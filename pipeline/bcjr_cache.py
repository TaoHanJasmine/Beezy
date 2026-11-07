"""Durable per-sequence BCJR results; compatibility excludes downstream and scheduling settings."""
import json
from pathlib import Path
import sqlite3

try:
    from ..ecc.bcjr import decode_hard_batch, _CYTHON_FULL_LOG_MAP
    from ..ecc.trellis_encoding import CODEBOOK_SHA256
except ImportError:
    from ecc.bcjr import decode_hard_batch, _CYTHON_FULL_LOG_MAP
    from ecc.trellis_encoding import CODEBOOK_SHA256


def compatibility(payload_steps, channel, drift_window, max_insertions_per_symbol):
    return {'format': 1, 'producer_compatibility': 'log_map_byte_risks_v1',
            'codebook': CODEBOOK_SHA256, 'backend': 'cython' if _CYTHON_FULL_LOG_MAP else 'numpy',
            'payload_steps': int(payload_steps),
            'channel': {k: float(channel[k]) for k in ('ps', 'pi', 'pd')},
            'drift_window': int(drift_window), 'max_insertions_per_symbol': int(max_insertions_per_symbol)}


def decode_cached(sequences, payload_steps, channel, *, cache_path, drift_window=3,
                  max_insertions_per_symbol=1, workers=24, batch_size=4096):
    if batch_size <= 0:
        raise ValueError('batch_size must be positive')
    reads = list(sequences)
    expected = compatibility(payload_steps, channel, drift_window, max_insertions_per_symbol)
    path = Path(cache_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(path) as db:
        db.execute('PRAGMA journal_mode=WAL')
        db.execute('CREATE TABLE IF NOT EXISTS metadata (id INTEGER PRIMARY KEY, config TEXT NOT NULL)')
        db.execute('CREATE TABLE IF NOT EXISTS results (sequence TEXT PRIMARY KEY, result TEXT NOT NULL)')
        stored = db.execute('SELECT config FROM metadata WHERE id=1').fetchone()
        if stored is None:
            if db.execute('SELECT COUNT(*) FROM results').fetchone()[0]:
                raise ValueError('Cache has results without compatibility metadata')
            db.execute('INSERT INTO metadata VALUES (1, ?)', (json.dumps(expected, sort_keys=True),))
            db.commit()
        elif json.loads(stored[0]) != expected:
            changed = [k for k in expected if json.loads(stored[0]).get(k) != expected[k]]
            raise ValueError('BCJR cache semantic inputs changed: ' + ', '.join(changed) + '; existing results preserved')
        results = {}
        for sequence in dict.fromkeys(reads):
            row = db.execute('SELECT result FROM results WHERE sequence=?', (sequence,)).fetchone()
            if row:
                results[sequence] = json.loads(row[0])
        reused = len(results)
        missing = [s for s in dict.fromkeys(reads) if s not in results]
        print(f'BCJR cache: {reused:,} reused, {len(missing):,} new sequences', flush=True)
        for start in range(0, len(missing), batch_size):
            batch = missing[start:start + batch_size]
            decoded = decode_hard_batch(batch, payload_steps, channel, drift_window=drift_window,
                                        max_insertions_per_symbol=max_insertions_per_symbol, workers=workers,
                                        return_diagnostics=True, return_byte_posterior_risks=True)
            if len(decoded) != len(batch):
                raise ValueError('BCJR batch length mismatch')
            serialized = [(s, json.dumps(r, separators=(',', ':'))) for s, r in zip(batch, decoded)]
            with db:
                db.executemany('INSERT INTO results VALUES (?, ?)', serialized)
            results.update(zip(batch, decoded))
            print(f'BCJR saved: {min(start + batch_size, len(missing)):,}/{len(missing):,} new sequences', flush=True)
        return [results[s] for s in reads], {'reused_unique_sequences': reused,
                                            'new_unique_sequences': len(missing), 'path': str(path)}
