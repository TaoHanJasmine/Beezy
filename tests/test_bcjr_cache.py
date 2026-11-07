import json
import sqlite3

import pytest

from ..pipeline import bcjr_cache as cache

CHANNEL = {'ps': .00233602, 'pi': .00067446, 'pd': .00441979}


def test_real_result_roundtrip_and_no_redecode(tmp_path, monkeypatch):
    path = tmp_path / 'results.sqlite'
    fresh, stats = cache.decode_cached(['ACGT' * 25], 24, CHANNEL, cache_path=path, workers=1)
    assert stats['new_unique_sequences'] == 1
    monkeypatch.setattr(cache, 'decode_hard_batch', lambda *a, **k: pytest.fail('Cache hit recomputed'))
    loaded, stats = cache.decode_cached(['ACGT' * 25] * 2, 24, CHANNEL, cache_path=path, workers=24, batch_size=1)
    assert json.dumps(loaded[0]) == json.dumps(fresh[0])
    assert loaded[0] == loaded[1]
    assert stats['reused_unique_sequences'] == 1 and stats['new_unique_sequences'] == 0


def test_increment_and_interrupted_batches(tmp_path, monkeypatch):
    path = tmp_path / 'results.sqlite'
    calls = []
    def producer(seqs, *args, **kwargs):
        calls.extend(seqs)
        if 'G' in seqs:
            raise RuntimeError('interrupted')
        return [(s, {'byte_posterior_risks': [0.1]}) for s in seqs]
    monkeypatch.setattr(cache, 'decode_hard_batch', producer)
    with pytest.raises(RuntimeError):
        cache.decode_cached(['A', 'C', 'G'], 24, CHANNEL, cache_path=path, batch_size=2)
    calls.clear()
    result, stats = cache.decode_cached(['C', 'A', 'T', 'C'], 24, CHANNEL, cache_path=path, batch_size=1)
    assert calls == ['T']
    assert [r[0] for r in result] == ['C', 'A', 'T', 'C']
    assert stats['reused_unique_sequences'] == 2


def test_production_pipeline_reuses_results_with_new_frequencies(tmp_path, monkeypatch):
    from ..pipeline.encoder import encode
    from ..pipeline.decoder import decode
    source = tmp_path / 'source.bin'
    source.write_bytes(bytes(range(110)))
    prefix = str(tmp_path / 'case')
    encode(artifact_prefix=prefix, source_path=source)
    from pathlib import Path
    encoded = Path(prefix + '_encoded_strands.txt').read_text()
    reads = tmp_path / 'reads.txt'
    reads.write_text(encoded * 2)
    output = tmp_path / 'output.bin'
    kwargs = dict(tree_seed_path=Path(prefix + '_seed'), channel={'ps': 0., 'pi': 0., 'pd': 0.},
                  drift_window=0, worker_count=1, output_path=output, return_diagnostics=True,
                  bcjr_cache_path=tmp_path / 'results.sqlite')
    _, first = decode(reads, **kwargs)
    assert output.read_bytes() == source.read_bytes()
    monkeypatch.setattr(cache, 'decode_hard_batch', lambda *a, **k: pytest.fail('Cached sequences recomputed'))
    reads.write_text(encoded * 3)
    _, second = decode(reads, **kwargs)
    assert output.read_bytes() == source.read_bytes()
    assert second['reads']['raw_count'] * 2 == first['reads']['raw_count'] * 3
    assert second['bcjr_cache']['new_unique_sequences'] == 0
    assert second['bcjr_cache']['reused_unique_sequences'] == first['reads']['unique_count']


@pytest.mark.parametrize('change', ['ps', 'pi', 'pd', 'payload_steps', 'drift_window', 'max_insertions_per_symbol', 'codebook', 'backend'])
def test_semantic_change_preserves_cache_without_recompute(tmp_path, monkeypatch, change):
    path = tmp_path / 'results.sqlite'
    monkeypatch.setattr(cache, 'decode_hard_batch', lambda reads, *a, **k: [('AC', {}) for _ in reads])
    cache.decode_cached(['AC'], 24, CHANNEL, cache_path=path)
    monkeypatch.setattr(cache, 'decode_hard_batch', lambda *a, **k: pytest.fail('Unexpected recomputation'))
    channel = dict(CHANNEL)
    kwargs = {'payload_steps': 24, 'drift_window': 3, 'max_insertions_per_symbol': 1}
    if change in channel:
        channel[change] += .001
    elif change in kwargs:
        kwargs[change] += 1
    elif change == 'codebook':
        monkeypatch.setattr(cache, 'CODEBOOK_SHA256', 'different')
    else:
        monkeypatch.setattr(cache, '_CYTHON_FULL_LOG_MAP', not cache._CYTHON_FULL_LOG_MAP)
    with pytest.raises(ValueError, match='semantic inputs changed'):
        cache.decode_cached(['AC'], channel=channel, cache_path=path, **kwargs)
    with sqlite3.connect(path) as db:
        assert db.execute('SELECT COUNT(*) FROM results').fetchone()[0] == 1
