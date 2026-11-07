"""Production recovery pipeline for Beezy DNA artifacts."""

import json
from pathlib import Path

from .config import (
    DEFAULT_BCJR_CHANNEL,
    DEFAULT_DRIFT_WINDOW,
    DEFAULT_MAX_INSERTIONS_PER_SYMBOL,
    DEFAULT_PAYLOAD_LENGTH,
    DEFAULT_TREE_LENGTH,
    DEFAULT_TREE_START,
    DEFAULT_WORKER_COUNT,
    resolve_artifact_layout,
)
from .data_io import decode_dna_to_file
from .read_frequency import read_raw_frequency_table

try:  # Package execution: ``python -m Beezy.main``.
    from ..ecc.bcjr import decode_hard_batch
    from ..ecc.trellis_encoding import CODEBOOK_SHA256
    from ..ecc.rs import vertical_rs_decode_hard
    from ..indexing.index_recovery.frequency_aggregation import (
        aggregate_bcjr_payloads_by_frequency,
    )
    from ..indexing.index_recovery.frequency_recovery import (
        recover_strand_indices_by_frequency,
    )
except ImportError:  # Direct execution through ``python main.py``.
    from ecc.bcjr import decode_hard_batch
    from ecc.trellis_encoding import CODEBOOK_SHA256
    from ecc.rs import vertical_rs_decode_hard
    from indexing.index_recovery.frequency_aggregation import (
        aggregate_bcjr_payloads_by_frequency,
    )
    from indexing.index_recovery.frequency_recovery import (
        recover_strand_indices_by_frequency,
    )


def _load_artifact_layout(tree_seed_path, encoding_parameters=None):
    resolved_tree_seed_path = Path(tree_seed_path)
    if not resolved_tree_seed_path.exists():
        raise ValueError(f"missing tree seed file: {resolved_tree_seed_path}")
    seed_payload = json.loads(
        resolved_tree_seed_path.read_text(encoding="utf-8")
    )
    if isinstance(seed_payload, list):
        raise ValueError(
            "legacy list tree seeds are unsupported: production decoding "
            "supports only object tree seeds, and non-default layouts must "
            "be supplied through explicit decoder parameters"
        )
    if not isinstance(seed_payload, dict):
        raise ValueError("tree seed must be a JSON object")
    required_seed_fields = {"source_size_bytes", "tree_seed"}
    allowed_seed_fields = required_seed_fields | {"file_whitening"}
    if not required_seed_fields.issubset(seed_payload) or not set(
        seed_payload
    ).issubset(allowed_seed_fields):
        raise ValueError(
            "tree seed object must contain source_size_bytes and tree_seed "
            "with only the optional file_whitening flag"
        )
    has_whitening_setting = "file_whitening" in seed_payload
    whitening_setting = seed_payload.get("file_whitening")
    if has_whitening_setting and whitening_setting is not False:
        raise ValueError("file_whitening must be false when present")

    tree_seed = seed_payload["tree_seed"]
    if not isinstance(tree_seed, list):
        raise ValueError("tree_seed must be a JSON list")
    parameters = dict(encoding_parameters or {})
    layout = resolve_artifact_layout(len(tree_seed), parameters)
    source_size_bytes = int(seed_payload["source_size_bytes"])
    if source_size_bytes < 0:
        raise ValueError("source_size_bytes must be non-negative")
    if source_size_bytes > 0:
        expected_data_strand_count = (
            source_size_bytes * 4 + layout["payload_length"] - 1
        ) // layout["payload_length"]
        if expected_data_strand_count != layout["data_strand_count"]:
            raise ValueError(
                "source_size_bytes is incompatible with the resolved data "
                "strand count"
            )
    layout["source_size_bytes"] = source_size_bytes
    layout["file_whitening"] = not has_whitening_setting
    return tree_seed, layout


def decode(
    reads_path,
    tree_seed_path,
    channel=None,
    write_output=True,
    output_path=None,
    drift_window=DEFAULT_DRIFT_WINDOW,
    max_insertions_per_symbol=DEFAULT_MAX_INSERTIONS_PER_SYMBOL,
    worker_count=DEFAULT_WORKER_COUNT,
    return_diagnostics=False,
    parity_strand_count=None,
    payload_length=DEFAULT_PAYLOAD_LENGTH,
    tree_start=DEFAULT_TREE_START,
    tree_length=DEFAULT_TREE_LENGTH,
    bcjr_cache_path=None,
):
    """Recover from explicit reads, seed, output, and protocol layout inputs."""
    channel_parameters = dict(
        DEFAULT_BCJR_CHANNEL if channel is None else channel
    )
    encoding_parameters = {
        "payload_length": payload_length,
        "tree_start": tree_start,
        "tree_length": tree_length,
    }
    if parity_strand_count is not None:
        encoding_parameters["parity_strand_count"] = parity_strand_count
    tree_seed, metadata = _load_artifact_layout(
        tree_seed_path,
        encoding_parameters=encoding_parameters,
    )
    data_strand_count = int(metadata["data_strand_count"])
    parity_strand_count = int(metadata["parity_strand_count"])
    metadata_strand_count = data_strand_count + parity_strand_count
    strand_count = len(tree_seed)
    if metadata_strand_count != strand_count:
        raise ValueError(
            "tree seed length does not match data plus parity strand counts: "
            f"{strand_count} != {data_strand_count} + {parity_strand_count}"
        )
    payload_steps = int(metadata["bcjr_payload_steps"])
    tree_start = int(metadata["tree_start"])
    tree_length = int(metadata["tree_length"])
    source_size_bytes = int(metadata["source_size_bytes"])
    if write_output and source_size_bytes > 0 and output_path is None:
        raise ValueError(
            "output_path is required when decoding a file artifact"
        )

    resolved_reads_path = Path(reads_path)
    raw_unique_reads = read_raw_frequency_table(resolved_reads_path)
    cache_diagnostics = None
    if bcjr_cache_path is not None:
        from .bcjr_cache import decode_cached
        raw_bcjr_results, cache_diagnostics = decode_cached(
            raw_unique_reads.iter_reads(), payload_steps, channel_parameters,
            cache_path=bcjr_cache_path, drift_window=drift_window,
            max_insertions_per_symbol=max_insertions_per_symbol, workers=worker_count,
        )
    else:
        raw_bcjr_results = decode_hard_batch(
            raw_unique_reads.iter_reads(),
            payload_steps,
            channel_parameters,
            drift_window=drift_window,
            max_insertions_per_symbol=max_insertions_per_symbol,
            workers=worker_count,
            return_diagnostics=True,
        )
    bcjr_results, candidate_frequencies, frequency_diagnostics = (
        aggregate_bcjr_payloads_by_frequency(
            raw_unique_reads,
            raw_bcjr_results,
            tree_start=tree_start,
            tree_length=tree_length,
        )
    )
    bcjr_failed_unique_read_count = frequency_diagnostics.pop(
        "bcjr_failed_unique_read_count"
    )
    bcjr_failed_read_count = frequency_diagnostics.pop(
        "bcjr_failed_read_count"
    )
    bcjr_decoded_strands = [result[0] for result in bcjr_results]
    ordered_strands, index_assignments, index_recovery_diagnostics = (
        recover_strand_indices_by_frequency(
            bcjr_decoded_strands,
            candidate_frequencies,
            tree_seed,
            tree_start=tree_start,
            tree_length=tree_length,
        )
    )

    rs_decoded_strands, rs_diagnostics = vertical_rs_decode_hard(
        ordered_strands,
        data_strand_count=data_strand_count,
        parity_strand_count=parity_strand_count,
        padding_bases=int(metadata["rs_padding_bases"]),
        strand_length=int(
            metadata.get("payload_length", metadata.get("rs_strand_length"))
        ),
        return_diagnostics=True,
    )

    resolved_output_path = None
    if write_output and source_size_bytes > 0:
        resolved_output_path = str(output_path)
        decode_dna_to_file(
            rs_decoded_strands,
            source_size_bytes,
            resolved_output_path,
            whiten=bool(metadata.get("file_whitening", False)),
        )

    diagnostics = {
        "decoder": {
            "route": "payload_frequency_index_recovery",
            "codec_fingerprint": CODEBOOK_SHA256,
        },
        "reads": {
            "raw_count": raw_unique_reads.raw_read_count,
            "unique_count": raw_unique_reads.unique_read_count,
        },
        "bcjr": {
            "algorithm": "log_map",
            "drift_window": int(drift_window),
            "max_insertions_per_symbol": int(max_insertions_per_symbol),
            "worker_count": int(worker_count),
            "failed_unique_read_count": bcjr_failed_unique_read_count,
            "failed_read_count": bcjr_failed_read_count,
            "complete": bcjr_failed_unique_read_count == 0,
        },
        "payload_aggregation": frequency_diagnostics,
        "index_recovery": index_recovery_diagnostics,
        "reed_solomon": rs_diagnostics,
        "output_path": resolved_output_path,
    }
    if cache_diagnostics is not None:
        diagnostics['bcjr_cache'] = cache_diagnostics
    if return_diagnostics:
        return rs_decoded_strands, diagnostics
    return rs_decoded_strands


__all__ = ["decode"]
