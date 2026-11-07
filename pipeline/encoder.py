"""Production encoding pipeline for Beezy DNA artifacts."""

import json
from pathlib import Path
import random

from .data_io import encode_file_to_dna
from .config import (
    DEFAULT_RANDOM_STRAND_COUNT,
    DEFAULT_PAYLOAD_LENGTH,
    DEFAULT_TREE_LENGTH,
    DEFAULT_TREE_START,
    default_parity_strand_count,
    encoding_parameter_overrides,
)

try:  # Package execution: ``python -m Beezy.main``.
    from ..ecc.trellis_encoding import encode_constrained
    from ..ecc.rs import vertical_rs_encode
    from ..indexing import construct_strand_indices
except ImportError:  # Direct execution through ``python main.py``.
    from ecc.trellis_encoding import encode_constrained
    from ecc.rs import vertical_rs_encode
    from indexing import construct_strand_indices


ENCODING_PARAMETERS_SUFFIX = "_encoding_parameters.json"
LEGACY_METADATA_SUFFIX = "_ecc_meta.json"


def generate_random_strands(payload_length, strand_count):
    """Generate uniformly random DNA strands."""
    return [
        "".join(random.choices("ATGC", k=payload_length))
        for _ in range(strand_count)
    ]


def resolve_parity_strand_count(
    data_strand_count,
    parity_strand_count=None,
):
    """Resolve an absolute RS count or use the frozen ten-percent default."""
    data_strand_count = int(data_strand_count)
    if data_strand_count <= 0:
        raise ValueError("data_strand_count must be positive")
    if parity_strand_count is not None:
        resolved_count = int(parity_strand_count)
        if resolved_count < 0:
            raise ValueError("parity_strand_count must be non-negative")
        return resolved_count
    return default_parity_strand_count(data_strand_count)


def encode(
    artifact_prefix,
    strand_count=DEFAULT_RANDOM_STRAND_COUNT,
    payload_length=DEFAULT_PAYLOAD_LENGTH,
    tree_length=DEFAULT_TREE_LENGTH,
    tree_start=DEFAULT_TREE_START,
    source_path=None,
    write_artifacts=True,
    parity_strand_count=None,
    tree_seed_path=None,
    whiten=True,
):
    """Encode data through vertical RS, indexing, and the constrained codec.

    File whitening is a frozen, reversible production transform. Random DNA
    generation does not need file whitening.
    """
    artifact_prefix = str(artifact_prefix)
    payload_length = int(payload_length)
    if payload_length <= 0:
        raise ValueError("payload_length must be positive")
    if payload_length % 4:
        raise ValueError("payload_length must be divisible by 4")
    if not isinstance(whiten, bool):
        raise ValueError("whiten must be a boolean")
    whiten_source = bool(source_path is not None and whiten)
    if source_path is not None:
        data_strands, source_size_bytes = encode_file_to_dna(
            str(source_path),
            payload_length,
            whiten=whiten_source,
        )
    else:
        data_strands = generate_random_strands(
            payload_length,
            strand_count,
        )
        source_size_bytes = 0
    if not data_strands:
        raise ValueError("the input produced no DNA strands")

    data_strand_count = len(data_strands)
    parity_strand_count = resolve_parity_strand_count(
        data_strand_count,
        parity_strand_count=parity_strand_count,
    )
    reference_strands, rs_padding_bases = vertical_rs_encode(
        data_strands,
        parity_strand_count,
    )
    reference_strand_length = len(reference_strands[0])
    resolved_tree_length = int(
        DEFAULT_TREE_LENGTH if tree_length is None else tree_length
    )
    tree_start = int(tree_start)
    if tree_start < 0 or resolved_tree_length < 0:
        raise ValueError("tree_start and tree_length must be non-negative")
    if tree_start + resolved_tree_length > reference_strand_length:
        raise ValueError(
            "tree slice exceeds the reference strand length: "
            f"start={tree_start}, length={resolved_tree_length}, "
            f"strand={reference_strand_length}"
        )

    indexed_strands, tree_seed, collision_count, collision_groups = (
        construct_strand_indices(
            reference_strands,
            tree_start=tree_start,
            tree_length=resolved_tree_length,
        )
    )

    encoded_strands = []
    payload_steps = None
    for indexed_sequence in indexed_strands:
        encoded_sequence, current_payload_steps = encode_constrained(
            indexed_sequence
        )
        if payload_steps is None:
            payload_steps = current_payload_steps
        elif current_payload_steps != payload_steps:
            raise ValueError("horizontal encoder produced inconsistent lengths")
        encoded_strands.append(encoded_sequence)

    parameters = encoding_parameter_overrides(
        data_strand_count,
        parity_strand_count,
        payload_length,
        tree_start,
        resolved_tree_length,
    )
    seed_payload = {
        "source_size_bytes": source_size_bytes,
        "tree_seed": tree_seed,
    }
    if source_path is not None and not whiten_source:
        # Current artifacts omit this field and use frozen whitening. Only the
        # historical/control path carries an explicit unwhitened marker.
        seed_payload["file_whitening"] = False

    if write_artifacts:
        Path(artifact_prefix + "_encoded_strands.txt").write_text(
            "\n".join(encoded_strands) + "\n",
            encoding="utf-8",
        )
        resolved_tree_seed_path = Path(
            tree_seed_path or artifact_prefix + "_seed"
        )
        resolved_tree_seed_path.write_text(
            json.dumps(seed_payload, ensure_ascii=False),
            encoding="utf-8",
        )
        resolved_tree_seed_path.chmod(0o600)
        parameters_path = Path(
            artifact_prefix + ENCODING_PARAMETERS_SUFFIX
        )
        if parameters:
            parameters_path.write_text(
                json.dumps(parameters, indent=2, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )
        elif parameters_path.exists():
            parameters_path.unlink()
        legacy_metadata_path = Path(artifact_prefix + LEGACY_METADATA_SUFFIX)
        if legacy_metadata_path.exists():
            legacy_metadata_path.unlink()

    print(
        "ECC编码完成："
        f"数据 {data_strand_count} 条，RS校验 {parity_strand_count} 条，"
        f"索引后 {len(indexed_strands[0])} nt，"
        f"编码链 {len(encoded_strands[0])} nt。"
    )
    return tree_seed, collision_count, collision_groups


__all__ = [
    "encode",
    "generate_random_strands",
    "resolve_parity_strand_count",
]
