"""Central production defaults and artifact-parameter derivation."""

import math
from types import MappingProxyType


# Artifact and encoding defaults.
DEFAULT_ARTIFACT_PREFIX = "beezy_run"
DEFAULT_PARITY_RATE = 0.10
DEFAULT_RANDOM_STRAND_COUNT = 100
DEFAULT_PAYLOAD_LENGTH = 44
DEFAULT_TREE_START = 0
DEFAULT_TREE_LENGTH = 44
HEADER_LENGTH = 4

# Simulation defaults.
DEFAULT_RANDOM_SEED = 20260801
DEFAULT_SIMULATION_DEPTH = 100
DEFAULT_SIMULATION_CHANNEL = MappingProxyType(
    {
        "ps": 0.0,
        "pd": 0.0,
        "pi": 0.0,
        "column": 0.0,
    }
)

# Decoder and BCJR defaults.
DEFAULT_BCJR_CHANNEL = MappingProxyType(
    {
        "ps": 0.0,
        "pd": 0.0,
        "pi": 0.0,
    }
)
DEFAULT_DRIFT_WINDOW = 3
DEFAULT_MAX_INSERTIONS_PER_SYMBOL = 1
DEFAULT_WORKER_COUNT = 24

ENCODING_PARAMETER_NAMES = frozenset(
    {
        "parity_strand_count",
        "payload_length",
        "tree_start",
        "tree_length",
    }
)


def default_parity_strand_count(data_strand_count):
    """Return the frozen ten-percent RS parity count."""
    data_strand_count = int(data_strand_count)
    if data_strand_count <= 0:
        raise ValueError("data_strand_count must be positive")
    return int(math.ceil(data_strand_count * DEFAULT_PARITY_RATE))


def derive_default_data_strand_count(total_strand_count):
    """Invert ``data + ceil(data * 0.10)`` for a seed length."""
    total_strand_count = int(total_strand_count)
    if total_strand_count <= 0:
        raise ValueError("tree_seed must not be empty")

    lower = 1
    upper = total_strand_count
    while lower <= upper:
        candidate = (lower + upper) // 2
        candidate_total = (
            candidate + default_parity_strand_count(candidate)
        )
        if candidate_total == total_strand_count:
            return candidate
        if candidate_total < total_strand_count:
            lower = candidate + 1
        else:
            upper = candidate - 1
    raise ValueError(
        "tree_seed length is incompatible with the default ten-percent "
        "RS layout"
    )


def encoding_parameter_overrides(
    data_strand_count,
    parity_strand_count,
    payload_length,
    tree_start,
    tree_length,
):
    """Return only decoder-relevant settings that differ from defaults."""
    parameters = {}
    if parity_strand_count != default_parity_strand_count(data_strand_count):
        parameters["parity_strand_count"] = int(parity_strand_count)
    if payload_length != DEFAULT_PAYLOAD_LENGTH:
        parameters["payload_length"] = int(payload_length)
    if tree_start != DEFAULT_TREE_START:
        parameters["tree_start"] = int(tree_start)
    if tree_length != DEFAULT_TREE_LENGTH:
        parameters["tree_length"] = int(tree_length)
    return parameters


def resolve_artifact_layout(total_strand_count, parameters=None):
    """Resolve the complete decoder layout from a seed and optional overrides."""
    parameters = dict(parameters or {})
    unknown = sorted(set(parameters) - ENCODING_PARAMETER_NAMES)
    if unknown:
        raise ValueError(
            "unknown encoding parameter(s): " + ", ".join(unknown)
        )

    total_strand_count = int(total_strand_count)
    if "parity_strand_count" in parameters:
        parity_strand_count = int(parameters["parity_strand_count"])
        data_strand_count = total_strand_count - parity_strand_count
        if data_strand_count <= 0 or parity_strand_count < 0:
            raise ValueError(
                "parity_strand_count is incompatible with tree_seed length"
            )
    else:
        data_strand_count = derive_default_data_strand_count(
            total_strand_count
        )
        parity_strand_count = default_parity_strand_count(
            data_strand_count
        )

    payload_length = int(
        parameters.get(
            "payload_length",
            DEFAULT_PAYLOAD_LENGTH,
        )
    )
    tree_start = int(parameters.get("tree_start", DEFAULT_TREE_START))
    tree_length = int(parameters.get("tree_length", DEFAULT_TREE_LENGTH))
    if payload_length <= 0:
        raise ValueError("payload_length must be positive")
    if payload_length % 4:
        raise ValueError("payload_length must be divisible by 4")
    if tree_start < 0 or tree_length < 0:
        raise ValueError("tree_start and tree_length must be non-negative")

    if tree_start + tree_length > payload_length:
        raise ValueError(
            "tree slice exceeds the payload length: "
            f"start={tree_start}, length={tree_length}, "
            f"payload={payload_length}"
        )
    indexed_strand_length = HEADER_LENGTH + payload_length

    return {
        "data_strand_count": data_strand_count,
        "parity_strand_count": parity_strand_count,
        "rs_padding_bases": 0,
        "indexed_strand_length": indexed_strand_length,
        "bcjr_payload_steps": indexed_strand_length // 2,
        "encoded_strand_length": indexed_strand_length * 2 + 4,
        "payload_length": payload_length,
        "tree_start": tree_start,
        "tree_length": tree_length,
    }


__all__ = [
    "DEFAULT_ARTIFACT_PREFIX",
    "DEFAULT_BCJR_CHANNEL",
    "DEFAULT_DRIFT_WINDOW",
    "DEFAULT_MAX_INSERTIONS_PER_SYMBOL",
    "DEFAULT_PARITY_RATE",
    "DEFAULT_RANDOM_SEED",
    "DEFAULT_RANDOM_STRAND_COUNT",
    "DEFAULT_SIMULATION_CHANNEL",
    "DEFAULT_SIMULATION_DEPTH",
    "DEFAULT_PAYLOAD_LENGTH",
    "DEFAULT_TREE_LENGTH",
    "DEFAULT_TREE_START",
    "DEFAULT_WORKER_COUNT",
    "ENCODING_PARAMETER_NAMES",
    "HEADER_LENGTH",
    "default_parity_strand_count",
    "derive_default_data_strand_count",
    "encoding_parameter_overrides",
    "resolve_artifact_layout",
]
