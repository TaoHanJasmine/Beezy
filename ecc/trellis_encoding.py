"""Structured rate-1/2 trellis definition and encoder."""

import hashlib
from types import MappingProxyType
from typing import NamedTuple

CODEBOOK_SHA256 = (
    "ad7ab89cd7f136f95633279a7ccee96c08cbec2cc011563d37182ef6bbbd8620"
)
DNA_ALPHABET = ("A", "C", "G", "T")
BASE_TO_BITS = MappingProxyType(
    {
        "A": (0, 0),
        "T": (0, 1),
        "G": (1, 0),
        "C": (1, 1),
    }
)
BITS_TO_BASE = MappingProxyType(
    {bits: base for base, bits in BASE_TO_BITS.items()}
)

# Symbols are two source bases.  Their order preserves the bit-plane mapping
# used when the selected trellis was constructed.
INPUT_PAIRS = (
    "AA", "TA", "AT", "TT",
    "GA", "CA", "GT", "CT",
    "AG", "TG", "AC", "TC",
    "GG", "CG", "GC", "CC",
)
PAIR_TO_SYMBOL = {pair: symbol for symbol, pair in enumerate(INPUT_PAIRS)}

# Rows and columns follow the bit-pair order used by ``INPUT_PAIRS``.  The
# upper bit is the GC indicator; each cell fixes the four encoded GC bits.
PAIR_PLANE_PATTERNS = ((0, 0), (1, 0), (0, 1), (1, 1))
GC_OUTPUT_TABLE = (
    ("1010", "0110", "1001", "1101"),
    ("0101", "1001", "0110", "0010"),
    ("1001", "1010", "0101", "0110"),
    ("0110", "0101", "1010", "1001"),
)

# Matrix columns are previous upper, previous lower, current upper, and
# current lower bit pairs, in that order.  Multiplication is over GF(2).
LOWER_GENERATOR_INPUTS = (
    "previous_upper_1",
    "previous_upper_2",
    "previous_lower_1",
    "previous_lower_2",
    "current_upper_1",
    "current_upper_2",
    "current_lower_1",
    "current_lower_2",
)
LOWER_GENERATOR_MATRIX = (
    (0, 0, 0, 1, 1, 0, 0, 0),
    (1, 0, 1, 1, 1, 0, 1, 0),
    (0, 0, 0, 1, 0, 1, 1, 1),
    (0, 1, 0, 0, 1, 1, 1, 0),
)


class ConstrainedTrellis(NamedTuple):
    """Immutable output and state-transition tables used by the codec."""

    output_table: tuple
    next_state_table: tuple
    initial_state: int
    termination_symbol: int
    number_states: int
    number_inputs: int


_PAIR_PLANE_INDEX = {
    pattern: index for index, pattern in enumerate(PAIR_PLANE_PATTERNS)
}


def _split_pair_planes(pair):
    first, second = (BASE_TO_BITS[base] for base in pair)
    return (first[0], second[0]), (first[1], second[1])


def _encode_branch(previous_pair, current_pair):
    previous_upper, previous_lower = _split_pair_planes(previous_pair)
    current_upper, current_lower = _split_pair_planes(current_pair)
    upper_output = GC_OUTPUT_TABLE[
        _PAIR_PLANE_INDEX[previous_upper]
    ][_PAIR_PLANE_INDEX[current_upper]]
    generator_input = (
        previous_upper
        + previous_lower
        + current_upper
        + current_lower
    )
    lower_output = tuple(
        sum(
            coefficient * bit
            for coefficient, bit in zip(row, generator_input)
        )
        % 2
        for row in LOWER_GENERATOR_MATRIX
    )
    return "".join(
        BITS_TO_BASE[(int(upper_bit), lower_bit)]
        for upper_bit, lower_bit in zip(upper_output, lower_output)
    )


def build_trellis():
    """Build the fixed DNA trellis from its GC and GF(2) components."""
    output_table = tuple(
        tuple(
            _encode_branch(previous_pair, current_pair)
            for current_pair in INPUT_PAIRS
        )
        for previous_pair in INPUT_PAIRS
    )
    input_symbols = tuple(range(len(INPUT_PAIRS)))
    next_state_table = tuple(input_symbols for _ in INPUT_PAIRS)
    return ConstrainedTrellis(
        output_table=output_table,
        next_state_table=next_state_table,
        initial_state=0,
        termination_symbol=PAIR_TO_SYMBOL["AA"],
        number_states=len(INPUT_PAIRS),
        number_inputs=len(INPUT_PAIRS),
    )


# Encoding and BCJR decoding still use this fully expanded lookup table.  The
# current input becomes the next state, and a final AA symbol terminates a
# strand.
TRELLIS = build_trellis()
OUTPUT_TABLE = TRELLIS.output_table
NEXT_STATE_TABLE = TRELLIS.next_state_table

_actual_codebook_hash = hashlib.sha256(
    "\n".join(",".join(row) for row in OUTPUT_TABLE).encode("ascii")
).hexdigest()
if _actual_codebook_hash != CODEBOOK_SHA256:
    raise RuntimeError("the output table does not match its fingerprint")


def encode_constrained(source_dna):
    """Encode an even-length DNA source and append the fixed ``AA`` tail."""
    source = source_dna.upper()
    if any(base not in DNA_ALPHABET for base in source):
        raise ValueError("source DNA must contain only A, C, G, and T")
    if len(source) % 2:
        raise ValueError("the constrained codec requires an even source length")

    state = TRELLIS.initial_state
    encoded = []
    for offset in range(0, len(source), 2):
        symbol = PAIR_TO_SYMBOL[source[offset : offset + 2]]
        encoded.append(TRELLIS.output_table[state][symbol])
        state = TRELLIS.next_state_table[state][symbol]

    encoded.append(TRELLIS.output_table[state][TRELLIS.termination_symbol])
    return "".join(encoded), len(source) // 2


def codec_info():
    """Return the stable production-code parameters written to metadata."""
    return {
        "codebook_sha256": CODEBOOK_SHA256,
        "rate": "1/2",
        "source_nt_per_step": 2,
        "encoded_nt_per_step": 4,
        "states": TRELLIS.number_states,
        "termination_pair": "AA",
        "max_homopolymer": 4,
        "gc_window": 12,
        "gc_min": 4,
        "gc_max": 8,
    }


__all__ = [
    "BASE_TO_BITS",
    "BITS_TO_BASE",
    "CODEBOOK_SHA256",
    "ConstrainedTrellis",
    "DNA_ALPHABET",
    "GC_OUTPUT_TABLE",
    "INPUT_PAIRS",
    "LOWER_GENERATOR_INPUTS",
    "LOWER_GENERATOR_MATRIX",
    "NEXT_STATE_TABLE",
    "OUTPUT_TABLE",
    "PAIR_PLANE_PATTERNS",
    "PAIR_TO_SYMBOL",
    "TRELLIS",
    "build_trellis",
    "codec_info",
    "encode_constrained",
]
