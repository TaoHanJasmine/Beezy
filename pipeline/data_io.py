"""Byte/DNA conversion and optional position-dependent whitening."""

import hashlib
from pathlib import Path


_DNA_ALPHABET = "ACGT"
_DNA_TO_VALUE = {base: value for value, base in enumerate(_DNA_ALPHABET)}
WHITENING_ALGORITHM = "shake256-block-xor"
WHITENING_VERSION = 1
WHITENING_SEED = 20260801
_WHITENING_DOMAIN = b"Beezy\x00whitening\x00shake256-block-xor\x00v1\x00"
_WHITENING_SEED_BYTES = WHITENING_SEED.to_bytes(8, byteorder="big")


def _payload_byte_length(payload_length):
    payload_length = int(payload_length)
    if payload_length <= 0:
        raise ValueError("payload_length must be positive")
    if payload_length % 4:
        raise ValueError("payload_length must be divisible by 4")
    return payload_length // 4


def _whitening_mask(block_index, byte_count):
    """Derive one independent block mask; this is not encryption."""
    if block_index < 0 or block_index >= 2**64:
        raise ValueError("whitening block index is out of range")
    digest = hashlib.shake_256(
        _WHITENING_DOMAIN
        + _WHITENING_SEED_BYTES
        + block_index.to_bytes(8, byteorder="big")
    )
    return digest.digest(byte_count)


def _transform_block(block, block_index):
    mask = _whitening_mask(block_index, len(block))
    return bytes(value ^ mask_value for value, mask_value in zip(block, mask))


def _bytes_to_dna(block):
    return "".join(
        _DNA_ALPHABET[(value >> shift) & 0b11]
        for value in block
        for shift in (6, 4, 2, 0)
    )


def _dna_to_bytes(sequence):
    if not isinstance(sequence, str):
        raise ValueError("each DNA sequence must be text")
    if len(sequence) % 4:
        raise ValueError("each DNA sequence length must be divisible by 4")
    output = bytearray()
    for offset in range(0, len(sequence), 4):
        value = 0
        for base in sequence[offset : offset + 4]:
            try:
                base_value = _DNA_TO_VALUE[base]
            except KeyError as error:
                raise ValueError(f"invalid DNA base: {base!r}") from error
            value = (value << 2) | base_value
        output.append(value)
    return bytes(output)


def bytes_to_dna_sequences(data, payload_length, *, whiten=False):
    """Convert bytes to fixed-length DNA payloads entirely in memory.

    Whitening XORs each padded byte block with a SHAKE-256 mask derived from
    the frozen protocol seed and the block's original zero-based position.
    XOR preserves the exact location of any later bit errors.
    """
    if not isinstance(data, (bytes, bytearray, memoryview)):
        raise TypeError("data must be bytes-like")
    data = bytes(data)
    byte_length = _payload_byte_length(payload_length)
    if not isinstance(whiten, bool):
        raise ValueError("whiten must be a boolean")
    if not data:
        return []

    padded_length = ((len(data) + byte_length - 1) // byte_length) * byte_length
    padded_data = data.ljust(padded_length, b"\x00")
    sequences = []
    for block_index, offset in enumerate(range(0, padded_length, byte_length)):
        block = padded_data[offset : offset + byte_length]
        if whiten:
            block = _transform_block(block, block_index)
        sequences.append(_bytes_to_dna(block))
    return sequences


def dna_sequences_to_bytes(sequences, original_byte_count, *, whiten=False):
    """Recover original bytes from ordered DNA payloads entirely in memory."""
    original_byte_count = int(original_byte_count)
    if original_byte_count < 0:
        raise ValueError("original_byte_count must be non-negative")
    if not isinstance(whiten, bool):
        raise ValueError("whiten must be a boolean")

    blocks = []
    expected_sequence_length = None
    for block_index, sequence in enumerate(sequences):
        if expected_sequence_length is None:
            expected_sequence_length = len(sequence)
        elif len(sequence) != expected_sequence_length:
            raise ValueError("DNA sequences must have a consistent length")
        block = _dna_to_bytes(sequence)
        if whiten:
            block = _transform_block(block, block_index)
        blocks.append(block)

    decoded = b"".join(blocks)
    if original_byte_count > len(decoded):
        raise ValueError("DNA sequences do not contain the requested byte count")
    return decoded[:original_byte_count]


def encode_file_to_dna(input_path, payload_length, *, whiten=False):
    """Read a file and convert it to fixed-length DNA payloads."""
    data = Path(input_path).read_bytes()
    sequences = bytes_to_dna_sequences(
        data,
        payload_length,
        whiten=whiten,
    )
    print("碱基总数：", len(data) * 4)
    return sequences, len(data)


def decode_dna_to_file(
    sequences,
    original_byte_count,
    output_path,
    *,
    whiten=False,
):
    """Recover ordered DNA payloads and write the original file bytes."""
    data = dna_sequences_to_bytes(
        sequences,
        original_byte_count,
        whiten=whiten,
    )
    Path(output_path).write_bytes(data)
    print(f"文件已成功还原至: {output_path}")


__all__ = [
    "WHITENING_ALGORITHM",
    "WHITENING_SEED",
    "WHITENING_VERSION",
    "bytes_to_dna_sequences",
    "decode_dna_to_file",
    "dna_sequences_to_bytes",
    "encode_file_to_dna",
]
