"""Metrics that compare recovered data with explicitly supplied truth."""

import hashlib
from pathlib import Path


def file_sha256(path):
    """Return the SHA-256 digest of a file as lowercase hexadecimal."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as input_file:
        for block in iter(lambda: input_file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def compare_file_hashes(original_path, recovered_path):
    """Compare an original file and a recovered file by SHA-256."""
    original_hash = file_sha256(original_path)
    recovered_hash = file_sha256(recovered_path)
    return {
        "algorithm": "sha256",
        "original_hash": original_hash,
        "recovered_hash": recovered_hash,
        "match": original_hash == recovered_hash,
    }


def compare_strands(reference_strands, recovered_strands):
    """Return strand- and base-error counts against explicit truth."""
    references = tuple(reference_strands)
    recovered = tuple(recovered_strands)
    shared_count = min(len(references), len(recovered))
    strand_errors = sum(
        references[index] != recovered[index]
        for index in range(shared_count)
    ) + abs(len(references) - len(recovered))
    base_errors = sum(
        sum(left != right for left, right in zip(reference, decoded))
        + abs(len(reference) - len(decoded))
        for reference, decoded in zip(references, recovered)
    )
    base_errors += sum(len(strand) for strand in references[shared_count:])
    base_errors += sum(len(strand) for strand in recovered[shared_count:])
    return {
        "reference_strand_count": len(references),
        "recovered_strand_count": len(recovered),
        "strand_error_count": strand_errors,
        "base_error_count": base_errors,
    }


__all__ = ["compare_file_hashes", "compare_strands", "file_sha256"]
