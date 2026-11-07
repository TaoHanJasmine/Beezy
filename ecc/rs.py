"""Vertical Reed-Solomon encoding and adaptive-erasure decoding."""

from collections import Counter
import math

import reedsolo


DNA_TO_INT = {"A": 0, "C": 1, "G": 2, "T": 3}
INT_TO_DNA = {value: base for base, value in DNA_TO_INT.items()}


def dna_to_bytes(dna_sequence):
    """Pack four DNA bases per byte and return bytes plus padding count."""
    padding_bases = (4 - len(dna_sequence) % 4) % 4
    padded_dna = dna_sequence + "A" * padding_bases
    packed_bytes = bytearray()
    for offset in range(0, len(padded_dna), 4):
        value = (
            (DNA_TO_INT[padded_dna[offset]] << 6)
            | (DNA_TO_INT[padded_dna[offset + 1]] << 4)
            | (DNA_TO_INT[padded_dna[offset + 2]] << 2)
            | DNA_TO_INT[padded_dna[offset + 3]]
        )
        packed_bytes.append(value)
    return packed_bytes, padding_bases


def bytes_to_dna(packed_bytes, padding_bases):
    """Unpack bytes to DNA and remove the encoder padding."""
    bases = []
    for value in packed_bytes:
        bases.extend(
            (
                INT_TO_DNA[(value >> 6) & 3],
                INT_TO_DNA[(value >> 4) & 3],
                INT_TO_DNA[(value >> 2) & 3],
                INT_TO_DNA[value & 3],
            )
        )
    if padding_bases:
        del bases[-padding_bases:]
    return "".join(bases)


def rs_chunk_layout(
    data_strand_count,
    parity_strand_count,
    chunk_limit=200,
):
    """Return global data/parity indices for independent GF(256) chunks."""
    if data_strand_count <= 0:
        raise ValueError("data_strand_count must be positive")
    if parity_strand_count < 0:
        raise ValueError("parity_strand_count must be non-negative")
    if chunk_limit <= 0:
        raise ValueError("chunk_limit must be positive")

    chunk_count = int(math.ceil(data_strand_count / chunk_limit))

    def fixed_layout_is_valid(chunk_count):
        highest_parity = int(
            math.ceil(parity_strand_count / chunk_count)
        )
        return min(chunk_limit, data_strand_count) + highest_parity <= 255

    use_fixed_chunks = fixed_layout_is_valid(chunk_count)
    if not use_fixed_chunks:
        for proposed_chunk_count in range(
            chunk_count + 1,
            data_strand_count + 1,
        ):
            max_data = int(
                math.ceil(data_strand_count / proposed_chunk_count)
            )
            max_parity = int(
                math.ceil(parity_strand_count / proposed_chunk_count)
            )
            if max_data + max_parity <= 255:
                chunk_count = proposed_chunk_count
                break
        else:
            raise ValueError(
                f"cannot distribute {parity_strand_count} parity strands "
                f"over {data_strand_count} data strands within GF(256)"
            )

    parity_per_chunk, extra_parity = divmod(
        parity_strand_count,
        chunk_count,
    )
    if use_fixed_chunks:
        data_sizes = [
            min(chunk_limit, data_strand_count - index * chunk_limit)
            for index in range(chunk_count)
        ]
    else:
        data_per_chunk, extra_data = divmod(
            data_strand_count,
            chunk_count,
        )
        data_sizes = [
            data_per_chunk + (1 if index < extra_data else 0)
            for index in range(chunk_count)
        ]

    layout = []
    data_start = 0
    parity_start = data_strand_count
    for chunk_index, data_size in enumerate(data_sizes):
        data_end = data_start + data_size
        chunk_parity_count = parity_per_chunk + (
            1 if chunk_index < extra_parity else 0
        )
        data_indices = tuple(range(data_start, data_end))
        parity_indices = tuple(
            range(parity_start, parity_start + chunk_parity_count)
        )
        if len(data_indices) + len(parity_indices) > 255:
            raise ValueError(
                f"RS chunk {chunk_index} exceeds the GF(256) limit"
            )
        layout.append((data_indices, parity_indices))
        data_start = data_end
        parity_start += chunk_parity_count
    return layout


def vertical_rs_encode(data_strands, parity_strand_count):
    """Encode equal-length DNA strands column-wise with Reed-Solomon."""
    if not data_strands:
        raise ValueError("data_strands must not be empty")
    if len({len(strand) for strand in data_strands}) != 1:
        raise ValueError("all DNA strands must have the same length")
    if any(
        base not in DNA_TO_INT
        for strand in data_strands
        for base in strand
    ):
        raise ValueError("DNA strands must contain only A, C, G, and T")

    layout = rs_chunk_layout(len(data_strands), parity_strand_count)
    data_byte_strands = []
    padding_bases = 0
    for strand in data_strands:
        packed_bytes, padding_bases = dna_to_bytes(strand)
        data_byte_strands.append(packed_bytes)
    bytes_per_strand = len(data_byte_strands[0])
    parity_byte_strands = []

    for data_indices, parity_indices in layout:
        chunk_parity_count = len(parity_indices)
        if chunk_parity_count == 0:
            continue
        codec = reedsolo.RSCodec(chunk_parity_count)
        chunk_parity = [bytearray() for _ in parity_indices]
        for byte_index in range(bytes_per_strand):
            column = bytearray(
                data_byte_strands[index][byte_index] for index in data_indices
            )
            encoded_column = codec.encode(column)
            for parity_index in range(chunk_parity_count):
                chunk_parity[parity_index].append(
                    encoded_column[len(data_indices) + parity_index]
                )
        parity_byte_strands.extend(chunk_parity)

    data_strands = [
        bytes_to_dna(strand, 0) for strand in data_byte_strands
    ]
    parity_strands = [
        bytes_to_dna(strand, 0) for strand in parity_byte_strands
    ]
    reference_strands = data_strands + parity_strands
    return reference_strands, padding_bases


def _normalize_hard_observations(received, total_count, strand_length):
    observations = {}
    inferred_length = strand_length
    for raw_index, raw_value in received.items():
        index = int(raw_index)
        if not 0 <= index < total_count:
            continue
        if not isinstance(raw_value, str):
            raise ValueError(
                f"RS position {index} must contain one DNA strand"
            )
        sequence = raw_value.upper()
        if any(base not in DNA_TO_INT for base in sequence):
            raise ValueError(f"RS position {index} contains an invalid DNA base")
        if inferred_length is None:
            inferred_length = len(sequence)
        if len(sequence) != inferred_length:
            raise ValueError(
                f"RS position {index} has length {len(sequence)}, expected "
                f"{inferred_length}"
            )
        observations[index] = dna_to_bytes(sequence)[0]
    if inferred_length is None:
        raise ValueError(
            "strand_length is required when every RS position is missing"
        )
    return observations, inferred_length


def vertical_rs_decode_hard(
    received,
    data_strand_count,
    parity_strand_count,
    padding_bases,
    strand_length=None,
    return_diagnostics=False,
):
    """Decode one hard DNA value per observed position; gaps are erasures."""
    total_count = data_strand_count + parity_strand_count
    layout = rs_chunk_layout(data_strand_count, parity_strand_count)
    observations, strand_length = _normalize_hard_observations(
        received, total_count, strand_length
    )
    bytes_per_strand = len(dna_to_bytes("A" * strand_length)[0])
    decoded_data = [bytearray() for _ in range(data_strand_count)]
    failed_columns = 0
    corrected_symbols = 0
    chunk_details = []
    decoded_columns = []
    unresolved_columns = []

    for chunk_index, (data_indices, parity_indices) in enumerate(layout):
        chunk_indices = data_indices + parity_indices
        missing_local = [
            local_index
            for local_index, global_index in enumerate(chunk_indices)
            if global_index not in observations
        ]
        chunk_parity_count = len(parity_indices)
        codec = (
            reedsolo.RSCodec(chunk_parity_count)
            if chunk_parity_count
            else None
        )
        chunk_failed = 0
        chunk_corrected = set()

        for byte_index in range(bytes_per_strand):
            column = bytearray(
                observations[global_index][byte_index]
                if global_index in observations
                else 0
                for global_index in chunk_indices
            )
            try:
                if codec is not None:
                    decoded_result = codec.decode(
                        column, erase_pos=missing_local
                    )
                    decoded_column = decoded_result[0]
                    missing_set = set(missing_local)
                    for local_index in decoded_result[2]:
                        if local_index not in missing_set:
                            chunk_corrected.add(int(local_index))
                            corrected_symbols += 1
                elif missing_local:
                    raise reedsolo.ReedSolomonError(
                        "erasures cannot be recovered without parity"
                    )
                else:
                    decoded_column = column[: len(data_indices)]
                for local_data_index, global_data_index in enumerate(
                    data_indices
                ):
                    decoded_data[global_data_index].append(
                        decoded_column[local_data_index]
                    )
                decoded_columns.append(
                    {"chunk": chunk_index, "byte": byte_index}
                )
            except reedsolo.ReedSolomonError:
                failed_columns += 1
                chunk_failed += 1
                unresolved_columns.append(
                    {"chunk": chunk_index, "byte": byte_index}
                )
                for global_data_index in data_indices:
                    observed = observations.get(global_data_index)
                    decoded_data[global_data_index].append(
                        observed[byte_index] if observed is not None else 0
                    )

        chunk_details.append(
            {
                "chunk": chunk_index,
                "positions": list(chunk_indices),
                "data_indices": list(data_indices),
                "parity_indices": list(parity_indices),
                "missing_positions": [
                    chunk_indices[index] for index in missing_local
                ],
                "corrected_positions": [
                    chunk_indices[index] for index in sorted(chunk_corrected)
                ],
                "failed_columns": chunk_failed,
                "mode": "failed" if chunk_failed else "hard",
            }
        )

    result = [
        bytes_to_dna(strand, padding_bases) for strand in decoded_data
    ]
    diagnostics = {
        "algorithm": "hard",
        "observed_positions": len(observations),
        "erasures": total_count - len(observations),
        "corrected_symbols": corrected_symbols,
        "failed_columns": failed_columns,
        "column_count": len(layout) * bytes_per_strand,
        "decoded_columns": decoded_columns,
        "unresolved_columns": unresolved_columns,
        "output_is_partial": bool(unresolved_columns),
        "chunks": len(layout),
        "failed_chunks": [
            detail["chunk"]
            for detail in chunk_details
            if detail["failed_columns"]
        ],
        "chunk_details": chunk_details,
    }
    if return_diagnostics:
        return result, diagnostics
    return result


def _normalize_posterior_risks(
    byte_posterior_risks,
    observations,
    bytes_per_strand,
):
    normalized = {}
    for global_index in observations:
        raw_risks = byte_posterior_risks.get(
            global_index,
            byte_posterior_risks.get(str(global_index)),
        )
        if raw_risks is None:
            raise ValueError(
                f"RS position {global_index} is missing byte posterior risks"
            )
        posterior_risks = tuple(float(value) for value in raw_risks)
        if len(posterior_risks) != bytes_per_strand:
            raise ValueError(
                f"RS position {global_index} has {len(posterior_risks)} byte "
                f"posterior risks, expected {bytes_per_strand}"
            )
        if any(not math.isfinite(value) for value in posterior_risks):
            raise ValueError(
                f"RS position {global_index} contains a non-finite byte "
                "posterior risk"
            )
        normalized[global_index] = posterior_risks
    return normalized


def vertical_rs_decode_adaptive(
    received,
    byte_posterior_risks,
    data_strand_count,
    parity_strand_count,
    padding_bases,
    strand_length=None,
    return_diagnostics=False,
):
    """Adapt erasures by posterior order only after hard decoding fails."""
    total_count = data_strand_count + parity_strand_count
    layout = rs_chunk_layout(data_strand_count, parity_strand_count)
    observations, strand_length = _normalize_hard_observations(
        received,
        total_count,
        strand_length,
    )
    bytes_per_strand = len(dna_to_bytes("A" * strand_length)[0])
    posterior_risks = _normalize_posterior_risks(
        byte_posterior_risks,
        observations,
        bytes_per_strand,
    )
    decoded_data = [bytearray() for _ in range(data_strand_count)]
    hard_failed_columns = 0
    adaptive_columns = 0
    adaptive_attempts = 0
    recovered_columns = 0
    failed_columns = 0
    corrected_symbols = 0
    accepted_erasure_counts = Counter()
    chunk_details = []
    decoded_columns = []
    unresolved_columns = []

    for chunk_index, (data_indices, parity_indices) in enumerate(layout):
        chunk_indices = data_indices + parity_indices
        missing_local = [
            local_index
            for local_index, global_index in enumerate(chunk_indices)
            if global_index not in observations
        ]
        missing_set = set(missing_local)
        chunk_parity_count = len(parity_indices)
        additional_erasure_limit = max(
            0,
            chunk_parity_count - len(missing_local),
        )
        codec = (
            reedsolo.RSCodec(chunk_parity_count)
            if chunk_parity_count
            else None
        )
        chunk_hard_failed = 0
        chunk_adaptive_columns = 0
        chunk_recovered = 0
        chunk_failed = 0
        chunk_corrected = set()

        for byte_index in range(bytes_per_strand):
            column = bytearray(
                observations[global_index][byte_index]
                if global_index in observations
                else 0
                for global_index in chunk_indices
            )
            decoded_result = None
            column_is_unresolved = False
            try:
                if codec is not None:
                    decoded_result = codec.decode(
                        column,
                        erase_pos=missing_local,
                    )
                elif missing_local:
                    raise reedsolo.ReedSolomonError(
                        "erasures cannot be recovered without parity"
                    )
                else:
                    decoded_column = column[: len(data_indices)]
            except reedsolo.ReedSolomonError:
                hard_failed_columns += 1
                chunk_hard_failed += 1
                ranked_local_positions = [
                    local_index
                    for local_index, global_index in enumerate(chunk_indices)
                    if global_index in observations
                ]
                ranked_local_positions.sort(
                    key=lambda local_index: (
                        -posterior_risks[
                            chunk_indices[local_index]
                        ][byte_index],
                        chunk_indices[local_index],
                    )
                )
                if additional_erasure_limit:
                    adaptive_columns += 1
                    chunk_adaptive_columns += 1
                for erase_count in range(1, additional_erasure_limit + 1):
                    adaptive_attempts += 1
                    try:
                        decoded_result = codec.decode(
                            column,
                            erase_pos=sorted(
                                missing_local
                                + ranked_local_positions[:erase_count]
                            ),
                        )
                    except reedsolo.ReedSolomonError:
                        continue
                    accepted_erasure_counts[erase_count] += 1
                    recovered_columns += 1
                    chunk_recovered += 1
                    break

                if decoded_result is None:
                    failed_columns += 1
                    chunk_failed += 1
                    column_is_unresolved = True
                    decoded_column = bytearray(
                        observations[global_data_index][byte_index]
                        if global_data_index in observations
                        else 0
                        for global_data_index in data_indices
                    )

            if decoded_result is not None:
                decoded_column = decoded_result[0]
                for local_index in decoded_result[2]:
                    if local_index not in missing_set:
                        chunk_corrected.add(int(local_index))
                        corrected_symbols += 1
            for local_data_index, global_data_index in enumerate(data_indices):
                decoded_data[global_data_index].append(
                    decoded_column[local_data_index]
                )
            column_address = {"chunk": chunk_index, "byte": byte_index}
            if column_is_unresolved:
                unresolved_columns.append(column_address)
            else:
                decoded_columns.append(column_address)

        chunk_details.append(
            {
                "chunk": chunk_index,
                "positions": list(chunk_indices),
                "data_indices": list(data_indices),
                "parity_indices": list(parity_indices),
                "missing_positions": [
                    chunk_indices[index] for index in missing_local
                ],
                "adaptive_erasure_budget": additional_erasure_limit,
                "corrected_positions": [
                    chunk_indices[index] for index in sorted(chunk_corrected)
                ],
                "hard_failed_columns": chunk_hard_failed,
                "adaptive_columns": chunk_adaptive_columns,
                "recovered_columns": chunk_recovered,
                "failed_columns": chunk_failed,
            }
        )

    result = [
        bytes_to_dna(strand, padding_bases) for strand in decoded_data
    ]
    diagnostics = {
        "algorithm": "hard_then_adaptive_erasures",
        "observed_positions": len(observations),
        "erasures": total_count - len(observations),
        "corrected_symbols": corrected_symbols,
        "hard_failed_columns": hard_failed_columns,
        "adaptive_columns": adaptive_columns,
        "adaptive_attempts": adaptive_attempts,
        "recovered_columns": recovered_columns,
        "failed_columns": failed_columns,
        "column_count": len(layout) * bytes_per_strand,
        "decoded_columns": decoded_columns,
        "unresolved_columns": unresolved_columns,
        "output_is_partial": bool(unresolved_columns),
        "accepted_erasure_count_histogram": {
            str(key): value
            for key, value in sorted(accepted_erasure_counts.items())
        },
        "chunks": len(layout),
        "failed_chunks": [
            detail["chunk"]
            for detail in chunk_details
            if detail["failed_columns"]
        ],
        "chunk_details": chunk_details,
    }
    if return_diagnostics:
        return result, diagnostics
    return result


__all__ = [
    "bytes_to_dna",
    "dna_to_bytes",
    "rs_chunk_layout",
    "vertical_rs_decode_adaptive",
    "vertical_rs_decode_hard",
    "vertical_rs_encode",
]
