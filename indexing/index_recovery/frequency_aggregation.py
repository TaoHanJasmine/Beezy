"""Payload-first frequency aggregation for production index recovery."""

from collections import Counter

from ..header import header_checksum_is_valid


HEADER_NT = 4
MINIMUM_FREQUENCY = 2
_HEADER_NT = HEADER_NT
_MINIMUM_FREQUENCY = MINIMUM_FREQUENCY


def _add_header_variant(group, decoded_strand, raw_frequency, result_index):
    header = decoded_strand[:_HEADER_NT]
    variant = group["headers"].get(header)
    if variant is None:
        group["headers"][header] = {
            "frequency": raw_frequency,
            "representative_result_index": result_index,
            "selected_strand": decoded_strand,
        }
    else:
        variant["frequency"] += raw_frequency


def aggregate_bcjr_payloads_by_frequency(
    raw_unique_reads,
    bcjr_results,
    tree_start=0,
    tree_length=None,
    minimum_payload_frequency=MINIMUM_FREQUENCY,
):
    """Select one observed header for every unambiguous repeated payload.

    ``raw_unique_reads`` and ``bcjr_results`` are parallel arrays. The read
    frequency weights its hard output, so decoding still runs only once per
    distinct raw read. Payload support is fully merged before the inclusive
    ``minimum_payload_frequency`` gate is applied.
    """
    minimum_payload_frequency = int(minimum_payload_frequency)
    if minimum_payload_frequency <= 0:
        raise ValueError("minimum_payload_frequency must be positive")
    if len(raw_unique_reads) != len(bcjr_results):
        raise ValueError(
            "raw read frequencies and BCJR results must have equal length"
        )

    singletons = {}
    groups = {}
    failed_unique_read_count = 0
    failed_read_count = 0
    for result_index, ((_, raw_frequency), result) in enumerate(
        zip(raw_unique_reads, bcjr_results)
    ):
        raw_frequency = int(raw_frequency)
        if raw_frequency <= 0:
            raise ValueError("raw read frequency must be positive")
        decoded_strand = result[0]
        if not isinstance(decoded_strand, str) or len(decoded_strand) < _HEADER_NT:
            failed_unique_read_count += 1
            failed_read_count += raw_frequency
            continue

        payload = decoded_strand[_HEADER_NT:]
        group = groups.get(payload)
        if group is not None:
            group["payload_frequency"] += raw_frequency
            _add_header_variant(
                group,
                decoded_strand,
                raw_frequency,
                result_index,
            )
            continue

        first_result_index = singletons.get(payload)
        if first_result_index is None:
            if raw_frequency < minimum_payload_frequency:
                singletons[payload] = result_index
                continue
            group = {
                "payload_frequency": raw_frequency,
                "headers": {},
            }
            groups[payload] = group
            _add_header_variant(
                group,
                decoded_strand,
                raw_frequency,
                result_index,
            )
            continue

        first_frequency = int(raw_unique_reads[first_result_index][1])
        first_decoded_strand = bcjr_results[first_result_index][0]
        group = {
            "payload_frequency": first_frequency + raw_frequency,
            "headers": {},
        }
        groups[payload] = group
        del singletons[payload]
        _add_header_variant(
            group,
            first_decoded_strand,
            first_frequency,
            first_result_index,
        )
        _add_header_variant(
            group,
            decoded_strand,
            raw_frequency,
            result_index,
        )

    selected = []
    decision_counts = Counter()
    eligible_groups = {
        payload: group
        for payload, group in groups.items()
        if group["payload_frequency"] >= minimum_payload_frequency
    }
    for payload, group in eligible_groups.items():
        headers = group["headers"]
        maximum_header_frequency = max(
            variant["frequency"] for variant in headers.values()
        )
        highest_headers = [
            header
            for header, variant in headers.items()
            if variant["frequency"] == maximum_header_frequency
        ]
        selected_header = None
        reason = None
        if (
            maximum_header_frequency >= minimum_payload_frequency
            and len(highest_headers) == 1
        ):
            selected_header = highest_headers[0]
            reason = (
                "unique_highest_header_frequency_at_least_two"
                if minimum_payload_frequency == MINIMUM_FREQUENCY
                else "unique_highest_header_frequency_at_least_threshold"
            )
        elif maximum_header_frequency >= minimum_payload_frequency:
            decision_counts["tied_highest_header_frequency_discarded"] += 1
        else:
            checksum_valid_headers = [
                header
                for header in headers
                if header_checksum_is_valid(
                    header + payload,
                    tree_start=tree_start,
                    tree_length=tree_length,
                )
            ]
            if len(checksum_valid_headers) == 1:
                selected_header = checksum_valid_headers[0]
                reason = (
                    "all_headers_frequency_one_unique_checksum_valid"
                    if minimum_payload_frequency == MINIMUM_FREQUENCY
                    else "all_headers_below_threshold_unique_checksum_valid"
                )
            elif not checksum_valid_headers:
                key = (
                    "all_headers_frequency_one_no_checksum_valid_discarded"
                    if minimum_payload_frequency == MINIMUM_FREQUENCY
                    else "all_headers_below_threshold_no_checksum_valid_discarded"
                )
                decision_counts[key] += 1
            else:
                key = (
                    "all_headers_frequency_one_multiple_checksum_valid_discarded"
                    if minimum_payload_frequency == MINIMUM_FREQUENCY
                    else "all_headers_below_threshold_multiple_checksum_valid_discarded"
                )
                decision_counts[key] += 1

        if selected_header is None:
            continue
        decision_counts[reason] += 1
        variant = headers[selected_header]
        selected.append(
            (
                variant["selected_strand"],
                variant["representative_result_index"],
                group["payload_frequency"],
                variant["frequency"],
                reason,
            )
        )

    selected.sort(key=lambda item: (-item[2], item[0]))
    retained_results = [bcjr_results[item[1]] for item in selected]
    retained_frequencies = [item[2] for item in selected]
    payload_frequency_histogram = Counter()
    payload_frequency_histogram.update(
        int(raw_unique_reads[result_index][1])
        for result_index in singletons.values()
    )
    payload_frequency_histogram.update(
        group["payload_frequency"] for group in groups.values()
    )
    diagnostics = {
        "unique_payload_count": len(singletons) + len(groups),
        "minimum_payload_frequency": minimum_payload_frequency,
    }
    discarded_below_threshold = (
        len(singletons) + len(groups) - len(eligible_groups)
    )
    if minimum_payload_frequency == MINIMUM_FREQUENCY:
        diagnostics["discarded_singleton_payload_count"] = (
            discarded_below_threshold
        )
    else:
        diagnostics["discarded_below_threshold_payload_count"] = (
            discarded_below_threshold
        )
    diagnostics.update({
        "retained_payload_count": len(eligible_groups),
        "selected_candidate_count": len(selected),
        "discarded_ambiguous_payload_count": (
            len(eligible_groups) - len(selected)
        ),
        "total_selected_read_support": sum(retained_frequencies),
        "maximum_selected_payload_frequency": max(
            retained_frequencies, default=0
        ),
        "frequency_histogram": dict(
            sorted(payload_frequency_histogram.items())
        ),
        "representative_decision_counts": dict(
            sorted(decision_counts.items())
        ),
        "bcjr_failed_unique_read_count": failed_unique_read_count,
        "bcjr_failed_read_count": failed_read_count,
    })
    return retained_results, retained_frequencies, diagnostics



__all__ = [
    "HEADER_NT",
    "MINIMUM_FREQUENCY",
    "aggregate_bcjr_payloads_by_frequency",
]
