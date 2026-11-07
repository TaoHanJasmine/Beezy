"""Protocol-v2 scoring using provenance, never sequence similarity, for indices."""

from collections.abc import Mapping
from numbers import Integral


METRIC_REVISION = 'fixed-position-bytes-v1'
ORDERING_METRIC_REVISION = 'source-reads-correct-over-assigned-support-v1'


def score_bytes(reference, ordered, original_byte_count=None):
    """Count equal valid bytes at fixed slots, with no alignment or shifting.

    Four DNA bases encode one byte bijectively. Comparing fixed 4-base groups
    is exactly byte equality, not per-base identity. For file XOR, applying
    the same position's inverse mask to both bytes preserves equality, so
    no source-truth index or decoded-byte guess is needed here.
    """
    capacities = [len(payload)//4 for payload in reference]
    if any(len(payload)%4 or set(payload)-set('ACGT') for payload in reference):
        raise ValueError('reference payloads must encode complete valid bytes')
    total = sum(capacities) if original_byte_count is None else original_byte_count
    if not isinstance(total,int) or total <= 0 or total > sum(capacities):
        raise ValueError('invalid reference byte count')
    if any(not 0 <= target < len(reference) for target in ordered):
        raise ValueError('target outside reference pool')
    rows=[]; correct=valid=extra=0; offset=0
    for target,(payload,capacity) in enumerate(zip(reference,capacities)):
        size=max(0,min(capacity,total-offset))
        recovered=ordered.get(target)
        mask=[]; equal=[]
        for j in range(size):
            group=recovered[4*j:4*j+4] if recovered is not None else ''
            present=len(group)==4 and not (set(group)-set('ACGT'))
            mask.append(bool(present))
            equal.append(bool(present and group==payload[4*j:4*j+4]))
        surplus=max(0,len(recovered)-len(payload)) if recovered is not None else 0
        correct+=sum(equal); valid+=sum(mask); extra+=surplus
        rows.append(dict(target=target,byte_offset=offset,reference_byte_count=size,
            correct_byte_count=sum(equal),valid_byte_mask=mask,correct_byte_mask=equal,
            recovered_nt=len(recovered) if recovered is not None else None,
            reference_nt=len(payload),extra_nt=surplus,
            incomplete_tail_nt=len(recovered)%4 if recovered is not None else 0))
        offset+=capacity
    return dict(metric_revision=METRIC_REVISION,recovery_unit='byte',
        recovery_rate=correct/total,correct_byte_count=correct,reference_byte_count=total,
        valid_recovered_byte_count=valid,missing_or_invalid_byte_count=total-valid,
        extra_nt=extra),rows


def score_pool(reference, ordered, assignments, candidate_sources, candidates, original_byte_count=None):
    result = score_indices(len(reference), assignments, candidate_sources)
    if set(ordered) != {target for target, _ in assignments}:
        raise ValueError("ordered output and assignments disagree")
    for target, candidate in assignments:
        if ordered[target] != candidates[candidate][4:]:
            raise ValueError("candidate payload differs from ordered output")
    byte_metrics,per_position=score_bytes(reference,ordered,original_byte_count)
    correct_payloads = set(reference)
    good = sum(candidate[4:] in correct_payloads for candidate in candidates)
    result.update(byte_metrics)
    result.update(candidate_accuracy=good/len(candidates) if candidates else None,
        candidate_count=len(candidates),correct_candidate_count=good,
        wrong_candidate_count=len(candidates)-good,
        exact_slot_payload_match_count=sum(value==reference[target] for target,value in ordered.items()))
    return result, per_position


def score_indices(reference_count, assignments, candidate_sources):
    """Score source-read histograms at zero-based assigned positions."""
    if not isinstance(reference_count, Integral) or isinstance(reference_count, bool) or reference_count <= 0:
        raise ValueError("reference_count must be a positive integer")
    assignments = list(assignments)
    targets = [target for target, _ in assignments]
    candidates = [candidate for _, candidate in assignments]
    if len(set(targets)) != len(targets) or len(set(candidates)) != len(candidates):
        raise ValueError("assignments must be one-to-one")
    correct = wrong = ambiguous = correct_reads = assigned_reads = 0
    for target, candidate in assignments:
        if not isinstance(target, Integral) or isinstance(target, bool) or not 0 <= target < reference_count:
            raise ValueError("target outside reference pool")
        if not isinstance(candidate, Integral) or isinstance(candidate, bool) or not 0 <= candidate < len(candidate_sources):
            raise ValueError("candidate outside candidate pool")
        sources = candidate_sources[candidate]
        if not isinstance(sources, Mapping):
            raise TypeError("candidate provenance must map source indices to read counts")
        if not sources:
            raise ValueError("candidate is missing provenance")
        for source, frequency in sources.items():
            if not isinstance(source, Integral) or isinstance(source, bool) or not 0 <= source < reference_count:
                raise ValueError("source outside reference pool")
            if not isinstance(frequency, Integral) or isinstance(frequency, bool) or frequency <= 0:
                raise ValueError("source read counts must be positive integers")
        assigned_reads += sum(int(frequency) for frequency in sources.values())
        correct_reads += int(sources.get(target, 0))
        if len(sources) != 1:
            ambiguous += 1
        elif target in sources:
            correct += 1
        else:
            wrong += 1
    count = len(assignments)
    accuracy = correct_reads / assigned_reads if assigned_reads else None
    return {
        "reference_count": reference_count,
        "recovered_count": count,
        "correctly_ordered_count": correct,
        "wrongly_ordered_count": wrong,
        "ambiguous_count": ambiguous,
        "coverage": count / reference_count,
        "ordering_accuracy": accuracy,
        "ordering_accuracy_lower": accuracy,
        "ordering_accuracy_upper": accuracy,
        "weighted_correct_read_count": correct_reads,
        "weighted_assigned_read_count": assigned_reads,
        "ordering_metric_revision": ORDERING_METRIC_REVISION,
    }
