"""Pool-boundary and interval algorithms for index recovery."""

from array import array
import bisect
from collections import Counter, defaultdict

from ..header import cycle_count
from ..tree import compact_tree_order_key
from .frequency_aggregation import HEADER_NT


CYCLE_PERIOD = 64
RESCUE_EXPECTED_LONG_GAP_BUDGET = 1.0
_HEADER_NT = HEADER_NT
_CYCLE_PERIOD = CYCLE_PERIOD
_RESCUE_EXPECTED_LONG_GAP_BUDGET = (
    RESCUE_EXPECTED_LONG_GAP_BUDGET
)


def _decoded_tree_order_key(strand, result_index, tree_start, tree_length):
    payload = strand[_HEADER_NT:]
    return (
        compact_tree_order_key(payload, tree_start, tree_length),
        payload,
        strand,
        result_index,
    )

def _normalize_decoded_strand(strand):
    if not isinstance(strand, str):
        return None
    normalized = strand.upper()
    if len(normalized) < _HEADER_NT or any(
        base not in "ATGC" for base in normalized
    ):
        return None
    return normalized


def _tree_rank_by_strand_index(tree_seed):
    strand_count = len(tree_seed)
    if strand_count <= 0:
        raise ValueError("tree_seed must not be empty")
    rank_by_target = array("i", [-1]) * strand_count
    for tree_rank, one_based_target in enumerate(tree_seed):
        target = int(one_based_target) - 1
        if not 0 <= target < strand_count or rank_by_target[target] >= 0:
            raise ValueError("tree_seed must be a one-based permutation")
        rank_by_target[target] = tree_rank
    return rank_by_target


def _next_rank_with_cycle(cursor, cycle):
    rank = cursor + (cycle - cursor) % _CYCLE_PERIOD
    return rank + _CYCLE_PERIOD if rank == cursor else rank


def _previous_rank_with_cycle(cursor, cycle):
    rank = cursor - (cursor - cycle) % _CYCLE_PERIOD
    return rank - _CYCLE_PERIOD if rank == cursor else rank


def _maximum_cardinality_interval_ranks(
    header_order,
    cycle_by_result_index,
    frequency_by_result_index,
    left_rank,
    right_rank,
):
    """Drop interval outliers through one exact sparse monotone alignment.

    Candidate/rank edges exist only when their original modulo-64 cycles
    agree. Paths are compared by retained candidate count, retained payload
    frequency, and finally distance from the interval's scaled expectation.
    """
    candidate_count = len(header_order)
    lower_rank = left_rank + 1
    upper_rank = right_rank - 1
    open_rank_count = max(0, upper_rank - lower_rank + 1)
    if candidate_count == 0 or open_rank_count == 0:
        return {}, {
            "candidate_count": candidate_count,
            "open_rank_count": open_rank_count,
            "state_count": 0,
            "selected_count": 0,
        }

    fenwick = array("i", [-1]) * (open_rank_count + 1)
    node_matches = array("I")
    node_frequency = array("Q")
    node_cost = array("q")
    node_predecessor = array("i")
    result_index_by_ordinal = sorted(header_order)
    result_index_ordinal = {
        result_index: ordinal
        for ordinal, result_index in enumerate(result_index_by_ordinal)
    }
    node_result_ordinal = array("I")
    node_rank = array("I")

    def is_better(left, right):
        if right < 0:
            return left >= 0
        if left < 0:
            return False
        if node_matches[left] != node_matches[right]:
            return node_matches[left] > node_matches[right]
        if node_frequency[left] != node_frequency[right]:
            return node_frequency[left] > node_frequency[right]
        if node_cost[left] != node_cost[right]:
            return node_cost[left] < node_cost[right]
        if node_rank[left] != node_rank[right]:
            return node_rank[left] < node_rank[right]
        return node_result_ordinal[left] < node_result_ordinal[right]

    def query(coordinate):
        best = -1
        fenwick_index = coordinate
        while fenwick_index > 0:
            candidate = fenwick[fenwick_index]
            if is_better(candidate, best):
                best = candidate
            fenwick_index -= fenwick_index & -fenwick_index
        return best

    def update(coordinate, node_index):
        fenwick_index = coordinate + 1
        while fenwick_index <= open_rank_count:
            if is_better(node_index, fenwick[fenwick_index]):
                fenwick[fenwick_index] = node_index
            fenwick_index += fenwick_index & -fenwick_index

    rank_span = right_rank - left_rank
    denominator = candidate_count + 1
    for position, result_index in enumerate(header_order):
        expected_rank = left_rank + round(
            (position + 1) * rank_span / denominator
        )
        cycle = cycle_by_result_index[result_index]
        first_rank = lower_rank + (cycle - lower_rank) % _CYCLE_PERIOD
        pending_updates = []
        for rank in range(first_rank, upper_rank + 1, _CYCLE_PERIOD):
            coordinate = rank - lower_rank
            predecessor = query(coordinate)
            node_index = len(node_matches)
            node_matches.append(
                (node_matches[predecessor] if predecessor >= 0 else 0) + 1
            )
            node_frequency.append(
                (
                    node_frequency[predecessor]
                    if predecessor >= 0
                    else 0
                )
                + int(frequency_by_result_index[result_index])
            )
            node_cost.append(
                (node_cost[predecessor] if predecessor >= 0 else 0)
                + abs(rank - expected_rank)
            )
            node_predecessor.append(predecessor)
            node_result_ordinal.append(result_index_ordinal[result_index])
            node_rank.append(rank)
            pending_updates.append((coordinate, node_index))
        # Do not let one candidate become its own predecessor.
        for coordinate, node_index in pending_updates:
            update(coordinate, node_index)

    best = query(open_rank_count)
    assignments = {}
    while best >= 0:
        result_index = result_index_by_ordinal[node_result_ordinal[best]]
        assignments[result_index] = node_rank[best]
        best = node_predecessor[best]
    return assignments, {
        "candidate_count": candidate_count,
        "open_rank_count": open_rank_count,
        "state_count": len(node_matches),
        "selected_count": len(assignments),
        "discarded_outlier_count": candidate_count - len(assignments),
        "objective": (
            "maximum_candidate_count_then_frequency_then_minimum_scaled_cost"
        ),
    }


def _exact_bidirectional_interval(
    interior,
    cycles,
    left_rank,
    right_rank,
    left_anchor,
    right_anchor,
    strand_count,
):
    """Return an interval only when both phase unwraps are identical."""
    forward = {}
    cursor = left_rank
    for result_index in interior:
        rank = _next_rank_with_cycle(cursor, cycles[result_index])
        if not cursor < rank < right_rank or rank >= strand_count:
            return None, "forward_does_not_reach_right_anchor"
        forward[result_index] = rank
        cursor = rank
    if right_anchor is not None and _next_rank_with_cycle(
        cursor, cycles[right_anchor]
    ) != right_rank:
        return None, "forward_does_not_reach_right_anchor"

    reverse = {}
    cursor = right_rank
    for result_index in reversed(interior):
        rank = _previous_rank_with_cycle(cursor, cycles[result_index])
        if not left_rank < rank < cursor or rank < 0:
            return None, "reverse_does_not_reach_left_anchor"
        reverse[result_index] = rank
        cursor = rank
    if left_anchor is not None and _previous_rank_with_cycle(
        cursor, cycles[left_anchor]
    ) != left_rank:
        return None, "reverse_does_not_reach_left_anchor"
    if forward != reverse:
        return None, "bidirectional_rank_disagreement"
    return forward, None


def _pool_interval_record(left_rank, right_rank, candidate_count):
    return {
        "left_boundary_kind": "pool_start",
        "left_boundary_result_index": None,
        "left_boundary_rank": left_rank,
        "right_boundary_kind": "pool_end",
        "right_boundary_result_index": None,
        "right_boundary_rank": right_rank,
        "interior_candidate_count": candidate_count,
    }


def _pool_bounded_interval_ranks(
    header_order,
    cycle_by_result_index,
    frequency_by_result_index,
    strand_count,
    enable_outlier_filter=False,
):
    """Recover the full pool from both ends, optionally filtering outliers."""
    additions = {}
    stages = {}
    accepted = []
    rejected = []
    left_rank = -1
    right_rank = strand_count
    interior = header_order
    record = _pool_interval_record(left_rank, right_rank, len(interior))
    if interior:
        exact, failure = _exact_bidirectional_interval(
            interior,
            cycle_by_result_index,
            left_rank,
            right_rank,
            None,
            None,
            strand_count,
        )
        if exact is not None:
            method = "initial_bidirectional_interval_alignment"
            recovered = exact
            record["recovery_method"] = "exact_bidirectional_rank_agreement"
        else:
            record.update(
                {
                    "exact_failure_reason": failure,
                    "outlier_filter_enabled": bool(enable_outlier_filter),
                }
            )
            if not enable_outlier_filter:
                record["reason"] = "strict_bidirectional_rank_disagreement"
                rejected.append(record)
                recovered = None
            else:
                open_rank_count = strand_count
                raw_density = len(interior) / open_rank_count
                density = min(raw_density, 1.0)
                expected_long_gap_count = (
                    open_rank_count
                    * density
                    * ((1.0 - density) ** _CYCLE_PERIOD)
                )
                alias_risk_accepted = (
                    expected_long_gap_count
                    <= _RESCUE_EXPECTED_LONG_GAP_BUDGET
                )
                record.update(
                    {
                        "outlier_fallback_open_rank_count": open_rank_count,
                        "outlier_fallback_candidate_count": len(interior),
                        "outlier_fallback_raw_candidate_density": raw_density,
                        "outlier_fallback_candidate_density": density,
                        "outlier_fallback_cycle_period": _CYCLE_PERIOD,
                        "outlier_fallback_expected_long_gap_count": (
                            expected_long_gap_count
                        ),
                        "outlier_fallback_expected_long_gap_budget": (
                            _RESCUE_EXPECTED_LONG_GAP_BUDGET
                        ),
                        "outlier_fallback_alias_risk_accepted": (
                            alias_risk_accepted
                        ),
                    }
                )
                if not alias_risk_accepted:
                    record["reason"] = (
                        "outlier_fallback_expected_long_gap_budget_exceeded"
                    )
                    rejected.append(record)
                    recovered = None
                else:
                    recovered, fallback_diagnostics = (
                        _maximum_cardinality_interval_ranks(
                            interior,
                            cycle_by_result_index,
                            frequency_by_result_index,
                            left_rank,
                            right_rank,
                        )
                    )
                    record["outlier_fallback"] = fallback_diagnostics
                    if not recovered:
                        record["reason"] = "outlier_fallback_returned_no_rank"
                        rejected.append(record)
                        recovered = None
                    else:
                        method = "initial_sparse_monotone_interval_alignment"
                        record["recovery_method"] = (
                            "maximum_cardinality_monotone_outlier_filter"
                        )

        if recovered is not None:
            additions.update(recovered)
            stages.update(dict.fromkeys(recovered, method))
            record["added_count"] = len(recovered)
            accepted.append(record)

    rejection_reasons = Counter(record["reason"] for record in rejected)
    diagnostics = {
        "cycle_period": _CYCLE_PERIOD,
        "pool_boundary_count": 2,
        "bounded_interval_candidate_count": len(interior),
        "accepted_interval_count": len(accepted),
        "rejected_interval_count": len(rejected),
        "rejection_reason_counts": dict(sorted(rejection_reasons.items())),
        "added_rank_count": len(additions),
        "addition_recovery_method_counts": dict(
            sorted(Counter(stages.values()).items())
        ),
        "outlier_filter_enabled": bool(enable_outlier_filter),
        "outlier_filter_used": any(
            record.get("recovery_method")
            == "maximum_cardinality_monotone_outlier_filter"
            for record in accepted
        ),
        "pool_boundaries_used": True,
        "accepted_intervals": accepted,
        "rejected_intervals": rejected,
    }
    return additions, stages, diagnostics


def _recover_initial_assignments(
    normalized,
    frequencies,
    candidate_indices,
    global_tree_order,
    tree_seed,
    tree_start,
    tree_length,
    enable_outlier_filter=False,
):
    """Recover one frequency prefix under either strict or rescue rules."""
    strand_count = len(tree_seed)
    candidate_set = set(candidate_indices)
    header_order = [
        result_index
        for result_index in global_tree_order
        if result_index in candidate_set
    ]
    cycle_by_result_index = {
        result_index: cycle_count(normalized[result_index])
        for result_index in header_order
    }
    frequency_by_result_index = {
        result_index: int(frequencies[result_index])
        for result_index in header_order
    }
    recovered_ranks, recovered_stages, interval_diagnostics = (
        _pool_bounded_interval_ranks(
            header_order,
            cycle_by_result_index,
            frequency_by_result_index,
            strand_count,
            enable_outlier_filter=enable_outlier_filter,
        )
    )
    assignments = {}
    stages = {}
    for result_index, tree_rank in recovered_ranks.items():
        target = int(tree_seed[tree_rank]) - 1
        assignments[target] = result_index
        stages[target] = recovered_stages[result_index]

    diagnostics = {
        "candidate_count": len(candidate_indices),
        "normalized_candidate_count": len(candidate_indices),
        "header_valid_candidate_count": len(header_order),
        "recovered_index_count": len(assignments),
        "outlier_filter_enabled": bool(enable_outlier_filter),
        "pool_bounded_bidirectional": interval_diagnostics,
        "recovery_method_counts": dict(
            sorted(Counter(stages.values()).items())
        ),
    }
    return assignments, stages, diagnostics


def _exact_interval_edges(
    candidate_indices,
    normalized_strands,
    decoded_order_keys,
    global_tree_order,
    placements,
    rank_by_target,
    strand_count,
):
    """Return exact tree-gap and original-cycle targets for candidates."""
    target_by_result = {
        result_index: target
        for target, result_index in placements.items()
        if result_index in decoded_order_keys
    }
    anchor_records = [
        (
            decoded_order_keys[result_index],
            rank_by_target[target],
            target,
            result_index,
        )
        for result_index in global_tree_order
        if (target := target_by_result.get(result_index)) is not None
    ]
    anchor_order_keys = tuple(record[0] for record in anchor_records)
    anchor_ranks = tuple(record[1] for record in anchor_records)
    ranks_are_monotone = all(
        left < right for left, right in zip(anchor_ranks, anchor_ranks[1:])
    )
    if not ranks_are_monotone:
        return {}, {
            "anchor_count": len(anchor_records),
            "anchor_ranks_strictly_increasing": False,
            "open_target_count": strand_count - len(placements),
            "candidate_edge_count": 0,
        }

    open_targets = set(range(strand_count)) - set(placements)
    targets_by_cycle_and_interval = defaultdict(set)
    for target in open_targets:
        tree_rank = rank_by_target[target]
        interval = bisect.bisect_left(anchor_ranks, tree_rank)
        targets_by_cycle_and_interval[
            (tree_rank % _CYCLE_PERIOD, interval)
        ].add(target)

    edges = {}
    for result_index in candidate_indices:
        if result_index not in decoded_order_keys:
            continue
        decoded_interval = bisect.bisect_left(
            anchor_order_keys,
            decoded_order_keys[result_index],
        )
        cycle = cycle_count(normalized_strands[result_index])
        edges[result_index] = set(
            targets_by_cycle_and_interval.get(
                (cycle, decoded_interval),
                (),
            )
        )
    return edges, {
        "anchor_count": len(anchor_records),
        "anchor_ranks_strictly_increasing": True,
        "open_target_count": len(open_targets),
        "candidate_edge_count": sum(map(len, edges.values())),
    }


def _mutual_singletons(edges):
    claimants = defaultdict(list)
    for result_index, targets in edges.items():
        if len(targets) == 1:
            claimants[next(iter(targets))].append(result_index)
    return {
        target: result_indices[0]
        for target, result_indices in claimants.items()
        if len(result_indices) == 1
    }
