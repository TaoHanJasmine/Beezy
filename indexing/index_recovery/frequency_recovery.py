"""Frequency-layer scheduling and global strand-index recovery."""

from collections import Counter, defaultdict
from time import perf_counter

from ..header import header_checksum_is_valid
from .frequency_aggregation import (
    HEADER_NT as _HEADER_NT,
    MINIMUM_FREQUENCY as _MINIMUM_FREQUENCY,
)
from .frequency_intervals import (
    _exact_interval_edges,
    _decoded_tree_order_key,
    _mutual_singletons,
    _normalize_decoded_strand,
    _recover_initial_assignments,
    _tree_rank_by_strand_index,
)

_INITIAL_PREFIX_NUMERATOR = 1
_INITIAL_PREFIX_DENOMINATOR = 4
_INITIAL_RECOVERY_OCCUPANCY_NUMERATOR = 1
_INITIAL_RECOVERY_OCCUPANCY_DENOMINATOR = 5
_RESCUE_TRIGGER_NUMERATOR = 1
_RESCUE_TRIGGER_DENOMINATOR = 2
_FREQUENCY_ONE_DUMMY_COST = 100000


def _ceiling_fraction(total, numerator, denominator):
    return (total * numerator + denominator - 1) // denominator


def _initial_prefix_count(strand_count):
    return _ceiling_fraction(
        strand_count,
        _INITIAL_PREFIX_NUMERATOR,
        _INITIAL_PREFIX_DENOMINATOR,
    )


def _occupancy_count(strand_count):
    return _ceiling_fraction(
        strand_count,
        _INITIAL_RECOVERY_OCCUPANCY_NUMERATOR,
        _INITIAL_RECOVERY_OCCUPANCY_DENOMINATOR,
    )


def _initial_frequency_prefix(retained_indices, frequencies, target_count):
    levels = sorted({frequencies[index] for index in retained_indices}, reverse=True)
    if not target_count:
        return levels, None
    cumulative = 0
    for level in levels:
        cumulative += sum(frequencies[index] == level for index in retained_indices)
        if cumulative >= target_count:
            return levels, level
    raise RuntimeError("retained candidates did not produce an initial frequency")


def _initial_recovery_attempt_decision(
    frequency,
    highest_frequency,
    candidate_count,
    recovered_index_count,
    occupancy_count,
):
    if recovered_index_count >= occupancy_count:
        return (
            "accept_minimum_initial_recovery_reached",
            "minimum_initial_recovery_reached",
        )
    if candidate_count < occupancy_count:
        return (
            "accept_all_available_candidates_below_minimum",
            "all_available_candidates_used_below_minimum",
        )
    if frequency == highest_frequency:
        return (
            "accept_highest_frequency_layer_below_minimum",
            "highest_frequency_layer_used_below_minimum",
        )
    return "retry_without_lowest_frequency_layer", None


def _run_strict_initial_recovery(
    normalized,
    frequencies,
    retained_indices,
    global_tree_order,
    frequency_levels,
    initial_frequency,
    tree_seed,
    tree_start,
    tree_length,
    occupancy_count,
):
    """Run strict initial recovery with complete-frequency-layer backoff."""
    selection = {
        "phase": "strict_alignment",
        "frequency": None,
        "acceptance_reason": None,
        "candidate_indices": [],
        "index_assignments": {},
        "recovery_methods": {},
        "initial_recovered_index_count": 0,
        "diagnostics": None,
        "build_attempts": [],
    }
    if initial_frequency is None:
        return selection

    initial_index = frequency_levels.index(initial_frequency)
    attempted_levels = reversed(frequency_levels[: initial_index + 1])
    for frequency in attempted_levels:
        candidate_indices = [
            index
            for index in retained_indices
            if frequencies[index] >= frequency
        ]
        local_assignments, local_stages, diagnostics = (
            _recover_initial_assignments(
                normalized,
                frequencies,
                candidate_indices,
                global_tree_order,
                tree_seed,
                tree_start,
                tree_length,
                enable_outlier_filter=False,
            )
        )
        decision, reason = _initial_recovery_attempt_decision(
            frequency,
            frequency_levels[0],
            len(candidate_indices),
            len(local_assignments),
            occupancy_count,
        )
        selection["build_attempts"].append(
            {
                "path": "strict_alignment",
                "sparse_alignment_enabled": False,
                "minimum_payload_frequency": frequency,
                "input_candidate_count": len(candidate_indices),
                "checksum_valid_candidate_count": diagnostics.get(
                    "header_valid_candidate_count", len(candidate_indices)
                ),
                "recovered_index_count": len(local_assignments),
                "candidate_count_to_total_index_count_ratio": (
                    len(candidate_indices) / len(tree_seed)
                ),
                "recovered_index_fraction": (
                    len(local_assignments) / len(tree_seed)
                ),
                "minimum_initial_recovered_index_count": occupancy_count,
                "minimum_initial_recovery_reached": (
                    len(local_assignments) >= occupancy_count
                ),
                "candidate_count_below_initial_recovery_minimum": (
                    len(candidate_indices) < occupancy_count
                ),
                "recovery_method_counts": diagnostics.get(
                    "recovery_method_counts", {}
                ),
                "outcome": decision,
            }
        )
        if decision.startswith("retry_"):
            continue

        selection.update(
            {
                "frequency": frequency,
                "acceptance_reason": reason,
                "candidate_indices": candidate_indices,
                "index_assignments": dict(local_assignments),
                "recovery_methods": dict(local_stages),
                "initial_recovered_index_count": len(local_assignments),
                "diagnostics": diagnostics,
            }
        )
        return selection
    return selection


def _run_sparse_initial_recovery(
    normalized,
    frequencies,
    retained_indices,
    global_tree_order,
    initial_frequency,
    tree_seed,
    tree_start,
    tree_length,
):
    """Run alias-risk-gated rescue once from the fixed initial prefix.

    The normal path's 20% occupancy backoff is intentionally absent here.
    Rescue feasibility is decided locally by the modulo-64 ``Egap`` model.
    """
    selection = {
        "phase": "sparse_alignment_rescue",
        "frequency": None,
        "acceptance_reason": None,
        "candidate_indices": [],
        "index_assignments": {},
        "recovery_methods": {},
        "initial_recovered_index_count": 0,
        "diagnostics": None,
        "build_attempts": [],
    }
    if initial_frequency is None:
        return selection

    candidate_indices = [
        index
        for index in retained_indices
        if frequencies[index] >= initial_frequency
    ]
    local_assignments, local_stages, diagnostics = (
        _recover_initial_assignments(
            normalized,
            frequencies,
            candidate_indices,
            global_tree_order,
            tree_seed,
            tree_start,
            tree_length,
            enable_outlier_filter=True,
        )
    )
    selection.update(
        {
            "frequency": initial_frequency,
            "acceptance_reason": "fixed_initial_prefix_sparse_alignment",
            "candidate_indices": candidate_indices,
            "index_assignments": dict(local_assignments),
            "recovery_methods": dict(local_stages),
            "initial_recovered_index_count": len(local_assignments),
            "diagnostics": diagnostics,
            "build_attempts": [
                {
                    "path": "sparse_alignment_rescue",
                    "sparse_alignment_enabled": True,
                    "minimum_payload_frequency": initial_frequency,
                    "input_candidate_count": len(candidate_indices),
                    "checksum_valid_candidate_count": diagnostics.get(
                        "header_valid_candidate_count", len(candidate_indices)
                    ),
                    "recovered_index_count": len(local_assignments),
                    "candidate_count_to_total_index_count_ratio": (
                        len(candidate_indices) / len(tree_seed)
                    ),
                    "recovered_index_fraction": (
                        len(local_assignments) / len(tree_seed)
                    ),
                    "recovery_method_counts": diagnostics.get(
                        "recovery_method_counts", {}
                    ),
                    "outcome": "accept_fixed_initial_prefix_sparse_alignment",
                }
            ],
        }
    )
    return selection


def _order_preserving_additions(placements, proposals, order_keys, ranks):
    """Defer every new proposal participating in an inversion; keep old anchors."""
    merged = {**placements, **proposals}
    targets = sorted(merged, key=lambda target: order_keys[merged[target]])
    conflicting = set()
    maximum = -1
    for target in targets:
        rank = ranks[target]
        if maximum >= rank:
            conflicting.add(target)
        maximum = max(maximum, rank)
    minimum = float("inf")
    for target in reversed(targets):
        rank = ranks[target]
        if minimum <= rank:
            conflicting.add(target)
        minimum = min(minimum, rank)
    return {target: candidate for target, candidate in proposals.items()
            if target not in conflicting}


def _commit_pending_to_fixed_point(
    pending_by_frequency,
    normalized,
    decoded_order_keys,
    global_tree_order,
    index_assignments,
    recovery_methods,
    rank_by_target,
    strand_count,
    protect_order=True,
):
    """Commit only candidate/target mutual singletons, high frequency first."""
    before_count = len(index_assignments)
    successful_rounds = 0
    blocked_edges = 0
    deferred_conflicts = 0
    while True:
        reserved_by_higher_frequency = set()
        for frequency in sorted(pending_by_frequency, reverse=True):
            pending = pending_by_frequency[frequency]
            if not pending:
                continue
            edges, _ = _exact_interval_edges(
                pending,
                normalized,
                decoded_order_keys,
                global_tree_order,
                index_assignments,
                rank_by_target,
                strand_count,
            )
            if reserved_by_higher_frequency:
                blocked_edges += sum(
                    len(targets & reserved_by_higher_frequency)
                    for targets in edges.values()
                )
                edges = {
                    result: targets - reserved_by_higher_frequency
                    for result, targets in edges.items()
                }
            additions = _mutual_singletons(edges)
            if protect_order:
                safe = _order_preserving_additions(
                    index_assignments, additions, decoded_order_keys, rank_by_target
                )
                deferred_conflicts += len(additions) - len(safe)
                additions = safe
            if additions:
                index_assignments.update(additions)
                recovery_methods.update(
                    {
                        target: (
                            f"frequency_{frequency}_"
                            "exact_interval_unique_match"
                        )
                        for target in additions
                    }
                )
                pending.difference_update(additions.values())
                successful_rounds += 1
                break
            reserved_by_higher_frequency.update(
                target for targets in edges.values() for target in targets
            )
        else:
            break
    return {
        "successful_assignment_rounds": successful_rounds,
        "newly_recovered_index_count": (
            len(index_assignments) - before_count
        ),
        "higher_frequency_reserved_edge_count": blocked_edges,
        "order_conflict_deferred_proposals": deferred_conflicts,
    }


def _descend_frequency_levels(
    selection,
    frequency_levels,
    frequencies,
    retained_indices,
    header_valid,
    normalized,
    decoded_order_keys,
    global_tree_order,
    rank_by_target,
    strand_count,
    protect_order=False,
):
    index_assignments = selection["index_assignments"]
    recovery_methods = selection["recovery_methods"]
    selected_frequency = selection["frequency"]
    pending = defaultdict(set)
    tiers = []
    if selected_frequency is None:
        return pending, tiers

    assigned_candidate_positions = set(index_assignments.values())
    for index in selection["candidate_indices"]:
        if index not in assigned_candidate_positions and header_valid[index]:
            pending[frequencies[index]].add(index)

    def commit(kind, frequency, introduced_count, introduced_valid_count):
        details = _commit_pending_to_fixed_point(
            pending,
            normalized,
            decoded_order_keys,
            global_tree_order,
            index_assignments,
            recovery_methods,
            rank_by_target,
            strand_count,
            protect_order=protect_order,
        )
        tiers.append(
            {
                "layer_source": kind,
                "payload_frequency": frequency,
                "introduced_candidate_count": introduced_count,
                "introduced_checksum_valid_candidate_count": (
                    introduced_valid_count
                ),
                **details,
                "cumulative_recovered_index_count": len(index_assignments),
                "cumulative_recovered_index_fraction": (
                    len(index_assignments) / strand_count
                ),
                "deferred_candidate_count": sum(map(len, pending.values())),
            }
        )

    if any(pending.values()):
        commit("initial_recovery_retry", selected_frequency, 0, 0)

    for frequency in (
        level for level in frequency_levels if level < selected_frequency
    ):
        if len(index_assignments) == strand_count:
            break
        introduced = {
            index
            for index in retained_indices
            if frequencies[index] == frequency and header_valid[index]
        }
        pending[frequency].update(introduced)
        commit(
            "new_frequency_layer",
            frequency,
            sum(frequencies[index] == frequency for index in retained_indices),
            len(introduced),
        )
    return pending, tiers


def _resolve_frequency_one_candidates_by_interval_matching(
    normalized,
    frequencies,
    decoded_order_keys,
    global_tree_order,
    index_assignments,
    rank_by_target,
    strand_count,
):
    """Match unused singletons within the existing anchor intervals.

    The finite dummy penalty and real-edge costs retain the trial protocol.
    They define a minimum-cost assignment, not a general lexicographic
    maximum-cardinality guarantee. New pairs are not checked for inversions
    against one another; existing assignments remain fixed.
    """
    started = perf_counter()
    used = set(index_assignments.values())
    eligible = sorted(
        index
        for index in decoded_order_keys
        if frequencies[index] == 1 and index not in used
    )
    diagnostics = {
        "frequency_one_resolution": "interval_matching",
        "frequency_one_candidate_count": len(eligible),
        "frequency_one_edge_count": 0,
        "frequency_one_matched_count": 0,
        "frequency_one_dummy_cost": _FREQUENCY_ONE_DUMMY_COST,
    }

    def finish(status, additions=None):
        additions = {} if additions is None else additions
        diagnostics.update(
            frequency_one_status=status,
            frequency_one_matched_count=len(additions),
            frequency_one_unmatched_target_count=(
                strand_count - len(index_assignments) - len(additions)
            ),
            frequency_one_resolution_seconds=perf_counter() - started,
        )
        return additions, diagnostics

    if len(index_assignments) == strand_count:
        return finish("all_targets_already_assigned")
    if not eligible:
        return finish("no_eligible_singletons")

    try:
        edges, edge_diagnostics = _exact_interval_edges(
            eligible,
            normalized,
            decoded_order_keys,
            global_tree_order,
            index_assignments,
            rank_by_target,
            strand_count,
        )
        diagnostics["frequency_one_edge_count"] = sum(map(len, edges.values()))
        diagnostics["frequency_one_interval_diagnostics"] = edge_diagnostics
        active = [index for index in eligible if edges.get(index)]
        targets = sorted({target for index in active for target in edges[index]})
        if not active or not targets:
            return finish("no_admissible_edges")

        # Optional T=1 dependency. The unchanged T>=2 path does not import SciPy.
        # Missing dependencies propagate; they are not treated as matching failure.
        from scipy.sparse import coo_matrix
        from scipy.sparse.csgraph import min_weight_full_bipartite_matching

        target_rows = {target: row for row, target in enumerate(targets)}
        decoded_ranks = {
            index: rank for rank, index in enumerate(global_tree_order)
        }
        rows = list(range(len(targets)))
        columns = [len(active) + row for row in range(len(targets))]
        costs = [_FREQUENCY_ONE_DUMMY_COST] * len(targets)
        for column, index in enumerate(active):
            expected_rank = round(
                (decoded_ranks[index] + 1) * strand_count
                / (len(global_tree_order) + 1)
            )
            for target in sorted(edges[index]):
                rows.append(target_rows[target])
                columns.append(column)
                costs.append(
                    1
                    + abs(int(rank_by_target[target]) - expected_rank)
                    + len(edges[index])
                )
        matrix = coo_matrix(
            (costs, (rows, columns)),
            shape=(len(targets), len(active) + len(targets)),
        ).tocsr()
        matched_rows, matched_columns = min_weight_full_bipartite_matching(matrix)
        additions = {
            targets[int(row)]: active[int(column)]
            for row, column in zip(matched_rows, matched_columns)
            if column < len(active)
        }
    except (MemoryError, ValueError) as error:
        diagnostics["frequency_one_failure_type"] = type(error).__name__
        diagnostics["frequency_one_failure_message"] = str(error)
        return finish("matching_failed")

    return finish("matched", additions)


def recover_strand_indices_by_frequency(
    bcjr_decoded_strands,
    candidate_frequencies,
    tree_seed,
    tree_start=0,
    tree_length=None,
    minimum_payload_frequency=_MINIMUM_FREQUENCY,
    frequency_one_resolution="none",
):
    """Recover positions, optionally matching remaining singletons at T=1.

    ``none`` retains the original result. ``interval_matching`` keeps the
    existing guarded recovery and adds a sparse singleton matching only when
    the frequency threshold is one. T>=2 never invokes the added step.
    """
    if frequency_one_resolution not in ("none", "interval_matching"):
        raise ValueError("frequency_one_resolution must be none or interval_matching")
    minimum_payload_frequency = int(minimum_payload_frequency)
    if minimum_payload_frequency <= 0:
        raise ValueError("minimum_payload_frequency must be positive")
    if len(bcjr_decoded_strands) != len(candidate_frequencies):
        raise ValueError("decoded strands and frequencies must have equal length")
    frequencies = [int(value) for value in candidate_frequencies]
    if any(value <= 0 for value in frequencies):
        raise ValueError("candidate frequencies must be positive")

    strand_count = len(tree_seed)
    rank_by_target = _tree_rank_by_strand_index(tree_seed)
    initial_prefix_count = _initial_prefix_count(strand_count)
    occupancy_count = _occupancy_count(strand_count)
    normalized = [_normalize_decoded_strand(value) for value in bcjr_decoded_strands]
    header_valid = [
        strand is not None
        and header_checksum_is_valid(strand, tree_start, tree_length)
        for strand in normalized
    ]
    retained_indices = sorted(
        (
            index
            for index, frequency in enumerate(frequencies)
            if frequency >= minimum_payload_frequency
        ),
        key=lambda index: (-frequencies[index], normalized[index] or "", index),
    )
    tree_candidate_indices = [
        index for index in retained_indices if header_valid[index]
    ]
    valid_retained_candidate_count = len(tree_candidate_indices)
    target_count = min(initial_prefix_count, valid_retained_candidate_count)
    frequency_levels, initial_frequency = _initial_frequency_prefix(
        tree_candidate_indices,
        frequencies,
        target_count,
    )
    decoded_order_keys = {
        index: _decoded_tree_order_key(
            normalized[index], index, tree_start, tree_length
        )
        for index in tree_candidate_indices
    }
    global_tree_order = sorted(
        tree_candidate_indices,
        key=decoded_order_keys.__getitem__,
    )
    direct_rescue_below_quarter = (
        valid_retained_candidate_count < initial_prefix_count
    )
    if direct_rescue_below_quarter and valid_retained_candidate_count:
        # Include every retained candidate, even if the threshold is absent.
        initial_frequency = minimum_payload_frequency
    strict_selection = None
    strict_pending = None
    strict_frequency_layers = None
    if not direct_rescue_below_quarter:
        strict_selection = _run_strict_initial_recovery(
            normalized,
            frequencies,
            tree_candidate_indices,
            global_tree_order,
            frequency_levels,
            initial_frequency,
            tree_seed,
            tree_start,
            tree_length,
            occupancy_count,
        )
        strict_pending, strict_frequency_layers = _descend_frequency_levels(
            strict_selection,
            frequency_levels,
            frequencies,
            tree_candidate_indices,
            header_valid,
            normalized,
            decoded_order_keys,
            global_tree_order,
            rank_by_target,
            strand_count,
        )

    rescue_trigger_denominator = min(
        valid_retained_candidate_count,
        strand_count,
    )
    rescue_trigger_count = _ceiling_fraction(
        rescue_trigger_denominator,
        _RESCUE_TRIGGER_NUMERATOR,
        _RESCUE_TRIGGER_DENOMINATOR,
    )
    strict_final_recovered_index_count = (
        len(strict_selection["index_assignments"])
        if strict_selection is not None
        else None
    )
    strict_path_below_required_fraction = (
        strict_selection is not None
        and rescue_trigger_denominator > 0
        and strict_final_recovered_index_count < rescue_trigger_count
    )
    rescue_triggered = (
        direct_rescue_below_quarter
        or strict_path_below_required_fraction
    )
    rescue_trigger_reason = (
        "insufficient_checksum_valid_candidates_for_strict_path"
        if direct_rescue_below_quarter
        else (
            "strict_path_recovered_below_required_fraction"
            if strict_path_below_required_fraction
            else None
        )
    )

    rescue_selection = None
    rescue_pending = None
    rescue_frequency_layers = None
    if rescue_triggered:
        rescue_selection = _run_sparse_initial_recovery(
            normalized,
            frequencies,
            tree_candidate_indices,
            global_tree_order,
            initial_frequency,
            tree_seed,
            tree_start,
            tree_length,
        )
        rescue_pending, rescue_frequency_layers = _descend_frequency_levels(
            rescue_selection,
            frequency_levels,
            frequencies,
            tree_candidate_indices,
            header_valid,
            normalized,
            decoded_order_keys,
            global_tree_order,
            rank_by_target,
            strand_count,
        )

    rescue_final_recovered_index_count = (
        len(rescue_selection["index_assignments"])
        if rescue_selection is not None
        else None
    )
    rescue_replaced_strict_path = (
        strict_selection is not None
        and rescue_selection is not None
        and rescue_final_recovered_index_count
        > strict_final_recovered_index_count
    )
    rescue_selected = (
        direct_rescue_below_quarter or rescue_replaced_strict_path
    )
    if rescue_selected:
        selection = rescue_selection
        pending = rescue_pending
        frequency_layers = rescue_frequency_layers
    else:
        selection = strict_selection
        pending = strict_pending
        frequency_layers = strict_frequency_layers

    build_attempts = (
        list(strict_selection["build_attempts"])
        if strict_selection is not None
        else []
    )
    if rescue_selection is not None:
        build_attempts.extend(rescue_selection["build_attempts"])

    # Freeze the original path decision before enabling safer commits. Letting
    # guarded coverage influence that decision can suppress a needed rescue.
    selection_pass_count = len(selection["index_assignments"])
    initial_methods = {
        target: method for target, method in selection["recovery_methods"].items()
        if method.startswith("initial_")
    }
    selection["index_assignments"] = {
        target: candidate for target, candidate in selection["index_assignments"].items()
        if target in initial_methods
    }
    selection["recovery_methods"] = initial_methods
    pending, frequency_layers = _descend_frequency_levels(
        selection, frequency_levels, frequencies, retained_indices, header_valid,
        normalized, decoded_order_keys, global_tree_order, rank_by_target,
        strand_count, True,
    )
    index_assignments = selection["index_assignments"]
    recovery_methods = selection["recovery_methods"]
    frequency_one_diagnostics = {}
    if minimum_payload_frequency == 1 and frequency_one_resolution == "interval_matching":
        additions, frequency_one_diagnostics = (
            _resolve_frequency_one_candidates_by_interval_matching(
                normalized,
                frequencies,
                decoded_order_keys,
                global_tree_order,
                index_assignments,
                rank_by_target,
                strand_count,
            )
        )
        index_assignments.update(additions)
        recovery_methods.update(
            dict.fromkeys(additions, "frequency_one_interval_matching")
        )
        if 1 in pending:
            pending[1].difference_update(additions.values())
    if selection["frequency"] is None:
        stop_reason = "no_recoverable_candidates"
    elif len(index_assignments) == strand_count:
        stop_reason = "all_indices_recovered"
    else:
        stop_reason = "minimum_payload_frequency_exhausted"

    ordered = {
        target: normalized[result][_HEADER_NT:]
        for target, result in index_assignments.items()
    }
    recovered_index_records = [
        {
            "recovered_strand_index": target,
            "tree_rank": int(rank_by_target[target]),
            "payload_frequency": frequencies[result],
            "recovery_method": recovery_methods[target],
        }
        for target, result in sorted(index_assignments.items())
    ]
    unresolved_strand_indices = sorted(
        set(range(strand_count)) - set(index_assignments)
    )
    unassigned_candidate_positions = sorted(
        set(range(len(normalized))) - set(index_assignments.values())
    )
    diagnostics = {
        "algorithm": "descending_frequency_frozen_tree_index_recovery",
        "recovery_revision": "fixed_initial_state_order_guard_v1",
        "path_selection_pass_recovered_index_count": selection_pass_count,
        "total_index_count": strand_count,
        "input_candidate_count": len(normalized),
        "retained_candidate_count": len(retained_indices),
        "checksum_valid_candidate_count": valid_retained_candidate_count,
        "discarded_singleton_candidate_count": sum(
            frequency == 1 and frequency < minimum_payload_frequency
            for frequency in frequencies
        ),
        "initial_recovery": {
            "path_counts_stage": "original_path_selection_pass",
            "target_candidate_count": target_count,
            "starting_minimum_payload_frequency": initial_frequency,
            "selected_minimum_payload_frequency": selection["frequency"],
            "selected_path": selection["phase"],
            "selection_reason": selection["acceptance_reason"],
            "selected_candidate_count": len(selection["candidate_indices"]),
            "initial_recovered_index_count": (
                selection["initial_recovered_index_count"]
            ),
            "minimum_required_recovered_index_count": occupancy_count,
            "build_attempts": build_attempts,
            "strict_path_initial_recovered_index_count": (
                strict_selection["initial_recovered_index_count"]
                if strict_selection is not None
                else None
            ),
            "strict_path_final_recovered_index_count": (
                strict_final_recovered_index_count
            ),
            "rescue_path_final_recovered_index_count": (
                rescue_final_recovered_index_count
            ),
        },
        "rescue": {
            "strict_path_skipped": direct_rescue_below_quarter,
            "triggered": rescue_triggered,
            "trigger_reason": rescue_trigger_reason,
            "required_recovered_index_fraction": 0.50,
            "reference_candidate_count": rescue_trigger_denominator,
            "minimum_required_recovered_index_count": rescue_trigger_count,
            "selected": rescue_selected,
            "replaced_strict_path": rescue_replaced_strict_path,
        },
        "frequency_layer_progress": frequency_layers,
        "recovery_method_counts": dict(
            sorted(Counter(recovery_methods.values()).items())
        ),
        "recovered_index_records": recovered_index_records,
        "summary": {
            "stop_reason": stop_reason,
            "recovered_index_count": len(index_assignments),
            "recovered_index_fraction": (
                len(index_assignments) / strand_count
            ),
            "unresolved_index_count": len(unresolved_strand_indices),
            "unresolved_strand_indices": unresolved_strand_indices,
            "unassigned_candidate_count": len(
                unassigned_candidate_positions
            ),
            "deferred_candidate_count": sum(map(len, pending.values())),
        },
    }
    diagnostics.update(frequency_one_diagnostics)
    return ordered, sorted(index_assignments.items()), diagnostics




__all__ = ["recover_strand_indices_by_frequency"]
