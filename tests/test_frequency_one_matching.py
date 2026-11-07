"""Singleton matching boundaries while retaining the guarded recovery path."""

import random

import pytest

from ..indexing.header import encode_header, header_checksum_is_valid
from ..indexing.index_construction import construct_strand_indices
from ..indexing.index_recovery import frequency_intervals as intervals
from ..indexing.index_recovery import frequency_recovery as recovery
from ..indexing.tree import build_base_trees


def _payloads(count, seed=719):
    rng = random.Random(seed)
    return ["".join(rng.choices("ATGC", k=44)) for _ in range(count)]


def _helper(candidates, tree, frequencies=None, placements=None):
    frequencies = [1] * len(candidates) if frequencies is None else frequencies
    placements = {} if placements is None else placements
    normalized = [intervals._normalize_decoded_strand(value) for value in candidates]
    keys = {
        index: intervals._decoded_tree_order_key(value, index, 0, 44)
        for index, value in enumerate(normalized)
        if value is not None and header_checksum_is_valid(value, 0, 44)
    }
    return recovery._resolve_frequency_one_candidates_by_interval_matching(
        normalized, frequencies, keys, sorted(keys, key=keys.__getitem__),
        placements, intervals._tree_rank_by_strand_index(tree), len(tree),
    )


@pytest.fixture
def sparse_case():
    indexed, tree, _, _ = construct_strand_indices(_payloads(512), tree_length=44)
    # A 200-rank gap leaves multiple modulo-64 placements. The unchanged
    # initialization rejects this sparse case rather than guessing an anchor.
    return [indexed[tree[rank] - 1] for rank in (100, 300)], tree


def test_disabled_strategy_retains_original_output(sparse_case):
    candidates, tree = sparse_case
    default = recovery.recover_strand_indices_by_frequency(
        candidates, [1, 1], tree, tree_length=44, minimum_payload_frequency=1,
    )
    disabled = recovery.recover_strand_indices_by_frequency(
        candidates, [1, 1], tree, tree_length=44, minimum_payload_frequency=1,
        frequency_one_resolution="none",
    )
    assert default == disabled
    assert default[1] == []


@pytest.mark.parametrize("threshold", [2, 3, 4, 5, 6])
def test_threshold_two_and_above_never_calls_new_matching(monkeypatch, threshold):
    indexed, tree, _, _ = construct_strand_indices(_payloads(70), tree_length=44)
    def unexpected(*args, **kwargs):
        raise AssertionError("T>=2 must bypass singleton matching")
    monkeypatch.setattr(recovery, "_resolve_frequency_one_candidates_by_interval_matching", unexpected)
    default = recovery.recover_strand_indices_by_frequency(
        indexed, [threshold] * 70, tree, tree_length=44,
        minimum_payload_frequency=threshold,
    )
    enabled = recovery.recover_strand_indices_by_frequency(
        indexed, [threshold] * 70, tree, tree_length=44,
        minimum_payload_frequency=threshold, frequency_one_resolution="interval_matching",
    )
    assert enabled == default
    assert not any(key.startswith("frequency_one_") for key in enabled[2])


def test_singleton_matching_updates_final_assignments_and_diagnostics(sparse_case):
    candidates, tree = sparse_case
    snapshot = candidates.copy()
    result = recovery.recover_strand_indices_by_frequency(
        candidates, [1, 1], tree, tree_length=44, minimum_payload_frequency=1,
        frequency_one_resolution="interval_matching",
    )
    ordered, pairs, diagnostic = result
    assert candidates == snapshot
    assert len(pairs) == len(ordered) == 2
    assert len({candidate for target, candidate in pairs}) == 2
    ranks = intervals._tree_rank_by_strand_index(tree)
    for target, candidate in pairs:
        assert ranks[target] % 64 == intervals.cycle_count(candidates[candidate])
        assert ordered[target] == candidates[candidate][4:]
    assert diagnostic["frequency_one_candidate_count"] == 2
    assert diagnostic["frequency_one_edge_count"] == 16
    assert diagnostic["frequency_one_matched_count"] == 2
    assert diagnostic["frequency_one_unmatched_target_count"] == 510
    assert diagnostic["summary"]["recovered_index_count"] == 2
    assert diagnostic["summary"]["unassigned_candidate_count"] == 0
    assert diagnostic["summary"]["deferred_candidate_count"] == 0
    assert diagnostic["recovery_method_counts"] == {"frequency_one_interval_matching": 2}
    assert all(record["recovery_method"] == "frequency_one_interval_matching"
               for record in diagnostic["recovered_index_records"])
    assert diagnostic["frequency_one_resolution_seconds"] >= 0


def test_only_valid_unused_frequency_one_sequences_enter_matching(sparse_case):
    candidates, tree = sparse_case
    invalid = ("A" if candidates[0][0] != "A" else "T") + candidates[0][1:]
    assert not header_checksum_is_valid(invalid, 0, 44)
    additions, diagnostic = _helper(candidates + [invalid, None], tree, [1, 2, 1, 1])
    assert list(additions.values()) == [0]
    assert diagnostic["frequency_one_candidate_count"] == 1
    target = next(iter(additions))
    existing = {target: 0}
    repeated, diagnostic = _helper(candidates + [invalid, None], tree, [1, 2, 1, 1], existing)
    assert repeated == {}
    assert existing == {target: 0}
    assert diagnostic["frequency_one_candidate_count"] == 0


def test_existing_anchor_intervals_and_occupied_positions_are_respected():
    indexed, tree, _, _ = construct_strand_indices(_payloads(200), tree_length=44)
    candidates = [indexed[tree[rank] - 1] for rank in (0, 65, 130)]
    existing = {tree[0] - 1: 0, tree[130] - 1: 2}
    snapshot = existing.copy()
    additions, diagnostic = _helper(candidates, tree, placements=existing)
    assert existing == snapshot
    assert len(additions) == 1
    ranks = intervals._tree_rank_by_strand_index(tree)
    for target, candidate in additions.items():
        assert target not in existing and candidate not in existing.values()
        assert 0 < ranks[target] < 130
        assert ranks[target] % 64 == 65 % 64


def test_new_pairs_are_not_subject_to_an_additional_order_guard(monkeypatch):
    payloads = sorted(_payloads(2), key=lambda value: intervals._decoded_tree_order_key("AAAA"+value, 0, 0, 44))
    candidates = [encode_header(build_base_trees(payload), rank)+payload
                  for payload, rank in zip(payloads, (1, 0))]
    def unexpected(*args, **kwargs):
        raise AssertionError("The new matching must not add an order guard")
    monkeypatch.setattr(recovery, "_order_preserving_additions", unexpected)
    additions, diagnostic = _helper(candidates, [1, 2])
    assert additions == {0: 1, 1: 0}
    assert diagnostic["frequency_one_matched_count"] == 2


def test_matching_is_repeatable_for_ambiguous_positions(sparse_case):
    candidates, tree = sparse_case
    first, _ = _helper(candidates, tree)
    second, _ = _helper(candidates, tree)
    assert first == second


def test_no_open_positions_or_admissible_edges_returns_without_matching(monkeypatch, sparse_case):
    candidates, tree = sparse_case
    complete = {index: index for index in range(2)}
    additions, diagnostic = _helper(candidates, [1, 2], placements=complete)
    assert additions == {}
    assert diagnostic["frequency_one_status"] == "all_targets_already_assigned"
    monkeypatch.setattr(recovery, "_exact_interval_edges", lambda *args: ({}, {}))
    additions, diagnostic = _helper(candidates, tree)
    assert additions == {}
    assert diagnostic["frequency_one_status"] == "no_admissible_edges"


@pytest.mark.parametrize("exception", [ValueError("solver failure"), MemoryError("matrix allocation")])
def test_matching_failure_keeps_original_recovery(monkeypatch, sparse_case, exception):
    import scipy.sparse.csgraph
    candidates, tree = sparse_case
    original = recovery.recover_strand_indices_by_frequency(
        candidates, [1, 1], tree, tree_length=44, minimum_payload_frequency=1,
    )
    def failed(*args, **kwargs):
        raise exception
    monkeypatch.setattr(scipy.sparse.csgraph, "min_weight_full_bipartite_matching", failed)
    result = recovery.recover_strand_indices_by_frequency(
        candidates, [1, 1], tree, tree_length=44, minimum_payload_frequency=1,
        frequency_one_resolution="interval_matching",
    )
    assert result[:2] == original[:2]
    assert result[2]["summary"] == original[2]["summary"]
    assert result[2]["frequency_one_status"] == "matching_failed"
    assert result[2]["frequency_one_failure_type"] == type(exception).__name__
    assert result[2]["frequency_one_matched_count"] == 0


def test_unknown_strategy_is_rejected():
    with pytest.raises(ValueError, match="frequency_one_resolution"):
        recovery.recover_strand_indices_by_frequency([], [], [1], frequency_one_resolution="unknown")
