"""Threshold boundaries without dependencies on experiment runners."""

import pytest

from ..indexing.index_construction import construct_strand_indices
from ..indexing.index_recovery.frequency_aggregation import aggregate_bcjr_payloads_by_frequency
from ..indexing.index_recovery.frequency_recovery import recover_strand_indices_by_frequency
from ..pipeline.data_io import bytes_to_dna_sequences


def test_aggregation_configurable_threshold_and_default_equivalence():
    indexed, _, _, _ = construct_strand_indices(['ACGT' * 11, 'TGCA' * 11], tree_length=44)
    raw = [('a', 1), ('b', 2)]
    hard = [(strand, None) for strand in indexed]
    default = aggregate_bcjr_payloads_by_frequency(raw, hard, tree_length=44)
    assert default == aggregate_bcjr_payloads_by_frequency(
        raw, hard, tree_length=44, minimum_payload_frequency=2)
    assert len(default[0]) == 1
    assert len(aggregate_bcjr_payloads_by_frequency(
        raw, hard, tree_length=44, minimum_payload_frequency=1)[0]) == 2
    assert not aggregate_bcjr_payloads_by_frequency(
        raw, hard, tree_length=44, minimum_payload_frequency=3)[0]


def test_threshold_applies_after_merging_payload_support():
    indexed, _, _, _ = construct_strand_indices(['ACGT' * 11, 'TGCA' * 11], tree_length=44)
    raw = [('a', 2), ('b', 2), ('c', 3)]
    hard = [(indexed[0], None), (indexed[0], None), (indexed[1], None)]
    retained, frequencies, diagnostics = aggregate_bcjr_payloads_by_frequency(
        raw, hard, tree_length=44, minimum_payload_frequency=4)
    assert retained == [(indexed[0], None)]
    assert frequencies == [4]
    assert diagnostics['discarded_below_threshold_payload_count'] == 1
    assert diagnostics['retained_payload_count'] == 1
    assert not aggregate_bcjr_payloads_by_frequency(
        raw, hard, tree_length=44, minimum_payload_frequency=5)[0]


def test_singleton_candidate_boundary_and_default_equivalence():
    payloads = bytes_to_dna_sequences(bytes(range(220)), 44, whiten=True)
    indexed, seed, _, _ = construct_strand_indices(payloads, tree_length=44)
    order = list(range(len(indexed) - 1, -1, -1))
    candidates = [indexed[i] for i in order]
    default = recover_strand_indices_by_frequency(candidates, [2] * len(order), seed, tree_length=44)
    explicit = recover_strand_indices_by_frequency(
        candidates, [2] * len(order), seed, tree_length=44, minimum_payload_frequency=2)
    assert default == explicit
    singleton = recover_strand_indices_by_frequency(
        candidates, [1] * len(order), seed, tree_length=44, minimum_payload_frequency=1)
    assert singleton[:2] == default[:2]
    assert len(singleton[1]) == len(seed)
    assert all(order[candidate] == target for target, candidate in singleton[1])
    assert not recover_strand_indices_by_frequency(candidates, [1] * len(order), seed, tree_length=44)[1]
    assert not recover_strand_indices_by_frequency(
        candidates, [2] * len(order), seed, tree_length=44, minimum_payload_frequency=3)[1]


@pytest.mark.parametrize('threshold', [0, -1])
def test_nonpositive_threshold_is_rejected(threshold):
    with pytest.raises(ValueError, match='minimum_payload_frequency'):
        aggregate_bcjr_payloads_by_frequency([], [], minimum_payload_frequency=threshold)
    with pytest.raises(ValueError, match='minimum_payload_frequency'):
        recover_strand_indices_by_frequency([], [], [1], minimum_payload_frequency=threshold)
