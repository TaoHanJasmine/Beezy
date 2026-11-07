import pytest

from ..evaluation.recovery_metrics import score_indices, score_bytes, score_pool
from ..pipeline.data_io import bytes_to_dna_sequences


def test_indices_not_payload_similarity():
    result = score_indices(4, [(0, 0), (2, 1)], [{2: 1}, {0: 1}])
    assert result["coverage"] == 0.5
    assert result["ordering_accuracy"] == 0


def test_fixed_bytes_no_alignment_and_invalid_zero():
    result, rows=score_bytes(['ACGTACGT','AAAAAAAA'],{0:'ACGGTACGT'})
    assert result['recovery_rate']==0
    assert result['valid_recovered_byte_count']==2
    assert result['extra_nt']==1
    assert rows[1]['valid_byte_mask']==[False,False]
    result,_=score_bytes(['ACGTACGT'],{0:'ACGTACN'})
    assert result['recovery_rate']==0.5
    assert result['missing_or_invalid_byte_count']==1


def test_partial_final_block_excludes_padding_and_missing_does_not_shift():
    result,_=score_bytes(['AAAACCCC','GGGGTTTT'],{1:'GGGGAAAA'},3)
    assert result['correct_byte_count']==1
    assert result['reference_byte_count']==3
    assert result['recovery_rate']==1/3


def test_wrong_index_can_still_have_equal_bytes_and_xor_invariance():
    from ..pipeline.data_io import dna_sequences_to_bytes, _transform_block
    source=bytes(range(22))
    ref=bytes_to_dna_sequences(source,44,whiten=True)
    ordered={0:ref[1],1:ref[0]}
    decoded=dna_sequences_to_bytes([ordered[0],ordered[1]],22,whiten=True)
    result,_=score_bytes(ref,ordered,len(source))
    assert result['correct_byte_count']==sum(a==b for a,b in zip(source,decoded))
    assert _transform_block(_transform_block(source[:11],1),1)==source[:11]
    m,_=score_pool(['AAAACCCC','AAAAGGGG'],{0:'AAAAGGGG'},[(0,0)],[{1: 1}],['TTTTAAAAGGGG'])
    assert m['ordering_accuracy']==0
    assert m['correct_byte_count']==1


def test_ambiguity_and_empty():
    result = score_indices(4, [(0, 0), (1, 1)], [{0: 1}, {1: 1, 2: 1}])
    assert result["ordering_accuracy"] == 2 / 3
    assert result["ordering_accuracy_lower"] == result["ordering_accuracy_upper"] == 2 / 3
    assert result["ambiguous_count"] == 1
    assert score_indices(4, [], [])["ordering_accuracy"] is None
    with pytest.raises(ValueError):
        score_indices(4, [(0, 0), (1, 0)], [{0: 1}])


def test_read_weights_and_unassigned_candidates():
    result = score_indices(3, [(0, 0), (1, 1)], [{0: 9}, {2: 1}, {2: 1000}])
    assert result['ordering_accuracy'] == .9
    assert result['weighted_correct_read_count'] == 9
    assert result['weighted_assigned_read_count'] == 10
    assert result['correctly_ordered_count'] == result['wrongly_ordered_count'] == 1


def test_mixed_origin_counts_do_not_use_majority_vote():
    result = score_indices(2, [(0, 0)], [{0: 2, 1: 8}])
    assert result['ordering_accuracy'] == .2
    assert result['weighted_assigned_read_count'] == 10


@pytest.mark.parametrize('histogram', [{}, {0: 0}, {0: -1}, {0: 1.5}, {0: True}, {2: 1}])
def test_invalid_source_histograms(histogram):
    with pytest.raises(ValueError):
        score_indices(2, [(0, 0)], [histogram])


def test_source_sets_cannot_supply_read_weights():
    with pytest.raises(TypeError, match='read counts'):
        score_indices(2, [(0, 0)], [{0, 1}])
