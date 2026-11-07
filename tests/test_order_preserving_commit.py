"""Regression checks for rejecting all participants of crossing proposals."""
from itertools import permutations
from ..indexing.index_recovery.frequency_recovery import _order_preserving_additions


def test_all_inversion_participants_are_deferred():
    for permutation in permutations(range(6)):
        proposals = {rank: candidate for candidate, rank in enumerate(permutation)}
        bad = {r for i, r in enumerate(permutation) for j, s in enumerate(permutation)
               if (i < j and r > s) or (i > j and r < s)}
        result = _order_preserving_additions({}, proposals, dict(enumerate(range(6))), list(range(6)))
        assert set(result) == set(permutation) - bad


def test_old_anchor_is_preserved_and_only_crossing_proposal_is_deferred():
    anchors = {3: 1}
    result = _order_preserving_additions(anchors, {4: 0, 5: 2},
        {0: 0, 1: 1, 2: 2}, list(range(6)))
    assert result == {5: 2}
    assert anchors == {3: 1}


def test_compatible_proposals_are_unchanged():
    proposals = {1: 1, 2: 2, 3: 3}
    assert _order_preserving_additions({0: 0, 4: 4}, proposals,
        dict(enumerate(range(5))), list(range(5))) == proposals
