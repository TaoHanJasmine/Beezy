"""Construct deterministic tree-order indices for an encoded strand pool."""

from collections import defaultdict

from .header import encode_header
from .tree import build_base_trees, order_by_base_trees


def construct_strand_indices(strands, tree_start=0, tree_length=None):
    """Order strands by their base trees and prepend tree-rank headers."""
    strands = list(strands)
    if not strands:
        raise ValueError("cannot construct indices for an empty strand pool")

    strand_indices_in_tree_order = order_by_base_trees(
        range(len(strands)),
        payload_for_item=strands.__getitem__,
        tree_start=tree_start,
        tree_length=tree_length,
        fallback_key=lambda strand_index: strand_index,
    )
    tree_seed = [
        strand_index + 1
        for strand_index in strand_indices_in_tree_order
    ]
    indexed_strands = list(strands)
    strand_indices_by_a_tree = defaultdict(list)
    for tree_rank, strand_index in enumerate(strand_indices_in_tree_order):
        base_trees = build_base_trees(
            strands[strand_index],
            tree_start,
            tree_length,
        )
        indexed_strands[strand_index] = (
            encode_header(base_trees, tree_rank) + strands[strand_index]
        )
        strand_indices_by_a_tree[base_trees.a_tree].append(strand_index + 1)

    collision_groups = [
        tuple(positions)
        for positions in strand_indices_by_a_tree.values()
        if len(positions) > 1
    ]
    collision_count = sum(len(group) - 1 for group in collision_groups)
    return indexed_strands, tree_seed, collision_count, collision_groups


__all__ = ["construct_strand_indices"]
