"""Beezy's 1 nt joint-tree checksum and 3 nt tree-order cycle count."""

import hashlib

from .tree import build_base_trees


def base_to_4ary(base):
    """Map the ordered DNA alphabet A/T/G/C to a 4-ary digit."""
    return {"A": 0, "T": 1, "G": 2, "C": 3}[base]


def ary4_to_base(digit):
    """Map a 4-ary digit to the ordered DNA alphabet A/T/G/C."""
    return {0: "A", 1: "T", 2: "G", 3: "C"}[digit]


def cycle_count(indexed_strand):
    """Read the three-base cycle count from an indexed strand."""
    if len(indexed_strand) < 4:
        raise ValueError("a Beezy indexed sequence must contain at least 4 nt")
    count = 0
    for base in indexed_strand[1:4]:
        count = count * 4 + base_to_4ary(base)
    return count


def restore_absolute_tree_rank(cycle_count, tree_rank_cursor):
    """Restore the nearest absolute tree rank around the current cursor."""
    expected_cycle_count = tree_rank_cursor % 64
    diff = cycle_count - expected_cycle_count
    if diff < -32:
        diff += 64
    elif diff > 32:
        diff -= 64
    return tree_rank_cursor + diff


def _joint_tree_checksum_digit(encoded_count, base_trees):
    """Compress framed cycle/root/tree content into one 4-ary digit."""
    digest = hashlib.blake2s(digest_size=1, person=b"BzyHdr1")
    fields = (
        encoded_count,
        base_trees.root_base,
        base_trees.a_tree,
        base_trees.t_tree,
        base_trees.g_tree,
        base_trees.c_tree,
    )
    for field_tag, sequence in enumerate(fields):
        encoded_sequence = sequence.encode("ascii")
        digest.update(bytes((field_tag,)))
        digest.update(len(encoded_sequence).to_bytes(4, "big"))
        digest.update(encoded_sequence)
    return digest.digest()[0] % 4


def header_checksum_is_valid(
    indexed_strand,
    tree_start=0,
    tree_length=None,
):
    """Validate the checksum against the cycle count, root, and four trees."""
    if len(indexed_strand) < 4:
        return False
    try:
        base_trees = build_base_trees(
            indexed_strand[4:],
            tree_start,
            tree_length,
        )
        checksum_base = ary4_to_base(
            _joint_tree_checksum_digit(indexed_strand[1:4], base_trees)
        )
    except (KeyError, ValueError):
        return False
    return checksum_base == indexed_strand[0]


def encode_header(base_trees, tree_rank):
    """Encode one joint-tree checksum nt and a three-nt cycle count."""
    count = tree_rank % 64
    count_digits = []
    remaining_count = count
    for _ in range(3):
        count_digits.append(remaining_count % 4)
        remaining_count //= 4
    count_digits.reverse()
    encoded_count = "".join(
        ary4_to_base(digit) for digit in count_digits
    )

    checksum_base = ary4_to_base(
        _joint_tree_checksum_digit(encoded_count, base_trees)
    )
    return checksum_base + encoded_count
