"""Focused tests for Beezy's tree and 4nt indexing core."""

from dataclasses import replace
import itertools
import unittest

from ..indexing import (
    BaseTree,
    BaseTrees,
    ary4_to_base,
    base_to_4ary,
    build_base_trees,
    construct_strand_indices,
    cycle_count,
    encode_header,
    header_checksum_is_valid,
    order_by_base_trees,
    restore_absolute_tree_rank,
    tree_order_key,
)
from ..indexing.tree import compact_tree_order_key


class IndexingTests(unittest.TestCase):
    def test_construct_strand_indices_adds_rank_headers_and_tree_seed(self):
        strands = ["TGCAT", "ATGCA", "CGTAA", "AATGC"]
        indexed_strands, tree_seed, collision_count, collision_groups = (
            construct_strand_indices(strands)
        )
        expected_order = order_by_base_trees(
            range(len(strands)),
            payload_for_item=strands.__getitem__,
            fallback_key=lambda strand_index: strand_index,
        )

        self.assertEqual(tree_seed, [index + 1 for index in expected_order])
        for tree_rank, one_based_strand_index in enumerate(tree_seed):
            strand_index = one_based_strand_index - 1
            indexed_strand = indexed_strands[strand_index]
            self.assertEqual(indexed_strand[4:], strands[strand_index])
            self.assertEqual(cycle_count(indexed_strand), tree_rank % 64)
            self.assertTrue(header_checksum_is_valid(indexed_strand))
        self.assertEqual(
            collision_count,
            sum(len(group) - 1 for group in collision_groups),
        )

    def test_4ary_dna_alphabet_round_trip(self):
        for digit, base in enumerate("ATGC"):
            self.assertEqual(base_to_4ary(base), digit)
            self.assertEqual(ary4_to_base(digit), base)

    def test_base_trees_and_order_key(self):
        payload = "ATGCAAGT"
        base_trees = build_base_trees(payload)

        self.assertEqual(
            base_trees,
            BaseTrees(
                root_base="A",
                a_tree="TAG",
                t_tree="G",
                g_tree="CT",
                c_tree="A",
            ),
        )
        self.assertEqual(tree_order_key(payload)[0], (1, 0, 2))

    def test_root_base_resolves_equal_four_tree_rotations(self):
        first = build_base_trees("ATGCA")
        second = build_base_trees("TGCAT")

        self.assertEqual(
            (first.a_tree, first.t_tree, first.g_tree, first.c_tree),
            (second.a_tree, second.t_tree, second.g_tree, second.c_tree),
        )
        self.assertEqual(first.root_base, "A")
        self.assertEqual(second.root_base, "T")
        self.assertNotEqual(
            tree_order_key("ATGCA")[:5],
            tree_order_key("TGCAT")[:5],
        )

    def test_root_and_four_trees_identify_complete_payloads(self):
        payloads = (
            "".join(bases)
            for bases in itertools.product("ATGC", repeat=5)
        )
        identities = [tree_order_key(payload)[:5] for payload in payloads]

        self.assertEqual(len(identities), 4**5)
        self.assertEqual(len(set(identities)), len(identities))

    def test_compact_tree_key_preserves_order_across_slices(self):
        payloads = [
            "",
            "A",
            "AA",
            "AT",
            "ATG",
            "T",
            "G",
            "C",
            "ATGCA",
            "TGCAT",
        ]
        for tree_start, tree_length in (
            (0, None),
            (0, 0),
            (1, None),
            (1, 2),
            (99, 4),
        ):
            expected = sorted(
                payloads,
                key=lambda payload: tree_order_key(
                    payload,
                    tree_start,
                    tree_length,
                ),
            )
            actual = sorted(
                payloads,
                key=lambda payload: compact_tree_order_key(
                    payload,
                    tree_start,
                    tree_length,
                ),
            )
            self.assertEqual(actual, expected)

    def test_compact_tree_key_delegates_slice_validation(self):
        with self.assertRaisesRegex(ValueError, "tree_start"):
            compact_tree_order_key("ATGC", tree_start=-1)
        with self.assertRaisesRegex(ValueError, "tree_length"):
            compact_tree_order_key("ATGC", tree_length=-1)

    def test_base_tree_materializes_nodes_and_traverses_them(self):
        tree = BaseTree()
        tree.insert("AT", "AT")
        tree.insert("A", "A")
        tree.insert("C", "C")
        tree.insert("AA", "AA")

        a_node = tree.root.children[0]
        self.assertIsNotNone(a_node)
        self.assertEqual(a_node.values, ["A"])
        self.assertIsNotNone(a_node.children[0])
        self.assertIsNotNone(a_node.children[1])
        self.assertEqual(
            tree.values_in_tree_order(),
            ["A", "AA", "AT", "C"],
        )

    def test_entity_trees_preserve_the_tree_order_key(self):
        payloads = [
            "C",
            "T",
            "G",
            "A",
            "ATGCAAGT",
            "AAGTATGC",
            "TTGCAAGT",
            "CGTAAAGT",
            "ATGCAAAT",
        ]
        expected = sorted(
            range(len(payloads)),
            key=lambda index: (tree_order_key(payloads[index]), index),
        )
        actual = order_by_base_trees(
            range(len(payloads)),
            payload_for_item=lambda index: payloads[index],
            fallback_key=lambda index: (payloads[index], index),
        )

        self.assertEqual(actual, expected)

    def test_header_round_trip(self):
        payload = "ATGCAAGT"
        base_trees = build_base_trees(payload)

        for rank in (0, 1, 63, 64, 65):
            header = encode_header(base_trees, rank)
            indexed = header + payload
            self.assertEqual(len(header), 4)
            self.assertEqual(cycle_count(indexed), rank % 64)
            self.assertTrue(header_checksum_is_valid(indexed))

    def test_header_checksum_uses_t_g_and_c_trees(self):
        payload = "ATGCAAGT"
        base_trees = build_base_trees(payload)
        header = encode_header(base_trees, 17)

        self.assertTrue(header_checksum_is_valid(header + payload))
        changes = {
            "root_base": "T",
            "a_tree": base_trees.a_tree + "A",
            "t_tree": base_trees.t_tree + "A",
            "g_tree": base_trees.g_tree + "A",
            "c_tree": base_trees.c_tree + "T",
        }
        for field_name, changed_value in changes.items():
            changed_trees = replace(
                base_trees,
                **{field_name: changed_value},
            )
            self.assertNotEqual(
                encode_header(changed_trees, 17)[0],
                header[0],
            )

    def test_cycle_count_restores_absolute_tree_rank(self):
        self.assertEqual(restore_absolute_tree_rank(0, 63), 64)
        self.assertEqual(restore_absolute_tree_rank(63, 64), 63)


if __name__ == "__main__":
    unittest.main()
