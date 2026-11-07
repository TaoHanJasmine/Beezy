"""A/T/G/C tree construction and deterministic Beezy ordering."""

from dataclasses import dataclass, field


TREE_BASE_ORDER = {"A": 0, "T": 1, "G": 2, "C": 3}


@dataclass(frozen=True)
class BaseTrees:
    """Root base and flattened A/T/G/C trees for one strand."""

    root_base: str
    a_tree: str
    t_tree: str
    g_tree: str
    c_tree: str


@dataclass
class TreeNode:
    """One node in an A/T/G/C prefix tree."""

    children: list = field(default_factory=lambda: [None, None, None, None])
    values: list = field(default_factory=list)


class BaseTree:
    """A prefix tree whose branches follow the A/T/G/C tree order."""

    def __init__(self):
        self.root = TreeNode()

    def insert(self, tree_sequence, value):
        """Insert one tree sequence and its associated value."""
        node = self.root
        for base in tree_sequence:
            child_index = TREE_BASE_ORDER[base]
            if node.children[child_index] is None:
                node.children[child_index] = TreeNode()
            node = node.children[child_index]
        node.values.append(value)

    def values_in_tree_order(self):
        """Return terminal values in A/T/G/C depth-first tree order."""
        ordered_values = []

        def visit(node):
            ordered_values.extend(node.values)
            for child in node.children:
                if child is not None:
                    visit(child)

        visit(self.root)
        return ordered_values


def build_base_trees(payload, tree_start=0, tree_length=None):
    """Build the flattened A/T/G/C trees for the configured payload slice."""
    if tree_start < 0:
        raise ValueError("tree_start must be non-negative")
    if tree_length is not None and tree_length < 0:
        raise ValueError("tree_length must be non-negative")
    segment = (
        payload[tree_start:]
        if tree_length is None
        else payload[tree_start:tree_start + tree_length]
    )
    base_tree_nodes = {"A": [], "T": [], "G": [], "C": []}
    for base, next_base in zip(segment, segment[1:]):
        base_tree_nodes[base].append(next_base)
    return BaseTrees(
        root_base=segment[:1],
        a_tree="".join(base_tree_nodes["A"]),
        t_tree="".join(base_tree_nodes["T"]),
        g_tree="".join(base_tree_nodes["G"]),
        c_tree="".join(base_tree_nodes["C"]),
    )


def order_by_base_trees(
    items,
    payload_for_item,
    tree_start=0,
    tree_length=None,
    fallback_key=None,
):
    """Order items by traversing their A/T/G/C trees in sequence."""
    records = []
    for item in items:
        payload = payload_for_item(item)
        records.append(
            (
                item,
                payload,
                build_base_trees(payload, tree_start, tree_length),
            )
        )
    if fallback_key is not None:
        records.sort(key=lambda record: fallback_key(record[0]))

    # Stable passes produce A/T/G/C/root order. A caller-provided fallback is
    # used only when configured tree slices are exactly equal.
    tree_sequences = [
        lambda record: record[2].root_base,
        lambda record: record[2].c_tree,
        lambda record: record[2].g_tree,
        lambda record: record[2].t_tree,
        lambda record: record[2].a_tree,
    ]
    for tree_sequence in tree_sequences:
        tree = BaseTree()
        for record in records:
            tree.insert(tree_sequence(record), record)
        records = tree.values_in_tree_order()

    return [record[0] for record in records]


def tree_order_key(payload, tree_start=0, tree_length=None):
    """Return the deterministic A-tree-first ordering key.

    The T/G/C trees resolve A-tree collisions, then the root base resolves
    cyclic rotations with identical successor trees. For a complete tree
    slice this key uniquely identifies every distinct payload.
    """
    base_trees = build_base_trees(payload, tree_start, tree_length)

    def encode_order(sequence):
        return tuple(TREE_BASE_ORDER[base] for base in sequence)

    return (
        encode_order(base_trees.a_tree),
        encode_order(base_trees.t_tree),
        encode_order(base_trees.g_tree),
        encode_order(base_trees.c_tree),
        encode_order(base_trees.root_base),
    )


def compact_tree_order_key(payload, tree_start=0, tree_length=None):
    """Encode ``tree_order_key`` as one order-equivalent bytes value."""
    base_trees = build_base_trees(payload, tree_start, tree_length)
    key = bytearray()
    for sequence in (
        base_trees.a_tree,
        base_trees.t_tree,
        base_trees.g_tree,
        base_trees.c_tree,
        base_trees.root_base,
    ):
        key.extend(TREE_BASE_ORDER[base] + 1 for base in sequence)
        key.append(0)
    return bytes(key)
