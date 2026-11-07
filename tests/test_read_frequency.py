"""Tests for streaming raw-read frequency counting."""

from array import array
from collections import Counter
from pathlib import Path
import tempfile
import unittest

from ..pipeline.read_frequency import (
    UINT64_MAX,
    _increment_uint64,
    read_raw_frequency_table,
)


class RawReadFrequencyTests(unittest.TestCase):
    def test_matches_counter_most_common_and_skips_blank_lines(self):
        observations = [
            "r2",
            "r1",
            "r2",
            "r3",
            "r1",
            "r2",
            "r4",
            "r4",
        ]
        with tempfile.TemporaryDirectory() as directory:
            reads_path = Path(directory) / "reads"
            reads_path.write_text(
                "\n\n  r2  \nr1\nr2\nr3\nr1\nr2\nr4\nr4\n \n",
                encoding="utf-8",
            )

            table = read_raw_frequency_table(reads_path)

        self.assertEqual(list(table), Counter(observations).most_common())
        self.assertEqual(list(table.iter_reads()), ["r2", "r1", "r4", "r3"])
        self.assertEqual(table.raw_read_count, len(observations))
        self.assertEqual(table.unique_read_count, 4)
        self.assertEqual(table.duplicate_read_count, 4)

    def test_frequency_ties_preserve_first_observation_order(self):
        with tempfile.TemporaryDirectory() as directory:
            reads_path = Path(directory) / "reads"
            reads_path.write_text(
                "third\nfirst\nsecond\nsecond\nfirst\nthird\n",
                encoding="utf-8",
            )

            table = read_raw_frequency_table(reads_path)

        self.assertEqual(
            list(table),
            [("third", 2), ("first", 2), ("second", 2)],
        )

    def test_singletons_follow_repeats_in_first_observation_order(self):
        observations = [
            "singleton-first",
            "repeat-late",
            "singleton-middle",
            "repeat-late",
            "singleton-last",
        ]
        with tempfile.TemporaryDirectory() as directory:
            reads_path = Path(directory) / "reads"
            reads_path.write_text(
                "\n".join(observations) + "\n",
                encoding="utf-8",
            )

            table = read_raw_frequency_table(reads_path)

        self.assertEqual(
            list(table),
            Counter(observations).most_common(),
        )

    def test_uint64_frequency_overflow_is_explicit(self):
        counts = array("Q", [UINT64_MAX])

        with self.assertRaisesRegex(OverflowError, "uint64"):
            _increment_uint64(counts, 0)


if __name__ == "__main__":
    unittest.main()
