"""Tests for production payload-frequency index recovery."""

import random
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from ..indexing import (
    build_base_trees,
    construct_strand_indices,
    encode_header,
)
from ..indexing.index_recovery.frequency_aggregation import (
    aggregate_bcjr_payloads_by_frequency,
)
from ..indexing.index_recovery.frequency_intervals import (
    _decoded_tree_order_key,
    _maximum_cardinality_interval_ranks,
    _pool_bounded_interval_ranks,
)
from ..indexing.index_recovery.frequency_recovery import (
    recover_strand_indices_by_frequency,
)
from ..pipeline.decoder import decode
from ..pipeline.encoder import encode


def _result(strand, marker):
    return (
        strand,
        {"byte_posterior_risks": [marker, marker + 0.01]},
    )


def _different_checksum(header):
    replacement = next(base for base in "ATGC" if base != header[0])
    return replacement + header[1:]


class PayloadRepresentativeTests(unittest.TestCase):
    def setUp(self):
        self.payload = "ATGC" * 20
        trees = build_base_trees(self.payload, 0, 80)
        self.header0 = encode_header(trees, 0)
        self.header1 = encode_header(trees, 1)

    def test_unique_highest_header_uses_total_payload_frequency(self):
        first = _result(self.header0 + self.payload, 0.9)
        second = _result(self.header1 + self.payload, 0.1)

        retained, frequencies, diagnostics = (
            aggregate_bcjr_payloads_by_frequency(
                [("read-a", 2), ("read-b", 1)],
                [first, second],
                tree_start=0,
                tree_length=80,
            )
        )

        self.assertEqual(retained, [first])
        self.assertEqual(frequencies, [3])
        self.assertEqual(diagnostics["total_selected_read_support"], 3)
        self.assertEqual(
            diagnostics["representative_decision_counts"],
            {"unique_highest_header_frequency_at_least_two": 1},
        )
        self.assertEqual(diagnostics["frequency_histogram"], {3: 1})

    def test_all_singleton_headers_select_unique_checksum_valid_observation(self):
        invalid = _different_checksum(self.header0)
        invalid_result = _result(invalid + self.payload, 0.01)
        valid_result = _result(self.header0 + self.payload, 0.99)

        retained, frequencies, diagnostics = (
            aggregate_bcjr_payloads_by_frequency(
                [("read-invalid", 1), ("read-valid", 1)],
                [invalid_result, valid_result],
                tree_start=0,
                tree_length=80,
            )
        )

        self.assertEqual(retained, [valid_result])
        self.assertEqual(frequencies, [2])
        self.assertEqual(
            diagnostics["representative_decision_counts"],
            {"all_headers_frequency_one_unique_checksum_valid": 1},
        )

    def test_all_singleton_headers_with_two_valid_observations_are_erased(self):
        retained, frequencies, diagnostics = (
            aggregate_bcjr_payloads_by_frequency(
                [("read-0", 1), ("read-1", 1)],
                [
                    _result(self.header0 + self.payload, 0.1),
                    _result(self.header1 + self.payload, 0.9),
                ],
                tree_start=0,
                tree_length=80,
            )
        )

        self.assertEqual(retained, [])
        self.assertEqual(frequencies, [])
        self.assertEqual(
            diagnostics["representative_decision_counts"],
            {
                "all_headers_frequency_one_multiple_checksum_valid_discarded": 1
            },
        )

    def test_tied_repeated_headers_are_erased_without_posterior_tiebreak(self):
        retained, frequencies, diagnostics = (
            aggregate_bcjr_payloads_by_frequency(
                [("read-0", 2), ("read-1", 2)],
                [
                    _result(self.header0 + self.payload, 0.99),
                    _result(self.header1 + self.payload, 0.01),
                ],
                tree_start=0,
                tree_length=80,
            )
        )

        self.assertEqual(retained, [])
        self.assertEqual(frequencies, [])
        self.assertEqual(
            diagnostics["representative_decision_counts"],
            {"tied_highest_header_frequency_discarded": 1},
        )

    def test_payload_frequency_one_is_discarded_before_header_histogram(self):
        retained, frequencies, diagnostics = (
            aggregate_bcjr_payloads_by_frequency(
                [("read-0", 1)],
                [_result(self.header0 + self.payload, 0.1)],
                tree_start=0,
                tree_length=80,
            )
        )

        self.assertEqual(retained, [])
        self.assertEqual(frequencies, [])
        self.assertEqual(
            diagnostics["discarded_singleton_payload_count"], 1
        )
        self.assertEqual(diagnostics["retained_payload_count"], 0)

    def test_invalid_bcjr_outputs_keep_weighted_failure_counts(self):
        _, _, diagnostics = aggregate_bcjr_payloads_by_frequency(
            [("read-none", 3), ("read-invalid", 2)],
            [
                _result(None, 0.1),
                _result("AX", 0.2),
            ],
            tree_start=0,
            tree_length=80,
        )

        self.assertEqual(diagnostics["bcjr_failed_unique_read_count"], 2)
        self.assertEqual(diagnostics["bcjr_failed_read_count"], 5)

    def test_same_header_keeps_first_complete_bcjr_result(self):
        first = _result(self.header0 + self.payload, 0.9)
        second = _result(self.header0 + self.payload, 0.1)

        retained, frequencies, _ = aggregate_bcjr_payloads_by_frequency(
            [("read-first", 1), ("read-second", 1)],
            [first, second],
            tree_start=0,
            tree_length=80,
        )

        self.assertEqual(frequencies, [2])
        self.assertIs(retained[0], first)
        self.assertEqual(
            retained[0][1]["byte_posterior_risks"],
            [0.9, 0.91],
        )


class PayloadRepresentativeTreeTests(unittest.TestCase):
    @staticmethod
    def _mock_selection(phase, index_assignments):
        return {
            "phase": phase,
            "frequency": 2,
            "acceptance_reason": "mocked_test_selection",
            "candidate_indices": list(range(100)),
            "index_assignments": dict(index_assignments),
            "recovery_methods": dict.fromkeys(
                index_assignments,
                "initial_bidirectional_interval_alignment",
            ),
            "initial_recovered_index_count": len(index_assignments),
            "diagnostics": {
                "candidate_count": 100,
                "recovered_index_count": len(index_assignments),
            },
            "build_attempts": [
                {
                    "path": phase,
                    "minimum_payload_frequency": 2,
                    "sparse_alignment_enabled": (
                        phase == "sparse_alignment_rescue"
                    ),
                }
            ],
        }

    @staticmethod
    def _global_recovery_side_effect(results):
        results = iter(results)

        def recover(
            _normalized,
            _frequencies,
            candidate_indices,
            *_args,
            **_kwargs,
        ):
            local_assignments, stages, diagnostics = next(results)
            return (
                {
                    target: candidate_indices[local_result]
                    for target, local_result in local_assignments.items()
                },
                stages,
                diagnostics,
            )

        return recover

    def test_dense_interval_alignment_discards_one_cycle_outlier(self):
        correct_indices = list(range(80))
        header_order = [*correct_indices[:40], 99, *correct_indices[40:]]
        cycle_by_result_index = {
            result_index: (result_index * 4) % 64
            for result_index in correct_indices
        }
        cycle_by_result_index[99] = 63
        frequency_by_result_index = dict.fromkeys(header_order, 2)

        assignments, diagnostics = _maximum_cardinality_interval_ranks(
            header_order,
            cycle_by_result_index,
            frequency_by_result_index,
            left_rank=-1,
            right_rank=320,
        )

        self.assertEqual(len(assignments), 80)
        self.assertNotIn(99, assignments)
        self.assertEqual(
            assignments,
            {result_index: result_index * 4 for result_index in correct_indices},
        )
        self.assertEqual(diagnostics["discarded_outlier_count"], 1)

    def test_decoded_tree_key_preserves_large_result_index_fallback(self):
        strand = "AAAAATGC"
        lower = _decoded_tree_order_key(strand, 2**64, 1, 2)
        higher = _decoded_tree_order_key(strand, 2**80, 1, 2)

        self.assertLess(lower, higher)

    def test_sparse_alignment_preserves_large_global_result_indices(self):
        lower = 2**64
        higher = 2**80
        assignments, _ = _maximum_cardinality_interval_ranks(
            [lower, higher],
            {lower: 0, higher: 0},
            {lower: 2**63, higher: 1},
            left_rank=-1,
            right_rank=128,
        )

        self.assertEqual(assignments, {lower: 0, higher: 64})

    def test_alias_risk_gate_accepts_sparse_interval_when_egap_is_at_most_one(self):
        candidate_count = 100
        header_order = list(range(candidate_count))
        with patch(
            "Beezy.indexing.index_recovery.frequency_intervals."
            "_exact_bidirectional_interval",
            return_value=(None, "mocked_disagreement"),
        ), patch(
            "Beezy.indexing.index_recovery.frequency_intervals."
            "_maximum_cardinality_interval_ranks",
            return_value=({0: 0}, {"selected_count": 1}),
        ) as sparse_mock:
            _, _, diagnostics = _pool_bounded_interval_ranks(
                header_order,
                dict.fromkeys(header_order, 0),
                dict.fromkeys(header_order, 2),
                strand_count=1000,
                enable_outlier_filter=True,
            )

        sparse_mock.assert_called_once()
        interval = diagnostics["accepted_intervals"][0]
        self.assertLessEqual(
            interval["outlier_fallback_expected_long_gap_count"],
            1.0,
        )
        self.assertTrue(interval["outlier_fallback_alias_risk_accepted"])

    def test_alias_risk_gate_rejects_interval_when_egap_exceeds_one(self):
        candidate_count = 50
        header_order = list(range(candidate_count))
        with patch(
            "Beezy.indexing.index_recovery.frequency_intervals."
            "_exact_bidirectional_interval",
            return_value=(None, "mocked_disagreement"),
        ), patch(
            "Beezy.indexing.index_recovery.frequency_intervals."
            "_maximum_cardinality_interval_ranks",
        ) as sparse_mock:
            additions, _, diagnostics = _pool_bounded_interval_ranks(
                header_order,
                dict.fromkeys(header_order, 0),
                dict.fromkeys(header_order, 2),
                strand_count=1000,
                enable_outlier_filter=True,
            )

        sparse_mock.assert_not_called()
        self.assertEqual(additions, {})
        interval = diagnostics["rejected_intervals"][0]
        self.assertGreater(
            interval["outlier_fallback_expected_long_gap_count"],
            1.0,
        )
        self.assertEqual(
            interval["reason"],
            "outlier_fallback_expected_long_gap_budget_exceeded",
        )

    def test_descending_frequency_tree_policy_recovers_clean_pool(self):
        rng = random.Random(20260808)
        payloads = [
            "".join(rng.choices("ATGC", k=80))
            for _ in range(70)
        ]
        indexed, tree_seed, _, _ = construct_strand_indices(payloads)
        rank_by_target = {
            one_based_target - 1: rank
            for rank, one_based_target in enumerate(tree_seed)
        }
        frequencies = [
            5 if rank_by_target[target] % 2 == 0 else 4
            for target in range(len(indexed))
        ]

        with patch(
            "Beezy.indexing.index_recovery.frequency_intervals."
            "_maximum_cardinality_interval_ranks",
            side_effect=AssertionError("clean strict path invoked rescue DP"),
        ):
            ordered, placements, diagnostics = (
                recover_strand_indices_by_frequency(
                    indexed,
                    frequencies,
                    tree_seed,
                    tree_start=0,
                    tree_length=80,
                )
            )

        self.assertEqual(placements, list(enumerate(range(len(indexed)))))
        self.assertEqual(ordered, dict(enumerate(payloads)))
        self.assertEqual(
            diagnostics["algorithm"],
            "descending_frequency_frozen_tree_index_recovery",
        )
        self.assertEqual(
            diagnostics["summary"]["stop_reason"],
            "all_indices_recovered",
        )
        self.assertFalse(diagnostics["rescue"]["triggered"])
        self.assertEqual(
            diagnostics["initial_recovery"]["selected_path"],
            "strict_alignment",
        )
        self.assertTrue(
            all(
                attempt["path"] == "strict_alignment"
                and not attempt["sparse_alignment_enabled"]
                for attempt in diagnostics["initial_recovery"]["build_attempts"]
            )
        )
        self.assertEqual(diagnostics["initial_recovery"]["target_candidate_count"], 18)
        records = diagnostics["recovered_index_records"]
        self.assertEqual(len(records), len(indexed))
        for record in records:
            self.assertEqual(
                set(record),
                {
                    "recovered_strand_index",
                    "tree_rank",
                    "payload_frequency",
                    "recovery_method",
                },
            )
            self.assertEqual(
                record["tree_rank"],
                rank_by_target[record["recovered_strand_index"]],
            )
        serialized_diagnostics = repr(diagnostics)
        self.assertNotIn("candidate_position", serialized_diagnostics)
        self.assertNotIn("placement", serialized_diagnostics)
        self.assertNotIn("backbone", serialized_diagnostics)
        self.assertNotIn("four_way", serialized_diagnostics)
        self.assertNotIn("strong_anchor", serialized_diagnostics)

    def test_quarter_prefix_preserves_available_hard_agreement_placements(self):
        rng = random.Random(20260808)
        payloads = [
            "".join(rng.choices("ATGC", k=80))
            for _ in range(70)
        ]
        indexed, tree_seed, _, _ = construct_strand_indices(payloads)
        frequencies = [
            2 if target < 34 else 1
            for target in range(len(indexed))
        ]

        ordered, placements, diagnostics = recover_strand_indices_by_frequency(
            indexed,
            frequencies,
            tree_seed,
            tree_start=0,
            tree_length=80,
        )

        self.assertEqual(
            diagnostics["initial_recovery"][
                "minimum_required_recovered_index_count"
            ],
            14,
        )
        self.assertEqual(diagnostics["initial_recovery"]["target_candidate_count"], 18)
        self.assertEqual(len(placements), 34)
        self.assertEqual(len(diagnostics["initial_recovery"]["build_attempts"]), 1)
        self.assertEqual(
            diagnostics["initial_recovery"]["selected_minimum_payload_frequency"],
            2,
        )
        self.assertEqual(
            diagnostics["summary"]["stop_reason"],
            "minimum_payload_frequency_exhausted",
        )
        self.assertEqual(
            ordered,
            {target: payloads[target] for target in range(34)},
        )
        self.assertTrue(
            all(
                record["payload_frequency"] == 2
                for record in diagnostics["recovered_index_records"]
            )
        )

    def test_backs_off_whole_frequency_layer_until_twenty_percent_placed(self):
        rng = random.Random(20260808)
        payloads = [
            "".join(rng.choices("ATGC", k=80))
            for _ in range(100)
        ]
        indexed, tree_seed, _, _ = construct_strand_indices(payloads)
        frequencies = [3 if target < 20 else 2 for target in range(100)]

        first_assignments = {0: 0}
        second_assignments = {target: target for target in range(20)}
        with patch(
            "Beezy.indexing.index_recovery.frequency_recovery."
            "_recover_initial_assignments",
            side_effect=self._global_recovery_side_effect([
                (
                    first_assignments,
                    {0: "initial_bidirectional_interval_alignment"},
                    {"candidate_count": 100, "recovered_index_count": 1},
                ),
                (
                    second_assignments,
                    {
                        target: "initial_bidirectional_interval_alignment"
                        for target in second_assignments
                    },
                    {"candidate_count": 20, "recovered_index_count": 20},
                ),
                (
                    first_assignments,
                    {0: "initial_bidirectional_interval_alignment"},
                    {"candidate_count": 100, "recovered_index_count": 1},
                ),
                (
                    second_assignments,
                    {
                        target: "initial_bidirectional_interval_alignment"
                        for target in second_assignments
                    },
                    {"candidate_count": 20, "recovered_index_count": 20},
                ),
            ]),
        ):
            _, _, diagnostics = recover_strand_indices_by_frequency(
                indexed,
                frequencies,
                tree_seed,
                tree_start=0,
                tree_length=80,
            )

        attempts = diagnostics["initial_recovery"]["build_attempts"]
        strict_attempts = [
            item for item in attempts if item["path"] == "strict_alignment"
        ]
        self.assertEqual(
            [item["minimum_payload_frequency"] for item in strict_attempts],
            [2, 3],
        )
        self.assertEqual(
            strict_attempts[0]["outcome"],
            "retry_without_lowest_frequency_layer",
        )
        self.assertEqual(
            strict_attempts[1]["outcome"],
            "accept_minimum_initial_recovery_reached",
        )
        self.assertEqual(
            diagnostics["initial_recovery"]["selected_minimum_payload_frequency"], 3
        )
        self.assertEqual(
            diagnostics["initial_recovery"]["initial_recovered_index_count"],
            20,
        )
        self.assertEqual(
            diagnostics["initial_recovery"]["selection_reason"],
            "minimum_initial_recovery_reached",
        )
        self.assertTrue(diagnostics["rescue"]["triggered"])
        self.assertFalse(diagnostics["rescue"]["replaced_strict_path"])

    def test_complete_strict_descent_prevents_rescue_after_empty_initial_recovery(self):
        rng = random.Random(20260808)
        payloads = [
            "".join(rng.choices("ATGC", k=80))
            for _ in range(100)
        ]
        indexed, tree_seed, _, _ = construct_strand_indices(payloads)
        frequencies = [3 if target < 20 else 2 for target in range(100)]
        with patch(
            "Beezy.indexing.index_recovery.frequency_recovery."
            "_recover_initial_assignments",
            side_effect=[
                ({}, {}, {"candidate_count": 100, "recovered_index_count": 0}),
                ({}, {}, {"candidate_count": 20, "recovered_index_count": 0}),
            ],
        ) as recover_mock:
            _, _, diagnostics = recover_strand_indices_by_frequency(
                indexed,
                frequencies,
                tree_seed,
                tree_start=0,
                tree_length=80,
            )

        attempts = diagnostics["initial_recovery"]["build_attempts"]
        self.assertEqual(
            [item["path"] for item in attempts],
            ["strict_alignment", "strict_alignment"],
        )
        self.assertEqual(
            [item["minimum_payload_frequency"] for item in attempts],
            [2, 3],
        )
        self.assertEqual(
            [
                call.kwargs["enable_outlier_filter"]
                for call in recover_mock.call_args_list
            ],
            [False, False],
        )
        self.assertFalse(diagnostics["rescue"]["triggered"])
        self.assertEqual(
            diagnostics["initial_recovery"]["selected_path"],
            "strict_alignment",
        )
        self.assertEqual(
            diagnostics["initial_recovery"][
                "strict_path_initial_recovered_index_count"
            ],
            0,
        )
        self.assertGreaterEqual(
            diagnostics["initial_recovery"]["strict_path_final_recovered_index_count"],
            diagnostics["rescue"]["minimum_required_recovered_index_count"],
        )

    def test_below_half_normal_result_restarts_and_uses_better_rescue(self):
        rng = random.Random(20260808)
        payloads = ["".join(rng.choices("ATGC", k=80)) for _ in range(100)]
        indexed, tree_seed, _, _ = construct_strand_indices(payloads)
        normal = self._mock_selection("strict_alignment", {0: 0})
        rescue = self._mock_selection(
            "sparse_alignment_rescue",
            {target: target for target in range(60)},
        )

        with patch(
            "Beezy.indexing.index_recovery.frequency_recovery."
            "_run_strict_initial_recovery",
            return_value=normal,
        ), patch(
            "Beezy.indexing.index_recovery.frequency_recovery."
            "_run_sparse_initial_recovery",
            return_value=rescue,
        ), patch(
            "Beezy.indexing.index_recovery.frequency_recovery."
            "_descend_frequency_levels",
            side_effect=[({}, []), ({}, []), ({}, [])],
        ):
            _, placements, diagnostics = recover_strand_indices_by_frequency(
                indexed,
                [2] * len(indexed),
                tree_seed,
                tree_start=0,
                tree_length=80,
            )

        self.assertEqual(len(placements), 60)
        self.assertEqual(diagnostics["rescue"]["reference_candidate_count"], 100)
        self.assertEqual(
            diagnostics["rescue"][
                "minimum_required_recovered_index_count"
            ],
            50,
        )
        self.assertTrue(diagnostics["rescue"]["triggered"])
        self.assertTrue(diagnostics["rescue"]["replaced_strict_path"])
        self.assertEqual(
            diagnostics["initial_recovery"]["selected_path"],
            "sparse_alignment_rescue",
        )

    def test_below_quarter_candidates_skip_normal_and_use_all_frequency_two(self):
        rng = random.Random(20260808)
        payloads = ["".join(rng.choices("ATGC", k=80)) for _ in range(100)]
        indexed, tree_seed, _, _ = construct_strand_indices(payloads)
        frequencies = [
            3 if target < 20 else 1
            for target in range(100)
        ]
        rescue_assignments = {target: target for target in range(8)}

        with patch(
            "Beezy.indexing.index_recovery.frequency_recovery."
            "_recover_initial_assignments",
            side_effect=self._global_recovery_side_effect([
                (
                    rescue_assignments,
                    {
                        target: "initial_bidirectional_interval_alignment"
                        for target in rescue_assignments
                    },
                    {"candidate_count": 20, "recovered_index_count": 8},
                ),
            ]),
        ) as recover_mock:
            _, _, diagnostics = recover_strand_indices_by_frequency(
                indexed,
                frequencies,
                tree_seed,
                tree_start=0,
                tree_length=80,
            )

        self.assertTrue(diagnostics["rescue"]["strict_path_skipped"])
        self.assertIsNone(
            diagnostics["initial_recovery"][
                "strict_path_final_recovered_index_count"
            ]
        )
        self.assertEqual(
            diagnostics["rescue"]["trigger_reason"],
            "insufficient_checksum_valid_candidates_for_strict_path",
        )
        self.assertEqual(
            [
                call.kwargs["enable_outlier_filter"]
                for call in recover_mock.call_args_list
            ],
            [True],
        )
        attempts = diagnostics["initial_recovery"]["build_attempts"]
        self.assertEqual(
            [attempt["minimum_payload_frequency"] for attempt in attempts],
            [2],
        )
        self.assertEqual(
            attempts[0]["outcome"],
            "accept_fixed_initial_prefix_sparse_alignment",
        )
        self.assertEqual(
            diagnostics["initial_recovery"]["selected_path"],
            "sparse_alignment_rescue",
        )

    def test_guarded_descent_keeps_selected_rescue_and_only_initial_anchors(self):
        rng = random.Random(17)
        payloads = ["".join(rng.choices("ATGC", k=80)) for _ in range(100)]
        indexed, tree_seed, _, _ = construct_strand_indices(payloads)
        strict = self._mock_selection("strict_alignment", {0: 0})
        rescue = self._mock_selection("sparse_alignment_rescue", {1: 1})
        calls = []

        def descend(selection, *args):
            guarded = args[-1] is True
            calls.append((selection["phase"], guarded))
            if guarded:
                self.assertEqual(selection["index_assignments"], {1: 1})
                self.assertEqual(set(selection["recovery_methods"]), {1})
                targets = range(2, 92)
            else:
                targets = [50] if selection["phase"] == "strict_alignment" else [51, 52]
            for target in targets:
                selection["index_assignments"][target] = target
                selection["recovery_methods"][target] = "frequency_2_exact_interval_unique_match"
            return {}, []

        module = "Beezy.indexing.index_recovery.frequency_recovery."
        with patch(module + "_run_strict_initial_recovery", return_value=strict), \
                patch(module + "_run_sparse_initial_recovery", return_value=rescue), \
                patch(module + "_descend_frequency_levels", side_effect=descend):
            _, assignments, diagnostics = recover_strand_indices_by_frequency(
                indexed, [2] * 100, tree_seed, tree_length=80)
        self.assertEqual(calls, [("strict_alignment", False),
            ("sparse_alignment_rescue", False), ("sparse_alignment_rescue", True)])
        self.assertEqual(len(assignments), 91)
        self.assertEqual(diagnostics["path_selection_pass_recovered_index_count"], 3)
        self.assertTrue(diagnostics["rescue"]["selected"])

    def test_rescue_uses_original_complete_layer_that_first_reaches_quarter(self):
        rng = random.Random(20260808)
        payloads = ["".join(rng.choices("ATGC", k=80)) for _ in range(100)]
        indexed, tree_seed, _, _ = construct_strand_indices(payloads)
        frequencies = [
            5 if target < 10 else 4 if target < 30 else 3
            for target in range(100)
        ]
        normal = self._mock_selection("strict_alignment", {})
        normal["frequency"] = 5

        with patch(
            "Beezy.indexing.index_recovery.frequency_recovery."
            "_run_strict_initial_recovery",
            return_value=normal,
        ), patch(
            "Beezy.indexing.index_recovery.frequency_recovery."
            "_recover_initial_assignments",
            return_value=(
                {},
                {},
                {"candidate_count": 30, "recovered_index_count": 0},
            ),
        ) as recover_mock, patch(
            "Beezy.indexing.index_recovery.frequency_recovery."
            "_descend_frequency_levels",
            return_value=({}, []),
        ):
            _, _, diagnostics = recover_strand_indices_by_frequency(
                indexed,
                frequencies,
                tree_seed,
                tree_start=0,
                tree_length=80,
            )

        rescue_candidates = recover_mock.call_args.args[2]
        self.assertEqual(len(rescue_candidates), 30)
        rescue_attempt = diagnostics["initial_recovery"]["build_attempts"][-1]
        self.assertEqual(rescue_attempt["path"], "sparse_alignment_rescue")
        self.assertEqual(rescue_attempt["minimum_payload_frequency"], 4)
        self.assertEqual(rescue_attempt["input_candidate_count"], 30)

    def test_rescue_trigger_denominator_is_capped_by_tree_seed_length(self):
        rng = random.Random(20260808)
        payloads = ["".join(rng.choices("ATGC", k=80)) for _ in range(100)]
        indexed, tree_seed, _, _ = construct_strand_indices(payloads)
        candidates = [*indexed, *indexed[:20]]
        normal = self._mock_selection(
            "strict_alignment",
            {target: target for target in range(50)},
        )

        with patch(
            "Beezy.indexing.index_recovery.frequency_recovery."
            "_run_strict_initial_recovery",
            return_value=normal,
        ) as phase_mock, patch(
            "Beezy.indexing.index_recovery.frequency_recovery."
            "_descend_frequency_levels",
            return_value=({}, []),
        ):
            _, _, diagnostics = recover_strand_indices_by_frequency(
                candidates,
                [2] * len(candidates),
                tree_seed,
                tree_start=0,
                tree_length=80,
            )

        phase_mock.assert_called_once()
        self.assertEqual(diagnostics["retained_candidate_count"], 120)
        self.assertEqual(diagnostics["rescue"]["reference_candidate_count"], 100)
        self.assertEqual(
            diagnostics["rescue"][
                "minimum_required_recovered_index_count"
            ],
            50,
        )
        self.assertFalse(diagnostics["rescue"]["triggered"])

    def test_keeps_final_build_when_candidate_prefix_drops_below_twenty_percent(self):
        rng = random.Random(20260808)
        payloads = [
            "".join(rng.choices("ATGC", k=80))
            for _ in range(70)
        ]
        indexed, tree_seed, _, _ = construct_strand_indices(payloads)
        frequencies = [
            3 if target < 10 else 2 if target < 40 else 1
            for target in range(70)
        ]

        first_assignments = {0: 0}
        second_assignments = {target: target for target in range(8)}

        def finish_normal_path_at_half_of_valid_candidates(selection, *_args):
            for target in range(20):
                selection["index_assignments"].setdefault(target, target)
                selection["recovery_methods"].setdefault(
                    target,
                    "frequency_2_exact_interval_unique_match",
                )
            return {}, []

        with patch(
            "Beezy.indexing.index_recovery.frequency_recovery."
            "_recover_initial_assignments",
            side_effect=self._global_recovery_side_effect([
                (
                    first_assignments,
                    {0: "initial_bidirectional_interval_alignment"},
                    {"candidate_count": 40, "recovered_index_count": 1},
                ),
                (
                    second_assignments,
                    {
                        target: "initial_bidirectional_interval_alignment"
                        for target in second_assignments
                    },
                    {"candidate_count": 10, "recovered_index_count": 8},
                ),
            ]),
        ), patch(
            "Beezy.indexing.index_recovery.frequency_recovery."
            "_descend_frequency_levels",
            side_effect=finish_normal_path_at_half_of_valid_candidates,
        ):
            _, _, diagnostics = recover_strand_indices_by_frequency(
                indexed,
                frequencies,
                tree_seed,
                tree_start=0,
                tree_length=80,
            )

        attempts = diagnostics["initial_recovery"]["build_attempts"]
        strict_attempts = [
            item for item in attempts if item["path"] == "strict_alignment"
        ]
        self.assertEqual(
            [item["minimum_payload_frequency"] for item in strict_attempts],
            [2, 3],
        )
        self.assertTrue(
            strict_attempts[1]["candidate_count_below_initial_recovery_minimum"]
        )
        self.assertEqual(
            strict_attempts[1]["outcome"],
            "accept_all_available_candidates_below_minimum",
        )
        self.assertEqual(
            diagnostics["initial_recovery"]["initial_recovered_index_count"],
            8,
        )
        self.assertEqual(
            diagnostics["initial_recovery"]["selection_reason"],
            "all_available_candidates_used_below_minimum",
        )

    def test_no_frequency_two_candidate_still_returns_no_placement(self):
        rng = random.Random(20260808)
        payloads = [
            "".join(rng.choices("ATGC", k=80))
            for _ in range(10)
        ]
        indexed, tree_seed, _, _ = construct_strand_indices(payloads)

        ordered, placements, diagnostics = recover_strand_indices_by_frequency(
            indexed,
            [1] * len(indexed),
            tree_seed,
            tree_start=0,
            tree_length=80,
        )

        self.assertEqual(ordered, {})
        self.assertEqual(placements, [])
        self.assertEqual(diagnostics["initial_recovery"]["target_candidate_count"], 0)
        self.assertEqual(
            diagnostics["summary"]["stop_reason"],
            "no_recoverable_candidates",
        )

    def test_highest_frequency_exhaustion_preserves_last_build(self):
        rng = random.Random(20260808)
        payloads = [
            "".join(rng.choices("ATGC", k=80))
            for _ in range(70)
        ]
        indexed, tree_seed, _, _ = construct_strand_indices(payloads)
        mocked_diagnostics = {
            "candidate_count": 70,
            "recovered_index_count": 1,
            "recovery_method_counts": {
                "initial_bidirectional_interval_alignment": 1
            },
        }

        with patch(
            "Beezy.indexing.index_recovery.frequency_recovery."
            "_recover_initial_assignments",
            side_effect=self._global_recovery_side_effect([
                (
                    {0: 0},
                    {0: "initial_bidirectional_interval_alignment"},
                    mocked_diagnostics,
                ),
            ]),
        ):
            _, placements, diagnostics = recover_strand_indices_by_frequency(
                indexed,
                [2] * len(indexed),
                tree_seed,
                tree_start=0,
                tree_length=80,
            )

        self.assertIn(0, dict(placements))
        self.assertEqual(diagnostics["initial_recovery"]["target_candidate_count"], 18)
        self.assertEqual(
            diagnostics["initial_recovery"]["initial_recovered_index_count"],
            1,
        )
        self.assertEqual(
            diagnostics["initial_recovery"]["selection_reason"],
            "highest_frequency_layer_used_below_minimum",
        )
        self.assertEqual(len(diagnostics["initial_recovery"]["build_attempts"]), 1)


class PayloadRepresentativePipelineTests(unittest.TestCase):
    def test_two_clean_copies_restore_source_end_to_end(self):
        with tempfile.TemporaryDirectory(
            prefix="beezy-frequency-recovery-test-"
        ) as directory:
            root = Path(directory)
            source = root / "source.bin"
            # Use distinct two-byte chunks so this test exercises the clean
            # multi-copy route rather than deliberate duplicate-payload
            # ambiguity.
            source.write_bytes(bytes(range(64)))
            prefix = str(root / "case")
            random.seed(20260808)
            tree_seed, _, _ = encode(
                artifact_prefix=prefix,
                payload_length=8,
                tree_length=8,
                source_path=source,
                parity_strand_count=4,
            )
            encoded_strands = (
                Path(prefix + "_encoded_strands.txt").read_text().splitlines()
            )
            reads = [strand for strand in encoded_strands for _ in range(2)]
            Path(prefix + "_reads").write_text(
                "\n".join(reads) + "\n",
                encoding="utf-8",
            )
            restored = root / "restored.bin"

            _, diagnostics = decode(
                reads_path=Path(prefix + "_reads"),
                tree_seed_path=Path(prefix + "_seed"),
                channel={"ps": 0.0, "pd": 0.0, "pi": 0.0, "column": 0.0},
                output_path=restored,
                drift_window=0,
                max_insertions_per_symbol=0,
                worker_count=1,
                return_diagnostics=True,
                parity_strand_count=4,
                payload_length=8,
                tree_length=8,
            )

            self.assertEqual(restored.read_bytes(), source.read_bytes())
            self.assertEqual(
                diagnostics["payload_aggregation"]["selected_candidate_count"],
                len(tree_seed),
            )
            self.assertEqual(
                diagnostics["payload_aggregation"][
                    "representative_decision_counts"
                ],
                {"unique_highest_header_frequency_at_least_two": len(tree_seed)},
            )
            self.assertEqual(
                diagnostics["index_recovery"]["summary"]["stop_reason"],
                "all_indices_recovered",
            )
            self.assertFalse(diagnostics["reed_solomon"]["output_is_partial"])


if __name__ == "__main__":
    unittest.main()
