"""Regression tests for posterior-guided production decoding."""

import random
import unittest
from unittest import mock

import numpy as np

from ..ecc import bcjr
from ..ecc.trellis_encoding import (
    ConstrainedTrellis,
    PAIR_TO_SYMBOL,
    TRELLIS,
    encode_constrained,
)
from ..ecc.rs import (
    vertical_rs_decode_hard,
    vertical_rs_decode_adaptive,
    vertical_rs_encode,
)

CHANNEL = {"ps": 0.03, "pd": 0.03, "pi": 0.03}


def _relabel_trellis(permutation):
    inverse = [0] * len(permutation)
    for old_state, new_state in enumerate(permutation):
        inverse[new_state] = old_state
    return ConstrainedTrellis(
        output_table=tuple(
            TRELLIS.output_table[inverse[state]]
            for state in range(TRELLIS.number_states)
        ),
        next_state_table=tuple(
            tuple(permutation[symbol] for symbol in range(TRELLIS.number_inputs))
            for _ in range(TRELLIS.number_states)
        ),
        initial_state=permutation[TRELLIS.initial_state],
        termination_symbol=TRELLIS.termination_symbol,
        number_states=TRELLIS.number_states,
        number_inputs=TRELLIS.number_inputs,
    )


def _encode_with_trellis(source, trellis):
    state = trellis.initial_state
    encoded = []
    for offset in range(0, len(source), 2):
        symbol = PAIR_TO_SYMBOL[source[offset : offset + 2]]
        encoded.append(trellis.output_table[state][symbol])
        state = trellis.next_state_table[state][symbol]
    encoded.append(
        trellis.output_table[state][trellis.termination_symbol]
    )
    return "".join(encoded), len(source) // 2


class AdaptiveDecoderTests(unittest.TestCase):
    def test_bcjr_follows_nonidentity_next_state_table(self):
        permutation = tuple((state + 5) % 16 for state in range(16))
        trellis = _relabel_trellis(permutation)
        self.assertEqual(trellis.initial_state, 5)
        self.assertTrue(
            all(
                trellis.next_state_table[state][symbol] != symbol
                for state in range(16)
                for symbol in range(16)
            )
        )

        source = "ATGCCGTAACGT"
        transmitted, payload_steps = _encode_with_trellis(source, trellis)
        cython_result = bcjr.decode_hard(
            transmitted,
            payload_steps,
            CHANNEL,
            drift_window=3,
            return_diagnostics=True,
            return_byte_posterior_risks=True,
            trellis=trellis,
        )
        with mock.patch.object(bcjr, "_CYTHON_FULL_LOG_MAP", False):
            numpy_result = bcjr.decode_hard(
                transmitted,
                payload_steps,
                CHANNEL,
                drift_window=3,
                return_diagnostics=True,
                return_byte_posterior_risks=True,
                trellis=trellis,
            )

        self.assertEqual(cython_result[0], source)
        self.assertEqual(numpy_result[0], source)
        self.assertEqual(cython_result[1]["initial_state"], 5)
        self.assertEqual(cython_result[1]["terminal_states"], [5])
        np.testing.assert_allclose(
            cython_result[1]["byte_posterior_risks"],
            numpy_result[1]["byte_posterior_risks"],
            atol=1e-12,
        )

    def test_bcjr_posterior_risks_share_the_hard_decode_pass(self):
        source = "ATGCCGTAACGT"
        transmitted, payload_steps = encode_constrained(source)
        received = list(transmitted)
        received[5] = "A" if received[5] != "A" else "C"
        del received[11]
        received.insert(17, "T")
        received = "".join(received)

        ordinary = bcjr.decode_hard(
            received,
            payload_steps,
            CHANNEL,
            drift_window=3,
        )
        bcjr_decoded_strand, diagnostics = bcjr.decode_hard(
            received,
            payload_steps,
            CHANNEL,
            drift_window=3,
            return_diagnostics=True,
            return_byte_posterior_risks=True,
        )

        self.assertEqual(bcjr_decoded_strand, ordinary)
        self.assertEqual(len(diagnostics["byte_posterior_risks"]), 3)
        self.assertTrue(
            all(0.0 <= risk <= 1.0 for risk in diagnostics["byte_posterior_risks"])
        )

    def test_numpy_and_cython_trellises_match_posterior_risks(self):
        source = "ATGCCGTAACGT"
        transmitted, payload_steps = encode_constrained(source)
        received = transmitted[:7] + "A" + transmitted[8:]
        cython_result = bcjr.decode_hard(
            received,
            payload_steps,
            CHANNEL,
            drift_window=3,
            return_diagnostics=True,
            return_byte_posterior_risks=True,
        )
        with mock.patch.object(bcjr, "_CYTHON_FULL_LOG_MAP", False):
            numpy_result = bcjr.decode_hard(
                received,
                payload_steps,
                CHANNEL,
                drift_window=3,
                return_diagnostics=True,
                return_byte_posterior_risks=True,
            )

        self.assertEqual(cython_result[0], numpy_result[0])
        np.testing.assert_allclose(
            cython_result[1]["byte_posterior_risks"],
            numpy_result[1]["byte_posterior_risks"],
            atol=1e-12,
        )

    def test_rs_retries_only_raised_columns_in_risk_order(self):
        source = ["AAAA"] * 10
        encoded, padding_bases = vertical_rs_encode(
            source,
            parity_strand_count=4,
        )
        received = {index: sequence for index, sequence in enumerate(encoded)}
        error_positions = {1, 3, 11}
        for position in error_positions:
            received[position] = "C" + received[position][1:]
        posterior_risks = {
            position: [0.9 if position in error_positions else 0.1]
            for position in received
        }

        _, hard_diagnostics = vertical_rs_decode_hard(
            received,
            data_strand_count=10,
            parity_strand_count=4,
            padding_bases=padding_bases,
            return_diagnostics=True,
        )
        decoded, diagnostics = vertical_rs_decode_adaptive(
            received,
            posterior_risks,
            data_strand_count=10,
            parity_strand_count=4,
            padding_bases=padding_bases,
            return_diagnostics=True,
        )
        transformed, transformed_diagnostics = vertical_rs_decode_adaptive(
            received,
            {
                position: [7.0 + 100.0 * value[0]]
                for position, value in posterior_risks.items()
            },
            data_strand_count=10,
            parity_strand_count=4,
            padding_bases=padding_bases,
            return_diagnostics=True,
        )

        self.assertEqual(hard_diagnostics["failed_columns"], 1)
        self.assertEqual(decoded, source)
        self.assertEqual(transformed, source)
        self.assertEqual(diagnostics["hard_failed_columns"], 1)
        self.assertEqual(diagnostics["adaptive_attempts"], 2)
        self.assertEqual(
            diagnostics["accepted_erasure_count_histogram"],
            {"2": 1},
        )
        self.assertEqual(
            diagnostics["accepted_erasure_count_histogram"],
            transformed_diagnostics["accepted_erasure_count_histogram"],
        )

    def test_rs_does_not_retry_a_successful_hard_column(self):
        source = ["AAAA"] * 10
        encoded, padding_bases = vertical_rs_encode(
            source,
            parity_strand_count=4,
        )
        received = {index: sequence for index, sequence in enumerate(encoded)}
        received[1] = "C" + received[1][1:]
        posterior_risks = {position: [float(position)] for position in received}

        decoded, diagnostics = vertical_rs_decode_adaptive(
            received,
            posterior_risks,
            data_strand_count=10,
            parity_strand_count=4,
            padding_bases=padding_bases,
            return_diagnostics=True,
        )

        self.assertEqual(decoded, source)
        self.assertEqual(diagnostics["hard_failed_columns"], 0)
        self.assertEqual(diagnostics["adaptive_attempts"], 0)

    def test_rs_reports_unresolved_column_addresses(self):
        source = ["AAAA"] * 4
        encoded, padding_bases = vertical_rs_encode(
            source,
            parity_strand_count=0,
        )
        received = {
            index: sequence for index, sequence in enumerate(encoded) if index != 1
        }
        posterior_risks = {index: [0.0] for index in received}

        _, diagnostics = vertical_rs_decode_adaptive(
            received,
            posterior_risks,
            data_strand_count=4,
            parity_strand_count=0,
            padding_bases=padding_bases,
            return_diagnostics=True,
        )

        self.assertEqual(diagnostics["column_count"], 1)
        self.assertEqual(diagnostics["decoded_columns"], [])
        self.assertEqual(
            diagnostics["unresolved_columns"],
            [{"chunk": 0, "byte": 0}],
        )
        self.assertTrue(diagnostics["output_is_partial"])


if __name__ == "__main__":
    unittest.main()
