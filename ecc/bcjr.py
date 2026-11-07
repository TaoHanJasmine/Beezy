"""Hard-output Log-MAP BCJR decoder for the constrained codec."""

from concurrent.futures import ProcessPoolExecutor
import math
import os
import time

import numpy as np

from .trellis_encoding import (
    DNA_ALPHABET,
    INPUT_PAIRS,
    PAIR_TO_SYMBOL,
    TRELLIS,
)


_BACKEND_REQUEST = os.environ.get("BEEZY_BCJR_BACKEND", "auto").lower()
if _BACKEND_REQUEST not in {"auto", "cython", "numpy"}:
    raise ValueError(
        "BEEZY_BCJR_BACKEND must be 'auto', 'cython', or 'numpy'"
    )

_CYTHON_KERNELS = None
_CYTHON_IMPORT_ERROR = None
if _BACKEND_REQUEST != "numpy":
    try:
        from . import _bcjr_cython as _CYTHON_KERNELS
    except ImportError as error:
        _CYTHON_IMPORT_ERROR = error
if _BACKEND_REQUEST == "cython" and _CYTHON_KERNELS is None:
    raise ImportError(
        "the Cython BCJR backend was requested but is not compiled; run "
        "`python setup.py build_ext --inplace` from the Beezy directory"
    ) from _CYTHON_IMPORT_ERROR
_CYTHON_FULL_LOG_MAP = (
    _CYTHON_KERNELS is not None
    and hasattr(_CYTHON_KERNELS, "run_log_map")
)


DNA_BASE_INDEX = {
    base: index for index, base in enumerate(DNA_ALPHABET)
}
NEG_INF = float("-inf")
_BATCH_DECODE_CONFIG = None


def _prepare_trellis(trellis):
    state_count = int(trellis.number_states)
    input_count = int(trellis.number_inputs)
    if state_count <= 0:
        raise ValueError("the trellis must contain at least one state")
    if input_count != len(INPUT_PAIRS):
        raise ValueError(
            "the DNA pair decoder requires exactly 16 input symbols"
        )
    if len(trellis.output_table) != state_count:
        raise ValueError("output table row count does not match the trellis")
    if len(trellis.next_state_table) != state_count:
        raise ValueError(
            "next-state table row count does not match the trellis"
        )
    if any(len(row) != input_count for row in trellis.output_table):
        raise ValueError(
            "output table column count does not match the trellis"
        )
    if any(len(row) != input_count for row in trellis.next_state_table):
        raise ValueError(
            "next-state table column count does not match the trellis"
        )

    initial_state = int(trellis.initial_state)
    termination_symbol = int(trellis.termination_symbol)
    if not 0 <= initial_state < state_count:
        raise ValueError("initial state lies outside the trellis")
    if not 0 <= termination_symbol < input_count:
        raise ValueError("termination symbol lies outside the trellis")

    next_state_ids = np.asarray(trellis.next_state_table, dtype=np.int16)
    if np.any(next_state_ids < 0) or np.any(next_state_ids >= state_count):
        raise ValueError("next-state table contains an invalid state")
    terminal_state_mask = np.zeros(state_count, dtype=np.uint8)
    terminal_state_mask[next_state_ids[:, termination_symbol]] = 1
    return (
        state_count,
        input_count,
        initial_state,
        termination_symbol,
        np.ascontiguousarray(next_state_ids),
        np.ascontiguousarray(terminal_state_mask),
    )


def _log_probability(probability):
    return NEG_INF if probability <= 0.0 else math.log(probability)


def _channel_logs(channel):
    probabilities = {
        key: float(channel[key]) for key in ("pi", "pd", "ps")
    }
    if any(not 0.0 <= value <= 1.0 for value in probabilities.values()):
        raise ValueError("pi, pd, and ps must lie in [0, 1]")
    pi = probabilities["pi"]
    pd = probabilities["pd"]
    ps = probabilities["ps"]
    return {
        "no_insert": _log_probability(1.0 - pi),
        "insert_base": _log_probability(pi / 4.0),
        "delete": _log_probability(pd),
        "match": _log_probability((1.0 - pd) * (1.0 - ps)),
        "substitute": _log_probability((1.0 - pd) * ps / 3.0),
    }


def _drift_bounds(final_drift, extra_drift):
    if final_drift >= 0:
        return -extra_drift, final_drift + extra_drift
    return final_drift - extra_drift, extra_drift


def _fallback_symbol_scores(
    received,
    expected_length,
    drift_min,
    drift_max,
    log_pr,
    max_insertions_per_symbol,
):
    drift_count = drift_max - drift_min + 1
    scores = np.full(
        (expected_length, 4, drift_count, drift_count),
        NEG_INF,
        dtype=np.float64,
    )
    insertion_choices = (
        (0, 1) if max_insertions_per_symbol else (0,)
    )
    for encoded_position in range(expected_length):
        for from_drift_index in range(drift_count):
            received_position = (
                encoded_position + drift_min + from_drift_index
            )
            if not 0 <= received_position <= len(received):
                continue
            for insertion_count in insertion_choices:
                insertion_score = (
                    log_pr["no_insert"]
                    if insertion_count == 0
                    else log_pr["insert_base"]
                )
                if insertion_score == NEG_INF:
                    continue
                aligned = received_position + insertion_count
                if aligned > len(received):
                    continue
                transient_drift = aligned - encoded_position
                if not drift_min <= transient_drift <= drift_max:
                    continue
                if log_pr["delete"] != NEG_INF:
                    next_index = transient_drift - 1 - drift_min
                    if 0 <= next_index < drift_count:
                        scores[
                            encoded_position,
                            :,
                            from_drift_index,
                            next_index,
                        ] = np.logaddexp(
                            scores[
                                encoded_position,
                                :,
                                from_drift_index,
                                next_index,
                            ],
                            insertion_score + log_pr["delete"],
                        )
                if aligned >= len(received):
                    continue
                next_index = transient_drift - drift_min
                if not 0 <= next_index < drift_count:
                    continue
                observation = DNA_BASE_INDEX[received[aligned]]
                event_scores = np.full(4, log_pr["substitute"])
                event_scores[observation] = log_pr["match"]
                scores[
                    encoded_position,
                    :,
                    from_drift_index,
                    next_index,
                ] = np.logaddexp(
                    scores[
                        encoded_position,
                        :,
                        from_drift_index,
                        next_index,
                    ],
                    insertion_score + event_scores,
                )
    return scores


def _fallback_branch_scores(symbol_scores, outputs, message_steps):
    drift_count = symbol_scores.shape[2]
    branches = np.full(
        (message_steps, len(outputs), drift_count, drift_count),
        NEG_INF,
        dtype=np.float64,
    )
    for step in range(message_steps):
        start = 4 * step
        for output_index, output in enumerate(outputs):
            combined = symbol_scores[start, DNA_BASE_INDEX[output[0]]]
            for phase in range(1, 4):
                phase_scores = symbol_scores[
                    start + phase, DNA_BASE_INDEX[output[phase]]
                ]
                combined = np.logaddexp.reduce(
                    combined[:, :, None] + phase_scores[None, :, :],
                    axis=1,
                )
            branches[step, output_index] = combined
    return branches


def _channel_branch_scores(
    received,
    message_steps,
    outputs,
    channel,
    drift_min,
    drift_max,
    max_insertions_per_symbol,
):
    expected_length = 4 * message_steps
    log_pr = _channel_logs(channel)
    if _CYTHON_KERNELS is not None:
        received_ids = np.fromiter(
            (DNA_BASE_INDEX[base] for base in received),
            dtype=np.uint8,
            count=len(received),
        )
        output_bases = np.asarray(
            [
                [DNA_BASE_INDEX[base] for base in output]
                for output in outputs
            ],
            dtype=np.uint8,
        )
        symbols = _CYTHON_KERNELS.build_symbol_scores(
            received_ids,
            expected_length,
            drift_min,
            drift_max,
            log_pr["no_insert"],
            log_pr["insert_base"],
            log_pr["delete"],
            log_pr["match"],
            log_pr["substitute"],
            max_insertions_per_symbol,
        )
        branches = _CYTHON_KERNELS.build_branch_scores(
            np.ascontiguousarray(symbols),
            np.ascontiguousarray(output_bases),
            message_steps,
        )
        return branches, "cython_channel_numpy_trellis"

    symbols = _fallback_symbol_scores(
        received,
        expected_length,
        drift_min,
        drift_max,
        log_pr,
        max_insertions_per_symbol,
    )
    return (
        _fallback_branch_scores(symbols, outputs, message_steps),
        "numpy",
    )


def _hard_byte_posterior_risks(
    decoded,
    payload_steps,
    alpha,
    beta,
    branches,
    output_ids,
    next_state_ids,
    state_count,
    drift_count,
    uniform_pair_prior,
    log_evidence,
):
    """Return model-posterior risks for complete hard-decoded bytes."""
    decoded = "".join(decoded)
    alpha_grid = alpha.reshape(
        payload_steps + 2, state_count, drift_count
    )
    beta_grid = beta.reshape(
        payload_steps + 2, state_count, drift_count
    )
    posterior_risks = []
    for step in range(0, payload_steps - 1, 2):
        first_symbol = PAIR_TO_SYMBOL[decoded[2 * step : 2 * step + 2]]
        second_symbol = PAIR_TO_SYMBOL[
            decoded[2 * step + 2 : 2 * step + 4]
        ]
        left = np.full((state_count, drift_count), NEG_INF)
        for state in range(state_count):
            next_state = next_state_ids[state, first_symbol]
            first_branch = branches[
                step, output_ids[state, first_symbol]
            ]
            contribution = np.logaddexp.reduce(
                alpha_grid[step, state, :, None] + first_branch,
                axis=0,
            ) + uniform_pair_prior
            left[next_state] = np.logaddexp(
                left[next_state], contribution
            )

        log_joint = NEG_INF
        for state in range(state_count):
            next_state = next_state_ids[state, second_symbol]
            second_branch = branches[
                step + 1, output_ids[state, second_symbol]
            ]
            contribution = np.logaddexp.reduce(
                (
                    left[state, :, None]
                    + second_branch
                    + beta_grid[step + 2, next_state][None, :]
                    + uniform_pair_prior
                ).ravel()
            )
            log_joint = float(np.logaddexp(log_joint, contribution))
        hard_probability = math.exp(min(0.0, log_joint - log_evidence))
        posterior_risks.append(
            min(1.0, max(0.0, 1.0 - hard_probability))
        )
    return posterior_risks


def decode_hard(
    received_dna,
    payload_steps,
    channel,
    drift_window=3,
    max_insertions_per_symbol=1,
    return_diagnostics=False,
    return_byte_posterior_risks=False,
    trellis=None,
):
    """Return one per-base MAP sequence with no rejection or erasure mark."""
    if drift_window < 0 or max_insertions_per_symbol not in (0, 1):
        raise ValueError(
            "drift_window must be non-negative and insertions per symbol 0 or 1"
        )
    if payload_steps < 0:
        raise ValueError("payload_steps must be non-negative")
    if return_byte_posterior_risks and not return_diagnostics:
        raise ValueError(
            "byte posterior risks require return_diagnostics=True"
        )

    active_trellis = TRELLIS if trellis is None else trellis
    (
        state_count,
        input_count,
        initial_state,
        termination_symbol,
        next_state_ids,
        terminal_state_mask,
    ) = _prepare_trellis(active_trellis)
    output_table = active_trellis.output_table

    received = received_dna.upper()
    if any(base not in DNA_BASE_INDEX for base in received):
        diagnostics = {
            "complete": True,
            "algorithm": "log_map",
            "abort_reason": "invalid_received_base",
        }
        if return_byte_posterior_risks:
            diagnostics["byte_posterior_risks"] = []
        return (None, diagnostics) if return_diagnostics else None

    message_steps = payload_steps + 1
    expected_length = 4 * message_steps
    final_drift = len(received) - expected_length
    drift_min, drift_max = _drift_bounds(final_drift, drift_window)
    drift_count = drift_max - drift_min + 1
    start_drift_index = -drift_min
    terminal_drift_index = final_drift - drift_min
    if not 0 <= start_drift_index < drift_count:
        start_drift_index = None
    if not 0 <= terminal_drift_index < drift_count:
        terminal_drift_index = None

    outputs = tuple(
        sorted({output for row in output_table for output in row})
    )
    output_index = {output: index for index, output in enumerate(outputs)}
    output_ids = np.asarray(
        [
            [
                output_index[output_table[state][symbol]]
                for symbol in range(input_count)
            ]
            for state in range(state_count)
        ],
        dtype=np.int16,
    )
    branches, backend = _channel_branch_scores(
        received,
        message_steps,
        outputs,
        channel,
        drift_min,
        drift_max,
        max_insertions_per_symbol,
    )

    uniform_pair_prior = -math.log(float(input_count))
    pair_posteriors = None
    edge_layers = None
    if _CYTHON_FULL_LOG_MAP:
        alpha, beta, pair_posteriors, generated_transitions = (
            _CYTHON_KERNELS.run_log_map(
                np.ascontiguousarray(branches),
                np.ascontiguousarray(output_ids),
                next_state_ids,
                terminal_state_mask,
                payload_steps,
                initial_state,
                termination_symbol,
                start_drift_index if start_drift_index is not None else -1,
                (
                    terminal_drift_index
                    if terminal_drift_index is not None
                    else -1
                ),
                uniform_pair_prior,
                max_insertions_per_symbol,
            )
        )
        backend = "cython_full_log_map"
    else:
        dense_state_count = state_count * drift_count
        alpha = np.full(
            (message_steps + 1, dense_state_count),
            NEG_INF,
            dtype=np.float64,
        )
        if start_drift_index is not None:
            alpha[
                0, initial_state * drift_count + start_drift_index
            ] = 0.0
        edge_layers = []
        generated_transitions = 0
        for step in range(message_steps):
            if step < payload_steps:
                allowed_inputs = np.arange(input_count, dtype=np.int16)
                prior = uniform_pair_prior
            else:
                allowed_inputs = np.asarray(
                    [termination_symbol], dtype=np.int16
                )
                prior = 0.0
            reachable = np.flatnonzero(np.isfinite(alpha[step]))
            combinations = len(allowed_inputs) * drift_count
            if len(reachable):
                from_array = np.repeat(reachable, combinations)
                state_array = np.repeat(
                    reachable // drift_count, combinations
                )
                from_drift_array = np.repeat(
                    reachable % drift_count, combinations
                )
                input_array = np.tile(
                    np.repeat(allowed_inputs, drift_count), len(reachable)
                )
                to_drift_array = np.tile(
                    np.arange(drift_count, dtype=np.int32),
                    len(reachable) * len(allowed_inputs),
                )
                score_array = branches[
                    step,
                    output_ids[state_array, input_array],
                    from_drift_array,
                    to_drift_array,
                ] + prior
                finite = np.isfinite(score_array)
                from_array = from_array[finite].astype(
                    np.int32, copy=False
                )
                state_array = state_array[finite].astype(
                    np.int32, copy=False
                )
                input_array = input_array[finite].astype(
                    np.int16, copy=False
                )
                to_drift_array = to_drift_array[finite]
                score_array = score_array[finite]
                next_state_array = next_state_ids[
                    state_array, input_array
                ].astype(np.int32, copy=False)
                to_array = next_state_array * drift_count + to_drift_array
            else:
                from_array = np.empty(0, dtype=np.int32)
                to_array = np.empty(0, dtype=np.int32)
                input_array = np.empty(0, dtype=np.int16)
                score_array = np.empty(0, dtype=np.float64)
            edge_layers.append(
                (from_array, to_array, input_array, score_array)
            )
            generated_transitions += len(score_array)
            if len(score_array):
                np.logaddexp.at(
                    alpha[step + 1],
                    to_array,
                    alpha[step, from_array] + score_array,
                )

    terminal_states = np.flatnonzero(terminal_state_mask)
    terminal_dense_indices = (
        terminal_states * drift_count + terminal_drift_index
        if terminal_drift_index is not None
        else None
    )
    log_evidence = (
        float(np.logaddexp.reduce(alpha[-1, terminal_dense_indices]))
        if terminal_dense_indices is not None
        else NEG_INF
    )
    common = {
        "complete": True,
        "algorithm": "log_map",
        "implementation_backend": backend,
        "encoder_states": state_count,
        "input_symbols": input_count,
        "initial_state": initial_state,
        "termination_symbol": termination_symbol,
        "terminal_states": terminal_states.tolist(),
        "payload_steps": payload_steps,
        "message_steps": message_steps,
        "expected_length": expected_length,
        "final_drift": final_drift,
        "drift_min": drift_min,
        "drift_max": drift_max,
        "generated_transitions": generated_transitions,
        "alpha_state_counts": [
            int(np.count_nonzero(np.isfinite(layer))) for layer in alpha
        ],
    }
    if log_evidence == NEG_INF:
        diagnostics = dict(
            common,
            abort_reason="no_valid_terminated_path",
            log_evidence=None,
        )
        return (None, diagnostics) if return_diagnostics else None

    if pair_posteriors is None:
        beta = np.full_like(alpha, NEG_INF)
        beta[-1, terminal_dense_indices] = 0.0
        for step in range(message_steps - 1, -1, -1):
            from_array, to_array, _, score_array = edge_layers[step]
            if len(score_array):
                np.logaddexp.at(
                    beta[step],
                    from_array,
                    score_array + beta[step + 1, to_array],
                )

    decoded = []
    partition_errors = []
    for step in range(payload_steps):
        if pair_posteriors is not None:
            by_pair = pair_posteriors[step]
        else:
            from_array, to_array, input_array, score_array = edge_layers[step]
            by_pair = np.full(input_count, NEG_INF, dtype=np.float64)
            values = (
                alpha[step, from_array]
                + score_array
                + beta[step + 1, to_array]
            )
            np.logaddexp.at(by_pair, input_array, values)
        partition = float(np.logaddexp.reduce(by_pair))
        partition_errors.append(abs(partition - log_evidence))
        normalized_pairs = by_pair - partition

        for pair_position in range(2):
            by_base = np.full(4, NEG_INF, dtype=np.float64)
            for base_index, base in enumerate(DNA_ALPHABET):
                indices = [
                    symbol
                    for symbol, pair in enumerate(INPUT_PAIRS)
                    if pair[pair_position] == base
                ]
                by_base[base_index] = np.logaddexp.reduce(
                    normalized_pairs[indices]
                )
            selected_index = int(np.argmax(by_base))
            decoded.append(DNA_ALPHABET[selected_index])

    diagnostics = dict(
        common,
        log_evidence=float(log_evidence),
        beta_state_counts=[
            int(np.count_nonzero(np.isfinite(layer))) for layer in beta
        ],
        max_partition_error=max(partition_errors, default=0.0),
    )
    if return_byte_posterior_risks:
        diagnostics["byte_posterior_risks"] = _hard_byte_posterior_risks(
            decoded,
            payload_steps,
            alpha,
            beta,
            branches,
            output_ids,
            next_state_ids,
            state_count,
            drift_count,
            uniform_pair_prior,
            float(log_evidence),
        )
    bcjr_decoded_strand = "".join(decoded)
    return (
        (bcjr_decoded_strand, diagnostics)
        if return_diagnostics
        else bcjr_decoded_strand
    )


def _initialize_batch_decoder(
    payload_steps,
    channel,
    drift_window,
    max_insertions_per_symbol,
    return_diagnostics,
    return_byte_posterior_risks,
):
    global _BATCH_DECODE_CONFIG
    _BATCH_DECODE_CONFIG = (
        payload_steps,
        channel,
        drift_window,
        max_insertions_per_symbol,
        return_diagnostics,
        return_byte_posterior_risks,
    )


def _decode_batch_read(received_dna):
    if _BATCH_DECODE_CONFIG is None:
        raise RuntimeError("batch decoder worker was not initialized")
    (
        payload_steps,
        channel,
        drift_window,
        max_insertions_per_symbol,
        return_diagnostics,
        return_byte_posterior_risks,
    ) = _BATCH_DECODE_CONFIG
    started = time.perf_counter() if return_diagnostics else None
    result = decode_hard(
        received_dna,
        payload_steps,
        channel,
        drift_window=drift_window,
        max_insertions_per_symbol=max_insertions_per_symbol,
        return_diagnostics=return_diagnostics,
        return_byte_posterior_risks=return_byte_posterior_risks,
    )
    if return_diagnostics:
        result[1]["worker_decode_seconds"] = time.perf_counter() - started
    return result


def decode_hard_batch(
    received_sequences,
    payload_steps,
    channel,
    drift_window=3,
    max_insertions_per_symbol=1,
    workers=24,
    chunksize=None,
    return_diagnostics=False,
    return_byte_posterior_risks=False,
):
    """Decode independent reads in order, optionally across processes."""
    if return_byte_posterior_risks and not return_diagnostics:
        raise ValueError(
            "byte posterior risks require return_diagnostics=True"
        )
    received = list(received_sequences)
    if not received:
        return []
    workers = int(workers)
    if workers == 0:
        workers = min(os.cpu_count() or 1, len(received))
    if workers < 0:
        raise ValueError("workers must be non-negative")
    workers = min(workers, len(received))
    if chunksize is None:
        chunksize = max(1, len(received) // (8 * workers))
    chunksize = int(chunksize)
    if chunksize <= 0:
        raise ValueError("chunksize must be positive")

    if workers == 1:
        results = []
        for sequence in received:
            started = time.perf_counter() if return_diagnostics else None
            result = decode_hard(
                sequence,
                payload_steps,
                channel,
                drift_window=drift_window,
                max_insertions_per_symbol=max_insertions_per_symbol,
                return_diagnostics=return_diagnostics,
                return_byte_posterior_risks=return_byte_posterior_risks,
            )
            if return_diagnostics:
                result[1]["worker_decode_seconds"] = (
                    time.perf_counter() - started
                )
            results.append(result)
        return results

    with ProcessPoolExecutor(
        max_workers=workers,
        initializer=_initialize_batch_decoder,
        initargs=(
            payload_steps,
            dict(channel),
            drift_window,
            max_insertions_per_symbol,
            return_diagnostics,
            return_byte_posterior_risks,
        ),
    ) as executor:
        return list(
            executor.map(_decode_batch_read, received, chunksize=chunksize)
        )


__all__ = ["decode_hard", "decode_hard_batch"]
