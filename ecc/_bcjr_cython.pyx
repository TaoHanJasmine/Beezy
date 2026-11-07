# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: initializedcheck=False
# cython: cdivision=True

"""Static Cython numerical kernels for the constrained BCJR decoder.

Validation, trellis construction, diagnostics, and the public API remain in
:mod:`Beezy.ecc.bcjr`.
"""

import numpy as np
cimport numpy as cnp

from libc.math cimport INFINITY, exp, fabs, log1p


cnp.import_array()

ctypedef cnp.float64_t float64_t
ctypedef cnp.int32_t int32_t
ctypedef cnp.int16_t int16_t
ctypedef cnp.int8_t int8_t
ctypedef cnp.uint8_t uint8_t


cdef inline double _logaddexp(double left, double right) noexcept nogil:
    """Stable two-term log-sum-exp without creating Python objects."""
    cdef double maximum
    if left == -INFINITY:
        return right
    if right == -INFINITY:
        return left
    maximum = left if left >= right else right
    # exp(-37) is below half an ulp at the score magnitudes used here, so the
    # correctly rounded sum is already ``maximum``.
    if fabs(left - right) > 37.0:
        return maximum
    return maximum + log1p(exp(-fabs(left - right)))


cpdef cnp.ndarray build_symbol_scores(
    cnp.ndarray[uint8_t, ndim=1, mode="c"] received_ids,
    int expected_length,
    int drift_min,
    int drift_max,
    double log_no_insert,
    double log_insert_base,
    double log_delete,
    double log_match,
    double log_substitute,
    int max_insertions_per_symbol,
):
    """Build exact single-symbol IDS likelihood matrices."""
    cdef int drift_count = drift_max - drift_min + 1
    cdef Py_ssize_t received_length = received_ids.shape[0]
    cdef cnp.ndarray[float64_t, ndim=4] scores_array = np.full(
        (expected_length, 4, drift_count, drift_count),
        -np.inf,
        dtype=np.float64,
    )
    cdef double[:, :, :, :] scores = scores_array
    cdef unsigned char[:] observations = received_ids
    cdef int encoded_position, drift_index, received_position
    cdef int insertion_count, aligned_position, transient_drift
    cdef int next_drift_index, base_index, observed_base
    cdef double insertion_score, event_score

    for encoded_position in range(expected_length):
        for drift_index in range(drift_count):
            received_position = encoded_position + drift_min + drift_index
            if received_position < 0 or received_position > received_length:
                continue

            for insertion_count in range(max_insertions_per_symbol + 1):
                if insertion_count == 0:
                    insertion_score = log_no_insert
                else:
                    insertion_score = log_insert_base
                if insertion_score == -INFINITY:
                    continue

                aligned_position = received_position + insertion_count
                if aligned_position > received_length:
                    continue
                transient_drift = aligned_position - encoded_position
                if transient_drift < drift_min or transient_drift > drift_max:
                    continue

                if log_delete != -INFINITY:
                    next_drift_index = transient_drift - 1 - drift_min
                    if 0 <= next_drift_index < drift_count:
                        event_score = insertion_score + log_delete
                        for base_index in range(4):
                            scores[
                                encoded_position,
                                base_index,
                                drift_index,
                                next_drift_index,
                            ] = _logaddexp(
                                scores[
                                    encoded_position,
                                    base_index,
                                    drift_index,
                                    next_drift_index,
                                ],
                                event_score,
                            )

                if aligned_position >= received_length:
                    continue
                next_drift_index = transient_drift - drift_min
                if next_drift_index < 0 or next_drift_index >= drift_count:
                    continue
                observed_base = observations[aligned_position]
                for base_index in range(4):
                    event_score = insertion_score + (
                        log_match
                        if base_index == observed_base
                        else log_substitute
                    )
                    scores[
                        encoded_position,
                        base_index,
                        drift_index,
                        next_drift_index,
                    ] = _logaddexp(
                        scores[
                            encoded_position,
                            base_index,
                            drift_index,
                            next_drift_index,
                        ],
                        event_score,
                    )

    return scores_array


cpdef cnp.ndarray build_branch_scores(
    cnp.ndarray[float64_t, ndim=4, mode="c"] symbol_scores_array,
    cnp.ndarray[uint8_t, ndim=2, mode="c"] expected_output_bases_array,
    int msg_length,
):
    """Compose 4-nt outputs using reusable 2-nt prefix/suffix matrices."""
    cdef int expected_length = symbol_scores_array.shape[0]
    cdef int drift_count = symbol_scores_array.shape[2]
    cdef int output_count = expected_output_bases_array.shape[0]
    cdef int encoded_width = expected_output_bases_array.shape[1]
    if encoded_width <= 0:
        raise ValueError("encoded output width must be positive")
    if encoded_width != 4:
        raise ValueError("the encoded output width must be four")
    if expected_length < msg_length * encoded_width:
        raise ValueError("symbol score table is shorter than the trellis")

    cdef cnp.ndarray[float64_t, ndim=4] branches_array = np.full(
        (msg_length, output_count, drift_count, drift_count),
        -np.inf,
        dtype=np.float64,
    )
    cdef cnp.ndarray[float64_t, ndim=3] prefix_array = np.full(
        (16, drift_count, drift_count), -np.inf, dtype=np.float64
    )
    cdef cnp.ndarray[float64_t, ndim=3] suffix_array = np.full(
        (16, drift_count, drift_count), -np.inf, dtype=np.float64
    )
    cdef cnp.ndarray[uint8_t, ndim=1] suffix_used_array = np.zeros(
        16, dtype=np.uint8
    )
    cdef double[:, :, :, :] symbol_scores = symbol_scores_array
    cdef unsigned char[:, :] expected_output_bases = expected_output_bases_array
    cdef double[:, :, :, :] branches = branches_array
    cdef double[:, :, :] prefix_scores = prefix_array
    cdef double[:, :, :] suffix_scores = suffix_array
    cdef unsigned char[:] suffix_used = suffix_used_array
    cdef int step, output_index, encoded_start, pair_index
    cdef int first_base, second_base, prefix_index, suffix_index
    cdef int from_drift, middle_drift, to_drift
    cdef double accumulator, left, right

    for output_index in range(output_count):
        suffix_index = (
            expected_output_bases[output_index, 2] * 4
            + expected_output_bases[output_index, 3]
        )
        suffix_used[suffix_index] = 1

    for step in range(msg_length):
        encoded_start = step * encoded_width

        # All 16 prefixes occur in the codebook.  Each one is shared by six
        # complete outputs, so compose it only once per step.
        for pair_index in range(16):
            first_base = pair_index // 4
            second_base = pair_index % 4
            for from_drift in range(drift_count):
                for to_drift in range(drift_count):
                    accumulator = -INFINITY
                    for middle_drift in range(drift_count):
                        left = symbol_scores[
                            encoded_start,
                            first_base,
                            from_drift,
                            middle_drift,
                        ]
                        right = symbol_scores[
                            encoded_start + 1,
                            second_base,
                            middle_drift,
                            to_drift,
                        ]
                        if left == -INFINITY or right == -INFINITY:
                            continue
                        accumulator = _logaddexp(accumulator, left + right)
                    prefix_scores[pair_index, from_drift, to_drift] = (
                        accumulator
                    )

        # Only eight of the 16 possible suffix dimers occur in the codebook.
        for pair_index in range(16):
            if suffix_used[pair_index] == 0:
                continue
            first_base = pair_index // 4
            second_base = pair_index % 4
            for from_drift in range(drift_count):
                for to_drift in range(drift_count):
                    accumulator = -INFINITY
                    for middle_drift in range(drift_count):
                        left = symbol_scores[
                            encoded_start + 2,
                            first_base,
                            from_drift,
                            middle_drift,
                        ]
                        right = symbol_scores[
                            encoded_start + 3,
                            second_base,
                            middle_drift,
                            to_drift,
                        ]
                        if left == -INFINITY or right == -INFINITY:
                            continue
                        accumulator = _logaddexp(accumulator, left + right)
                    suffix_scores[pair_index, from_drift, to_drift] = (
                        accumulator
                    )

        for output_index in range(output_count):
            prefix_index = (
                expected_output_bases[output_index, 0] * 4
                + expected_output_bases[output_index, 1]
            )
            suffix_index = (
                expected_output_bases[output_index, 2] * 4
                + expected_output_bases[output_index, 3]
            )
            for from_drift in range(drift_count):
                for to_drift in range(drift_count):
                    accumulator = -INFINITY
                    for middle_drift in range(drift_count):
                        left = prefix_scores[
                            prefix_index, from_drift, middle_drift
                        ]
                        right = suffix_scores[
                            suffix_index, middle_drift, to_drift
                        ]
                        if left == -INFINITY or right == -INFINITY:
                            continue
                        accumulator = _logaddexp(accumulator, left + right)
                    branches[
                        step, output_index, from_drift, to_drift
                    ] = accumulator

    return branches_array


cpdef tuple run_log_map(
    cnp.ndarray[float64_t, ndim=4, mode="c"] branch_scores_array,
    cnp.ndarray[int16_t, ndim=2, mode="c"] output_ids_array,
    cnp.ndarray[int16_t, ndim=2, mode="c"] next_state_ids_array,
    cnp.ndarray[uint8_t, ndim=1, mode="c"] terminal_state_mask_array,
    int payload_steps,
    int initial_state,
    int termination_symbol,
    int start_drift_index,
    int terminal_drift_index,
    double uniform_input_log_probability,
    int max_insertions_per_symbol,
):
    """Run exact forward, backward, and pair-posterior recursions."""
    cdef int message_steps = branch_scores_array.shape[0]
    cdef int drift_count = branch_scores_array.shape[2]
    cdef int state_count = output_ids_array.shape[0]
    cdef int symbol_count = output_ids_array.shape[1]
    cdef int dense_state_count = state_count * drift_count
    if state_count <= 0 or symbol_count <= 0:
        raise ValueError("the trellis must contain states and input symbols")
    if (
        next_state_ids_array.shape[0] != state_count
        or next_state_ids_array.shape[1] != symbol_count
    ):
        raise ValueError("next-state table shape does not match the trellis")
    if terminal_state_mask_array.shape[0] != state_count:
        raise ValueError("terminal-state mask does not match the trellis")
    if not 0 <= initial_state < state_count:
        raise ValueError("initial state lies outside the trellis")
    if not 0 <= termination_symbol < symbol_count:
        raise ValueError("termination symbol lies outside the trellis")
    if payload_steps < 0 or message_steps != payload_steps + 1:
        raise ValueError("message must contain payload steps plus one tail")
    if max_insertions_per_symbol < 0:
        raise ValueError("max_insertions_per_symbol must be non-negative")

    cdef cnp.ndarray[float64_t, ndim=2] alpha_array = np.full(
        (message_steps + 1, dense_state_count), -np.inf, dtype=np.float64
    )
    cdef cnp.ndarray[float64_t, ndim=2] beta_array = np.full(
        (message_steps + 1, dense_state_count), -np.inf, dtype=np.float64
    )
    cdef cnp.ndarray[float64_t, ndim=2] posterior_array = np.full(
        (payload_steps, symbol_count), -np.inf, dtype=np.float64
    )
    cdef double[:, :, :, :] branch_scores = branch_scores_array
    cdef short[:, :] output_ids = output_ids_array
    cdef short[:, :] next_state_ids = next_state_ids_array
    cdef unsigned char[:] terminal_state_mask = terminal_state_mask_array
    cdef double[:, :] alpha = alpha_array
    cdef double[:, :] beta = beta_array
    cdef double[:, :] posterior = posterior_array
    cdef int step, state, symbol, symbol_start, symbol_limit, output_index
    cdef int from_drift, to_drift, from_dense, to_dense, to_state
    cdef int to_drift_start, to_drift_stop
    cdef double prior, alpha_value, beta_value, branch_value, path_value
    cdef long long generated_transitions = 0

    if 0 <= start_drift_index < drift_count:
        alpha[
            0, initial_state * drift_count + start_drift_index
        ] = 0.0

    with nogil:
        for step in range(message_steps):
            if step < payload_steps:
                symbol_start = 0
                symbol_limit = symbol_count
                prior = uniform_input_log_probability
            else:
                symbol_start = termination_symbol
                symbol_limit = 1
                prior = 0.0
            for state in range(state_count):
                for from_drift in range(drift_count):
                    from_dense = state * drift_count + from_drift
                    alpha_value = alpha[step, from_dense]
                    if alpha_value == -INFINITY:
                        continue
                    # A branch emits four encoded bases.  A deletion lowers
                    # drift by at most one per base, while insertions raise it
                    # by at most the configured per-base limit.
                    to_drift_start = from_drift - 4
                    if to_drift_start < 0:
                        to_drift_start = 0
                    to_drift_stop = (
                        from_drift + 4 * max_insertions_per_symbol + 1
                    )
                    if to_drift_stop > drift_count:
                        to_drift_stop = drift_count
                    for symbol in range(
                        symbol_start, symbol_start + symbol_limit
                    ):
                        output_index = output_ids[state, symbol]
                        to_state = next_state_ids[state, symbol]
                        for to_drift in range(to_drift_start, to_drift_stop):
                            branch_value = branch_scores[
                                step, output_index, from_drift, to_drift
                            ]
                            if branch_value == -INFINITY:
                                continue
                            to_dense = to_state * drift_count + to_drift
                            path_value = alpha_value + branch_value + prior
                            alpha[step + 1, to_dense] = _logaddexp(
                                alpha[step + 1, to_dense], path_value
                            )
                            generated_transitions += 1

        if 0 <= terminal_drift_index < drift_count:
            for state in range(state_count):
                if terminal_state_mask[state] != 0:
                    beta[
                        message_steps,
                        state * drift_count + terminal_drift_index,
                    ] = 0.0

        # Posterior accumulation uses the same suffix term as beta.  Fusing
        # both operations removes a third full traversal of every transition.
        for step in range(message_steps - 1, -1, -1):
            if step < payload_steps:
                symbol_start = 0
                symbol_limit = symbol_count
                prior = uniform_input_log_probability
            else:
                symbol_start = termination_symbol
                symbol_limit = 1
                prior = 0.0
            for state in range(state_count):
                for from_drift in range(drift_count):
                    from_dense = state * drift_count + from_drift
                    if alpha[step, from_dense] == -INFINITY:
                        continue
                    to_drift_start = from_drift - 4
                    if to_drift_start < 0:
                        to_drift_start = 0
                    to_drift_stop = (
                        from_drift + 4 * max_insertions_per_symbol + 1
                    )
                    if to_drift_stop > drift_count:
                        to_drift_stop = drift_count
                    for symbol in range(
                        symbol_start, symbol_start + symbol_limit
                    ):
                        output_index = output_ids[state, symbol]
                        to_state = next_state_ids[state, symbol]
                        for to_drift in range(to_drift_start, to_drift_stop):
                            branch_value = branch_scores[
                                step, output_index, from_drift, to_drift
                            ]
                            if branch_value == -INFINITY:
                                continue
                            to_dense = to_state * drift_count + to_drift
                            beta_value = beta[step + 1, to_dense]
                            if beta_value == -INFINITY:
                                continue
                            path_value = branch_value + prior + beta_value
                            beta[step, from_dense] = _logaddexp(
                                beta[step, from_dense], path_value
                            )
                            if step < payload_steps:
                                path_value = alpha[step, from_dense] + path_value
                                posterior[step, symbol] = _logaddexp(
                                    posterior[step, symbol], path_value
                                )

    return alpha_array, beta_array, posterior_array, generated_transitions
