"""FP32 recurrent state for the Gated DeltaNet text layers."""
from dataclasses import dataclass
from functools import lru_cache

import mlx.core as mx


@lru_cache
def unused_indices():
    return mx.zeros((1,), dtype=mx.int32)


@lru_cache
def conv_kernel():
    return mx.fast.metal_kernel(
        name="local_delta_causal_conv4",
        input_names=["sequence", "weights", "windows"],
        output_names=["output"],
        source=r"""
        uint i = thread_position_in_grid.x;
        uint width = sequence_shape[1];
        uint size = INDEXED ? windows_shape[0] : sequence_shape[0] - 3;
        if (i >= size * width) return;
        uint channel = i % width;
        float dot = 0;
        for (uint tap = 0; tap < 4; ++tap) {
            uint row = INDEXED ? windows[(i / width) * 4 + tap] : i / width + tap;
            dot = fma(float(sequence[row * width + channel]), float(weights[channel * 4 + tap]), dot);
        }
        T raw = T(dot);
        T gate = T(1.0f / (1.0f + exp(-float(raw))));
        output[i] = T(float(raw) * float(gate));
        """,
    )


def causal_conv4(sequence, weights, windows=None):
    size = sequence.shape[0] - 3 if windows is None else windows.shape[0]
    width = sequence.shape[1]
    output, = conv_kernel()(
        inputs=[sequence, weights, unused_indices() if windows is None else windows],
        template=[("T", sequence.dtype), ("INDEXED", int(windows is not None))],
        grid=(size * width, 1, 1), threadgroup=(256, 1, 1),
        output_shapes=[(size, width)], output_dtypes=[sequence.dtype],
    )
    return output


@lru_cache(maxsize=1)
def prefill_convolution_kernel():
    return mx.fast.metal_kernel(
        name="local_delta_prefill_convolution_qkv",
        input_names=["raw", "history", "weights", "silu"], output_names=["q", "k", "v"],
        compile_options={"math_mode": "safe"},
        source=r"""
        uint group = thread_position_in_grid.x / 32;
        uint lane = thread_index_in_simdgroup;
        uint token = group / 80;
        uint head = group % 80;
        uint ch = head * 128 + 4 * lane;
        float value[4], squares[4];
        for (uint c = 0; c < 4; ++c) {
            uint channel = ch + c;
            float dot = 0.0f;
            for (uint tap = 0; tap < 4; ++tap) {
                uint position = token + tap;
                float x = position < 3 ? float(history[position * 10240 + channel])
                                       : float(raw[(position - 3) * 16384 + channel]);
                dot = fma(x, float(weights[channel * 4 + tap]), dot);
            }
            bfloat16_t rounded = bfloat16_t(dot);
            value[c] = float(silu[as_type<ushort>(rounded)]);
            squares[c] = value[c] * value[c];
        }
        if (head < 32) {
            // Match the four contiguous per-lane adds in MLX's row reduction.
            float sum = ((squares[0] + squares[1]) + squares[2]) + squares[3];
            float inv = metal::precise::rsqrt(simd_sum(sum) + 1e-6f);
            uint dst = (token * 16 + head % 16) * 128 + 4 * lane;
            for (uint c = 0; c < 4; ++c) {
                float normalized = value[c] * inv;
                if (head < 16) q[dst+c] = normalized * 0.08838834764831845f;
                else k[dst+c] = normalized;
            }
        } else {
            uint dst = (token * 48 + head - 32) * 128 + 4 * lane;
            for (uint c = 0; c < 4; ++c) v[dst+c] = bfloat16_t(value[c]);
        }
        """,
    )


def prefill_convolution(projection, history, weight, silu):
    rows = projection.shape[0]
    return prefill_convolution_kernel()(
        inputs=[projection, history, weight, silu],
        output_shapes=[(rows, 16, 128), (rows, 16, 128), (rows, 48, 128)],
        output_dtypes=[mx.float32, mx.float32, mx.bfloat16],
        grid=(rows * 80 * 32, 1, 1), threadgroup=(256, 1, 1),
    )


@lru_cache(maxsize=1)
def prefill_norm_gate_kernel():
    return mx.fast.metal_kernel(
        name="local_delta_prefill_norm_gate",
        input_names=["value", "gate", "weight", "silu", "eps"], output_names=["out"],
        compile_options={"math_mode": "safe"},
        source=r"""
        uint group = thread_position_in_grid.x / 32;
        uint lane = thread_index_in_simdgroup;
        uint start = group * 128 + 4 * lane;
        float x[4];
        float dot = 0.0f;
        for (uint c = 0; c < 4; ++c) {
            x[c] = float(value[start+c]);
            dot = fma(x[c], x[c], dot);
        }
        float inv = metal::precise::rsqrt(simd_sum(dot) / 128.0f + eps[0]);
        for (uint c = 0; c < 4; ++c) {
            // Keep the BF16 rounding at the MLX RMSNorm output boundaries.
            bfloat16_t rounded = bfloat16_t(x[c] * inv);
            bfloat16_t normalized = bfloat16_t(float(rounded) * float(weight[lane * 4+c]));
            out[start+c] = bfloat16_t(float(normalized) * silu[as_type<ushort>(gate[start+c])]);
        }
        """,
    )


@lru_cache(maxsize=1)
def prefill_epsilon(epsilon):
    return mx.array([epsilon], dtype=mx.float32)


def prefill_norm_gate(value, gate, weight, epsilon, silu):
    return prefill_norm_gate_kernel()(
        inputs=[value, gate, weight, silu, prefill_epsilon(epsilon)],
        output_shapes=[value.shape], output_dtypes=[mx.bfloat16],
        grid=(value.size // 4, 1, 1), threadgroup=(256, 1, 1),
    )[0]


@lru_cache
def recurrent_kernel(record_updates=False, prepared_tree=False, checkpoint=False, scheduled_tree=False):
    outputs = ["output", "next_state"]
    record = ""
    if record_updates:
        outputs += ["updates", "normalized_keys"]
        record = r"""
            if (lane == 0) updates[out_index] = update;
            if (row % (DV * (HV / HK)) == 0) {
                for (uint i = 0; i < DK / 32; ++i)
                    normalized_keys[qk_base + i * 32] = k[i];
            }
        """
    forks = r"""
        uint spine = uint(route[0]);
        uint fork = uint(route[1]);
        uint second_fork = uint(route[2]);
    """ if prepared_tree else r"""
        constexpr uint spine = SPINE;
        uint fork = SPINE > 0 ? uint(parents[SPINE]) : 0;
        uint second_fork = fork;
        if (MULTI_FORK) {
            for (uint t = SPINE + 1; t < value_shape[0]; ++t) {
                if (parents[t] != int(t) - 1 && parents[t] != int(fork)) {
                    second_fork = uint(parents[t]);
                    break;
                }
            }
        }
    """
    restore = r"""
            int slot = route[t + 3];
            if (slot >= 0) {
                for (uint i = 0; i < DK / 32; ++i)
                    state[i] = MULTI_FORK && slot == 1 ? second_state[i] : fork_state[i];
            }
    """ if prepared_tree else r"""
            if (SPINE > 0 && t >= SPINE && (t == SPINE || parents[t] != int(t) - 1)) {
                for (uint i = 0; i < DK / 32; ++i)
                    state[i] = MULTI_FORK && uint(parents[t]) == second_fork ? second_state[i] : fork_state[i];
            } else if (TREE && SPINE == 0 && t != 0 && parents[t] != int(t) - 1) {
                uint base = (uint(parents[t]) * HV * DV + row) * DK;
                for (uint i = 0; i < DK / 32; ++i)
                    state[i] = next_state[base + i * 32 + lane];
            }
    """
    name = "local_delta_recurrence_updates" if record_updates else "local_delta_recurrence"
    if prepared_tree:
        name += "_prepared"
    if checkpoint:
        name += "_checkpoint"
    checkpoint_test = "t == uint(checkpoint[0])" if checkpoint else "t + 1 == spine"
    if scheduled_tree:
        name += "_reused_slots"
        forks = ""
        checkpoint_test = "t == uint(route[0])"
        restore = r"""
            uint command = uint(route[t + 1]);
            uint read_slot = command & 3u;
            if (read_slot) {
                for (uint i = 0; i < DK / 32; ++i)
                    state[i] = MULTI_FORK && read_slot == 2 ? second_state[i] : fork_state[i];
            }
        """
    save_checkpoint = r"""
            if (""" + checkpoint_test + r""") {
                for (uint i = 0; i < DK / 32; ++i)
                    next_state[row * DK + i * 32 + lane] = state[i];
            }
    """
    checkpoint_tail = save_checkpoint if checkpoint else ""
    save_spine = "" if checkpoint else save_checkpoint
    save_forks = r"""
            if (PREPARED || SPINE > 0) {
                if (t == fork) {
                    for (uint i = 0; i < DK / 32; ++i)
                        fork_state[i] = state[i];
                }
                if (MULTI_FORK && t == second_fork) {
                    for (uint i = 0; i < DK / 32; ++i)
                        second_state[i] = state[i];
                }
        """ + save_spine + r"""
            }
    """
    if scheduled_tree:
        save_forks = r"""
            uint write_slot = command >> 2u;
            if (write_slot == 1) {
                for (uint i = 0; i < DK / 32; ++i)
                    fork_state[i] = state[i];
            } else if (MULTI_FORK && write_slot == 2) {
                for (uint i = 0; i < DK / 32; ++i)
                    second_state[i] = state[i];
            }
        """ + save_checkpoint
    save_final = "" if checkpoint else r"""
        if (!TRACE && !PREPARED && SPINE == 0) {
            for (uint i = 0; i < DK / 32; ++i)
                next_state[row * DK + i * 32 + lane] = state[i];
        }
    """
    return mx.fast.metal_kernel(
        name=name,
        input_names=["query", "key", "value", "decay", "beta", "previous",
                     "route" if prepared_tree else "parents", *(["checkpoint"] if checkpoint else [])],
        output_names=outputs,
        source=r"""
        // One SIMD group owns one row (one value channel) of the FP32 state.
        uint lane = thread_position_in_grid.x % 32;
        uint row = thread_position_in_grid.x / 32;
        uint head = row / DV;
        uint query_head = head / (HV / HK);
        float state[DK / 32];
        float fork_state[DK / 32];
        float second_state[DK / 32];
        """ + forks + r"""
        for (uint i = 0; i < DK / 32; ++i)
            state[i] = previous[row * DK + i * 32 + lane];

        for (uint t = 0; t < value_shape[0]; ++t) {
        """ + restore + r"""
            uint qk_base = (t * HK + query_head) * DK + lane;
            uint out_index = t * HV * DV + row;
            float d = decay[t * HV + head];
            float k[DK / 32];
            float q[DK / 32];
            float k_scale = 1;
            float q_scale = 1;
            if (NORMALIZE) {
                float q_norm = 0;
                float k_norm = 0;
                for (uint i = 0; i < DK / 32; ++i) {
                    q[i] = float(query[qk_base + i * 32]);
                    k[i] = float(key[qk_base + i * 32]);
                    q_norm += q[i] * q[i];
                    k_norm += k[i] * k[i];
                }
                k_scale = rsqrt(simd_sum(k_norm) + 1e-6f);
                q_scale = rsqrt(simd_sum(q_norm) + 1e-6f) * rsqrt(float(DK));
            }
            float read = 0;
            for (uint i = 0; i < DK / 32; ++i) {
                k[i] = NORMALIZE ? k[i] * k_scale : float(key[qk_base + i * 32]);
                state[i] *= d;
                read += state[i] * k[i];
            }

            float update = (float(value[out_index]) - simd_sum(read)) * beta[t * HV + head];
        """ + record + r"""
            float result = 0;
            for (uint i = 0; i < DK / 32; ++i) {
                state[i] = fma(update, k[i], state[i]);
                result += state[i] * (NORMALIZE ? q[i] * q_scale : float(query[qk_base + i * 32]));
            }
            result = simd_sum(result);
            if (lane == 0)
                output[out_index] = T(result);
            if (TRACE) {
                uint base = (t * HV * DV + row) * DK;
                for (uint i = 0; i < DK / 32; ++i)
                    next_state[base + i * 32 + lane] = state[i];
            }
        """ + save_forks + checkpoint_tail + r"""
        }

        """ + save_final,
    )


def recurrent(q, k, v, decay, beta, state, group_rows, trace=False, normalize=False, parents=None,
              record_updates=False, spine=None, multi_fork=False, *, prepared_route=None, checkpoint=None,
              scheduled_tree=False):
    _, value_heads, value_width = v.shape
    _, key_heads, key_width = k.shape
    prepared = prepared_route is not None
    trace = not prepared and (trace or parents is not None) and spine is None
    if checkpoint is not None and (trace or not record_updates):
        raise ValueError("A DeltaNet checkpoint requires compact replay records")
    if scheduled_tree and (not prepared or not record_updates or checkpoint is not None):
        raise ValueError("Reusable DeltaNet slots require a prepared tree and compact replay records")
    state_shape = (v.shape[0], *state.shape) if trace else state.shape
    shapes = [v.shape, state_shape]
    dtypes = [v.dtype, mx.float32]
    if record_updates:
        shapes += [v.shape, k.shape]
        dtypes += [mx.float32, mx.float32]
    routing = prepared_route if prepared else unused_indices() if parents is None else parents
    inputs = [q, k, v, decay, beta, state, routing]
    if checkpoint is not None:
        inputs.append(checkpoint)
    return recurrent_kernel(record_updates, prepared, checkpoint is not None, scheduled_tree)(
        inputs=inputs,
        template=[
            ("T", v.dtype), ("HV", value_heads), ("HK", key_heads),
            ("DK", key_width), ("DV", value_width), ("TRACE", int(trace)), ("NORMALIZE", int(normalize)),
            ("TREE", int(parents is not None)),
            ("SPINE", spine or 0), ("PREPARED", int(prepared)),
            ("MULTI_FORK", int(multi_fork)),
        ],
        grid=(value_heads * value_width * 32, 1, 1),
        threadgroup=(group_rows * 32, 1, 1),
        output_shapes=shapes,
        output_dtypes=dtypes,
    )


@lru_cache
def prefill_recurrent_kernel():
    return mx.fast.metal_kernel(
        name="local_delta_prefill_four_rows",
        input_names=["query", "key", "value", "decay", "beta", "previous"],
        output_names=["output", "next_state"],
        source=r"""
        // Four adjacent value channels share the same query and key loads.
        uint lane = thread_position_in_grid.x % 32;
        uint row = thread_position_in_grid.x / 32 * 4;
        uint head = row / DV;
        uint query_head = head / (HV / HK);
        float4 state[DK / 32];
        for (uint i = 0; i < DK / 32; ++i)
            for (uint r = 0; r < 4; ++r)
                state[i][r] = previous[(row + r) * DK + i * 32 + lane];

        for (uint t = 0; t < value_shape[0]; ++t) {
            uint qk_base = (t * HK + query_head) * DK + lane;
            uint out_index = t * HV * DV + row;
            float d = decay[t * HV + head];
            float k[DK / 32];
            float4 read(0.0f);
            for (uint i = 0; i < DK / 32; ++i) {
                k[i] = float(key[qk_base + i * 32]);
                state[i] *= d;
                read += state[i] * k[i];
            }
            float4 input = float4(float(value[out_index]), float(value[out_index + 1]),
                                  float(value[out_index + 2]), float(value[out_index + 3]));
            float4 total(simd_sum(read.x), simd_sum(read.y), simd_sum(read.z), simd_sum(read.w));
            float4 update = (input - total) * beta[t * HV + head];
            float4 result(0.0f);
            for (uint i = 0; i < DK / 32; ++i) {
                state[i] = fma(update, float4(k[i]), state[i]);
                result += state[i] * float(query[qk_base + i * 32]);
            }
            float4 sum(simd_sum(result.x), simd_sum(result.y), simd_sum(result.z), simd_sum(result.w));
            if (lane == 0)
                for (uint r = 0; r < 4; ++r)
                    output[out_index + r] = T(sum[r]);
        }
        for (uint i = 0; i < DK / 32; ++i)
            for (uint r = 0; r < 4; ++r)
                next_state[(row + r) * DK + i * 32 + lane] = state[i][r];
        """,
    )


def prefill_recurrent(q, k, v, decay, beta, state, group_rows):
    _, hv, dv = v.shape
    _, hk, dk = k.shape
    return prefill_recurrent_kernel()(
        inputs=[q, k, v, decay, beta, state],
        template=[("T", v.dtype), ("HV", hv), ("HK", hk), ("DK", dk), ("DV", dv)],
        grid=(hv * dv * 32 // 4, 1, 1), threadgroup=(group_rows * 32, 1, 1),
        output_shapes=[v.shape, state.shape], output_dtypes=[v.dtype, mx.float32],
    )


@lru_cache
def restore_delta_kernel():
    return mx.fast.metal_kernel(
        name="local_delta_restore_prefix",
        input_names=["previous", "updates", "keys", "decay", "indices"],
        output_names=["state"],
        source=r"""
        uint i = thread_position_in_grid.x;
        uint hv = previous_shape[0];
        uint dv = previous_shape[1];
        uint dk = previous_shape[2];
        uint hk = keys_shape[1];
        if (i >= hv * dv * dk) return;
        uint row = i / dk;
        uint head = row / dv;
        uint key_head = head / (hv / hk);
        uint feature = i % dk;
        float value = previous[i];
        for (uint step = 0; step < KEEP; ++step) {
            uint t = INDEXED ? uint(indices[step]) : step;
            float scaled = value * decay[t * hv + head];
            float key = keys[(t * hk + key_head) * dk + feature];
            value = fma(updates[t * hv * dv + row], key, scaled);
        }
        state[i] = value;
        """,
    )


@dataclass
class DeltaRollback:
    previous: mx.array
    updates: mx.array
    keys: mx.array
    decay: mx.array

    def restore(self, keep, indices=None):
        state, = restore_delta_kernel()(
            inputs=[self.previous, self.updates, self.keys, self.decay,
                    unused_indices() if indices is None else indices],
            template=[("KEEP", keep), ("INDEXED", int(indices is not None))],
            grid=(self.previous.size, 1, 1), threadgroup=(256, 1, 1),
            output_shapes=[self.previous.shape], output_dtypes=[mx.float32],
        )
        return state


def reference(q, k, v, decay, beta, state):
    """Direct recurrence used by the kernel and cache verification command."""
    repeat = v.shape[1] // k.shape[1]
    q = mx.repeat(q, repeat, axis=1)
    k = mx.repeat(k, repeat, axis=1)
    outputs = []
    for pos in range(v.shape[0]):
        state = state * decay[pos, :, None, None]
        read = (state * k[pos, :, None, :]).sum(axis=-1)
        update = (v[pos].astype(mx.float32) - read) * beta[pos, :, None]
        state = state + update[..., None] * k[pos, :, None, :]
        outputs.append((state * q[pos, :, None, :]).sum(axis=-1))
    return mx.stack(outputs).astype(v.dtype), state
