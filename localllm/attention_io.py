"""Fused BF16 activation preprocessing and output packing for short GQA."""
from functools import lru_cache
from math import log2

import mlx.core as mx

from .packed_projection import FRAGMENT_COORDS


Q_HEADS = 24
K_HEADS = 4
WIDTH = 256
ROPE = 64


@lru_cache
def qk_kernel():
    return mx.fast.metal_kernel(
        name="local_qk_normalize_rotate",
        input_names=["qkv", "q_weight", "k_weight", "positions", "log_base", "eps"],
        output_names=["queries", "keys"],
        source=r"""
        uint t = threadgroup_position_in_grid.x;
        uint h = threadgroup_position_in_grid.y;
        uint tid = thread_index_in_threadgroup;
        bool query = h < QH;
        uint head = query ? h : h - QH;
        uint source_head = query ? 2 * head : 2 * QH + head;
        uint feature = tid * 4;
        uint base = t * qkv_shape[1] + source_head * 256 + feature;
        auto weights = query ? q_weight : k_weight;

        // Match the FP32 RMS reduction and its BF16 writeback before rotation.
        threadgroup float sums[2];
        float4 x;
        float sum = 0;
        for (uint i = 0; i < 4; ++i) {
            x[i] = float(qkv[base + i]);
            sum += x[i] * x[i];
        }
        sum = simd_sum(sum);
        if ((tid % 32) == 0) sums[tid / 32] = sum;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        float normalizer = metal::precise::rsqrt((sums[0] + sums[1]) / 256.0f + eps);

        float4 values;
        for (uint i = 0; i < 4; ++i)
            values[i] = float(bfloat16_t((x[i] * normalizer) * weights[feature + i]));

        uint heads = query ? QH : KH;
        uint row = TREE ? t * heads + head : head * qkv_shape[0] + t;
        auto output = query ? queries : keys;
        uint at = row * 256 + feature;
        int position = TREE ? positions[t] : positions[0] + int(t);
        for (uint i = 0; i < 4; ++i) {
            // The other half of the rotary pair is eight lanes away.
            float peer = simd_shuffle_down(values[i], 8);
            uint f = feature + i;
            if (f < 32) {
                float phase = float(position) * metal::exp2(-float(f) * (log_base / 32.0f));
                float cs = metal::fast::cos(phase);
                float sn = metal::fast::sin(phase);
                output[at + i] = bfloat16_t(values[i] * cs - peer * sn);
                output[at + i + 32] = bfloat16_t(values[i] * sn + peer * cs);
            } else if (f >= 64) {
                output[at + i] = bfloat16_t(values[i]);
            }
        }
        """,
    )


def normalize_rotate_qk(raw, q_weight, k_weight, offset, base, eps, *, tree):
    if not tree and not isinstance(offset, mx.array):
        offset = mx.array([offset], dtype=mx.int32)
    tokens = raw.shape[0]
    if tree:
        q_shape, k_shape = (tokens, Q_HEADS, WIDTH), (tokens, K_HEADS, WIDTH)
    else:
        q_shape, k_shape = (1, Q_HEADS, tokens, WIDTH), (1, K_HEADS, tokens, WIDTH)
    q, k = qk_kernel()(
        inputs=[raw, q_weight, k_weight, offset, log2(base), eps],
        grid=(64 * tokens, Q_HEADS + K_HEADS, 1), threadgroup=(64, 1, 1),
        template=[("QH", Q_HEADS), ("KH", K_HEADS), ("TREE", tree)],
        output_shapes=[q_shape, k_shape], output_dtypes=[mx.bfloat16, mx.bfloat16],
    )
    return (q.transpose(1, 0, 2)[None], k.transpose(1, 0, 2)[None]) if tree else (q, k)


@lru_cache
def gate_kernel():
    return mx.fast.metal_kernel(
        name="local_attention_gate_to_fragments",
        ensure_row_contiguous=False,
        input_names=["attention", "gate"],
        output_names=["packed"],
        source=r"""
        uint i = thread_position_in_grid.x;
        uint tokens = gate_shape[0];
        uint width = gate_shape[1] * gate_shape[2];
        uint span = width * 8;
        uint lane = i / 2 % 32;
        """ + FRAGMENT_COORDS + r"""
        uint token = (i / span) * 8 + col + i % 2;
        if (token >= tokens) {
            packed[i] = bfloat16_t(0);
            return;
        }
        uint feature = ((i % span) / 64) * 8 + row;
        uint head = feature / gate_shape[2];
        uint d = feature % gate_shape[2];
        uint a = head * attention_strides[1] + token * attention_strides[2] + d * attention_strides[3];
        uint g = token * gate_strides[0] + head * gate_strides[1] + d * gate_strides[2];
        bfloat16_t value = gate[g];
        auto low_half = 1 / (1 + metal::precise::exp(metal::abs(value)));
        bfloat16_t probability = value < 0 ? low_half : 1 - low_half;
        packed[i] = bfloat16_t(float(attention[a]) * float(probability));
        """,
    )


def pack_gated_attention(attention, gate):
    tokens, heads, width = gate.shape
    panels = (tokens + 7) // 8
    packed, = gate_kernel()(
        inputs=[attention, gate],
        grid=(panels * 8 * heads * width, 1, 1), threadgroup=(256, 1, 1),
        output_shapes=[(panels * heads * width // 8, 64)],
        output_dtypes=[mx.bfloat16],
    )
    return packed


@lru_cache
def suffix_update_kernel():
    return mx.fast.metal_kernel(
        name="local_joint_attention_suffix_update",
        input_names=["old_k", "old_v", "new_k", "new_v", "write_meta", "route"],
        output_names=["keys", "values"],
        ensure_row_contiguous=False,
        source=r"""
        uint i = thread_position_in_grid.x;
        uint capacity = old_k_shape[2];
        uint position = i / 32 % capacity;
        uint head = i / 32 / capacity;
        uint feature = (i % 32) * 8;
        uint begin = uint(write_meta[0]);
        uint replay = uint(write_meta[1]);
        uint end = begin + replay + uint(new_k_shape[2]);
        uint4 k, v;
        if (position >= begin + replay && position < end) {
            uint source = position - begin - replay;
            uint ki = head * new_k_strides[1] + source * new_k_strides[2] + feature;
            uint vi = head * new_v_strides[1] + source * new_v_strides[2] + feature;
            k = reinterpret_cast<device const uint4*>(new_k + ki)[0];
            v = reinterpret_cast<device const uint4*>(new_v + vi)[0];
        } else {
            uint source = position;
            if (position >= begin && position < begin + replay)
                source = uint(route[position - begin]);
            uint ki = head * old_k_strides[1] + source * old_k_strides[2] + feature;
            uint vi = head * old_v_strides[1] + source * old_v_strides[2] + feature;
            k = reinterpret_cast<device const uint4*>(old_k + ki)[0];
            v = reinterpret_cast<device const uint4*>(old_v + vi)[0];
        }
        reinterpret_cast<device uint4*>(keys)[i] = k;
        reinterpret_cast<device uint4*>(values)[i] = v;
        """,
    )


def update_attention_suffix(keys, values, k, v, write_meta, route):
    return suffix_update_kernel()(
        inputs=[keys, values, k, v, write_meta, route],
        grid=(keys.size // 8, 1, 1), threadgroup=(256, 1, 1),
        output_shapes=[keys.shape, values.shape], output_dtypes=[keys.dtype, values.dtype],
    )
