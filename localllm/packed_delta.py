"""Fused BF16 DeltaNet projection boundaries for short linear verification."""
from functools import lru_cache

import mlx.core as mx

from .packed_projection import fragment_order, pack_activations, packed_partials


@lru_cache
def convolution_input_kernel():
    return mx.fast.metal_kernel(
        name="local_delta_projected_conv",
        input_names=["partial", "history", "weight"],
        output_names=["qkv", "gate", "sequence"],
        source=r"""
        uint channel = thread_position_in_grid.x;
        uint width = partial_shape[2];
        if (channel >= width) return;
        uint channels = history_shape[1];
        uint tokens = partial_shape[1];
        uint stride = tokens * width;
        if (channel >= channels) {
            uint gate_width = width - channels;
            for (uint token = 0; token < tokens; ++token) {
                float sum = 0;
                for (uint split = 0; split < partial_shape[0]; ++split)
                    sum += partial[split * stride + token * width + channel];
                gate[token * gate_width + channel - channels] = T(sum);
            }
            return;
        }
        T s0 = history[channel];
        T s1 = history[channels + channel];
        T s2 = history[2 * channels + channel];
        sequence[channel] = s0;
        sequence[channels + channel] = s1;
        sequence[2 * channels + channel] = s2;
        float w0 = float(weight[channel * 4]);
        float w1 = float(weight[channel * 4 + 1]);
        float w2 = float(weight[channel * 4 + 2]);
        float w3 = float(weight[channel * 4 + 3]);
        for (uint token = 0; token < tokens; ++token) {
            float sum = 0;
            for (uint split = 0; split < partial_shape[0]; ++split)
                sum += partial[split * stride + token * width + channel];
            T current = T(sum);
            float dot = fma(float(s0), w0, 0.0f);
            dot = fma(float(s1), w1, dot);
            dot = fma(float(s2), w2, dot);
            dot = fma(float(current), w3, dot);
            T value = T(dot);
            T sigmoid = T(1.0f / (1.0f + exp(-float(value))));
            qkv[token * channels + channel] = T(float(value) * float(sigmoid));
            sequence[(token + 3) * channels + channel] = current;
            s0 = s1;
            s1 = s2;
            s2 = current;
        }
        """,
    )


def projected_convolution(x, projection, history, conv_weight):
    tokens, channels = x.shape[0], history.shape[1]
    partial = packed_partials(pack_activations(x), tokens, projection)
    width = partial.shape[-1]
    return convolution_input_kernel()(
        inputs=[partial, history, conv_weight], template=[("T", x.dtype)],
        grid=(width, 1, 1), threadgroup=(256, 1, 1),
        output_shapes=[(tokens, channels), (tokens, width - channels), (tokens + 3, channels)],
        output_dtypes=[x.dtype] * 3,
    )


@lru_cache
def gated_fragments_kernel():
    return mx.fast.metal_kernel(
        name="local_delta_norm_gate_fragments",
        input_names=["value", "gate", "norm", "inverse", "epsilon"],
        output_names=["packed"],
        source=r"""
        uint lane = thread_position_in_grid.x % 32;
        uint group = thread_position_in_grid.x / 32;
        uint heads = value_shape[1];
        uint token = group / heads;
        uint head = group % heads;
        uint width = value_shape[2];
        float inv_rms = 0;
        if (token < value_shape[0]) {
            float square = 0;
            for (uint channel = lane; channel < width; channel += 32) {
                float x = float(value[group * width + channel]);
                square = fma(x, x, square);
            }
            inv_rms = rsqrt(simd_sum(square) / width + epsilon[0]);
        }
        for (uint channel = lane; channel < width; channel += 32) {
            uint feature = head * width + channel;
            uint panel = (token / 8) * heads * width * 8;
            uint index = panel + (feature / 8) * 64 + inverse[(feature % 8) * 8 + token % 8];
            T output = T(0);
            if (token < value_shape[0]) {
                uint i = group * width + channel;
                T normalized = T(float(value[i]) * inv_rms * float(norm[channel]));
                float z = float(gate[i]);
                output = T(float(normalized) * (z / (1.0f + exp(-z))));
            }
            packed[index] = output;
        }
        """,
    )


def gated_projection(value, gate, norm, epsilon, projection):
    tokens, heads, width = value.shape
    panels = (tokens + 7) // 8
    packed, = gated_fragments_kernel()(
        inputs=[value, gate, norm, fragment_order()[1], mx.array([epsilon], dtype=mx.float32)],
        template=[("T", value.dtype)],
        grid=(panels * 8 * heads * 32, 1, 1), threadgroup=(256, 1, 1),
        output_shapes=[(panels * heads * width // 8, 64)], output_dtypes=[value.dtype],
    )
    return packed_partials(packed, tokens, projection).sum(axis=0).astype(value.dtype)
