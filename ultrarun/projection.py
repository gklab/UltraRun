"""Short-block BF16 projections on Apple M2 Ultra."""
from functools import lru_cache

import mlx.core as mx


@lru_cache
def split_gemv_kernel():
    return mx.fast.metal_kernel(
        name="local_bf16_split_gemv",
        input_names=["x", "weight"],
        output_names=["partials"],
        source=r"""
        uint row = thread_position_in_grid.x / 32;
        uint lane = thread_position_in_grid.x % 32;
        uint tile = thread_position_in_grid.y;
        if (row >= weight_shape[0]) return;
        uint cols = weight_shape[1] / 4;
        auto weights = reinterpret_cast<device const ushort4*>(weight);
        auto tokens = reinterpret_cast<device const ushort4*>(x);
        float4 sums[TOKENS] = {};

        // Reuse each four-value BF16 load for every token in the verify block.
        // Widening the BF16 bit pattern to FP32 is exact.
        uint end = min((tile + 1) * TILE, cols);
        for (uint column = tile * TILE + lane; column < end; column += 32) {
            float4 w = as_type<float4>(uint4(weights[row * cols + column]) << 16);
            for (uint t = 0; t < TOKENS; ++t) {
                float4 v = as_type<float4>(uint4(tokens[t * cols + column]) << 16);
                sums[t] = fma(w, v, sums[t]);
            }
        }
        for (uint t = 0; t < TOKENS; ++t) {
            float total = simd_sum(dot(sums[t], float4(1)));
            if (lane == 0)
                partials[(tile * TOKENS + t) * weight_shape[0] + row] = total;
        }
        """,
    )


def split_gemv(x, weight):
    tokens, width = x.shape
    rows = weight.shape[0]
    tiles = (width + 511) // 512
    partials, = split_gemv_kernel()(
        inputs=[x, weight],
        template=[("TOKENS", tokens), ("TILE", 128)],
        grid=(((rows + 7) // 8) * 256, tiles, 1),
        threadgroup=(256, 1, 1),
        output_shapes=[(tiles, tokens, rows)],
        output_dtypes=[mx.float32],
    )
    return partials.sum(axis=0).astype(mx.bfloat16)


def tiled_argmax(x, weight, tile_rows):
    """Score the complete vocabulary using wide-first BF16 projections."""
    token_ids, scores = [], []
    for offset in range(0, weight.shape[0], tile_rows):
        tile_logits = weight[offset:offset + tile_rows] @ x.T
        column_winners = mx.argmax(tile_logits, axis=0)
        score = mx.take_along_axis(tile_logits, column_winners[None], axis=0)[0]
        token_ids.append(column_winners + offset)
        scores.append(score)
    winners = mx.argmax(mx.stack(scores), axis=0)
    return mx.take_along_axis(mx.stack(token_ids), winners[None], axis=0)[0]
