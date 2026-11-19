"""Verify a bounded Metal top-16 selection against MLX's stable full sort."""

from functools import lru_cache
from statistics import median
from time import perf_counter

import mlx.core as mx


@lru_cache(maxsize=2)
def tile_top16_kernel(mapped=False):
    return mx.fast.metal_kernel(
        name="dflash2_tile_top16_mapped_bf16" if mapped else "dflash2_tile_top16_bf16",
        input_names=["words"],
        output_names=["indices", "values"] if mapped else ["indices"],
        source=r"""
        uint lane = thread_position_in_grid.x % 32;
        uint tile = thread_position_in_grid.x / 32;
        uint offset = tile * 256;
        uint row_offset = offset % words_shape[1];
        """ + (r"""
        uint half_vocab = words_shape[1] / 2;
        uint shift = half_vocab - DRAFT_HALF;
        uint storage = row_offset < DRAFT_HALF ? row_offset
            : row_offset < 2 * DRAFT_HALF ? row_offset + shift
            : row_offset < half_vocab + DRAFT_HALF ? row_offset - DRAFT_HALF : row_offset;
        uint load_offset = offset - row_offset + storage;
        """ if mapped else "uint load_offset = offset;") + r"""
        uint keys[8];
        #pragma unroll
        for (uint i = 0; i < 8; ++i) {
            uint local_id = i * 32 + lane;
            uint bits = words[load_offset + local_id];
            uint magnitude = bits & 0x7fffu;
            // Preserve stable numeric order for ties, signed zeros, and NaNs.
            uint order = (bits & 0x8000u) != 0 ? (bits ^ 0xffffu) : (bits ^ 0x8000u);
            if (magnitude == 0) order = 0x8000u;
            if (magnitude > 0x7f80u) order = 0xffffu;
            keys[i] = ((order + 1) << 8) | local_id;
        }
        for (int rank = 15; rank >= 0; --rank) {
            uint lane_best = 0;
            #pragma unroll
            for (uint i = 0; i < 8; ++i) lane_best = max(lane_best, keys[i]);
            uint winner = simd_max(lane_best);
            if (lane == 0) {
                indices[tile * 16 + rank] = row_offset + (winner & 255u);
                """ + ("values[tile * 16 + rank] = words[load_offset + (winner & 255u)];"
                       if mapped else "") + r"""
            }
            #pragma unroll
            for (uint i = 0; i < 8; ++i) keys[i] = (keys[i] == winner) ? 0u : keys[i];
        }
        """,
    )


def reference_top16(logits):
    return mx.argpartition(logits, -16, axis=-1)[..., -16:]


def bounded_top16(logits):
    """Keep only the top 16 per 256 consecutive IDs, then merge in ID order."""
    rows, vocab = logits.shape
    if logits.dtype != mx.bfloat16 or vocab % 256:
        raise ValueError("Expected 256-aligned BF16 logits")
    tiles = vocab // 256
    local, = tile_top16_kernel()(
        inputs=[mx.contiguous(logits).view(mx.uint16)], grid=(rows * tiles * 32, 1, 1),
        threadgroup=(128, 1, 1), output_shapes=[(rows, tiles * 16)], output_dtypes=[mx.uint32],
    )
    small = mx.take_along_axis(logits, local, axis=-1)
    winners = reference_top16(small)
    return mx.take_along_axis(local, winners, axis=-1)


def mapped_top16(logits, order):
    """Merge ranked BF16 logits in token order without copying the full vocabulary."""
    rows, vocab = logits.shape
    if (logits.dtype != mx.bfloat16 or vocab != order.size
            or vocab % 512 or order.draft_rows % 512):
        raise ValueError("Expected 256-aligned BF16 vocabulary segments")
    tiles = vocab // 256
    shape = (rows, tiles * 16)
    local, values = tile_top16_kernel(mapped=True)(
        inputs=[mx.contiguous(logits).view(mx.uint16)], template=[("DRAFT_HALF", order.draft_rows // 2)],
        grid=(rows * tiles * 32, 1, 1), threadgroup=(128, 1, 1),
        output_shapes=[shape, shape], output_dtypes=[mx.uint32, mx.uint16],
    )
    scores = values.view(mx.bfloat16)
    winners = reference_top16(scores)
    return mx.take_along_axis(local, winners, axis=-1), mx.take_along_axis(scores, winners, axis=-1)


def verify_top16(logits):
    expected = reference_top16(logits)
    actual = bounded_top16(logits)
    mx.eval(expected, actual)
    if not mx.array_equal(expected, actual).item():
        raise AssertionError("Metal top-16 changed candidate IDs or stable order")


def probe_top16(real_logits, samples=9):
    edge_values = mx.array([0.0, -0.0, float("inf"), float("-inf"), float("nan"), 1.0, -1.0, 0.5],
                           dtype=mx.bfloat16)
    verify_top16(mx.broadcast_to(edge_values[None, :, None], (1, 8, 64)).reshape(1, 512))
    timings = {}
    for rows, scores in sorted(real_logits.items()):
        verify_top16(scores)
        methods = {"full_sort": reference_top16, "tiled": bounded_top16}
        durations = {name: [] for name in methods}
        for i in range(samples):
            order = tuple(methods) if i % 2 == 0 else tuple(reversed(methods))
            for name in order:
                started = perf_counter()
                mx.eval(methods[name](scores))
                durations[name].append((perf_counter() - started) * 1000)
        timings[rows] = {name: median(times) for name, times in durations.items()}
    return timings
