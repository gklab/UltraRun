"""Joint lossless BF16 packets for paired projections."""
from dataclasses import dataclass
from functools import lru_cache

import mlx.core as mx

from .lossless_projection import DECODE
from .packed_projection import FRAGMENT_COORDS, SWIGLU_BF16

JOINT_HEADER = r"""
inline uint4 restore_four(ushort4 m, uint e, uint base) {
    return uint4(restore_pair(m[0], e & 255, base),
                 restore_pair(m[1], (e >> 8) & 255, base),
                 restore_pair(m[2], (e >> 16) & 255, base),
                 restore_pair(m[3], e >> 24, base));
}
"""


PAIRED_EXPONENT_ORIGIN = 64
PAIRED_INDEX_BITS = 20


PAIRED_PACKET = r"""
            uint index = payload_index * 32 + lane;
            uint4 gate_pairs, up_pairs;
            if (verbatim) {
                gate_pairs = fallback[2 * index];
                up_pairs = fallback[2 * index + 1];
            } else {
                uint4 digits4 = reinterpret_cast<device const uint4*>(digits)[index];
                uint2 exponents2 = reinterpret_cast<device const uint2*>(exponents)[index];
                gate_pairs = restore_four(as_type<ushort4>(digits4.xy), exponents2.x, first_base);
                up_pairs = restore_four(as_type<ushort4>(digits4.zw), exponents2.y, second_base);
            }
"""


LOAD_JOINT = r"""
            uint descriptor_index = 2 * (output_tile * packets + packet);
            uint2 description(descriptor[descriptor_index], descriptor[descriptor_index + 1]);
            uint payload_index = description.y;
            uint first_base = description.x & 255u;
            uint second_base = description.x >> 8;
            bool verbatim = first_base == 255u;
""" + PAIRED_PACKET


LOAD_COMPACT_JOINT = r"""
            uint desc = descriptor[output_tile * packets + packet];
            uint payload_index = desc & 0xfffffu;
            uint low = (desc >> 20) & 63u;
            uint first_base = low + 64u;
            uint second_base = (desc >> 26) + 64u;
            bool verbatim = low == 63u;
""" + PAIRED_PACKET


def encode_paired(raw, *, compact_descriptors=False):
    blocks, tiles, fragment = raw.shape
    if raw.dtype != mx.uint16 or blocks % 2 or tiles % 4 or fragment != 64:
        raise ValueError("Expected paired packed BF16 projections")
    grouped = raw.reshape(2, -1, 4, 32, 2).transpose(1, 0, 2, 3, 4)
    exponent = (grouped >> 7) & 255
    limits = exponent.max(axis=(2, 3, 4))
    minimum = exponent.min(axis=(2, 3, 4))
    bases = mx.maximum(limits.astype(mx.int32) - 15, 0).astype(mx.uint16)
    fits = mx.all(minimum >= bases, axis=1)
    n = fits.shape[0]
    good = int(mx.sum(fits).item())
    bad = n - good
    indexes = mx.zeros((n,), dtype=mx.uint32)
    if good:
        fit_indices = mx.sort(mx.argpartition((~fits).astype(mx.uint8), good - 1)[:good])
        selected = grouped[fit_indices].transpose(0, 3, 1, 2, 4)
        signs_and_fractions = ((selected & 127) | ((selected >> 8) & 128)).astype(mx.uint8)
        digits = mx.contiguous(signs_and_fractions.reshape(good, 32, 16)).view(mx.uint32)
        deltas = ((selected >> 7) & 255) - bases[fit_indices, None, :, None, None]
        shifts = (deltas[..., 0] | (deltas[..., 1] << 4)).astype(mx.uint8)
        exponents = mx.contiguous(shifts.reshape(good, 32, 8)).view(mx.uint32)
        indexes = indexes.at[fit_indices].add(mx.arange(good, dtype=mx.uint32))
    else:
        digits = mx.zeros((1, 32, 4), dtype=mx.uint32)
        exponents = mx.zeros((1, 32, 2), dtype=mx.uint32)
    if bad:
        escape_indices = mx.sort(mx.argpartition(fits.astype(mx.uint8), bad - 1)[:bad])
        raw_pairs = grouped[escape_indices].transpose(0, 3, 1, 2, 4).reshape(bad, 32, 16)
        escaped = mx.contiguous(raw_pairs).view(mx.uint32)
        indexes = indexes.at[escape_indices].add(mx.arange(bad, dtype=mx.uint32))
    else:
        escaped = mx.zeros((1, 32, 8), dtype=mx.uint32)
    # One descriptor holds both exponent bases and the shared packet index.
    tags = (bases[:, 0].astype(mx.uint32) | (bases[:, 1].astype(mx.uint32) << 8))
    tags = mx.where(fits, tags, mx.array(255, dtype=mx.uint32))
    descriptors = None
    if compact_descriptors and n <= 1 << PAIRED_INDEX_BITS:
        origin = PAIRED_EXPONENT_ORIGIN
        in_range = mx.all((bases >= origin) & (bases < origin + 63), axis=1)
        if mx.all(~fits | in_range).item():
            offsets = bases.astype(mx.uint32) - origin
            encoded_bases = (offsets[:, 0] << 20) | (offsets[:, 1] << 26)
            words = mx.where(fits, encoded_bases, mx.array(63 << 20, dtype=mx.uint32)) | indexes
            descriptors = words.reshape(blocks // 2, tiles // 4, 1)
    if descriptors is None:
        descriptors = mx.stack((tags, indexes), axis=1).reshape(blocks // 2, tiles // 4, 2)
    return LosslessPaired((digits, exponents, escaped, mx.contiguous(descriptors)))


@lru_cache
def unpack_joint_kernel(row_major, compact_descriptor=False, transposed=False, group_width=0):
    writes = r"""
        uint k_tiles = packets * 4;
        uint stride = k_tiles * 4;
        uint dst = (output_tile * 8 + row) * stride + packet * 16 + col / 2;
        uint up_offset = descriptor_shape[0] * 8 * stride;
        for (uint i = 0; i < 4; ++i) {
            raw[dst + i * 4] = gate_pairs[i];
            raw[up_offset + dst + i * 4] = up_pairs[i];
        }
    """ if row_major else r"""
        uint base = (output_tile * packets + packet) * 4;
        uint up_offset = descriptor_shape[0] * packets * 4;
        for (uint i = 0; i < 4; ++i) {
            raw[(base + i) * 32 + lane] = gate_pairs[i];
            raw[(base + up_offset + i) * 32 + lane] = up_pairs[i];
        }
    """
    if transposed:
        writes = r"""
        uint feature = output_tile * 8 + row;
        """
        if group_width:
            writes += f"""
        constexpr uint half_width = {group_width}u;
        uint group_start = feature / half_width * packets * 32;
        feature %= half_width;
            """
        else:
            writes += r"""
        uint half_width = descriptor_shape[0] * 8;
        uint group_start = 0;
            """
        writes += r"""
        uint width = half_width * 2;
        for (uint i = 0; i < 4; ++i) {
            uint position = packet * 32 + i * 8 + col;
            uint dst = (group_start + position) * width + feature;
            raw[dst] = ushort(gate_pairs[i]);
            raw[dst + width] = ushort(gate_pairs[i] >> 16);
            raw[dst + half_width] = ushort(up_pairs[i]);
            raw[dst + half_width + width] = ushort(up_pairs[i] >> 16);
        }
        """
    load = LOAD_COMPACT_JOINT if compact_descriptor else LOAD_JOINT
    suffix = "compact" if compact_descriptor else "base"
    layout = f"transpose_{group_width}" if transposed else str(int(row_major))
    return mx.fast.metal_kernel(
        name=f"local_unpack_bf16_joint_gate_{layout}_{suffix}",
        input_names=["digits", "exponents", "escaped", "descriptor"], output_names=["raw"],
        header=DECODE + JOINT_HEADER, source=r"""
        uint i = thread_position_in_grid.x;
        uint lane = i % 32;
        uint packets = descriptor_shape[1];
        uint output_tile = i / 32 / packets;
        uint packet = i / 32 % packets;
        if (output_tile >= descriptor_shape[0]) return;
        auto fallback = reinterpret_cast<device const uint4*>(escaped);
        """ + FRAGMENT_COORDS + load + writes,
    )


def unpack_joint(payload, *, row_major=False, transposed=False, groups=1):
    blocks, chunks, descriptor_words = payload[-1].shape
    if groups < 1 or (groups != 1 and (not transposed or blocks % groups)):
        raise ValueError("Grouped weights require evenly sized transposed projections")
    if transposed:
        shape = (chunks * 32, blocks * 16 // groups)
        if groups > 1:
            shape = (groups, *shape)
        dtype = mx.uint16
    elif row_major:
        shape, dtype = (blocks * 16, chunks * 16), mx.uint32
    else:
        shape, dtype = (blocks * 2, chunks * 4, 32), mx.uint32
    group_width = blocks * 8 // groups if groups > 1 else 0
    out, = unpack_joint_kernel(row_major, descriptor_words == 1, transposed, group_width)(
        inputs=payload, grid=(blocks * chunks * 32, 1, 1), threadgroup=(256, 1, 1),
        output_shapes=[shape], output_dtypes=[dtype],
    )
    return out.view(mx.bfloat16 if row_major or transposed else mx.uint16)


SWIGLU_OUTPUT = r"""
        uint token = k_split * 8 + col;
        uint sum_slot = k_split * 64 + lane * 2;
        float2 gate = float2(0.0f);
        float2 up = float2(0.0f);
        for (uint part = 0; part < SPLITS; ++part) {
            uint index = part * stride + sum_slot;
            gate += float2(partials[index], partials[index + 1]);
            index += SPLITS * stride;
            up += float2(partials[index], partials[index + 1]);
        }
        // These operand fragments already have the layout of the down projection's input.
        uint dst = (k_split * half_blocks + output_tile) * 64 + lane * 2;
        output[dst] = token < TOKENS ? swiglu_bf16(gate.x, up.x) : bfloat16_t(0);
        output[dst + 1] = token + 1 < TOKENS ? swiglu_bf16(gate.y, up.y) : bfloat16_t(0);
"""


PROJECT_SUM = r"""

        uint token = k_split * 8 + col;
        uint sum_slot = k_split * 64 + lane * 2;
        float2 first(0.0f), second(0.0f);
        for (uint group = 0; group < 8; ++group) {
            float2 group_first(0.0f), group_second(0.0f);
            for (uint part = group; part < SPLITS; part += 8) {
                uint at = part * stride + sum_slot;
                group_first += float2(partials[at], partials[at + 1]);
                at += SPLITS * stride;
                group_second += float2(partials[at], partials[at + 1]);
            }
            first += group_first;
            second += group_second;
        }
"""


PROJECT_OUTPUT = r"""
        uint width = half_blocks * 16;
        uint feature = output_tile * 8 + row;
        uint other = half_blocks * 8;
        uint index = token * width + feature;
        if (token < TOKENS) {
            output[index] = bfloat16_t(first[0]);
            output[index + other] = bfloat16_t(second[0]);
        }
        if (token + 1 < TOKENS) {
            output[index + width] = bfloat16_t(first[1]);
            output[index + width + other] = bfloat16_t(second[1]);
        }
"""


SEQUENCE_OUTPUT = r"""
        uint feature = output_tile * 8 + row;
        uint columns = history_shape[1];
        uint z_columns = half_blocks * 16 - columns;
        uint channels[2] = {feature, feature + half_blocks * 8};
        float2 values[2] = {first, second};
        #pragma unroll
        for (uint part = 0; part < 2; ++part) {
            uint channel = channels[part];
            if (channel < columns) {
                // Each feature has one lane that owns token zero.
                if (token == 0) {
                    #pragma unroll
                    for (uint tap = 0; tap < 3; ++tap) {
                        uint offset = tap * columns + channel;
                        sequence[offset] = history[offset];
                    }
                }
                uint offset = (token + 3) * columns + channel;
                if (token < TOKENS) sequence[offset] = bfloat16_t(values[part][0]);
                if (token + 1 < TOKENS) sequence[offset + columns] = bfloat16_t(values[part][1]);
            } else {
                uint offset = token * z_columns + channel - columns;
                if (token < TOKENS) gate[offset] = bfloat16_t(values[part][0]);
                if (token + 1 < TOKENS) gate[offset + z_columns] = bfloat16_t(values[part][1]);
            }
        }
"""


RETAINED_PROJECT = r"""
        if (!use_complement) {
            if (k_split == 0 && col == 0) {
                uint first = output_tile * 8 + row;
                uint second = first + half_blocks * 8;
                int a = row_index[first];
                int b = row_index[second];
                output[first] = a < 0 ? bfloat16_t(-INFINITY) : saved[a];
                output[second] = b < 0 ? bfloat16_t(-INFINITY) : saved[b];
            }
            return;
        }
"""


ARGMAX_OUTPUT = r"""
        float4 best(float(bfloat16_t(first.x)), float(bfloat16_t(first.y)),
                    float(bfloat16_t(second.x)), float(bfloat16_t(second.y)));
        uint4 rows(row);
        for (uint mask : {2u, 4u, 16u}) {
            for (uint i = 0; i < 4; ++i) {
                float other_value = simd_shuffle_xor(best[i], mask);
                uint other_row = simd_shuffle_xor(rows[i], mask);
                if (other_value > best[i] || (other_value == best[i] && other_row < rows[i])) {
                    best[i] = other_value;
                    rows[i] = other_row;
                }
            }
        }
        if (row == 0) {
            uint width = half_blocks * 2;
            for (uint i = 0; i < 4; ++i) {
                uint t = token + i % 2;
                uint tile = output_tile + (i / 2) * half_blocks;
                if (t < TOKENS) {
                    score[t * width + tile] = bfloat16_t(best[i]);
                    winner[t * width + tile] = tile * 8 + rows[i];
                }
            }
        }
"""


LOCAL_REDUCTION = r"""
        constexpr uint stride = TOKENS > 8 ? 128 : 64;
        threadgroup float partials[2 * SPLITS * stride];
        uint gate_slot = k_split * stride + lane * 2;
        uint up_slot = gate_slot + SPLITS * stride;
        partials[gate_slot] = accum.thread_elements()[0];
        partials[gate_slot + 1] = accum.thread_elements()[1];

        partials[up_slot] = up_accum.thread_elements()[0];
        partials[up_slot + 1] = up_accum.thread_elements()[1];
        if (TOKENS > 8) {
            partials[gate_slot + 64] = tail.thread_elements()[0];
            partials[gate_slot + 65] = tail.thread_elements()[1];
            partials[up_slot + 64] = up_tail.thread_elements()[0];
            partials[up_slot + 65] = up_tail.thread_elements()[1];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (k_split >= (TOKENS > 8 ? 2 : 1)) return;
"""


PARTIALS_OUTPUT = r"""
        uint n = half_blocks * 16;
        uint first = output_tile * 8 + row;
        uint spacing = half_blocks * 8;
        uint i = (k_split * TOKENS + col) * n + first;
        if (col < TOKENS) {
            output[i] = accum.thread_elements()[0];
            output[i + spacing] = up_accum.thread_elements()[0];
        }
        if (col + 1 < TOKENS) {
            output[i + n] = accum.thread_elements()[1];
            output[i + n + spacing] = up_accum.thread_elements()[1];
        }
        if (col + 8 < TOKENS) {
            output[i + 8 * n] = tail.thread_elements()[0];
            output[i + 8 * n + spacing] = up_tail.thread_elements()[0];
        }
        if (col + 9 < TOKENS) {
            output[i + 9 * n] = tail.thread_elements()[1];
            output[i + 9 * n + spacing] = up_tail.thread_elements()[1];
        }
"""


@lru_cache
def paired_kernel(mode, packed_exponents=False, compact_descriptor=False):
    partials = mode == "partials"
    retained = mode == "project_retained"
    sequence = mode == "project_sequence"
    tiles = r"""
        uint k_split = threadgroup_position_in_grid.y;
        uint output_tile = thread_position_in_grid.x / 32;
    """ if partials else r"""
        uint k_split = thread_index_in_threadgroup / 32;
        uint output_tile = threadgroup_position_in_grid.x;
    """
    bounds_check = "if (output_tile >= half_blocks) return;" if partials else ""

    epilogue = {
        "swiglu": LOCAL_REDUCTION + SWIGLU_OUTPUT,
        "project": LOCAL_REDUCTION + PROJECT_SUM + PROJECT_OUTPUT,
        "project_retained": LOCAL_REDUCTION + PROJECT_SUM + PROJECT_OUTPUT,
        "project_sequence": LOCAL_REDUCTION + PROJECT_SUM + SEQUENCE_OUTPUT,
        "argmax": LOCAL_REDUCTION + PROJECT_SUM + ARGMAX_OUTPUT,
        "partials": PARTIALS_OUTPUT,
    }[mode]
    input_names = ["x", "digits", "exponents", "escaped", "descriptor"]
    if retained:
        input_names.extend(("use_complement", "row_index", "saved"))
    if sequence:
        input_names.append("history")
    packet_load = LOAD_COMPACT_JOINT if compact_descriptor else LOAD_JOINT
    kernel_name = f"local_lossless_paired_{mode}"
    if compact_descriptor:
        kernel_name += "_compact_descriptor"
    if packed_exponents:
        packet_load = packet_load.replace("restore_four(", "restore_packed_exponents(")
        kernel_name += "_wide_exponents"
    return mx.fast.metal_kernel(
        name=kernel_name,
        input_names=input_names,
        output_names=["sequence", "gate"] if sequence else (["score", "winner"] if mode == "argmax" else ["output"]),
        header="#include <metal_simdgroup_matrix>\n" + DECODE + JOINT_HEADER + SWIGLU_BF16,
        source=r"""
        uint lane = thread_index_in_threadgroup % 32;
        """ + tiles + r"""
        uint packets = descriptor_shape[1];
        uint half_blocks = descriptor_shape[0];
        """ + bounds_check + r"""
        auto fallback = reinterpret_cast<device const uint4*>(escaped);
        auto inputs = reinterpret_cast<device const ushort2*>(x);
        """ + FRAGMENT_COORDS + (RETAINED_PROJECT if retained else "") + r"""
        metal::simdgroup_matrix<float, 8, 8> a, b, accum, tail, up_accum, up_tail;
        accum.thread_elements()[0] = accum.thread_elements()[1] = 0.0f;
        up_accum.thread_elements()[0] = up_accum.thread_elements()[1] = 0.0f;
        if (TOKENS > 8) {
            tail.thread_elements()[0] = tail.thread_elements()[1] = 0.0f;
            up_tail.thread_elements()[0] = up_tail.thread_elements()[1] = 0.0f;
        }

        for (uint packet = k_split * 16; packet < min((k_split + 1) * 16, packets); ++packet) {
            """ + packet_load + r"""
            #pragma unroll
            for (uint j = 0; j < 4; ++j) {
                uint k = packet * 4 + j;
                uint gp = gate_pairs[j], up = up_pairs[j];
                float2 wg = as_type<float2>(uint2(gp << 16, gp & 0xffff0000u));
                float2 wu = as_type<float2>(uint2(up << 16, up & 0xffff0000u));
                float2 v = as_type<float2>(uint2(inputs[k * 32 + lane]) << 16);
                a.thread_elements()[0] = wg[0];
                a.thread_elements()[1] = wg[1];
                b.thread_elements()[0] = v[0];
                b.thread_elements()[1] = v[1];
                simdgroup_multiply_accumulate(accum, a, b, accum);
                a.thread_elements()[0] = wu[0];
                a.thread_elements()[1] = wu[1];
                simdgroup_multiply_accumulate(up_accum, a, b, up_accum);
                if (TOKENS > 8) {
                    float2 next = as_type<float2>(uint2(inputs[(packets * 4 + k) * 32 + lane]) << 16);
                    b.thread_elements()[0] = next[0];
                    b.thread_elements()[1] = next[1];
                    simdgroup_multiply_accumulate(up_tail, a, b, up_tail);
                    a.thread_elements()[0] = wg[0];
                    a.thread_elements()[1] = wg[1];
                    simdgroup_multiply_accumulate(tail, a, b, tail);
                }
            }
        }
        """ + epilogue,
    )


@dataclass(frozen=True)
class LosslessPaired:
    payload: tuple[mx.array, mx.array, mx.array, mx.array]

    def dispatch(self, x, tokens, mode, shapes, dtypes, *extra):
        blocks, packets, descriptor_words = self.payload[-1].shape
        splits = (packets + 15) // 16
        threads = splits * 32
        packed_exponents = mode == "swiglu" and 8 < tokens <= 16
        return paired_kernel(mode, packed_exponents, descriptor_words == 1)(
            inputs=[x, *self.payload, *extra], template=[("TOKENS", tokens), ("SPLITS", splits)],
            grid=(threads * blocks, 1, 1), threadgroup=(threads, 1, 1),
            output_shapes=shapes, output_dtypes=dtypes,
        )

    def project(self, x, tokens, *, swiglu=False):
        blocks = self.payload[-1].shape[0]
        shape = (((tokens + 7) // 8) * blocks, 64) if swiglu else (tokens, blocks * 16)
        mode = "swiglu" if swiglu else "project"
        result, = self.dispatch(x, tokens, mode, [shape], [mx.bfloat16])
        return result

    def project_retained(self, x, use_complement, row_index, saved):
        """Fill a single row from the complement or from the retained logit slots."""
        blocks = self.payload[-1].shape[0]
        result, = self.dispatch(x, 1, "project_retained", [(1, blocks * 16)], [mx.bfloat16],
                                use_complement, row_index, saved)
        return result

    def project_sequence(self, x, tokens, history):
        """Write QKV alongside three convolution history rows, and return Z separately."""
        blocks = self.payload[-1].shape[0]
        columns = history.shape[1]
        return self.dispatch(x, tokens, "project_sequence",
                             [(tokens + 3, columns), (tokens, blocks * 16 - columns)],
                             [mx.bfloat16, mx.bfloat16], history)

    def partials(self, x, tokens):
        # Keep the same 512-column FP32 partials and reduction order as the down projection.
        blocks, packets, descriptor_words = self.payload[-1].shape
        splits = (packets + 15) // 16
        groups = 4
        threads = groups * 32
        result, = paired_kernel("partials", compact_descriptor=descriptor_words == 1)(
            inputs=[x, *self.payload], template=[("TOKENS", tokens)],
            grid=(((blocks + groups - 1) // groups) * threads, splits, 1), threadgroup=(threads, 1, 1),
            output_shapes=[(splits, tokens, blocks * 16)], output_dtypes=[mx.float32],
        )
        return result

    def argmax_tiles(self, x, tokens):
        width = self.payload[-1].shape[0] * 2
        return self.dispatch(
            x, tokens, "argmax", [(tokens, width)] * 2, [mx.bfloat16, mx.uint32],
        )

    def argmax(self, x, tokens):
        scores, indices = self.argmax_tiles(x, tokens)
        winner = mx.argmax(scores, axis=-1)
        return mx.take_along_axis(indices, winner[:, None], axis=-1)[:, 0]
