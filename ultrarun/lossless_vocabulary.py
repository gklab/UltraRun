"""Staged BF16 vocabulary encoding and direct row gather."""
from dataclasses import dataclass
from functools import lru_cache

import mlx.core as mx

from .lossless_gate_up import LosslessPaired, encode_paired, unpack_joint
from .lossless_projection import DECODE
from .packed_projection import fragment_order, pack_weight


@dataclass(frozen=True)
class VocabularyOrder:
    size: int
    draft_rows: int

    def to_storage(self, ids):
        prefix, half = self.draft_rows // 2, self.size // 2
        return mx.where(ids < prefix, ids, mx.where(
            ids < 2 * prefix, ids + half - prefix,
            mx.where(ids < half + prefix, ids - prefix, ids),
        ))

    def to_token_ids(self, slots):
        prefix, half = self.draft_rows // 2, self.size // 2
        return mx.where(slots < prefix, slots, mx.where(
            slots < half, slots + prefix,
            mx.where(slots < half + prefix, slots - half + prefix, slots),
        ))

    def token_order(self, values, rows_per_item=1):
        prefix = self.draft_rows // (2 * rows_per_item)
        half = self.size // (2 * rows_per_item)
        draft_first, rest_first, draft_second, rest_second = mx.split(
            values, (prefix, half, half + prefix), axis=-1,
        )
        return mx.concatenate((draft_first, draft_second, rest_first, rest_second), axis=-1)

    def draft_view(self, coded):
        # Pair the two halves of the fixed draft range at the beginning of the descriptor.
        return LosslessPaired((*coded.payload[:3], coded.payload[3][:self.draft_rows // 16]))

    def argmax(self, coded, x, tokens):
        return self.argmax_from_tiles(*coded.argmax_tiles(x, tokens))

    def argmax_from_tiles(self, scores, indices):
        # Resolve equal scores in token ID order before using the physical tile index.
        best = mx.argmax(self.token_order(scores, rows_per_item=8), axis=-1)
        tile = self.to_storage(best * 8) // 8
        slot = mx.take_along_axis(indices, tile[:, None], axis=-1)[:, 0]
        return self.to_token_ids(slot)


def encode_vocabulary(weight, budget, order=None):
    rows, width = weight.shape
    if weight.dtype != mx.bfloat16 or rows % 16 or width % 32:
        raise ValueError("Expected a BF16 vocabulary aligned to 16 rows and 32 columns")
    half = rows // 2
    chunks = []
    compressed_offset = escaped_offset = 0
    # Encode at most 62,080 rows per pair to bound the temporary working set.
    rows_per_half = 31040
    for start in range(0, half, rows_per_half):
        budget.check(force=True)
        end = min(start + rows_per_half, half)
        if order is None:
            pair = mx.concatenate((weight[start:end], weight[start + half:end + half]), axis=0)
        else:
            slots = mx.arange(start, end, dtype=mx.int32)
            rows_to_read = order.to_token_ids(mx.concatenate((slots, slots + half)))
            pair = weight[rows_to_read]
        packed = pack_weight(pair)
        mx.eval(packed)
        coded = encode_paired(packed)
        if not mx.array_equal(unpack_joint(coded.payload), packed).item():
            raise ValueError("Lossless packet validation failed for vocabulary projection")
        digits, exponent, escaped, descriptor = coded.payload
        tags = descriptor[..., 0]
        delta = mx.where((tags & 255) == 255, escaped_offset, compressed_offset)
        indices = descriptor[..., 1] + delta.astype(mx.uint32)
        chunk = (digits, exponent, escaped, mx.stack((tags, indices), axis=-1))
        mx.eval(*chunk)
        chunks.append(chunk)
        compressed_offset += digits.shape[0]
        escaped_offset += escaped.shape[0]
        del pair, packed, coded, chunk, digits, exponent, escaped, descriptor, tags, delta, indices
        mx.clear_cache()
        budget.check(force=True)
    full = tuple(mx.concatenate(items, axis=0) for items in zip(*chunks))
    mx.eval(*full)
    del chunks
    mx.clear_cache()
    budget.check(force=True)
    return LosslessPaired(full)


@lru_cache
def gather_kernel(rotate=False):
    names = ["digits", "exponents", "escaped", "descriptor", "rows", "inverse"]
    read_row = "uint row = uint(rows[item]);"
    if rotate:
        names.append("prior")
        read_row = r"""
        int route = rows[item];
        if (route >= 0) {
            raw[i] = prior[uint(route) * pairs_per_row + i % pairs_per_row];
            return;
        }
        uint row = uint(-1 - route);
        """
    return mx.fast.metal_kernel(
        name="local_lossless_paired_vocab_rotation" if rotate else "local_lossless_paired_vocab_rows",
        input_names=names,
        output_names=["raw"], header=DECODE,
        source=r"""
        uint i = thread_position_in_grid.x;
        uint packets = descriptor_shape[1];
        uint pairs_per_row = packets * 16;
        uint item = i / pairs_per_row;
        if (item >= rows_shape[0]) return;
        """ + read_row + r"""
        uint half_blocks = descriptor_shape[0];
        uint operand = row / (half_blocks * 8);
        uint tile = row / 8 - operand * half_blocks;
        uint pair = i % pairs_per_row;
        uint piece = pair / 4 % 4;
        uint lane = inverse[(row % 8) * 8 + (pair % 4) * 2] / 2;
        uint meta = (tile * packets + pair / 16) * 2;
        uint tags = descriptor[meta];
        uint loc = (descriptor[meta + 1] * 32 + lane) * 8 + operand * 4 + piece;
        if ((tags & 255) == 255) {
            raw[i] = escaped[loc];
        } else {
            auto mantissa = reinterpret_cast<device const ushort*>(digits);
            auto exponent = reinterpret_cast<device const uchar*>(exponents);
            raw[i] = restore_pair(mantissa[loc], exponent[loc], (tags >> (operand * 8)) & 255);
        }
        """,
    )


def vocabulary_rows(coded, rows, *, previous=None):
    count = rows.shape[0]
    pairs = coded.payload[-1].shape[1] * 16
    if count == 0:
        return mx.zeros((0, pairs * 2), dtype=mx.bfloat16)
    inputs = [*coded.payload, rows, fragment_order()[1]]
    if previous is not None:
        inputs.append(previous.view(mx.uint32))
    output, = gather_kernel(previous is not None)(
        inputs=inputs,
        grid=(count * pairs, 1, 1), threadgroup=(256, 1, 1),
        output_shapes=[(count, pairs)], output_dtypes=[mx.uint32],
    )
    return output.view(mx.bfloat16)


def rotate_vocabulary_rows(coded, routes, previous):
    """Gather old row indices and complemented storage slots into their new order."""
    if previous is None:
        previous = mx.zeros((1, 4), dtype=mx.bfloat16)
    return vocabulary_rows(coded, routes, previous=previous)
