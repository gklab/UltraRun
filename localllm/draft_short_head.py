"""Score a small BF16 draft vocabulary before optionally scoring its complement."""
import math
from time import perf_counter

import mlx.core as mx

from .lossless_gate_up import encode_paired, unpack_joint
from .memory import GIB
from .model import project
from .packed_projection import pack_activations, pack_weight


CORE_ROWS = 8192
FULL_ROWS = 32768
MAX_PROMPT_ROWS = 128
MAX_RETAINED_ROWS = 384
REQUEST_RESERVE_BYTES = 128 * 2**20
MIN_PATH_MASS = math.log(0.35)
MIN_CONFIDENCE = math.log(0.9)


def prepare_short_head(model, budget):
    """Create two verified lossless partitions within the startup memory budget."""
    reserve = 2 * GIB
    if (model.draft_vocabulary_rows != FULL_ROWS or model.config["hidden_size"] != 5120
            or mx.get_active_memory() + budget.cache_limit + reserve > budget.limit):
        return None
    budget.check(force=True, required_bytes=reserve)
    start = perf_counter()
    partitions = []
    for begin, end in ((0, CORE_ROWS), (CORE_ROWS, FULL_ROWS)):
        rows = model.vocabulary_rows(mx.arange(begin, end, dtype=mx.int32))
        packed = pack_weight(rows)
        mx.eval(packed)
        coded = encode_paired(packed)
        if not mx.array_equal(unpack_joint(coded.payload), packed).item():
            raise ValueError("Lossless draft vocabulary partition failed BF16 validation.")
        partitions.append(coded)
        del rows, packed
        mx.clear_cache()
        budget.check(force=True)
    print(f"Draft vocabulary partitions prepared in {perf_counter() - start:.3f} s", flush=True)
    return tuple(partitions)


def retained_prompt_rows(model, ids):
    """Use the small head when the prompt has few distinct complement rows."""
    if getattr(model, "short_draft_head", None) is None:
        return None
    rows = tuple(dict.fromkeys(token for token in ids if CORE_ROWS <= token < FULL_ROWS))
    return rows if len(rows) <= MAX_PROMPT_ROWS else None


@mx.compile
def needs_complement(scores, threshold):
    scores = scores.astype(mx.float32)
    return mx.any(scores.max(axis=-1) - mx.logsumexp(scores, axis=-1) < threshold)


class RequestDraftHead:
    def __init__(self, model, prompt_rows):
        self.model = model
        self.core, self.complement = model.short_draft_head
        self.retained = set()
        self.rows = None
        self.row_index = mx.full((FULL_ROWS - CORE_ROWS,), -1, dtype=mx.int32)
        self.empty = mx.zeros((1, 1), dtype=mx.bfloat16)
        self.threshold = mx.array(MIN_CONFIDENCE, dtype=mx.float32)
        self.previous_extra = None
        self.previous_rows = None
        self.combined = None
        self.complement_flags = []
        self.retain(prompt_rows)

    def retain(self, tokens):
        count = len(self.retained)
        if count == MAX_RETAINED_ROWS:
            return
        added = list(dict.fromkeys(token for token in tokens
                                  if CORE_ROWS <= token < FULL_ROWS and token not in self.retained))
        added = added[:MAX_RETAINED_ROWS - count]
        if not added:
            return
        token_ids = mx.array(added, dtype=mx.int32)
        rows = self.model.vocabulary_rows(token_ids)
        self.rows = rows if self.rows is None else mx.concatenate((self.rows, rows), axis=0)
        positions = mx.arange(count, count + len(added), dtype=mx.int32)
        self.row_index = mx.put_along_axis(self.row_index, token_ids - CORE_ROWS, positions, axis=0)
        self.retained.update(added)

    def score(self, hidden, head):
        packed = pack_activations(hidden)
        primary = self.core.project(packed, 1)
        if self.previous_extra is not head.extra or self.previous_rows is not self.rows:
            if head.extra is None:
                self.combined = self.rows
            elif self.rows is None:
                self.combined = head.extra
            else:
                self.combined = mx.concatenate((head.extra, self.rows), axis=0)
            self.previous_extra = head.extra
            self.previous_rows = self.rows
        scores = project(hidden, self.combined) if self.combined is not None else None
        visible = primary if scores is None else mx.concatenate((primary, scores), axis=-1)
        fallback = needs_complement(visible, self.threshold)
        self.complement_flags.append(fallback)
        extra_count = head.extra.shape[0] if head.extra is not None else 0
        saved = self.empty if self.rows is None else scores[:, extra_count:]
        rest = self.complement.project_retained(packed, fallback, self.row_index, saved)
        return mx.concatenate((primary, rest, scores[:, :extra_count]) if extra_count else (primary, rest), axis=-1)

    def metrics(self):
        flags = self.complement_flags
        return {
            "draft_conditional_vocab_attempts": len(flags),
            "draft_conditional_vocab_complements": mx.sum(mx.stack(flags)).item() if flags else 0,
            "draft_conditional_vocab_retained": len(self.retained),
        }
