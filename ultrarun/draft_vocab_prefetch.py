"""Reuse target vocabulary tile winners to anticipate request-local draft rows."""
from time import perf_counter

import mlx.core as mx

from .lossless_vocabulary import rotate_vocabulary_rows
from .memory import MemoryBudgetError
from .model import DraftVocabulary


PREFETCH_RESERVE_BYTES = 256 * 2**20
MAX_PREFETCH_ROWS = 160
ROWS_PER_ROUND = 12
MAX_IDLE_ROUNDS = 12
TOP_TILES = 16
MAX_SCORE_GAP = 8.0


class DraftVocabPrefetch:
    def __init__(self, model, ids, extra_ids, confirmed_quota, stats):
        self.model = model
        self.order = model.vocabulary_order
        self.base_rows = model.draft_vocabulary_rows
        self.extra_ids = extra_ids
        self.confirmed = set(extra_ids)
        self.confirmed_quota = confirmed_quota
        self.stats = stats
        self.age = 0
        self.prefetched = {}
        self.next_candidates = ()
        self.hints = None
        prompt_ids = mx.array(sorted(set(ids)), dtype=mx.int32)
        self.in_prompt = mx.zeros((model.vocab_size,), dtype=mx.int32).at[prompt_ids].add(1) != 0
        mx.eval(self.in_prompt)
        stats.update(draft_vocab_prefetched=0, draft_vocab_prefetch_confirmed=0,
                     draft_vocab_prefetch_peak_rows=0, draft_vocab_prefetch_gathered=0,
                     draft_vocab_prefetch_readout_s=0.0)

    def capture(self, scores, indices):
        scores = scores.astype(mx.float32)
        tokens = self.order.to_token_ids(indices)
        allowed = (tokens >= self.base_rows) & ~self.in_prompt[tokens]
        rare_scores = mx.where(allowed, scores, -mx.inf)
        winners = mx.argpartition(-rare_scores, TOP_TILES - 1, axis=-1)[:, :TOP_TILES]
        candidates = mx.take_along_axis(tokens, winners, axis=-1)
        gaps = mx.max(scores, axis=-1, keepdims=True) - mx.take_along_axis(rare_scores, winners, axis=-1)
        self.hints = candidates, gaps

    def observe(self, path):
        if self.hints is None:
            self.next_candidates = ()
            return
        start = perf_counter()
        candidates, gaps = self.hints
        self.hints = None
        accepted = set(path)
        ranking = {}
        for node, (ids, margins) in enumerate(zip(candidates.tolist(), gaps.tolist())):
            for token, gap in zip(ids, margins):
                if 0 <= gap <= MAX_SCORE_GAP:
                    priority = gap + (node not in accepted)
                    ranking[token] = min(ranking.get(token, priority), priority)
        self.next_candidates = sorted(ranking, key=lambda token: (ranking[token], token))
        self.stats["draft_vocab_prefetch_readout_s"] += perf_counter() - start

    def update(self, shortlist, head, pair_ids, budget):
        confirmed = list(dict.fromkeys(token for token in pair_ids
                                       if token >= self.base_rows and token not in self.confirmed))
        confirmed = confirmed[:self.confirmed_quota]
        self.stats["draft_vocab_prefetch_confirmed"] += sum(token in self.prefetched for token in confirmed)
        self.stats["draft_vocab_added"] += len(confirmed)
        self.stats["draft_vocab_updates"] += bool(confirmed)
        self.confirmed_quota -= len(confirmed)
        self.confirmed.update(confirmed)

        self.age += 1
        prefetched = {token: age for token, age in self.prefetched.items()
                      if self.age - age <= MAX_IDLE_ROUNDS and token not in self.confirmed}
        inserted = 0
        for token in self.next_candidates:
            if token in self.confirmed:
                continue
            inserted += token not in prefetched
            prefetched[token] = self.age
            if inserted == ROWS_PER_ROUND:
                break
        if len(prefetched) > MAX_PREFETCH_ROWS:
            recent = sorted(prefetched, key=prefetched.__getitem__, reverse=True)[:MAX_PREFETCH_ROWS]
            prefetched = {token: prefetched[token] for token in recent}
        self.prefetched = prefetched
        self.stats["draft_vocab_prefetched"] += inserted
        self.stats["draft_vocab_prefetch_peak_rows"] = max(self.stats["draft_vocab_prefetch_peak_rows"],
                                                         len(prefetched))

        desired = sorted(self.confirmed | prefetched.keys())
        if desired == self.extra_ids:
            return shortlist, head
        previous = {token: i for i, token in enumerate(self.extra_ids)}
        missing = [token for token in desired if token not in previous]
        row_bytes = 2 * self.model.config["hidden_size"]
        scratch = (len(previous) + 2 * len(desired) + 2 * len(missing)) * row_bytes
        scratch += 4 * (self.base_rows + len(previous) + 3 * len(desired))
        if mx.get_active_memory() + budget.cache_limit + scratch > budget.limit:
            raise MemoryBudgetError("Insufficient working-set headroom for draft vocabulary row rotation.")

        prefix = mx.arange(self.base_rows, dtype=mx.int32)
        if desired:
            # These IDs are outside the fixed draft prefix. Convert new rows to physical storage slots.
            prefix_half = self.base_rows // 2
            boundary = self.order.size // 2 + prefix_half
            routes = [previous[token] if token in previous else -1 - (token - prefix_half if token < boundary else token)
                      for token in desired]
            rows = rotate_vocabulary_rows(self.model.packed_head, mx.array(routes, dtype=mx.int32), head.extra)
            shortlist = mx.concatenate((prefix, mx.array(desired, dtype=mx.int32)))
        else:
            shortlist, rows = prefix, None
        self.extra_ids = desired
        self.stats["draft_vocab_prefetch_gathered"] += len(missing)
        return shortlist, DraftVocabulary(head.prefix, rows)
