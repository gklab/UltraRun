"""Test a single reuse of the previous DSpine continuation after BF16 verification."""

import argparse
from collections import Counter
from dataclasses import replace
import json

import mlx.core as mx

from localllm.host import thermal_state
from localllm.memory import GIB, metal_job
from scripts.audit_dspine_complement import capture_mtp
from scripts.bench_dflash2 import pages
from scripts.bench_dspine import (
    DSpineDraft, ROOT, confirmation_ratio, layout_tree, rank_tree, verify_checkpoint,
    with_packed_ffn, with_packed_projections, with_prompt_vocabulary,
)
from scripts.bench_dspine_adaptive import advance, phase_costs
from scripts.bench_dspine_ffn import paired_decode
from scripts.bench_ready import benchmark_target, prepare_prefix


WIDTH_RATIO = 1.70
BONUS_RANK_LIMITS = (1, 4, 16)


def fresh_plan(pending, lattice, width):
    nodes, parents, costs = rank_tree(pending, *lattice, width)
    if len(nodes) > 8 and confirmation_ratio(costs) < WIDTH_RATIO:
        return nodes[:8], parents[:8]
    return nodes, parents


def old_tail(lattice, confirmed, width):
    """Realign using only the verified path, including the target's bonus token."""
    options, logp = lattice
    depth = len(confirmed)
    cap = min(width, 8)
    if not depth or cap < 2 or len(options) - depth < cap - 1:
        return None
    previous = 0
    for row, token in enumerate(confirmed):
        choices = options[row]
        if token not in choices:
            return None
        scores = logp[row][previous]
        previous = choices.index(token)
    selected = scores[previous]
    rank = 1 + sum(score > selected or (score == selected and candidate < confirmed[-1])
                   for score, candidate in zip(scores, choices, strict=True))
    nodes, parents, _ = rank_tree(confirmed[-1], options[depth:], logp[depth:], cap, previous_slot=previous)
    return nodes, parents, rank


class ReusingDraft:
    def __init__(self, base, rank_limit):
        self.base = base
        self.config = base.config
        self.rank_limit = rank_limit
        self.lattice = None
        self.confirmed = ()
        self.fresh = self.reused = 0
        self.reused_tokens = 0
        self.using_tail = False

    def context(self, features, start, budget):
        return self.base.context(features, start, budget)

    def observe_verified(self, tokens):
        self.confirmed = tokens
        if self.using_tail:
            self.reused_tokens += len(tokens)

    def propose_tree(self, model, pending, prefix, width):
        self.using_tail = False
        lattice, self.lattice = self.lattice, None
        if lattice is not None and self.confirmed and self.confirmed[-1] == pending:
            plan = old_tail(lattice, self.confirmed, width)
            if plan is not None:
                nodes, parents, bonus_rank = plan
                if bonus_rank <= self.rank_limit:
                    self.reused += 1
                    self.using_tail = True
                    return layout_tree(nodes, parents)
        self.lattice = self.base.candidate_lattice(model, pending, prefix)
        self.fresh += 1
        return layout_tree(*fresh_plan(pending, self.lattice, width))


def measure_options(model, draft, trace, prefix, budget):
    ids = trace["output"]
    prompt_len = len(trace["input"])
    count = len(ids) - 1
    options = []
    for i, pending in enumerate(ids[:-1]):
        offset = prompt_len + i
        local = [tuple(t[:, :, :offset] for t in pair) for pair in prefix]
        lattice = draft.candidate_lattice(model, pending, local)
        nodes, parents = fresh_plan(pending, lattice, min(16, count - i + 1))
        truth = ids[i + 1:i + 17]
        confirmed = advance(nodes, parents, truth, len(nodes))
        nxt = i + confirmed
        plan = old_tail(lattice, truth[:confirmed], min(8, count - nxt + 1))
        rank, extra = None, 0
        if plan is not None:
            reused_nodes, reused_parents, rank = plan
            extra = advance(reused_nodes, reused_parents, ids[nxt + 1:nxt + 9], len(reused_nodes))
        options.append((16 if len(nodes) > 8 else 8, confirmed, rank, extra))
        if len(options) % 256 == 0:
            print("reuse_anchors", len(options), flush=True)
        budget.check()
    return options


def replay(options, rank_limit, costs, reuse_ms):
    count = len(options)
    i, ms, reuse_tokens = 0, 0.0, 0
    phases = Counter()
    while i < count:
        width, amount, rank, extra = options[i]
        assert 1 <= amount <= count - i and 0 <= extra <= count - i - amount
        i += amount
        ms += costs[width]
        phases["fresh"] += 1
        if rank is not None and rank <= rank_limit and extra:
            i += extra
            ms += reuse_ms
            phases["reused"] += 1
            reuse_tokens += extra
    return {"ms": ms, "tok_s": count * 1000 / ms, "rounds": phases, "reuse_tokens": reuse_tokens}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bonus-rank-limit", type=int, choices=BONUS_RANK_LIMITS,
                        help="repeat a selected policy without recomputing the teacher trace")
    rank_limit = parser.parse_args().bonus_rank_limit
    verify_checkpoint()
    trace = json.loads((ROOT / "runs/frontier-target-margin-audit.json").read_text())["code"]
    assert len(trace["input"]) == 4096 and len(trace["output"]) == 1024
    before, thermal = pages(), thermal_state()
    with metal_job(80, 2) as budget:
        draft = DSpineDraft()
        mx.eval(draft.weights, draft.code_projection, draft.sender)
        draft = with_packed_ffn(draft, budget)
        draft = with_packed_projections(draft, budget)
        mx.clear_cache()
        model = benchmark_target(budget)
        budget.admit(model, 5120, 512, False, extra_bytes=2 * GIB)
        draft = with_prompt_vocabulary(draft, model, trace["input"])
        draft.wide_cost_ratio = WIDTH_RATIO
        prepared = prepare_prefix(model, draft, trace, budget)
        if rank_limit is None:
            cache, first, prompt = prepared
            teacher = [replace(state) for state in cache], first, prompt
            native, prefix = capture_mtp(model, draft, trace, teacher, budget)
            print("mtp_replayed", len(native), "exact_tokens", len(trace["output"]), flush=True)
            del native, teacher
            options = measure_options(model, draft, trace, prefix, budget)
            del prefix
            mx.clear_cache()
            budget.check(force=True)

            costs = phase_costs()
            # Charge an explicit millisecond for CPU ranking the old lattice.
            reuse_ms = costs[8] - 16.94 + 1.0
            results = {k: replay(options, k, costs, reuse_ms) for k in (0, *BONUS_RANK_LIMITS)}
            reported = {"fresh_ms": costs, "reuse_ms": reuse_ms, "policies": results}
            print("reuse_replay", json.dumps(reported), flush=True)
            (ROOT / "runs/dspine-lattice-reuse-replay.json").write_text(
                json.dumps({"result": reported, "options": options}) + "\n")
            rank_limit = min(BONUS_RANK_LIMITS, key=lambda rank: results[rank]["ms"])
        print("tested_rank_limit", rank_limit, flush=True)
        candidate = ReusingDraft(draft, rank_limit)
        paired_decode(model, (draft, candidate), trace, prepared, 16, True, budget,
                      labels=("fresh", "reuse_tail"))
        print("reuse_counts", json.dumps({"fresh": candidate.fresh, "reused": candidate.reused,
              "reused_confirmed_tokens": candidate.reused_tokens}), flush=True)
        after = pages()
        print("host", json.dumps({"thermal": [thermal, thermal_state()],
              "faults": {key: after[key] - before[key] for key in before}, **budget.metrics()}), flush=True)


if __name__ == "__main__":
    main()
