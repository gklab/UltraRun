"""Read DSpine's early token codes and verify the candidates with full BF16 Qwen."""

import argparse
from collections import Counter
from dataclasses import replace
import json
from statistics import median
from time import perf_counter

import mlx.core as mx

from localllm.host import thermal_state
from localllm.memory import GIB, metal_job
from scripts.audit_dspine_complement import capture_mtp
from scripts.bench_dflash2 import pages
from scripts.bench_dspine import (
    DSpineDraft, ROOT, layout_tree, verify_checkpoint, with_packed_ffn, with_packed_projections,
    with_prompt_vocabulary,
)
from scripts.bench_dspine_adaptive import advance, phase_costs
from scripts.bench_dspine_ffn import paired_decode
from scripts.bench_dspine_reuse import WIDTH_RATIO, fresh_plan
from scripts.bench_ready import benchmark_target, linear, prepare_prefix


DEPTHS = (2, 3, 4)
BASE = "full_5"


class ShallowDraft:
    def __init__(self, base, depth):
        assert 0 < depth < base.config.num_hidden_layers
        self.base, self.depth = base, depth
        self.config = base.config

    def context(self, features, start, budget):
        return self.base.context(features, start, budget, layer_count=self.depth)


class EarlyCodeDraft(ShallowDraft):
    verification_width = 8

    def __init__(self, base, depth, codebook, row_norms):
        super().__init__(base, depth)
        self.codebook, self.row_norms = codebook, row_norms

    def candidates(self, model, pending, prefix, width):
        codes = self.base.hidden(model, pending, prefix[:self.depth], width, early_depth=self.depth)
        scores = linear(codes, self.codebook).astype(mx.float32) / self.row_norms
        return self.base.proposal_vocab[0][mx.argmax(scores, axis=-1)]

    def propose_tree(self, model, pending, prefix, width):
        guessed = self.candidates(model, pending, prefix, width).tolist()
        return [pending, *guessed], [-1, *range(width - 1)], width - 1


class EarlyHeadDraft(ShallowDraft):
    """Keep final vocabulary ranking and re-score the actual last feature write."""

    verification_width = 16

    def propose_tree(self, model, pending, prefix, width):
        choices, scores, rms = self.base.lattice(model, pending, prefix, self.config.block_size,
                                                backbone_depth=self.depth)
        logits = scores / mx.maximum(rms[:, None, :], self.base.eps)
        logp = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
        mx.eval(choices, logp)
        return layout_tree(*fresh_plan(pending, (choices.tolist(), logp.tolist()), width))


def measure(model, base, early, trace, prefix, budget):
    ids, context = trace["output"], len(trace["input"])
    names = (BASE, *early)
    results = {key: [] for key in names}
    for i, pending in enumerate(ids[:-1]):
        offset = context + i
        local = [tuple(t[:, :, :offset] for t in pair) for pair in prefix]
        truth = ids[i + 1:i + 17]
        turn = i % len(names)
        for name in (*names[turn:], *names[:turn]):
            start = perf_counter()
            if name == BASE:
                lattice = base.candidate_lattice(model, pending, local)
                nodes, parents = fresh_plan(pending, lattice, min(16, len(truth) + 1))
            else:
                draft = early[name]
                nodes, parents, _ = draft.propose_tree(model, pending, local, min(draft.verification_width, len(truth) + 1))
            draft_ms = (perf_counter() - start) * 1000
            verified = advance(nodes, parents, truth, len(nodes))
            results[name].append((len(nodes), verified, draft_ms))
        if (i + 1) % 256 == 0:
            print("early_code_anchors", i + 1, flush=True)
        budget.check()
    return results


def play(rows, price, draft_ms):
    i, ms = 0, 0.0
    widths = Counter()
    while i < len(rows):
        width, kept, _ = rows[i]
        widths[width] += 1
        ms += price[16 if width > 8 else 8] + draft_ms
        i += min(kept, len(rows) - i)
    return {"tok_s": len(rows) * 1000 / ms, "rounds": sum(widths.values()), "widths": widths, "draft_ms": draft_ms}


def oracle(rows, target_ms, draft_ms):
    """Estimate a perfect chooser of the existing plans at each verified prefix."""
    count = len(rows[BASE])
    future = [0.0] * (count + 1)
    for i in range(count - 1, -1, -1):
        future[i] = min(target_ms[16 if curve[i][0] > 8 else 8] + draft_ms[name] + future[i + curve[i][1]]
                        for name, curve in rows.items())
    return count * 1000 / future[0]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workload", choices=("code", "chinese_technical_report", "english_technical_tutorial"),
                        default="code")
    parser.add_argument("--depth", type=int, choices=DEPTHS, help="repeat a live depth without an offline replay")
    parser.add_argument("--four-layer-head", action="store_true",
                        help="leave after four layers but use final RMSNorm, vocabulary and adaptive tree")
    args = parser.parse_args()
    if args.depth and args.four_layer_head:
        parser.error("choose either --depth or --four-layer-head")
    verify_checkpoint()
    trace = json.loads((ROOT / "runs/frontier-target-margin-audit.json").read_text())[args.workload]
    assert len(trace["input"]) == 4096 and len(trace["output"]) == 1024
    before, thermal = pages(), thermal_state()
    with metal_job(80, 2) as budget:
        base = DSpineDraft()
        mx.eval(base.weights, base.code_projection, base.sender)
        base = with_packed_ffn(base, budget)
        base = with_packed_projections(base, budget)
        mx.clear_cache()
        model = benchmark_target(budget)
        budget.admit(model, 5120, 512, False, extra_bytes=2 * GIB)
        base = with_prompt_vocabulary(base, model, trace["input"])
        base.context_window = (128, 2048)
        base.wide_cost_ratio = WIDTH_RATIO
        if args.four_layer_head:
            early = {"head_4": EarlyHeadDraft(base, 4)}
        else:
            codebook = base.bridge("codebook")[base.proposal_vocab[0]]
            row_norms = mx.maximum(mx.linalg.norm(codebook.astype(mx.float32), axis=-1), base.eps)
            mx.eval(codebook, row_norms)
            print("codebook", json.dumps({"rows": codebook.shape[0], "dimensions": codebook.shape[1],
                  "extra_bytes": codebook.nbytes + row_norms.nbytes}), flush=True)
            early = {f"early_{depth}": EarlyCodeDraft(base, depth, codebook, row_norms)
                     for depth in ((args.depth,) if args.depth else DEPTHS)}
        prepared = prepare_prefix(model, base, trace, budget)
        cache, first, prompt = prepared
        selected = f"early_{args.depth}"
        if args.depth is None:
            teacher = [replace(state) for state in cache], first, prompt
            native, prefix = capture_mtp(model, base, trace, teacher, budget)
            print("mtp_replayed", len(native), "exact_tokens", len(trace["output"]), flush=True)
            del teacher, native
            rows = measure(model, base, early, trace, prefix, budget)
            del prefix
            mx.clear_cache()
            budget.check(force=True)
            target_ms = {width: cost - 16.94 for width, cost in phase_costs().items()}
            split = len(rows[BASE]) // 2
            draft_ms = {name: median(row[2] for row in points[16:split]) for name, points in rows.items()}
            report = {name: {"first_half": play(points[:split], target_ms, draft_ms[name]),
                             "second_half": play(points[split:], target_ms, draft_ms[name]),
                             "whole": play(points, target_ms, draft_ms[name])}
                      for name, points in rows.items()}
            print("early_replay", json.dumps({"target_ms": target_ms, "depths": report,
                  "perfect_chooser_tok_s": oracle(rows, target_ms, draft_ms)}), flush=True)
            selected = max(early, key=lambda name: report[name]["first_half"]["tok_s"])
        print("workload", args.workload, "tested_depth", selected, flush=True)
        chosen = early[selected]
        candidate = cache, first, prompt[:chosen.depth]
        paired_decode(model, (base, chosen), trace, prepared, (16, chosen.verification_width), True, budget,
                      labels=(BASE, selected), prepared_by_arm=(prepared, candidate), align_tokens=True)
        after = pages()
        print("host", json.dumps({"thermal": [thermal, thermal_state()],
              "faults": {key: after[key] - before[key] for key in before}, **budget.metrics()}), flush=True)


if __name__ == "__main__":
    main()
