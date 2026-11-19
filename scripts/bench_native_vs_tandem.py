"""Compare deployed native MTP and packed Tandem on one BF16 prefill."""

import argparse
from collections import Counter
from copy import copy
from dataclasses import replace
import json
from pathlib import Path
from unittest.mock import patch

import mlx.core as mx

from ultrarun import cli
from ultrarun.draft_short_head import prepare_short_head
from ultrarun.host import thermal_state
from ultrarun.memory import GIB, metal_job
from ultrarun.model import Model
from ultrarun.profiles import OPTIMIZED_DRAFT, OPTIMIZED_MODEL
from ultrarun.server import warm_verification_graphs
from scripts.bench_dflash2 import ROOT, bind_draft, load_draft, observe_layers, pages, tiled_candidates
from scripts.bench_dflash2_packed import fork_site_runner, pack_variants, short_vocabulary
from scripts.bench_dflash2_tandem import copy_verifier, draft_runner, exercise, verified_directories
from scripts.dflash2_ranked_head import CompactShortlist, lattice_for, profile_shortlist
from scripts.bench_verbatim_ffn_live import iterator


TARGET = ROOT / "weights/Qwen3.8-27B"
WORKLOADS = ("code", "chinese_technical_report", "english_technical_tutorial")


class PromptFeatures:
    def __init__(self, model, draft, prompt_length):
        self.layers = model.layers
        model.layers = list(self.layers)
        self.expected_rows = min(prompt_length, draft.config.sliding_window - 1)
        self.start = prompt_length - self.expected_rows
        self.taps = tuple(draft.config.target_layer_ids)
        self.blocks = {i: [] for i in self.taps}
        observe_layers(model, self.taps, self.record)

    def record(self, layer, offset, hidden):
        end = offset + hidden.shape[0]
        if end > self.start:
            begin = max(offset, self.start)
            self.blocks[layer].append((begin, hidden[begin - offset:]))

    def finish(self, model):
        model.layers = self.layers
        layers = []
        for tap in self.taps:
            position = self.start
            pieces = []
            for begin, values in sorted(self.blocks[tap], key=lambda pair: pair[0]):
                assert begin == position, (tap, begin, position)
                position += values.shape[0]
                pieces.append(values)
            assert position == self.start + self.expected_rows
            layers.append(mx.concatenate(pieces))
        features = mx.concatenate(layers, axis=-1)[None]
        mx.eval(features)
        self.blocks.clear()
        return features


def native_stream(model, cache, mtp_cache, positional, keywords, samples):
    options = keywords.copy()
    if (head := options.get("short_head")) is not None:
        new = copy(head)
        new.retained = head.retained.copy()
        new.complement_flags = []
        options["short_head"] = new
    return iterator(samples)(model, [replace(s) for s in cache], replace(mtp_cache), *positional, **options)


def paired_decode(model, draft, upstream, trace, observed, cap, budget, *, block, ranked_head, shortlist=None):
    runners = {"tandem": draft_runner(block)}
    fork_stats = {}
    if ranked_head:
        heads = {"tandem": None}
        if shortlist is not None:
            heads["tandem_shortlist"] = shortlist
        for name, compact in heads.items():
            fork_stats[name] = Counter()
            runners[name] = fork_site_runner(block, fork_stats[name], lattice=lattice_for(model, block, compact=compact))

    def run(engine, cache, mtp_cache, ids, hidden, pending, limit, stop, *args, **kwargs):
        assert kwargs["draft_schedule"] == "frontier_prefetch"
        features = observed.finish(engine)
        assert pending == trace["output"][0]
        warm_limit = 65
        selector_checked = {name: [False] for name in runners}

        def warm(trial):
            seen = dict.fromkeys(trial, 1)
            try:
                while min(seen.values()) < warm_limit:
                    arm = min(trial, key=seen.__getitem__)
                    seen[arm] = next(trial[arm])
            finally:
                for generator in trial.values():
                    generator.close()

        def make_native(samples):
            return native_stream(engine, cache, mtp_cache, (ids, hidden, pending, limit, stop, *args), kwargs, samples)

        def make_streams(samples, output_count):
            streams = {"mtp": make_native(samples)}
            for name, runner in runners.items():
                streams[name] = runner(copy_verifier(engine), draft, trace, cache, pending, features, cap,
                                       budget, upstream, selector_checked[name], output_count)
            return streams

        warm(make_streams([], warm_limit))
        for count in fork_stats.values():
            count.clear()
        mx.clear_cache()
        budget.check(force=True)
        mx.reset_peak_memory()
        before, thermal = pages(), thermal_state()
        samples = []
        results, elapsed = exercise(make_streams(samples, limit), limit, report_progress=True)
        actual, stats = results["mtp"]
        assert actual == trace["output"]
        print("result", json.dumps({
            "arm": "mtp", "tokens": len(actual), "rounds": stats["target_decode_passes"],
            "decode_s": elapsed["mtp"], "decode_tok_s": (len(actual) - 1) / elapsed["mtp"],
            "phase_s": {"draft": stats["draft_s"], "verify": stats["target_s"], "commit": stats["accept_s"]},
            "mean_advance": (len(actual) - 1) / stats["target_decode_passes"],
            "width8_or_less": sum(width <= 8 for _, width, _ in samples), "exact": True,
        }), flush=True)
        for name in runners:
            tandem = results[name]
            assert tandem["matches_reference"]
            count = tandem["rounds"]
            if name in fork_stats:
                assert fork_stats[name]["calls"] == count
            print("result", json.dumps({
                "arm": name, "tokens": tandem["tokens"] + 1, "rounds": count, "draft_block": block,
                "decode_s": elapsed[name], "decode_tok_s": tandem["tokens"] / elapsed[name],
                "phase_s": tandem["phase_s"], "mean_advance": tandem["mean_confirmed"],
                "width8_or_less": sum(n for width, n in tandem["verifier_widths"].items() if width <= 8),
                "exact": True,
            }), flush=True)
        after = pages()
        print("decode_memory", json.dumps({
            "peak_gib": mx.get_peak_memory() / GIB, "thermal": [thermal, thermal_state()],
            "faults": {key: val - before[key] for key, val in after.items()}, **budget.metrics(),
        }), flush=True)
        return actual, stats

    with patch.object(cli, "speculative", run):
        # Use the deployed prefill path, including the original MTP cache priming.
        eos = json.loads((TARGET / "generation_config.json").read_text())["eos_token_id"]
        stop = (eos,) if isinstance(eos, int) else tuple(eos)
        _, summary = cli.generate(model, trace["input"], 1024, 512, stop=stop, budget=budget,
                                  draft_schedule="frontier_prefetch", **OPTIMIZED_DRAFT)
    assert summary["draft_schedule"] == "frontier_prefetch"
    return summary["prefill_peak_gib"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workload", action="append", choices=WORKLOADS)
    parser.add_argument("--cap", type=int, choices=(8, 16), default=8)
    parser.add_argument("--tandem-block", type=int, choices=(8, 16), default=16)
    parser.add_argument("--full-vocab", action="store_true")
    parser.add_argument("--ranked-head", action="store_true", help="score the full BF16 head in storage order")
    parser.add_argument("--compare-compact-head", action="store_true",
                        help="also benchmark a directly ranked 32K-plus-prompt shortlist")
    parser.add_argument("--memory-gib", type=int, default=80)
    parser.add_argument("--upstream", type=Path, default=Path("/private/tmp/localllm-ref/dflash2/model_mlx.py"))
    args = parser.parse_args()
    if args.ranked_head and not args.full_vocab:
        parser.error("--ranked-head requires --full-vocab")
    if args.compare_compact_head and not args.ranked_head:
        parser.error("--compare-compact-head requires --ranked-head")
    if args.cap > args.tandem_block:
        parser.error("--cap cannot exceed the number of Tandem positions")
    path = verified_directories(("tandem_b16",))["tandem_b16"]
    cases = json.loads((ROOT / "runs/frontier-target-margin-audit.json").read_text())
    before = pages()
    with metal_job(args.memory_gib, 2) as budget:
        model = Model(TARGET, mtp=True, budget=budget, **OPTIMIZED_MODEL)
        warm_verification_graphs(model, budget)
        model.short_draft_head = prepare_short_head(model, budget)
        budget.admit(model, 5120, 512, True, extra_bytes=9 * GIB)
        base, upstream = load_draft(args.upstream, path)
        assert base.config.block_size == 16
        if args.ranked_head:
            assert base.config.output_multiplier == 1.0 and not base.config.final_logit_softcapping
        bind_draft(model, base, {})
        upstream.rank_candidates = tiled_candidates
        ffn, packed = pack_variants(base, budget)
        del ffn, base
        mx.clear_cache()
        budget.check(force=True)
        for workload in args.workload or ("code", "chinese_technical_report"):
            trace = cases[workload]
            assert len(trace["input"]) == 4096 and len(trace["output"]) == 1024
            print("workload", workload, "cap", args.cap, "full_vocab", args.full_vocab,
                  "draft_block", args.tandem_block, "ranked_head", args.ranked_head, flush=True)
            active = packed if args.full_vocab else short_vocabulary(packed, model, trace["input"])
            shortlist = CompactShortlist(model, trace["input"]) if args.compare_compact_head else None
            if shortlist is not None:
                print("compact_head", json.dumps({"base_rows": shortlist.rows, "prompt_rows": shortlist.extras.size,
                         "extra_bytes": 0 if shortlist.weights is None else shortlist.weights.nbytes}), flush=True)
                profile_shortlist(model, shortlist)
            observed = PromptFeatures(model, active, len(trace["input"]))
            prefill_peak = paired_decode(model, active, upstream, trace, observed, args.cap, budget,
                                         block=args.tandem_block, ranked_head=args.ranked_head, shortlist=shortlist)
            print("prefill_peak_gib", prefill_peak, flush=True)
            del active, observed, shortlist
            mx.clear_cache()
            budget.check(force=True)
        after = pages()
        print("host", json.dumps({k: val - before[k] for k, val in after.items()}), flush=True)


if __name__ == "__main__":
    main()
