"""Compare native MTP with full and shorter prompt attention on one prefill."""

import argparse
from copy import copy
from dataclasses import replace
import inspect
import json
from statistics import median
from time import perf_counter
from unittest.mock import patch

import mlx.core as mx

from localllm import cli
from localllm.draft_short_head import prepare_short_head
from localllm.host import thermal_state
from localllm.logical_attention import logical_attention
from localllm.memory import GIB, metal_job
from localllm.model import Model
from localllm.profiles import OPTIMIZED_DRAFT, OPTIMIZED_MODEL
from localllm.server import warm_verification_graphs
from scripts.bench_dflash2 import pages
from scripts.bench_ready import ROOT, TARGET, digest


def shorten_prompt(prompt):
    return tuple(mx.concatenate((t[:, :, :128], t[:, :, -2048:]), axis=2) for t in prompt)


def measure_attention():
    c = json.loads((TARGET / "config.json").read_text())["text_config"]
    hq, hk, dim = c["num_attention_heads"], c["num_key_value_heads"], c["head_dim"]
    mx.random.seed(17)
    with metal_job(4, 1) as budget:
        query = mx.random.normal((1, hq, 1, dim)).astype(mx.bfloat16)
        prompt, tail = [tuple(mx.random.normal((1, hk, rows, dim)).astype(mx.bfloat16) for _ in range(2))
                        for rows in (4095, 1024)]
        windows = (prompt, shorten_prompt(prompt))
        mx.eval(query, windows, tail)
        for rows in (1, 512, 1024):
            key, value = (t[:, :, :rows] for t in tail)

            def work(window):
                return logical_attention(query, key, value, dim ** -0.5, None,
                                         aligned_bf16=True, prompt_kv=window)

            for window in windows:
                actual = work(window)
                reference = mx.fast.scaled_dot_product_attention(
                    query, mx.concatenate((window[0], key), axis=2),
                    mx.concatenate((window[1], value), axis=2), scale=dim ** -0.5)
                assert mx.allclose(actual, reference, atol=0.002, rtol=0.02).item()
            times = [[], []]
            for turn in range(48):
                for i in (turn % 2, 1 - turn % 2):
                    t = perf_counter()
                    mx.eval(work(windows[i]))
                    times[i].append((perf_counter() - t) * 1_000_000)
            print("micro", json.dumps({"suffix_tokens": rows, "full_us": median(times[0][8:]),
                  "short_us": median(times[1][8:]), "peak_gib": mx.get_peak_memory() / GIB,
                  "thermal": thermal_state(), **budget.metrics()}), flush=True)
            budget.check(force=True)


def round_iterator():
    source = inspect.getsource(cli.speculative)
    boundary = '    stats["accepted_per_pass_histogram"] = acceptance'
    assert source.count(boundary) == 1
    namespace = vars(cli).copy()
    exec(compile(source.replace(boundary, "        yield len(generated)\n" + boundary), __file__, "exec"), namespace)
    return namespace["speculative"]


def compare(trace, iterator):
    def run(model, cache, mtp, *args, **kwargs):
        prompt = mtp.prompt_kv
        assert prompt is not None and all(t.shape[2] == 4095 for t in prompt)
        assert sum(state.prompt_kv is not None for state in cache) == 16
        shortened = replace(mtp, draft_prompt_kv=shorten_prompt(prompt))
        mx.eval(shortened.draft_prompt_kv)
        alt = kwargs.copy()
        if (head := alt.get("short_head")) is not None:
            short_head = copy(head)
            short_head.retained = head.retained.copy()
            short_head.complement_flags = []
            alt["short_head"] = short_head
        streams = {
            "full": iterator(model, [replace(s) for s in cache], replace(mtp), *args, **kwargs),
            "short": iterator(model, [replace(s) for s in cache], shortened, *args, **alt),
        }
        results = {}
        times = dict.fromkeys(streams, 0.0)
        before, thermal = pages(), thermal_state()
        turn = 0
        while streams:
            order = tuple(streams) if turn % 2 == 0 else tuple(reversed(streams))
            for label in order:
                start = perf_counter()
                try:
                    next(streams[label])
                except StopIteration as done:
                    results[label] = done.value
                    del streams[label]
                times[label] += perf_counter() - start
            turn += 1
        after = pages()
        for label, (tokens, metrics) in results.items():
            expected = trace["output"]
            divergence = next((i for i, (a, b) in enumerate(zip(tokens, expected)) if a != b), None)
            summary = {k: metrics[k] for k in ("target_decode_passes", "draft_s", "target_s", "accept_s",
                                               "draft_scores_evaluated", "drafts_per_pass_histogram")}
            print("result", label, json.dumps({"tokens": len(tokens), "exact_output": tokens == expected,
                  "first_divergence": divergence, "token_sha256": digest(tokens),
                  "decode_tok_s": (len(tokens) - 1) / times[label], **summary}), flush=True)
        print("decode", json.dumps({"peak_gib": mx.get_peak_memory() / GIB, "thermal": [thermal, thermal_state()],
              "faults": {key: after[key] - before[key] for key in before}}), flush=True)
        assert results["full"][0] == trace["output"], "Baseline must reproduce the archived BF16 output"
        return results["full"]
    return run


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workload", action="append",
                        choices=("code", "english_technical_tutorial", "chinese_technical_report"))
    parser.add_argument("--attention-only", action="store_true", help="time the attention kernel without loading weights")
    args = parser.parse_args()
    if args.attention_only:
        measure_attention()
        return
    traces = json.loads((ROOT / "runs/frontier-target-margin-audit.json").read_text())
    stops = json.loads((TARGET / "generation_config.json").read_text())["eos_token_id"]
    stops = (stops,) if isinstance(stops, int) else tuple(stops)
    iterator = round_iterator()
    start_pages = pages()
    with metal_job(80, 2) as budget:
        model = Model(TARGET, budget=budget, mtp=True, **OPTIMIZED_MODEL)
        warm_verification_graphs(model, budget)
        model.short_draft_head = prepare_short_head(model, budget)
        print("load_peak_gib", mx.get_peak_memory() / GIB, flush=True)
        for label in args.workload or ("code",):
            trace = traces[label]
            assert len(trace["input"]) == 4096 and len(trace["output"]) == 1024
            print("workload", label, "draft_prompt_rows", [4095, 2176], flush=True)
            with patch.object(cli, "speculative", compare(trace, iterator)):
                _, stats = cli.generate(model, trace["input"], 1024, 512, stop=stops, budget=budget,
                                        draft_schedule="frontier_prefetch", **OPTIMIZED_DRAFT)
            assert stats["draft_schedule"] == "frontier_prefetch"
            print("memory", json.dumps({"prefill_peak_gib": stats["prefill_peak_gib"], **budget.metrics()}), flush=True)
            mx.clear_cache()
            budget.check(force=True)
        end_pages = pages()
        print("host_faults", json.dumps({k: end_pages[k] - start_pages[k] for k in start_pages}), flush=True)


if __name__ == "__main__":
    main()
