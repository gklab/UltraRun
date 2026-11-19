"""Use exact MTP embedding lane sums in single and mixed-hit batches."""

from collections import Counter
from copy import copy
from dataclasses import replace
from functools import lru_cache
from hashlib import sha256
import inspect
import json
from statistics import median
from textwrap import dedent
from time import perf_counter
from unittest.mock import patch

import mlx.core as mx

from localllm import cli, frontier, model as modeling
from localllm.draft_projection import CAPACITY, FLOAT4, continue_lane_prefix, multi_gemv
from localllm.draft_short_head import prepare_short_head
from localllm.host import thermal_state
from localllm.memory import GIB, metal_job
from localllm.profiles import OPTIMIZED_DRAFT, OPTIMIZED_MODEL
from localllm.server import warm_verification_graphs
from scripts.bench_dflash2 import pages
from scripts.bench_dflash2_tandem import exercise
from scripts.bench_ready import ROOT, TARGET, digest
from scripts.bench_verbatim_ffn import replace_once


PRODUCTION = cli.speculative


@lru_cache(maxsize=1)
def projection_kernel():
    return mx.fast.metal_kernel(
        name="probe_mtp_mixed_lane_prefix",
        input_names=["weights", "hidden_weights", "hidden", "embeddings", "missing", "sums", "slots", "tokens"],
        output_names=["output"], header=FLOAT4, source=r"""
        uint lane = thread_position_in_grid.x % 32;
        uint first = thread_position_in_grid.x / 32 * ROWS;
        uint rows = weights_shape[0];
        uint cols = hidden_shape[1] / 4;
        float accum[PATHS][ROWS] = {};
        int pending_row[PATHS];
        #pragma unroll
        for (uint p = 0; p < PATHS; ++p) {
            pending_row[p] = missing[p];
            if (pending_row[p] < 0) {
                uint slot = uint(slots[tokens[p]]);
                #pragma unroll
                for (uint r = 0; r < ROWS; ++r) {
                    accum[p][r] = sums[(slot * rows + first + r) * 32 + lane];
                }
            }
        }
        if (HAS_MISSES) {
            for (uint col = lane; col < cols; col += 32) {
                float4 embed[PATHS];
                #pragma unroll
                for (uint p = 0; p < PATHS; ++p) {
                    if (pending_row[p] >= 0) embed[p] = load4(embeddings, uint(pending_row[p]) * cols + col);
                }
                #pragma unroll
                for (uint r = 0; r < ROWS; ++r) {
                    float4 weight = load4(weights, (first + r) * cols * 2 + col);
                    #pragma unroll
                    for (uint p = 0; p < PATHS; ++p) {
                        if (pending_row[p] >= 0) accum[p][r] = step4(weight, embed[p], accum[p][r]);
                    }
                }
            }
        }
        for (uint col = lane; col < cols; col += 32) {
            float4 input[PATHS];
            #pragma unroll
            for (uint p = 0; p < PATHS; ++p) input[p] = load4(hidden, p * cols + col);
            #pragma unroll
            for (uint r = 0; r < ROWS; ++r) {
                float4 weight = load4(hidden_weights, (first + r) * cols + col);
                #pragma unroll
                for (uint p = 0; p < PATHS; ++p) accum[p][r] = step4(weight, input[p], accum[p][r]);
            }
        }
        #pragma unroll
        for (uint p = 0; p < PATHS; ++p) {
            #pragma unroll
            for (uint r = 0; r < ROWS; ++r) {
                #pragma unroll
                for (uint shift = 16; shift != 0; shift >>= 1)
                    accum[p][r] += simd_shuffle_down(accum[p][r], shift);
                if (lane == 0) output[p * rows + first + r] = bfloat16_t(accum[p][r]);
            }
        }
        """,
    )


class MaskedInput:
    def __init__(self, model, ids):
        self.model = model
        self.ids = {token for token, _ in Counter(ids).most_common(CAPACITY)}
        width = model.config["hidden_size"]
        assert model.mtp_fc.shape == (5120, 10240) and model.mtp_fc.dtype == mx.bfloat16
        self.hidden_weights = mx.contiguous(model.mtp_fc[:, width:])
        self.empty = mx.zeros((1, width), dtype=mx.bfloat16)
        self.counts = Counter()
        self.single_call, self.batch_call = alternate_entries()
        mx.eval(self.hidden_weights, self.empty)

    def project(self, normalized, tokens, known, memo):
        sources, row_index = [], []
        for token in known:
            if token in self.ids:
                row_index.append(-1)
            else:
                row_index.append(len(sources))
                sources.append(token)
        embed = self.empty
        if sources:
            embed = self.model.embedding[mx.array(sources, dtype=mx.int32)]
            embed = mx.fast.rms_norm(embed, self.model.mtp_norms[0],
                                     self.model.config["rms_norm_eps"]).astype(mx.bfloat16)
        paths, rows = normalized.shape
        per_simd = 1 if paths == 1 else 2
        out, = projection_kernel()(
            inputs=[self.model.mtp_fc, self.hidden_weights, normalized, embed, mx.array(row_index, dtype=mx.int32),
                    memo.sums, memo.slots, tokens],
            template=[("PATHS", paths), ("ROWS", per_simd), ("HAS_MISSES", bool(sources))],
            grid=(rows // per_simd * 32, 1, 1), threadgroup=(256, 1, 1),
            output_shapes=[(paths, rows)], output_dtypes=[mx.bfloat16],
        )
        self.counts[f"{paths}_{'mixed' if sources else 'cached'}"] += 1
        return out

    def single(self, model, hidden, tokens, state, offset, known):
        if len(known) == 1 and state.draft_lanes is not None and known[0] in self.ids:
            return self.single_call(model, hidden, tokens, state, offset, masked=(self, known))
        return model.draft(hidden, tokens, state, offset)

    def batch(self, model, hidden, tokens, states, offsets, head, known):
        memo = states[0].draft_lanes
        if memo is not None and all(state.draft_lanes is memo for state in states) and any(t in self.ids for t in known):
            return self.batch_call(model, hidden, tokens, states, offsets, head, masked=(self, known, memo))
        return model.draft_batch(hidden, tokens, states, offsets, head)


def alternate_entries():
    one = dedent(inspect.getsource(modeling.Model.draft))
    one = replace_once(one, "cached_embedding=None):", "cached_embedding=None, masked=None):")
    one = replace_once(one, "    if cached_embedding is None:", """    if masked is not None:
        x = masked[0].project(hidden, next_tokens, masked[1], cache.draft_lanes)
    elif cached_embedding is None:""")
    batch = dedent(inspect.getsource(modeling.Model.draft_batch))
    batch = replace_once(batch, "caches, offsets, head):", "caches, offsets, head, *, masked):")
    batch = replace_once(batch, '    inputs = mx.fast.rms_norm(self.embedding[tokens], embed_norm, eps).astype(mx.bfloat16)\n', "")
    batch = replace_once(batch, "    x = multi_gemv(mx.concatenate((inputs, hidden), axis=-1), self.mtp_fc)",
                         "    x = masked[0].project(hidden, tokens, masked[1], masked[2])")
    scope = vars(modeling).copy()
    exec(compile(one + "\n" + batch, __file__, "exec"), scope)
    return scope["draft"], scope["draft_batch"]


def micro(engine, memo, budget, ids):
    hits = list(dict.fromkeys(token for token in ids if token in engine.ids))
    miss = next(i for i in range(1000) if i not in engine.ids)
    cases = (hits[:1], hits[:2], [hits[0], miss], hits[:3], [hits[0], miss, hits[1]])
    model = engine.model
    for known in cases:
        n = len(known)
        tokens = mx.array(known, dtype=mx.int32)
        normalized = mx.random.normal((n, 5120)).astype(mx.bfloat16)
        mx.eval(tokens, normalized)

        def baseline():
            emb = model.embedding[tokens]
            emb = mx.fast.rms_norm(emb, model.mtp_norms[0], model.config["rms_norm_eps"]).astype(mx.bfloat16)
            x = mx.concatenate((emb, normalized), axis=-1)
            if n == 1:
                return continue_lane_prefix(model.mtp_fc, x, memo, tokens)
            return multi_gemv(x, model.mtp_fc)

        def candidate():
            return engine.project(normalized, tokens, known, memo)

        assert mx.array_equal(baseline().view(mx.uint16), candidate().view(mx.uint16)).item(), known
        times = ([], [])
        for trial in range(24):
            for arm in (trial % 2, 1 - trial % 2):
                started = perf_counter()
                mx.eval((baseline, candidate)[arm]())
                times[arm].append((perf_counter() - started) * 1000)
        print("projection", json.dumps({
            "paths": n, "hits": sum(t in engine.ids for t in known), "bitwise": True,
            "base_ms": median(times[0]), "masked_ms": median(times[1]),
            "paired_saved_ms": median(a - b for a, b in zip(*times, strict=True)),
        }), flush=True)
        budget.check()


def iterator(engine, route):
    builder = frontier.build_frontier
    source = inspect.getsource(PRODUCTION)
    scope = vars(cli).copy()
    if engine is not None:
        draft = inspect.getsource(builder)
        draft = replace_once(draft, "            output = model.draft(hidden, tokens, states[0], offsets[0])",
                             "            output = engine.single(model, hidden, tokens, states[0], offsets[0], (nodes[active[0]].token,))")
        draft = replace_once(draft, "            output, scores = model.draft_batch(hidden, tokens, states, offsets, head)",
                             "            output, scores = engine.batch(model, hidden, tokens, states, offsets, head, [nodes[i].token for i in active])")
        provisional = {**vars(frontier), "engine": engine}
        exec(compile(draft, __file__, "exec"), provisional)
        builder = provisional["build_frontier"]
        source = replace_once(source, "            root = model.draft(hidden, mx.array(pair_ids, dtype=mx.int32), mtp_cache, mtp_offset)[-1:]",
                              "            root = engine.single(model, hidden, mx.array(pair_ids, dtype=mx.int32), mtp_cache, mtp_offset, pair_ids)[-1:]")
        scope["engine"] = engine
    verified = "        verified = model.hidden(batch, cache, offset, keep_replay=bool(proposals), tree=tree)"
    source = replace_once(source, verified,
                          "        route.update(json.dumps((tokens, parents if tree is not None else None)).encode())\n" + verified)
    end = '    stats["accepted_per_pass_histogram"] = acceptance'
    source = replace_once(source, end, "        yield len(generated)\n" + end)
    scope.update(build_frontier=builder, route=route, json=json)
    exec(compile(source, __file__, "exec"), scope)
    return scope["speculative"]


def compare(trace, budget):
    def hook(model, cache, mtp, *args, **kwargs):
        assert mtp.draft_lanes is not None
        budget.check(force=True, required_bytes=GIB)
        engine = MaskedInput(model, trace["input"])
        micro(engine, mtp.draft_lanes, budget, trace["input"])

        def streams(routes):
            def make(name, candidate):
                options = kwargs.copy()
                if (head := options.get("short_head")) is not None:
                    current = copy(head)
                    current.retained = head.retained.copy()
                    current.complement_flags = []
                    options["short_head"] = current
                return iterator(candidate, routes[name])(
                    model, [replace(s) for s in cache], replace(mtp), *args, **options)
            return {"base": make("base", None), "masked": make("masked", engine)}

        warm = streams({"base": sha256(), "masked": sha256()})
        try:
            for stream in warm.values():
                while next(stream) < 65:
                    pass
        finally:
            for stream in warm.values():
                stream.close()
        engine.counts.clear()
        mx.clear_cache()
        budget.check(force=True)
        mx.reset_peak_memory()
        routes = {"base": sha256(), "masked": sha256()}
        before, thermal = pages(), thermal_state()
        results, elapsed = exercise(streams(routes), 1024, report_progress=True)
        assert all(tokens == trace["output"] for tokens, _ in results.values())
        assert routes["masked"].digest() == routes["base"].digest()
        for name, (tokens, stats) in results.items():
            print("live", json.dumps({"arm": name, "decode_tok_s": 1023 / elapsed[name],
                                      "output_tokens": len(tokens), "sha256": digest(tokens),
                                      "rounds": stats["target_decode_passes"],
                                      "draft_s": stats["draft_s"], "target_s": stats["target_s"]}), flush=True)
        print("usage", json.dumps({"routed": engine.counts, "identical_tree_sequence": True,
                                   "extra_mib": engine.hidden_weights.nbytes / 2**20,
                                   "peak_gib": mx.get_peak_memory() / GIB, "thermal": [thermal, thermal_state()],
                                   "faults": {k: v - before[k] for k, v in pages().items()},
                                   **budget.metrics()}), flush=True)
        return results["base"]
    return hook


def main():
    trace = json.loads((ROOT / "runs/frontier-target-margin-audit.json").read_text())["code"]
    assert (len(trace["input"]), len(trace["output"])) == (4096, 1024)
    stop = json.loads((TARGET / "generation_config.json").read_text())["eos_token_id"]
    stop = (stop,) if isinstance(stop, int) else tuple(stop)
    before = pages()
    with metal_job(64, 2) as budget:
        model = modeling.Model(TARGET, mtp=True, budget=budget, **OPTIMIZED_MODEL)
        warm_verification_graphs(model, budget)
        model.short_draft_head = prepare_short_head(model, budget)
        with patch.object(cli, "speculative", compare(trace, budget)):
            actual, timing = cli.generate(model, trace["input"], 1024, 512, stop=stop, budget=budget,
                                          draft_schedule="frontier_prefetch", **OPTIMIZED_DRAFT)
        assert actual == trace["output"] and timing["draft_schedule"] == "frontier_prefetch"
        print("host", json.dumps({"prefill_peak_gib": timing["prefill_peak_gib"],
                                  "faults": {k: v - before[k] for k, v in pages().items()},
                                  **budget.metrics()}), flush=True)


if __name__ == "__main__":
    main()
