"""Attribute hot BF16 verifier slowdown using exact, synchronized layer stages."""

import argparse
from collections import defaultdict
from dataclasses import replace
import inspect
import json
from statistics import median
from time import perf_counter, sleep
from unittest.mock import patch

import mlx.core as mx

from ultrarun import cli, model as modeling
from ultrarun.attention_io import pack_gated_attention
from ultrarun.draft_short_head import prepare_short_head
from ultrarun.draft_vocab_prefetch import DraftVocabPrefetch
from ultrarun.host import thermal_state
from ultrarun.memory import GIB, metal_job
from ultrarun.packed_projection import residual_norm_pack
from ultrarun.profiles import OPTIMIZED_DRAFT, OPTIMIZED_MODEL
from ultrarun.server import warm_verification_graphs
from ultrarun.spec_tree import VerificationTree
from scripts.bench_dflash2 import pages
from scripts.bench_ready import ROOT, TARGET, digest

from down_read_decode_cost import scan
from full_target_weight_stream import materialize_catalog


WIDTHS = (8, 16)


def attention_prefix(x, attention, gate, out_weight, norm, epsilon, _gate_up, _down):
    packed = pack_gated_attention(attention, gate)
    output = modeling.project_attention_output(packed, x.shape[0], out_weight)
    return residual_norm_pack(x, output, norm, epsilon)


def compile_attention_prefix():
    # Build a separate compiled graph with the same cache writes, Q/K rotation,
    # attention reduction and residual normalization, ending before the FFN.
    source = inspect.getsource(modeling.compile_reserved_attention)
    scope = {**vars(modeling), "finish_attention_projections": attention_prefix}
    exec(compile(source, __file__, "exec"), scope)
    return scope["compile_reserved_attention"]


def same_bits(left, right):
    if left is None or right is None:
        return left is right
    if left.shape != right.shape or left.dtype != right.dtype:
        return False
    if left.dtype in (mx.bfloat16, mx.float32):
        dtype = mx.uint16 if left.dtype == mx.bfloat16 else mx.uint32
        left, right = left.view(dtype), right.view(dtype)
    return mx.array_equal(left, right).item()


class Fixture:
    def __init__(self, model, prefix, trace, width, compile_prefix):
        self.model = model
        self.prefix = [replace(state) for state in prefix]
        self.width, self.compile_prefix = width, compile_prefix
        self.offset = len(trace["input"])
        branches = 1 if width == 8 else 2
        spine = width - branches
        self.tree = VerificationTree([-1, *range(spine - 1), *range(branches)], self.offset,
                                     model.config["linear_conv_kernel_dim"] - 1, spine=spine)
        self.ids = mx.array(trace["output"][:width], dtype=mx.int32)
        self.hints = DraftVocabPrefetch(model, trace["input"], (), 128, {})
        self.use_path = list(self.tree.paths[spine])
        self.frozen = []
        self.final = None
        mx.eval(self.ids)

    def score(self, hidden, states):
        prediction = self.model.predictions(hidden, observe_vocabulary_tiles=self.hints.capture)
        self.model.evaluate([hidden, prediction, *self.hints.hints], states)
        info, self.hints.hints = self.hints.hints, None
        return prediction, info

    def full(self):
        states = [replace(state) for state in self.prefix]
        started = perf_counter()
        x = self.model.hidden(self.ids, states, self.offset, keep_replay=True, tree=self.tree)
        y, hints = self.score(x, states)
        return (x, y, hints, states), (perf_counter() - started) * 1000

    def mixer(self, layer, state, hidden):
        if not layer.gdn:
            return layer.reserved_attention(hidden, state, self.offset, self.tree)
        mixed = layer.attention_result(hidden, state, self.offset, True, self.tree)
        return residual_norm_pack(hidden, mixed, layer.w["post_attention_layernorm.weight"],
                                  layer.residual_norm_epsilon)

    def staged(self, *, freeze=False):
        states = [replace(state) for state in self.prefix]
        phases = defaultdict(float)
        started = perf_counter()
        x = self.model.embedding[self.ids]
        mx.eval(x)
        phases["embedding"] = (perf_counter() - started) * 1000
        with patch.object(modeling, "compile_reserved_attention", self.compile_prefix):
            for layer, state in zip(self.model.layers, states, strict=True):
                began = perf_counter()
                incoming = x
                residual, packed = self.mixer(layer, state, incoming)
                phase = "delta_and_norm" if layer.gdn else "attention_and_norm"
                self.model.evaluate([residual, packed], [state])
                mixed_at = perf_counter()
                gate = layer.w["ffn_in"].project(packed, self.width, swiglu=True)
                mx.eval(gate)
                gate_at = perf_counter()
                down = layer.w["mlp.down_proj.weight"].partials(gate, self.width)
                x = residual + down.sum(axis=0).astype(residual.dtype)
                mx.eval(x)
                finished = perf_counter()
                if freeze:
                    self.frozen.append((incoming, residual, packed, gate))
                phases[phase] += (mixed_at - began) * 1000
                phases["ffn_gate_up"] += (gate_at - mixed_at) * 1000
                phases["ffn_down_reduce"] += (finished - gate_at) * 1000
        begin = perf_counter()
        y, hints = self.score(x, states)
        if freeze:
            self.final = x
        phases["vocab_and_hints"] = (perf_counter() - begin) * 1000
        phases["staged_total"] = (perf_counter() - started) * 1000
        return (x, y, hints, states), dict(phases)

    def grouped_stages(self, order):
        """Replay each stage on all original inputs, synchronizing once per group."""
        layers = self.model.layers
        assert len(self.frozen) == len(layers)

        def mix(is_delta):
            arrays, states = [], []
            with patch.object(modeling, "compile_reserved_attention", self.compile_prefix):
                for layer, before, (hidden, *_rest) in zip(layers, self.prefix, self.frozen, strict=True):
                    if layer.gdn == is_delta:
                        state = replace(before)
                        arrays.extend(self.mixer(layer, state, hidden))
                        states.append(state)
            self.model.evaluate(arrays, states)

        def gate_up():
            mx.eval([layer.w["ffn_in"].project(packed, self.width, swiglu=True)
                     for layer, (_, _, packed, _) in zip(layers, self.frozen, strict=True)])

        def down_reduce():
            mx.eval([residual + layer.w["mlp.down_proj.weight"].partials(gate, self.width).sum(axis=0).astype(residual.dtype)
                     for layer, (_, residual, _, gate) in zip(layers, self.frozen, strict=True)])

        methods = {
            "delta_and_norm": lambda: mix(True), "attention_and_norm": lambda: mix(False),
            "ffn_gate_up": gate_up, "ffn_down_reduce": down_reduce,
            "vocab_and_hints": lambda: self.score(self.final, []),
        }
        names = tuple(methods)
        shift = order % len(names)
        times = {}
        for label in names[shift:] + names[:shift]:
            started = perf_counter()
            methods[label]()
            times[label] = 1000 * (perf_counter() - started)
        return times

    def parity(self, full, staged):
        x, y, h, original = full
        a, b, g, other = staged
        assert same_bits(x, a) and same_bits(y, b), (self.width, "hidden or prediction")
        assert all(same_bits(u, v) for u, v in zip(h, g, strict=True)), (self.width, "vocabulary hints")
        for states in (original, other):
            self.model.accept_tree(states, self.tree, self.offset, self.use_path)
            self.model.evaluate([], states)
        through = self.offset + len(self.use_path)
        for i, (l, r) in enumerate(zip(original, other, strict=True)):
            for field in ("keys", "values", "conv", "delta"):
                left, right = getattr(l, field), getattr(r, field)
                if left is not None and field in ("keys", "values") and l.prompt_kv is not None:
                    used = through - l.prompt_kv[0].shape[2]
                    left, right = left[:, :, :used], right[:, :, :used]
                assert same_bits(left, right), (self.width, i, field)
        return digest(y.tolist())


def scan_weights(catalog):
    started = perf_counter()
    mx.eval(tuple(scan(array) for group in catalog.values() for array in group))
    return (perf_counter() - started) * 1000


def sample(label, fixtures, catalog, budget):
    raw, readings, groups = [], {w: defaultdict(list) for w in fixtures}, {w: defaultdict(list) for w in fixtures}
    routes = {}
    for i in range(4):
        for width in (WIDTHS if i % 2 == 0 else WIDTHS[::-1]):
            fixture = fixtures[width]
            full, normal = fixture.full()
            stages, parts = fixture.staged()
            if i == 0:
                routes[width] = fixture.parity(full, stages)
            readings[width]["normal"].append(normal)
            for name, value in parts.items():
                readings[width][name].append(value)
            if fixture.frozen:
                for name, value in fixture.grouped_stages(i).items():
                    groups[width][name].append(value)
            del full, stages
        raw.append(scan_weights(catalog))
        budget.check()
    report = {w: {name: median(values) for name, values in row.items()} for w, row in readings.items()}
    print("phase", json.dumps({
        "name": label, "thermal": thermal_state(), "weights_scan_ms": median(raw), "median_ms": report,
        "grouped_median_ms": {w: {name: median(values) for name, values in group.items()} for w, group in groups.items()},
        "prediction_digests": routes, "peak_gib": mx.get_peak_memory() / GIB, **budget.metrics(),
    }), flush=True)
    return report, routes


def request_hook(trace, native, budget, stress_seconds):
    def run(model, cache, mtp, *args, **kwargs):
        assert sum(state.prompt_kv is not None for state in cache) == 16
        compile_prefix = compile_attention_prefix()
        fixtures = {w: Fixture(model, cache, trace, w, compile_prefix) for w in WIDTHS}
        catalog = materialize_catalog(model)
        print("catalog_bytes", json.dumps({name: sum(t.nbytes for t in arrays)
                                           for name, arrays in catalog.items()}), flush=True)
        # Warm both execution routes before cooling to avoid charging compilation.
        for fixture in fixtures.values():
            fixture.parity(fixture.full()[0], fixture.staged(freeze=bool(stress_seconds))[0])
            if fixture.frozen:
                fixture.grouped_stages(0)
        scan_weights(catalog)
        sleep(20)
        early, first = sample("before", fixtures, catalog, budget)
        t0, before = perf_counter(), pages()
        actual, metrics = native(model, cache, mtp, *args, **kwargs)
        seconds = perf_counter() - t0
        assert actual == trace["output"]
        after = pages()
        print("live", json.dumps({
            "passes": metrics["target_decode_passes"], "decode_tok_s": (len(actual) - 1) / seconds,
            "decoded": len(actual), "exact": True, "target_s": metrics["target_s"],
            "thermal": thermal_state(), "faults": {k: after[k] - val for k, val in before.items()},
        }), flush=True)
        # Saved lightweight layer-state copies retain the original prompt cache.
        if stress_seconds:
            until = perf_counter() + stress_seconds
            n = 0
            times = []
            baseline = early[16]["normal"]
            while perf_counter() < until:
                _, duration = fixtures[16].full()
                times.append(duration)
                n += 1
                budget.check()
                if len(times) == 24:
                    current = median(times)
                    print("stress", json.dumps({"fixed_width16_passes": n, "median_ms": current,
                                                "slowdown": current / baseline,
                                                "thermal": thermal_state()}), flush=True)
                    times.clear()
                    if n >= 96 and current >= 1.35 * baseline:
                        break
        late, second = sample("after", fixtures, catalog, budget)
        sleep(20)
        _, third = sample("after_idle", fixtures, catalog, budget)
        assert first == second == third
        print("changes_ms", json.dumps({
            w: {name: late[w][name] - early[w][name] for name in early[w]}
            for w in WIDTHS
        }), flush=True)
        return actual, metrics
    return run


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workload", choices=("code", "chinese_technical_report", "english_technical_tutorial"),
                        default="code")
    parser.add_argument("--stress-seconds", type=int, choices=(0, 30, 45), default=0)
    args = parser.parse_args()
    trace = json.loads((ROOT / "runs/frontier-target-margin-audit.json").read_text())[args.workload]
    assert len(trace["input"]) == 4096 and len(trace["output"]) == 1024
    eos = json.loads((TARGET / "generation_config.json").read_text())["eos_token_id"]
    stop = (eos,) if isinstance(eos, int) else tuple(eos)
    before = pages()
    with metal_job(80, 2) as budget:
        model = modeling.Model(TARGET, mtp=True, budget=budget, **OPTIMIZED_MODEL)
        warm_verification_graphs(model, budget)
        model.short_draft_head = prepare_short_head(model, budget)
        native = cli.speculative
        with patch.object(cli, "speculative", request_hook(trace, native, budget, args.stress_seconds)):
            actual, timing = cli.generate(model, trace["input"], 1024, 512, stop=stop, budget=budget,
                                          draft_schedule="frontier_prefetch", **OPTIMIZED_DRAFT)
        assert actual == trace["output"] and timing["draft_schedule"] == "frontier_prefetch"
        after = pages()
        print("host", json.dumps({
            "prefill_peak_gib": timing["prefill_peak_gib"],
            "faults": {key: after[key] - total for key, total in before.items()}, **budget.metrics(),
        }), flush=True)


if __name__ == "__main__":
    main()
