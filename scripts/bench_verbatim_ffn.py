"""Compare compressed and verbatim BF16 FFN packets under sustained verification."""

from collections import defaultdict
from copy import copy
from dataclasses import replace
from functools import lru_cache
import inspect
import json
from statistics import median
from time import perf_counter

import mlx.core as mx

from ultrarun import lossless_gate_up as gates, lossless_projection as down
from ultrarun.host import thermal_state
from ultrarun.memory import GIB, metal_job
from ultrarun.model import Model
from ultrarun.packed_projection import pack_activations
from ultrarun.profiles import OPTIMIZED_MODEL
from ultrarun.server import warm_verification_graphs
from ultrarun.spec_tree import VerificationTree
from scripts.bench_dflash2 import pages
from scripts.bench_ready import ROOT, TARGET, digest
from scripts.bench_verify_buffer_limits import partition_prompt


LAYERS = (0, 9, 23, 48)
WIDTHS = (8, 16)
FFN = ("ffn_in", "mlp.down_proj.weight")


def verbatim_gate(weight):
    raw = gates.unpack_joint(weight.payload)
    blocks, packets, words = weight.payload[-1].shape
    count = blocks * packets
    escaped = mx.contiguous(raw.reshape(2, count, 4, 32, 2).transpose(1, 3, 0, 2, 4))
    escaped = escaped.reshape(count, 32, 16).view(mx.uint32)
    ids = mx.arange(count, dtype=mx.uint32)
    if words == 1:
        assert count <= 1 << gates.PAIRED_INDEX_BITS
        desc = (ids | (63 << 20)).reshape(blocks, packets, 1)
    else:
        desc = mx.stack((mx.full((count,), 255, dtype=mx.uint32), ids), axis=1).reshape(blocks, packets, 2)
    # Preserve the production packet shape and Metal device address space.
    digits, exponents = (mx.zeros_like(t[:1]) for t in weight.payload[:2])
    result = gates.LosslessPaired((digits, exponents, escaped, desc))
    assert mx.array_equal(gates.unpack_joint(result.payload), raw).item()
    return result


def verbatim_down(weight):
    raw = down.decoded_bits(weight.payload)
    blocks, packets = weight.payload[-1].shape
    count = blocks * packets
    escaped = mx.contiguous(raw.reshape(count, 4, 32, 2).transpose(0, 2, 1, 3))
    escaped = escaped.reshape(count, 32, 8).view(mx.uint32)
    desc = ((mx.arange(count, dtype=mx.uint32) << 8) | 255).reshape(blocks, packets)
    digits, exponents = (mx.zeros_like(t[:1]) for t in weight.payload[:2])
    result = down.LosslessDown((digits, exponents, escaped, desc))
    assert mx.array_equal(down.decoded_bits(result.payload), raw).item()
    return result


def replace_once(text, old, new):
    assert text.count(old) == 1, old
    return text.replace(old, new)


@lru_cache
def direct_kernels():
    # Change only the loader in the existing kernels. Keep FP32 MMA and reduction order.
    src = inspect.getsource(gates.paired_kernel)
    src = replace_once(src, "    return mx.fast.metal_kernel(", r'''
    packet_load = r"""
            uint address = (output_tile * packets + packet) * 32 + lane;
            uint4 gate_pairs = fallback[2 * address];
            uint4 up_pairs = fallback[2 * address + 1];
    """
    kernel_name += "_direct_bf16"
    return mx.fast.metal_kernel(''')
    gate_scope = vars(gates).copy()
    exec(compile(src, __file__, "exec"), gate_scope)

    src = inspect.getsource(down.project_kernel)
    src = src.replace('name="local_lossless_packet4_mma"', 'name="local_direct_bf16_packet4_mma"')
    src = replace_once(src, '''decode_packet(m, e, fallback, descriptor[output_tile * packets + packet],
                                        lane, TOKENS > 8 && TOKENS <= 16)''',
                       "fallback[(output_tile * packets + packet) * 32 + lane]")
    down_scope = vars(down).copy()
    exec(compile(src, __file__, "exec"), down_scope)
    return gate_scope["paired_kernel"], down_scope["project_kernel"]


def direct_gate(weight, packed, width):
    kernel, _ = direct_kernels()
    blocks, packets, words = weight.payload[-1].shape
    splits = (packets + 15) // 16
    threads = splits * 32
    result, = kernel("swiglu", width > 8, words == 1)(
        inputs=[packed, *weight.payload], template=[("TOKENS", width), ("SPLITS", splits)],
        grid=(threads * blocks, 1, 1), threadgroup=(threads, 1, 1),
        output_shapes=[(((width + 7) // 8) * blocks, 64)], output_dtypes=[mx.bfloat16],
    )
    return result


def direct_down(weight, packed, width):
    _, kernel = direct_kernels()
    blocks, packets = weight.payload[-1].shape
    splits = (packets + 15) // 16
    result, = kernel()(
        inputs=[packed, *weight.payload], template=[("TOKENS", width)],
        grid=(((blocks + 7) // 8) * 256, splits, 1), threadgroup=(256, 1, 1),
        output_shapes=[(splits, width, blocks * 8)], output_dtypes=[mx.float32],
    )
    return result


def micro(weights, arrays, budget):
    results = {}
    for width in WIDTHS:
        packed = arrays[width]
        modes = ("compressed", "verbatim_packets", "direct_bf16")

        def chain(mode):
            values = []
            for original, verbatim in weights:
                g, d = original if mode == "compressed" else verbatim
                if mode == "direct_bf16":
                    mid = direct_gate(g, packed, width)
                    partial = direct_down(d, mid, width)
                else:
                    mid = g.project(packed, width, swiglu=True)
                    partial = d.partials(mid, width)
                values.append((mid, partial))
            return values

        reference = chain("compressed")
        for name in modes[1:]:
            for want, got in zip(reference, chain(name), strict=True):
                assert mx.array_equal(want[0].view(mx.uint16), got[0].view(mx.uint16)).item()
                assert mx.array_equal(want[1].view(mx.uint32), got[1].view(mx.uint32)).item()
        del reference
        times = defaultdict(list)
        for i in range(12):
            pivot = i % len(modes)
            for mode in modes[pivot:] + modes[:pivot]:
                t = perf_counter()
                mx.eval(chain(mode))
                times[mode].append(1000 * (perf_counter() - t))
            budget.check()
        results[width] = {mode: median(vals) for mode, vals in times.items()}
    return results


def main():
    trace = json.loads((ROOT / "runs/frontier-target-margin-audit.json").read_text())["code"]
    before = pages()
    mx.random.seed(184)
    with metal_job(80, 2) as budget:
        model = Model(TARGET, mtp=True, budget=budget, **OPTIMIZED_MODEL)
        warm_verification_graphs(model, budget)
        work = []
        alternate = copy(model)
        alternate.layers = list(model.layers)
        counts = [0, 0]
        for index in LAYERS:
            layer = copy(model.layers[index])
            layer.w = dict(layer.w)
            alternate.layers[index] = layer
            source = tuple(layer.w[name] for name in FFN)
            budget.check(force=True, required_bytes=2 * GIB)
            raw = (verbatim_gate(source[0]), verbatim_down(source[1]))
            mx.eval([w.payload for w in raw])
            for i, weights in enumerate((source, raw)):
                counts[i] += sum(a.nbytes for w in weights for a in w.payload)
            layer.w.update(zip(FFN, raw, strict=True))
            work.append((source, raw))
            mx.clear_cache()
        print("weights", json.dumps({"layers": LAYERS, "compressed_bytes": counts[0], "verbatim_bytes": counts[1]}), flush=True)
        arrays = {width: pack_activations(mx.random.normal((width, model.config["hidden_size"])).astype(mx.bfloat16))
                  for width in WIDTHS}
        mx.eval(arrays)
        print("micro early", json.dumps({"ms": micro(work, arrays, budget), "thermal": thermal_state(),
              "peak_gib": mx.get_peak_memory() / GIB, **budget.metrics()}), flush=True)

        ids, output = trace["input"], trace["output"]
        cache = model.cache()
        budget.admit(model, 5120, 512, False)
        for _, hidden in model.prefill(mx.array(ids, dtype=mx.int32), cache, 512):
            model.evaluate(hidden, cache)
            budget.check()
        assert mx.argmax(model.logits(hidden[-1:]), axis=-1).item() == output[0]
        partition_prompt(model, cache, len(ids))
        fixed = {}
        for width, spine in ((8, 7), (16, 14)):
            parents = [-1, *range(spine - 1), *([1] if width == 8 else [1, spine])]
            fixed[width] = (mx.array(output[:width], dtype=mx.int32), VerificationTree(parents, len(ids), 3, spine=spine))

        def verify(verifier, width):
            states = [replace(s) for s in cache]
            tokens, tree = fixed[width]
            hidden = verifier.hidden(tokens, states, len(ids), keep_replay=True, tree=tree)
            predictions = verifier.predictions(hidden)
            verifier.evaluate([hidden, predictions], states)
            return hidden, predictions

        ref = {width: verify(model, width) for width in WIDTHS}
        for width, (h, t) in ref.items():
            ah, at = verify(alternate, width)
            assert mx.array_equal(h.view(mx.uint16), ah.view(mx.uint16)).item()
            assert mx.array_equal(t, at).item()
        print("fixed_exact", json.dumps({width: digest(out.tolist()) for width, (_, out) in ref.items()}), flush=True)

        started = perf_counter()
        rounds = 0
        while perf_counter() - started < 17:
            verify(model, 8)
            rounds += 1
            budget.check()
        print("sustained", rounds, thermal_state(), flush=True)

        trial = defaultdict(list)
        for i in range(10):
            for width in WIDTHS if i % 2 == 0 else WIDTHS[::-1]:
                pair = (model, alternate) if i % 2 == 0 else (alternate, model)
                times = {}
                for verifier in pair:
                    start = perf_counter()
                    verify(verifier, width)
                    times[verifier is alternate] = 1000 * (perf_counter() - start)
                trial[width].append(times)
                budget.check()
        print("fixed_hot", json.dumps({"ms": {width: {"compressed": median(t[False] for t in times),
              "verbatim_packets": median(t[True] for t in times), "paired_delta": median(t[True] - t[False] for t in times)}
              for width, times in trial.items()}, "thermal": thermal_state(), **budget.metrics()}), flush=True)
        print("micro late", json.dumps({"ms": micro(work, arrays, budget), "thermal": thermal_state(),
              "peak_gib": mx.get_peak_memory() / GIB, **budget.metrics()}), flush=True)
        after = pages()
        print("faults", json.dumps({key: after[key] - before[key] for key in before}), flush=True)


if __name__ == "__main__":
    main()
