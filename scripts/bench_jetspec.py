"""Measure serial JetSpec++ BF16 drafting with fresh, verified target context."""

import argparse
from copy import copy
import json
from pathlib import Path
from random import Random
from time import perf_counter

import mlx.core as mx
from mlx.utils import tree_flatten

from localllm.memory import GIB, metal_job
from localllm.packed_projection import pack_activations
from localllm.server import warm_verification_graphs
from scripts.bench_dflash2 import ROOT, decode, load_target, prefill, upstream_module


class ContextCache:
    """Store verified context only for the fixed 4K/1K benchmark."""

    def __init__(self):
        self.offset = 0
        self.keys = self.values = None

    def update_and_fetch(self, keys, values):
        end = self.offset + keys.shape[2]
        if end > 5120:
            raise ValueError("JetSpec++ cache exceeds the benchmark's 5120-token context")
        if self.keys is not None:
            keys = mx.concatenate((self.keys, keys), axis=2)
            values = mx.concatenate((self.values, values), axis=2)
        self.keys, self.values, self.offset = keys, values, end
        return keys, values


def causal_mask(length, offset):
    return mx.arange(length + offset)[None, :] <= offset + mx.arange(length)[:, None]


def load_jetspec(upstream_path, checkpoint):
    if (checkpoint / "model.safetensors").stat().st_size != 3_460_432_504:
        raise ValueError("Expected the original JetSpec++ BF16 checkpoint")
    config = json.loads((checkpoint / "config.json").read_bytes())
    spec = config["dflash_config"]
    assert spec["causal_head"] and spec["stock_drafter"] and spec["drafter_mlp"]
    assert spec["noise_source"] == "context" and spec["jacobi_forcing"]
    assert "inject_layer_ids" not in spec and config["block_size"] == 16
    upstream = upstream_module(upstream_path, checkpoint)
    upstream.KVCache = ContextCache
    upstream.create_causal_mask = causal_mask
    model = upstream.load_draft(str(checkpoint))
    assert type(model) is upstream.DFlashDraftModel
    assert not model.config.sliding_window and set(model.config.layer_types) == {"full_attention"}
    assert all(weight.dtype == mx.bfloat16 for _, weight in tree_flatten(model.parameters()))
    for layer in model.layers:
        layer.self_attn.is_causal = True
    return model


class RefiningDraft:
    def __init__(self, model, predict, prompt, refinements, seed):
        self.model = model
        self.predict = predict
        self.config = model.config
        self.context = list(prompt)
        self.refinements = refinements
        self.seed = seed
        self.previous = []

    def make_cache(self):
        return self.model.make_cache()

    def propose(self, anchors, features, cache, temperature, logits_start):
        assert temperature == 0 and logits_start == 1
        if self.previous:
            self.context.extend(self.previous[:features.shape[1]])
        assert len(self.context) == cache[0].offset + features.shape[1]
        anchor = anchors[0, 0].item()
        count = anchors.shape[1] - 1
        noise = Random(self.seed)
        end = len(self.context)
        indices = [noise.randrange(end + 1) for _ in range(count)]
        draft_tokens = [anchor if i == end else self.context[i] for i in indices]
        inputs = mx.array([[anchor, *draft_tokens]], dtype=mx.int32)
        for iteration in range(self.refinements + 1):
            # Each iteration sees the same verified prefix. Keep only its newest cache view.
            latest = [copy(state) for state in cache]
            hidden = self.model.hidden_states(inputs, features, latest, logits_start=1)
            proposal = self.predict(hidden)
            mx.eval(proposal)
            if iteration < self.refinements:
                if mx.array_equal(proposal, inputs[:, 1:]).item():
                    break
                inputs = mx.concatenate((anchors[:, :1], proposal.astype(mx.int32)), axis=1)
        cache[:] = latest
        self.previous = [anchor, *proposal[0].tolist()]
        return proposal, None, None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upstream", type=Path, required=True)
    parser.add_argument("--case", default="code")
    parser.add_argument("--refinements", type=int, nargs="+", choices=(0, 1, 3), default=[0, 1])
    args = parser.parse_args()
    trace = json.loads((ROOT / "runs/frontier-target-margin-audit.json").read_bytes())[args.case]
    assert len(trace["input"]) == 4096 and len(trace["output"]) == 1024
    with metal_job(80, 2) as budget:
        target = load_target(budget)
        budget.admit(target, 5120, 512, False, extra_bytes=6 * GIB)
        draft = load_jetspec(args.upstream, ROOT / "weights/Qwen3.8-27B-JetSpecPP")
        assert draft.config.hidden_size == target.embedding.shape[1] and draft.config.vocab_size == target.vocab_size
        assert draft.config.input_embedding_scale == draft.config.output_multiplier == 1.0
        assert draft.config.final_logit_softcapping is None
        draft.embed_tokens = lambda ids: target.embedding[ids]

        def predict(hidden):
            return target.vocabulary_order.argmax(target.packed_head, pack_activations(hidden[0]), hidden.shape[1])[None]

        budget.admit(target, 5120, 512, False, extra_bytes=2 * GIB)
        warm_verification_graphs(target, budget)
        mx.reset_peak_memory()
        started = perf_counter()
        cache, first, features = prefill(target, draft, trace["input"])
        assert first == trace["output"][0]
        assert features.shape[1] == 4096
        prefill_s = perf_counter() - started
        prefill_peak = mx.get_peak_memory() / GIB
        print("prefill", args.case, prefill_s, prefill_peak, flush=True)
        for refinements in args.refinements:
            wrapper = RefiningDraft(draft, predict, trace["input"], refinements, seed=38)
            mx.clear_cache()
            mx.reset_peak_memory()
            result = decode(target, wrapper, trace, 16, [copy(state) for state in cache], first, features, budget)
            result.update(refinements=refinements, context_adapter=False, prefill_s=prefill_s,
                          prefill_peak_gib=prefill_peak)
            print("jetspec_result", args.case, json.dumps(result), flush=True)
        budget.check(force=True)


if __name__ == "__main__":
    main()
