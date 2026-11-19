"""Measure losslessly packed DSpine QKV, attention output, and short context projections."""

import argparse
import json

import mlx.core as mx

from ultrarun.host import thermal_state
from ultrarun.memory import GIB, metal_job
from scripts.bench_dflash2 import pages
from scripts.bench_dspine import (
    DSpineDraft, ROOT, verify_checkpoint, with_packed_ffn, with_packed_projections, with_prompt_vocabulary,
)
from scripts.bench_dspine_ffn import paired_decode, paired_time
from scripts.bench_ready import benchmark_target, prepare_prefix


def micro(drafts, budget):
    config = drafts[0].config
    inputs = len(config.target_layer_ids) * config.hidden_size
    for width in (1, 4, 8, 16, 32):
        features = mx.random.normal((1, width, inputs)).astype(mx.bfloat16)
        mx.eval(features)

        def context(arm):
            result = drafts[arm].context(features, 4096, budget)
            mx.eval(result)
            return [a for pair in result for a in pair]

        left, right = context(0), context(1)
        assert all(a.shape == b.shape and a.dtype == b.dtype == mx.bfloat16
                   for a, b in zip(left, right, strict=True))
        if width > 16:
            assert all(mx.array_equal(a.view(mx.uint16), b.view(mx.uint16)).item()
                       for a, b in zip(left, right, strict=True))
        print("short_context", json.dumps({"width": width, **paired_time(context)}), flush=True)
        budget.check(force=True)

    for width in (8, 16):
        hidden = mx.random.normal((1, width, config.hidden_size)).astype(mx.bfloat16)
        attention = mx.random.normal((1, config.num_attention_heads, width, config.head_dim)).astype(mx.bfloat16)
        mx.eval(hidden, attention)

        def project(arm):
            draft = drafts[arm]
            out = [(*draft.qkv(hidden, i, 4096), draft.attention_output(attention, i))
                   for i in range(config.num_hidden_layers)]
            mx.eval(out)

        print("attn_projection", json.dumps({"width": width, **paired_time(project)}), flush=True)
        budget.check(force=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--micro-only", action="store_true")
    mode.add_argument("--short-draft-vocab", action="store_true")
    args = parser.parse_args()
    verify_checkpoint()
    trace = json.loads((ROOT / "runs/frontier-target-margin-audit.json").read_text())["code"]
    assert len(trace["input"]) == 4096 and len(trace["output"]) == 1024
    mx.random.seed(503)
    thermal, before = thermal_state(), pages()
    with metal_job(10 if args.micro_only else 80, 2) as budget:
        draft = DSpineDraft()
        mx.eval(draft.weights, draft.code_projection, draft.sender)
        base = with_packed_ffn(draft, budget)
        del draft
        mx.clear_cache()
        candidate = with_packed_projections(base, budget)
        if args.short_draft_vocab:
            base = candidate
        else:
            micro((base, candidate), budget)
        if not args.micro_only:
            model = benchmark_target(budget)
            budget.admit(model, 5120, 512, False, extra_bytes=2 * GIB)
            labels = ("packed_ffn", "packed_projections")
            if args.short_draft_vocab:
                candidate = with_prompt_vocabulary(base, model, trace["input"])
                labels = ("packed_projections", "short_draft_vocab")
            drafts = base, candidate
            prepared = prepare_prefix(model, base, trace, budget)
            _, pending, prefix = prepared

            def proposal(arm):
                return drafts[arm].propose_tree(model, pending, prefix, 16)

            same = proposal(0) == proposal(1)
            print("draft_4k_prefix", json.dumps({"same_proposals": same, **paired_time(proposal)}), flush=True)
            paired_decode(model, drafts, trace, prepared, 16, True, budget, labels=labels)
        after = pages()
        print("host", json.dumps({"thermal": [thermal, thermal_state()],
              "faults": {k: after[k] - before[k] for k in before}, **budget.metrics()}), flush=True)


if __name__ == "__main__":
    main()
