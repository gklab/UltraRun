"""Measure Tencent's BF16 DSpine draft against the native BF16 verifier."""

import argparse
from copy import copy
import hashlib
import heapq
import json
from math import exp, sqrt
from types import SimpleNamespace

import mlx.core as mx
from mlx import nn

from localllm.lossless_gate_up import LosslessPaired, encode_paired, unpack_joint
from localllm.lossless_projection import decoded_bits, encode_down
from localllm.memory import GIB, metal_job
from localllm.packed_projection import pack_activations, pack_weight
from scripts.bench_ready import DraftContext, ROOT, benchmark_target, decode, linear, prepare_prefix


DRAFT = ROOT / "weights/Qwen3.8-27B-DSpine-b16"
CHECKSUMS = {
    "config.json": "c2aa9e3ff22766ea7df0fa5b5f5f7d2035ae722671c6b7eb58b65578705e6815",
    "model.safetensors": "ac90f1df842ea66b23e4224f3e86624add547526f689ffe9e9c6b7a5937116b1",
}
BRIDGE = "selected_prefix_bridge."
VERSION_KEYS = {
    "draft_block_causal_version", "last_write_refine_version",
    BRIDGE + "independent_write_version", BRIDGE + "terminal_disabled_version",
}


def verify_checkpoint():
    for file, expected in CHECKSUMS.items():
        with (DRAFT / file).open("rb") as stream:
            if hashlib.file_digest(stream, "sha256").hexdigest() != expected:
                raise ValueError(f"DSpine checkpoint checksum mismatch for {file}")


def top16(logits):
    # A chunk outside the top 16 maxima cannot supply any of the top 16 tokens.
    rows = logits.shape[0]
    padding = -logits.shape[1] % 64
    if padding:
        logits = mx.pad(logits, [(0, 0), (0, padding)], constant_values=-float("inf"))
    chunks = logits.reshape(rows, -1, 64)
    maxima = mx.max(chunks, axis=-1).astype(mx.float32)
    blocks = mx.sort(mx.argsort(-maxima, axis=-1)[:, :16], axis=-1)
    ids = (blocks[..., None] * 64 + mx.arange(64)).reshape(rows, -1)
    scores = mx.take_along_axis(logits, ids, axis=-1).astype(mx.float32)
    picks = mx.argsort(-scores, axis=-1)[:, :16]
    return mx.sort(mx.take_along_axis(ids, picks, axis=-1), axis=-1).astype(mx.int32)


def paired_rows(weight, x):
    return weight.project(pack_activations(x[0]), x.shape[1])[None]


class DSpineDraft(DraftContext):
    normalize = staticmethod(mx.fast.rms_norm)

    def __init__(self):
        config = self.config = SimpleNamespace(**json.loads((DRAFT / "config.json").read_text()))
        config.target_layer_ids = config.dflash_config["target_layer_ids"]
        config.mask_token_id = config.dflash_config["mask_token_id"]
        self.weights = mx.load(str(DRAFT / "model.safetensors"))
        integral = {key for key, arr in self.weights.items() if arr.dtype == mx.int64}
        assert integral == VERSION_KEYS and len(self.weights) == 71
        assert all(self.weights[key].item() == 1 for key in VERSION_KEYS)
        assert all(arr.dtype == mx.bfloat16 for key, arr in self.weights.items() if key not in VERSION_KEYS)
        assert not mx.any(self.bridge("out.weight") != 0).item()
        dims = config.dflash_config["selected_prefix_code_bridge"]
        self.eps = dims["eps"]
        self.scale = sqrt(dims["code_size"])
        self.sender = mx.concatenate((self.bridge("predecessor_gate.weight"), self.bridge("prev.weight")))
        self.code_projection = mx.concatenate((self.bridge("gate.weight"), self.sender))
        self.checked_widths = set()
        self.packed_ffn = None
        self.packed_attention = None
        self.packed_context = None
        self.proposal_vocab = None
        self.wide_cost_ratio = None
        self.context_window = None

    def bridge(self, name):
        return self.weights[BRIDGE + name]

    def rotate(self, words, offset):
        return mx.fast.rope(words, self.config.head_dim, traditional=False,
                            base=self.config.rope_theta, scale=1.0, offset=offset)

    def token_code(self, ids):
        return (self.scale * self.bridge("codebook")[ids].astype(mx.float32)).astype(mx.bfloat16)

    def context_features(self, features):
        rows = features.shape[1]
        if self.packed_context is None or rows > 16:
            return super().context_features(features)
        packed = pack_activations(features[0])
        projected = self.packed_context.partials(packed, rows).sum(axis=0).astype(mx.bfloat16)[None]
        return self.normalize(projected, self.weights["hidden_norm.weight"], self.config.rms_norm_eps)

    def context_kv(self, x, index, offset):
        if self.packed_attention is None or x.shape[1] > 16:
            return super().context_kv(x, index, offset)
        key, value = mx.split(paired_rows(self.packed_attention[index][1], x), 2, axis=-1)
        return self.format_kv(key, value, self.layer(index), offset)

    def qkv(self, h, index, offset):
        config, layer = self.config, self.layer(index)
        if self.packed_attention is None:
            q = linear(h, layer("self_attn.q_proj"))
            k, v = self.kv(h, layer, offset)
        else:
            qkv = paired_rows(self.packed_attention[index][0], h)
            half_q = config.num_attention_heads * config.head_dim // 2
            kv = config.num_key_value_heads * config.head_dim
            left, key, right, value = mx.split(qkv, [half_q, half_q + kv, 2 * half_q + kv], axis=-1)
            q = mx.concatenate((left, right), axis=-1)
            k, v = self.format_kv(key, value, layer, offset)
        q = q.reshape(1, h.shape[1], config.num_attention_heads, config.head_dim)
        q = self.normalize(q, layer("self_attn.q_norm"), config.rms_norm_eps).transpose(0, 2, 1, 3)
        return self.rotate(q, offset), k, v

    def attention_output(self, attention, index):
        x = attention.transpose(0, 2, 1, 3).reshape(1, attention.shape[2], -1)
        if self.packed_attention is not None:
            return paired_rows(self.packed_attention[index][2], x)
        return linear(x, self.layer(index)("self_attn.o_proj"))

    def feed_forward(self, hidden, index):
        if self.packed_ffn is not None:
            gate_up, down = self.packed_ffn[index]
            rows = hidden.shape[1]
            activated = gate_up.project(pack_activations(hidden[0]), rows, swiglu=True)
            return down.partials(activated, rows).sum(axis=0).astype(mx.bfloat16)[None]
        weights = self.layer(index)
        gate = nn.silu(linear(hidden, weights("mlp.gate_proj")))
        return linear(gate * linear(hidden, weights("mlp.up_proj")), weights("mlp.down_proj"))

    def read_codes(self, hidden, layer):
        reader = (self.bridge("read_weight") if layer == self.config.num_hidden_layers - 1
                  else self.bridge("early_read_weights")[layer])
        raw = hidden.astype(mx.float32)
        rms = mx.sqrt(mx.maximum(mx.mean(mx.square(raw), axis=-1, keepdims=True), 2**-126))
        normalized = (raw / (rms + self.eps)).astype(mx.bfloat16)
        projected = linear(normalized, reader).astype(mx.float32)
        length = mx.sqrt(mx.sum(mx.square(projected), axis=-1, keepdims=True))
        estimates = (self.scale * projected / (length + self.eps)).astype(mx.bfloat16)
        return raw, rms, estimates

    def write(self, hidden, anchor_code, layer, forced_senders=None):
        raw, rms, estimates = self.read_codes(hidden, layer)
        codes = mx.concatenate((anchor_code, estimates[:, 1:]), axis=1)
        packed = linear(codes, self.code_projection).astype(mx.float32)
        receiver, sender, values = mx.split(packed, 3, axis=-1)
        if forced_senders is not None:
            count = forced_senders.shape[1]
            gate, value = mx.split(forced_senders, 2, axis=-1)
            sender = mx.concatenate((sender[:, :1], gate, sender[:, count + 1:]), axis=1)
            values = mx.concatenate((values[:, :1], value, values[:, count + 1:]), axis=1)
        bias = self.bridge("gate.bias").astype(mx.float32)
        adjacent = mx.sigmoid(receiver[:, 1:] + sender[:, :-1] + bias) * values[:, :-1]
        messages = mx.concatenate((mx.zeros_like(receiver[:, :1]), adjacent), axis=1).astype(mx.bfloat16)
        delta = linear(messages, self.bridge("feature_write_weights")[layer]).astype(mx.float32)
        out = mx.concatenate((hidden[:, :1], (raw[:, 1:] + rms[:, 1:] * delta[:, 1:]).astype(mx.bfloat16)), axis=1)
        return out, (raw[0, 1:], rms[0, 1:], receiver[0, 1:])

    def hidden(self, model, pending, prefix, width, *, early_depth=None, backbone_depth=None,
               seeds=(), seed_embeddings=False):
        config = self.config
        depth = config.num_hidden_layers if backbone_depth is None else backbone_depth
        embedded = seeds if seed_embeddings else ()
        ids = mx.array([[pending, *embedded, *([config.mask_token_id] * (width - 1 - len(embedded)))]],
                       dtype=mx.int32)
        residual = model.embedding[ids]
        anchor_code = self.token_code(ids[:, :1])
        forced_senders = None
        if seeds:
            if len(seeds) >= width - 1:
                raise ValueError("At least one draft position must follow the supplied seed")
            codes = self.token_code(mx.array([seeds], dtype=mx.int32))
            forced_senders = linear(codes, self.sender).astype(mx.float32)
        offset = prefix[0][0].shape[2]
        # The verifier's 4K/1K workload fits entirely inside the draft window.
        assert offset + width <= config.sliding_window
        h = self.normalize(residual, self.layer(0)("input_layernorm"), config.rms_norm_eps)
        for i, (known_k, known_v) in enumerate(prefix[:depth]):
            weights = self.layer(i)
            q, k, v = self.qkv(h, i, offset)
            keys = draft_attention_input(known_k, k, self.context_window)
            values = draft_attention_input(known_v, v, self.context_window)
            attention = mx.fast.scaled_dot_product_attention(
                q, keys, values, scale=config.head_dim**-0.5, mask="causal",
            )
            residual = residual + self.attention_output(attention, i)
            h = self.normalize(residual, weights("post_attention_layernorm"), config.rms_norm_eps)
            out = residual + self.feed_forward(h, i)
            if i + 1 == early_depth:
                return self.read_codes(out, i)[2][0, 1:]
            residual, last_write = self.write(out, anchor_code, i, forced_senders)
            norm = (self.layer(i + 1)("input_layernorm") if i + 1 < depth
                    else self.weights["norm.weight"])
            f = residual.astype(mx.float32)
            n = (f * mx.rsqrt(mx.mean(mx.square(f), axis=-1, keepdims=True) + config.rms_norm_eps)).astype(mx.bfloat16)
            h = (n.astype(mx.float32) * norm.astype(mx.float32)).astype(mx.bfloat16)
        return h[0, 1:], last_write

    def lattice(self, model, pending, prefix, width, *, extra_senders=None, backbone_depth=None,
                seeds=(), seed_embeddings=False):
        hidden, (raw, rms, receiver) = self.hidden(
            model, pending, prefix, width, backbone_depth=backbone_depth, seeds=seeds, seed_embeddings=seed_embeddings,
        )
        rows = width - 1
        packed = pack_activations(hidden)
        if self.proposal_vocab is None:
            logits = model.vocabulary_order.token_order(model.packed_head.project(packed, rows))
            token_ids = None
        else:
            token_ids, head, extra = self.proposal_vocab
            logits = head.project(packed, rows)
            if extra is not None:
                logits = mx.concatenate((logits, linear(hidden, extra)), axis=-1)
        choices = top16(logits)
        if width not in self.checked_widths:
            reference = mx.sort(mx.argsort(-logits.astype(mx.float32), axis=-1)[:, :16], axis=-1)
            assert mx.array_equal(choices, reference).item(), "DSpine vocabulary selection differs from stable top-16"
            self.checked_widths.add(width)
        if token_ids is not None:
            choices = token_ids[choices]

        head = model.vocabulary_rows(choices.reshape(-1)).reshape(rows, 16, -1).astype(mx.float32)
        norm = self.weights["norm.weight"].astype(mx.float32)
        base = mx.sum((raw[:, None, :] * norm) * head, axis=-1)
        # Compute entries only for the selected tokens instead of building both vocabulary-wide tables.
        write_layer = -1 if backbone_depth is None else backbone_depth - 1
        write = self.bridge("feature_write_weights")[write_layer]
        target = ((head * norm).astype(mx.bfloat16) @ write).astype(mx.float32)
        prev = mx.concatenate((mx.full((1, 16), pending, dtype=mx.int32), choices[:-1]))
        if extra_senders is not None:
            prev = mx.concatenate((prev, extra_senders), axis=1)
        source_gate, source_value = mx.split(linear(self.token_code(prev), self.sender).astype(mx.float32), 2, axis=-1)
        bias = self.bridge("gate.bias").astype(mx.float32)
        message = mx.sigmoid(receiver[:, None, :] + source_gate + bias) * source_value
        refined = message @ target.swapaxes(1, 2)
        scores = base[:, None, :] + rms[:, None, :] * refined
        return choices, scores, rms

    def propose(self, model, pending, prefix, _cache, proposals, _reader_enabled):
        options, scores, _ = self.lattice(model, pending, prefix, proposals + 1)
        best = mx.argmax(scores, axis=-1)
        selected = best[0, 0]
        tokens = [mx.take(options[0], selected)]
        for row, next_slot in zip(options[1:], best[1:]):
            selected = mx.take(next_slot, selected)
            tokens.append(mx.take(row, selected))
        return mx.stack(tokens)

    def candidate_lattice(self, model, pending, prefix):
        options, scores, rms = self.lattice(model, pending, prefix, self.config.block_size)
        logits = scores / mx.maximum(rms[:, None, :], self.eps)
        logp = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
        mx.eval(options, logp)
        return options.tolist(), logp.tolist()

    def ranked_tree(self, model, pending, prefix, width):
        options, logp = self.candidate_lattice(model, pending, prefix)
        return rank_tree(pending, options, logp, width)

    def propose_tree(self, model, pending, prefix, width):
        nodes, parents, costs = self.ranked_tree(model, pending, prefix, width)
        if self.wide_cost_ratio is not None and len(nodes) > 8 and confirmation_ratio(costs) < self.wide_cost_ratio:
            return layout_tree(nodes[:8], parents[:8])
        return layout_tree(nodes, parents)


def draft_attention_input(known, block, window):
    if window is None:
        return mx.concatenate((known, block), axis=2)
    leading, recent = window
    if known.shape[2] <= leading + recent:
        return mx.concatenate((known, block), axis=2)
    tail = known[:, :, -recent:]
    segments = (known[:, :, :leading], tail, block) if leading else (tail, block)
    return mx.concatenate(segments, axis=2)


def checked_paired(weight, name):
    original = pack_weight(weight)
    coded = encode_paired(original, compact_descriptors=True)
    if not mx.array_equal(unpack_joint(coded.payload), original).item():
        raise ValueError(f"{name} did not round-trip")
    return coded


def checked_split_k(weight, name):
    original = pack_weight(weight)
    coded = encode_down(original)
    if not mx.array_equal(decoded_bits(coded.payload), original).item():
        raise ValueError(f"{name} did not round-trip")
    return coded


def with_packed_ffn(base, budget):
    draft = copy(base)
    stored_bytes = original_bytes = 0
    payloads = []
    for i in range(draft.config.num_hidden_layers):
        budget.check(force=True, required_bytes=2 * GIB)
        weight = base.layer(i)
        gate, up, down = (weight(f"mlp.{name}_proj") for name in ("gate", "up", "down"))
        paired = checked_paired(mx.concatenate((gate, up), axis=0), f"DSpine layer {i} gate/up")
        coded = checked_split_k(down, f"DSpine layer {i} down")
        original_bytes += sum(w.nbytes for w in (gate, up, down))
        stored_bytes += sum(part.nbytes for part in (*paired.payload, *coded.payload))
        payloads.append((paired, coded))
        mx.clear_cache()
    draft.packed_ffn = payloads
    draft.weights = {name: weight for name, weight in base.weights.items() if ".mlp." not in name}
    draft.checked_widths = set()
    print("packed_ffn", json.dumps({"layers": len(payloads), "roundtrip_exact": True,
          "original_bytes": original_bytes, "packed_bytes": stored_bytes,
          "peak_gib": mx.get_peak_memory() / GIB, **budget.metrics()}), flush=True)
    return draft


def with_packed_projections(base, budget):
    draft = copy(base)
    budget.check(force=True, required_bytes=2 * GIB)
    draft.packed_context = checked_split_k(base.weights["fc.weight"], "DSpine context projection")
    original_bytes = base.weights["fc.weight"].nbytes
    bytes_stored = sum(x.nbytes for x in draft.packed_context.payload)
    kv_descriptor_bytes = 0
    layers = []
    for i in range(base.config.num_hidden_layers):
        weights = base.layer(i)
        q, k, v, o = (weights(f"self_attn.{name}_proj") for name in ("q", "k", "v", "o"))
        half = q.shape[0] // 2
        combined = mx.concatenate((q[:half], k, q[half:], v), axis=0)
        qkv = checked_paired(combined, f"DSpine layer {i} QKV")
        # The trailing descriptor rows refer to the K/V packets in the same payload.
        kv = LosslessPaired((*qkv.payload[:-1], mx.contiguous(qkv.payload[-1][half // 8:])))
        if not mx.array_equal(unpack_joint(kv.payload), pack_weight(mx.concatenate((k, v), axis=0))).item():
            raise ValueError(f"DSpine layer {i} shared K/V descriptor did not round-trip")
        out = checked_paired(o, f"DSpine layer {i} attention output")
        layers.append((qkv, kv, out))
        original_bytes += sum(weight.nbytes for weight in (q, k, v, o))
        bytes_stored += sum(x.nbytes for x in (*qkv.payload, *out.payload))
        kv_descriptor_bytes += kv.payload[-1].nbytes
        del combined
        mx.clear_cache()
        budget.check(force=True, required_bytes=GIB)
    draft.packed_attention = layers
    draft.checked_widths = set()
    # Keep the context and K/V weights for large prefill panels.
    draft.weights = {k: v for k, v in base.weights.items()
                     if not k.endswith(("self_attn.q_proj.weight", "self_attn.o_proj.weight"))}
    print("packed_projections", json.dumps({"layers": len(layers), "roundtrip_exact": True,
          "original_bytes": original_bytes, "packed_bytes": bytes_stored,
          "kv_descriptor_view_bytes": kv_descriptor_bytes, **budget.metrics()}), flush=True)
    return draft


def with_prompt_vocabulary(base, model, input_ids):
    draft = copy(base)
    size = model.vocabulary_order.draft_rows
    assert size == 32768
    extra_ids = sorted({i for i in input_ids if i >= size})
    ids = mx.array([*range(size), *extra_ids], dtype=mx.int32)
    head = model.vocabulary_order.draft_view(model.packed_head)
    extra = model.vocabulary_rows(ids[size:]) if extra_ids else None
    if extra is not None:
        mx.eval(extra)
    draft.proposal_vocab = ids, head, extra
    draft.checked_widths = set()
    print("prompt_vocabulary", json.dumps({"fixed_rows": size, "prompt_rows": len(extra_ids),
          "extra_bytes": 0 if extra is None else extra.nbytes}), flush=True)
    return draft


def rank_tree(pending, options, logp, width, *, previous_slot=0):
    """Add candidates in probability order with an ancestor and at most two extra edges."""
    nodes = [pending]
    parents = [-1]
    # Store path log costs so one ranking can serve both verification widths.
    costs = [0.0]
    has_child = [False]
    heap = []

    def extend(parent, depth, previous, cost):
        if depth < len(options):
            for slot, log_probability in enumerate(logp[depth][previous]):
                heapq.heappush(heap, (cost - log_probability, parent, depth, slot))

    extend(0, 0, previous_slot, 0.0)
    forks = 0
    while heap and len(nodes) < width:
        cost, parent, depth, slot = heapq.heappop(heap)
        side = has_child[parent]
        if side and forks == 2:
            continue
        node = len(nodes)
        nodes.append(options[depth][slot])
        parents.append(parent)
        costs.append(cost)
        has_child[parent] = True
        has_child.append(False)
        forks += side
        extend(node, depth + 1, slot, cost)
    return nodes, parents, costs


def confirmation_ratio(costs):
    """Ratio of expected confirmations with all ranked nodes versus the first eight."""
    mass = [exp(-cost) for cost in costs]
    return sum(mass) / sum(mass[:8])


def layout_tree(nodes, ranked_parents):
    children = [[] for _ in nodes]
    for i, parent in enumerate(ranked_parents[1:], 1):
        children[parent].append(i)
    stack = [(0, -1)]
    order, parents = [], []
    while stack:
        node, parent = stack.pop()
        index = len(order)
        order.append(nodes[node])
        parents.append(parent)
        stack.extend((child, index) for child in reversed(children[node]))
    spine = next((i for i in range(1, len(order)) if parents[i] != i - 1), len(order) - 1)
    return order, parents, spine


def candidate_tree(pending, options, logp, width):
    nodes, parents, _ = rank_tree(pending, options, logp, width)
    return layout_tree(nodes, parents)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workload", choices=("code", "chinese_technical_report", "english_technical_tutorial"), default="code")
    parser.add_argument("--layout", action="append", choices=("chain8", "chain16", "tree8", "tree16"))
    parser.add_argument("--packed-ffn", action="store_true", help="use bit-preserving packed BF16 FFN weights")
    parser.add_argument("--packed-projections", action="store_true", help="pack BF16 draft attention and short context projections")
    parser.add_argument("--short-draft-vocab", action="store_true",
                        help="score the 32K draft prefix and prompt token IDs for proposals")
    parser.add_argument("--wide-cost-ratio", type=float, help="use 16 verification positions only above this value ratio")
    args = parser.parse_args()
    if args.wide_cost_ratio is not None and not 1.0 <= args.wide_cost_ratio <= 2.0:
        parser.error("--wide-cost-ratio must be in [1, 2]")
    verify_checkpoint()
    trace = json.loads((ROOT / "runs/frontier-target-margin-audit.json").read_text())[args.workload]
    assert len(trace["input"]) == 4096 and len(trace["output"]) == 1024
    with metal_job(80, 2) as budget:
        budget.check(force=True, required_bytes=6 * GIB)
        draft = DSpineDraft()
        mx.eval(draft.weights, draft.code_projection, draft.sender)
        bf16_count = len(draft.weights) - len(VERSION_KEYS)
        if args.packed_ffn:
            draft = with_packed_ffn(draft, budget)
            mx.clear_cache()
        if args.packed_projections:
            draft = with_packed_projections(draft, budget)
            mx.clear_cache()
        model = benchmark_target(budget)
        budget.admit(model, 5120, 512, False, extra_bytes=GIB)
        if args.short_draft_vocab:
            draft = with_prompt_vocabulary(draft, model, trace["input"])
        draft.wide_cost_ratio = args.wide_cost_ratio
        print("load", json.dumps({"weights_bf16": bf16_count, "packed_ffn": args.packed_ffn,
              "packed_projections": args.packed_projections,
              "peak_gib": mx.get_peak_memory() / GIB, **budget.metrics()}), flush=True)
        prefix = prepare_prefix(model, draft, trace, budget)
        for layout in args.layout or ("chain8", "chain16", "tree16"):
            tree = layout.startswith("tree")
            width = int(layout[4 if tree else 5:])
            decode(model, draft, trace, prefix, width, False, budget, tree_proposals=tree)
            mx.clear_cache()
            budget.check(force=True)


if __name__ == "__main__":
    main()
