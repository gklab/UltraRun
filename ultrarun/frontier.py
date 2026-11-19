"""Bounded best-first expansion of native MTP candidates."""
from bisect import bisect_right
from collections import Counter, defaultdict
from dataclasses import dataclass, replace
from functools import lru_cache
import heapq
import math

import mlx.core as mx

from .draft_short_head import MIN_PATH_MASS
from .phrase_history import MAX_ANCHORED_TOKENS, MAX_PHRASE_TOKENS


NODE_LIMIT = 11
NARROW_WIDTH = 8
PHRASE_WIDTH_LIMIT = NODE_LIMIT + MAX_PHRASE_TOKENS
MIN_MASS = math.log(0.12)
RESERVE_MASS = math.log(0.06)
WIDEN_MASS = math.log(0.35)
WIDEN_MIN_GAIN = 0.72
WIDEN_CONTINUATION = 0.82
PEER_GAP = math.log(2.0)
ALT_GAP = 2.0
MAX_SIDES = 2
MAX_GROUP = 3
ANCHOR_LENGTH = 4
STRONG_ANCHOR_LENGTH = 2
MIN_ANCHORED_TAIL = 2
RANK_THREADS = 128
RANK_VALUES_PER_THREAD = 4
CONFIDENCE_BINS = (0.45, 0.65, 0.8, 0.9, 0.97, 0.995)
CALIBRATION_PRIOR = 30
COUNTERFACTUAL_WEIGHT = 0.25
PATH_REACH_COEFFICIENTS = (-0.175, -0.2076, 0.0221, 0.0626, 0.0713, 0.0122,
                           -0.3137, 0.0518, -0.0303, 0.0401)


@dataclass
class Node:
    parent: int
    token: int
    depth: int
    mass: float
    state: object = None
    hidden: mx.array | None = None


class FrontierCalibration:
    """Estimate draft rank accuracy from already verified target predictions."""

    def __init__(self):
        self.counts = defaultdict(Counter)
        self.samples = 0
        self.counterfactual_samples = 0
        self.pending = {}
        self.reserve_nodes = 0
        self.fourth_rank_nodes = 0
        self.kept_narrow = 0
        self.early_narrow = 0
        self.skipped_rows = 0
        self.scored_boundaries = 0
        self.alternate_boundary_hits = 0
        self.reorder_rounds = 0

    @property
    def reorder_paths(self):
        return (self.scored_boundaries >= 24
                and self.alternate_boundary_hits + 4 >= 0.43 * (self.scored_boundaries + 9))

    def adjust(self, depth, raw):
        band = min(2, (depth + 2) // 3)
        cell = band, bisect_right(CONFIDENCE_BINS, math.exp(raw[0]))
        count = self.counts[cell]
        total = count.total() + CALIBRATION_PRIOR
        probabilities = [(count[rank] + CALIBRATION_PRIOR * math.exp(value)) / total
                         for rank, value in enumerate(raw)]
        return cell, [math.log(value) if value else -math.inf for value in probabilities]

    def observe(self, path, predictions):
        committed = set(path)
        terminal = max(committed, default=-1)
        for index, (cell, tokens) in self.pending.items():
            correct = predictions[index]
            rank = next((i for i, token in enumerate(tokens) if token == correct), len(tokens))
            if index == terminal:
                self.scored_boundaries += 1
                self.alternate_boundary_hits += 0 < rank < len(tokens)
            if index in committed:
                self.counts[cell][rank] += 1
                self.samples += 1
            else:
                self.counts[cell][rank] += COUNTERFACTUAL_WEIGHT
                self.counterfactual_samples += 1
        self.pending.clear()


def fitted_path_priority(nominal, depth, first_logp, rank, gap, parent_mass, root):
    probability = min(1.0 - 1e-8, max(1e-8, math.exp(nominal)))
    first = math.exp(first_logp)
    relative_depth = (depth - 3) / 7
    first_rank = float(rank == 0)
    features = (1.0, relative_depth, first - 0.5, rank / 3, min(5.0, gap) / 5.0 - 0.5,
                math.exp(parent_mass) - 0.5, first_rank, (first - 0.65) * (1.0 - first_rank),
                relative_depth * (1.0 - first_rank), float(root))
    logit = math.log(probability) - math.log1p(-probability)
    logit += sum(weight * value for weight, value in zip(PATH_REACH_COEFFICIENTS, features))
    return -math.log1p(math.exp(-max(-35.0, min(35.0, logit))))


def ranked_scores(logits, shortlist, columns):
    score = logits.astype(mx.float32)
    norm = mx.logsumexp(score, axis=-1, keepdims=True)
    return shortlist[columns], mx.take_along_axis(score, columns, axis=-1) - norm


@lru_cache
def reference_ranking(ranks):
    @mx.compile
    def find(logits, shortlist):
        columns = mx.arange(logits.shape[-1])
        remaining = logits.astype(mx.float32)
        chosen = []
        for _ in range(ranks):
            pick = mx.argmax(remaining, axis=-1, keepdims=True)
            chosen.append(pick)
            remaining = mx.where(columns == pick, -mx.inf, remaining)
        return ranked_scores(logits, shortlist, mx.concatenate(chosen, axis=-1))
    return find


@lru_cache
def partial_rank_kernel(with_mass=False):
    return mx.fast.metal_kernel(
        name="local_frontier_rank_and_mass" if with_mass else "local_frontier_top_ranks",
        input_names=["logits"],
        output_names=["ranks", "stats"] if with_mass else ["ranks"],
        header=f"#define FRONTIER_MASS {int(with_mass)}\n",
        source=r"""
        uint group = threadgroup_position_in_grid.x;
        uint row = threadgroup_position_in_grid.y;
        uint tid = thread_index_in_threadgroup;
        uint lane = tid % 32;
        uint count = logits_shape[1];
        uint blocks = (count + THREADS * VALUES - 1) / (THREADS * VALUES);
        uint best[RANKS] = {};
        #if FRONTIER_MASS
        float values[VALUES];
        float maximum = -INFINITY;
        #endif
        #pragma unroll
        for (uint i = 0; i < VALUES; ++i) {
            uint index = group * THREADS * VALUES + i * THREADS + tid;
            #if FRONTIER_MASS
            bfloat16_t value = index < count ? logits[row * count + index] : bfloat16_t(-INFINITY);
            values[i] = float(value);
            maximum = max(maximum, values[i]);
            if (index >= count) continue;
            #else
            if (index >= count) continue;
            bfloat16_t value = logits[row * count + index];
            #endif
            uint bits = as_type<ushort>(value);
            if ((bits & 0x7fffu) == 0) bits = 0;
            // Sort by BF16 score, then by the original shortlist position.
            uint score = bits & 0x8000u ? bits ^ 0xffffu : bits ^ 0x8000u;
            uint key = (score << 16) | (65535u - index);
            #pragma unroll
            for (uint j = 0; j < RANKS; ++j) {
                uint saved = max(best[j], key);
                key = min(best[j], key);
                best[j] = saved;
            }
        }
        uint part = (row * blocks + group) * (THREADS / 32) + tid / 32;
        #if FRONTIER_MASS
        maximum = simd_max(maximum);
        float mass = 0.0f;
        if (maximum > -INFINITY) {
            #pragma unroll
            for (uint i = 0; i < VALUES; ++i)
                mass += exp(values[i] - maximum);
        }
        mass = simd_sum(mass);
        if (lane == 0) {
            stats[part * 2] = maximum;
            stats[part * 2 + 1] = mass;
        }
        #endif
        uint at = part * RANKS;
        #pragma unroll
        for (uint j = 0; j < RANKS; ++j) {
            uint selected = simd_max(best[0]);
            if (lane == 0) ranks[at + j] = selected;
            if (selected == best[0]) {
                #pragma unroll
                for (uint k = 0; k + 1 < RANKS; ++k) best[k] = best[k + 1];
                best[RANKS - 1] = 0;
            }
        }
        """,
    )


@lru_cache
def finish_rank_and_mass_kernel():
    return mx.fast.metal_kernel(
        name="local_frontier_finish_rank_and_mass",
        input_names=["logits", "shortlist", "candidates", "stats"],
        output_names=["tokens", "log_probs"],
        source=r"""
        uint row = threadgroup_position_in_grid.x;
        uint tid = thread_index_in_threadgroup;
        uint lane = tid % 32;
        uint group = tid / 32;
        uint last = stats_shape[1];
        threadgroup float maxima[8];
        threadgroup float sums[8];
        threadgroup uint candidates_by_group[8 * RANKS];
        float maximum = -INFINITY;
        uint local[RANKS] = {};
        for (uint i = tid; i < last; i += 256) {
            uint offset = row * last + i;
            maximum = max(maximum, stats[offset * 2]);
            #pragma unroll
            for (uint j = 0; j < RANKS; ++j) {
                uint key = candidates[offset * RANKS + j];
                #pragma unroll
                for (uint k = 0; k < RANKS; ++k) {
                    uint top = max(local[k], key);
                    key = min(local[k], key);
                    local[k] = top;
                }
            }
        }
        float warp_max = simd_max(maximum);
        if (lane == 0) maxima[group] = warp_max;
        #pragma unroll
        for (uint i = 0; i < RANKS; ++i) {
            uint selected = simd_max(local[0]);
            if (lane == 0) candidates_by_group[group * RANKS + i] = selected;
            if (selected == local[0]) {
                #pragma unroll
                for (uint k = 1; k < RANKS; ++k) local[k - 1] = local[k];
                local[RANKS - 1] = 0;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        float top = simd_max(lane < 8 ? maxima[lane] : -INFINITY);
        float mass = 0.0f;
        for (uint i = tid; i < last; i += 256) {
            uint offset = (row * last + i) * 2;
            mass += stats[offset + 1] * exp(stats[offset] - top);
        }
        mass = simd_sum(mass);
        if (lane == 0) sums[group] = mass;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (tid < 32) {
            float mass = simd_sum(lane < 8 ? sums[lane] : 0.0f);
            float norm = top + log(mass);
            uint candidate = lane < 8 * RANKS ? candidates_by_group[lane] : 0;
            #pragma unroll
            for (uint i = 0; i < RANKS; ++i) {
                uint selected = simd_max(candidate);
                if (lane == 0) {
                    uint slot = 65535u - (selected & 65535u);
                    tokens[row * RANKS + i] = shortlist[slot];
                    log_probs[row * RANKS + i] = float(logits[row * logits_shape[1] + slot]) - norm;
                }
                if (candidate == selected) candidate = 0;
            }
        }
        """,
    )


@lru_cache
def finish_ranking(ranks):
    @mx.compile
    def finish(logits, shortlist, partials):
        remaining = partials
        chosen = []
        for _ in range(ranks):
            pick = mx.max(remaining, axis=-1, keepdims=True)
            chosen.append(pick)
            remaining = mx.where(remaining == pick, 0, remaining)
        keys = mx.concatenate(chosen, axis=-1)
        columns = (65535 - (keys & 65535)).astype(mx.int32)
        return ranked_scores(logits, shortlist, columns)
    return finish


def top_candidates(logits, shortlist, ranks):
    rows, columns = logits.shape
    if logits.dtype != mx.bfloat16 or not ranks <= columns <= 65536:
        return reference_ranking(min(ranks, columns))(logits, shortlist)
    values = RANK_THREADS * RANK_VALUES_PER_THREAD
    blocks = (columns + values - 1) // values
    parts = blocks * (RANK_THREADS // 32)
    # The frontier scores up to three 32K BF16 rows at a time.
    with_mass = (1 <= rows <= MAX_GROUP and 32768 <= columns <= 36864
                 and shortlist.dtype == mx.int32 and ranks in (3, 4))
    output_shapes = [(rows, parts * ranks)]
    output_dtypes = [mx.uint32]
    if with_mass:
        output_shapes.append((rows, parts, 2))
        output_dtypes.append(mx.float32)
    outputs = partial_rank_kernel(with_mass)(
        inputs=[logits], template=[("THREADS", RANK_THREADS), ("VALUES", RANK_VALUES_PER_THREAD), ("RANKS", ranks)],
        grid=(blocks * RANK_THREADS, rows, 1), threadgroup=(RANK_THREADS, 1, 1),
        output_shapes=output_shapes, output_dtypes=output_dtypes,
    )
    if with_mass:
        tokens, log_probs = finish_rank_and_mass_kernel()(
            inputs=[logits, shortlist, *outputs], template=[("RANKS", ranks)],
            grid=(rows * 256, 1, 1), threadgroup=(256, 1, 1),
            output_shapes=[(rows, ranks), (rows, ranks)], output_dtypes=[mx.int32, mx.float32],
        )
        return tokens, log_probs
    return finish_ranking(ranks)(logits, shortlist, outputs[0])


def append_phrase(history, nodes, children, parent, tokens):
    for token in tokens:
        child = len(nodes)
        nodes.append(Node(parent, token, nodes[parent].depth + 1, nodes[parent].mass))
        children[parent].append(child)
        children.append([])
        parent = child
    history.added_nodes += len(tokens)
    history.rounds += 1


def anchored_candidate(history, remaining, width_limit):
    capacity = min(MAX_ANCHORED_TOKENS, remaining, width_limit - 1)
    minimum = ANCHOR_LENGTH + MIN_ANCHORED_TAIL
    if capacity < minimum:
        return (), ANCHOR_LENGTH
    continuation = history.continuation((), max_tokens=capacity)
    if continuation is None or len(continuation[0]) < minimum:
        return (), ANCHOR_LENGTH
    tokens, suffix, support = continuation
    required = STRONG_ANCHOR_LENGTH if suffix >= 8 or (suffix == 6 and support >= 2) else ANCHOR_LENGTH
    return tokens, required


def append_anchored_phrase(history, nodes, children, candidate, width_limit, required=ANCHOR_LENGTH):
    parent = 0
    for matched, token in enumerate(candidate):
        if not children[parent]:
            break
        parent = children[parent][0]
        if nodes[parent].token != token:
            return False
    else:
        return False
    room = width_limit - len(nodes)
    tail = candidate[matched:matched + room] if room >= MIN_ANCHORED_TAIL else ()
    if matched < required or len(tail) < MIN_ANCHORED_TAIL:
        return False
    append_phrase(history, nodes, children, parent, tail)
    history.anchor_rounds += 1
    history.early_anchor_rounds += matched < ANCHOR_LENGTH
    return True


def append_history_phrase(history, nodes, children, remaining, width_limit):
    size = len(nodes)
    ceiling = NARROW_WIDTH if size <= NARROW_WIDTH else PHRASE_WIDTH_LIMIT
    room = min(ceiling, width_limit) - size
    if room <= 0:
        return

    forks = sum(len(row) > 1 for row in children)
    paths = [()]
    offers = []
    for index, node in enumerate(nodes):
        if index:
            paths.append((*paths[node.parent], node.token))
        if node.depth >= remaining or (len(children[index]) == 1 and forks >= MAX_SIDES):
            continue
        continuation = history.continuation(paths[index])
        if continuation is None:
            continue
        tokens, suffix_length, support = continuation
        if any(nodes[child].token == tokens[0] for child in children[index]):
            continue
        offers.append(((suffix_length + 0.1 * support, node.mass), index, tokens))
    if not offers:
        return

    _, parent, tokens = max(offers)
    tokens = tokens[:min(room, remaining - nodes[parent].depth)]
    append_phrase(history, nodes, children, parent, tokens)


def build_frontier(model, root, initial, offset, pending, remaining, head, shortlist,
                   *, phrases=None, calibration=None, short_head=None):
    nodes = [Node(-1, pending, 0, 0.0, initial, root)]
    children = [[]]
    queue = []
    score_rows = 0
    side_edges = 0
    selection_mass = [0.0]
    samples = {}
    width_limit = min(model.short_tokens, PHRASE_WIDTH_LIMIT) if phrases is not None else NODE_LIMIT
    candidate, required = anchored_candidate(phrases, remaining, width_limit) if phrases is not None else ((), ANCHOR_LENGTH)
    used_anchor = False
    reorder = calibration is not None and calibration.reorder_paths
    if reorder:
        calibration.reorder_rounds += 1

    def enqueue(parents, scores, states=()):
        nonlocal score_rows
        picks, log_probs = top_candidates(scores, shortlist, 4 if calibration is not None else 3)
        kv = [value for state in states for value in (state.keys, state.values)]
        mx.eval(picks, log_probs, *kv)
        score_rows += len(parents)
        for parent, ids, probs in zip(parents, picks.tolist(), log_probs.tolist()):
            choices = probs
            if calibration is not None:
                cell, choices = calibration.adjust(nodes[parent].depth, probs)
                samples[parent] = (cell, ids)
            best = max(choices)
            for rank, (token, logp, adjusted) in enumerate(zip(ids, probs, choices)):
                mass = nodes[parent].mass + logp
                priority = selection_mass[parent] + adjusted
                if rank and probs[0] - logp > ALT_GAP and best - adjusted > ALT_GAP:
                    if calibration is None or priority < RESERVE_MASS:
                        continue
                # Keep the original cumulative probability for extension and width decisions.
                fitted = priority
                if reorder and len(probs) > 1:
                    fitted = fitted_path_priority(priority, nodes[parent].depth + 1, probs[0], rank,
                                                  probs[0] - probs[1], selection_mass[parent], parent == 0)
                heapq.heappush(queue, (-fitted, -mass, parent, rank, token, -priority))

    def best_native_mass():
        if calibration is None:
            return -queue[0][1] if queue else -math.inf
        return max((-edge[1] for edge in queue), default=-math.inf)

    def widen_gain(unscored=()):
        probabilities = [math.exp(-cost) for _, native_cost, parent, _, _, cost in queue
                         if (-native_cost >= MIN_MASS or -cost >= RESERVE_MASS)
                         and nodes[parent].depth < remaining
                         and (side_edges < MAX_SIDES or not children[parent])]
        # The top-three sum is maximized when each unscored parent gives all
        # its probability to one child. Allow slack for FP32 log scores.
        probabilities.extend(math.exp(selection_mass[parent]) * 1.00001 for parent in unscored)
        gains = (probability * WIDEN_CONTINUATION**i for probability in probabilities for i in range(3))
        return sum(heapq.nlargest(3, gains))

    enqueue((0,), model.draft_scores(root, head))
    while queue and len(nodes) < NODE_LIMIT:
        width = len(nodes)
        if width > 2:
            if calibration is not None:
                eligible = [edge for edge in queue if -edge[1] >= MIN_MASS or -edge[5] >= RESERVE_MASS]
                if not eligible:
                    break
                if len(eligible) < len(queue):
                    queue = eligible
                    heapq.heapify(queue)
            elif best_native_mass() < MIN_MASS:
                break
        if width == NARROW_WIDTH:
            if best_native_mass() < WIDEN_MASS:
                break
            if calibration is not None and widen_gain() < WIDEN_MIN_GAIN:
                calibration.kept_narrow += 1
                break
        ceiling = NARROW_WIDTH if width < NARROW_WIDTH else NODE_LIMIT
        picks = []
        best = queue[0][0]
        while queue and len(picks) < min(MAX_GROUP, ceiling - width):
            if picks and queue[0][0] - best > PEER_GAP:
                break
            _, native_cost, parent, rank, token, cost = heapq.heappop(queue)
            is_side = bool(children[parent])
            depth = nodes[parent].depth + 1
            if (is_side and side_edges == MAX_SIDES) or depth > remaining:
                continue
            child = len(nodes)
            nodes.append(Node(parent, token, depth, -native_cost))
            selection_mass.append(-cost)
            if calibration is not None:
                calibration.reserve_nodes += -native_cost < MIN_MASS
                calibration.fourth_rank_nodes += rank == 3
            children[parent].append(child)
            children.append([])
            side_edges += is_side
            picks.append(child)
        if not picks:
            continue
        if candidate and append_anchored_phrase(phrases, nodes, children, candidate, width_limit, required):
            used_anchor = True
            break
        if len(nodes) == NODE_LIMIT:
            break

        # Each descendant has at most its parent's probability.
        active = [n for n in picks if nodes[n].depth < remaining
                  and (len(nodes) <= 2 or nodes[n].mass >= MIN_MASS)]
        if len(nodes) == NARROW_WIDTH:
            upper = max([nodes[n].mass for n in active] + [best_native_mass()])
            if upper < WIDEN_MASS:
                break
            if calibration is not None and active and widen_gain(active) < WIDEN_MIN_GAIN:
                calibration.kept_narrow += 1
                calibration.early_narrow += 1
                calibration.skipped_rows += len(active)
                break
        if not active:
            continue

        states = [replace(nodes[nodes[n].parent].state) for n in active]
        hidden = mx.concatenate([nodes[nodes[n].parent].hidden for n in active])
        tokens = mx.array([nodes[n].token for n in active], dtype=mx.int32)
        offsets = [offset + nodes[n].depth - 1 for n in active]
        if len(active) == 1:
            output = model.draft(hidden, tokens, states[0], offsets[0])
            if short_head is not None and nodes[active[0]].mass >= MIN_PATH_MASS:
                scores = short_head.score(output, head)
            else:
                scores = model.draft_scores(output, head)
        else:
            output, scores = model.draft_batch(hidden, tokens, states, offsets, head)
        for i, node in enumerate(active):
            nodes[node].state = states[i]
            nodes[node].hidden = output[i:i + 1]
        enqueue(active, scores, states)

    # Copied nodes go straight to verification and have no draft states.
    if phrases is not None and not used_anchor:
        append_history_phrase(phrases, nodes, children, remaining, width_limit)

    # Depth-first layout emits the primary path first and at most two saved forks.
    spine = 1
    node = 0
    while children[node]:
        node = children[node][0]
        spine += 1
    ordered = []
    pending_nodes = [0]
    while pending_nodes:
        node = pending_nodes.pop()
        ordered.append(node)
        pending_nodes.extend(reversed(children[node]))
    positions = {node: i for i, node in enumerate(ordered)}
    proposals = [nodes[node].token for node in ordered[1:spine]]
    alternatives = [nodes[node].token for node in ordered[spine:]]
    parents = [positions[nodes[node].parent] for node in ordered[spine:]]
    if calibration is not None:
        calibration.pending = {positions[node]: sample for node, sample in samples.items()}
    return proposals, alternatives, parents, score_rows
