"""Retain checkpoints of the resident model and its native drafter."""

from dataclasses import dataclass, replace

import mlx.core as mx

from .model import LayerState


@dataclass
class Checkpoint:
    tokens: list[int]
    layers: list[LayerState]
    draft: LayerState
    hidden: mx.array


def resume_layer(state, length):
    """Bring segmented verification caches back to the normal prefill layout."""
    if state.prompt_kv is not None:
        keys, values = state.prompt_kv
        suffix = length - keys.shape[2]
        if suffix:
            keys = mx.concatenate((keys, state.keys[:, :, :suffix]), axis=2)
            values = mx.concatenate((values, state.values[:, :, :suffix]), axis=2)
        state.keys, state.values = keys, values
        state.prompt_kv = None
    state.logical_attention = False
    state.compile_delta_tree = False
    state.draft_lanes = None


class ChatSession:
    """Keep the prompt and the last committed response for one conversation."""

    def __init__(self):
        self.checkpoints = ()

    def take(self, ids):
        hit = next((item for item in self.checkpoints if ids[:len(item.tokens)] == item.tokens), None)
        self.checkpoints = ()
        if hit is not None:
            length = len(hit.tokens)
            for state in hit.layers:
                resume_layer(state, length)
            resume_layer(hit.draft, length - 1)
        return hit

    def remember_prompt(self, ids, cache, draft, hidden):
        # Use COW snapshots so a re-tokenized reply tail can fall back to the prompt.
        self.checkpoints = (Checkpoint(ids, [replace(state) for state in cache], replace(draft), hidden),)

    def remember_output(self, ids, generated, cache, draft, hidden):
        # The final predicted token is emitted before being consumed by the model.
        draft.draft_lanes = None
        self.checkpoints = (Checkpoint(ids + generated[:-1], cache, draft, hidden), *self.checkpoints)
