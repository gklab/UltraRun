"""Parameters shared by the deployed performance profile and its benchmark."""

DRAFT_VOCAB_ROWS = 32768

OPTIMIZED_MODEL = {
    "merge": True,
    "skinny": True,
    "reserve_kv": True,
    "fused_delta": True,
    "wide_verify": True,
    "packed_main": True,
    "lossless_down": True,
    "lossless_gate_up": True,
    "lossless_inputs": True,
    "lossless_attention_outputs": True,
    "packed_head": True,
    "packed_head_only": True,
    "lossless_vocabulary": True,
    "lossless_draft_rows": DRAFT_VOCAB_ROWS,
    "shared_vocabulary": True,
    "packed_attention": True,
    "fast_draft": True,
    "fused_ffn": True,
    "split_attention": True,
    "fused_attention_io": True,
    "fused_residual": True,
    "compact_delta": True,
    "compiled_delta": True,
    "direct_delta_sequence": True,
    "async_verify": True,
    "layer_prefill": True,
    "lossless_mtp_ffn": True,
    "wide_blocks": True,
    "compact_tree": True,
}

OPTIMIZED_DRAFT = {
    "draft_depth": 10,
    "draft_vocab": DRAFT_VOCAB_ROWS,
    "draft_confirmed_vocab": True,
    "draft_lane_cache": True,
    "draft_paired_root": True,
    "draft_confidence": 0.12,
    "device_draft": True,
    "draft_extension_confidence": 0.5,
    "draft_root_branch_gap": 1.0,
}
