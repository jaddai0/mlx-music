"""Weight mapping from PyTorch MOSS-SoundEffect checkpoint to MLX."""

from __future__ import annotations

from typing import Any


def map_weights(weights: dict[str, Any]) -> dict[str, Any]:
    """Map PyTorch MOSS-SoundEffect weight keys to MLX model keys.

    The HF checkpoint uses:
        language_model.embed_tokens.weight
        language_model.layers.N.self_attn.{q,k,v,o}_proj.weight
        language_model.layers.N.mlp.{gate,up,down}_proj.weight
        language_model.layers.N.{input,post_attention}_layernorm.weight
        language_model.norm.weight
        emb_ext.N.weight
        lm_heads.N.weight

    Our MLX model wraps Qwen3 as language_model.model.*, so:
        language_model.X -> language_model.model.X

    emb_ext.N and lm_heads.N pass through directly.
    No transpositions needed (MLX Linear stores weight as (out, in) same as PyTorch).
    """
    mapped = {}
    for key, value in weights.items():
        new_key = key

        # Map backbone keys: language_model.X -> language_model.model.X
        if key.startswith("language_model.") and not key.startswith("language_model.model."):
            suffix = key[len("language_model."):]
            new_key = f"language_model.model.{suffix}"

        # emb_ext.N.weight -> direct
        # lm_heads.N.weight -> direct

        mapped[new_key] = value

    return mapped


# Layers/modules that should NOT be quantized
QUANTIZATION_SKIP_PATTERNS = [
    "emb_ext.",                          # Audio embeddings (small)
    "language_model.model.embed_tokens", # Text embedding
    "language_model.model.norm",         # Final norm
    "lm_heads.1.", "lm_heads.2.", "lm_heads.3.", "lm_heads.4.",   # Audio heads
    "lm_heads.5.", "lm_heads.6.", "lm_heads.7.", "lm_heads.8.",
    "lm_heads.9.", "lm_heads.10.", "lm_heads.11.", "lm_heads.12.",
    "lm_heads.13.", "lm_heads.14.", "lm_heads.15.", "lm_heads.16.",
    "lm_heads.17.", "lm_heads.18.", "lm_heads.19.", "lm_heads.20.",
    "lm_heads.21.", "lm_heads.22.", "lm_heads.23.", "lm_heads.24.",
    "lm_heads.25.", "lm_heads.26.", "lm_heads.27.", "lm_heads.28.",
    "lm_heads.29.", "lm_heads.30.", "lm_heads.31.", "lm_heads.32.",
]


def should_quantize(key: str) -> bool:
    """Check if a weight key should be quantized."""
    for pattern in QUANTIZATION_SKIP_PATTERNS:
        if pattern in key:
            return False
    # Also skip any norm/layernorm layers
    if "layernorm" in key.lower() or key.endswith(".norm.weight"):
        return False
    return True
