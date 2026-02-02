"""
LoRA (Low-Rank Adaptation) for Stable Audio.

Stable Audio uses DiT (Diffusion Transformer) architecture with:
- GQAttention: Grouped Query Attention (fewer KV heads than Q heads)
- CrossAttention: Standard cross-attention for text conditioning

For GQA, we support different ranks for Q vs KV projections since
they have different dimensions:
- Q: num_attention_heads * head_dim
- K/V: num_key_value_heads * head_dim

This module provides convenience functions to apply LoRA to Stable Audio
models, reusing the base LoRA implementation from ACE-Step.

Recommended configuration:
- Q rank: 64-128 (larger for more capacity)
- KV rank: 32-64 (can be smaller due to fewer heads)
- Alpha: Equal to rank for 1.0 scaling
"""

from typing import List, Optional

import mlx.nn as nn

from mlx_music.training.ace_step.lora_layers import (
    LoRALinear,
    apply_lora_to_model,
    get_lora_parameters,
    merge_lora_weights,
    save_lora_weights,
    load_lora_weights,
)

# Default targets for Stable Audio attention layers
STABLE_AUDIO_LORA_TARGETS = [
    "q_proj",
    "k_proj",
    "v_proj",
    "out_proj",
]


def apply_lora_to_stable_audio(
    model: nn.Module,
    q_rank: int = 64,
    kv_rank: int = 32,
    alpha: Optional[float] = None,
    dropout: float = 0.0,
    target_modules: Optional[List[str]] = None,
) -> int:
    """
    Apply LoRA to a Stable Audio model with GQA-aware configuration.

    For Grouped Query Attention (GQA), the KV projections have fewer
    parameters than Q projections, so we support different ranks.

    Args:
        model: Stable Audio DiT model
        q_rank: LoRA rank for Q and output projections (default: 64)
        kv_rank: LoRA rank for K and V projections (default: 32)
        alpha: LoRA scaling factor (default: equal to respective rank)
        dropout: LoRA dropout probability (default: 0.0)
        target_modules: Which modules to apply LoRA to
            Default: ["q_proj", "k_proj", "v_proj", "out_proj"]

    Returns:
        Number of layers converted to LoRA

    Example:
        model = load_stable_audio_model()
        num_lora = apply_lora_to_stable_audio(model, q_rank=64, kv_rank=32)
        print(f"Applied LoRA to {num_lora} layers")
    """
    if target_modules is None:
        target_modules = STABLE_AUDIO_LORA_TARGETS

    # For simplicity, use the higher rank for all layers
    # A more sophisticated implementation could track module paths
    # and apply different ranks to Q vs KV
    rank = max(q_rank, kv_rank)
    effective_alpha = alpha if alpha is not None else float(rank)

    return apply_lora_to_model(
        model,
        rank=rank,
        alpha=effective_alpha,
        dropout=dropout,
        target_modules=target_modules,
    )


def apply_gqa_aware_lora(
    model: nn.Module,
    q_rank: int = 64,
    kv_rank: int = 32,
    dropout: float = 0.0,
) -> int:
    """
    Apply GQA-aware LoRA with different ranks for Q vs KV.

    This is a more sophisticated version that tracks module paths
    and applies appropriate ranks based on the projection type.

    Args:
        model: Stable Audio model
        q_rank: Rank for q_proj and out_proj
        kv_rank: Rank for k_proj and v_proj
        dropout: LoRA dropout

    Returns:
        Number of layers converted
    """
    replaced = 0
    visited = set()

    def _get_children(module: nn.Module):
        """Get child modules using MLX's children() method."""
        if hasattr(module, 'children') and callable(module.children):
            children_dict = module.children()
            if isinstance(children_dict, dict):
                return list(children_dict.items())
        return []

    def _replace_in_module(module: nn.Module, prefix: str = "") -> int:
        nonlocal replaced

        # Prevent cycles
        if id(module) in visited:
            return replaced
        visited.add(id(module))

        for name, child in _get_children(module):
            full_name = f"{prefix}.{name}" if prefix else name

            if isinstance(child, nn.Linear) and not isinstance(child, LoRALinear):
                # Determine rank based on projection type
                if name in ("q_proj", "out_proj"):
                    rank = q_rank
                    alpha = float(q_rank)
                elif name in ("k_proj", "v_proj"):
                    rank = kv_rank
                    alpha = float(kv_rank)
                else:
                    continue  # Skip non-target layers

                # Replace with LoRA
                lora_layer = LoRALinear(
                    in_features=child.weight.shape[1],
                    out_features=child.weight.shape[0],
                    rank=rank,
                    alpha=alpha,
                    dropout=dropout,
                    original_layer=child,
                )
                setattr(module, name, lora_layer)
                replaced += 1

            elif isinstance(child, nn.Module) and not isinstance(child, LoRALinear):
                _replace_in_module(child, full_name)

        return replaced

    return _replace_in_module(model)


# Re-export common utilities
__all__ = [
    "LoRALinear",
    "apply_lora_to_stable_audio",
    "apply_gqa_aware_lora",
    "get_lora_parameters",
    "merge_lora_weights",
    "save_lora_weights",
    "load_lora_weights",
    "STABLE_AUDIO_LORA_TARGETS",
]
