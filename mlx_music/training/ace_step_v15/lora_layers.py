"""
LoRA configuration for ACE-Step v1.5 fine-tuning.

Reuses the core LoRA implementation from ace_step.lora_layers with
v1.5-specific target module names based on Qwen3 architecture:
- Attention: q_proj, k_proj, v_proj, o_proj (GQA with 16/8 heads)
- MLP: gate_proj, up_proj, down_proj (SiLU-gated Qwen3MLP)

Layer dimensions in v1.5 (24 DiT layers):
  q_proj: 2048 -> 2048 (16 heads * 128 dim)
  k_proj: 2048 -> 1024 (8 KV heads * 128 dim)
  v_proj: 2048 -> 1024 (8 KV heads * 128 dim)
  o_proj: 2048 -> 2048
  gate_proj: 2048 -> 6144
  up_proj: 2048 -> 6144
  down_proj: 6144 -> 2048
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

# V1.5 attention projection targets (in AceStepV15Attention)
V15_ATTENTION_TARGETS = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
]

# V1.5 MLP targets (in Qwen3MLP)
V15_MLP_TARGETS = [
    "gate_proj",
    "up_proj",
    "down_proj",
]

# Combined attention + MLP targets
V15_ALL_TARGETS = V15_ATTENTION_TARGETS + V15_MLP_TARGETS


def apply_lora_to_v15_model(
    model: nn.Module,
    rank: int = 64,
    alpha: float = 64.0,
    dropout: float = 0.0,
    target_modules: Optional[List[str]] = None,
    include_mlp: bool = False,
) -> int:
    """Apply LoRA to an ACE-Step v1.5 model.

    Args:
        model: ACE-Step v1.5 model (typically the DiT decoder)
        rank: LoRA rank (default: 64)
        alpha: LoRA scaling factor (default: 64.0)
        dropout: LoRA dropout probability (default: 0.0)
        target_modules: Override target module names. If None, uses attention targets.
        include_mlp: Also target MLP projections (more params, more capacity)

    Returns:
        Number of layers converted to LoRA
    """
    if target_modules is None:
        target_modules = V15_ALL_TARGETS if include_mlp else V15_ATTENTION_TARGETS

    return apply_lora_to_model(
        model,
        rank=rank,
        alpha=alpha,
        dropout=dropout,
        target_modules=target_modules,
    )


__all__ = [
    "V15_ATTENTION_TARGETS",
    "V15_MLP_TARGETS",
    "V15_ALL_TARGETS",
    "apply_lora_to_v15_model",
    # Re-exported
    "LoRALinear",
    "get_lora_parameters",
    "merge_lora_weights",
    "save_lora_weights",
    "load_lora_weights",
]
