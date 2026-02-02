"""
Stable Audio training infrastructure.

Provides LoRA fine-tuning, v-prediction loss, and training utilities
for Stable Audio Open models.

Stable Audio uses EDM-style v-prediction training with:
- GQAttention: Grouped Query Attention (fewer KV heads)
- CrossAttention: Text conditioning
- EDM noise schedule with Karras sigmas
"""

from .lora_layers import (
    LoRALinear,
    apply_lora_to_stable_audio,
    apply_gqa_aware_lora,
    get_lora_parameters,
    merge_lora_weights,
    save_lora_weights,
    load_lora_weights,
    STABLE_AUDIO_LORA_TARGETS,
)
from .loss import StableAudioLoss

__all__ = [
    # LoRA
    "LoRALinear",
    "apply_lora_to_stable_audio",
    "apply_gqa_aware_lora",
    "get_lora_parameters",
    "merge_lora_weights",
    "save_lora_weights",
    "load_lora_weights",
    "STABLE_AUDIO_LORA_TARGETS",
    # Loss
    "StableAudioLoss",
]
