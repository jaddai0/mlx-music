"""
MusicGen training infrastructure.

Provides LoRA fine-tuning and cross-entropy loss for training
MusicGen autoregressive audio generation models.

Unlike ACE-Step and Stable Audio (diffusion models), MusicGen is
autoregressive and predicts audio codebook tokens. Training uses:
- Cross-entropy loss on codebook predictions
- Teacher forcing with causal masking
- Separate LoRA for text encoder vs audio decoder (optional)
"""

from .lora_layers import (
    LoRALinear,
    apply_lora_to_musicgen,
    get_lora_parameters,
    merge_lora_weights,
    save_lora_weights,
    load_lora_weights,
    MUSICGEN_LORA_TARGETS,
)
from .loss import (
    MusicGenLoss,
    shift_codes_right,
    create_causal_mask,
)

__all__ = [
    # LoRA
    "LoRALinear",
    "apply_lora_to_musicgen",
    "get_lora_parameters",
    "merge_lora_weights",
    "save_lora_weights",
    "load_lora_weights",
    "MUSICGEN_LORA_TARGETS",
    # Loss
    "MusicGenLoss",
    "shift_codes_right",
    "create_causal_mask",
]
