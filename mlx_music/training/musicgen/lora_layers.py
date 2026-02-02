"""
LoRA (Low-Rank Adaptation) for MusicGen.

MusicGen has two main components that can be fine-tuned:
1. Text encoder (T5-like transformer)
2. Audio decoder (autoregressive transformer)

For most use cases, fine-tuning only the audio decoder is sufficient
and more memory efficient. The text encoder can optionally be fine-tuned
for domain-specific text understanding.

Target layers:
- MusicGenAttention: q_proj, k_proj, v_proj, out_proj
- Text encoder attention (optional): similar structure

Recommended configuration:
- Rank: 32-64 (MusicGen has smaller attention dimensions than diffusion models)
- Alpha: Equal to rank
- Only fine-tune decoder for style transfer
- Fine-tune both for domain adaptation with new vocabulary
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

# Default targets for MusicGen decoder attention
MUSICGEN_LORA_TARGETS = [
    "q_proj",
    "k_proj",
    "v_proj",
    "out_proj",
]


def apply_lora_to_musicgen(
    model: nn.Module,
    rank: int = 32,
    alpha: float = 32.0,
    dropout: float = 0.0,
    target_modules: Optional[List[str]] = None,
    finetune_decoder_only: bool = True,
) -> int:
    """
    Apply LoRA to a MusicGen model.

    By default, only the audio decoder is fine-tuned. Set
    finetune_decoder_only=False to also fine-tune the text encoder.

    Args:
        model: MusicGen model
        rank: LoRA rank (default: 32)
        alpha: LoRA scaling factor (default: 32.0)
        dropout: LoRA dropout probability (default: 0.0)
        target_modules: Which modules to apply LoRA to
            Default: ["q_proj", "k_proj", "v_proj", "out_proj"]
        finetune_decoder_only: If True, skip text encoder (default: True)

    Returns:
        Number of layers converted to LoRA

    Example:
        model = load_musicgen_model()

        # Style transfer: only decoder
        num_lora = apply_lora_to_musicgen(model, rank=32)

        # Domain adaptation: both encoder and decoder
        num_lora = apply_lora_to_musicgen(
            model, rank=32, finetune_decoder_only=False
        )
    """
    if target_modules is None:
        target_modules = MUSICGEN_LORA_TARGETS

    total_replaced = 0

    # Apply to decoder
    if hasattr(model, "decoder"):
        decoder_replaced = apply_lora_to_model(
            model.decoder,
            rank=rank,
            alpha=alpha,
            dropout=dropout,
            target_modules=target_modules,
        )
        total_replaced += decoder_replaced

    # Optionally apply to text encoder
    if not finetune_decoder_only and hasattr(model, "text_encoder"):
        encoder_replaced = apply_lora_to_model(
            model.text_encoder,
            rank=rank,
            alpha=alpha,
            dropout=dropout,
            target_modules=target_modules,
        )
        total_replaced += encoder_replaced

    # If model doesn't have explicit decoder/encoder attributes,
    # try applying to the whole model
    if total_replaced == 0:
        total_replaced = apply_lora_to_model(
            model,
            rank=rank,
            alpha=alpha,
            dropout=dropout,
            target_modules=target_modules,
        )

    return total_replaced


# Re-export common utilities
__all__ = [
    "LoRALinear",
    "apply_lora_to_musicgen",
    "get_lora_parameters",
    "merge_lora_weights",
    "save_lora_weights",
    "load_lora_weights",
    "MUSICGEN_LORA_TARGETS",
]
