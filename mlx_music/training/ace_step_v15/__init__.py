"""
ACE-Step v1.5 training infrastructure.

Provides LoRA fine-tuning, flow matching loss, and dataset utilities
for training ACE-Step v1.5 music generation models.

Reuses the core LoRA implementation from ace_step.lora_layers with
v1.5-specific target module names (q_proj, k_proj, etc.).
"""

from .lora_layers import (
    V15_ATTENTION_TARGETS,
    V15_ALL_TARGETS,
    apply_lora_to_v15_model,
)
from .loss import ACEStepV15Loss, create_v15_train_step
from .dataset import (
    AudioExampleV15,
    AudioBatchV15,
    AudioDatasetV15,
    PreEncodedDatasetV15,
)
from .trainer import (
    TrainingConfigV15,
    ACEStepV15Trainer,
)

# Re-export from v1 (model-agnostic)
from mlx_music.training.ace_step.lora_layers import (
    LoRALinear,
    get_lora_parameters,
    merge_lora_weights,
    save_lora_weights,
    load_lora_weights,
)

__all__ = [
    # LoRA (v1.5 specific)
    "V15_ATTENTION_TARGETS",
    "V15_ALL_TARGETS",
    "apply_lora_to_v15_model",
    # LoRA (shared)
    "LoRALinear",
    "get_lora_parameters",
    "merge_lora_weights",
    "save_lora_weights",
    "load_lora_weights",
    # Loss
    "ACEStepV15Loss",
    "create_v15_train_step",
    # Dataset
    "AudioExampleV15",
    "AudioBatchV15",
    "AudioDatasetV15",
    "PreEncodedDatasetV15",
    # Trainer
    "TrainingConfigV15",
    "ACEStepV15Trainer",
]
