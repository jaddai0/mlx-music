"""
ACE-Step training infrastructure.

Provides LoRA fine-tuning, flow matching loss, and dataset utilities
for training ACE-Step music generation models.
"""

from .lora_layers import (
    LoRALinear,
    apply_lora_to_model,
    get_lora_parameters,
    merge_lora_weights,
    save_lora_weights,
    load_lora_weights,
)
from .loss import ACEStepLoss, create_train_step
from .dataset import (
    AudioExample,
    AudioBatch,
    AudioDataset,
    PreEncodedDataset,
)
from .trainer import (
    TrainingConfig,
    TrainingState,
    ACEStepTrainer,
)

__all__ = [
    # LoRA
    "LoRALinear",
    "apply_lora_to_model",
    "get_lora_parameters",
    "merge_lora_weights",
    "save_lora_weights",
    "load_lora_weights",
    # Loss
    "ACEStepLoss",
    "create_train_step",
    # Dataset
    "AudioExample",
    "AudioBatch",
    "AudioDataset",
    "PreEncodedDataset",
    # Trainer
    "TrainingConfig",
    "TrainingState",
    "ACEStepTrainer",
]
