"""
Training infrastructure for mlx-music models.

Provides LoRA fine-tuning, dataset loading, and optimization utilities
for ACE-Step, Stable Audio, and MusicGen.
"""

from .common import (
    # EMA
    EMAModel,
    NoOpEMA,
    create_ema,
    # LR Schedulers
    LRScheduler,
    CosineAnnealingLR,
    OneCycleLR,
    LinearWarmupLR,
    ConstantLR,
    create_scheduler,
    # Gradient Accumulation
    GradientAccumulator,
    NoOpAccumulator,
    create_accumulator,
    # Loss Functions
    flow_matching_loss,
    v_prediction_loss,
    snr_weighted_loss,
    min_snr_weighted_loss,
)

__all__ = [
    # EMA
    "EMAModel",
    "NoOpEMA",
    "create_ema",
    # LR Schedulers
    "LRScheduler",
    "CosineAnnealingLR",
    "OneCycleLR",
    "LinearWarmupLR",
    "ConstantLR",
    "create_scheduler",
    # Gradient Accumulation
    "GradientAccumulator",
    "NoOpAccumulator",
    "create_accumulator",
    # Loss Functions
    "flow_matching_loss",
    "v_prediction_loss",
    "snr_weighted_loss",
    "min_snr_weighted_loss",
]
