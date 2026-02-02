"""
Common training utilities for mlx-music.

Provides EMA, learning rate scheduling, gradient accumulation,
and loss functions shared across model types.
"""

from .ema import EMAModel, NoOpEMA, create_ema
from .lr_scheduler import (
    LRScheduler,
    CosineAnnealingLR,
    OneCycleLR,
    LinearWarmupLR,
    ConstantLR,
    create_scheduler,
)
from .gradient_accumulator import (
    GradientAccumulator,
    NoOpAccumulator,
    create_accumulator,
)
from .loss import (
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
