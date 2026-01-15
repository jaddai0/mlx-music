"""
Stable Audio Open implementation for MLX.

A diffusion-based text-to-audio model from Stability AI.

Example:
    >>> from mlx_music import StableAudio
    >>> model = StableAudio.from_pretrained("stabilityai/stable-audio-open-1.0")
    >>> output = model.generate(
    ...     prompt="ambient electronic music with soft pads",
    ...     duration=30.0,
    ... )
    >>> import soundfile as sf
    >>> sf.write("output.wav", output.audio.T, output.sample_rate)
"""

from mlx_music.models.stable_audio.config import (
    DiTConfig,
    EDMSchedulerConfig,
    ProjectionConfig,
    StableAudioConfig,
    VAEConfig,
)
from mlx_music.models.stable_audio.conditioning import (
    ConditioningManager,
    NumberEmbedding,
    ProjectionModel,
    TimestepEmbedding,
)
from mlx_music.models.stable_audio.model import (
    GenerationOutput,
    StableAudio,
)
from mlx_music.models.stable_audio.scheduler import (
    EDMDPMSolverMultistepScheduler,
    SchedulerOutput,
    retrieve_timesteps,
)
from mlx_music.models.stable_audio.transformer import (
    CrossAttention,
    DiTBlock,
    GQAttention,
    RMSNorm,
    RotaryEmbedding,
    StableAudioDiT,
)
from mlx_music.models.stable_audio.vae import (
    AutoencoderOobleck,
    OobleckDecoder,
    OobleckEncoder,
    Snake1d,
    snake,
)

__all__ = [
    # Main model
    "StableAudio",
    "GenerationOutput",
    # Configuration
    "StableAudioConfig",
    "DiTConfig",
    "VAEConfig",
    "ProjectionConfig",
    "EDMSchedulerConfig",
    # Transformer
    "StableAudioDiT",
    "DiTBlock",
    "GQAttention",
    "CrossAttention",
    "RotaryEmbedding",
    "RMSNorm",
    # VAE
    "AutoencoderOobleck",
    "OobleckEncoder",
    "OobleckDecoder",
    "Snake1d",
    "snake",
    # Conditioning
    "ProjectionModel",
    "TimestepEmbedding",
    "NumberEmbedding",
    "ConditioningManager",
    # Scheduler
    "EDMDPMSolverMultistepScheduler",
    "SchedulerOutput",
    "retrieve_timesteps",
]
