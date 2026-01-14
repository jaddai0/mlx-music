"""
ACE-Step model implementation for MLX.

ACE-Step is a diffusion-based music generation model that transforms
text prompts and lyrics into full songs.

Architecture:
- Linear Transformer (24 blocks, 2560 dim)
- DCAE (Deep Compression AutoEncoder) for audio latents
- HiFi-GAN vocoder for audio synthesis
- UMT5 text encoder for text conditioning
"""

from mlx_music.models.ace_step.dcae import DCAE, DCAEConfig
from mlx_music.models.ace_step.model import ACEStep, GenerationConfig, GenerationOutput
from mlx_music.models.ace_step.scheduler import (
    FlowMatchEulerDiscreteScheduler,
    FlowMatchHeunDiscreteScheduler,
)
from mlx_music.models.ace_step.text_encoder import (
    UMT5TextEncoder,
    PlaceholderTextEncoder,
    get_text_encoder,
)
from mlx_music.models.ace_step.transformer import ACEStepConfig, ACEStepTransformer
from mlx_music.models.ace_step.vocoder import HiFiGANVocoder, MusicDCAEPipeline, VocoderConfig

__all__ = [
    "ACEStep",
    "ACEStepConfig",
    "ACEStepTransformer",
    "DCAE",
    "DCAEConfig",
    "FlowMatchEulerDiscreteScheduler",
    "FlowMatchHeunDiscreteScheduler",
    "GenerationConfig",
    "GenerationOutput",
    "HiFiGANVocoder",
    "MusicDCAEPipeline",
    "PlaceholderTextEncoder",
    "UMT5TextEncoder",
    "VocoderConfig",
    "get_text_encoder",
]
