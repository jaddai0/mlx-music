"""
MusicGen MLX implementation.

Text-to-music generation using autoregressive language modeling.
"""

from .config import (
    MusicGenConfig,
    MusicGenDecoderConfig,
    MusicGenAudioEncoderConfig,
    MusicGenTextEncoderConfig,
)
from .model import MusicGen
from .generation import GenerationConfig, GenerationOutput
from .kv_cache import (
    KVCache,
    CrossAttentionCache,
    MusicGenCacheState,
    create_causal_mask,
)

__all__ = [
    "MusicGen",
    "MusicGenConfig",
    "MusicGenDecoderConfig",
    "MusicGenAudioEncoderConfig",
    "MusicGenTextEncoderConfig",
    "GenerationConfig",
    "GenerationOutput",
    "KVCache",
    "CrossAttentionCache",
    "MusicGenCacheState",
    "create_causal_mask",
]
