"""
MusicGen configuration classes.

Defines configurations for decoder, audio encoder, and text encoder components.
"""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class MusicGenDecoderConfig:
    """Configuration for the MusicGen decoder (transformer LM)."""

    # Model architecture
    vocab_size: int = 2048
    hidden_size: int = 1024
    num_hidden_layers: int = 24
    num_attention_heads: int = 16
    ffn_dim: int = 4096
    max_position_embeddings: int = 2048

    # Audio-specific
    num_codebooks: int = 4
    audio_channels: int = 1

    # Special tokens
    pad_token_id: int = 2048
    bos_token_id: int = 2048

    # Regularization
    dropout: float = 0.1
    attention_dropout: float = 0.0
    activation_dropout: float = 0.0

    # Other
    activation_function: str = "gelu"
    scale_embedding: bool = False
    use_cache: bool = True

    @property
    def head_dim(self) -> int:
        """Calculate head dimension from hidden_size and num_attention_heads."""
        return self.hidden_size // self.num_attention_heads

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "MusicGenDecoderConfig":
        """Create config from dictionary, filtering unknown fields."""
        known_fields = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in config_dict.items() if k in known_fields}
        return cls(**filtered)


@dataclass
class MusicGenAudioEncoderConfig:
    """Configuration for the audio encoder (EnCodec)."""

    # Core settings
    sampling_rate: int = 32000
    audio_channels: int = 1
    codebook_size: int = 2048
    codebook_dim: int = 128

    # Architecture
    hidden_size: int = 128
    num_filters: int = 64
    num_residual_layers: int = 1
    num_lstm_layers: int = 2
    upsampling_ratios: List[int] = field(default_factory=lambda: [8, 5, 4, 4])

    # Convolution settings
    kernel_size: int = 7
    last_kernel_size: int = 7
    residual_kernel_size: int = 3
    dilation_growth_rate: int = 2
    use_causal_conv: bool = False
    use_conv_shortcut: bool = False
    pad_mode: str = "reflect"
    compress: int = 2
    norm_type: str = "weight_norm"

    # Bandwidth
    target_bandwidths: List[float] = field(default_factory=lambda: [2.2])

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "MusicGenAudioEncoderConfig":
        """Create config from dictionary, filtering unknown fields."""
        known_fields = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in config_dict.items() if k in known_fields}
        return cls(**filtered)


@dataclass
class MusicGenTextEncoderConfig:
    """Configuration for the text encoder (T5)."""

    # Model
    model_name: str = "t5-base"
    d_model: int = 768
    d_ff: int = 3072
    d_kv: int = 64
    num_heads: int = 12
    num_layers: int = 12
    vocab_size: int = 32128

    # Position
    n_positions: int = 512
    relative_attention_num_buckets: int = 32
    relative_attention_max_distance: int = 128

    # Regularization
    dropout_rate: float = 0.1
    layer_norm_epsilon: float = 1e-6

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "MusicGenTextEncoderConfig":
        """Create config from dictionary, filtering unknown fields."""
        known_fields = {f.name for f in cls.__dataclass_fields__.values()}
        # Handle _name_or_path -> model_name
        if "_name_or_path" in config_dict:
            config_dict = {**config_dict, "model_name": config_dict["_name_or_path"]}
        filtered = {k: v for k, v in config_dict.items() if k in known_fields}
        return cls(**filtered)


@dataclass
class MusicGenConfig:
    """
    Full MusicGen configuration containing all component configs.

    Supports both standard MusicGen and MusicGen-Melody variants.
    """

    # Component configs
    decoder: MusicGenDecoderConfig = field(default_factory=MusicGenDecoderConfig)
    audio_encoder: MusicGenAudioEncoderConfig = field(
        default_factory=MusicGenAudioEncoderConfig
    )
    text_encoder: MusicGenTextEncoderConfig = field(
        default_factory=MusicGenTextEncoderConfig
    )

    # Model type
    model_type: str = "musicgen"  # "musicgen" or "musicgen_melody"

    # Melody-specific (only used when model_type == "musicgen_melody")
    num_chroma: int = 12
    chroma_length: int = 235

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "MusicGenConfig":
        """Create full config from nested dictionary."""
        decoder_config = MusicGenDecoderConfig.from_dict(
            config_dict.get("decoder", {})
        )
        audio_encoder_config = MusicGenAudioEncoderConfig.from_dict(
            config_dict.get("audio_encoder", {})
        )
        text_encoder_config = MusicGenTextEncoderConfig.from_dict(
            config_dict.get("text_encoder", {})
        )

        return cls(
            decoder=decoder_config,
            audio_encoder=audio_encoder_config,
            text_encoder=text_encoder_config,
            model_type=config_dict.get("model_type", "musicgen"),
            num_chroma=config_dict.get("num_chroma", 12),
            chroma_length=config_dict.get("chroma_length", 235),
        )

    @classmethod
    def from_pretrained(cls, model_path: str) -> "MusicGenConfig":
        """Load config from pretrained model directory."""
        config_path = Path(model_path) / "config.json"
        if not config_path.exists():
            raise FileNotFoundError(f"Config not found at {config_path}")

        with open(config_path, "r") as f:
            config_dict = json.load(f)

        return cls.from_dict(config_dict)

    @property
    def is_melody(self) -> bool:
        """Check if this is a melody-conditioned model."""
        return self.model_type == "musicgen_melody"

    @property
    def hidden_size(self) -> int:
        """Convenience accessor for decoder hidden size."""
        return self.decoder.hidden_size

    @property
    def num_hidden_layers(self) -> int:
        """Convenience accessor for decoder layer count."""
        return self.decoder.num_hidden_layers

    @property
    def num_attention_heads(self) -> int:
        """Convenience accessor for decoder attention heads."""
        return self.decoder.num_attention_heads

    @property
    def num_codebooks(self) -> int:
        """Convenience accessor for number of audio codebooks."""
        return self.decoder.num_codebooks

    @property
    def sampling_rate(self) -> int:
        """Convenience accessor for audio sampling rate."""
        return self.audio_encoder.sampling_rate

    @property
    def frame_rate(self) -> int:
        """Calculate the audio frame rate (tokens per second)."""
        # Frame rate = sampling_rate / product(upsampling_ratios)
        import math

        hop_length = math.prod(self.audio_encoder.upsampling_ratios)
        return self.audio_encoder.sampling_rate // hop_length


# Preset configurations for common model variants
MUSICGEN_SMALL_CONFIG = MusicGenDecoderConfig(
    hidden_size=1024,
    num_hidden_layers=24,
    num_attention_heads=16,
    ffn_dim=4096,
)

MUSICGEN_MEDIUM_CONFIG = MusicGenDecoderConfig(
    hidden_size=1536,
    num_hidden_layers=48,
    num_attention_heads=24,
    ffn_dim=6144,
)

MUSICGEN_LARGE_CONFIG = MusicGenDecoderConfig(
    hidden_size=2048,
    num_hidden_layers=48,
    num_attention_heads=32,
    ffn_dim=8192,
)
