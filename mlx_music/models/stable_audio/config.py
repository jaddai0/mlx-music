"""
Configuration classes for Stable Audio Open.

Contains dataclasses for the main model, transformer, VAE, and conditioning components.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import json


@dataclass
class DiTConfig:
    """Configuration for StableAudioDiT transformer.

    Based on stabilityai/stable-audio-open-1.0 architecture.
    """

    # Core dimensions
    sample_size: int = 1024
    in_channels: int = 64
    out_channels: int = 64
    num_layers: int = 24
    attention_head_dim: int = 64
    num_attention_heads: int = 24
    num_key_value_heads: int = 12  # GQA: fewer KV heads than query heads

    # Hidden dimension is num_attention_heads * attention_head_dim = 1536

    # Cross-attention for text conditioning
    cross_attention_dim: int = 768

    # Feed-forward network
    ff_mult: float = 4.0  # FFN hidden = hidden_size * ff_mult

    # Position embeddings (rotary)
    max_seq_len: int = 8192

    # Global conditioning
    global_states_input_dim: int = 1536  # projection model output

    # Timestep embedding
    timestep_features_dim: int = 256

    @classmethod
    def from_dict(cls, config: Dict[str, Any]) -> "DiTConfig":
        """Create config from dictionary."""
        return cls(**{k: v for k, v in config.items() if k in cls.__dataclass_fields__})


@dataclass
class VAEConfig:
    """Configuration for AutoencoderOobleck (VAE).

    Oobleck uses 1D convolutions for audio processing.
    """

    # Encoder/decoder
    encoder_hidden_size: int = 128
    decoder_hidden_size: int = 128
    latent_channels: int = 64

    # Audio I/O
    audio_channels: int = 2  # Stereo

    # Downsampling ratios for encoder
    downsampling_ratios: Tuple[int, ...] = (2, 4, 4, 8, 8)
    # Total compression: 2 * 4 * 4 * 8 * 8 = 2048

    # Channel multipliers for encoder/decoder blocks
    channel_multiples: Tuple[int, ...] = (1, 2, 4, 8, 16)

    @classmethod
    def from_dict(cls, config: Dict[str, Any]) -> "VAEConfig":
        """Create config from dictionary."""
        if "downsampling_ratios" in config and isinstance(config["downsampling_ratios"], list):
            config["downsampling_ratios"] = tuple(config["downsampling_ratios"])
        if "channel_multiples" in config and isinstance(config["channel_multiples"], list):
            config["channel_multiples"] = tuple(config["channel_multiples"])

        result = cls(**{k: v for k, v in config.items() if k in cls.__dataclass_fields__})

        # Validate tuple lengths match (required for encoder/decoder symmetry)
        if len(result.downsampling_ratios) != len(result.channel_multiples):
            raise ValueError(
                f"downsampling_ratios length ({len(result.downsampling_ratios)}) "
                f"must match channel_multiples length ({len(result.channel_multiples)})"
            )

        return result


@dataclass
class ProjectionConfig:
    """Configuration for the conditioning projection model."""

    # Input dimensions
    text_encoder_dim: int = 768  # T5 encoder output dimension

    # Timing conditioning
    num_timing_features: int = 2  # seconds_start, seconds_total

    # Output dimension (matches transformer global_states_input_dim)
    output_dim: int = 1536

    @classmethod
    def from_dict(cls, config: Dict[str, Any]) -> "ProjectionConfig":
        """Create config from dictionary."""
        return cls(**{k: v for k, v in config.items() if k in cls.__dataclass_fields__})


@dataclass
class StableAudioConfig:
    """Main configuration for Stable Audio Open model.

    Contains sub-configurations for all components.
    """

    # Sub-component configs
    transformer: DiTConfig = field(default_factory=DiTConfig)
    vae: VAEConfig = field(default_factory=VAEConfig)
    projection: ProjectionConfig = field(default_factory=ProjectionConfig)

    # Audio parameters
    sample_rate: int = 44100
    max_duration_seconds: float = 47.0  # Model limit

    # Generation defaults
    default_num_inference_steps: int = 100
    default_guidance_scale: float = 7.0

    @classmethod
    def from_dict(cls, config: Dict[str, Any]) -> "StableAudioConfig":
        """Create config from dictionary."""
        transformer_config = config.get("transformer", {})
        vae_config = config.get("vae", {})
        projection_config = config.get("projection", {})

        return cls(
            transformer=DiTConfig.from_dict(transformer_config) if transformer_config else DiTConfig(),
            vae=VAEConfig.from_dict(vae_config) if vae_config else VAEConfig(),
            projection=ProjectionConfig.from_dict(projection_config) if projection_config else ProjectionConfig(),
            sample_rate=config.get("sample_rate", 44100),
            max_duration_seconds=config.get("max_duration_seconds", 47.0),
            default_num_inference_steps=config.get("default_num_inference_steps", 100),
            default_guidance_scale=config.get("default_guidance_scale", 7.0),
        )

    @classmethod
    def from_pretrained(cls, model_path: Union[str, Path]) -> "StableAudioConfig":
        """Load config from pretrained model directory.

        Args:
            model_path: Path to model directory containing config files

        Returns:
            StableAudioConfig instance

        Raises:
            FileNotFoundError: If model_path doesn't exist
            ValueError: If config files are malformed
        """
        model_path = Path(model_path)

        if not model_path.exists():
            raise FileNotFoundError(f"Model path does not exist: {model_path}")

        if not model_path.is_dir():
            raise ValueError(f"Model path must be a directory: {model_path}")

        # Maximum config file size (10MB) to prevent memory exhaustion
        MAX_CONFIG_SIZE = 10 * 1024 * 1024

        def load_json_config(path: Path) -> Dict[str, Any]:
            """Load JSON config with size validation."""
            if not path.exists():
                return {}

            file_size = path.stat().st_size
            if file_size > MAX_CONFIG_SIZE:
                raise ValueError(
                    f"Config file too large: {path} ({file_size} bytes). "
                    f"Maximum allowed: {MAX_CONFIG_SIZE} bytes"
                )

            try:
                with open(path) as f:
                    return json.load(f)
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON in {path}: {e}")

        # Try to load from model_index.json (HuggingFace diffusers format)
        model_index_path = model_path / "model_index.json"
        if model_index_path.exists():
            load_json_config(model_index_path)  # Validate but ignore for now

        # Load transformer config
        transformer_config_path = model_path / "transformer" / "config.json"
        transformer_config = load_json_config(transformer_config_path)

        # Load VAE config
        vae_config_path = model_path / "vae" / "config.json"
        vae_config = load_json_config(vae_config_path)

        # Load projection config
        projection_config_path = model_path / "projection_model" / "config.json"
        projection_config = load_json_config(projection_config_path)

        return cls.from_dict({
            "transformer": transformer_config,
            "vae": vae_config,
            "projection": projection_config,
        })


# Scheduler configuration
@dataclass
class EDMSchedulerConfig:
    """Configuration for EDM DPM-Solver scheduler."""

    sigma_min: float = 0.3
    sigma_max: float = 500.0
    sigma_data: float = 1.0

    # Karras sigmas
    rho: float = 7.0

    # Solver settings
    algorithm_type: str = "dpmsolver++"
    solver_order: int = 2

    # EDM scaling
    prediction_type: str = "v_prediction"  # Stable Audio uses v-prediction

    @classmethod
    def from_dict(cls, config: Dict[str, Any]) -> "EDMSchedulerConfig":
        """Create config from dictionary."""
        return cls(**{k: v for k, v in config.items() if k in cls.__dataclass_fields__})


__all__ = [
    "DiTConfig",
    "VAEConfig",
    "ProjectionConfig",
    "StableAudioConfig",
    "EDMSchedulerConfig",
]
