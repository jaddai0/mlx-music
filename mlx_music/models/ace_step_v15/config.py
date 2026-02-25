"""ACE-Step v1.5 model configuration for MLX."""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional


@dataclass
class AceStepV15Config:
    """Configuration for ACE-Step v1.5 DiT model.

    All variants (turbo, base, sft, shift1, shift3, continuous) share the same
    architecture. The difference is in training (is_turbo) and runtime inference
    parameters (shift, inference_steps).
    """

    # Core transformer dimensions
    vocab_size: int = 64003
    hidden_size: int = 2048
    intermediate_size: int = 6144
    num_hidden_layers: int = 24
    num_attention_heads: int = 16
    num_key_value_heads: int = 8
    head_dim: int = 128
    hidden_act: str = "silu"

    # Position embeddings
    max_position_embeddings: int = 32768
    rope_theta: float = 1000000.0
    rope_scaling: Optional[Dict] = None

    # Attention configuration
    attention_bias: bool = False
    attention_dropout: float = 0.0
    use_sliding_window: bool = True
    sliding_window: int = 128
    layer_types: Optional[List[str]] = None

    # FSQ (Finite Scalar Quantization)
    fsq_dim: int = 2048
    fsq_input_levels: List[int] = field(default_factory=lambda: [8, 8, 8, 5, 5, 5])
    fsq_input_num_quantizers: int = 1

    # Text encoder
    text_hidden_dim: int = 1024

    # Lyric encoder
    num_lyric_encoder_hidden_layers: int = 8
    patch_size: int = 2

    # Audio acoustic / DiT
    audio_acoustic_hidden_dim: int = 64
    pool_window_size: int = 5
    in_channels: int = 192
    num_audio_decoder_hidden_layers: int = 24

    # Timbre encoder
    timbre_hidden_dim: int = 64
    num_timbre_encoder_hidden_layers: int = 4
    timbre_fix_frame: int = 750
    num_attention_pooler_hidden_layers: int = 2

    # Flow matching
    data_proportion: float = 0.5
    timestep_mu: float = -0.4
    timestep_sigma: float = 1.0

    # Initialization
    initializer_range: float = 0.02
    rms_norm_eps: float = 1e-6

    # Variant flags
    is_turbo: bool = True
    model_version: str = "turbo"
    dtype: str = "bfloat16"

    # Cache
    use_cache: bool = True
    tie_word_embeddings: bool = True

    def __post_init__(self):
        if self.num_key_value_heads is None:
            self.num_key_value_heads = self.num_attention_heads

        if self.layer_types is None:
            self.layer_types = [
                "sliding_attention" if (i + 1) % 2 else "full_attention"
                for i in range(self.num_hidden_layers)
            ]

        if self.use_sliding_window is False:
            self.sliding_window = None

    @property
    def num_kv_groups(self) -> int:
        """Number of query heads per KV head (for GQA)."""
        return self.num_attention_heads // self.num_key_value_heads

    @property
    def fsq_codebook_size(self) -> int:
        """Total FSQ codebook size = product of all levels."""
        result = 1
        for level in self.fsq_input_levels:
            result *= level
        return result

    @property
    def latent_hz(self) -> float:
        """Latent frame rate after VAE compression (25 Hz for Oobleck)."""
        return 25.0

    @property
    def audio_code_hz(self) -> float:
        """Audio code rate after pooling (5 Hz = 25 / pool_window_size)."""
        return self.latent_hz / self.pool_window_size

    @classmethod
    def from_dict(cls, config_dict: Dict) -> "AceStepV15Config":
        """Create config from a dictionary, ignoring unknown keys."""
        valid_fields = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in config_dict.items() if k in valid_fields}
        return cls(**filtered)

    @classmethod
    def from_json(cls, path: str | Path) -> "AceStepV15Config":
        """Load config from a JSON file (e.g., config.json from HuggingFace)."""
        path = Path(path)
        with open(path) as f:
            config_dict = json.load(f)
        return cls.from_dict(config_dict)

    def to_dict(self) -> Dict:
        """Serialize config to dictionary."""
        from dataclasses import asdict
        return asdict(self)


# Variant presets - these define inference-time defaults, not architecture differences.
# All variants share the same architecture; weights differ due to training.
VARIANT_DEFAULTS = {
    "turbo": {
        "is_turbo": True,
        "model_version": "turbo",
        "default_inference_steps": 8,
        "default_shift": 1.0,
        "supports_cfg": False,
        "supported_tasks": ["text2music", "repaint", "cover"],
    },
    "base": {
        "is_turbo": False,
        "model_version": "base",
        "default_inference_steps": 50,
        "default_shift": 1.0,
        "supports_cfg": True,
        "supported_tasks": ["text2music", "repaint", "cover", "extract", "lego", "complete"],
    },
    "sft": {
        "is_turbo": False,
        "model_version": "sft",
        "default_inference_steps": 50,
        "default_shift": 1.0,
        "supports_cfg": True,
        "supported_tasks": ["text2music", "repaint", "cover", "extract", "lego", "complete"],
    },
    "turbo-shift1": {
        "is_turbo": True,
        "model_version": "turbo",
        "default_inference_steps": 8,
        "default_shift": 1.0,
        "supports_cfg": False,
        "supported_tasks": ["text2music", "repaint", "cover"],
    },
    "turbo-shift3": {
        "is_turbo": True,
        "model_version": "turbo",
        "default_inference_steps": 8,
        "default_shift": 3.0,
        "supports_cfg": False,
        "supported_tasks": ["text2music", "repaint", "cover"],
    },
    "turbo-continuous": {
        "is_turbo": True,
        "model_version": "turbo",
        "default_inference_steps": 8,
        "default_shift": 1.0,
        "supports_cfg": False,
        "supported_tasks": ["text2music", "repaint", "cover"],
    },
}

# LM model sizes and their HuggingFace repo names
LM_VARIANTS = {
    "0.6B": "acestep-5Hz-lm-0.6B",
    "1.7B": "acestep-5Hz-lm-1.7B",
    "4B": "acestep-5Hz-lm-4B",
}


__all__ = ["AceStepV15Config", "VARIANT_DEFAULTS", "LM_VARIANTS"]
