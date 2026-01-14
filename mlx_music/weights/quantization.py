"""
Quantization utilities for mlx-music.

Uses MLX's native quantization for efficient inference:
- INT4: 4-bit quantization, ~4x memory reduction
- INT8: 8-bit quantization, ~2x memory reduction, higher quality
- Mixed: INT8 for attention, INT4 for FFN (balanced)

The native quantization actually stores weights in lower precision
and uses optimized kernels for faster inference.
"""

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import mlx.core as mx
import mlx.nn as nn


class QuantizationMode(Enum):
    """Quantization modes."""

    NONE = "none"
    INT4 = "int4"
    INT8 = "int8"
    MIXED = "mixed"  # INT8 attention + INT4 FFN


@dataclass
class QuantizationConfig:
    """Configuration for model quantization."""

    mode: QuantizationMode = QuantizationMode.NONE

    # Per-component settings (for mixed mode)
    attention_bits: int = 8
    ffn_bits: int = 4
    embedding_bits: int = 8

    # Group size for quantization (smaller = more accurate, larger = faster)
    group_size: int = 64

    # Layers to exclude from quantization (by name substring)
    exclude_layers: List[str] = field(default_factory=lambda: [
        "rotary_emb",
        "time_proj",
        "norm",
        "lyric_embs",
        "gamma",  # Layer scale parameters
    ])

    @classmethod
    def for_quality(cls) -> "QuantizationConfig":
        """Config prioritizing output quality (INT8)."""
        return cls(
            mode=QuantizationMode.INT8,
            attention_bits=8,
            ffn_bits=8,
            group_size=64,
        )

    @classmethod
    def for_speed(cls) -> "QuantizationConfig":
        """Config prioritizing inference speed (INT4)."""
        return cls(
            mode=QuantizationMode.INT4,
            attention_bits=4,
            ffn_bits=4,
            group_size=64,
        )

    @classmethod
    def for_balanced(cls) -> "QuantizationConfig":
        """Balanced config - INT8 attention, INT4 FFN (recommended)."""
        return cls(
            mode=QuantizationMode.MIXED,
            attention_bits=8,
            ffn_bits=4,
            group_size=64,
        )


def _should_exclude(name: str, config: QuantizationConfig) -> bool:
    """Check if a layer should be excluded from quantization."""
    for exclude in config.exclude_layers:
        if exclude in name:
            return True
    return False


def _get_bits_for_layer(name: str, config: QuantizationConfig) -> int:
    """Get quantization bits for a specific layer."""
    if config.mode == QuantizationMode.NONE:
        return 16  # No quantization
    elif config.mode == QuantizationMode.INT4:
        return 4
    elif config.mode == QuantizationMode.INT8:
        return 8
    else:  # MIXED
        if "attn" in name.lower() or "attention" in name.lower():
            return config.attention_bits
        elif "ff" in name.lower() or "mlp" in name.lower() or "conv" in name.lower():
            return config.ffn_bits
        else:
            return config.embedding_bits


def quantize_model(
    model: nn.Module,
    config: Optional[QuantizationConfig] = None,
    bits: Optional[int] = None,
) -> nn.Module:
    """
    Quantize a model using MLX's native quantization.

    This uses MLX's optimized quantized kernels for actual memory
    reduction and faster inference, not fake quantization.

    Args:
        model: Model to quantize (modified in-place)
        config: Quantization configuration
        bits: Override bits (4 or 8), ignored if config provided

    Returns:
        Quantized model (same object, modified in-place)

    Example:
        >>> from mlx_music.weights.quantization import quantize_model, QuantizationConfig
        >>> model = ACEStepTransformer(config)
        >>> model.load_weights(...)
        >>> quantize_model(model, QuantizationConfig.for_balanced())
    """
    if config is None:
        if bits is None:
            bits = 4
        config = QuantizationConfig(
            mode=QuantizationMode.INT4 if bits == 4 else QuantizationMode.INT8
        )

    if config.mode == QuantizationMode.NONE:
        return model

    # Create predicate for nn.quantize
    def class_predicate(path: str, module: nn.Module) -> Union[bool, dict]:
        """Determine if/how to quantize a module.

        Always returns a dict with explicit bits to avoid relying on
        the default bits parameter in nn.quantize().
        """
        if _should_exclude(path, config):
            return False

        # Only quantize Linear layers (Conv layers don't support quantization yet)
        if not isinstance(module, nn.Linear):
            return False

        layer_bits = _get_bits_for_layer(path, config)
        if layer_bits >= 16:
            return False

        # Always return dict with explicit bits - don't rely on nn.quantize default
        return {"group_size": config.group_size, "bits": layer_bits}

    # Apply quantization - class_predicate always returns dict with bits,
    # so these defaults are only fallbacks and shouldn't be used
    nn.quantize(
        model,
        class_predicate=class_predicate,
    )

    return model


def get_model_size(model: nn.Module) -> Tuple[int, float]:
    """
    Get model size in parameters and bytes.

    Returns:
        Tuple of (num_parameters, size_in_mb)
    """
    num_params = 0
    total_bytes = 0

    def count_params(params, prefix=""):
        nonlocal num_params, total_bytes
        for name, value in params.items():
            full_name = f"{prefix}.{name}" if prefix else name
            if isinstance(value, dict):
                count_params(value, full_name)
            elif isinstance(value, mx.array):
                num_params += value.size
                total_bytes += value.nbytes

    count_params(model.parameters())

    return num_params, total_bytes / (1024 * 1024)


def save_quantized_model(
    model: nn.Module,
    output_path: Union[str, Path],
    config: Optional[QuantizationConfig] = None,
) -> Path:
    """
    Save a quantized model to disk.

    Args:
        model: Quantized model to save
        output_path: Output directory or file path
        config: Optional config to save alongside weights

    Returns:
        Path to saved weights file
    """
    import json

    output_path = Path(output_path)
    if output_path.suffix != ".safetensors":
        output_path = output_path / "model.safetensors"

    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Flatten parameters for saving
    def flatten_params(params, prefix=""):
        flat = {}
        for name, value in params.items():
            full_name = f"{prefix}.{name}" if prefix else name
            if isinstance(value, dict):
                flat.update(flatten_params(value, full_name))
            elif isinstance(value, mx.array):
                flat[full_name] = value
        return flat

    flat_params = flatten_params(model.parameters())

    # Save weights
    mx.save_safetensors(str(output_path), flat_params)

    # Save config if provided
    if config is not None:
        config_path = output_path.parent / "quantization_config.json"
        config_dict = {
            "mode": config.mode.value,
            "attention_bits": config.attention_bits,
            "ffn_bits": config.ffn_bits,
            "embedding_bits": config.embedding_bits,
            "group_size": config.group_size,
            "exclude_layers": config.exclude_layers,
        }
        with open(config_path, "w") as f:
            json.dump(config_dict, f, indent=2)

    return output_path


def load_quantization_config(path: Union[str, Path]) -> Optional[QuantizationConfig]:
    """Load quantization config from a directory."""
    import json

    path = Path(path)
    config_path = path / "quantization_config.json" if path.is_dir() else path

    if not config_path.exists():
        return None

    with open(config_path) as f:
        config_dict = json.load(f)

    return QuantizationConfig(
        mode=QuantizationMode(config_dict["mode"]),
        attention_bits=config_dict.get("attention_bits", 8),
        ffn_bits=config_dict.get("ffn_bits", 4),
        embedding_bits=config_dict.get("embedding_bits", 8),
        group_size=config_dict.get("group_size", 64),
        exclude_layers=config_dict.get("exclude_layers", []),
    )
