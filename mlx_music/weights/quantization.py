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

    # Track if we've shown the Conv warning
    conv_warning_shown = [False]

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
            # Warn once about Conv layers being skipped
            if isinstance(module, (nn.Conv1d, nn.Conv2d)) and not conv_warning_shown[0]:
                import warnings
                warnings.warn(
                    "Conv layers (Conv1d, Conv2d) detected but not quantized. "
                    "MLX native quantization currently only supports Linear layers. "
                    "Memory savings may be less than expected."
                )
                conv_warning_shown[0] = True
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


# =============================================================================
# Conv1d Quantization Support (Experimental)
# =============================================================================

class QuantizedConv1d(nn.Module):
    """
    Quantized Conv1d layer for kernel_size=1 (pointwise convolution).

    Conv1d with kernel_size=1 is mathematically equivalent to a Linear layer
    applied along the channel dimension. This allows us to quantize pointwise
    convolutions using MLX's quantized linear operations.

    Note: Only kernel_size=1 is supported. Larger kernels require different
    treatment and are not quantized.
    """

    # Default quantization parameters
    DEFAULT_BITS: int = 4
    DEFAULT_GROUP_SIZE: int = 64

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        bits: int = DEFAULT_BITS,
        group_size: int = DEFAULT_GROUP_SIZE,
        bias: bool = True,
    ):
        """
        Initialize quantized Conv1d.

        Args:
            in_channels: Number of input channels
            out_channels: Number of output channels
            bits: Quantization bits (4 or 8)
            group_size: Group size for quantization
            bias: Whether to include bias

        Raises:
            ValueError: If bits or group_size are invalid
        """
        super().__init__()

        # Validate parameters
        if bits not in (4, 8):
            raise ValueError(f"bits must be 4 or 8, got {bits}")
        if group_size < 1 or group_size > 256:
            raise ValueError(f"group_size must be 1-256, got {group_size}")

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.bits = bits
        self.group_size = group_size
        self._has_bias = bias

        # Lazy initialization: Linear layer created on first use or when weights loaded.
        # Design tradeoff: Small latency spike on first forward pass in exchange for
        # avoiding duplicate memory allocation during from_conv1d() weight conversion.
        # For inference pipelines with strict latency requirements, call warmup()
        # or _ensure_linear() explicitly after loading weights.
        self._linear: Optional[nn.Linear] = None
        self._is_quantized = False

    def _ensure_linear(self):
        """Lazily initialize the underlying Linear layer."""
        if self._linear is None:
            self._linear = nn.Linear(self.in_channels, self.out_channels, bias=self._has_bias)

    def quantize_weights(self):
        """Quantize the underlying linear layer weights."""
        self._ensure_linear()
        if not self._is_quantized:
            nn.quantize(
                self._linear,
                bits=self.bits,
                group_size=self.group_size,
            )
            self._is_quantized = True

    def warmup(self):
        """
        Eagerly initialize the underlying Linear layer.

        Call this after loading weights to ensure predictable first-call latency.
        Useful for inference pipelines with strict latency requirements.
        """
        self._ensure_linear()

    def __call__(self, x: mx.array) -> mx.array:
        """
        Apply quantized pointwise convolution.

        Args:
            x: Input of shape (batch, seq_len, in_channels) - NLC format

        Returns:
            Output of shape (batch, seq_len, out_channels)
        """
        self._ensure_linear()
        # For kernel_size=1, Conv1d is just a linear projection
        # Input: (batch, seq, in_channels)
        # Output: (batch, seq, out_channels)
        return self._linear(x)

    @classmethod
    def from_conv1d(
        cls,
        conv: nn.Conv1d,
        bits: int = 4,
        group_size: int = 64,
    ) -> "QuantizedConv1d":
        """
        Create a QuantizedConv1d from an existing Conv1d layer.

        Only works for kernel_size=1 convolutions.

        Args:
            conv: Source Conv1d layer
            bits: Quantization bits
            group_size: Group size for quantization

        Returns:
            QuantizedConv1d instance

        Raises:
            ValueError: If kernel_size != 1
        """
        # Check kernel size
        # MLX Conv1d weight shape is (out_channels, kernel_size, in_channels)
        kernel_size = conv.weight.shape[1]
        if kernel_size != 1:
            raise ValueError(
                f"QuantizedConv1d only supports kernel_size=1, got {kernel_size}"
            )

        in_channels = conv.weight.shape[2]
        out_channels = conv.weight.shape[0]
        has_bias = hasattr(conv, "bias") and conv.bias is not None

        # Create quantized layer
        quantized = cls(
            in_channels=in_channels,
            out_channels=out_channels,
            bits=bits,
            group_size=group_size,
            bias=has_bias,
        )

        # Initialize linear layer directly with weights (skip lazy init)
        quantized._linear = nn.Linear(in_channels, out_channels, bias=has_bias)

        # Copy weights (squeeze kernel dimension)
        # Conv1d weight: (out, kernel=1, in) -> Linear weight: (out, in)
        quantized._linear.weight = conv.weight.squeeze(1)
        if has_bias:
            quantized._linear.bias = conv.bias

        # Quantize
        quantized.quantize_weights()

        return quantized


def quantize_conv1d_layers(
    model: nn.Module,
    config: Optional[QuantizationConfig] = None,
    bits: int = 4,
    group_size: int = 64,
    verbose: bool = True,
) -> Tuple[nn.Module, int]:
    """
    Quantize Conv1d layers with kernel_size=1 in a model.

    This is a separate pass that handles Conv1d layers which MLX's
    native nn.quantize() doesn't support.

    Args:
        model: Model to process (modified in-place)
        config: Optional quantization config
        bits: Quantization bits if config not provided
        group_size: Group size if config not provided
        verbose: Whether to print progress

    Returns:
        Tuple of (model, num_layers_quantized)
    """
    if config is not None:
        bits = 4 if config.mode == QuantizationMode.INT4 else 8
        group_size = config.group_size

    num_quantized = 0
    num_skipped = 0
    visited = set()

    def process_module(module: nn.Module, path: str = ""):
        nonlocal num_quantized, num_skipped

        # Prevent cycles
        module_id = id(module)
        if module_id in visited:
            return
        visited.add(module_id)

        # Use MLX's children() method to get child modules
        if not hasattr(module, 'children') or not callable(module.children):
            return

        children_dict = module.children()
        if not isinstance(children_dict, dict):
            return

        for name, child in children_dict.items():
            full_path = f"{path}.{name}" if path else name

            if isinstance(child, nn.Conv1d):
                import logging

                # MLX Conv1d weight shape is (out_channels, kernel_size, in_channels)
                kernel_size = child.weight.shape[1]
                if kernel_size == 1:
                    try:
                        quantized = QuantizedConv1d.from_conv1d(
                            child, bits=bits, group_size=group_size
                        )
                        setattr(module, name, quantized)
                        num_quantized += 1
                        if verbose:
                            print(f"Quantized Conv1d: {full_path}")
                    except ValueError as e:
                        # Expected validation errors (invalid bits, group_size)
                        logging.warning(f"Skipped {full_path}: {e}")
                        num_skipped += 1
                    except Exception as e:
                        # Unexpected errors - log with full context for debugging
                        logging.error(f"Failed to quantize Conv1d {full_path}: {type(e).__name__}: {e}")
                        if verbose:
                            print(f"Failed to quantize {full_path}: {e}")
                        num_skipped += 1
                else:
                    num_skipped += 1
                    if verbose:
                        print(f"Skipped Conv1d (kernel_size={kernel_size}): {full_path}")

            elif isinstance(child, nn.Module):
                process_module(child, full_path)

            elif isinstance(child, list):
                for i, item in enumerate(child):
                    if isinstance(item, nn.Module):
                        process_module(item, f"{full_path}.{i}")

    process_module(model)

    if verbose:
        print(f"Conv1d quantization: {num_quantized} quantized, {num_skipped} skipped")

    return model, num_quantized


# =============================================================================
# Memory-Aware Quantization
# =============================================================================

@dataclass
class QuantizationStats:
    """Statistics from quantization process."""

    initial_params: int
    final_params: int
    initial_memory_mb: float
    final_memory_mb: float
    linear_layers_quantized: int
    conv1d_layers_quantized: int
    layers_skipped: int

    @property
    def memory_saved_mb(self) -> float:
        return self.initial_memory_mb - self.final_memory_mb

    @property
    def compression_ratio(self) -> float:
        if self.final_memory_mb == 0:
            return 0.0
        return self.initial_memory_mb / self.final_memory_mb

    def __repr__(self) -> str:
        return (
            f"QuantizationStats(\n"
            f"  initial_memory={self.initial_memory_mb:.2f} MB,\n"
            f"  final_memory={self.final_memory_mb:.2f} MB,\n"
            f"  memory_saved={self.memory_saved_mb:.2f} MB,\n"
            f"  compression_ratio={self.compression_ratio:.2f}x,\n"
            f"  linear_quantized={self.linear_layers_quantized},\n"
            f"  conv1d_quantized={self.conv1d_layers_quantized},\n"
            f"  skipped={self.layers_skipped}\n"
            f")"
        )


def get_metal_memory_info(cache_ttl_ms: float = 0.0) -> Dict[str, float]:
    """
    Get current Metal memory usage.

    Args:
        cache_ttl_ms: Cache TTL in milliseconds. If > 0, returns cached result
                      if last query was within TTL. Useful for training loops
                      where memory stats are logged frequently.

    Returns:
        Dict with memory info in MB. Returns zeros if Metal API unavailable.

    Note: For training loops, consider using cache_ttl_ms=100 to reduce API
    overhead while still capturing meaningful memory trends.
    """
    import logging
    import time

    # Simple time-based caching to reduce Metal API overhead
    if cache_ttl_ms > 0:
        now = time.time() * 1000
        if hasattr(get_metal_memory_info, "_cache"):
            cached_time, cached_result = get_metal_memory_info._cache
            if now - cached_time < cache_ttl_ms:
                return cached_result

    try:
        active = mx.metal.get_active_memory() / (1024 * 1024)
        peak = mx.metal.get_peak_memory() / (1024 * 1024)
        cache = mx.metal.get_cache_memory() / (1024 * 1024)
        result = {
            "active_mb": active,
            "peak_mb": peak,
            "cache_mb": cache,
        }
        # Store in cache if caching enabled
        if cache_ttl_ms > 0:
            get_metal_memory_info._cache = (time.time() * 1000, result)
        return result
    except AttributeError as e:
        # Metal API not available (e.g., running on non-Apple hardware)
        logging.debug(f"Metal memory API unavailable: {e}")
        return {
            "active_mb": 0.0,
            "peak_mb": 0.0,
            "cache_mb": 0.0,
        }
    except Exception as e:
        # Unexpected error - log for debugging
        logging.warning(f"Failed to get Metal memory info: {e}")
        return {
            "active_mb": 0.0,
            "peak_mb": 0.0,
            "cache_mb": 0.0,
        }


def quantize_model_with_stats(
    model: nn.Module,
    config: Optional[QuantizationConfig] = None,
    bits: Optional[int] = None,
    quantize_conv1d: bool = True,
    verbose: bool = True,
) -> Tuple[nn.Module, QuantizationStats]:
    """
    Quantize a model with memory tracking and statistics.

    This is a comprehensive quantization function that:
    1. Tracks memory before/after quantization
    2. Quantizes Linear layers using MLX's native quantization
    3. Optionally quantizes Conv1d layers (kernel_size=1 only)
    4. Returns detailed statistics

    Args:
        model: Model to quantize (modified in-place)
        config: Quantization configuration
        bits: Override bits (4 or 8), ignored if config provided
        quantize_conv1d: Whether to also quantize Conv1d layers
        verbose: Whether to print progress

    Returns:
        Tuple of (quantized_model, stats)

    Example:
        >>> model = ACEStepTransformer(config)
        >>> model.load_weights(...)
        >>> model, stats = quantize_model_with_stats(
        ...     model,
        ...     QuantizationConfig.for_balanced(),
        ...     verbose=True
        ... )
        >>> print(stats)
    """
    # Measure initial state
    initial_params, initial_memory_mb = get_model_size(model)
    initial_metal = get_metal_memory_info()

    if verbose:
        print(f"Initial model size: {initial_memory_mb:.2f} MB ({initial_params:,} params)")
        print(f"Initial Metal memory: {initial_metal['active_mb']:.2f} MB active")

    # Quantize Linear layers
    linear_count_before = sum(
        1 for _, m in _iter_modules(model) if isinstance(m, nn.Linear)
    )

    quantize_model(model, config=config, bits=bits)

    # Count quantized linear layers (now they're QuantizedLinear)
    linear_count_after = sum(
        1 for _, m in _iter_modules(model) if isinstance(m, nn.Linear)
    )
    linear_layers_quantized = linear_count_before - linear_count_after

    # Quantize Conv1d layers if requested
    conv1d_layers_quantized = 0
    if quantize_conv1d:
        _, conv1d_layers_quantized = quantize_conv1d_layers(
            model, config=config, verbose=verbose
        )

    # Measure final state
    final_params, final_memory_mb = get_model_size(model)
    final_metal = get_metal_memory_info()

    if verbose:
        print(f"Final model size: {final_memory_mb:.2f} MB ({final_params:,} params)")
        print(f"Final Metal memory: {final_metal['active_mb']:.2f} MB active")
        print(f"Memory saved: {initial_memory_mb - final_memory_mb:.2f} MB")

    stats = QuantizationStats(
        initial_params=initial_params,
        final_params=final_params,
        initial_memory_mb=initial_memory_mb,
        final_memory_mb=final_memory_mb,
        linear_layers_quantized=linear_layers_quantized,
        conv1d_layers_quantized=conv1d_layers_quantized,
        layers_skipped=0,  # TODO: track skipped layers
    )

    return model, stats


def _iter_modules(model: nn.Module, prefix: str = "", _visited: set = None):
    """Iterate over all modules in a model."""
    if _visited is None:
        _visited = set()

    # Avoid cycles
    model_id = id(model)
    if model_id in _visited:
        return
    _visited.add(model_id)

    yield prefix, model

    # Use MLX's children() method to get child modules
    if hasattr(model, 'children') and callable(model.children):
        children_dict = model.children()
        if isinstance(children_dict, dict):
            for name, child in children_dict.items():
                full_name = f"{prefix}.{name}" if prefix else name
                if isinstance(child, nn.Module):
                    yield from _iter_modules(child, full_name, _visited)
                elif isinstance(child, list):
                    for i, item in enumerate(child):
                        if isinstance(item, nn.Module):
                            yield from _iter_modules(item, f"{full_name}.{i}", _visited)
