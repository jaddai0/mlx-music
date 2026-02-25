"""
Weight mapping and conversion for ACE-Step v1.5 PyTorch -> MLX.

Handles:
- Direct key mapping (PyTorch HF naming -> MLX naming)
- Conv1d weight transposition (NCL -> NLC)
- VAE weight normalization decomposition (weight_g, weight_v -> weight)
- Snake parameter reshaping ([1, C, 1] -> [C])
- Special parameter handling (scale_shift_table, null_condition_emb)
"""

import logging
from pathlib import Path
from typing import Dict, Optional, Tuple

import mlx.core as mx
import mlx.nn as nn

logger = logging.getLogger(__name__)


def _transpose_conv1d(weight: mx.array) -> mx.array:
    """Transpose Conv1d weight from PyTorch [out, in, kernel] to MLX [out, kernel, in]."""
    return mx.transpose(weight, axes=(0, 2, 1))


def _transpose_conv_transpose1d(weight: mx.array) -> mx.array:
    """Transpose ConvTranspose1d weight from PyTorch [in, out, kernel] to MLX [out, kernel, in]."""
    return mx.transpose(weight, axes=(1, 2, 0))


def _decompose_weight_norm(weight_g: mx.array, weight_v: mx.array) -> mx.array:
    """Decompose weight normalization: weight = g * (v / ||v||).

    PyTorch weight_norm stores:
    - weight_g: [out_channels, 1, 1] gain
    - weight_v: [out_channels, in_channels, kernel_size] direction

    Returns:
        Combined weight [out_channels, in_channels, kernel_size]
    """
    # Compute L2 norm over in_channels and kernel_size dims
    norm = mx.sqrt(mx.sum(weight_v * weight_v, axis=(1, 2), keepdims=True) + 1e-12)
    return weight_g * (weight_v / norm)


def _reshape_snake_param(param: mx.array) -> mx.array:
    """Reshape Snake parameter from [1, C, 1] to [C]."""
    return param.squeeze()


def map_dit_weights(pytorch_weights: Dict[str, mx.array]) -> Dict[str, mx.array]:
    """Map ACE-Step v1.5 model weights from PyTorch to MLX naming.

    The PyTorch model uses HuggingFace naming:
      decoder.layers.0.self_attn.q_proj.weight
      encoder.lyric_encoder.layers.0.self_attn.q_proj.weight
      tokenizer.attention_pooler.layers.0.self_attn.q_proj.weight
      detokenizer.layers.0.self_attn.q_proj.weight
      null_condition_emb

    The MLX model uses similar naming but needs:
    1. Conv1d weights transposed
    2. Special handling for proj_in (Sequential index 1 -> direct)
    3. Special handling for proj_out (Sequential index 1 -> direct)
    """
    mlx_weights = {}
    conv1d_keys = set()

    for key, value in pytorch_weights.items():
        new_key = key
        new_value = value

        # Handle proj_in: PyTorch uses Sequential with Lambda wrappers
        # decoder.proj_in.1.weight -> decoder.proj_in.weight
        # decoder.proj_in.1.bias -> decoder.proj_in.bias
        if "proj_in.1." in key:
            new_key = key.replace("proj_in.1.", "proj_in.")
            # Conv1d weight: [out_ch, in_ch, kernel] -> [out_ch, kernel, in_ch]
            if "weight" in new_key:
                new_value = _transpose_conv1d(value)
                conv1d_keys.add(new_key)

        # Handle proj_out: decoder.proj_out.1.weight -> decoder.proj_out.weight
        elif "proj_out.1." in key:
            new_key = key.replace("proj_out.1.", "proj_out.")
            # ConvTranspose1d weight: PyTorch [in_ch, out_ch, kernel] -> MLX [out_ch, kernel, in_ch]
            if "weight" in new_key:
                new_value = _transpose_conv_transpose1d(value)
                conv1d_keys.add(new_key)

        else:
            new_key = key

        mlx_weights[new_key] = new_value

    n_transposed = len(conv1d_keys)
    if n_transposed > 0:
        logger.info(f"Transposed {n_transposed} Conv1d weights")

    return mlx_weights


def map_vae_weights(pytorch_weights: Dict[str, mx.array]) -> Dict[str, mx.array]:
    """Map VAE weights from diffusers format to MLX.

    Handles:
    1. Weight normalization decomposition: weight_g + weight_v -> weight
    2. Conv1d/ConvTranspose1d weight transposition
    3. Snake parameter reshaping: [1, C, 1] -> [C]
    """
    mlx_weights = {}

    # First pass: collect weight_g and weight_v pairs
    wn_groups: Dict[str, Dict[str, mx.array]] = {}
    regular_keys = {}

    for key, value in pytorch_weights.items():
        if key.endswith(".weight_g"):
            base = key[: -len(".weight_g")]
            wn_groups.setdefault(base, {})["weight_g"] = value
        elif key.endswith(".weight_v"):
            base = key[: -len(".weight_v")]
            wn_groups.setdefault(base, {})["weight_v"] = value
        else:
            regular_keys[key] = value

    # Decompose weight normalization pairs
    n_decomposed = 0
    for base, parts in wn_groups.items():
        if "weight_g" in parts and "weight_v" in parts:
            combined = _decompose_weight_norm(parts["weight_g"], parts["weight_v"])
            # ConvTranspose1d layers (decoder.block.*.conv_t1) use different transposition
            if ".conv_t1" in base:
                # PyTorch ConvTranspose1d: [in, out, kernel] -> MLX: [out, kernel, in]
                combined = _transpose_conv_transpose1d(combined)
            else:
                # Conv1d: [out, in, kernel] -> MLX: [out, kernel, in]
                combined = _transpose_conv1d(combined)
            mlx_weights[f"{base}.weight"] = combined
            n_decomposed += 1
        else:
            # Incomplete pair, keep as-is
            for suffix, val in parts.items():
                mlx_weights[f"{base}.{suffix}"] = val

    # Process regular keys
    n_snake = 0
    for key, value in regular_keys.items():
        if ".snake" in key and key.endswith((".alpha", ".beta")):
            # Reshape snake params: [1, C, 1] -> [C]
            mlx_weights[key] = _reshape_snake_param(value)
            n_snake += 1
        else:
            mlx_weights[key] = value

    logger.info(
        f"VAE weight conversion: {n_decomposed} weight norms decomposed, "
        f"{n_snake} snake params reshaped"
    )
    return mlx_weights


def load_safetensors(path: Path) -> Dict[str, mx.array]:
    """Load weights from a safetensors file as MLX arrays.

    Uses MLX's native safetensors loader which handles bfloat16 directly.
    """
    return dict(mx.load(str(path)))  # mx.load handles .safetensors natively


def load_dit_weights(
    model_dir: Path,
    filename: str = "model.safetensors",
) -> Dict[str, mx.array]:
    """Load and convert DiT model weights.

    Args:
        model_dir: Directory containing model.safetensors
        filename: Weight filename

    Returns:
        Dict of MLX-format weights
    """
    path = Path(model_dir) / filename
    if not path.exists():
        raise FileNotFoundError(f"Weight file not found: {path}")

    raw = load_safetensors(path)
    logger.info(f"Loaded {len(raw)} weight tensors from {path}")
    return map_dit_weights(raw)


def load_vae_weights(
    vae_dir: Path,
    filename: str = "diffusion_pytorch_model.safetensors",
) -> Dict[str, mx.array]:
    """Load and convert VAE weights.

    Args:
        vae_dir: Directory containing diffusion_pytorch_model.safetensors
        filename: Weight filename

    Returns:
        Dict of MLX-format weights
    """
    path = Path(vae_dir) / filename
    if not path.exists():
        raise FileNotFoundError(f"Weight file not found: {path}")

    raw = load_safetensors(path)
    logger.info(f"Loaded {len(raw)} weight tensors from {path}")
    return map_vae_weights(raw)


def load_silence_latent(path: Path) -> mx.array:
    """Load silence_latent.pt and convert to MLX format.

    PyTorch format: [1, 64, T] (NCL)
    MLX format: [1, T, 64] (NLC)
    """
    import torch

    data = torch.load(str(path), map_location="cpu", weights_only=True)
    if isinstance(data, torch.Tensor):
        arr = mx.array(data.numpy())
    else:
        arr = mx.array(data)

    # Transpose from NCL to NLC if needed
    if arr.ndim == 3 and arr.shape[1] == 64:
        arr = mx.transpose(arr, axes=(0, 2, 1))

    return arr


__all__ = [
    "map_dit_weights",
    "map_vae_weights",
    "load_dit_weights",
    "load_vae_weights",
    "load_safetensors",
    "load_silence_latent",
]
