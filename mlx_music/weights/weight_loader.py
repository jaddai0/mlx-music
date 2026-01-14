"""
Weight loading utilities for ACE-Step models.

Handles loading SafeTensors weights from HuggingFace or local paths,
converting from PyTorch format to MLX format.
"""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import mlx.core as mx
import numpy as np
from huggingface_hub import hf_hub_download, snapshot_download
from safetensors import safe_open
from tqdm import tqdm


@dataclass
class WeightMapping:
    """Defines how to map weights from PyTorch to MLX format."""

    mlx_key: str
    torch_key: str
    transform: Optional[Callable[[mx.array], mx.array]] = None


def load_safetensors(
    path: Union[str, Path],
    dtype: mx.Dtype = mx.bfloat16,
) -> Dict[str, mx.array]:
    """
    Load weights from a SafeTensors file into MLX arrays.

    Args:
        path: Path to the .safetensors file
        dtype: Target dtype for weights (default: bfloat16)

    Returns:
        Dictionary mapping weight names to MLX arrays
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Weight file not found: {path}")

    weights = {}

    # Try using mlx.core's load function first (handles bfloat16 properly)
    try:
        # MLX can load safetensors directly
        weights = mx.load(str(path))
        # Convert to target dtype if needed
        converted_weights = {}
        for key, arr in weights.items():
            if arr.dtype != dtype and arr.dtype in (mx.float32, mx.float16, mx.bfloat16):
                converted_weights[key] = arr.astype(dtype)
            else:
                converted_weights[key] = arr
        return converted_weights
    except FileNotFoundError:
        raise  # Re-raise file not found errors
    except (RuntimeError, ValueError) as e:
        # MLX load failed (possibly format issue), fall back to safetensors
        import warnings
        warnings.warn(f"MLX native load failed for {path}, falling back to safetensors: {e}")

    # Fallback to safetensors with numpy (doesn't support bfloat16)
    # This will convert bfloat16 to float32 first
    with safe_open(str(path), framework="numpy") as f:
        for key in f.keys():
            try:
                tensor = f.get_tensor(key)
                # Convert numpy to MLX array
                arr = mx.array(tensor)
                # Convert to target dtype if needed
                if arr.dtype != dtype and arr.dtype in (mx.float32, mx.float16, mx.bfloat16):
                    arr = arr.astype(dtype)
                weights[key] = arr
            except TypeError as e:
                if "bfloat16" in str(e):
                    # Skip bfloat16 tensors when using numpy framework
                    print(f"Warning: Skipping bfloat16 tensor {key}")
                else:
                    raise

    return weights


def load_sharded_safetensors(
    directory: Union[str, Path],
    dtype: mx.Dtype = mx.bfloat16,
    show_progress: bool = True,
) -> Dict[str, mx.array]:
    """
    Load weights from multiple sharded SafeTensors files.

    Args:
        directory: Directory containing .safetensors shards
        dtype: Target dtype for weights
        show_progress: Whether to show progress bar

    Returns:
        Combined dictionary of all weights
    """
    directory = Path(directory)

    # Check for index file
    index_path = directory / "model.safetensors.index.json"
    if index_path.exists():
        with open(index_path) as f:
            index = json.load(f)
        weight_map = index.get("weight_map", {})
        shard_files = set(weight_map.values())
    else:
        # Find all .safetensors files
        shard_files = list(directory.glob("*.safetensors"))

    weights = {}
    iterator = tqdm(shard_files, desc="Loading weights") if show_progress else shard_files

    for shard_file in iterator:
        shard_path = directory / shard_file if isinstance(shard_file, str) else shard_file
        shard_weights = load_safetensors(shard_path, dtype=dtype)
        weights.update(shard_weights)

    return weights


def download_model(
    repo_id: str,
    local_dir: Optional[Union[str, Path]] = None,
    revision: Optional[str] = None,
) -> Path:
    """
    Download a model from HuggingFace Hub.

    Args:
        repo_id: HuggingFace repository ID (e.g., "ACE-Step/ACE-Step-v1-3.5B")
        local_dir: Local directory to save the model
        revision: Specific revision/branch to download

    Returns:
        Path to the downloaded model directory
    """
    try:
        # Try offline first
        model_path = Path(snapshot_download(
            repo_id,
            local_dir=str(local_dir) if local_dir else None,
            revision=revision,
            local_files_only=True,
        ))
    except Exception:
        # Fall back to network download
        model_path = Path(snapshot_download(
            repo_id,
            local_dir=str(local_dir) if local_dir else None,
            revision=revision,
        ))

    return model_path


def transpose_conv2d(weight: mx.array) -> mx.array:
    """Transpose Conv2d weights from PyTorch to MLX format.

    PyTorch: (out_channels, in_channels, height, width)
    MLX:     (out_channels, height, width, in_channels)
    """
    return mx.transpose(weight, axes=(0, 2, 3, 1))


def transpose_conv1d(weight: mx.array) -> mx.array:
    """Transpose Conv1d weights from PyTorch to MLX format.

    PyTorch: (out_channels, in_channels, kernel_size)
    MLX:     (out_channels, kernel_size, in_channels)
    """
    return mx.transpose(weight, axes=(0, 2, 1))


def transpose_conv_transpose1d(weight: mx.array) -> mx.array:
    """Transpose ConvTranspose1d weights from PyTorch to MLX format.

    PyTorch: (in_channels, out_channels, kernel_size)
    MLX:     (out_channels, kernel_size, in_channels)
    """
    return mx.transpose(weight, axes=(1, 2, 0))


# ACE-Step specific weight mappings
# Note: Most weights have 1:1 naming, so we use pass-through for simplicity.
# Only Conv2d and Conv1d weights need transposition from PyTorch to MLX format.

ACE_STEP_TRANSFORMER_MAPPINGS: List[WeightMapping] = [
    # Patch embedding - Conv2d weights need transposition
    # proj_in.early_conv_layers.0: Conv2d(8, 2048, kernel_size=(16, 1), stride=(16, 1))
    WeightMapping(
        "proj_in.early_conv_layers.0.weight",
        "proj_in.early_conv_layers.0.weight",
        transform=transpose_conv2d,
    ),
    # proj_in.early_conv_layers.1: GroupNorm (weight/bias are 1D, no transpose needed)
    # proj_in.early_conv_layers.2: Conv2d(2048, 2560, kernel_size=(1, 1))
    WeightMapping(
        "proj_in.early_conv_layers.2.weight",
        "proj_in.early_conv_layers.2.weight",
        transform=transpose_conv2d,
    ),
]


def generate_transformer_block_mappings(num_blocks: int = 24) -> List[WeightMapping]:
    """Generate weight mappings for all transformer blocks.

    ACE-Step transformer block structure:
    - transformer_blocks.{i}.attn: LiteLAAttention (self-attention)
        - to_q, to_k, to_v, to_out.0 (Linear layers)
    - transformer_blocks.{i}.cross_attn: SDPACrossAttention
        - to_q, to_k, to_v, to_out.0 (query from latent, k/v from encoder)
        - add_q_proj, add_k_proj, add_v_proj, to_add_out (for joint attention)
    - transformer_blocks.{i}.ff: GLUMBConv
        - inverted_conv.conv: Conv1d (1x1 pointwise)
        - depth_conv.conv: Conv1d (depthwise grouped)
        - point_conv.conv: Conv1d (1x1 pointwise)
    - transformer_blocks.{i}.scale_shift_table: AdaLN modulation
    """
    mappings = []

    for i in range(num_blocks):
        prefix = f"transformer_blocks.{i}"

        # Feed-forward (GLUMBConv) - Conv1d weights need transposition
        # inverted_conv: Conv1d(in, hidden*2, kernel_size=1)
        mappings.append(
            WeightMapping(
                f"{prefix}.ff.inverted_conv.conv.weight",
                f"{prefix}.ff.inverted_conv.conv.weight",
                transform=transpose_conv1d,
            )
        )
        # depth_conv: Conv1d(hidden*2, hidden*2, kernel_size=3, groups=hidden*2)
        mappings.append(
            WeightMapping(
                f"{prefix}.ff.depth_conv.conv.weight",
                f"{prefix}.ff.depth_conv.conv.weight",
                transform=transpose_conv1d,
            )
        )
        # point_conv: Conv1d(hidden, out, kernel_size=1)
        mappings.append(
            WeightMapping(
                f"{prefix}.ff.point_conv.conv.weight",
                f"{prefix}.ff.point_conv.conv.weight",
                transform=transpose_conv1d,
            )
        )

    return mappings


def convert_torch_to_mlx(
    torch_weights: Dict[str, mx.array],
    mappings: List[WeightMapping],
    strict: bool = False,
) -> Dict[str, mx.array]:
    """
    Convert PyTorch weights to MLX format using mappings.

    Args:
        torch_weights: Dictionary of PyTorch weights (already as mx.arrays)
        mappings: List of weight mappings defining the conversion
        strict: If True, raise error on missing weights

    Returns:
        Dictionary of MLX-formatted weights
    """
    mlx_weights = {}
    missing_keys = []

    for mapping in mappings:
        if mapping.torch_key in torch_weights:
            weight = torch_weights[mapping.torch_key]
            if mapping.transform is not None:
                weight = mapping.transform(weight)
            mlx_weights[mapping.mlx_key] = weight
        else:
            missing_keys.append(mapping.torch_key)

    if strict and missing_keys:
        raise KeyError(f"Missing weights: {missing_keys[:10]}...")

    # Also include any unmapped weights (pass-through)
    mapped_torch_keys = {m.torch_key for m in mappings}
    for key, value in torch_weights.items():
        if key not in mapped_torch_keys:
            mlx_weights[key] = value

    return mlx_weights


def load_weights_with_string_keys(
    module: "nn.Module",
    weights: Dict[str, mx.array],
    strict: bool = False,
) -> None:
    """
    Load weights into a module, handling numeric keys as string keys.

    MLX's tree_unflatten converts numeric path components (e.g., "blocks.0.weight")
    into list indices, but our DCAE model uses dict keys with string indices.
    This function manually sets weights to handle this case.

    Args:
        module: The nn.Module to load weights into
        weights: Dictionary of flat weight paths to arrays
        strict: If True, raise error on missing weights
    """
    import re

    # Get current model parameters as a flat dict
    def flatten_params(d, prefix=""):
        flat = {}
        for k, v in d.items():
            full_key = f"{prefix}.{k}" if prefix else k
            if isinstance(v, dict):
                flat.update(flatten_params(v, full_key))
            elif isinstance(v, mx.array):
                flat[full_key] = v
        return flat

    current_params = flatten_params(module.parameters())

    # For each weight, find matching param and update
    missing = []
    updated = 0

    for key, value in weights.items():
        if key in current_params:
            # Navigate to the parameter and set it
            parts = key.split(".")
            obj = module
            for i, part in enumerate(parts[:-1]):
                if hasattr(obj, part):
                    obj = getattr(obj, part)
                elif isinstance(obj, dict) and part in obj:
                    obj = obj[part]
                else:
                    # Try as dict key
                    if hasattr(obj, "__getitem__"):
                        try:
                            obj = obj[part]
                        except (KeyError, TypeError):
                            break
                    else:
                        break
            else:
                # Set the final attribute
                final_part = parts[-1]
                if hasattr(obj, final_part):
                    setattr(obj, final_part, value)
                    updated += 1
                elif isinstance(obj, dict) and final_part in obj:
                    obj[final_part] = value
                    updated += 1
                else:
                    missing.append(key)
        else:
            missing.append(key)

    if strict and missing:
        raise KeyError(f"Missing {len(missing)} weights: {missing[:10]}...")


def load_ace_step_weights(
    model_path: Union[str, Path],
    component: str = "transformer",
    dtype: mx.Dtype = mx.bfloat16,
) -> Tuple[Dict[str, mx.array], Dict[str, Any]]:
    """
    Load ACE-Step model weights and config.

    Args:
        model_path: Path to model directory or HuggingFace repo ID
        component: Which component to load ("transformer", "dcae", "vocoder", "text_encoder")
        dtype: Target dtype for weights

    Returns:
        Tuple of (weights dict, config dict)
    """
    model_path = Path(model_path)

    # Map component to subdirectory
    component_dirs = {
        "transformer": "ace_step_transformer",
        "dcae": "music_dcae_f8c8",
        "vocoder": "music_vocoder",
        "text_encoder": "umt5-base",
    }

    if component not in component_dirs:
        raise ValueError(f"Unknown component: {component}. Choose from {list(component_dirs.keys())}")

    component_dir = model_path / component_dirs[component]

    if not component_dir.exists():
        raise FileNotFoundError(f"Component directory not found: {component_dir}")

    # Load config
    config_path = component_dir / "config.json"
    if config_path.exists():
        with open(config_path) as f:
            config = json.load(f)
    else:
        config = {}

    # Load weights
    weight_file = component_dir / "diffusion_pytorch_model.safetensors"
    if not weight_file.exists():
        weight_file = component_dir / "model.safetensors"

    if weight_file.exists():
        weights = load_safetensors(weight_file, dtype=dtype)
    else:
        # Try sharded loading
        weights = load_sharded_safetensors(component_dir, dtype=dtype)

    # Apply transformations for transformer component
    if component == "transformer":
        all_mappings = ACE_STEP_TRANSFORMER_MAPPINGS + generate_transformer_block_mappings(
            num_blocks=config.get("num_layers", 24)
        )
        weights = convert_torch_to_mlx(weights, all_mappings, strict=False)

    return weights, config
