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
    with safe_open(str(path), framework="numpy") as f:
        for key in f.keys():
            tensor = f.get_tensor(key)
            # Convert numpy to MLX array
            arr = mx.array(tensor)
            # Convert to target dtype if needed
            if arr.dtype != dtype and arr.dtype in (mx.float32, mx.float16, mx.bfloat16):
                arr = arr.astype(dtype)
            weights[key] = arr

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


# ACE-Step specific weight mappings
ACE_STEP_TRANSFORMER_MAPPINGS: List[WeightMapping] = [
    # Time embedding
    WeightMapping("time_proj.weight", "time_proj.weight"),
    WeightMapping("timestep_embedder.linear_1.weight", "timestep_embedder.linear_1.weight"),
    WeightMapping("timestep_embedder.linear_1.bias", "timestep_embedder.linear_1.bias"),
    WeightMapping("timestep_embedder.linear_2.weight", "timestep_embedder.linear_2.weight"),
    WeightMapping("timestep_embedder.linear_2.bias", "timestep_embedder.linear_2.bias"),

    # t_block (AdaLN conditioning)
    WeightMapping("t_block.1.weight", "t_block.1.weight"),
    WeightMapping("t_block.1.bias", "t_block.1.bias"),

    # Embedders
    WeightMapping("speaker_embedder.weight", "speaker_embedder.weight"),
    WeightMapping("genre_embedder.weight", "genre_embedder.weight"),
    WeightMapping("lyric_embs.weight", "lyric_embs.weight"),

    # Patch embedding
    WeightMapping("proj_in.conv1.weight", "proj_in.conv1.weight", transform=transpose_conv2d),
    WeightMapping("proj_in.conv1.bias", "proj_in.conv1.bias"),
    WeightMapping("proj_in.conv2.weight", "proj_in.conv2.weight", transform=transpose_conv2d),
    WeightMapping("proj_in.conv2.bias", "proj_in.conv2.bias"),

    # Final layer
    WeightMapping("final_layer.norm.weight", "final_layer.norm.weight"),
    WeightMapping("final_layer.linear.weight", "final_layer.linear.weight"),
    WeightMapping("final_layer.linear.bias", "final_layer.linear.bias"),
    WeightMapping("final_layer.adaLN_modulation.1.weight", "final_layer.adaLN_modulation.1.weight"),
    WeightMapping("final_layer.adaLN_modulation.1.bias", "final_layer.adaLN_modulation.1.bias"),

    # RoPE
    WeightMapping("rotary_emb.inv_freq", "rotary_emb.inv_freq"),
]


def generate_transformer_block_mappings(num_blocks: int = 24) -> List[WeightMapping]:
    """Generate weight mappings for all transformer blocks."""
    mappings = []

    for i in range(num_blocks):
        prefix = f"transformer_blocks.{i}"

        # Layer norms
        mappings.extend([
            WeightMapping(f"{prefix}.norm1.linear.weight", f"{prefix}.norm1.linear.weight"),
            WeightMapping(f"{prefix}.norm1.linear.bias", f"{prefix}.norm1.linear.bias"),
        ])

        # Self-attention
        mappings.extend([
            WeightMapping(f"{prefix}.attn1.to_q.weight", f"{prefix}.attn1.to_q.weight"),
            WeightMapping(f"{prefix}.attn1.to_k.weight", f"{prefix}.attn1.to_k.weight"),
            WeightMapping(f"{prefix}.attn1.to_v.weight", f"{prefix}.attn1.to_v.weight"),
            WeightMapping(f"{prefix}.attn1.to_out.0.weight", f"{prefix}.attn1.to_out.0.weight"),
            WeightMapping(f"{prefix}.attn1.to_out.0.bias", f"{prefix}.attn1.to_out.0.bias"),
            WeightMapping(f"{prefix}.attn1.norm_q.weight", f"{prefix}.attn1.norm_q.weight"),
            WeightMapping(f"{prefix}.attn1.norm_k.weight", f"{prefix}.attn1.norm_k.weight"),
        ])

        # Cross-attention (if present)
        mappings.extend([
            WeightMapping(f"{prefix}.attn2.to_q.weight", f"{prefix}.attn2.to_q.weight"),
            WeightMapping(f"{prefix}.attn2.to_k.weight", f"{prefix}.attn2.to_k.weight"),
            WeightMapping(f"{prefix}.attn2.to_v.weight", f"{prefix}.attn2.to_v.weight"),
            WeightMapping(f"{prefix}.attn2.to_out.0.weight", f"{prefix}.attn2.to_out.0.weight"),
            WeightMapping(f"{prefix}.attn2.to_out.0.bias", f"{prefix}.attn2.to_out.0.bias"),
        ])

        # Feed-forward (GLUMBConv)
        mappings.extend([
            WeightMapping(f"{prefix}.ff.net.0.weight", f"{prefix}.ff.net.0.weight"),
            WeightMapping(f"{prefix}.ff.net.0.bias", f"{prefix}.ff.net.0.bias"),
            WeightMapping(f"{prefix}.ff.net.2.weight", f"{prefix}.ff.net.2.weight"),
            WeightMapping(f"{prefix}.ff.net.2.bias", f"{prefix}.ff.net.2.bias"),
        ])

        # Scale-shift table for AdaLN
        mappings.append(
            WeightMapping(f"{prefix}.scale_shift_table", f"{prefix}.scale_shift_table")
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
