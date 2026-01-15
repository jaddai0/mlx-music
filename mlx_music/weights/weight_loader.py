"""
Weight loading utilities for ACE-Step models.

Handles loading SafeTensors weights from HuggingFace or local paths,
converting from PyTorch format to MLX format.
"""

import json
import os
import re
import unicodedata
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from urllib.parse import unquote

import mlx.core as mx
import numpy as np
from huggingface_hub import snapshot_download
from huggingface_hub.utils import LocalEntryNotFoundError, EntryNotFoundError
from safetensors import safe_open
from tqdm import tqdm


class PathTraversalError(ValueError):
    """Raised when a path traversal attempt is detected."""
    pass


# Maximum allowed path length to prevent buffer overflow attacks
_MAX_PATH_LENGTH = 4096

# Maximum JSON file size to prevent memory exhaustion (10 MB)
_MAX_JSON_SIZE = 10 * 1024 * 1024

# Maximum number of shard files to prevent resource exhaustion
_MAX_SHARD_COUNT = 1000

# Maximum number of weight tensors to prevent memory exhaustion
_MAX_WEIGHT_COUNT = 100_000

# Maximum number of transformer blocks to prevent excessive memory allocation
_MAX_NUM_LAYERS = 1000

# Pattern to detect dangerous control characters (ASCII 0-31 except tab)
_CONTROL_CHAR_PATTERN = re.compile(r'[\x00-\x08\x0b\x0c\x0e-\x1f]')


def _validate_safe_path(base_dir: Path, filename: str) -> Path:
    """
    Validate that a filename doesn't escape the base directory.

    Performs comprehensive security checks including:
    - URL decoding to catch %2e%2e (encoded ..)
    - Unicode normalization to prevent homoglyph attacks
    - Null byte injection detection
    - Control character rejection
    - Symlink resolution
    - Length limits

    Args:
        base_dir: The trusted base directory
        filename: The filename to validate (potentially untrusted)

    Returns:
        Safe resolved path within base_dir

    Raises:
        PathTraversalError: If the path would escape base_dir
        ValueError: If the filename is invalid
    """
    # Check for empty filename
    if not filename:
        raise ValueError("Filename cannot be empty")

    # Check for path length limits
    if len(filename) > _MAX_PATH_LENGTH:
        raise PathTraversalError(
            f"Filename too long ({len(filename)} > {_MAX_PATH_LENGTH})"
        )

    # URL decode to catch encoded path traversal (%2e%2e = .., %2f = /)
    # Do this twice to catch double-encoding
    decoded = unquote(unquote(filename))

    # Unicode normalize to NFC form to prevent homoglyph attacks
    # (e.g., fullwidth dot \uFF0E looking like normal dot)
    decoded = unicodedata.normalize("NFC", decoded)

    # Check for null bytes (can truncate strings in some contexts)
    if "\x00" in decoded or "\x00" in filename:
        raise PathTraversalError(
            f"Invalid filename: null byte detected"
        )

    # Check for control characters (except newline/tab which are handled by path)
    if _CONTROL_CHAR_PATTERN.search(decoded) or _CONTROL_CHAR_PATTERN.search(filename):
        raise PathTraversalError(
            f"Invalid filename: control characters detected"
        )

    # Check for path traversal in decoded version
    if ".." in decoded or decoded.startswith("/") or decoded.startswith("\\"):
        raise PathTraversalError(
            f"Invalid filename '{filename}': path traversal characters detected"
        )

    # Check for Windows drive letters (also check decoded)
    if (len(decoded) > 1 and decoded[1] == ":") or (len(filename) > 1 and filename[1] == ":"):
        raise PathTraversalError(
            f"Invalid filename '{filename}': absolute Windows path not allowed"
        )

    # Normalize the base directory
    base_dir = base_dir.resolve()

    # Construct and resolve the full path (this follows symlinks)
    full_path = (base_dir / filename).resolve()

    # Verify the resolved path is within base_dir
    # This catches symlinks pointing outside the directory
    try:
        full_path.relative_to(base_dir)
    except ValueError:
        raise PathTraversalError(
            f"Invalid filename '{filename}': resolved path escapes base directory"
        )

    return full_path


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
                    warnings.warn(
                        f"Skipping bfloat16 tensor '{key}' - numpy fallback doesn't support bfloat16. "
                        f"Some model weights may be missing."
                    )
                else:
                    raise

    return weights


def load_single_safetensors(
    path: Union[str, Path],
    dtype: mx.Dtype = mx.bfloat16,
) -> Dict[str, mx.array]:
    """
    Load weights from a single SafeTensors file.

    Args:
        path: Path to the .safetensors file
        dtype: Target dtype for weights

    Returns:
        Dictionary mapping weight names to MLX arrays
    """
    return load_safetensors(path, dtype=dtype)


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

    Raises:
        PathTraversalError: If index file contains invalid paths
        json.JSONDecodeError: If index file is malformed
    """
    directory = Path(directory).resolve()

    # Check for index file
    index_path = directory / "model.safetensors.index.json"
    if index_path.exists():
        # Check file size before loading to prevent memory exhaustion
        index_size = index_path.stat().st_size
        if index_size > _MAX_JSON_SIZE:
            raise ValueError(
                f"Index file too large: {index_size} bytes (max: {_MAX_JSON_SIZE})"
            )

        with open(index_path, encoding="utf-8") as f:
            index = json.load(f)

        # Validate index structure
        if not isinstance(index, dict):
            raise ValueError("Invalid index file: expected a JSON object")

        weight_map = index.get("weight_map", {})
        if not isinstance(weight_map, dict):
            raise ValueError("Invalid index file: weight_map must be a dict")

        # Get unique shard files and validate each one
        raw_shard_files = set(weight_map.values())

        # Check shard count limit
        if len(raw_shard_files) > _MAX_SHARD_COUNT:
            raise ValueError(
                f"Too many shard files: {len(raw_shard_files)} (max: {_MAX_SHARD_COUNT})"
            )

        shard_files = []
        for shard_filename in raw_shard_files:
            if not isinstance(shard_filename, str):
                raise ValueError(
                    f"Invalid index file: shard filename must be string, got {type(shard_filename)}"
                )
            # Validate path is safe (no traversal)
            safe_path = _validate_safe_path(directory, shard_filename)
            shard_files.append(safe_path)
    else:
        # Find all .safetensors files directly in the directory
        # (glob is safe - only returns files within the directory)
        shard_files = list(directory.glob("*.safetensors"))

        # Check shard count limit
        if len(shard_files) > _MAX_SHARD_COUNT:
            raise ValueError(
                f"Too many shard files: {len(shard_files)} (max: {_MAX_SHARD_COUNT})"
            )

    weights = {}
    iterator = tqdm(shard_files, desc="Loading weights") if show_progress else shard_files

    for shard_path in iterator:
        # shard_path is already a validated Path object
        shard_weights = load_safetensors(shard_path, dtype=dtype)
        weights.update(shard_weights)

        # Check weight count limit to prevent memory exhaustion
        if len(weights) > _MAX_WEIGHT_COUNT:
            raise ValueError(
                f"Too many weight tensors: {len(weights)} (max: {_MAX_WEIGHT_COUNT})"
            )

    return weights


def load_pytorch_bin(
    path: Union[str, Path],
    dtype: mx.Dtype = mx.bfloat16,
) -> Dict[str, mx.array]:
    """
    Load weights from a PyTorch .bin file into MLX arrays.

    Args:
        path: Path to the .bin file
        dtype: Target dtype for weights (default: bfloat16)

    Returns:
        Dictionary mapping weight names to MLX arrays

    Note:
        Requires PyTorch to be installed. Uses weights_only=True for security.
    """
    try:
        import torch
    except ImportError:
        raise ImportError(
            "PyTorch is required to load .bin files. "
            "Install with: pip install torch"
        )

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Weight file not found: {path}")

    # Load with weights_only=True for security (prevents arbitrary code execution)
    # map_location="cpu" prevents GPU memory usage during loading
    torch_weights = torch.load(str(path), map_location="cpu", weights_only=True)

    # Convert to MLX arrays
    weights = {}
    for key, tensor in torch_weights.items():
        if isinstance(tensor, torch.Tensor):
            # Convert bfloat16 to float32 first (numpy doesn't support bf16)
            if tensor.dtype == torch.bfloat16:
                tensor = tensor.float()
            # Convert to numpy then MLX
            np_array = tensor.numpy()
            arr = mx.array(np_array)
            # Convert to target dtype
            if arr.dtype != dtype and arr.dtype in (mx.float32, mx.float16, mx.bfloat16):
                arr = arr.astype(dtype)
            weights[key] = arr

    return weights


def load_sharded_pytorch(
    directory: Union[str, Path],
    dtype: mx.Dtype = mx.bfloat16,
    show_progress: bool = True,
) -> Dict[str, mx.array]:
    """
    Load weights from multiple sharded PyTorch .bin files.

    Args:
        directory: Directory containing .bin shards and pytorch_model.bin.index.json
        dtype: Target dtype for weights
        show_progress: Whether to show progress bar

    Returns:
        Combined dictionary of all weights

    Raises:
        FileNotFoundError: If index file or shards not found
        PathTraversalError: If index file contains invalid paths
        json.JSONDecodeError: If index file is malformed
    """
    directory = Path(directory).resolve()

    # Check for index file
    index_path = directory / "pytorch_model.bin.index.json"
    if not index_path.exists():
        raise FileNotFoundError(
            f"PyTorch index file not found: {index_path}. "
            f"Expected pytorch_model.bin.index.json for sharded weights."
        )

    # Check file size before loading
    index_size = index_path.stat().st_size
    if index_size > _MAX_JSON_SIZE:
        raise ValueError(
            f"Index file too large: {index_size} bytes (max: {_MAX_JSON_SIZE})"
        )

    with open(index_path, encoding="utf-8") as f:
        index = json.load(f)

    # Validate index structure
    if not isinstance(index, dict):
        raise ValueError("Invalid index file: expected a JSON object")

    weight_map = index.get("weight_map", {})
    if not isinstance(weight_map, dict):
        raise ValueError("Invalid index file: weight_map must be a dict")

    # Get unique shard files and validate each one
    raw_shard_files = set(weight_map.values())

    # Check shard count limit
    if len(raw_shard_files) > _MAX_SHARD_COUNT:
        raise ValueError(
            f"Too many shard files: {len(raw_shard_files)} (max: {_MAX_SHARD_COUNT})"
        )

    shard_files = []
    for shard_filename in raw_shard_files:
        if not isinstance(shard_filename, str):
            raise ValueError(
                f"Invalid index file: shard filename must be string, got {type(shard_filename)}"
            )
        # Validate path is safe (no traversal)
        safe_path = _validate_safe_path(directory, shard_filename)
        shard_files.append(safe_path)

    # Load each shard
    weights = {}
    iterator = tqdm(shard_files, desc="Loading PyTorch weights") if show_progress else shard_files

    for shard_path in iterator:
        shard_weights = load_pytorch_bin(shard_path, dtype=dtype)
        weights.update(shard_weights)

        # Check weight count limit
        if len(weights) > _MAX_WEIGHT_COUNT:
            raise ValueError(
                f"Too many weight tensors: {len(weights)} (max: {_MAX_WEIGHT_COUNT})"
            )

    return weights


def is_local_path(path_or_repo: str) -> bool:
    """Check if the input is a local path rather than a HuggingFace repo ID."""
    # A local path starts with /, ~, or .
    # A repo ID is in the form "org/repo" or "repo"
    if path_or_repo.startswith(("/", "~", ".")) or Path(path_or_repo).exists():
        return True
    # Check for Windows-style paths
    if len(path_or_repo) > 1 and path_or_repo[1] == ":":
        return True
    return False


def download_model(
    model_id: str,
    local_dir: Optional[Union[str, Path]] = None,
    revision: Optional[str] = None,
) -> Path:
    """
    Get or download a model from HuggingFace Hub.

    Supports both local paths and HuggingFace repository IDs.

    Args:
        model_id: Local path or HuggingFace repository ID (e.g., "ACE-Step/ACE-Step-v1-3.5B")
        local_dir: Local directory to save the model (only used for HF downloads)
        revision: Specific revision/branch to download (only used for HF downloads)

    Returns:
        Path to the model directory

    Raises:
        FileNotFoundError: If local path doesn't exist or HF model not found
        ValueError: If model_id is invalid

    Examples:
        >>> download_model("/path/to/local/model")  # Local path
        >>> download_model("ACE-Step/ACE-Step-v1-3.5B")  # HuggingFace repo
        >>> download_model("ACE-Step/ACE-Step-v1-3.5B", revision="v1.0")  # Specific version
    """
    # Validate model_id
    if not model_id or not isinstance(model_id, str):
        raise ValueError(f"model_id must be a non-empty string, got {type(model_id)}")

    model_id = model_id.strip()
    if not model_id:
        raise ValueError("model_id cannot be empty or whitespace")

    # Handle local paths directly
    if is_local_path(model_id):
        model_path = Path(model_id).expanduser().resolve()
        if not model_path.exists():
            raise FileNotFoundError(f"Local model path not found: {model_path}")
        return model_path

    # Download from HuggingFace Hub
    print(f"Downloading model from HuggingFace Hub: {model_id}...")

    try:
        # Try offline first (cached models)
        model_path = Path(snapshot_download(
            model_id,
            local_dir=str(local_dir) if local_dir else None,
            revision=revision,
            local_files_only=True,
        ))
        print("Using cached model.")
    except (LocalEntryNotFoundError, EntryNotFoundError, OSError):
        # Model not cached locally, fall back to network download
        # OSError can occur for network issues or missing cache
        model_path = Path(snapshot_download(
            model_id,
            local_dir=str(local_dir) if local_dir else None,
            revision=revision,
        ))
        print("Download complete.")

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

    # Load config with size limit and validation
    config_path = component_dir / "config.json"
    if config_path.exists():
        # Check config file size
        config_size = config_path.stat().st_size
        if config_size > _MAX_JSON_SIZE:
            raise ValueError(
                f"Config file too large: {config_size} bytes (max: {_MAX_JSON_SIZE})"
            )

        with open(config_path, encoding="utf-8") as f:
            config = json.load(f)

        # Validate config structure
        if not isinstance(config, dict):
            raise ValueError("Invalid config file: expected a JSON object")

        # Validate num_layers if present (used for generating mappings)
        if "num_layers" in config:
            num_layers = config["num_layers"]
            if not isinstance(num_layers, int) or num_layers < 0 or num_layers > _MAX_NUM_LAYERS:
                raise ValueError(
                    f"Invalid num_layers in config: {num_layers} "
                    f"(must be integer in range 0-{_MAX_NUM_LAYERS})"
                )
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

    # Validate that weights were loaded
    if not weights:
        raise ValueError(
            f"No weights loaded from {component_dir}. "
            f"Expected safetensors files but found none or all were empty."
        )

    # Apply transformations for transformer component
    if component == "transformer":
        all_mappings = ACE_STEP_TRANSFORMER_MAPPINGS + generate_transformer_block_mappings(
            num_blocks=config.get("num_layers", 24)
        )
        weights = convert_torch_to_mlx(weights, all_mappings, strict=False)

    return weights, config
