"""
LoRA (Low-Rank Adaptation) layers for ACE-Step fine-tuning.

LoRA enables efficient fine-tuning by training low-rank decomposition matrices
instead of full weight matrices. This dramatically reduces memory requirements
and training time while maintaining quality.

For ACE-Step, we target:
- LiteLAAttention: to_q, to_k, to_v, to_out (linear self-attention)
- SDPACrossAttention: to_q, to_k, to_v, to_out, add_* projections

Recommended rank: 64-128 for audio quality
Recommended alpha: Equal to rank (alpha/rank = 1.0 scaling)

Reference:
    "LoRA: Low-Rank Adaptation of Large Language Models" (Hu et al., 2021)
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten, tree_unflatten


class LoRALinear(nn.Module):
    """
    Linear layer with LoRA adaptation.

    Implements: y = Wx + (alpha/rank) * BAx

    Where:
    - W is the frozen original weight
    - B is the low-rank down-projection (in_features -> rank)
    - A is the low-rank up-projection (rank -> out_features)
    - alpha controls the strength of the adaptation

    Args:
        in_features: Input dimension
        out_features: Output dimension
        rank: LoRA rank (lower = more compression, higher = more capacity)
        alpha: LoRA scaling factor (typically equal to rank)
        dropout: Dropout probability for LoRA path (default: 0.0)
        bias: Whether to include bias (default: True)
        original_layer: Optional existing nn.Linear to wrap
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int = 64,
        alpha: float = 64.0,
        dropout: float = 0.0,
        bias: bool = True,
        original_layer: Optional[nn.Linear] = None,
    ):
        super().__init__()

        # Validate parameters
        if rank <= 0:
            raise ValueError(f"rank must be positive, got {rank}")
        if alpha <= 0:
            raise ValueError(f"alpha must be positive, got {alpha}")
        if not 0.0 <= dropout < 1.0:
            raise ValueError(f"dropout must be in [0, 1), got {dropout}")

        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank

        # Original (frozen) weights
        if original_layer is not None:
            self.weight = original_layer.weight
            # nn.Linear always has bias attribute, but it may be None
            self.bias = original_layer.bias if original_layer.bias is not None else None
        else:
            # Initialize fresh weights (for testing)
            self.weight = mx.random.normal(shape=(out_features, in_features)) * 0.02
            self.bias = mx.zeros((out_features,)) if bias else None

        # LoRA matrices (trainable)
        # A: down-projection, initialized with Kaiming uniform
        # B: up-projection, initialized with zeros (so LoRA starts as identity)
        self.lora_A = mx.random.normal(shape=(rank, in_features)) * (1.0 / rank ** 0.5)
        self.lora_B = mx.zeros((out_features, rank))

        # Optional dropout on LoRA path
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else None

    def __call__(self, x: mx.array) -> mx.array:
        """
        Forward pass with LoRA adaptation.

        Args:
            x: Input tensor (..., in_features)

        Returns:
            Output tensor (..., out_features)
        """
        # Original linear transformation (frozen)
        result = mx.matmul(x, self.weight.T)
        if self.bias is not None:
            result = result + self.bias

        # LoRA adaptation path
        lora_x = x
        if self.dropout is not None:
            lora_x = self.dropout(lora_x)

        # Low-rank adaptation: (B @ A) @ x
        lora_out = mx.matmul(lora_x, self.lora_A.T)  # (..., rank)
        lora_out = mx.matmul(lora_out, self.lora_B.T)  # (..., out_features)

        # Scale and add
        result = result + self.scaling * lora_out

        return result

    def merge_weights(self) -> mx.array:
        """
        Merge LoRA weights into the base weight matrix.

        Returns:
            Merged weight matrix (out_features, in_features)
        """
        # W_merged = W + scaling * B @ A
        delta_w = self.scaling * mx.matmul(self.lora_B, self.lora_A)
        return self.weight + delta_w

    def get_lora_params(self) -> Dict[str, mx.array]:
        """Return only the LoRA parameters (for optimizer)."""
        return {
            "lora_A": self.lora_A,
            "lora_B": self.lora_B,
        }


def _get_module_children(module: nn.Module) -> List[Tuple[str, Any]]:
    """
    Get child attributes from an MLX module.

    MLX stores module children differently than PyTorch. MLX provides
    a .children() method that returns a dict of named children.

    Args:
        module: The parent module

    Returns:
        List of (name, child) tuples
    """
    # MLX modules have a children() method that returns named children
    if hasattr(module, 'children') and callable(module.children):
        children_dict = module.children()
        if isinstance(children_dict, dict):
            return list(children_dict.items())

    # Fallback to checking vars() for non-standard modules
    children = []
    try:
        instance_vars = vars(module)
    except TypeError:
        instance_vars = {}

    for name, value in instance_vars.items():
        if name.startswith('_'):
            continue
        children.append((name, value))

    return children


def _replace_linear_with_lora(
    module: nn.Module,
    target_names: List[str],
    rank: int,
    alpha: float,
    dropout: float = 0.0,
    _depth: int = 0,
    _visited: Optional[set] = None,
) -> int:
    """
    Recursively replace Linear layers with LoRALinear.

    Args:
        module: Module to process
        target_names: Names of Linear layers to replace (e.g., ["to_q", "to_k", "to_v"])
        rank: LoRA rank
        alpha: LoRA scaling factor
        dropout: LoRA dropout
        _depth: Internal recursion depth counter (do not set manually)
        _visited: Set of visited module ids to prevent cycles

    Returns:
        Number of layers replaced

    Raises:
        RecursionError: If module tree exceeds max depth (100)
    """
    # Prevent infinite recursion on deeply nested or circular structures
    max_depth = 100
    if _depth > max_depth:
        raise RecursionError(f"Module tree exceeds max depth ({max_depth})")

    # Track visited modules to prevent cycles
    if _visited is None:
        _visited = set()
    if id(module) in _visited:
        return 0
    _visited.add(id(module))

    replaced = 0

    # Get child attributes from the module
    for name, child in _get_module_children(module):
        if child is None:
            continue

        if isinstance(child, list):
            # Handle list attributes (like to_out = [Linear, Dropout])
            for i, item in enumerate(child):
                if isinstance(item, nn.Linear) and not isinstance(item, LoRALinear) and name in target_names:
                    lora_layer = LoRALinear(
                        in_features=item.weight.shape[1],
                        out_features=item.weight.shape[0],
                        rank=rank,
                        alpha=alpha,
                        dropout=dropout,
                        original_layer=item,
                    )
                    child[i] = lora_layer
                    replaced += 1
        elif isinstance(child, nn.Linear) and not isinstance(child, LoRALinear) and name in target_names:
            # Direct Linear layer replacement
            lora_layer = LoRALinear(
                in_features=child.weight.shape[1],
                out_features=child.weight.shape[0],
                rank=rank,
                alpha=alpha,
                dropout=dropout,
                original_layer=child,
            )
            setattr(module, name, lora_layer)
            replaced += 1
        elif isinstance(child, nn.Module) and not isinstance(child, LoRALinear):
            # Recurse into submodules (but not into LoRALinear which is also nn.Module)
            replaced += _replace_linear_with_lora(
                child, target_names, rank, alpha, dropout, _depth + 1, _visited
            )

    return replaced


def apply_lora_to_model(
    model: nn.Module,
    rank: int = 64,
    alpha: float = 64.0,
    dropout: float = 0.0,
    target_modules: Optional[List[str]] = None,
) -> int:
    """
    Apply LoRA to an ACE-Step model.

    Replaces specified Linear layers with LoRALinear layers.
    Default targets are the attention projections in LiteLAAttention
    and SDPACrossAttention.

    Args:
        model: ACE-Step model to modify
        rank: LoRA rank (default: 64)
        alpha: LoRA scaling factor (default: 64.0)
        dropout: LoRA dropout probability (default: 0.0)
        target_modules: List of module names to apply LoRA to.
            Default: ["to_q", "to_k", "to_v", "to_out", "add_q_proj", "add_k_proj", "add_v_proj", "to_add_out"]

    Returns:
        Number of layers converted to LoRA

    Example:
        model = load_ace_step_model()
        num_lora = apply_lora_to_model(model, rank=64, alpha=64.0)
        print(f"Applied LoRA to {num_lora} layers")

        # Freeze base weights, only train LoRA
        lora_params = get_lora_parameters(model)
        optimizer = mx.optimizers.AdamW(learning_rate=1e-4)
        optimizer.init(lora_params)
    """
    if target_modules is None:
        # Default ACE-Step attention targets
        target_modules = [
            "to_q",
            "to_k",
            "to_v",
            "to_out",  # Will match to_out[0] in lists
            "add_q_proj",
            "add_k_proj",
            "add_v_proj",
            "to_add_out",
        ]

    return _replace_linear_with_lora(model, target_modules, rank, alpha, dropout)


def get_lora_parameters(model: nn.Module) -> Dict[str, mx.array]:
    """
    Extract only LoRA parameters from a model.

    Use this to create an optimizer that only trains LoRA weights.

    Args:
        model: Model with LoRA layers applied

    Returns:
        Dictionary of LoRA parameters (lora_A, lora_B for each LoRA layer)

    Example:
        lora_params = get_lora_parameters(model)
        optimizer = mx.optimizers.AdamW(learning_rate=1e-4)
        optimizer.init(lora_params)
    """
    lora_params = {}
    flat = tree_flatten(model.parameters())

    for name, param in flat:
        if "lora_A" in name or "lora_B" in name:
            lora_params[name] = param

    return tree_unflatten(list(lora_params.items()))


def merge_lora_weights(model: nn.Module, _visited: Optional[set] = None) -> None:
    """
    Merge LoRA weights into base weights.

    After training, call this to merge the LoRA adaptations into
    the base weights for efficient inference (no LoRA overhead).

    Args:
        model: Model with trained LoRA layers
        _visited: Internal set for cycle detection

    Note:
        This modifies the model in-place. The LoRA layers remain
        but their adaptations are folded into the base weights.
        The LoRA matrices are reset to zero.
    """
    # Track visited modules to prevent cycles
    if _visited is None:
        _visited = set()
    if id(model) in _visited:
        return
    _visited.add(id(model))

    # Get child attributes from the module
    for name, child in _get_module_children(model):
        if child is None:
            continue

        if isinstance(child, list):
            for item in child:
                if isinstance(item, LoRALinear):
                    # Merge and reset
                    item.weight = item.merge_weights()
                    item.lora_A = mx.zeros_like(item.lora_A)
                    item.lora_B = mx.zeros_like(item.lora_B)
        elif isinstance(child, LoRALinear):
            child.weight = child.merge_weights()
            child.lora_A = mx.zeros_like(child.lora_A)
            child.lora_B = mx.zeros_like(child.lora_B)
        elif isinstance(child, nn.Module):
            merge_lora_weights(child, _visited)


def save_lora_weights(model: nn.Module, path: Path | str) -> None:
    """
    Save only LoRA weights to a file.

    Args:
        model: Model with LoRA layers
        path: Path to save weights (safetensors format)

    Example:
        save_lora_weights(model, "ace_step_lora.safetensors")
    """
    lora_params = {}
    flat = tree_flatten(model.parameters())

    for name, param in flat:
        if "lora_A" in name or "lora_B" in name:
            lora_params[name] = param

    mx.save_safetensors(str(path), lora_params)


def load_lora_weights(
    model: nn.Module,
    path: Path | str,
    allowed_dir: Optional[Path] = None,
) -> int:
    """
    Load LoRA weights from a safetensors file.

    Security: Only .safetensors format is allowed (not pickle-based formats).
    The safetensors format is safe because it only stores tensor data,
    not arbitrary Python objects that could execute code.

    Args:
        model: Model with LoRA layers (must have matching architecture)
        path: Path to saved LoRA weights (must be .safetensors format)
        allowed_dir: Optional base directory to restrict file access.
                    If provided, path must be within this directory.

    Returns:
        Number of parameters loaded

    Raises:
        ValueError: If path is outside allowed_dir or has invalid extension.
        FileNotFoundError: If path doesn't exist.

    Example:
        model = load_ace_step_model()
        apply_lora_to_model(model, rank=64)
        load_lora_weights(model, "ace_step_lora.safetensors")
    """
    path = Path(path).resolve()

    # Security: Validate file extension (only safetensors allowed)
    # safetensors is a safe format that doesn't use pickle deserialization
    if path.suffix.lower() != ".safetensors":
        raise ValueError(
            f"Only .safetensors files are allowed for LoRA weights. Got: {path.suffix}"
        )

    # Security: Validate path is within allowed directory
    if allowed_dir is not None:
        allowed_dir = Path(allowed_dir).resolve()
        try:
            path.relative_to(allowed_dir)
        except ValueError:
            raise ValueError(
                f"LoRA weights path must be within allowed directory: {allowed_dir}. "
                f"Got: {path}"
            )

    if not path.exists():
        raise FileNotFoundError(f"LoRA weights file not found: {path}")

    if not path.is_file():
        raise ValueError(f"LoRA weights path is not a file: {path}")

    # Load using safetensors format (mx.load detects format from extension)
    # safetensors is safe - it only stores tensor data, not executable code
    loaded = mx.load(str(path))

    # Get current parameters
    flat = tree_flatten(model.parameters())
    param_dict = dict(flat)

    # Update matching LoRA parameters only
    loaded_count = 0
    for name, value in loaded.items():
        # Security: Only update parameters that match expected LoRA pattern
        if ("lora_A" in name or "lora_B" in name) and name in param_dict:
            param_dict[name] = value
            loaded_count += 1

    # Apply back to model
    model.update(tree_unflatten(list(param_dict.items())))

    return loaded_count
