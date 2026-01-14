"""
DCAE (Deep Compression AutoEncoder) for ACE-Step.

Encodes mel-spectrograms to latent space and decodes back.
Architecture matches the actual checkpoint structure.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import mlx.core as mx
import mlx.nn as nn


@dataclass
class DCAEConfig:
    """Configuration for DCAE."""

    in_channels: int = 2  # Stereo input
    latent_channels: int = 8
    attention_head_dim: int = 32
    base_channels: int = 128

    # Normalization factors for latent space
    scaling_factor: float = 0.41407
    shift_factor: float = -1.9091
    scale_factor: float = 0.1786

    # Mel normalization
    min_mel: float = -11.0
    max_mel: float = 3.0

    @classmethod
    def from_dict(cls, config: Dict[str, Any]) -> "DCAEConfig":
        """Create config from dictionary."""
        return cls(**{k: v for k, v in config.items() if k in cls.__dataclass_fields__})


class GroupNorm2d(nn.Module):
    """GroupNorm for NHWC format with weight/bias named correctly."""

    def __init__(self, num_groups: int, num_channels: int, eps: float = 1e-6):
        super().__init__()
        self.num_groups = num_groups
        self.num_channels = num_channels
        self.eps = eps
        self.weight = mx.ones((num_channels,))
        self.bias = mx.zeros((num_channels,))

    def __call__(self, x: mx.array) -> mx.array:
        # x: (batch, height, width, channels) - NHWC
        batch, h, w, c = x.shape
        group_size = c // self.num_groups
        x = x.reshape(batch, h, w, self.num_groups, group_size)
        # GroupNorm normalizes over spatial dims (h, w) AND channels within each group
        # For shape (batch, h, w, groups, group_size), normalize over axes (1, 2, 4)
        mean = mx.mean(x, axis=(1, 2, 4), keepdims=True)
        var = mx.var(x, axis=(1, 2, 4), keepdims=True)
        x = (x - mean) / mx.sqrt(var + self.eps)
        x = x.reshape(batch, h, w, c)
        return x * self.weight + self.bias


class ResBlock(nn.Module):
    """
    Residual block matching checkpoint structure.

    Checkpoint keys:
    - norm.weight, norm.bias
    - conv1.weight, conv1.bias
    - conv2.weight (no bias)
    """

    def __init__(self, channels: int):
        super().__init__()
        self.norm = GroupNorm2d(num_groups=32, num_channels=channels)
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)

    def __call__(self, x: mx.array) -> mx.array:
        residual = x

        x = self.norm(x)
        x = nn.silu(x)
        x = self.conv1(x)
        x = nn.silu(x)
        x = self.conv2(x)

        return x + residual


class DownsampleConv(nn.Module):
    """
    Downsample with strided convolution.

    Checkpoint key: conv.weight, conv.bias
    """

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1)

    def __call__(self, x: mx.array) -> mx.array:
        return self.conv(x)


class UpsampleConv(nn.Module):
    """
    Upsample with interpolation + convolution.

    Checkpoint key: conv.weight, conv.bias
    """

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)

    def __call__(self, x: mx.array) -> mx.array:
        # Nearest neighbor 2x upsampling
        x = mx.repeat(x, 2, axis=1)  # height
        x = mx.repeat(x, 2, axis=2)  # width
        return self.conv(x)


class MultiscaleProjection(nn.Module):
    """
    Multi-scale projection for attention using grouped convolution.

    Checkpoint weights:
    - proj_in.weight: (dim*3, 1, 5, 5) - depthwise 5x5 conv expanding dim -> dim*3
    - proj_out.weight: (dim*3, 32, 1, 1) - 1x1 grouped pointwise conv keeping dim*3

    Architecture:
    - proj_in: depthwise conv, dim -> dim*3, kernel=5, groups=dim
    - proj_out: grouped pointwise conv, dim*3 -> dim*3, kernel=1, groups=dim*3/32=96
    - Output is dim*3 channels (Q, K, V concatenated), split and added to each
    """

    def __init__(self, dim: int, pool_size: int = 5):
        super().__init__()
        self.dim = dim
        self.pool_size = pool_size

        # proj_in: depthwise 5x5 conv, dim -> dim*3 (expand for Q, K, V)
        # groups=dim, so each input channel produces 3 output channels
        # Weight shape PyTorch: (dim*3, 1, 5, 5)
        self.proj_in = nn.Conv2d(
            dim,
            dim * 3,
            kernel_size=pool_size,
            padding=pool_size // 2,
            groups=dim,
            bias=False,
        )

        # proj_out: 1x1 grouped pointwise conv, dim*3 -> dim*3
        # groups=96 (for dim=1024, so dim*3/32=96)
        # Weight shape PyTorch: (dim*3, 32, 1, 1)
        num_groups = (dim * 3) // 32  # = 96 for dim=1024
        self.proj_out = nn.Conv2d(
            dim * 3,
            dim * 3,
            kernel_size=1,
            groups=num_groups,
            bias=False,
        )

    def __call__(self, x: mx.array, height: int, width: int) -> Tuple[mx.array, mx.array, mx.array]:
        """
        Apply multiscale projection.

        Args:
            x: Flattened spatial input (batch, seq, dim)
            height: Original spatial height
            width: Original spatial width

        Returns:
            Tuple of (q_ms, k_ms, v_ms) each of shape (batch, seq, dim)
        """
        batch, seq, dim = x.shape

        # Reshape to spatial for convolution: (batch, height, width, dim)
        x = x.reshape(batch, height, width, dim)

        # Apply proj_in -> (batch, height, width, dim*3)
        x = self.proj_in(x)

        # Apply proj_out -> (batch, height, width, dim*3)
        x = self.proj_out(x)

        # Split into Q, K, V components
        q_ms = x[:, :, :, :dim]
        k_ms = x[:, :, :, dim : dim * 2]
        v_ms = x[:, :, :, dim * 2 :]

        # Flatten back to sequence
        q_ms = q_ms.reshape(batch, seq, dim)
        k_ms = k_ms.reshape(batch, seq, dim)
        v_ms = v_ms.reshape(batch, seq, dim)

        return q_ms, k_ms, v_ms


class DCAEAttention(nn.Module):
    """
    Attention module matching checkpoint structure.

    Checkpoint keys:
    - to_q.weight: (dim, dim)
    - to_k.weight: (dim, dim)
    - to_v.weight: (dim, dim)
    - to_out.weight: (dim, dim*2) - takes concatenated attention + v
    - to_qkv_multiscale.0.proj_in/proj_out.weight
    - norm_out.weight, norm_out.bias
    """

    def __init__(self, dim: int, head_dim: int = 32):
        super().__init__()
        self.dim = dim
        self.head_dim = head_dim
        self.num_heads = dim // head_dim
        self.scale = head_dim ** -0.5

        # QKV projections
        self.to_q = nn.Linear(dim, dim, bias=False)
        self.to_k = nn.Linear(dim, dim, bias=False)
        self.to_v = nn.Linear(dim, dim, bias=False)

        # Output projection: takes 2*dim (concatenated attention output + v)
        self.to_out = nn.Linear(dim * 2, dim, bias=False)

        # Multi-scale projection (dict for proper registration)
        self.to_qkv_multiscale = {"0": MultiscaleProjection(dim)}

        # Output norm
        self.norm_out = GroupNorm2d(num_groups=32, num_channels=dim)

    def __call__(self, x: mx.array) -> mx.array:
        # x: (batch, height, width, channels) - NHWC
        batch, height, width, channels = x.shape

        # Flatten spatial dims
        x_flat = x.reshape(batch, height * width, channels)

        # Project Q, K, V
        q = self.to_q(x_flat)
        k = self.to_k(x_flat)
        v = self.to_v(x_flat)

        # Apply multi-scale projection (returns separate q, k, v components)
        q_ms, k_ms, v_ms = self.to_qkv_multiscale["0"](x_flat, height, width)
        q = q + q_ms
        k = k + k_ms
        v = v + v_ms

        # Reshape for multi-head attention
        q = q.reshape(batch, height * width, self.num_heads, self.head_dim)
        k = k.reshape(batch, height * width, self.num_heads, self.head_dim)
        v_attn = v.reshape(batch, height * width, self.num_heads, self.head_dim)

        # Transpose to (batch, heads, seq, head_dim)
        q = mx.transpose(q, axes=(0, 2, 1, 3))
        k = mx.transpose(k, axes=(0, 2, 1, 3))
        v_attn = mx.transpose(v_attn, axes=(0, 2, 1, 3))

        # Attention
        attn = mx.matmul(q, mx.transpose(k, axes=(0, 1, 3, 2))) * self.scale
        attn = mx.softmax(attn, axis=-1)
        attn_out = mx.matmul(attn, v_attn)

        # Reshape back
        attn_out = mx.transpose(attn_out, axes=(0, 2, 1, 3))
        attn_out = attn_out.reshape(batch, height * width, channels)

        # Concatenate attention output with v for gated output
        out = mx.concatenate([attn_out, v], axis=-1)  # (batch, seq, dim*2)

        # Output projection: (dim*2) -> dim
        out = self.to_out(out)

        # Reshape to spatial
        out = out.reshape(batch, height, width, channels)

        # Norm
        out = self.norm_out(out)

        return out


class DCAEConvOut(nn.Module):
    """
    Conv output block for attention blocks using GLU (Gated Linear Unit).

    Checkpoint keys:
    - conv_inverted.weight: (dim*8, dim, 1, 1) - expands to 8x
    - conv_inverted.bias: (dim*8,)
    - conv_depth.weight: (dim*8, 1, 3, 3) - depthwise conv
    - conv_depth.bias: (dim*8,)
    - conv_point.weight: (dim, dim*4, 1, 1) - projects from gated half
    - norm.weight, norm.bias

    Architecture (GLU pattern):
    1. conv_inverted: dim -> dim*8 (expand 8x)
    2. conv_depth: depthwise 3x3 conv
    3. Split into two halves of dim*4 each
    4. GLU gating: x * sigmoid(gate)
    5. conv_point: dim*4 -> dim
    """

    def __init__(self, dim: int):
        super().__init__()
        hidden = dim * 8  # 8x expansion
        gated = dim * 4   # After GLU split

        self.norm = GroupNorm2d(num_groups=32, num_channels=dim)
        self.conv_inverted = nn.Conv2d(dim, hidden, kernel_size=1)
        self.conv_depth = nn.Conv2d(hidden, hidden, kernel_size=3, padding=1, groups=hidden)
        self.conv_point = nn.Conv2d(gated, dim, kernel_size=1, bias=False)

    def __call__(self, x: mx.array) -> mx.array:
        residual = x

        x = self.norm(x)
        x = self.conv_inverted(x)
        x = nn.silu(x)
        x = self.conv_depth(x)

        # GLU gating: split and apply sigmoid gate
        x, gate = mx.split(x, 2, axis=-1)  # Split along channels (NHWC)
        x = x * mx.sigmoid(gate)

        x = self.conv_point(x)

        return x + residual


class AttentionBlock(nn.Module):
    """
    Full attention block combining attention and conv output.

    Checkpoint keys:
    - attn.* (see DCAEAttention)
    - conv_out.* (see DCAEConvOut)
    """

    def __init__(self, dim: int, head_dim: int = 32):
        super().__init__()
        self.attn = DCAEAttention(dim, head_dim)
        self.conv_out = DCAEConvOut(dim)

    def __call__(self, x: mx.array) -> mx.array:
        x = x + self.attn(x)
        x = self.conv_out(x)
        return x


class DCAEEncoder(nn.Module):
    """
    DCAE Encoder matching checkpoint structure.

    Checkpoint structure:
    - conv_in: 2 → 128
    - down_blocks.0: 2 ResBlocks(128) + Downsample(128→256)
    - down_blocks.1: 2 ResBlocks(256) + Downsample(256→512)
    - down_blocks.2: 3 ResBlocks(512) + Downsample(512→1024)
    - down_blocks.3: 3 AttentionBlocks(1024)
    - conv_out: 1024 → 8
    """

    def __init__(self, config: DCAEConfig):
        super().__init__()
        self.config = config

        ch = config.base_channels  # 128

        # Input convolution
        self.conv_in = nn.Conv2d(config.in_channels, ch, kernel_size=3, padding=1)

        # Build down_blocks as nested dicts
        self.down_blocks = {}

        # Block 0: 2 ResBlocks(128) + Downsample(128→256)
        self.down_blocks["0"] = {
            "0": ResBlock(ch),
            "1": ResBlock(ch),
            "2": DownsampleConv(ch, ch * 2),
        }

        # Block 1: 2 ResBlocks(256) + Downsample(256→512)
        self.down_blocks["1"] = {
            "0": ResBlock(ch * 2),
            "1": ResBlock(ch * 2),
            "2": DownsampleConv(ch * 2, ch * 4),
        }

        # Block 2: 3 ResBlocks(512) + Downsample(512→1024)
        self.down_blocks["2"] = {
            "0": ResBlock(ch * 4),
            "1": ResBlock(ch * 4),
            "2": ResBlock(ch * 4),
            "3": DownsampleConv(ch * 4, ch * 8),
        }

        # Block 3: 3 AttentionBlocks(1024)
        self.down_blocks["3"] = {
            "0": AttentionBlock(ch * 8, config.attention_head_dim),
            "1": AttentionBlock(ch * 8, config.attention_head_dim),
            "2": AttentionBlock(ch * 8, config.attention_head_dim),
        }

        # Store order for forward pass
        self._block_order = [
            ("0", ["0", "1", "2"]),
            ("1", ["0", "1", "2"]),
            ("2", ["0", "1", "2", "3"]),
            ("3", ["0", "1", "2"]),
        ]

        # Output convolution
        self.conv_out = nn.Conv2d(ch * 8, config.latent_channels, kernel_size=3, padding=1)

    def __call__(self, x: mx.array) -> mx.array:
        x = self.conv_in(x)

        for block_idx, sub_indices in self._block_order:
            for sub_idx in sub_indices:
                x = self.down_blocks[block_idx][sub_idx](x)
            # Evaluate after each major block to prevent graph explosion
            mx.eval(x)

        x = self.conv_out(x)
        return x


class DCAEDecoder(nn.Module):
    """
    DCAE Decoder matching checkpoint structure.

    Note: Blocks are numbered in REVERSE order of execution!

    Checkpoint structure (in execution order):
    - conv_in: 8 → 1024
    - up_blocks.3: 3 AttentionBlocks(1024)  [FIRST]
    - up_blocks.2: Upsample(1024→512) + 3 ResBlocks(512)
    - up_blocks.1: Upsample(512→256) + 3 ResBlocks(256)
    - up_blocks.0: Upsample(256→128) + 3 ResBlocks(128) [LAST]
    - norm_out: 128
    - conv_out: 128 → 2
    """

    def __init__(self, config: DCAEConfig):
        super().__init__()
        self.config = config

        ch = config.base_channels  # 128

        # Input from latent
        self.conv_in = nn.Conv2d(config.latent_channels, ch * 8, kernel_size=3, padding=1)

        # Build up_blocks as nested dicts
        # Note: Block numbering is reversed from execution order!
        self.up_blocks = {}

        # Block 3: 3 AttentionBlocks(1024) - executed FIRST
        self.up_blocks["3"] = {
            "0": AttentionBlock(ch * 8, config.attention_head_dim),
            "1": AttentionBlock(ch * 8, config.attention_head_dim),
            "2": AttentionBlock(ch * 8, config.attention_head_dim),
        }

        # Block 2: Upsample(1024→512) + 3 ResBlocks(512)
        self.up_blocks["2"] = {
            "0": UpsampleConv(ch * 8, ch * 4),
            "1": ResBlock(ch * 4),
            "2": ResBlock(ch * 4),
            "3": ResBlock(ch * 4),
        }

        # Block 1: Upsample(512→256) + 3 ResBlocks(256)
        self.up_blocks["1"] = {
            "0": UpsampleConv(ch * 4, ch * 2),
            "1": ResBlock(ch * 2),
            "2": ResBlock(ch * 2),
            "3": ResBlock(ch * 2),
        }

        # Block 0: Upsample(256→128) + 3 ResBlocks(128) - executed LAST
        self.up_blocks["0"] = {
            "0": UpsampleConv(ch * 2, ch),
            "1": ResBlock(ch),
            "2": ResBlock(ch),
            "3": ResBlock(ch),
        }

        # Store order for forward pass (3→2→1→0, NOT 0→1→2→3)
        self._block_order = [
            ("3", ["0", "1", "2"]),
            ("2", ["0", "1", "2", "3"]),
            ("1", ["0", "1", "2", "3"]),
            ("0", ["0", "1", "2", "3"]),
        ]

        # Output
        self.norm_out = GroupNorm2d(num_groups=32, num_channels=ch)
        self.conv_out = nn.Conv2d(ch, config.in_channels, kernel_size=3, padding=1)

    def __call__(self, x: mx.array) -> mx.array:
        x = self.conv_in(x)

        # Execute in order: 3, 2, 1, 0
        for block_idx, sub_indices in self._block_order:
            for sub_idx in sub_indices:
                x = self.up_blocks[block_idx][sub_idx](x)
            # Evaluate after each major block to prevent graph explosion
            mx.eval(x)

        x = self.norm_out(x)
        x = nn.silu(x)
        x = self.conv_out(x)

        return x


class DCAE(nn.Module):
    """
    Deep Compression AutoEncoder for audio latents.

    Encodes stereo mel-spectrograms to 8-channel latent space
    and decodes back to mel-spectrograms.
    """

    def __init__(self, config: DCAEConfig):
        super().__init__()
        self.config = config
        self.encoder = DCAEEncoder(config)
        self.decoder = DCAEDecoder(config)

    def encode(self, mel: mx.array) -> mx.array:
        """
        Encode mel-spectrogram to latent.

        Args:
            mel: Normalized mel-spectrogram (batch, 2, 128, time) in NCHW format

        Returns:
            Latent (batch, 8, H, W) in NCHW format
        """
        if mel.ndim != 4:
            raise ValueError(
                f"Expected 4D input (batch, channels, height, width), got {mel.ndim}D"
            )
        if mel.shape[1] != 2:
            raise ValueError(
                f"Expected 2 channels (stereo mel), got {mel.shape[1]} channels"
            )

        # Convert NCHW to NHWC for MLX convolutions
        mel = mx.transpose(mel, axes=(0, 2, 3, 1))  # (B, H, W, C)

        latent = self.encoder(mel)

        # Convert back to NCHW
        latent = mx.transpose(latent, axes=(0, 3, 1, 2))  # (B, C, H, W)

        # Apply scaling
        latent = (latent - self.config.shift_factor) * self.config.scale_factor

        return latent

    def decode(self, latent: mx.array) -> mx.array:
        """
        Decode latent to mel-spectrogram.

        Args:
            latent: Scaled latent (batch, 8, H, W) in NCHW format

        Returns:
            Mel-spectrogram (batch, 2, 128, time) in NCHW format
        """
        if latent.ndim != 4:
            raise ValueError(
                f"Expected 4D input (batch, channels, height, width), got {latent.ndim}D"
            )
        if latent.shape[1] != self.config.latent_channels:
            raise ValueError(
                f"Expected {self.config.latent_channels} latent channels, "
                f"got {latent.shape[1]} channels"
            )

        # Unscale latent
        latent = latent / self.config.scale_factor + self.config.shift_factor

        # Convert NCHW to NHWC for MLX convolutions
        latent = mx.transpose(latent, axes=(0, 2, 3, 1))  # (B, H, W, C)

        mel = self.decoder(latent)

        # Convert back to NCHW
        mel = mx.transpose(mel, axes=(0, 3, 1, 2))  # (B, C, H, W)

        return mel

    def normalize_mel(self, mel: mx.array) -> mx.array:
        """Normalize mel-spectrogram to [0, 1] range then to [-1, 1]."""
        mel = (mel - self.config.min_mel) / (self.config.max_mel - self.config.min_mel)
        mel = (mel - 0.5) / 0.5
        return mel

    def denormalize_mel(self, mel: mx.array) -> mx.array:
        """Denormalize mel-spectrogram back to log-mel scale."""
        mel = mel * 0.5 + 0.5
        mel = mel * (self.config.max_mel - self.config.min_mel) + self.config.min_mel
        return mel

    @classmethod
    def from_pretrained(
        cls,
        model_path: str,
        dtype: mx.Dtype = mx.bfloat16,
    ) -> "DCAE":
        """Load DCAE from pretrained weights."""
        import json
        from pathlib import Path

        from mlx_music.weights.weight_loader import (
            load_safetensors,
            load_weights_with_string_keys,
            transpose_conv2d,
        )

        model_path = Path(model_path)

        # Load config
        config_path = model_path / "config.json"
        if config_path.exists():
            with open(config_path) as f:
                config_dict = json.load(f)
            config = DCAEConfig.from_dict(config_dict)
        else:
            config = DCAEConfig()

        # Create model
        model = cls(config)

        # Load weights
        weight_file = model_path / "diffusion_pytorch_model.safetensors"
        if weight_file.exists():
            weights = load_safetensors(weight_file, dtype=dtype)

            # Transpose Conv2d weights from PyTorch to MLX format
            transposed = {}
            for key, value in weights.items():
                # All 4D weights are Conv2d in PyTorch: (out, in, H, W)
                # MLX expects: (out, H, W, in)
                # This includes conv, proj_in, proj_out weights
                if "weight" in key and value.ndim == 4:
                    transposed[key] = transpose_conv2d(value)
                else:
                    transposed[key] = value

            # Use custom loader for dict-keyed blocks
            load_weights_with_string_keys(model, transposed, strict=False)

        return model
