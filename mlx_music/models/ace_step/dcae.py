"""
DCAE (Deep Compression AutoEncoder) for ACE-Step.

Encodes mel-spectrograms to latent space and decodes back.
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

    # Encoder
    encoder_block_out_channels: List[int] = field(
        default_factory=lambda: [128, 256, 512, 1024]
    )
    encoder_layers_per_block: List[int] = field(default_factory=lambda: [2, 2, 3, 3])

    # Decoder
    decoder_block_out_channels: List[int] = field(
        default_factory=lambda: [128, 256, 512, 1024]
    )
    decoder_layers_per_block: List[int] = field(default_factory=lambda: [3, 3, 3, 3])

    # Normalization
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


class RMSNorm2d(nn.Module):
    """RMS Normalization for 2D feature maps."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = mx.ones((dim,))

    def __call__(self, x: mx.array) -> mx.array:
        # x: (batch, channels, height, width)
        # Normalize over channel dimension
        rms = mx.sqrt(mx.mean(x**2, axis=1, keepdims=True) + self.eps)
        x = x / rms
        return x * self.weight[None, :, None, None]


class ResBlock2d(nn.Module):
    """Residual block with 2D convolutions."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        use_attention: bool = False,
        attention_head_dim: int = 32,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.norm1 = RMSNorm2d(in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)

        self.norm2 = RMSNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)

        # Skip connection if channels change
        if in_channels != out_channels:
            self.skip = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        else:
            self.skip = None

        self.use_attention = use_attention
        if use_attention:
            self.attn = EfficientViTBlock(out_channels, attention_head_dim)

    def __call__(self, x: mx.array) -> mx.array:
        residual = x

        x = self.norm1(x)
        x = nn.silu(x)
        x = self.conv1(x)

        x = self.norm2(x)
        x = nn.silu(x)
        x = self.conv2(x)

        if self.skip is not None:
            residual = self.skip(residual)

        x = x + residual

        if self.use_attention:
            x = self.attn(x)

        return x


class EfficientViTBlock(nn.Module):
    """Efficient Vision Transformer block with multi-scale attention."""

    def __init__(
        self,
        dim: int,
        head_dim: int = 32,
        qkv_multiscale: int = 5,
    ):
        super().__init__()
        self.dim = dim
        self.head_dim = head_dim
        self.num_heads = dim // head_dim
        self.scale = head_dim**-0.5

        self.norm = RMSNorm2d(dim)

        # QKV projection
        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1)
        self.proj = nn.Conv2d(dim, dim, kernel_size=1)

        # Multi-scale pooling for efficiency
        self.pool_size = qkv_multiscale

    def __call__(self, x: mx.array) -> mx.array:
        residual = x
        batch, channels, height, width = x.shape

        x = self.norm(x)

        # QKV projection
        qkv = self.qkv(x)  # (B, 3*C, H, W)
        qkv = qkv.reshape(batch, 3, self.num_heads, self.head_dim, height, width)
        q, k, v = qkv[:, 0], qkv[:, 1], qkv[:, 2]

        # Reshape for attention: (B, heads, H*W, head_dim)
        q = q.reshape(batch, self.num_heads, self.head_dim, -1)
        q = mx.transpose(q, axes=(0, 1, 3, 2))  # (B, heads, H*W, head_dim)

        k = k.reshape(batch, self.num_heads, self.head_dim, -1)
        k = mx.transpose(k, axes=(0, 1, 3, 2))

        v = v.reshape(batch, self.num_heads, self.head_dim, -1)
        v = mx.transpose(v, axes=(0, 1, 3, 2))

        # Scaled dot-product attention
        attn = mx.matmul(q, mx.transpose(k, axes=(0, 1, 3, 2))) * self.scale
        attn = mx.softmax(attn, axis=-1)
        out = mx.matmul(attn, v)

        # Reshape back: (B, heads, H*W, head_dim) → (B, C, H, W)
        out = mx.transpose(out, axes=(0, 1, 3, 2))  # (B, heads, head_dim, H*W)
        out = out.reshape(batch, channels, height, width)

        out = self.proj(out)

        return out + residual


class Downsample2d(nn.Module):
    """2x downsampling with strided convolution."""

    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, stride=2, padding=1)

    def __call__(self, x: mx.array) -> mx.array:
        return self.conv(x)


class Upsample2d(nn.Module):
    """2x upsampling with interpolation and convolution."""

    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, padding=1)

    def __call__(self, x: mx.array) -> mx.array:
        # Bilinear upsampling 2x
        batch, channels, height, width = x.shape
        x = mx.repeat(x, 2, axis=2)
        x = mx.repeat(x, 2, axis=3)
        return self.conv(x)


class DCAEEncoder(nn.Module):
    """DCAE Encoder: Mel-spectrogram → Latent."""

    def __init__(self, config: DCAEConfig):
        super().__init__()
        self.config = config

        # Initial convolution
        self.conv_in = nn.Conv2d(
            config.in_channels,
            config.encoder_block_out_channels[0],
            kernel_size=3,
            padding=1,
        )

        # Encoder stages
        self.down_blocks = []
        in_ch = config.encoder_block_out_channels[0]

        for i, (out_ch, n_layers) in enumerate(
            zip(config.encoder_block_out_channels, config.encoder_layers_per_block)
        ):
            blocks = []

            # Use attention in later stages (3, 4)
            use_attention = i >= 2

            for j in range(n_layers):
                blocks.append(
                    ResBlock2d(
                        in_ch if j == 0 else out_ch,
                        out_ch,
                        use_attention=use_attention,
                        attention_head_dim=config.attention_head_dim,
                    )
                )

            self.down_blocks.append(blocks)

            # Downsample except for last stage
            if i < len(config.encoder_block_out_channels) - 1:
                self.down_blocks.append([Downsample2d(out_ch)])

            in_ch = out_ch

        # Latent projection
        self.conv_out = nn.Conv2d(
            config.encoder_block_out_channels[-1],
            config.latent_channels,
            kernel_size=3,
            padding=1,
        )

    def __call__(self, x: mx.array) -> mx.array:
        x = self.conv_in(x)

        for block_group in self.down_blocks:
            for block in block_group:
                x = block(x)

        x = self.conv_out(x)
        return x


class DCAEDecoder(nn.Module):
    """DCAE Decoder: Latent → Mel-spectrogram."""

    def __init__(self, config: DCAEConfig):
        super().__init__()
        self.config = config

        # Reverse channel order for decoder
        channels = list(reversed(config.decoder_block_out_channels))
        layers = list(reversed(config.decoder_layers_per_block))

        # Initial convolution from latent
        self.conv_in = nn.Conv2d(
            config.latent_channels,
            channels[0],
            kernel_size=3,
            padding=1,
        )

        # Decoder stages
        self.up_blocks = []
        in_ch = channels[0]

        for i, (out_ch, n_layers) in enumerate(zip(channels, layers)):
            # Upsample at start of each stage (except first)
            if i > 0:
                self.up_blocks.append([Upsample2d(in_ch)])

            blocks = []
            use_attention = i < 2  # Attention in first two stages (reversed)

            for j in range(n_layers):
                blocks.append(
                    ResBlock2d(
                        in_ch if j == 0 else out_ch,
                        out_ch,
                        use_attention=use_attention,
                        attention_head_dim=config.attention_head_dim,
                    )
                )
                in_ch = out_ch

            self.up_blocks.append(blocks)

        # Output convolution
        self.norm_out = RMSNorm2d(channels[-1])
        self.conv_out = nn.Conv2d(
            channels[-1],
            config.in_channels,
            kernel_size=3,
            padding=1,
        )

    def __call__(self, x: mx.array) -> mx.array:
        x = self.conv_in(x)

        for block_group in self.up_blocks:
            for block in block_group:
                x = block(x)

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
            mel: Normalized mel-spectrogram (batch, 2, 128, time)

        Returns:
            Latent (batch, 8, H, W)
        """
        latent = self.encoder(mel)

        # Apply scaling
        latent = (latent - self.config.shift_factor) * self.config.scale_factor

        return latent

    def decode(self, latent: mx.array) -> mx.array:
        """
        Decode latent to mel-spectrogram.

        Args:
            latent: Scaled latent (batch, 8, H, W)

        Returns:
            Mel-spectrogram (batch, 2, 128, time)
        """
        # Unscale latent
        latent = latent / self.config.scale_factor + self.config.shift_factor

        mel = self.decoder(latent)

        return mel

    def normalize_mel(self, mel: mx.array) -> mx.array:
        """Normalize mel-spectrogram to [0, 1] range then to [-1, 1]."""
        # Min-max normalization
        mel = (mel - self.config.min_mel) / (self.config.max_mel - self.config.min_mel)
        # Transform to [-1, 1]
        mel = (mel - 0.5) / 0.5
        return mel

    def denormalize_mel(self, mel: mx.array) -> mx.array:
        """Denormalize mel-spectrogram back to log-mel scale."""
        # Undo [-1, 1] transform
        mel = mel * 0.5 + 0.5
        # Undo min-max normalization
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

        from mlx_music.weights.weight_loader import load_safetensors

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
            model.load_weights(list(weights.items()))

        return model
