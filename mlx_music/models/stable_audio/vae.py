"""
AutoencoderOobleck (VAE) for Stable Audio Open.

A 1D convolutional variational autoencoder designed for audio,
using Snake activation functions for improved audio quality.

The VAE compresses audio by a factor of 2048 (from 44.1kHz to ~21.5Hz latent rate).
"""

import math
from typing import List, Optional, Tuple

import mlx.core as mx
import mlx.nn as nn

from mlx_music.models.stable_audio.config import VAEConfig


def snake(x: mx.array, alpha: mx.array) -> mx.array:
    """
    Snake activation function.

    Snake(x) = x + (1/alpha) * sin^2(alpha * x)

    This activation is particularly well-suited for audio generation
    as it preserves periodic structure.

    Args:
        x: Input tensor
        alpha: Learnable frequency parameter

    Returns:
        Activated tensor
    """
    # Use absolute value + epsilon to prevent division by zero
    # This handles both near-zero and negative alpha values safely
    safe_alpha = mx.maximum(mx.abs(alpha), 1e-9)
    return x + (1.0 / safe_alpha) * mx.sin(alpha * x) ** 2


class Snake1d(nn.Module):
    """Snake activation with learnable alpha for 1D convolutions."""

    def __init__(self, channels: int):
        """
        Initialize Snake activation.

        Args:
            channels: Number of channels (alpha is per-channel)
        """
        super().__init__()
        # Initialize alpha to 1.0
        self.alpha = mx.ones((channels,))

    def __call__(self, x: mx.array) -> mx.array:
        """
        Apply Snake activation.

        Args:
            x: Input of shape (batch, length, channels) - MLX format

        Returns:
            Activated tensor of same shape
        """
        # Expand alpha for broadcasting
        alpha = self.alpha[None, None, :]  # (1, 1, channels)
        return snake(x, alpha)


class Conv1d(nn.Module):
    """
    1D Convolution layer for audio processing.

    MLX uses (batch, length, channels) format, so we adapt accordingly.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
        groups: int = 1,
        bias: bool = True,
    ):
        super().__init__()

        # Validate parameters
        if kernel_size <= 0:
            raise ValueError(f"kernel_size must be positive, got {kernel_size}")
        if stride <= 0:
            raise ValueError(f"stride must be positive, got {stride}")
        if in_channels % groups != 0:
            raise ValueError(
                f"in_channels ({in_channels}) must be divisible by groups ({groups})"
            )

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups

        # Weight shape for MLX conv: (out_channels, kernel_size, in_channels // groups)
        # Standard PyTorch shape: (out_channels, in_channels // groups, kernel_size)
        # We store in MLX format directly
        scale = math.sqrt(1.0 / (in_channels * kernel_size))
        self.weight = mx.random.uniform(
            low=-scale,
            high=scale,
            shape=(out_channels, kernel_size, in_channels // groups),
        )
        if bias:
            self.bias = mx.zeros((out_channels,))
        else:
            self.bias = None

    def __call__(self, x: mx.array) -> mx.array:
        """
        Apply 1D convolution.

        Args:
            x: Input of shape (batch, length, in_channels)

        Returns:
            Output of shape (batch, new_length, out_channels)
        """
        # Apply padding manually if needed
        if self.padding > 0:
            x = mx.pad(x, [(0, 0), (self.padding, self.padding), (0, 0)])

        # MLX conv1d: input (N, L, C_in), weight (C_out, K, C_in)
        y = mx.conv1d(x, self.weight, stride=self.stride, groups=self.groups)

        if self.bias is not None:
            y = y + self.bias[None, None, :]

        return y


class ConvTranspose1d(nn.Module):
    """1D Transposed Convolution for upsampling."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        output_padding: int = 0,
        groups: int = 1,
        bias: bool = True,
    ):
        super().__init__()

        # Validate parameters
        if kernel_size <= 0:
            raise ValueError(f"kernel_size must be positive, got {kernel_size}")
        if stride <= 0:
            raise ValueError(f"stride must be positive, got {stride}")
        if in_channels % groups != 0:
            raise ValueError(
                f"in_channels ({in_channels}) must be divisible by groups ({groups})"
            )

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding
        self.groups = groups

        # Weight shape for MLX conv_transpose
        scale = math.sqrt(1.0 / (in_channels * kernel_size))
        self.weight = mx.random.uniform(
            low=-scale,
            high=scale,
            shape=(out_channels, kernel_size, in_channels // groups),
        )
        if bias:
            self.bias = mx.zeros((out_channels,))
        else:
            self.bias = None

    def __call__(self, x: mx.array) -> mx.array:
        """
        Apply transposed 1D convolution.

        Args:
            x: Input of shape (batch, length, in_channels)

        Returns:
            Output of shape (batch, new_length, out_channels)
        """
        # Use conv_general for transpose convolution
        y = mx.conv_general(
            x,
            self.weight,
            stride=self.stride,
            padding=self.padding,
            groups=self.groups,
            flip=True,  # Flip kernel for transpose
        )

        # Handle output padding
        if self.output_padding > 0:
            y = mx.pad(y, [(0, 0), (0, self.output_padding), (0, 0)])

        if self.bias is not None:
            y = y + self.bias[None, None, :]

        return y


class ResidualUnit(nn.Module):
    """Residual unit with Snake activation for Oobleck VAE."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        dilation: int = 1,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        # Dilated convolution
        self.snake1 = Snake1d(in_channels)
        self.conv1 = Conv1d(
            in_channels,
            out_channels,
            kernel_size=7,
            padding=3 * dilation,
            dilation=dilation,
        )

        self.snake2 = Snake1d(out_channels)
        self.conv2 = Conv1d(
            out_channels,
            out_channels,
            kernel_size=1,
        )

        # Skip connection if channels change
        if in_channels != out_channels:
            self.skip = Conv1d(in_channels, out_channels, kernel_size=1)
        else:
            self.skip = None

    def __call__(self, x: mx.array) -> mx.array:
        """Apply residual unit."""
        residual = x

        # Main path
        y = self.snake1(x)
        y = self.conv1(y)
        y = self.snake2(y)
        y = self.conv2(y)

        # Skip connection
        if self.skip is not None:
            residual = self.skip(residual)

        return y + residual


class EncoderBlock(nn.Module):
    """Encoder block with downsampling."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int,
    ):
        super().__init__()

        # Residual units
        self.res_units = [
            ResidualUnit(in_channels, in_channels, dilation=1),
            ResidualUnit(in_channels, in_channels, dilation=3),
            ResidualUnit(in_channels, in_channels, dilation=9),
        ]

        # Downsampling
        self.snake = Snake1d(in_channels)
        self.conv = Conv1d(
            in_channels,
            out_channels,
            kernel_size=2 * stride,
            stride=stride,
            padding=stride // 2,
        )

    def __call__(self, x: mx.array) -> mx.array:
        """Apply encoder block with downsampling."""
        # Residual units
        for res_unit in self.res_units:
            x = res_unit(x)

        # Downsample
        x = self.snake(x)
        x = self.conv(x)

        return x


class DecoderBlock(nn.Module):
    """Decoder block with upsampling."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int,
    ):
        super().__init__()

        # Upsampling
        self.snake = Snake1d(in_channels)
        self.conv = ConvTranspose1d(
            in_channels,
            out_channels,
            kernel_size=2 * stride,
            stride=stride,
            padding=stride // 2,
        )

        # Residual units
        self.res_units = [
            ResidualUnit(out_channels, out_channels, dilation=1),
            ResidualUnit(out_channels, out_channels, dilation=3),
            ResidualUnit(out_channels, out_channels, dilation=9),
        ]

    def __call__(self, x: mx.array) -> mx.array:
        """Apply decoder block with upsampling."""
        # Upsample
        x = self.snake(x)
        x = self.conv(x)

        # Residual units
        for res_unit in self.res_units:
            x = res_unit(x)

        return x


class OobleckEncoder(nn.Module):
    """
    Oobleck Encoder for audio-to-latent conversion.

    Progressively downsamples audio using strided convolutions.
    """

    def __init__(self, config: VAEConfig):
        super().__init__()
        self.config = config

        # Initial projection
        self.conv_in = Conv1d(
            config.audio_channels,
            config.encoder_hidden_size,
            kernel_size=7,
            padding=3,
        )

        # Encoder blocks
        self.blocks = []
        in_channels = config.encoder_hidden_size

        for i, (stride, mult) in enumerate(
            zip(config.downsampling_ratios, config.channel_multiples)
        ):
            out_channels = config.encoder_hidden_size * mult
            self.blocks.append(EncoderBlock(in_channels, out_channels, stride))
            in_channels = out_channels

        # Final projection to latent space
        self.snake_out = Snake1d(in_channels)
        self.conv_out = Conv1d(
            in_channels,
            config.latent_channels * 2,  # Mean and log_var
            kernel_size=3,
            padding=1,
        )

    def __call__(self, x: mx.array) -> Tuple[mx.array, mx.array]:
        """
        Encode audio to latent distribution.

        Args:
            x: Audio waveform (batch, samples, channels)

        Returns:
            Tuple of (mean, log_var) for the latent distribution
        """
        # Initial projection
        x = self.conv_in(x)

        # Encoder blocks
        for block in self.blocks:
            x = block(x)

        # Final projection
        x = self.snake_out(x)
        x = self.conv_out(x)

        # Split into mean and log_var
        mean, log_var = mx.split(x, 2, axis=-1)

        return mean, log_var


class OobleckDecoder(nn.Module):
    """
    Oobleck Decoder for latent-to-audio conversion.

    Progressively upsamples latents to audio waveform.
    """

    def __init__(self, config: VAEConfig):
        super().__init__()
        self.config = config

        # Compute final encoder channels
        final_mult = config.channel_multiples[-1]
        final_channels = config.encoder_hidden_size * final_mult

        # Initial projection from latent space
        self.conv_in = Conv1d(
            config.latent_channels,
            final_channels,
            kernel_size=7,
            padding=3,
        )

        # Decoder blocks (reverse order)
        self.blocks = []
        in_channels = final_channels

        for stride, mult in zip(
            reversed(config.downsampling_ratios),
            reversed(config.channel_multiples[:-1]),
        ):
            out_channels = config.encoder_hidden_size * mult
            self.blocks.append(DecoderBlock(in_channels, out_channels, stride))
            in_channels = out_channels

        # Handle the last block to base channels
        if len(self.blocks) < len(config.downsampling_ratios):
            self.blocks.append(
                DecoderBlock(
                    in_channels,
                    config.decoder_hidden_size,
                    config.downsampling_ratios[0],
                )
            )
            in_channels = config.decoder_hidden_size

        # Final projection to audio
        self.snake_out = Snake1d(in_channels)
        self.conv_out = Conv1d(
            in_channels,
            config.audio_channels,
            kernel_size=7,
            padding=3,
        )

    def __call__(self, z: mx.array) -> mx.array:
        """
        Decode latents to audio.

        Args:
            z: Latent representation (batch, latent_length, latent_channels)

        Returns:
            Audio waveform (batch, samples, channels)
        """
        # Initial projection
        x = self.conv_in(z)

        # Decoder blocks
        for block in self.blocks:
            x = block(x)

        # Final projection
        x = self.snake_out(x)
        x = self.conv_out(x)

        # Clamp to valid audio range
        x = mx.clip(x, -1.0, 1.0)

        return x


class AutoencoderOobleck(nn.Module):
    """
    AutoencoderOobleck VAE for Stable Audio Open.

    A variational autoencoder that compresses 44.1kHz stereo audio
    into a 64-channel latent representation at ~21.5Hz.

    Compression ratio: 2048x (2 * 4 * 4 * 8 * 8)
    """

    def __init__(self, config: VAEConfig):
        super().__init__()
        self.config = config

        self.encoder = OobleckEncoder(config)
        self.decoder = OobleckDecoder(config)

        # Compute compression ratio
        self.compression_ratio = 1
        for r in config.downsampling_ratios:
            self.compression_ratio *= r

    @classmethod
    def from_config(cls, config: VAEConfig) -> "AutoencoderOobleck":
        """Create VAE from config."""
        return cls(config)

    def encode(
        self,
        x: mx.array,
        sample: bool = True,
    ) -> mx.array:
        """
        Encode audio to latent representation.

        Args:
            x: Audio waveform (batch, samples, channels) or (batch, channels, samples)
            sample: Whether to sample from posterior or return mean

        Returns:
            Latent representation (batch, latent_length, latent_channels)

        Raises:
            ValueError: If input shape is invalid
        """
        # Validate input dimensions
        if x.ndim != 3:
            raise ValueError(
                f"Expected 3D input (batch, samples, channels) or (batch, channels, samples), "
                f"got shape {x.shape}"
            )

        # Determine audio format based on dimension sizes
        # Channels are typically 1-2, while samples are typically thousands+
        expected_channels = self.config.audio_channels
        last_dim = x.shape[-1]
        second_dim = x.shape[1]

        # Use dimension magnitude to detect format
        # If last dim is small (<=8) and matches expected channels, it's samples-last format
        # If second dim is small (<=8) and matches expected channels, it's channels-first format
        if last_dim <= 8 and (last_dim == expected_channels or last_dim <= 2):
            pass  # Already in (batch, samples, channels) format
        elif second_dim <= 8 and (second_dim == expected_channels or second_dim <= 2):
            # Transpose from (batch, channels, samples) to (batch, samples, channels)
            x = mx.transpose(x, [0, 2, 1])
        else:
            # Ambiguous - assume samples-last if last dim is small
            if last_dim < second_dim:
                pass  # Assume (batch, samples, channels)
            else:
                raise ValueError(
                    f"Cannot determine audio format from shape {x.shape}. "
                    f"Expected either (batch, samples, {expected_channels}) or "
                    f"(batch, {expected_channels}, samples)"
                )

        # Get distribution parameters
        mean, log_var = self.encoder(x)

        # Clamp log_var for numerical stability (prevent extreme std values)
        log_var = mx.clip(log_var, -30.0, 20.0)

        if sample:
            # Reparameterization trick
            std = mx.exp(0.5 * log_var)
            eps = mx.random.normal(mean.shape)
            z = mean + std * eps
        else:
            z = mean

        return z

    def decode(self, z: mx.array) -> mx.array:
        """
        Decode latents to audio.

        Args:
            z: Latent representation (batch, latent_length, latent_channels)

        Returns:
            Audio waveform (batch, samples, channels)
        """
        return self.decoder(z)

    def __call__(
        self,
        x: mx.array,
        sample: bool = True,
    ) -> Tuple[mx.array, mx.array, mx.array]:
        """
        Full forward pass (encode -> sample -> decode).

        Args:
            x: Audio waveform
            sample: Whether to sample from posterior

        Returns:
            Tuple of (reconstruction, mean, log_var)
        """
        mean, log_var = self.encoder(x)

        # Clamp log_var for numerical stability
        log_var = mx.clip(log_var, -30.0, 20.0)

        if sample:
            std = mx.exp(0.5 * log_var)
            eps = mx.random.normal(mean.shape)
            z = mean + std * eps
        else:
            z = mean

        reconstruction = self.decoder(z)

        return reconstruction, mean, log_var

    @property
    def sample_rate(self) -> int:
        """Target audio sample rate."""
        return 44100

    @property
    def latent_rate(self) -> float:
        """Latent sequence rate in Hz."""
        return self.sample_rate / self.compression_ratio

    @property
    def latent_channels(self) -> int:
        """Number of latent channels."""
        return self.config.latent_channels


__all__ = [
    "AutoencoderOobleck",
    "OobleckEncoder",
    "OobleckDecoder",
    "Snake1d",
    "snake",
]
