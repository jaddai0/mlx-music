"""
Oobleck VAE for ACE-Step v1.5.

1D waveform VAE with Snake activation for 48kHz stereo audio.
Config: 2ch, 48kHz, downsampling [2,4,4,6,10] = 1920x compression,
64-dim latent at 25Hz.

All operations use MLX NLC format (batch, time, channels).
Weight normalization is decomposed at load time.
"""

import math
from typing import List, Optional, Tuple

import mlx.core as mx
import mlx.nn as nn


class Snake1d(nn.Module):
    """Snake activation: x + (1/beta) * sin^2(alpha * x).

    Parameters alpha and beta are per-channel in log space.
    """

    def __init__(self, channels: int):
        super().__init__()
        self.alpha = mx.zeros((channels,))
        self.beta = mx.zeros((channels,))

    def __call__(self, x: mx.array) -> mx.array:
        # x: (B, T, C) in NLC format
        alpha = mx.exp(self.alpha)
        beta = mx.exp(self.beta)
        sin_val = mx.sin(alpha * x)
        return x + (1.0 / (beta + 1e-9)) * sin_val * sin_val


class OobleckResidualUnit(nn.Module):
    """Residual block with dilated convolution + pointwise convolution."""

    def __init__(self, dimension: int, dilation: int = 1):
        super().__init__()
        pad = ((7 - 1) * dilation) // 2

        self.snake1 = Snake1d(dimension)
        self.conv1 = nn.Conv1d(
            dimension, dimension, kernel_size=7, dilation=dilation, padding=pad
        )
        self.snake2 = Snake1d(dimension)
        self.conv2 = nn.Conv1d(dimension, dimension, kernel_size=1)

    def __call__(self, x: mx.array) -> mx.array:
        residual = x
        out = self.snake1(x)
        out = self.conv1(out)
        out = self.snake2(out)
        out = self.conv2(out)

        # Handle size mismatch from padding
        if residual.shape[1] != out.shape[1]:
            diff = residual.shape[1] - out.shape[1]
            pad_left = diff // 2
            residual = residual[:, pad_left : pad_left + out.shape[1], :]

        return residual + out


class OobleckEncoderBlock(nn.Module):
    """Encoder block: 3x ResUnits (dilation 1,3,9) -> Snake -> Downsample."""

    def __init__(self, input_dim: int, output_dim: int, stride: int = 1):
        super().__init__()
        self.res_unit1 = OobleckResidualUnit(input_dim, dilation=1)
        self.res_unit2 = OobleckResidualUnit(input_dim, dilation=3)
        self.res_unit3 = OobleckResidualUnit(input_dim, dilation=9)
        self.snake1 = Snake1d(input_dim)
        self.conv1 = nn.Conv1d(
            input_dim,
            output_dim,
            kernel_size=2 * stride,
            stride=stride,
            padding=math.ceil(stride / 2),
        )

    def __call__(self, x: mx.array) -> mx.array:
        x = self.res_unit1(x)
        x = self.res_unit2(x)
        x = self.res_unit3(x)
        x = self.snake1(x)
        x = self.conv1(x)
        return x


class OobleckDecoderBlock(nn.Module):
    """Decoder block: Snake -> Upsample -> 3x ResUnits (dilation 1,3,9)."""

    def __init__(self, input_dim: int, output_dim: int, stride: int = 1):
        super().__init__()
        self.snake1 = Snake1d(input_dim)
        self.conv_t1 = nn.ConvTranspose1d(
            input_dim,
            output_dim,
            kernel_size=2 * stride,
            stride=stride,
            padding=math.ceil(stride / 2),
        )
        self.res_unit1 = OobleckResidualUnit(output_dim, dilation=1)
        self.res_unit2 = OobleckResidualUnit(output_dim, dilation=3)
        self.res_unit3 = OobleckResidualUnit(output_dim, dilation=9)

    def __call__(self, x: mx.array) -> mx.array:
        x = self.snake1(x)
        x = self.conv_t1(x)
        x = self.res_unit1(x)
        x = self.res_unit2(x)
        x = self.res_unit3(x)
        return x


class OobleckEncoder(nn.Module):
    """Full Oobleck encoder: Conv -> 5 blocks -> Snake -> Conv.

    Produces 128-dim output split into mean (64) + scale (64) for VAE.
    """

    def __init__(
        self,
        encoder_hidden_size: int = 128,
        audio_channels: int = 2,
        downsampling_ratios: List[int] = None,
        channel_multiples: List[int] = None,
    ):
        super().__init__()
        if downsampling_ratios is None:
            downsampling_ratios = [2, 4, 4, 6, 10]
        if channel_multiples is None:
            channel_multiples = [1, 2, 4, 8, 16]

        self.conv1 = nn.Conv1d(audio_channels, encoder_hidden_size, kernel_size=7, padding=3)

        full_multiples = [1] + channel_multiples
        self.block = []
        for i, stride in enumerate(downsampling_ratios):
            in_ch = encoder_hidden_size * full_multiples[i]
            out_ch = encoder_hidden_size * full_multiples[i + 1]
            self.block.append(OobleckEncoderBlock(in_ch, out_ch, stride))

        d_model = encoder_hidden_size * full_multiples[-1]
        self.snake1 = Snake1d(d_model)
        self.conv2 = nn.Conv1d(d_model, encoder_hidden_size, kernel_size=3, padding=1)

    def __call__(self, x: mx.array) -> mx.array:
        """
        Args:
            x: (B, T, audio_channels) NLC format

        Returns:
            (B, T/hop_length, encoder_hidden_size)
        """
        x = self.conv1(x)
        for block in self.block:
            x = block(x)
        x = self.snake1(x)
        x = self.conv2(x)
        return x


class OobleckDecoder(nn.Module):
    """Full Oobleck decoder: Conv -> 5 blocks -> Snake -> Conv."""

    def __init__(
        self,
        channels: int = 128,
        input_channels: int = 64,
        audio_channels: int = 2,
        upsampling_ratios: List[int] = None,
        channel_multiples: List[int] = None,
    ):
        super().__init__()
        if upsampling_ratios is None:
            upsampling_ratios = [10, 6, 4, 4, 2]
        if channel_multiples is None:
            channel_multiples = [1, 2, 4, 8, 16]

        full_multiples = [1] + channel_multiples
        self.conv1 = nn.Conv1d(
            input_channels, channels * full_multiples[-1], kernel_size=7, padding=3
        )

        self.block = []
        n = len(upsampling_ratios)
        for i, stride in enumerate(upsampling_ratios):
            in_ch = channels * full_multiples[n - i]
            out_ch = channels * full_multiples[n - i - 1]
            self.block.append(OobleckDecoderBlock(in_ch, out_ch, stride))

        self.snake1 = Snake1d(channels)
        self.conv2 = nn.Conv1d(channels, audio_channels, kernel_size=7, padding=3, bias=False)

    def __call__(self, x: mx.array) -> mx.array:
        """
        Args:
            x: (B, T/hop_length, input_channels) NLC format

        Returns:
            (B, T, audio_channels) NLC format
        """
        x = self.conv1(x)
        for block in self.block:
            x = block(x)
        x = self.snake1(x)
        x = self.conv2(x)
        return x


class AutoencoderOobleck(nn.Module):
    """Oobleck VAE for 48kHz stereo audio.

    Encoder: stereo audio (B, T, 2) -> latent (B, T/1920, 64)
    Decoder: latent (B, T/1920, 64) -> stereo audio (B, T, 2)
    """

    def __init__(
        self,
        encoder_hidden_size: int = 128,
        downsampling_ratios: List[int] = None,
        channel_multiples: List[int] = None,
        decoder_channels: int = 128,
        decoder_input_channels: int = 64,
        audio_channels: int = 2,
        sampling_rate: int = 48000,
    ):
        super().__init__()
        if downsampling_ratios is None:
            downsampling_ratios = [2, 4, 4, 6, 10]
        if channel_multiples is None:
            channel_multiples = [1, 2, 4, 8, 16]

        self.encoder = OobleckEncoder(
            encoder_hidden_size, audio_channels, downsampling_ratios, channel_multiples
        )
        self.decoder = OobleckDecoder(
            decoder_channels,
            decoder_input_channels,
            audio_channels,
            downsampling_ratios[::-1],
            channel_multiples,
        )

        self.hop_length = 1
        for r in downsampling_ratios:
            self.hop_length *= r

        self.sampling_rate = sampling_rate
        self.latent_dim = decoder_input_channels

    def encode(self, x: mx.array) -> Tuple[mx.array, mx.array]:
        """Encode audio to latent space.

        Args:
            x: (B, T, audio_channels) stereo audio in NLC format

        Returns:
            (mean, log_var) each of shape (B, T/hop_length, latent_dim)
        """
        h = self.encoder(x)  # (B, T/hop, encoder_hidden_size)
        mean, scale = mx.split(h, 2, axis=-1)  # each (B, T/hop, 64)
        # Softplus for positive scale, then log variance
        std = mx.log(1 + mx.exp(scale)) + 1e-4
        log_var = 2 * mx.log(std)
        return mean, log_var

    def encode_mean(self, x: mx.array) -> mx.array:
        """Encode audio to latent mean (no sampling, for inference)."""
        h = self.encoder(x)
        mean, _ = mx.split(h, 2, axis=-1)
        return mean

    def decode(self, z: mx.array) -> mx.array:
        """Decode latent to audio.

        Args:
            z: (B, T/hop_length, latent_dim) latent representation

        Returns:
            (B, T, audio_channels) reconstructed audio
        """
        return self.decoder(z)

    @classmethod
    def from_config(cls, config: dict) -> "AutoencoderOobleck":
        """Create from diffusers config dict."""
        return cls(
            encoder_hidden_size=config.get("encoder_hidden_size", 128),
            downsampling_ratios=config.get("downsampling_ratios", [2, 4, 4, 6, 10]),
            channel_multiples=config.get("channel_multiples", [1, 2, 4, 8, 16]),
            decoder_channels=config.get("decoder_channels", 128),
            decoder_input_channels=config.get("decoder_input_channels", 64),
            audio_channels=config.get("audio_channels", 2),
            sampling_rate=config.get("sampling_rate", 48000),
        )


__all__ = [
    "Snake1d",
    "OobleckResidualUnit",
    "OobleckEncoderBlock",
    "OobleckDecoderBlock",
    "OobleckEncoder",
    "OobleckDecoder",
    "AutoencoderOobleck",
]
