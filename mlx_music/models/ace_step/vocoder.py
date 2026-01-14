"""
HiFi-GAN Vocoder for ACE-Step.

Converts mel-spectrograms to audio waveforms using the
ADaMoSHiFiGANV1 architecture with ConvNeXt backbone.

Key features:
- Weight normalization on conv layers
- ConvNeXt backbone for mel processing
- HiFi-GAN generator head
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import mlx.core as mx
import mlx.nn as nn


@dataclass
class VocoderConfig:
    """Configuration for HiFi-GAN vocoder."""

    # Input
    input_channels: int = 128  # Mel bins
    sampling_rate: int = 44100
    n_fft: int = 2048
    win_length: int = 2048
    hop_length: int = 512
    n_mels: int = 128
    f_min: float = 40.0
    f_max: float = 16000.0

    # ConvNeXt Backbone
    depths: List[int] = field(default_factory=lambda: [3, 3, 9, 3])
    dims: List[int] = field(default_factory=lambda: [128, 256, 384, 512])
    kernel_size: int = 7

    # HiFi-GAN Generator (from checkpoint analysis)
    upsample_rates: List[int] = field(default_factory=lambda: [4, 4, 2, 2, 2, 2, 2])
    upsample_kernel_sizes: List[int] = field(
        default_factory=lambda: [8, 8, 4, 4, 4, 4, 4]
    )
    upsample_initial_channel: int = 1024
    resblock_kernel_sizes: List[int] = field(default_factory=lambda: [3, 7, 11, 13])
    resblock_dilations: List[int] = field(default_factory=lambda: [1, 3, 5])
    pre_conv_kernel_size: int = 13
    post_conv_kernel_size: int = 13

    @classmethod
    def from_dict(cls, config: Dict[str, Any]) -> "VocoderConfig":
        """Create config from dictionary."""
        return cls(**{k: v for k, v in config.items() if k in cls.__dataclass_fields__})


class WeightNormConv1d(nn.Module):
    """
    Conv1d with weight normalization.

    Weight normalization decomposes the weight into magnitude (weight_g) and
    direction (weight_v):
        weight = weight_g * weight_v / ||weight_v||

    Checkpoint keys: weight_g, weight_v, bias (optional)
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
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups

        # Weight normalization parameters
        # weight_v: (out_channels, kernel_size, in_channels_per_group) - MLX format
        in_per_group = in_channels // groups
        self.weight_v = mx.random.normal(shape=(out_channels, kernel_size, in_per_group)) * 0.02
        # weight_g: (out_channels, 1, 1) - magnitude per output channel
        self.weight_g = mx.ones((out_channels, 1, 1))

        if bias:
            self.bias = mx.zeros((out_channels,))
        else:
            self.bias = None

    def _compute_weight(self) -> mx.array:
        """Compute normalized weight from weight_g and weight_v."""
        # Normalize weight_v over the kernel and input dimensions
        norm = mx.sqrt(mx.sum(self.weight_v ** 2, axis=(1, 2), keepdims=True) + 1e-5)  # Use 1e-5 for bfloat16 numerical stability
        return self.weight_g * self.weight_v / norm

    def __call__(self, x: mx.array) -> mx.array:
        # x: (batch, time, channels) - NLC format for MLX
        weight = self._compute_weight()
        y = mx.conv1d(x, weight, self.stride, self.padding, self.dilation, self.groups)
        if self.bias is not None:
            y = y + self.bias
        return y


class WeightNormConvTranspose1d(nn.Module):
    """
    ConvTranspose1d with weight normalization for upsampling.

    PyTorch ConvTranspose1d weight: (in_channels, out_channels, kernel_size)
    MLX conv_transpose1d weight: (out_channels, kernel_size, in_channels)

    Checkpoint keys: weight_g, weight_v, bias
    Note: weight_g shape in checkpoint is (in_channels, 1, 1) for PyTorch format
    """

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
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding
        self.groups = groups

        # Weight normalization parameters
        # MLX format: (out_channels, kernel_size, in_channels_per_group)
        in_per_group = in_channels // groups
        self.weight_v = mx.random.normal(shape=(out_channels, kernel_size, in_per_group)) * 0.02
        # weight_g: (1, 1, in_channels) for per-input-channel normalization
        # PyTorch ConvTranspose normalizes over (out, kernel) dims, scaling per input channel
        self.weight_g = mx.ones((1, 1, in_per_group))

        if bias:
            self.bias = mx.zeros((out_channels,))
        else:
            self.bias = None

    def _compute_weight(self) -> mx.array:
        """Compute normalized weight from weight_g and weight_v.

        For ConvTranspose with weight shape (out, kernel, in):
        - Norm is computed over (out, kernel) axes -> shape (1, 1, in)
        - weight_g scales per input channel -> shape (1, 1, in)
        """
        norm = mx.sqrt(mx.sum(self.weight_v ** 2, axis=(0, 1), keepdims=True) + 1e-5)
        return self.weight_g * self.weight_v / norm

    def __call__(self, x: mx.array) -> mx.array:
        # x: (batch, time, channels) - NLC format
        weight = self._compute_weight()

        # MLX conv_transpose1d expects weight shape (out_channels, kernel, in_channels/groups)
        y = mx.conv_transpose1d(
            x, weight,
            stride=self.stride,
            padding=self.padding,
            groups=self.groups,
        )

        # Apply output_padding manually (MLX doesn't have native support yet)
        # Output padding adds extra elements to one side of the output
        if self.output_padding > 0:
            y = mx.pad(y, [(0, 0), (0, self.output_padding), (0, 0)])

        if self.bias is not None:
            y = y + self.bias

        return y


class LayerNorm1d(nn.Module):
    """Layer normalization for 1D sequences (NLC format)."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = mx.ones((dim,))
        self.bias = mx.zeros((dim,))
        self.eps = eps

    def __call__(self, x: mx.array) -> mx.array:
        # x: (batch, time, channels) - NLC format
        mean = mx.mean(x, axis=-1, keepdims=True)
        var = mx.var(x, axis=-1, keepdims=True)
        x = (x - mean) / mx.sqrt(var + self.eps)
        return x * self.weight + self.bias


class ConvNeXtBlock(nn.Module):
    """
    ConvNeXt block for backbone.

    Checkpoint keys:
    - dwconv.weight, dwconv.bias
    - gamma (layer scale)
    - norm.weight, norm.bias
    - pwconv1.weight, pwconv1.bias
    - pwconv2.weight, pwconv2.bias
    """

    def __init__(self, dim: int, kernel_size: int = 7):
        super().__init__()

        # Depthwise convolution (groups=dim)
        padding = kernel_size // 2
        self.dwconv = nn.Conv1d(
            dim, dim, kernel_size=kernel_size, padding=padding, groups=dim
        )

        self.norm = LayerNorm1d(dim)
        self.pwconv1 = nn.Linear(dim, dim * 4)
        self.pwconv2 = nn.Linear(dim * 4, dim)

        # Layer scale parameter
        self.gamma = mx.ones((dim,)) * 1e-6

    def __call__(self, x: mx.array) -> mx.array:
        # x: (batch, time, channels) - NLC format
        residual = x

        x = self.dwconv(x)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = nn.gelu(x)
        x = self.pwconv2(x)
        x = x * self.gamma

        return x + residual


class ConvNeXtBackbone(nn.Module):
    """
    ConvNeXt backbone for mel processing.

    Checkpoint structure:
    - channel_layers.{0-3}.0: Conv1d for stem/downsampling
    - channel_layers.{0-3}.1: LayerNorm
    - stages.{0-3}.{0-N}: ConvNeXt blocks
    - norm: Final LayerNorm
    """

    def __init__(self, config: VocoderConfig):
        super().__init__()
        self.config = config

        # Channel layers as dict for checkpoint key matching
        self.channel_layers = {}

        # Stem (channel_layers.0)
        # Conv1d: input_channels -> dims[0]
        self.channel_layers["0"] = {
            "0": nn.Conv1d(
                config.input_channels,
                config.dims[0],
                kernel_size=config.kernel_size,
                padding=config.kernel_size // 2,
            ),
            "1": LayerNorm1d(config.dims[0]),
        }

        # Channel transitions for stages 1-3
        for i in range(1, 4):
            self.channel_layers[str(i)] = {
                "0": LayerNorm1d(config.dims[i - 1]),
                "1": nn.Conv1d(config.dims[i - 1], config.dims[i], kernel_size=1),
            }

        # Stages as dict
        self.stages = {}
        for i, (depth, dim) in enumerate(zip(config.depths, config.dims)):
            self.stages[str(i)] = {}
            for j in range(depth):
                self.stages[str(i)][str(j)] = ConvNeXtBlock(dim, config.kernel_size)

        # Final norm
        self.norm = LayerNorm1d(config.dims[-1])

    def __call__(self, x: mx.array) -> mx.array:
        # x: (batch, time, channels) - NLC format

        # Stem
        x = self.channel_layers["0"]["0"](x)
        x = self.channel_layers["0"]["1"](x)

        # Stage 0
        for j in range(self.config.depths[0]):
            x = self.stages["0"][str(j)](x)
        # Evaluate after stage to prevent graph explosion
        mx.eval(x)

        # Stages 1-3 with channel transitions
        for i in range(1, 4):
            # Channel transition
            x = self.channel_layers[str(i)]["0"](x)
            x = self.channel_layers[str(i)]["1"](x)

            # Blocks
            for j in range(self.config.depths[i]):
                x = self.stages[str(i)][str(j)](x)
            # Evaluate after each stage to prevent graph explosion
            mx.eval(x)

        x = self.norm(x)
        return x


class HiFiGANResBlock(nn.Module):
    """
    HiFi-GAN residual block with weight normalization.

    Checkpoint structure:
    - convs1.{0,1,2}: WeightNormConv1d with dilations [1, 3, 5]
    - convs2.{0,1,2}: WeightNormConv1d (post-activation)
    """

    def __init__(self, channels: int, kernel_size: int = 3, dilations: List[int] = [1, 3, 5]):
        super().__init__()
        self.channels = channels

        # Two sets of convolutions: convs1 (dilated) and convs2 (no dilation)
        self.convs1 = {}
        self.convs2 = {}

        for i, dilation in enumerate(dilations):
            padding = (kernel_size * dilation - dilation) // 2
            self.convs1[str(i)] = WeightNormConv1d(
                channels, channels, kernel_size, padding=padding, dilation=dilation
            )
            self.convs2[str(i)] = WeightNormConv1d(
                channels, channels, kernel_size, padding=kernel_size // 2
            )

        self.num_layers = len(dilations)

    def __call__(self, x: mx.array) -> mx.array:
        for i in range(self.num_layers):
            residual = x
            x = nn.leaky_relu(x, negative_slope=0.1)
            x = self.convs1[str(i)](x)
            x = nn.leaky_relu(x, negative_slope=0.1)
            x = self.convs2[str(i)](x)
            x = x + residual
        return x


class HiFiGANHead(nn.Module):
    """
    HiFi-GAN generator head with weight normalization.

    Checkpoint structure:
    - conv_pre: WeightNormConv1d
    - ups.{0-6}: WeightNormConvTranspose1d
    - resblocks.{0-27}: HiFiGANResBlock (4 per upsample stage)
    - conv_post: WeightNormConv1d
    """

    def __init__(self, config: VocoderConfig):
        super().__init__()
        self.config = config

        # Pre-conv from backbone output
        self.conv_pre = WeightNormConv1d(
            config.dims[-1],
            config.upsample_initial_channel,
            kernel_size=config.pre_conv_kernel_size,
            padding=config.pre_conv_kernel_size // 2,
        )

        # Upsampling layers as dict
        self.ups = {}
        ch = config.upsample_initial_channel

        for i, (rate, kernel) in enumerate(
            zip(config.upsample_rates, config.upsample_kernel_sizes)
        ):
            ch_out = ch // 2
            self.ups[str(i)] = WeightNormConvTranspose1d(
                ch,
                ch_out,
                kernel_size=kernel,
                stride=rate,
                padding=(kernel - rate) // 2,
            )
            ch = ch_out

        # Residual blocks (4 per upsample stage)
        # Total: 7 stages * 4 resblocks = 28 resblocks
        self.resblocks = {}
        ch = config.upsample_initial_channel

        resblock_idx = 0
        for i in range(len(config.upsample_rates)):
            ch = ch // 2
            for k in config.resblock_kernel_sizes:
                self.resblocks[str(resblock_idx)] = HiFiGANResBlock(
                    ch, k, config.resblock_dilations
                )
                resblock_idx += 1

        self.num_upsamples = len(config.upsample_rates)
        self.num_kernels = len(config.resblock_kernel_sizes)

        # Post-conv to audio (mono output)
        self.conv_post = WeightNormConv1d(
            ch, 1, kernel_size=config.post_conv_kernel_size,
            padding=config.post_conv_kernel_size // 2,
        )

    def __call__(self, x: mx.array) -> mx.array:
        x = self.conv_pre(x)
        x = nn.leaky_relu(x, negative_slope=0.1)

        for i in range(self.num_upsamples):
            x = self.ups[str(i)](x)
            x = nn.leaky_relu(x, negative_slope=0.1)

            # Apply resblocks and average
            xs = None
            for j in range(self.num_kernels):
                rb_idx = i * self.num_kernels + j
                if xs is None:
                    xs = self.resblocks[str(rb_idx)](x)
                else:
                    xs = xs + self.resblocks[str(rb_idx)](x)
            # Use float multiplication instead of integer division for clarity
            x = xs * (1.0 / self.num_kernels)

            # Evaluate after each upsample stage to prevent graph explosion
            mx.eval(x)

        x = nn.leaky_relu(x, negative_slope=0.1)
        x = self.conv_post(x)
        x = mx.tanh(x)

        return x


class HiFiGANVocoder(nn.Module):
    """
    Full HiFi-GAN vocoder for mel-to-audio conversion.

    Combines ConvNeXt backbone with HiFi-GAN generator head.

    Checkpoint structure:
    - backbone.*
    - head.*
    - mel_transform.* (not loaded into model)
    """

    def __init__(self, config: VocoderConfig):
        super().__init__()
        self.config = config
        self.backbone = ConvNeXtBackbone(config)
        self.head = HiFiGANHead(config)

    def __call__(self, mel: mx.array) -> mx.array:
        """
        Convert mel-spectrogram to audio.

        Args:
            mel: Log-mel spectrogram (batch, n_mels, time) in NCL format

        Returns:
            Audio waveform (batch, 1, samples) in NCL format
        """
        # Convert NCL to NLC for MLX Conv1d
        mel = mx.transpose(mel, axes=(0, 2, 1))  # (B, T, C)

        # Backbone: mel → features
        features = self.backbone(mel)

        # Head: features → audio
        audio = self.head(features)

        # Convert back to NCL format
        audio = mx.transpose(audio, axes=(0, 2, 1))  # (B, C, T)

        return audio

    def decode(self, mel: mx.array) -> mx.array:
        """Alias for __call__ for consistency."""
        return self(mel)

    @classmethod
    def from_pretrained(
        cls,
        model_path: str,
        dtype: mx.Dtype = mx.bfloat16,
    ) -> "HiFiGANVocoder":
        """Load vocoder from pretrained weights."""
        import json
        from pathlib import Path

        from mlx_music.weights.weight_loader import (
            load_safetensors,
            transpose_conv1d,
            transpose_conv_transpose1d,
            load_weights_with_string_keys,
        )

        model_path = Path(model_path)

        # Load config
        config_path = model_path / "config.json"
        if config_path.exists():
            with open(config_path) as f:
                config_dict = json.load(f)
            config = VocoderConfig.from_dict(config_dict)
        else:
            config = VocoderConfig()

        # Create model
        model = cls(config)

        # Load weights
        weight_file = model_path / "diffusion_pytorch_model.safetensors"
        if weight_file.exists():
            weights = load_safetensors(weight_file, dtype=dtype)

            # Transpose weights from PyTorch to MLX format
            # Conv1d: (out, in, kernel) -> (out, kernel, in)
            # ConvTranspose1d: (in, out, kernel) -> (out, kernel, in)
            transposed = {}
            for key, value in weights.items():
                # Skip mel_transform weights (not part of model)
                if key.startswith("mel_transform."):
                    continue

                if value.ndim == 3:
                    # Check if this is a ConvTranspose weight (in head.ups.*)
                    if "head.ups." in key and "weight_v" in key:
                        transposed[key] = transpose_conv_transpose1d(value)
                    else:
                        transposed[key] = transpose_conv1d(value)
                elif "head.ups." in key and "weight_g" in key:
                    # weight_g for ConvTranspose: (in, 1, 1) -> (out, 1, 1)
                    # We need to know the output channels. For upsampling layers:
                    # ups.0: in=1024, out=512 -> weight_g needs to be (512, 1, 1)
                    # Just transpose the first dimension by using the weight_v info
                    # Actually, we can derive from the layer structure
                    # For now, compute the norm at load time and store pre-normalized
                    transposed[key] = value
                else:
                    transposed[key] = value

            # For ConvTranspose layers, compute effective weight and store
            # This avoids the weight_g dimension mismatch
            processed = {}
            ups_weights = {}  # Collect weight_g, weight_v pairs

            for key, value in transposed.items():
                if "head.ups." in key:
                    # Parse: head.ups.0.weight_g -> layer=0, type=weight_g
                    parts = key.split(".")
                    layer_idx = parts[2]  # "0", "1", etc.
                    param_type = parts[3]  # "weight_g", "weight_v", "bias"

                    if layer_idx not in ups_weights:
                        ups_weights[layer_idx] = {}
                    ups_weights[layer_idx][param_type] = value
                else:
                    processed[key] = value

            # Process ConvTranspose weights - handle weight normalization correctly
            # PyTorch ConvTranspose: weight (in, out, k), weight_g (in, 1, 1)
            # MLX format after transpose: weight_v (out, k, in), weight_g needs (1, 1, in)
            for layer_idx, layer_weights in ups_weights.items():
                if "weight_g" in layer_weights and "weight_v" in layer_weights:
                    weight_v = layer_weights["weight_v"]  # Already transposed: (out, kernel, in)
                    weight_g = layer_weights["weight_g"]  # PyTorch format: (in, 1, 1)

                    # Transpose weight_g to match MLX format: (in, 1, 1) -> (1, 1, in)
                    weight_g_mlx = mx.transpose(weight_g, axes=(1, 2, 0))

                    processed[f"head.ups.{layer_idx}.weight_v"] = weight_v
                    processed[f"head.ups.{layer_idx}.weight_g"] = weight_g_mlx

                if "bias" in layer_weights:
                    processed[f"head.ups.{layer_idx}.bias"] = layer_weights["bias"]

            # Load into model
            load_weights_with_string_keys(model, processed, strict=False)

        return model


class MusicDCAEPipeline:
    """
    Complete audio processing pipeline.

    Combines DCAE and HiFi-GAN for:
    - audio → mel → latent (encode)
    - latent → mel → audio (decode)
    """

    def __init__(
        self,
        dcae: "DCAE",
        vocoder: HiFiGANVocoder,
        sample_rate: int = 44100,
    ):
        self.dcae = dcae
        self.vocoder = vocoder
        self.sample_rate = sample_rate

    def decode(self, latent: mx.array) -> mx.array:
        """
        Decode latent to audio.

        Args:
            latent: Latent (batch, 8, H, W)

        Returns:
            Stereo audio (batch, 2, samples)
        """
        # Decode to mel
        mel = self.dcae.decode(latent)

        # Denormalize mel
        mel = self.dcae.denormalize_mel(mel)

        # Vocoder: mel → audio (process channels separately)
        audio_ch1 = self.vocoder.decode(mel[:, 0:1, :, :].squeeze(1))
        audio_ch2 = self.vocoder.decode(mel[:, 1:2, :, :].squeeze(1))

        # Combine channels
        audio = mx.concatenate([audio_ch1, audio_ch2], axis=1)

        return audio

    @classmethod
    def from_pretrained(
        cls,
        model_path: str,
        dtype: mx.Dtype = mx.bfloat16,
    ) -> "MusicDCAEPipeline":
        """Load complete pipeline from pretrained weights."""
        from pathlib import Path

        from mlx_music.models.ace_step.dcae import DCAE

        model_path = Path(model_path)

        # Load DCAE
        dcae_path = model_path / "music_dcae_f8c8"
        dcae = DCAE.from_pretrained(str(dcae_path), dtype=dtype)

        # Load vocoder
        vocoder_path = model_path / "music_vocoder"
        vocoder = HiFiGANVocoder.from_pretrained(str(vocoder_path), dtype=dtype)

        return cls(dcae, vocoder)
