"""
ACE-Step Transformer implementation for MLX.

The main diffusion transformer that processes audio latents
conditioned on text, lyrics, and speaker embeddings.

Architecture matches ACE-Step-v1-3.5B checkpoint structure.
"""

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import mlx.core as mx
import mlx.nn as nn

from mlx_music.models.ace_step.attention import (
    GLUMBConv,
    LinearTransformerBlock,
    Qwen2RotaryEmbedding,
    RMSNorm,
)


@dataclass
class ACEStepConfig:
    """Configuration for ACE-Step transformer."""

    # Core dimensions
    in_channels: int = 8
    out_channels: int = 8
    inner_dim: int = 2560
    num_layers: int = 24
    num_attention_heads: int = 20
    attention_head_dim: int = 128

    # Patch embedding
    patch_size: Tuple[int, int] = (16, 1)
    max_height: int = 16
    max_width: int = 32768

    # Position embeddings
    max_position: int = 32768
    rope_theta: float = 1000000.0

    # Feed-forward
    mlp_ratio: float = 2.5

    # Conditioning
    text_embedding_dim: int = 768
    speaker_embedding_dim: int = 512
    lyric_hidden_size: int = 1024
    lyric_encoder_vocab_size: int = 6693

    # SSL heads (for training, optional for inference)
    ssl_names: List[str] = field(default_factory=lambda: ["mert", "m-hubert"])
    ssl_latent_dims: List[int] = field(default_factory=lambda: [1024, 768])
    ssl_encoder_depths: List[int] = field(default_factory=lambda: [8, 8])

    @classmethod
    def from_dict(cls, config: Dict[str, Any]) -> "ACEStepConfig":
        """Create config from dictionary."""
        # Handle tuple conversion
        if "patch_size" in config and isinstance(config["patch_size"], list):
            config["patch_size"] = tuple(config["patch_size"])
        return cls(**{k: v for k, v in config.items() if k in cls.__dataclass_fields__})


class GroupNormLayer(nn.Module):
    """GroupNorm layer for proper weight naming.

    Checkpoint expects: proj_in.early_conv_layers.1.weight/bias
    """

    def __init__(self, num_groups: int, num_channels: int, eps: float = 1e-6):
        super().__init__()
        self.num_groups = num_groups
        self.num_channels = num_channels
        self.eps = eps
        self.weight = mx.ones((num_channels,))
        self.bias = mx.zeros((num_channels,))

    def __call__(self, x: mx.array) -> mx.array:
        """Apply group normalization (NHWC format)."""
        batch, h, w, c = x.shape
        group_size = c // self.num_groups

        # Reshape for group norm: (B, H, W, G, C//G)
        x = x.reshape(batch, h, w, self.num_groups, group_size)

        # GroupNorm normalizes over spatial dims (h, w) AND channels within each group
        # For shape (batch, h, w, groups, group_size), normalize over axes (1, 2, 4)
        mean = mx.mean(x, axis=(1, 2, 4), keepdims=True)
        var = mx.var(x, axis=(1, 2, 4), keepdims=True)
        x = (x - mean) / mx.sqrt(var + self.eps)

        # Reshape back
        x = x.reshape(batch, h, w, c)

        # Apply affine transform
        x = x * self.weight + self.bias

        return x


class PatchEmbed(nn.Module):
    """
    Patch embedding for audio latents.

    Converts (B, C, H, W) latent tensors to (B, N, D) sequences.
    Uses 3-layer early convolution as in ACE-Step:
    - Conv2d with patch_size stride
    - GroupNorm
    - Conv2d 1x1

    Weight naming: proj_in.early_conv_layers.{0,1,2}.{weight,bias}
    """

    def __init__(
        self,
        height: int = 16,
        width: int = 4096,
        patch_size: Tuple[int, int] = (16, 1),
        in_channels: int = 8,
        embed_dim: int = 2560,
        bias: bool = True,
    ):
        super().__init__()
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.embed_dim = embed_dim
        self.height = height // patch_size[0]
        self.width = width // patch_size[1]

        hidden_channels = in_channels * 256  # 8 * 256 = 2048

        # Early convolution layers - use list for proper indexed naming
        # Checkpoint expects: proj_in.early_conv_layers.0/1/2.weight/bias
        self.early_conv_layers = [
            # Layer 0: patch conv - Conv2d(8, 2048, kernel_size=(16, 1), stride=(16, 1))
            nn.Conv2d(
                in_channels,
                hidden_channels,
                kernel_size=patch_size,
                stride=patch_size,
                padding=0,
                bias=bias,
            ),
            # Layer 1: GroupNorm(32, 2048)
            GroupNormLayer(num_groups=32, num_channels=hidden_channels),
            # Layer 2: 1x1 conv - Conv2d(2048, 2560, kernel_size=(1, 1))
            nn.Conv2d(
                hidden_channels,
                embed_dim,
                kernel_size=(1, 1),
                stride=(1, 1),
                padding=0,
                bias=bias,
            ),
        ]

    def __call__(self, x: mx.array) -> mx.array:
        """
        Embed patches.

        Args:
            x: Input tensor (batch, channels, height, width) in NCHW format

        Returns:
            Sequence tensor (batch, num_patches, embed_dim)
        """
        # Convert NCHW to NHWC for MLX convolutions
        x = mx.transpose(x, axes=(0, 2, 3, 1))  # (B, H, W, C)

        # Apply early conv layers
        x = self.early_conv_layers[0](x)  # Patch conv
        x = self.early_conv_layers[1](x)  # GroupNorm
        x = self.early_conv_layers[2](x)  # 1x1 conv

        # Flatten spatial dimensions: (B, H, W, C) -> (B, H*W, C)
        batch, height, width, channels = x.shape
        x = x.reshape(batch, height * width, channels)

        return x


class T2IFinalLayer(nn.Module):
    """
    Final layer for diffusion transformer (Sana-style).

    Applies AdaLN modulation with scale_shift_table and projects back to latent space.
    """

    def __init__(
        self,
        hidden_size: int,
        patch_size: Tuple[int, int],
        out_channels: int,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.out_channels = out_channels
        self.patch_size = patch_size

        # Output dimension per patch
        patch_dim = out_channels * patch_size[0] * patch_size[1]

        self.norm_final = RMSNorm(hidden_size, elementwise_affine=False)
        self.linear = nn.Linear(hidden_size, patch_dim, bias=True)

        # Scale-shift table for AdaLN
        self.scale_shift_table = mx.zeros((2, hidden_size))

    def unpatchify(
        self,
        hidden_states: mx.array,
        output_length: int,
    ) -> mx.array:
        """Reconstruct latent from patches."""
        new_height, new_width = 1, hidden_states.shape[1]
        batch = hidden_states.shape[0]

        # Reshape: (B, N, patch_dim) -> (B, 1, N, ph, pw, C)
        hidden_states = hidden_states.reshape(
            batch,
            new_height,
            new_width,
            self.patch_size[0],
            self.patch_size[1],
            self.out_channels,
        )

        # Rearrange: nhwpqc -> nchpwq (NCHW output)
        hidden_states = mx.transpose(hidden_states, axes=(0, 5, 1, 3, 2, 4))
        output = hidden_states.reshape(
            batch,
            self.out_channels,
            new_height * self.patch_size[0],
            new_width * self.patch_size[1],
        )

        # Pad or trim to match output_length
        current_width = output.shape[3]
        if output_length > current_width:
            # Pad
            output = mx.pad(output, [(0, 0), (0, 0), (0, 0), (0, output_length - current_width)])
        elif output_length < current_width:
            # Trim
            output = output[:, :, :, :output_length]

        return output

    def __call__(
        self,
        x: mx.array,
        t: mx.array,
        output_length: int,
    ) -> mx.array:
        """
        Apply final layer with conditioning.

        Args:
            x: Hidden states (batch, seq, hidden_size)
            t: Timestep embedding (batch, hidden_size)
            output_length: Target output width

        Returns:
            Output tensor (batch, out_channels, H, W)
        """
        # Get modulation parameters from scale_shift_table + t
        params = self.scale_shift_table[None, :, :] + t[:, None, :]
        shift, scale = params[:, 0, :], params[:, 1, :]

        # Apply modulated normalization
        x = self.norm_final(x)
        x = x * (1 + scale[:, None, :]) + shift[:, None, :]

        # Project to output
        x = self.linear(x)

        # Unpatchify
        output = self.unpatchify(x, output_length)

        return output


class Timesteps(nn.Module):
    """Sinusoidal timestep embedding (diffusers compatible)."""

    def __init__(
        self,
        num_channels: int = 256,
        flip_sin_to_cos: bool = True,
        downscale_freq_shift: float = 0,
    ):
        super().__init__()
        self.num_channels = num_channels
        self.flip_sin_to_cos = flip_sin_to_cos
        self.downscale_freq_shift = downscale_freq_shift

    def __call__(self, timesteps: mx.array) -> mx.array:
        """
        Create sinusoidal timestep embeddings.

        Args:
            timesteps: Timestep values (batch,)

        Returns:
            Sinusoidal embeddings (batch, num_channels)
        """
        half_dim = self.num_channels // 2
        exponent = -math.log(10000) * mx.arange(half_dim, dtype=mx.float32)
        exponent = exponent / (half_dim - self.downscale_freq_shift)

        emb = timesteps[:, None].astype(mx.float32) * mx.exp(exponent)[None, :]

        if self.flip_sin_to_cos:
            emb = mx.concatenate([mx.cos(emb), mx.sin(emb)], axis=-1)
        else:
            emb = mx.concatenate([mx.sin(emb), mx.cos(emb)], axis=-1)

        return emb


class TimestepEmbedding(nn.Module):
    """Timestep embedding MLP."""

    def __init__(self, in_channels: int, time_embed_dim: int):
        super().__init__()
        self.linear_1 = nn.Linear(in_channels, time_embed_dim)
        self.linear_2 = nn.Linear(time_embed_dim, time_embed_dim)

    def __call__(self, sample: mx.array) -> mx.array:
        sample = self.linear_1(sample)
        sample = nn.silu(sample)
        sample = self.linear_2(sample)
        return sample


class ACEStepTransformer(nn.Module):
    """
    ACE-Step Diffusion Transformer (ACEStepTransformer2DModel).

    Main model class that processes audio latents conditioned on
    text prompts, lyrics, and speaker embeddings.

    Weight naming matches ACE-Step checkpoint for direct loading.
    """

    def __init__(self, config: ACEStepConfig):
        super().__init__()
        self.config = config
        self.inner_dim = config.num_attention_heads * config.attention_head_dim

        # Timestep embedding
        self.time_proj = Timesteps(num_channels=256, flip_sin_to_cos=True, downscale_freq_shift=0)
        self.timestep_embedder = TimestepEmbedding(256, self.inner_dim)

        # t_block: condition projection for AdaLN (6 params per layer)
        # Checkpoint expects: t_block.1.weight/bias (index 0 is SiLU with no params)
        self.t_block = [
            nn.SiLU(),
            nn.Linear(self.inner_dim, 6 * self.inner_dim, bias=True),
        ]

        # Patch embedding
        self.proj_in = PatchEmbed(
            height=config.max_height,
            width=config.max_width,
            patch_size=config.patch_size,
            in_channels=config.in_channels,
            embed_dim=self.inner_dim,
            bias=True,
        )

        # Conditioning embedders
        self.speaker_embedder = nn.Linear(config.speaker_embedding_dim, self.inner_dim, bias=True)
        self.genre_embedder = nn.Linear(config.text_embedding_dim, self.inner_dim, bias=True)

        # Lyric encoding (simple embedding + projection, full encoder loaded separately)
        self.lyric_embs = nn.Embedding(config.lyric_encoder_vocab_size, config.lyric_hidden_size)
        self.lyric_proj = nn.Linear(config.lyric_hidden_size, self.inner_dim, bias=True)

        # RoPE for positional encoding
        self.rotary_emb = Qwen2RotaryEmbedding(
            dim=config.attention_head_dim,
            max_position_embeddings=config.max_position,
            base=config.rope_theta,
        )

        # Transformer blocks
        self.transformer_blocks = [
            LinearTransformerBlock(
                dim=self.inner_dim,
                num_attention_heads=config.num_attention_heads,
                attention_head_dim=config.attention_head_dim,
                mlp_ratio=config.mlp_ratio,
                add_cross_attention=True,
                add_cross_attention_dim=self.inner_dim,  # Context is projected to inner_dim
            )
            for _ in range(config.num_layers)
        ]

        # SSL projectors (for training loss, not needed for inference)
        # Checkpoint expects: projectors.0.0/2/4.weight (indices 1,3 are SiLU with no params)
        projector_dim = 2 * self.inner_dim
        self.projectors = [
            [
                nn.Linear(self.inner_dim, projector_dim),
                nn.SiLU(),
                nn.Linear(projector_dim, projector_dim),
                nn.SiLU(),
                nn.Linear(projector_dim, ssl_dim),
            ]
            for ssl_dim in config.ssl_latent_dims
        ]

        # Final layer
        self.final_layer = T2IFinalLayer(
            hidden_size=self.inner_dim,
            patch_size=config.patch_size,
            out_channels=config.out_channels,
        )

    def encode(
        self,
        encoder_text_hidden_states: mx.array,
        text_attention_mask: mx.array,
        speaker_embeds: mx.array,
        lyric_token_idx: Optional[mx.array] = None,
        lyric_mask: Optional[mx.array] = None,
    ) -> Tuple[mx.array, mx.array]:
        """
        Encode conditioning inputs.

        Returns:
            encoder_hidden_states: Concatenated conditioning (batch, seq, inner_dim)
            encoder_hidden_mask: Attention mask (batch, seq)
        """
        batch_size = encoder_text_hidden_states.shape[0]

        # Speaker embedding (1 token)
        encoder_spk_hidden_states = self.speaker_embedder(speaker_embeds)[:, None, :]
        speaker_mask = mx.ones((batch_size, 1))

        # Genre/text embedding (project to inner_dim)
        encoder_text_hidden_states = self.genre_embedder(encoder_text_hidden_states)

        # Lyric embedding (simple projection for now)
        if lyric_token_idx is not None:
            lyric_emb = self.lyric_embs(lyric_token_idx)
            encoder_lyric_hidden_states = self.lyric_proj(lyric_emb)
        else:
            # Create empty lyric states
            encoder_lyric_hidden_states = mx.zeros((batch_size, 1, self.inner_dim))
            lyric_mask = mx.ones((batch_size, 1))

        # Concatenate all conditioning
        encoder_hidden_states = mx.concatenate(
            [encoder_spk_hidden_states, encoder_text_hidden_states, encoder_lyric_hidden_states],
            axis=1,
        )
        encoder_hidden_mask = mx.concatenate(
            [speaker_mask, text_attention_mask, lyric_mask],
            axis=1,
        )

        return encoder_hidden_states, encoder_hidden_mask

    def decode(
        self,
        hidden_states: mx.array,
        attention_mask: mx.array,
        encoder_hidden_states: mx.array,
        encoder_hidden_mask: mx.array,
        timestep: mx.array,
        output_length: int,
    ) -> mx.array:
        """
        Decode latents with transformer blocks.

        Args:
            hidden_states: Noisy latents (batch, channels, height, width)
            attention_mask: Self-attention mask (batch, seq)
            encoder_hidden_states: Conditioning (batch, seq_enc, inner_dim)
            encoder_hidden_mask: Conditioning mask (batch, seq_enc)
            timestep: Timestep (batch,)
            output_length: Target output width

        Returns:
            Predicted output (batch, out_channels, height, width)
        """
        # Timestep embedding
        t_emb = self.time_proj(timestep).astype(hidden_states.dtype)
        embedded_timestep = self.timestep_embedder(t_emb)

        # AdaLN conditioning (6 * inner_dim)
        # t_block is [SiLU, Linear]
        temb = self.t_block[0](embedded_timestep)
        temb = self.t_block[1](temb)

        # Patch embedding
        hidden_states = self.proj_in(hidden_states)

        # Get RoPE frequencies
        rotary_freqs_cis = self.rotary_emb(hidden_states, seq_len=hidden_states.shape[1])
        encoder_rotary_freqs_cis = self.rotary_emb(
            encoder_hidden_states, seq_len=encoder_hidden_states.shape[1]
        )

        # Transformer blocks
        for i, block in enumerate(self.transformer_blocks):
            hidden_states = block(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_hidden_mask,
                rotary_freqs_cis=rotary_freqs_cis,
                rotary_freqs_cis_cross=encoder_rotary_freqs_cis,
                temb=temb,
            )
            # Evaluate every 6 blocks to prevent graph explosion
            if (i + 1) % 6 == 0:
                mx.eval(hidden_states)

        # Final layer
        output = self.final_layer(hidden_states, embedded_timestep, output_length)

        return output

    def __call__(
        self,
        hidden_states: mx.array,
        timestep: mx.array,
        encoder_text_hidden_states: Optional[mx.array] = None,
        text_attention_mask: Optional[mx.array] = None,
        speaker_embeds: Optional[mx.array] = None,
        lyric_token_idx: Optional[mx.array] = None,
        lyric_mask: Optional[mx.array] = None,
        attention_mask: Optional[mx.array] = None,
    ) -> mx.array:
        """
        Forward pass.

        Args:
            hidden_states: Noisy latents (batch, channels, height, width)
            timestep: Diffusion timestep (batch,)
            encoder_text_hidden_states: Text embeddings (batch, seq, text_dim)
            text_attention_mask: Mask for text (batch, seq)
            speaker_embeds: Speaker embeddings (batch, speaker_dim)
            lyric_token_idx: Lyric token indices (batch, seq)
            lyric_mask: Mask for lyrics (batch, seq)
            attention_mask: Mask for self-attention (batch, seq_latent)

        Returns:
            Predicted noise/velocity (batch, channels, height, width)
        """
        batch_size = hidden_states.shape[0]
        output_length = hidden_states.shape[-1]

        # Default masks
        if attention_mask is None:
            seq_len = (hidden_states.shape[2] // self.config.patch_size[0]) * \
                      (hidden_states.shape[3] // self.config.patch_size[1])
            attention_mask = mx.ones((batch_size, seq_len))

        if text_attention_mask is None and encoder_text_hidden_states is not None:
            text_attention_mask = mx.ones((batch_size, encoder_text_hidden_states.shape[1]))

        if speaker_embeds is None:
            speaker_embeds = mx.zeros((batch_size, self.config.speaker_embedding_dim))

        if encoder_text_hidden_states is None:
            encoder_text_hidden_states = mx.zeros((batch_size, 1, self.config.text_embedding_dim))
            text_attention_mask = mx.ones((batch_size, 1))

        # Encode conditioning
        encoder_hidden_states, encoder_hidden_mask = self.encode(
            encoder_text_hidden_states=encoder_text_hidden_states,
            text_attention_mask=text_attention_mask,
            speaker_embeds=speaker_embeds,
            lyric_token_idx=lyric_token_idx,
            lyric_mask=lyric_mask,
        )

        # Decode
        output = self.decode(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_hidden_mask=encoder_hidden_mask,
            timestep=timestep,
            output_length=output_length,
        )

        return output
