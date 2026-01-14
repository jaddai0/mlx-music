"""
ACE-Step Transformer implementation for MLX.

The main diffusion transformer that processes audio latents
conditioned on text, lyrics, and speaker embeddings.
"""

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import mlx.core as mx
import mlx.nn as nn

from mlx_music.models.ace_step.attention import (
    AdaLNSingle,
    AttentionConfig,
    CrossAttention,
    FeedForward,
    LinearAttention,
    RMSNorm,
    RotaryEmbedding,
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


class PatchEmbed(nn.Module):
    """
    Patch embedding for audio latents.

    Converts (B, C, H, W) latent tensors to (B, N, D) sequences.
    """

    def __init__(
        self,
        in_channels: int = 8,
        embed_dim: int = 2560,
        patch_size: Tuple[int, int] = (16, 1),
    ):
        super().__init__()
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.embed_dim = embed_dim

        # Two-stage convolution
        mid_channels = embed_dim // 2 + embed_dim // 4  # ~2048

        # First conv: in_channels -> mid_channels
        self.conv1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=mid_channels,
            kernel_size=patch_size,
            stride=patch_size,
        )

        # Second conv: mid_channels -> embed_dim (1x1 conv)
        self.conv2 = nn.Conv2d(
            in_channels=mid_channels,
            out_channels=embed_dim,
            kernel_size=(1, 1),
            stride=(1, 1),
        )

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

        # Apply convolutions (MLX Conv2d expects NHWC)
        x = self.conv1(x)
        x = nn.silu(x)
        x = self.conv2(x)

        # Flatten spatial dimensions: (B, H, W, C) -> (B, H*W, C)
        batch, height, width, channels = x.shape
        x = x.reshape(batch, height * width, channels)

        return x


class T2IFinalLayer(nn.Module):
    """
    Final layer for diffusion transformer.

    Applies AdaLN modulation and projects back to latent space.
    """

    def __init__(
        self,
        hidden_dim: int,
        out_channels: int,
        patch_size: Tuple[int, int],
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.out_channels = out_channels
        self.patch_size = patch_size

        # Output dimension per patch
        patch_dim = out_channels * patch_size[0] * patch_size[1]

        self.norm = RMSNorm(hidden_dim)
        self.linear = nn.Linear(hidden_dim, patch_dim)

        # AdaLN modulation (scale, shift)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_dim, 2 * hidden_dim),
        )

    def __call__(
        self,
        x: mx.array,
        condition: mx.array,
    ) -> mx.array:
        """
        Apply final layer with conditioning.

        Args:
            x: Hidden states (batch, seq, hidden_dim)
            condition: Conditioning vector (batch, hidden_dim)

        Returns:
            Output tensor ready for unpatchify
        """
        # Get modulation parameters
        mod = self.adaLN_modulation(condition)
        shift, scale = mx.split(mod, 2, axis=-1)

        # Apply modulated normalization
        x = self.norm(x)
        x = x * (1 + scale[:, None, :]) + shift[:, None, :]

        # Project to output
        x = self.linear(x)

        return x


class LinearTransformerBlock(nn.Module):
    """
    Single transformer block with linear attention.

    Consists of:
    1. Self-attention (linear)
    2. Cross-attention (to text/lyric conditioning)
    3. Feed-forward network

    All with AdaLN-single conditioning.
    """

    def __init__(
        self,
        config: ACEStepConfig,
        has_cross_attention: bool = True,
    ):
        super().__init__()
        self.config = config
        self.has_cross_attention = has_cross_attention

        # Attention config
        attn_config = AttentionConfig(
            dim=config.inner_dim,
            num_heads=config.num_attention_heads,
            head_dim=config.attention_head_dim,
            use_linear_attention=True,
            use_rope=True,
            rope_theta=config.rope_theta,
            max_position=config.max_position,
        )

        # Layer norm with linear projection for AdaLN
        self.norm1 = nn.Sequential(
            RMSNorm(config.inner_dim),
            nn.Linear(config.inner_dim, config.inner_dim),
        )

        # Self-attention
        self.attn1 = LinearAttention(attn_config)

        # Cross-attention (optional)
        if has_cross_attention:
            self.norm2 = RMSNorm(config.inner_dim)
            self.attn2 = CrossAttention(
                attn_config,
                context_dim=config.text_embedding_dim,
            )

        # Feed-forward
        self.norm3 = RMSNorm(config.inner_dim)
        hidden_dim = int(config.inner_dim * config.mlp_ratio)
        self.ff = FeedForward(config.inner_dim, hidden_dim)

        # AdaLN parameters: [shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp]
        self.scale_shift_table = mx.zeros((6, config.inner_dim))

    def __call__(
        self,
        hidden_states: mx.array,
        condition: mx.array,
        encoder_hidden_states: Optional[mx.array] = None,
        attention_mask: Optional[mx.array] = None,
        encoder_attention_mask: Optional[mx.array] = None,
    ) -> mx.array:
        """
        Forward pass through the block.

        Args:
            hidden_states: Input (batch, seq, dim)
            condition: Timestep conditioning (batch, dim)
            encoder_hidden_states: Text/lyric embeddings (batch, seq_enc, enc_dim)
            attention_mask: Mask for self-attention
            encoder_attention_mask: Mask for cross-attention

        Returns:
            Output tensor (batch, seq, dim)
        """
        # Get AdaLN parameters
        params = condition[:, None, :] + self.scale_shift_table[None, :, :]
        shift_msa = params[:, 0, :]
        scale_msa = params[:, 1, :]
        gate_msa = params[:, 2, :]
        shift_mlp = params[:, 3, :]
        scale_mlp = params[:, 4, :]
        gate_mlp = params[:, 5, :]

        # Self-attention with AdaLN
        residual = hidden_states
        x = self.norm1(hidden_states)
        x = x * (1 + scale_msa[:, None, :]) + shift_msa[:, None, :]
        x = self.attn1(x, attention_mask=attention_mask)
        hidden_states = residual + gate_msa[:, None, :] * x

        # Cross-attention (if enabled)
        if self.has_cross_attention and encoder_hidden_states is not None:
            residual = hidden_states
            x = self.norm2(hidden_states)
            x = self.attn2(x, encoder_hidden_states, encoder_attention_mask)
            hidden_states = residual + x

        # Feed-forward with AdaLN
        residual = hidden_states
        x = self.norm3(hidden_states)
        x = x * (1 + scale_mlp[:, None, :]) + shift_mlp[:, None, :]
        x = self.ff(x)
        hidden_states = residual + gate_mlp[:, None, :] * x

        return hidden_states


class TimestepEmbedding(nn.Module):
    """Timestep embedding using sinusoidal encoding."""

    def __init__(self, channels: int, time_embed_dim: int):
        super().__init__()
        self.channels = channels

        self.linear_1 = nn.Linear(channels, time_embed_dim)
        self.linear_2 = nn.Linear(time_embed_dim, time_embed_dim)

    def __call__(self, timesteps: mx.array) -> mx.array:
        """
        Embed timesteps.

        Args:
            timesteps: Timestep values (batch,)

        Returns:
            Embedding (batch, time_embed_dim)
        """
        # Sinusoidal embedding
        half_dim = self.channels // 2
        freqs = mx.exp(
            -math.log(10000) * mx.arange(half_dim, dtype=mx.float32) / half_dim
        )
        args = timesteps[:, None].astype(mx.float32) * freqs[None, :]
        embedding = mx.concatenate([mx.cos(args), mx.sin(args)], axis=-1)

        # MLP
        embedding = self.linear_1(embedding)
        embedding = nn.silu(embedding)
        embedding = self.linear_2(embedding)

        return embedding


class ACEStepTransformer(nn.Module):
    """
    ACE-Step Diffusion Transformer.

    Main model class that processes audio latents conditioned on
    text prompts, lyrics, and speaker embeddings.
    """

    def __init__(self, config: ACEStepConfig):
        super().__init__()
        self.config = config

        # Timestep embedding
        self.time_proj = nn.Identity()  # Placeholder, actual projection in TimestepEmbedding
        self.timestep_embedder = TimestepEmbedding(256, config.inner_dim)

        # t_block: condition projection for AdaLN
        self.t_block = nn.Sequential(
            nn.SiLU(),
            nn.Linear(config.inner_dim, 6 * config.inner_dim),
        )

        # Patch embedding
        self.proj_in = PatchEmbed(
            in_channels=config.in_channels,
            embed_dim=config.inner_dim,
            patch_size=config.patch_size,
        )

        # Conditioning embedders
        self.speaker_embedder = nn.Linear(config.speaker_embedding_dim, config.inner_dim)
        self.genre_embedder = nn.Linear(config.text_embedding_dim, config.inner_dim)

        # Lyric encoding
        self.lyric_embs = nn.Embedding(config.lyric_encoder_vocab_size, config.lyric_hidden_size)
        self.lyric_proj = nn.Linear(config.lyric_hidden_size, config.inner_dim)

        # RoPE for positional encoding
        self.rotary_emb = RotaryEmbedding(
            dim=config.attention_head_dim,
            max_position=config.max_position,
            theta=config.rope_theta,
        )

        # Transformer blocks
        self.transformer_blocks = [
            LinearTransformerBlock(config, has_cross_attention=True)
            for _ in range(config.num_layers)
        ]

        # Final layer
        self.final_layer = T2IFinalLayer(
            hidden_dim=config.inner_dim,
            out_channels=config.out_channels,
            patch_size=config.patch_size,
        )

    def unpatchify(
        self,
        x: mx.array,
        height: int,
        width: int,
    ) -> mx.array:
        """
        Reconstruct latent from patches.

        Args:
            x: Patches (batch, num_patches, patch_dim)
            height: Original height in patches
            width: Original width in patches

        Returns:
            Reconstructed latent (batch, channels, H, W)
        """
        batch = x.shape[0]
        patch_h, patch_w = self.config.patch_size
        channels = self.config.out_channels

        # Reshape: (B, H*W, C*ph*pw) -> (B, H, W, C, ph, pw)
        x = x.reshape(batch, height, width, channels, patch_h, patch_w)

        # Rearrange: (B, H, W, C, ph, pw) -> (B, C, H*ph, W*pw)
        x = mx.transpose(x, axes=(0, 3, 1, 4, 2, 5))
        x = x.reshape(batch, channels, height * patch_h, width * patch_w)

        return x

    def __call__(
        self,
        hidden_states: mx.array,
        timestep: mx.array,
        encoder_hidden_states: Optional[mx.array] = None,
        attention_mask: Optional[mx.array] = None,
        encoder_attention_mask: Optional[mx.array] = None,
        speaker_embeds: Optional[mx.array] = None,
        lyric_token_idx: Optional[mx.array] = None,
        lyric_mask: Optional[mx.array] = None,
    ) -> mx.array:
        """
        Forward pass.

        Args:
            hidden_states: Noisy latents (batch, channels, height, width)
            timestep: Diffusion timestep (batch,)
            encoder_hidden_states: Text embeddings (batch, seq, dim)
            attention_mask: Mask for self-attention
            encoder_attention_mask: Mask for text
            speaker_embeds: Speaker embeddings (batch, speaker_dim)
            lyric_token_idx: Lyric token indices (batch, seq)
            lyric_mask: Mask for lyrics

        Returns:
            Predicted noise/velocity (batch, channels, height, width)
        """
        batch_size = hidden_states.shape[0]
        height = hidden_states.shape[2] // self.config.patch_size[0]
        width = hidden_states.shape[3] // self.config.patch_size[1]

        # Timestep embedding
        t_emb = self.timestep_embedder(timestep)

        # Add speaker conditioning if provided
        if speaker_embeds is not None:
            speaker_cond = self.speaker_embedder(speaker_embeds)
            t_emb = t_emb + speaker_cond

        # Add text/genre conditioning (mean pool)
        if encoder_hidden_states is not None:
            if encoder_attention_mask is not None:
                mask = encoder_attention_mask[:, :, None]
                genre_cond = mx.sum(encoder_hidden_states * mask, axis=1) / mx.sum(mask, axis=1)
            else:
                genre_cond = mx.mean(encoder_hidden_states, axis=1)
            genre_cond = self.genre_embedder(genre_cond)
            t_emb = t_emb + genre_cond

        # Compute AdaLN condition
        condition = self.t_block(t_emb)
        # Split into 6 parameters per layer (we'll pass the full condition)
        condition_per_layer = t_emb  # Simplified: pass base embedding

        # Patch embed
        hidden_states = self.proj_in(hidden_states)

        # Encode lyrics if provided
        if lyric_token_idx is not None:
            lyric_emb = self.lyric_embs(lyric_token_idx)
            lyric_emb = self.lyric_proj(lyric_emb)
            # Concatenate with text embeddings
            if encoder_hidden_states is not None:
                encoder_hidden_states = mx.concatenate(
                    [encoder_hidden_states, lyric_emb], axis=1
                )
                if encoder_attention_mask is not None and lyric_mask is not None:
                    encoder_attention_mask = mx.concatenate(
                        [encoder_attention_mask, lyric_mask], axis=1
                    )
            else:
                encoder_hidden_states = lyric_emb
                encoder_attention_mask = lyric_mask

        # Transformer blocks
        for block in self.transformer_blocks:
            hidden_states = block(
                hidden_states=hidden_states,
                condition=condition_per_layer,
                encoder_hidden_states=encoder_hidden_states,
                attention_mask=attention_mask,
                encoder_attention_mask=encoder_attention_mask,
            )

        # Final layer
        hidden_states = self.final_layer(hidden_states, t_emb)

        # Unpatchify
        output = self.unpatchify(hidden_states, height, width)

        return output
