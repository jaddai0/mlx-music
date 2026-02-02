"""
StableAudioDiT Transformer for Stable Audio Open.

A 24-layer diffusion transformer with:
- Grouped Query Attention (GQA): 24 query heads, 12 KV heads
- Half-dimension Rotary Position Embeddings (RoPE)
- Cross-attention for text conditioning
- Adaptive Layer Normalization (AdaLN) for timestep/global conditioning
"""

import math
from typing import Optional, Tuple

import mlx.core as mx
import mlx.nn as nn

from mlx_music.models.stable_audio.config import DiTConfig


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = mx.ones((dim,))

    def __call__(self, x: mx.array) -> mx.array:
        # Standard RMSNorm: sqrt(mean(x^2) + eps)
        # Adding eps BEFORE sqrt is correct and efficient - ensures valid denominator
        ms = mx.mean(x * x, axis=-1, keepdims=True)
        rms = mx.sqrt(ms + self.eps)
        return (x / rms) * self.weight


class RotaryEmbedding(nn.Module):
    """
    Rotary Position Embeddings for half of the head dimensions.

    Stable Audio applies RoPE only to the first half of each head's dimension,
    leaving the second half unchanged.
    """

    def __init__(
        self,
        dim: int,
        max_seq_len: int = 8192,
        base: float = 10000.0,
    ):
        super().__init__()

        # Validate dimension is even for RoPE
        if dim % 2 != 0:
            raise ValueError(f"RoPE requires even dimension, got {dim}")

        self.dim = dim
        self.max_seq_len = max_seq_len
        self.base = base

        # RoPE applies to half the dimension
        self.rope_dim = dim // 2

        # Compute inverse frequencies
        inv_freq = 1.0 / (base ** (mx.arange(0, self.rope_dim, 2, dtype=mx.float32) / self.rope_dim))
        self.inv_freq = inv_freq

        # Build initial cache
        self._build_cache(max_seq_len)

    def _build_cache(self, seq_len: int):
        """Build cos/sin cache for positions."""
        self.max_seq_len_cached = seq_len
        t = mx.arange(seq_len, dtype=mx.float32)
        freqs = mx.outer(t, self.inv_freq)
        # Shape: (seq_len, rope_dim // 2)
        self.cos_cached = mx.cos(freqs)
        self.sin_cached = mx.sin(freqs)
        mx.eval(self.cos_cached, self.sin_cached)

    def __call__(self, seq_len: int) -> Tuple[mx.array, mx.array]:
        """
        Get cos/sin embeddings for sequence length.

        Returns:
            Tuple of (cos, sin) of shape (seq_len, rope_dim // 2)
        """
        if seq_len > self.max_seq_len_cached:
            self._build_cache(seq_len)

        return self.cos_cached[:seq_len], self.sin_cached[:seq_len]


def apply_rotary_emb_half(
    x: mx.array,
    cos: mx.array,
    sin: mx.array,
) -> mx.array:
    """
    Apply rotary embeddings to first half of dimensions only.

    Args:
        x: Input of shape (batch, seq, heads, head_dim)
        cos: Cosines of shape (seq, rope_dim // 2)
        sin: Sines of shape (seq, rope_dim // 2)

    Returns:
        Tensor with RoPE applied to first half of head_dim
    """
    # Split into rotary and passthrough parts
    head_dim = x.shape[-1]
    rope_dim = head_dim // 2
    x_rope = x[..., :rope_dim]
    x_pass = x[..., rope_dim:]

    # Reshape for rotation: (batch, seq, heads, rope_dim//2, 2)
    x_rope_complex = x_rope.reshape(*x_rope.shape[:-1], -1, 2)
    x_real = x_rope_complex[..., 0]
    x_imag = x_rope_complex[..., 1]

    # Expand cos/sin for broadcasting: (seq, rope_dim//2) -> (1, seq, 1, rope_dim//2)
    cos = cos[None, :, None, :]
    sin = sin[None, :, None, :]

    # Apply rotation
    x_rot_real = x_real * cos - x_imag * sin
    x_rot_imag = x_real * sin + x_imag * cos

    # Recombine
    x_rotated = mx.stack([x_rot_real, x_rot_imag], axis=-1)
    x_rotated = x_rotated.reshape(*x_rope.shape)

    # Concatenate with passthrough
    return mx.concatenate([x_rotated, x_pass], axis=-1)


class GQAttention(nn.Module):
    """
    Grouped Query Attention (GQA).

    Uses fewer KV heads than query heads for efficiency.
    Each KV head is shared across multiple query heads.
    """

    def __init__(
        self,
        dim: int,
        num_attention_heads: int = 24,
        num_key_value_heads: int = 12,
        head_dim: int = 64,
    ):
        super().__init__()

        # Validate GQA head configuration
        if num_attention_heads % num_key_value_heads != 0:
            raise ValueError(
                f"num_attention_heads ({num_attention_heads}) must be divisible by "
                f"num_key_value_heads ({num_key_value_heads})"
            )

        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.head_dim = head_dim
        self.num_head_groups = num_attention_heads // num_key_value_heads

        # QKV projections
        self.q_proj = nn.Linear(dim, num_attention_heads * head_dim, bias=False)
        self.k_proj = nn.Linear(dim, num_key_value_heads * head_dim, bias=False)
        self.v_proj = nn.Linear(dim, num_key_value_heads * head_dim, bias=False)
        self.out_proj = nn.Linear(num_attention_heads * head_dim, dim, bias=False)

        self.scale = head_dim ** -0.5

    def __call__(
        self,
        x: mx.array,
        cos: Optional[mx.array] = None,
        sin: Optional[mx.array] = None,
        mask: Optional[mx.array] = None,
    ) -> mx.array:
        """
        Apply grouped query attention.

        Args:
            x: Input of shape (batch, seq, dim)
            cos: RoPE cosines
            sin: RoPE sines
            mask: Optional attention mask

        Returns:
            Output of shape (batch, seq, dim)
        """
        batch, seq, _ = x.shape

        # Project to Q, K, V
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        # Reshape: (batch, seq, heads, head_dim)
        q = q.reshape(batch, seq, self.num_attention_heads, self.head_dim)
        k = k.reshape(batch, seq, self.num_key_value_heads, self.head_dim)
        v = v.reshape(batch, seq, self.num_key_value_heads, self.head_dim)

        # Apply RoPE if provided
        if cos is not None and sin is not None:
            q = apply_rotary_emb_half(q, cos, sin)
            k = apply_rotary_emb_half(k, cos, sin)

        # Repeat KV heads to match query heads
        # (batch, seq, kv_heads, head_dim) -> (batch, seq, num_heads, head_dim)
        k = mx.repeat(k, self.num_head_groups, axis=2)
        v = mx.repeat(v, self.num_head_groups, axis=2)

        # Transpose for attention: (batch, heads, seq, head_dim)
        q = mx.transpose(q, [0, 2, 1, 3])
        k = mx.transpose(k, [0, 2, 1, 3])
        v = mx.transpose(v, [0, 2, 1, 3])

        # Scaled dot-product attention
        attn = (q @ mx.transpose(k, [0, 1, 3, 2])) * self.scale

        if mask is not None:
            attn = attn + mask

        attn = mx.softmax(attn, axis=-1)
        out = attn @ v

        # Reshape back: (batch, heads, seq, head_dim) -> (batch, seq, dim)
        out = mx.transpose(out, [0, 2, 1, 3])
        out = out.reshape(batch, seq, -1)

        return self.out_proj(out)


class CrossAttention(nn.Module):
    """Cross-attention for text conditioning."""

    def __init__(
        self,
        dim: int,
        cross_attention_dim: int = 768,
        num_attention_heads: int = 24,
        head_dim: int = 64,
    ):
        super().__init__()
        self.num_attention_heads = num_attention_heads
        self.head_dim = head_dim

        # Projections
        self.q_proj = nn.Linear(dim, num_attention_heads * head_dim, bias=False)
        self.k_proj = nn.Linear(cross_attention_dim, num_attention_heads * head_dim, bias=False)
        self.v_proj = nn.Linear(cross_attention_dim, num_attention_heads * head_dim, bias=False)
        self.out_proj = nn.Linear(num_attention_heads * head_dim, dim, bias=False)

        self.scale = head_dim ** -0.5

    def __call__(
        self,
        x: mx.array,
        encoder_hidden_states: mx.array,
        mask: Optional[mx.array] = None,
    ) -> mx.array:
        """
        Apply cross-attention.

        Args:
            x: Query input of shape (batch, seq, dim)
            encoder_hidden_states: KV input of shape (batch, enc_seq, cross_dim)
            mask: Optional attention mask

        Returns:
            Output of shape (batch, seq, dim)
        """
        batch, seq, _ = x.shape
        enc_seq = encoder_hidden_states.shape[1]

        # Project
        q = self.q_proj(x)
        k = self.k_proj(encoder_hidden_states)
        v = self.v_proj(encoder_hidden_states)

        # Reshape
        q = q.reshape(batch, seq, self.num_attention_heads, self.head_dim)
        k = k.reshape(batch, enc_seq, self.num_attention_heads, self.head_dim)
        v = v.reshape(batch, enc_seq, self.num_attention_heads, self.head_dim)

        # Transpose: (batch, heads, seq, head_dim)
        q = mx.transpose(q, [0, 2, 1, 3])
        k = mx.transpose(k, [0, 2, 1, 3])
        v = mx.transpose(v, [0, 2, 1, 3])

        # Attention
        attn = (q @ mx.transpose(k, [0, 1, 3, 2])) * self.scale

        if mask is not None:
            attn = attn + mask

        attn = mx.softmax(attn, axis=-1)
        out = attn @ v

        # Reshape back
        out = mx.transpose(out, [0, 2, 1, 3])
        out = out.reshape(batch, seq, -1)

        return self.out_proj(out)


class FeedForward(nn.Module):
    """Feed-forward network with SiLU gating."""

    def __init__(self, dim: int, mult: float = 4.0):
        super().__init__()
        inner_dim = int(dim * mult)

        self.net = nn.Sequential(
            nn.Linear(dim, inner_dim * 2, bias=False),
            # SiLU gating applied manually
        )
        self.proj_out = nn.Linear(inner_dim, dim, bias=False)

    def __call__(self, x: mx.array) -> mx.array:
        # Project up with gating
        h = self.net(x)
        h, gate = mx.split(h, 2, axis=-1)
        h = h * nn.silu(gate)
        return self.proj_out(h)


class AdaLNModulation(nn.Module):
    """
    Adaptive Layer Normalization modulation.

    Computes scale and shift parameters from conditioning.
    """

    def __init__(self, cond_dim: int, hidden_dim: int, num_modulations: int = 6):
        super().__init__()
        self.num_modulations = num_modulations
        self.silu = nn.SiLU()
        self.linear = nn.Linear(cond_dim, hidden_dim * num_modulations, bias=False)

    def __call__(self, x: mx.array) -> Tuple[mx.array, ...]:
        """
        Compute modulation parameters.

        Args:
            x: Conditioning of shape (batch, cond_dim)

        Returns:
            Tuple of modulation parameters, each of shape (batch, hidden_dim)
        """
        x = self.silu(x)
        mods = self.linear(x)
        return mx.split(mods, self.num_modulations, axis=-1)


class DiTBlock(nn.Module):
    """
    Diffusion Transformer block.

    Contains:
    - Self-attention with GQA and RoPE
    - Cross-attention for text conditioning
    - Feed-forward network
    - AdaLN for timestep/global modulation
    """

    def __init__(self, config: DiTConfig):
        super().__init__()
        dim = config.num_attention_heads * config.attention_head_dim

        # Normalization layers
        self.norm1 = RMSNorm(dim)
        self.norm2 = RMSNorm(dim)
        self.norm3 = RMSNorm(dim)
        self.norm_cross = RMSNorm(dim)

        # Self-attention with GQA
        self.self_attn = GQAttention(
            dim=dim,
            num_attention_heads=config.num_attention_heads,
            num_key_value_heads=config.num_key_value_heads,
            head_dim=config.attention_head_dim,
        )

        # Cross-attention
        self.cross_attn = CrossAttention(
            dim=dim,
            cross_attention_dim=config.cross_attention_dim,
            num_attention_heads=config.num_attention_heads,
            head_dim=config.attention_head_dim,
        )

        # Feed-forward
        self.ff = FeedForward(dim, mult=config.ff_mult)

        # AdaLN modulation - projects from conditioning dim to hidden dim
        self.adaLN = AdaLNModulation(
            cond_dim=config.global_states_input_dim,
            hidden_dim=dim,
            num_modulations=6,
        )

    def __call__(
        self,
        x: mx.array,
        encoder_hidden_states: mx.array,
        global_cond: mx.array,
        cos: mx.array,
        sin: mx.array,
        encoder_attention_mask: Optional[mx.array] = None,
    ) -> mx.array:
        """
        Forward pass through DiT block.

        Args:
            x: Input of shape (batch, seq, dim)
            encoder_hidden_states: Text embeddings
            global_cond: Global conditioning (batch, cond_dim)
            cos, sin: RoPE embeddings
            encoder_attention_mask: Optional cross-attention mask

        Returns:
            Output of same shape as input
        """
        # Get modulation parameters
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN(global_cond)

        # Expand modulations for broadcasting
        shift_msa = shift_msa[:, None, :]
        scale_msa = scale_msa[:, None, :]
        gate_msa = gate_msa[:, None, :]
        shift_mlp = shift_mlp[:, None, :]
        scale_mlp = scale_mlp[:, None, :]
        gate_mlp = gate_mlp[:, None, :]

        # Self-attention with AdaLN
        h = self.norm1(x)
        h = h * (1 + scale_msa) + shift_msa
        h = self.self_attn(h, cos, sin)
        x = x + gate_msa * h

        # Cross-attention
        h = self.norm_cross(x)
        h = self.cross_attn(h, encoder_hidden_states, encoder_attention_mask)
        x = x + h

        # Feed-forward with AdaLN
        h = self.norm3(x)
        h = h * (1 + scale_mlp) + shift_mlp
        h = self.ff(h)
        x = x + gate_mlp * h

        return x


class StableAudioDiT(nn.Module):
    """
    Stable Audio Diffusion Transformer.

    A 24-layer transformer for audio diffusion with:
    - Patch embedding for latent inputs
    - Rotary position embeddings
    - Grouped query attention
    - Cross-attention for text conditioning
    - Adaptive layer normalization for timestep
    """

    # Frequency for calling mx.eval to prevent computation graph explosion.
    # Every N blocks, we materialize intermediate results to bound memory usage.
    # 6 blocks balances memory efficiency vs. graph optimization benefits.
    MATERIALIZE_BLOCK_FREQUENCY: int = 6

    def __init__(self, config: DiTConfig):
        super().__init__()
        self.config = config

        hidden_dim = config.num_attention_heads * config.attention_head_dim

        # Input projection
        self.proj_in = nn.Linear(config.in_channels, hidden_dim)

        # Timestep embedding
        self.timestep_proj = nn.Sequential(
            nn.Linear(config.timestep_features_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, config.global_states_input_dim),
        )

        # Global conditioning projection
        self.global_proj = nn.Linear(config.global_states_input_dim, config.global_states_input_dim)

        # Rotary embeddings
        self.rotary_emb = RotaryEmbedding(
            dim=config.attention_head_dim,
            max_seq_len=config.max_seq_len,
        )

        # Transformer blocks
        self.blocks = [DiTBlock(config) for _ in range(config.num_layers)]

        # Output projection
        self.norm_out = RMSNorm(hidden_dim)
        self.proj_out = nn.Linear(hidden_dim, config.out_channels)

    def _get_timestep_embedding(
        self,
        timesteps: mx.array,
        dim: int,
    ) -> mx.array:
        """
        Get sinusoidal timestep embedding.

        Args:
            timesteps: Timesteps of shape (batch,)
            dim: Embedding dimension

        Returns:
            Embeddings of shape (batch, dim)
        """
        half_dim = dim // 2
        freqs = mx.exp(-math.log(10000.0) * mx.arange(0, half_dim) / half_dim)

        args = timesteps[:, None] * freqs[None, :]
        embedding = mx.concatenate([mx.cos(args), mx.sin(args)], axis=-1)

        return embedding

    def __call__(
        self,
        hidden_states: mx.array,
        timestep: mx.array,
        encoder_hidden_states: mx.array,
        global_embed: mx.array,
        encoder_attention_mask: Optional[mx.array] = None,
    ) -> mx.array:
        """
        Forward pass through the transformer.

        Args:
            hidden_states: Latent input (batch, seq, in_channels)
            timestep: Current diffusion timestep (batch,)
            encoder_hidden_states: Text embeddings (batch, text_seq, cross_dim)
            global_embed: Global conditioning (batch, global_dim)
            encoder_attention_mask: Optional mask for text

        Returns:
            Denoised output (batch, seq, out_channels)
        """
        batch, seq_len, _ = hidden_states.shape

        # Project input
        x = self.proj_in(hidden_states)

        # Timestep embedding
        t_emb = self._get_timestep_embedding(timestep, self.config.timestep_features_dim)
        t_emb = self.timestep_proj(t_emb)

        # Combine with global conditioning
        global_cond = self.global_proj(global_embed) + t_emb

        # Get rotary embeddings
        cos, sin = self.rotary_emb(seq_len)

        # Process through blocks with memory checkpointing
        for i, block in enumerate(self.blocks):
            x = block(
                x,
                encoder_hidden_states,
                global_cond,
                cos,
                sin,
                encoder_attention_mask,
            )
            # Periodically materialize to prevent graph explosion and reduce memory
            if (i + 1) % self.MATERIALIZE_BLOCK_FREQUENCY == 0:
                mx.eval(x)

        # Output projection
        x = self.norm_out(x)
        x = self.proj_out(x)

        return x


__all__ = [
    "StableAudioDiT",
    "DiTBlock",
    "GQAttention",
    "CrossAttention",
    "RotaryEmbedding",
    "RMSNorm",
]
