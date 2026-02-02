"""
Attention mechanisms for ACE-Step.

Implements:
- Linear attention with ReLU kernel (O(n) complexity) for self-attention
- Standard scaled dot-product attention for cross-attention
- Rotary Position Embeddings (RoPE)
- GLUMBConv feed-forward network
"""

import math
from dataclasses import dataclass
from typing import Optional, Tuple, Union

import mlx.core as mx
import mlx.nn as nn


@dataclass
class AttentionConfig:
    """Configuration for attention layers."""

    dim: int = 2560
    num_heads: int = 20
    head_dim: int = 128
    dropout: float = 0.0
    mlp_ratio: float = 2.5
    use_rope: bool = True
    rope_theta: float = 1000000.0
    max_position: int = 32768


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization (without elementwise affine)."""

    def __init__(self, dim: int, eps: float = 1e-6, elementwise_affine: bool = False):
        super().__init__()
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        if elementwise_affine:
            self.weight = mx.ones((dim,))

    def __call__(self, x: mx.array) -> mx.array:
        # Standard RMSNorm: sqrt(mean(x^2) + eps)
        # Adding eps BEFORE sqrt is correct and efficient - ensures valid denominator
        ms = mx.mean(x * x, axis=-1, keepdims=True)
        rms = mx.sqrt(ms + self.eps)
        x = x / rms
        if self.elementwise_affine:
            x = x * self.weight
        return x


class Qwen2RotaryEmbedding(nn.Module):
    """
    Rotary Position Embeddings (RoPE) as used in Qwen2.

    Pre-computes cos/sin caches for efficient position encoding.
    """

    def __init__(
        self,
        dim: int,
        max_position_embeddings: int = 32768,
        base: float = 10000.0,
    ):
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base

        # Compute inverse frequencies
        inv_freq = 1.0 / (base ** (mx.arange(0, dim, 2, dtype=mx.float32) / dim))
        self.inv_freq = inv_freq

        # Build initial cache
        self._build_cache(max_position_embeddings)

    def _build_cache(self, seq_len: int):
        """Build cos/sin cache for positions up to seq_len."""
        self.max_seq_len_cached = seq_len
        t = mx.arange(seq_len, dtype=mx.float32)
        freqs = mx.outer(t, self.inv_freq)
        # Concatenate for interleaved pattern
        emb = mx.concatenate([freqs, freqs], axis=-1)
        self.cos_cached = mx.cos(emb)
        self.sin_cached = mx.sin(emb)
        # Evaluate cache to prevent graph accumulation on rebuild
        mx.eval(self.cos_cached, self.sin_cached)

    def __call__(self, x: mx.array, seq_len: Optional[int] = None) -> Tuple[mx.array, mx.array]:
        """
        Get cos/sin embeddings for the given sequence length.

        Args:
            x: Input tensor (used only for dtype)
            seq_len: Sequence length to get embeddings for

        Returns:
            Tuple of (cos, sin) embeddings of shape (seq_len, dim)
        """
        if seq_len is None:
            seq_len = x.shape[1] if x.ndim > 1 else x.shape[0]

        if seq_len > self.max_seq_len_cached:
            self._build_cache(seq_len)

        return (
            self.cos_cached[:seq_len].astype(x.dtype),
            self.sin_cached[:seq_len].astype(x.dtype),
        )


def apply_rotary_emb(
    x: mx.array,
    freqs_cis: Tuple[mx.array, mx.array],
) -> mx.array:
    """
    Apply rotary embeddings to input tensor.

    Args:
        x: Input tensor of shape (batch, heads, seq, head_dim)
        freqs_cis: Tuple of (cos, sin) tensors of shape (seq, head_dim)

    Returns:
        Tensor with rotary embeddings applied
    """
    cos, sin = freqs_cis
    # Expand for broadcasting: (seq, dim) -> (1, 1, seq, dim)
    cos = cos[None, None, :, :]
    sin = sin[None, None, :, :]

    # Split into real and imaginary parts for rotation (single reshape for efficiency)
    x_reshaped = x.reshape(*x.shape[:-1], -1, 2)
    x_real, x_imag = x_reshaped[..., 0], x_reshaped[..., 1]

    # Create rotated version: [-x_imag, x_real] interleaved
    x_rotated = mx.stack([-x_imag, x_real], axis=-1).reshape(x.shape)

    # Apply rotation (compute in float32 for numerical stability)
    cos_f32 = cos.astype(mx.float32)
    sin_f32 = sin.astype(mx.float32)
    out = x.astype(mx.float32) * cos_f32 + x_rotated.astype(mx.float32) * sin_f32
    return out.astype(x.dtype)


class ConvLayerWithNaming(nn.Module):
    """1D Convolution layer that exposes 'conv' attribute for proper weight naming.

    Checkpoint expects: ff.inverted_conv.conv.weight, ff.depth_conv.conv.weight, etc.
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        kernel_size: int = 3,
        stride: int = 1,
        dilation: int = 1,
        groups: int = 1,
        padding: Optional[int] = None,
        use_bias: bool = False,
        use_norm: bool = False,
        use_act: bool = False,
    ):
        super().__init__()
        if padding is None:
            # Same padding
            padding = (kernel_size // 2) * dilation

        # MLX Conv1d expects NLC format, weight shape: (out, kernel, in/groups)
        # Named 'conv' to match checkpoint: ff.inverted_conv.conv.weight
        self.conv = nn.Conv1d(
            in_dim,
            out_dim,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=use_bias,
        )
        self.norm = RMSNorm(out_dim, elementwise_affine=False) if use_norm else None
        self.act = nn.SiLU() if use_act else None

    def __call__(self, x: mx.array) -> mx.array:
        # x: (batch, seq, channels) - NLC format for MLX
        x = self.conv(x)
        if self.norm is not None:
            x = self.norm(x)
        if self.act is not None:
            x = self.act(x)
        return x


class GLUMBConv(nn.Module):
    """
    GLU-style Convolution block used in ACE-Step feed-forward.

    Structure:
    1. Inverted conv: in_features -> hidden_features * 2 (1x1)
    2. Depth conv: hidden_features * 2 -> hidden_features * 2 (grouped)
    3. GLU activation: split and gate
    4. Point conv: hidden_features -> out_features (1x1)

    Weight naming matches checkpoint:
    - ff.inverted_conv.conv.weight/bias
    - ff.depth_conv.conv.weight/bias
    - ff.point_conv.conv.weight
    """

    def __init__(
        self,
        in_features: int,
        hidden_features: int,
        out_features: Optional[int] = None,
        kernel_size: int = 3,
    ):
        super().__init__()
        out_features = out_features or in_features

        # Inverted convolution (1x1 pointwise)
        # Named to match: ff.inverted_conv.conv.weight
        self.inverted_conv = ConvLayerWithNaming(
            in_features,
            hidden_features * 2,
            kernel_size=1,
            use_bias=True,
            use_act=True,  # SiLU
        )

        # Depthwise convolution (grouped)
        # Named to match: ff.depth_conv.conv.weight
        self.depth_conv = ConvLayerWithNaming(
            hidden_features * 2,
            hidden_features * 2,
            kernel_size=kernel_size,
            groups=hidden_features * 2,
            use_bias=True,
            use_act=False,
        )

        # Pointwise convolution (1x1)
        # Named to match: ff.point_conv.conv.weight
        self.point_conv = ConvLayerWithNaming(
            hidden_features,
            out_features,
            kernel_size=1,
            use_bias=False,
            use_act=False,
        )

        self.glu_act = nn.SiLU()

    def __call__(self, x: mx.array) -> mx.array:
        # x: (batch, seq, channels) - already in NLC format
        x = self.inverted_conv(x)
        x = self.depth_conv(x)

        # GLU: split and gate
        x, gate = mx.split(x, 2, axis=-1)
        gate = self.glu_act(gate)
        x = x * gate

        x = self.point_conv(x)
        return x


class Attention(nn.Module):
    """
    Attention module matching diffusers Attention API.

    Supports both self-attention and cross-attention with optional
    added KV projections for joint attention.
    """

    def __init__(
        self,
        query_dim: int,
        heads: int = 8,
        dim_head: int = 64,
        cross_attention_dim: Optional[int] = None,
        added_kv_proj_dim: Optional[int] = None,
        out_dim: Optional[int] = None,
        bias: bool = True,
        context_pre_only: bool = False,
        qk_norm: Optional[str] = None,
    ):
        super().__init__()
        self.heads = heads
        self.dim_head = dim_head
        inner_dim = heads * dim_head
        out_dim = out_dim or query_dim

        self.is_cross_attention = cross_attention_dim is not None
        self.context_pre_only = context_pre_only

        # Query projection
        self.to_q = nn.Linear(query_dim, inner_dim, bias=bias)

        # Key/Value projections (from context if cross-attention)
        kv_dim = cross_attention_dim if cross_attention_dim else query_dim
        self.to_k = nn.Linear(kv_dim, inner_dim, bias=bias)
        self.to_v = nn.Linear(kv_dim, inner_dim, bias=bias)

        # Additional KV projections for joint attention
        if added_kv_proj_dim is not None:
            self.add_q_proj = nn.Linear(added_kv_proj_dim, inner_dim, bias=bias)
            self.add_k_proj = nn.Linear(added_kv_proj_dim, inner_dim, bias=bias)
            self.add_v_proj = nn.Linear(added_kv_proj_dim, inner_dim, bias=bias)
            if not context_pre_only:
                self.to_add_out = nn.Linear(inner_dim, added_kv_proj_dim, bias=bias)

        # Output projection
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, out_dim, bias=bias),
            nn.Dropout(0.0),  # Placeholder for API compatibility
        )

        # QK normalization
        if qk_norm is not None:
            self.norm_q = RMSNorm(dim_head)
            self.norm_k = RMSNorm(dim_head)
        else:
            self.norm_q = None
            self.norm_k = None


class LiteLAAttention(nn.Module):
    """
    Linear Attention with ReLU kernel (LiteLA) for self-attention.

    This is the main self-attention mechanism in ACE-Step.
    Uses O(n) linear attention instead of O(n^2) softmax attention.
    """

    def __init__(
        self,
        dim: int,
        heads: int = 20,
        dim_head: int = 128,
        bias: bool = True,
    ):
        super().__init__()
        self.heads = heads
        self.dim_head = dim_head
        self.inner_dim = heads * dim_head
        self.eps = 1e-6  # Increased from 1e-15 for bfloat16 numerical stability
        self.pad_val = 1.0

        # Projections
        self.to_q = nn.Linear(dim, self.inner_dim, bias=bias)
        self.to_k = nn.Linear(dim, self.inner_dim, bias=bias)
        self.to_v = nn.Linear(dim, self.inner_dim, bias=bias)

        # Output projection - use list for proper weight naming (to_out.0.weight)
        self.to_out = [
            nn.Linear(self.inner_dim, dim, bias=bias),
        ]

        # QK normalization (not used in ACE-Step)
        self.norm_q = None
        self.norm_k = None

    def __call__(
        self,
        hidden_states: mx.array,
        attention_mask: Optional[mx.array] = None,
        rotary_freqs_cis: Optional[Tuple[mx.array, mx.array]] = None,
    ) -> mx.array:
        """
        Apply linear attention.

        Args:
            hidden_states: Input (batch, seq, dim)
            attention_mask: Optional mask (batch, seq)
            rotary_freqs_cis: Optional RoPE frequencies (cos, sin)

        Returns:
            Output (batch, seq, dim)
        """
        batch_size, seq_len, _ = hidden_states.shape

        # Project to Q, K, V
        query = self.to_q(hidden_states)
        key = self.to_k(hidden_states)
        value = self.to_v(hidden_states)

        # Reshape to (batch, heads, head_dim, seq) for linear attention
        query = query.reshape(batch_size, seq_len, self.heads, self.dim_head)
        key = key.reshape(batch_size, seq_len, self.heads, self.dim_head)
        value = value.reshape(batch_size, seq_len, self.heads, self.dim_head)

        # Apply QK normalization if present
        if self.norm_q is not None:
            query = self.norm_q(query)
        if self.norm_k is not None:
            key = self.norm_k(key)

        # Transpose to (batch, heads, seq, head_dim) for RoPE
        query = mx.transpose(query, axes=(0, 2, 1, 3))
        key = mx.transpose(key, axes=(0, 2, 1, 3))
        value = mx.transpose(value, axes=(0, 2, 1, 3))

        # Apply RoPE
        if rotary_freqs_cis is not None:
            query = apply_rotary_emb(query, rotary_freqs_cis)
            key = apply_rotary_emb(key, rotary_freqs_cis)

        # Transpose for linear attention: query to (batch, heads, head_dim, seq)
        query = mx.transpose(query, axes=(0, 1, 3, 2))
        # key stays as (batch, heads, seq, head_dim)
        # value to (batch, heads, head_dim, seq)
        value = mx.transpose(value, axes=(0, 1, 3, 2))

        # Apply attention mask
        if attention_mask is not None:
            # mask: (batch, seq) -> (batch, 1, seq, 1) for key
            mask = attention_mask[:, None, :, None].astype(key.dtype)
            key = key * mask
            # For value: (batch, heads, head_dim, seq) needs (batch, 1, 1, seq)
            value = value * mx.transpose(mask, axes=(0, 1, 3, 2))

        # ReLU kernel activation
        query = mx.maximum(query, 0)
        key = mx.maximum(key, 0)

        # Convert to float32 for numerical stability
        query = query.astype(mx.float32)
        key = key.astype(mx.float32)
        value = value.astype(mx.float32)

        # Pad value for normalization (add a row of 1s)
        # value: (batch, heads, head_dim, seq) -> (batch, heads, head_dim+1, seq)
        value = mx.pad(value, [(0, 0), (0, 0), (0, 1), (0, 0)], constant_values=self.pad_val)

        # Linear attention: V @ K^T @ Q
        # vk: (batch, heads, head_dim+1, head_dim)
        vk = mx.matmul(value, key)

        # out: (batch, heads, head_dim+1, seq)
        out = mx.matmul(vk, query)

        # Normalize using the padded row (use mx.maximum for extra stability)
        normalizer = mx.maximum(out[:, :, -1:], self.eps)
        out = out[:, :, :-1] / normalizer

        # Reshape back: (batch, heads, head_dim, seq) -> (batch, seq, heads * head_dim)
        out = mx.transpose(out, axes=(0, 3, 1, 2))  # (batch, seq, heads, head_dim)
        out = out.reshape(batch_size, seq_len, -1)

        # Convert back to original dtype
        out = out.astype(hidden_states.dtype)

        # Output projection
        out = self.to_out[0](out)

        return out


class SDPACrossAttention(nn.Module):
    """
    Scaled Dot-Product Cross Attention for ACE-Step.

    Used for cross-attention between latent and encoder hidden states.
    """

    def __init__(
        self,
        dim: int,
        cross_attention_dim: int,
        heads: int = 20,
        dim_head: int = 128,
        bias: bool = True,
    ):
        super().__init__()
        self.heads = heads
        self.dim_head = dim_head
        self.inner_dim = heads * dim_head
        self.scale = dim_head ** -0.5

        # Query from latent
        self.to_q = nn.Linear(dim, self.inner_dim, bias=bias)
        # Key/Value from encoder (context)
        self.to_k = nn.Linear(cross_attention_dim, self.inner_dim, bias=bias)
        self.to_v = nn.Linear(cross_attention_dim, self.inner_dim, bias=bias)

        # Additional projections for joint attention (latent side of context)
        self.add_q_proj = nn.Linear(cross_attention_dim, self.inner_dim, bias=bias)
        self.add_k_proj = nn.Linear(cross_attention_dim, self.inner_dim, bias=bias)
        self.add_v_proj = nn.Linear(cross_attention_dim, self.inner_dim, bias=bias)

        # Output - use list for proper weight naming (to_out.0.weight)
        self.to_out = [
            nn.Linear(self.inner_dim, dim, bias=bias),
        ]
        self.to_add_out = nn.Linear(self.inner_dim, cross_attention_dim, bias=bias)

        # QK normalization (not used by default)
        self.norm_q = None
        self.norm_k = None

    def __call__(
        self,
        hidden_states: mx.array,
        encoder_hidden_states: mx.array,
        attention_mask: Optional[mx.array] = None,
        encoder_attention_mask: Optional[mx.array] = None,
        rotary_freqs_cis: Optional[Tuple[mx.array, mx.array]] = None,
        rotary_freqs_cis_cross: Optional[Tuple[mx.array, mx.array]] = None,
    ) -> mx.array:
        """
        Apply cross attention.

        Args:
            hidden_states: Query input (batch, seq_q, dim)
            encoder_hidden_states: Key/Value input (batch, seq_kv, cross_dim)
            attention_mask: Mask for hidden_states (batch, seq_q)
            encoder_attention_mask: Mask for encoder (batch, seq_kv)
            rotary_freqs_cis: RoPE for query
            rotary_freqs_cis_cross: RoPE for key

        Returns:
            Output (batch, seq_q, dim)
        """
        batch_size, seq_q, _ = hidden_states.shape
        seq_kv = encoder_hidden_states.shape[1]

        # Project query from hidden states, key/value from encoder
        query = self.to_q(hidden_states)
        key = self.to_k(encoder_hidden_states)
        value = self.to_v(encoder_hidden_states)

        # Reshape to multi-head: (batch, seq, heads, head_dim)
        query = query.reshape(batch_size, seq_q, self.heads, self.dim_head)
        key = key.reshape(batch_size, seq_kv, self.heads, self.dim_head)
        value = value.reshape(batch_size, seq_kv, self.heads, self.dim_head)

        # Transpose to (batch, heads, seq, head_dim)
        query = mx.transpose(query, axes=(0, 2, 1, 3))
        key = mx.transpose(key, axes=(0, 2, 1, 3))
        value = mx.transpose(value, axes=(0, 2, 1, 3))

        # Apply QK normalization if present
        if self.norm_q is not None:
            query = self.norm_q(query)
        if self.norm_k is not None:
            key = self.norm_k(key)

        # Apply RoPE
        if rotary_freqs_cis is not None:
            query = apply_rotary_emb(query, rotary_freqs_cis)
        if rotary_freqs_cis_cross is not None:
            key = apply_rotary_emb(key, rotary_freqs_cis_cross)

        # Create attention mask from both masks
        attn_mask = None
        if attention_mask is not None and encoder_attention_mask is not None:
            # Combined mask: (batch, seq_q, seq_kv)
            combined = attention_mask[:, :, None] * encoder_attention_mask[:, None, :]
            # Convert to additive mask (use -1e4 for float16 compatibility, -65504 is float16 min)
            attn_mask = mx.where(combined == 1, mx.zeros_like(combined), mx.array(-1e4))
            # Expand for heads: (batch, 1, seq_q, seq_kv) -> (batch, heads, seq_q, seq_kv)
            attn_mask = attn_mask[:, None, :, :].astype(query.dtype)

        # Scaled dot-product attention
        scores = mx.matmul(query, mx.transpose(key, axes=(0, 1, 3, 2))) * self.scale

        if attn_mask is not None:
            scores = scores + attn_mask

        weights = mx.softmax(scores, axis=-1)
        out = mx.matmul(weights, value)

        # Reshape back: (batch, heads, seq_q, head_dim) -> (batch, seq_q, inner_dim)
        out = mx.transpose(out, axes=(0, 2, 1, 3))
        out = out.reshape(batch_size, seq_q, -1)

        # Output projection
        out = self.to_out[0](out)

        return out


class LinearTransformerBlock(nn.Module):
    """
    ACE-Step Transformer block with:
    - Linear self-attention (LiteLA) - named 'attn' to match checkpoint
    - Cross-attention to encoder hidden states - named 'cross_attn' to match checkpoint
    - GLUMBConv feed-forward - named 'ff' to match checkpoint
    - AdaLN-single conditioning via scale_shift_table

    Weight naming matches checkpoint:
    - transformer_blocks.{i}.attn.to_q/k/v/to_out.0
    - transformer_blocks.{i}.cross_attn.to_q/k/v/to_out.0/add_*
    - transformer_blocks.{i}.ff.inverted_conv/depth_conv/point_conv
    - transformer_blocks.{i}.scale_shift_table
    """

    def __init__(
        self,
        dim: int,
        num_attention_heads: int,
        attention_head_dim: int,
        mlp_ratio: float = 2.5,
        add_cross_attention: bool = True,
        add_cross_attention_dim: Optional[int] = None,
    ):
        super().__init__()
        self.dim = dim
        self.add_cross_attention = add_cross_attention

        # RMSNorm layers (without elementwise affine - no learnable params, not in checkpoint)
        # These are used for AdaLN modulation
        self.norm1 = RMSNorm(dim, elementwise_affine=False)
        self.norm2 = RMSNorm(dim, elementwise_affine=False)

        # Self-attention (linear attention) - named 'attn' to match checkpoint
        self.attn = LiteLAAttention(
            dim=dim,
            heads=num_attention_heads,
            dim_head=attention_head_dim,
            bias=True,
        )

        # Cross-attention (standard SDPA) - named 'cross_attn' to match checkpoint
        if add_cross_attention and add_cross_attention_dim is not None:
            self.cross_attn = SDPACrossAttention(
                dim=dim,
                cross_attention_dim=add_cross_attention_dim,
                heads=num_attention_heads,
                dim_head=attention_head_dim,
                bias=True,
            )

        # Feed-forward (GLUMBConv) - named 'ff' to match checkpoint
        hidden_features = int(dim * mlp_ratio)
        self.ff = GLUMBConv(
            in_features=dim,
            hidden_features=hidden_features,
            out_features=dim,
            kernel_size=3,
        )

        # AdaLN scale-shift table (6 values: shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp)
        self.scale_shift_table = mx.zeros((6, dim))

    def __call__(
        self,
        hidden_states: mx.array,
        attention_mask: Optional[mx.array] = None,
        encoder_hidden_states: Optional[mx.array] = None,
        encoder_attention_mask: Optional[mx.array] = None,
        rotary_freqs_cis: Optional[Tuple[mx.array, mx.array]] = None,
        rotary_freqs_cis_cross: Optional[Tuple[mx.array, mx.array]] = None,
        temb: Optional[mx.array] = None,
    ) -> mx.array:
        """
        Forward pass through the block.

        Args:
            hidden_states: Input (batch, seq, dim)
            attention_mask: Self-attention mask (batch, seq)
            encoder_hidden_states: Context for cross-attention (batch, seq_enc, dim)
            encoder_attention_mask: Cross-attention mask (batch, seq_enc)
            rotary_freqs_cis: RoPE for self-attention
            rotary_freqs_cis_cross: RoPE for cross-attention
            temb: Timestep embedding for AdaLN (batch, 6*dim)

        Returns:
            Output (batch, seq, dim)
        """
        batch_size = hidden_states.shape[0]

        # Get AdaLN parameters from temb
        if temb is not None:
            # temb: (batch, 6*dim) -> (batch, 6, dim)
            temb_reshaped = temb.reshape(batch_size, 6, -1)
            # Add scale_shift_table: (batch, 6, dim) + (1, 6, dim)
            params = temb_reshaped + self.scale_shift_table[None, :, :]
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
                params[:, 0], params[:, 1], params[:, 2],
                params[:, 3], params[:, 4], params[:, 5]
            )
        else:
            shift_msa = scale_msa = gate_msa = None
            shift_mlp = scale_mlp = gate_mlp = None

        # Self-attention with AdaLN
        norm_hidden = self.norm1(hidden_states)
        if scale_msa is not None:
            norm_hidden = norm_hidden * (1 + scale_msa[:, None, :]) + shift_msa[:, None, :]

        attn_out = self.attn(
            norm_hidden,
            attention_mask=attention_mask,
            rotary_freqs_cis=rotary_freqs_cis,
        )

        if gate_msa is not None:
            attn_out = gate_msa[:, None, :] * attn_out
        hidden_states = hidden_states + attn_out

        # Cross-attention
        if self.add_cross_attention and encoder_hidden_states is not None:
            cross_out = self.cross_attn(
                hidden_states=hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                attention_mask=attention_mask,
                encoder_attention_mask=encoder_attention_mask,
                rotary_freqs_cis=rotary_freqs_cis,
                rotary_freqs_cis_cross=rotary_freqs_cis_cross,
            )
            hidden_states = hidden_states + cross_out

        # Feed-forward with AdaLN
        norm_hidden = self.norm2(hidden_states)
        if scale_mlp is not None:
            norm_hidden = norm_hidden * (1 + scale_mlp[:, None, :]) + shift_mlp[:, None, :]

        ff_out = self.ff(norm_hidden)

        if gate_mlp is not None:
            ff_out = gate_mlp[:, None, :] * ff_out
        hidden_states = hidden_states + ff_out

        return hidden_states


# Backward compatibility aliases
FeedForward = GLUMBConv
RotaryEmbedding = Qwen2RotaryEmbedding
CrossAttention = SDPACrossAttention
LinearAttention = LiteLAAttention
AdaLNSingle = None  # Not used directly
