"""Transformer blocks for MOSS Audio Tokenizer (MLX)."""

from __future__ import annotations

import math

import mlx.core as mx
import mlx.nn as nn


# =============================================================================
# Normalization & Scaling
# =============================================================================


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.alpha = mx.ones((1, 1, dim))

    def __call__(self, x: mx.array) -> mx.array:
        var = self.eps + mx.mean(x * x, axis=2, keepdims=True)
        return x * (self.alpha * mx.rsqrt(var))


class LayerScale(nn.Module):
    def __init__(self, channels: int, init: float = 1e-4):
        super().__init__()
        self.scale = mx.full((channels,), init)

    def __call__(self, x: mx.array) -> mx.array:
        return self.scale * x


def create_norm(norm_type: str, dim: int) -> nn.Module:
    if norm_type == "layer_norm":
        return nn.LayerNorm(dim, eps=1e-5)
    elif norm_type == "rms_norm":
        return RMSNorm(dim, eps=1e-5)
    else:
        raise ValueError(f"Unknown norm type: {norm_type}")


# =============================================================================
# Rotary Position Embedding (complex-number style)
# =============================================================================


def apply_rope(
    q: mx.array,
    k: mx.array,
    offset: int,
    max_period: float = 10_000,
) -> tuple[mx.array, mx.array]:
    """Apply complex-number style RoPE.

    q, k: (B, H, T, D) where D is even.
    offset: scalar position offset.
    """
    B, H, T, D = q.shape
    half_d = D // 2

    ds = mx.arange(half_d).astype(mx.float32)
    freqs = mx.exp(ds * (-math.log(max_period) * 2 / D))

    ts = mx.arange(T).astype(mx.float32) + offset
    # (T, half_d)
    angles = ts[:, None] * freqs[None, :]

    cos_a = mx.cos(angles)  # (T, half_d)
    sin_a = mx.sin(angles)  # (T, half_d)

    # Reshape q, k -> (..., half_d, 2) to extract real/imag pairs
    q_pairs = q.reshape(B, H, T, half_d, 2)
    k_pairs = k.reshape(B, H, T, half_d, 2)

    qr, qi = q_pairs[..., 0], q_pairs[..., 1]
    kr, ki = k_pairs[..., 0], k_pairs[..., 1]

    # Complex rotation: (qr + j*qi) * (cos + j*sin)
    qor = qr * cos_a - qi * sin_a
    qoi = qr * sin_a + qi * cos_a
    kor = kr * cos_a - ki * sin_a
    koi = kr * sin_a + ki * cos_a

    qo = mx.stack([qor, qoi], axis=-1).reshape(B, H, T, D)
    ko = mx.stack([kor, koi], axis=-1).reshape(B, H, T, D)
    return qo, ko


# =============================================================================
# Sinusoidal Positional Embedding
# =============================================================================


def create_sin_embedding(
    positions: mx.array,
    dim: int,
    max_period: float = 10000,
) -> mx.array:
    """Create sinusoidal positional embedding. positions: (B, T, 1)."""
    half_dim = dim // 2
    adim = mx.arange(half_dim).reshape(1, 1, -1).astype(mx.float32)
    phase = positions / (max_period ** (adim / (half_dim - 1)))
    return mx.concatenate([mx.cos(phase), mx.sin(phase)], axis=-1)


# =============================================================================
# Gating FFN
# =============================================================================


class ActivationGating(nn.Module):
    def __init__(self, dim: int, dim_feedforward: int):
        super().__init__()
        if dim_feedforward == 4 * dim:
            hidden = (21 * dim) // 8
        else:
            hidden = (2 * dim_feedforward) // 3
        self.linear_in = nn.Linear(dim, 2 * hidden, bias=False)
        self.linear_out = nn.Linear(hidden, dim, bias=False)

    def __call__(self, x: mx.array) -> mx.array:
        x = self.linear_in(x)
        B, T, _ = x.shape
        x = x.reshape(B, T, 2, -1)
        x = nn.gelu(x[:, :, 0, :]) * x[:, :, 1, :]
        return self.linear_out(x)


# =============================================================================
# Multi-Head Attention
# =============================================================================


class MultiheadAttention(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        causal: bool = False,
        context: int | None = None,
        use_rope: bool = False,
        max_period: float = 10000.0,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.causal = causal
        self.context = context
        self.use_rope = use_rope
        self.max_period = max_period
        self.dim_per_head = embed_dim // num_heads

        self.in_proj = nn.Linear(embed_dim, 3 * embed_dim, bias=False)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=False)

    def __call__(self, x: mx.array, offset: int = 0) -> mx.array:
        B, T, _ = x.shape

        projected = self.in_proj(x)
        projected = projected.reshape(B, T, 3, self.num_heads, self.dim_per_head)
        projected = projected.transpose(0, 2, 3, 1, 4)  # (B, 3, H, T, D)
        q, k, v = projected[:, 0], projected[:, 1], projected[:, 2]

        if self.use_rope:
            q, k = apply_rope(q, k, offset, self.max_period)

        if self.causal:
            total_len = offset + T
            if self.context is not None:
                # Windowed causal mask
                mask = mx.zeros((T, total_len))
                for i in range(T):
                    pos = offset + i
                    start = max(0, pos - self.context + 1)
                    mask = mask.at[i, start:pos + 1].add(1.0)
                mask = mx.where(mask > 0, mx.zeros_like(mask), mx.full(mask.shape, -1e9))
            else:
                # Standard causal mask
                causal_mask = nn.MultiHeadAttention.create_additive_causal_mask(T)
                mask = causal_mask
        else:
            mask = None

        scale = 1.0 / math.sqrt(self.dim_per_head)
        attn = (q @ k.transpose(0, 1, 3, 2)) * scale
        if mask is not None:
            attn = attn + mask
        attn = mx.softmax(attn, axis=-1)
        x = (attn @ v).transpose(0, 2, 1, 3).reshape(B, T, self.embed_dim)
        return self.out_proj(x)


# =============================================================================
# Transformer Layer
# =============================================================================


class TransformerLayer(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        dim_feedforward: int = 2048,
        causal: bool = False,
        context: int | None = None,
        use_rope: bool = False,
        max_period: float = 10000.0,
        norm: str = "layer_norm",
        layer_scale: float | None = None,
        gating: str = "none",
    ):
        super().__init__()
        self.self_attn = MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            causal=causal,
            context=context,
            use_rope=use_rope,
            max_period=max_period,
        )
        self.norm1 = create_norm(norm, d_model)
        self.norm2 = create_norm(norm, d_model)

        if gating == "none":
            self.linear1 = nn.Linear(d_model, dim_feedforward, bias=False)
            self.linear2 = nn.Linear(dim_feedforward, d_model, bias=False)
            self.gating_module = None
        else:
            self.linear1 = None
            self.linear2 = None
            self.gating_module = ActivationGating(d_model, dim_feedforward)

        if layer_scale is not None:
            self.layer_scale_1 = LayerScale(d_model, init=layer_scale)
            self.layer_scale_2 = LayerScale(d_model, init=layer_scale)
        else:
            self.layer_scale_1 = None
            self.layer_scale_2 = None

    def __call__(self, x: mx.array, offset: int = 0) -> mx.array:
        # Self-attention block
        h = self.norm1(x)
        h = self.self_attn(h, offset=offset)
        if self.layer_scale_1 is not None:
            h = self.layer_scale_1(h)
        x = x + h

        # FFN block
        h = self.norm2(x)
        if self.gating_module is not None:
            h = self.gating_module(h)
        else:
            h = self.linear2(nn.gelu(self.linear1(h)))
        if self.layer_scale_2 is not None:
            h = self.layer_scale_2(h)
        x = x + h
        return x


# =============================================================================
# Transformer Stack
# =============================================================================


class Transformer(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        num_layers: int,
        dim_feedforward: int = 2048,
        causal: bool = False,
        context: int | None = None,
        positional_embedding: str = "sin",
        max_period: float = 10_000,
        positional_scale: float = 1.0,
        norm: str = "layer_norm",
        layer_scale: float | None = None,
        gating: str = "none",
    ):
        super().__init__()
        self.positional_embedding = positional_embedding
        self.max_period = max_period
        self.positional_scale = positional_scale

        use_rope = positional_embedding in {"rope", "sin_rope"}

        self.layers = [
            TransformerLayer(
                d_model=d_model,
                num_heads=num_heads,
                dim_feedforward=dim_feedforward,
                causal=causal,
                context=context,
                use_rope=use_rope,
                max_period=max_period,
                norm=norm,
                layer_scale=layer_scale,
                gating=gating,
            )
            for _ in range(num_layers)
        ]

    def __call__(self, x: mx.array, offset: int = 0) -> mx.array:
        B, T, C = x.shape

        if self.positional_embedding in {"sin", "sin_rope"}:
            positions = (mx.arange(T) + offset).reshape(1, -1, 1).astype(mx.float32)
            pos_emb = create_sin_embedding(positions, C, max_period=self.max_period)
            x = x + self.positional_scale * pos_emb

        for layer in self.layers:
            x = layer(x, offset=offset)
        return x


# =============================================================================
# Projected Transformer
# =============================================================================


class ProjectedTransformer(nn.Module):
    """Transformer with input/output projections and conv_layout handling."""

    def __init__(
        self,
        input_dimension: int,
        output_dimension: int,
        d_model: int,
        context: int | None = None,
        conv_layout: bool = False,
        **kwargs,
    ):
        super().__init__()
        self.conv_layout = conv_layout
        self.downsample_ratio = 1

        if d_model != input_dimension:
            self.input_proj = nn.Linear(input_dimension, d_model, bias=False)
        else:
            self.input_proj = None

        self.transformer = Transformer(d_model=d_model, context=context, **kwargs)

        if d_model != output_dimension:
            self.output_proj = nn.Linear(d_model, output_dimension, bias=False)
        else:
            self.output_proj = None

    def __call__(self, x: mx.array, offset: int = 0) -> mx.array:
        # x: (B, D, T) in conv_layout
        x = x.transpose(0, 2, 1)  # (B, T, D)
        if self.input_proj is not None:
            x = self.input_proj(x)
        x = self.transformer(x, offset=offset)
        if self.output_proj is not None:
            x = self.output_proj(x)
        x = x.transpose(0, 2, 1)  # (B, D, T)
        return x
