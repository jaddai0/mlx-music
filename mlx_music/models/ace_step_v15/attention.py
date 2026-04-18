"""
Attention mechanisms for ACE-Step v1.5.

Implements:
- GQA (Grouped Query Attention) with 16 heads, 8 KV heads
- RMSNorm on Q/K (Qwen3-style)
- Rotary Position Embeddings (RoPE)
- Sliding window attention via additive masks
- Qwen3-style SiLU gated MLP
"""

import math
from typing import Dict, Optional, Tuple

import mlx.core as mx
import mlx.nn as nn


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization with learnable weight."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = mx.ones((dim,))

    def __call__(self, x: mx.array) -> mx.array:
        ms = mx.mean(x * x, axis=-1, keepdims=True)
        x = x * mx.rsqrt(ms + self.eps)
        return x * self.weight


class RotaryEmbedding(nn.Module):
    """Rotary Position Embeddings (RoPE) with cos/sin cache."""

    def __init__(
        self,
        dim: int,
        max_position_embeddings: int = 32768,
        base: float = 1000000.0,
    ):
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base

        inv_freq = 1.0 / (base ** (mx.arange(0, dim, 2, dtype=mx.float32) / dim))
        self._inv_freq = inv_freq
        self._build_cache(max_position_embeddings)

    def _build_cache(self, seq_len: int):
        self._max_seq_len_cached = seq_len
        t = mx.arange(seq_len, dtype=mx.float32)
        freqs = mx.outer(t, self._inv_freq)
        emb = mx.concatenate([freqs, freqs], axis=-1)
        self._cos_cached = mx.cos(emb)
        self._sin_cached = mx.sin(emb)
        # Materialize cache to prevent graph accumulation
        mx.eval(self._cos_cached, self._sin_cached)

    def __call__(self, seq_len: int, dtype: mx.Dtype = mx.float32) -> Tuple[mx.array, mx.array]:
        if seq_len > self._max_seq_len_cached:
            self._build_cache(seq_len)
        return (
            self._cos_cached[:seq_len].astype(dtype),
            self._sin_cached[:seq_len].astype(dtype),
        )


def _rotate_half(x: mx.array) -> mx.array:
    """Rotate half of the hidden dims: [-x2, x1]."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return mx.concatenate([-x2, x1], axis=-1)


def apply_rotary_pos_emb(
    q: mx.array,
    k: mx.array,
    cos: mx.array,
    sin: mx.array,
) -> Tuple[mx.array, mx.array]:
    """Apply RoPE to query and key tensors.

    Args:
        q: (batch, heads, seq, head_dim)
        k: (batch, heads, seq, head_dim)
        cos: (seq, head_dim)
        sin: (seq, head_dim)

    Returns:
        Rotated (q, k) tuple.
    """
    cos = cos[None, None, :, :]
    sin = sin[None, None, :, :]

    q_f32 = q.astype(mx.float32)
    k_f32 = k.astype(mx.float32)
    cos_f32 = cos.astype(mx.float32)
    sin_f32 = sin.astype(mx.float32)

    q_embed = q_f32 * cos_f32 + _rotate_half(q_f32) * sin_f32
    k_embed = k_f32 * cos_f32 + _rotate_half(k_f32) * sin_f32

    return q_embed.astype(q.dtype), k_embed.astype(k.dtype)


def create_sliding_mask(
    seq_len: int,
    sliding_window: Optional[int],
    is_sliding: bool,
    is_causal: bool = False,
    dtype: mx.Dtype = mx.float32,
) -> Optional[mx.array]:
    """Create additive attention mask for sliding window attention.

    Args:
        seq_len: Sequence length.
        sliding_window: Window size (|i-j| <= window).
        is_sliding: Whether this layer uses sliding window.
        is_causal: Whether to apply causal mask.
        dtype: Output dtype.

    Returns:
        Additive mask of shape (1, 1, seq_len, seq_len) or None for full attention.
    """
    if not is_sliding or sliding_window is None:
        return None

    indices = mx.arange(seq_len)
    diff = indices[:, None] - indices[None, :]

    valid = mx.abs(diff) <= sliding_window

    if is_causal:
        valid = valid & (diff >= 0)

    mask = mx.where(valid, mx.zeros((seq_len, seq_len), dtype=dtype), mx.array(float("-inf"), dtype=dtype))
    return mask[None, None, :, :]


def _repeat_kv(x: mx.array, n_rep: int) -> mx.array:
    """Repeat KV heads to match Q heads for GQA.

    Args:
        x: (batch, kv_heads, seq, head_dim)
        n_rep: Number of repetitions per KV head.

    Returns:
        (batch, kv_heads * n_rep, seq, head_dim)
    """
    if n_rep == 1:
        return x
    batch, kv_heads, seq_len, head_dim = x.shape
    x = mx.expand_dims(x, axis=2)
    x = mx.broadcast_to(x, (batch, kv_heads, n_rep, seq_len, head_dim))
    return x.reshape(batch, kv_heads * n_rep, seq_len, head_dim)


class AceStepV15Attention(nn.Module):
    """GQA attention for ACE-Step v1.5 with Q/K RMSNorm and sliding window.

    Supports both self-attention and cross-attention modes.
    """

    def __init__(
        self,
        hidden_size: int,
        num_attention_heads: int,
        num_key_value_heads: int,
        head_dim: int,
        attention_bias: bool = False,
        rms_norm_eps: float = 1e-6,
        layer_idx: int = 0,
        is_cross_attention: bool = False,
        sliding_window: Optional[int] = None,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_attention_heads
        self.num_kv_heads = num_key_value_heads
        self.head_dim = head_dim
        self.num_kv_groups = num_attention_heads // num_key_value_heads
        self.scale = head_dim ** -0.5
        self.is_cross_attention = is_cross_attention
        self.sliding_window = sliding_window
        self.layer_idx = layer_idx

        self.q_proj = nn.Linear(hidden_size, num_attention_heads * head_dim, bias=attention_bias)
        self.k_proj = nn.Linear(hidden_size, num_key_value_heads * head_dim, bias=attention_bias)
        self.v_proj = nn.Linear(hidden_size, num_key_value_heads * head_dim, bias=attention_bias)
        self.o_proj = nn.Linear(num_attention_heads * head_dim, hidden_size, bias=attention_bias)

        self.q_norm = RMSNorm(head_dim, eps=rms_norm_eps)
        self.k_norm = RMSNorm(head_dim, eps=rms_norm_eps)

    def __call__(
        self,
        hidden_states: mx.array,
        attention_mask: Optional[mx.array] = None,
        encoder_hidden_states: Optional[mx.array] = None,
        position_embeddings: Optional[Tuple[mx.array, mx.array]] = None,
        kv_cache: Optional[Tuple[mx.array, mx.array]] = None,
    ) -> Tuple[mx.array, Optional[Tuple[mx.array, mx.array]]]:
        """
        Args:
            hidden_states: (batch, seq, hidden_size)
            attention_mask: Additive mask (1, 1, seq, seq) or None
            encoder_hidden_states: For cross-attention (batch, enc_seq, hidden_size)
            position_embeddings: (cos, sin) for RoPE
            kv_cache: Cached (key, value) for cross-attention reuse

        Returns:
            (output, updated_kv_cache) tuple
        """
        batch, seq_len, _ = hidden_states.shape

        # Project Q
        q = self.q_proj(hidden_states)
        q = q.reshape(batch, seq_len, self.num_heads, self.head_dim)
        q = self.q_norm(q)
        q = mx.transpose(q, axes=(0, 2, 1, 3))  # (B, H, T, D)

        is_cross = self.is_cross_attention and encoder_hidden_states is not None

        if is_cross:
            if kv_cache is not None:
                k, v = kv_cache
                new_cache = kv_cache
            else:
                enc_seq = encoder_hidden_states.shape[1]
                k = self.k_proj(encoder_hidden_states)
                k = k.reshape(batch, enc_seq, self.num_kv_heads, self.head_dim)
                k = self.k_norm(k)
                k = mx.transpose(k, axes=(0, 2, 1, 3))

                v = self.v_proj(encoder_hidden_states)
                v = v.reshape(batch, enc_seq, self.num_kv_heads, self.head_dim)
                v = mx.transpose(v, axes=(0, 2, 1, 3))
                new_cache = (k, v)
        else:
            k = self.k_proj(hidden_states)
            k = k.reshape(batch, seq_len, self.num_kv_heads, self.head_dim)
            k = self.k_norm(k)
            k = mx.transpose(k, axes=(0, 2, 1, 3))

            v = self.v_proj(hidden_states)
            v = v.reshape(batch, seq_len, self.num_kv_heads, self.head_dim)
            v = mx.transpose(v, axes=(0, 2, 1, 3))

            if position_embeddings is not None:
                cos, sin = position_embeddings
                q, k = apply_rotary_pos_emb(q, k, cos, sin)

            new_cache = None

        # Expand KV heads for GQA
        k = _repeat_kv(k, self.num_kv_groups)
        v = _repeat_kv(v, self.num_kv_groups)

        # Compute attention
        scores = mx.matmul(q, mx.transpose(k, axes=(0, 1, 3, 2))) * self.scale

        if attention_mask is not None:
            scores = scores + attention_mask

        weights = mx.softmax(scores.astype(mx.float32), axis=-1).astype(scores.dtype)
        out = mx.matmul(weights, v)

        # Reshape: (B, H, T, D) -> (B, T, H*D)
        out = mx.transpose(out, axes=(0, 2, 1, 3))
        out = out.reshape(batch, seq_len, -1)
        out = self.o_proj(out)

        return out, new_cache


class Qwen3MLP(nn.Module):
    """SiLU-gated MLP: down(silu(gate(x)) * up(x))."""

    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

    def __call__(self, x: mx.array) -> mx.array:
        return self.down_proj(nn.silu(self.gate_proj(x)) * self.up_proj(x))


__all__ = [
    "RMSNorm",
    "RotaryEmbedding",
    "apply_rotary_pos_emb",
    "create_sliding_mask",
    "AceStepV15Attention",
    "Qwen3MLP",
]
