"""
DiT (Diffusion Transformer) layers for ACE-Step v1.5.

Implements:
- TimestepEmbedding: sinusoidal + MLP + 6-way projection for AdaLN
- AceStepEncoderLayer: bidirectional self-attn + MLP (shared by lyric/timbre/pooler)
- AceStepDiTLayer: self-attn(AdaLN) + cross-attn + MLP(AdaLN)
"""

import math
from typing import Dict, Optional, Tuple

import mlx.core as mx
import mlx.nn as nn

from mlx_music.models.ace_step_v15.attention import (
    AceStepV15Attention,
    Qwen3MLP,
    RMSNorm,
)


class TimestepEmbedding(nn.Module):
    """Sinusoidal timestep embedding with MLP and 6-way AdaLN projection.

    Converts scalar timestep t into:
    - temb: (batch, hidden_size) embedding
    - timestep_proj: (batch, 6, hidden_size) AdaLN parameters
    """

    def __init__(self, in_channels: int, time_embed_dim: int, scale: float = 1000.0):
        super().__init__()
        self.in_channels = in_channels
        self.scale = scale

        self.linear_1 = nn.Linear(in_channels, time_embed_dim, bias=True)
        self.linear_2 = nn.Linear(time_embed_dim, time_embed_dim, bias=True)
        self.time_proj = nn.Linear(time_embed_dim, time_embed_dim * 6, bias=True)

    def _timestep_embedding(self, t: mx.array, dim: int, max_period: float = 10000.0) -> mx.array:
        """Create sinusoidal timestep embeddings.

        Args:
            t: (batch,) timestep values
            dim: Embedding dimension
            max_period: Controls minimum frequency

        Returns:
            (batch, dim) sinusoidal embeddings
        """
        t = t * self.scale
        half = dim // 2
        freqs = mx.exp(
            -math.log(max_period) * mx.arange(0, half, dtype=mx.float32) / half
        )
        args = t[:, None].astype(mx.float32) * freqs[None, :]
        embedding = mx.concatenate([mx.cos(args), mx.sin(args)], axis=-1)
        if dim % 2:
            embedding = mx.concatenate([embedding, mx.zeros_like(embedding[:, :1])], axis=-1)
        return embedding

    def __call__(self, t: mx.array) -> Tuple[mx.array, mx.array]:
        """
        Args:
            t: (batch,) timestep values in [0, 1]

        Returns:
            (temb, timestep_proj) where:
            - temb: (batch, time_embed_dim)
            - timestep_proj: (batch, 6, time_embed_dim)
        """
        t_freq = self._timestep_embedding(t, self.in_channels)
        temb = self.linear_1(t_freq.astype(t.dtype))
        temb = nn.silu(temb)
        temb = self.linear_2(temb)

        proj = self.time_proj(nn.silu(temb))
        # Reshape to (batch, 6, hidden_size)
        timestep_proj = proj.reshape(proj.shape[0], 6, -1)
        return temb, timestep_proj


class AceStepEncoderLayer(nn.Module):
    """Bidirectional encoder layer: self-attn + MLP with residual connections.

    Shared by lyric encoder, timbre encoder, attention pooler, and detokenizer.
    No causal mask, no AdaLN - plain pre-norm transformer.
    """

    def __init__(
        self,
        hidden_size: int,
        num_attention_heads: int,
        num_key_value_heads: int,
        head_dim: int,
        intermediate_size: int,
        attention_bias: bool = False,
        rms_norm_eps: float = 1e-6,
        layer_idx: int = 0,
        sliding_window: Optional[int] = None,
        layer_type: str = "full_attention",
    ):
        super().__init__()
        self.hidden_size = hidden_size

        is_sliding = layer_type == "sliding_attention"
        self.self_attn = AceStepV15Attention(
            hidden_size=hidden_size,
            num_attention_heads=num_attention_heads,
            num_key_value_heads=num_key_value_heads,
            head_dim=head_dim,
            attention_bias=attention_bias,
            rms_norm_eps=rms_norm_eps,
            layer_idx=layer_idx,
            is_cross_attention=False,
            sliding_window=sliding_window if is_sliding else None,
        )
        self.input_layernorm = RMSNorm(hidden_size, eps=rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(hidden_size, eps=rms_norm_eps)
        self.mlp = Qwen3MLP(hidden_size, intermediate_size)

    def __call__(
        self,
        hidden_states: mx.array,
        position_embeddings: Tuple[mx.array, mx.array],
        attention_mask: Optional[mx.array] = None,
    ) -> mx.array:
        """
        Args:
            hidden_states: (batch, seq, hidden_size)
            position_embeddings: (cos, sin) for RoPE
            attention_mask: Additive mask or None

        Returns:
            (batch, seq, hidden_size)
        """
        # Self-attention with residual
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states, _ = self.self_attn(
            hidden_states=hidden_states,
            position_embeddings=position_embeddings,
            attention_mask=attention_mask,
        )
        hidden_states = residual + hidden_states

        # MLP with residual
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states


class AceStepDiTLayer(nn.Module):
    """DiT layer with AdaLN conditioning.

    Three sub-layers:
    1. Self-attention with AdaLN: norm(x) * (1 + scale) + shift, gated residual
    2. Cross-attention to encoder hidden states (standard residual)
    3. MLP with AdaLN: same modulation pattern as self-attention
    """

    def __init__(
        self,
        hidden_size: int,
        num_attention_heads: int,
        num_key_value_heads: int,
        head_dim: int,
        intermediate_size: int,
        attention_bias: bool = False,
        rms_norm_eps: float = 1e-6,
        layer_idx: int = 0,
        sliding_window: Optional[int] = None,
        layer_type: str = "full_attention",
        use_cross_attention: bool = True,
    ):
        super().__init__()

        is_sliding = layer_type == "sliding_attention"

        # Self-attention with AdaLN
        self.self_attn_norm = RMSNorm(hidden_size, eps=rms_norm_eps)
        self.self_attn = AceStepV15Attention(
            hidden_size=hidden_size,
            num_attention_heads=num_attention_heads,
            num_key_value_heads=num_key_value_heads,
            head_dim=head_dim,
            attention_bias=attention_bias,
            rms_norm_eps=rms_norm_eps,
            layer_idx=layer_idx,
            is_cross_attention=False,
            sliding_window=sliding_window if is_sliding else None,
        )

        # Cross-attention (optional)
        self.use_cross_attention = use_cross_attention
        if use_cross_attention:
            self.cross_attn_norm = RMSNorm(hidden_size, eps=rms_norm_eps)
            self.cross_attn = AceStepV15Attention(
                hidden_size=hidden_size,
                num_attention_heads=num_attention_heads,
                num_key_value_heads=num_key_value_heads,
                head_dim=head_dim,
                attention_bias=attention_bias,
                rms_norm_eps=rms_norm_eps,
                layer_idx=layer_idx,
                is_cross_attention=True,
            )

        # MLP with AdaLN
        self.mlp_norm = RMSNorm(hidden_size, eps=rms_norm_eps)
        self.mlp = Qwen3MLP(hidden_size, intermediate_size)

        # Scale-shift table for AdaLN: 6 params (shift, scale, gate) x2 for self-attn and MLP
        self.scale_shift_table = mx.zeros((1, 6, hidden_size))

    def __call__(
        self,
        hidden_states: mx.array,
        position_embeddings: Tuple[mx.array, mx.array],
        temb: mx.array,
        attention_mask: Optional[mx.array] = None,
        encoder_hidden_states: Optional[mx.array] = None,
        encoder_attention_mask: Optional[mx.array] = None,
        cross_kv_cache: Optional[Dict[int, Tuple[mx.array, mx.array]]] = None,
        layer_idx: int = 0,
    ) -> Tuple[mx.array, Optional[Dict[int, Tuple[mx.array, mx.array]]]]:
        """
        Args:
            hidden_states: (batch, seq, hidden_size)
            position_embeddings: (cos, sin) for RoPE
            temb: (batch, 6, hidden_size) timestep AdaLN parameters
            attention_mask: Additive mask for self-attention
            encoder_hidden_states: (batch, enc_seq, hidden_size) for cross-attn
            encoder_attention_mask: Additive mask for cross-attention
            cross_kv_cache: Dict mapping layer_idx -> (k, v) cached tensors
            layer_idx: Layer index for cache key

        Returns:
            (hidden_states, updated_cross_kv_cache)
        """
        # Extract AdaLN parameters: (shift, scale, gate) for self-attn and MLP
        combined = self.scale_shift_table + temb
        shift_msa = combined[:, 0:1, :]    # (B, 1, D)
        scale_msa = combined[:, 1:2, :]
        gate_msa = combined[:, 2:3, :]
        shift_mlp = combined[:, 3:4, :]
        scale_mlp = combined[:, 4:5, :]
        gate_mlp = combined[:, 5:6, :]

        # 1. Self-attention with AdaLN
        norm_hidden = self.self_attn_norm(hidden_states)
        norm_hidden = norm_hidden * (1 + scale_msa) + shift_msa
        attn_out, _ = self.self_attn(
            hidden_states=norm_hidden,
            position_embeddings=position_embeddings,
            attention_mask=attention_mask,
        )
        hidden_states = hidden_states + attn_out * gate_msa

        # 2. Cross-attention
        updated_cache = cross_kv_cache
        if self.use_cross_attention and encoder_hidden_states is not None:
            norm_hidden = self.cross_attn_norm(hidden_states)

            # Check for cached KV
            cached_kv = None
            if cross_kv_cache is not None and layer_idx in cross_kv_cache:
                cached_kv = cross_kv_cache[layer_idx]

            cross_out, new_kv = self.cross_attn(
                hidden_states=norm_hidden,
                encoder_hidden_states=encoder_hidden_states,
                attention_mask=encoder_attention_mask,
                kv_cache=cached_kv,
            )
            hidden_states = hidden_states + cross_out

            # Update cache
            if new_kv is not None:
                if updated_cache is None:
                    updated_cache = {}
                updated_cache[layer_idx] = new_kv

        # 3. MLP with AdaLN
        norm_hidden = self.mlp_norm(hidden_states)
        norm_hidden = norm_hidden * (1 + scale_mlp) + shift_mlp
        ff_out = self.mlp(norm_hidden)
        hidden_states = hidden_states + ff_out * gate_mlp

        return hidden_states, updated_cache


__all__ = [
    "TimestepEmbedding",
    "AceStepEncoderLayer",
    "AceStepDiTLayer",
]
