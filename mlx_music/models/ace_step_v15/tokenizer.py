"""
Audio tokenization for ACE-Step v1.5.

Implements:
- ResidualFSQ: Finite Scalar Quantization (levels [8,8,8,5,5,5])
- AttentionPooler: CLS-based attention pooling (2 encoder layers)
- AceStepAudioTokenizer: project -> reshape -> pool -> quantize
- AudioTokenDetokenizer: expand 5Hz->25Hz + encoder layers + project
"""

import math
from typing import List, Optional, Tuple

import mlx.core as mx
import mlx.nn as nn

from mlx_music.models.ace_step_v15.attention import (
    RMSNorm,
    RotaryEmbedding,
    create_sliding_mask,
)
from mlx_music.models.ace_step_v15.config import AceStepV15Config
from mlx_music.models.ace_step_v15.dit import AceStepEncoderLayer


class ResidualFSQ(nn.Module):
    """Finite Scalar Quantization with residual quantizers.

    Quantizes continuous vectors into discrete codes using per-dimension rounding
    to predefined levels. Each dimension has its own codebook size.

    levels=[8,8,8,5,5,5] gives 8*8*8*5*5*5 = 64000 total codes.
    """

    def __init__(
        self,
        dim: int,
        levels: List[int],
        num_quantizers: int = 1,
    ):
        super().__init__()
        self.dim = dim
        self.levels = levels
        self.num_quantizers = num_quantizers
        self.codebook_dim = len(levels)

        # Project from dim to codebook_dim and back
        self.project_in = nn.Linear(dim, self.codebook_dim * num_quantizers, bias=True)
        self.project_out = nn.Linear(self.codebook_dim * num_quantizers, dim, bias=True)

        # Store levels as array for quantization
        self._levels = mx.array(levels, dtype=mx.float32)
        self._half_levels = (self._levels - 1) / 2

    def _quantize(self, z: mx.array) -> Tuple[mx.array, mx.array]:
        """Quantize continuous values to discrete levels.

        Args:
            z: (..., codebook_dim) continuous values

        Returns:
            (quantized, indices) tuple
        """
        # Scale to [-1, 1] then to [0, levels-1]
        half = self._half_levels
        z_scaled = mx.tanh(z) * half + half

        # Round to nearest integer (straight-through for gradients)
        indices = mx.round(z_scaled).astype(mx.int32)
        indices = mx.clip(indices, 0, self._levels.astype(mx.int32) - 1)

        # Convert back to continuous
        quantized = (indices.astype(mx.float32) - half) / half

        return quantized, indices

    def _indices_to_codes(self, indices: mx.array) -> mx.array:
        """Convert indices back to continuous codes.

        Args:
            indices: (..., codebook_dim) integer indices

        Returns:
            (..., codebook_dim) continuous values in [-1, 1]
        """
        half = self._half_levels
        return (indices.astype(mx.float32) - half) / half

    def get_output_from_indices(self, indices: mx.array) -> mx.array:
        """Reconstruct output from quantization indices.

        Args:
            indices: (..., num_quantizers, codebook_dim) or (..., codebook_dim)

        Returns:
            (..., dim) reconstructed continuous values
        """
        if indices.ndim > 2 and indices.shape[-2] == self.num_quantizers:
            # Multi-quantizer: reshape
            codes = self._indices_to_codes(indices)
            codes = codes.reshape(*codes.shape[:-2], -1)
        else:
            codes = self._indices_to_codes(indices)
        return self.project_out(codes)

    def __call__(self, x: mx.array) -> Tuple[mx.array, mx.array]:
        """Quantize input.

        Args:
            x: (..., dim) continuous input

        Returns:
            (quantized_output, indices):
            - quantized_output: (..., dim) projected back to original dim
            - indices: (..., num_quantizers, codebook_dim)
        """
        z = self.project_in(x)

        if self.num_quantizers > 1:
            z = z.reshape(*z.shape[:-1], self.num_quantizers, self.codebook_dim)
            quantized, indices = self._quantize(z)
            quantized_flat = quantized.reshape(*quantized.shape[:-2], -1)
        else:
            quantized, indices = self._quantize(z)
            indices = mx.expand_dims(indices, axis=-2)
            quantized_flat = quantized

        output = self.project_out(quantized_flat)
        return output, indices


class AttentionPooler(nn.Module):
    """CLS-token based attention pooling.

    Prepends a learnable CLS token to patch sequences, processes through
    encoder layers, then extracts the CLS output as the pooled representation.

    Input: (B, T, P, D) -> Output: (B, T, D)
    """

    def __init__(self, config: AceStepV15Config):
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Linear(config.hidden_size, config.hidden_size)
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = RotaryEmbedding(
            config.head_dim, config.max_position_embeddings, config.rope_theta
        )
        self.special_token = mx.random.normal((1, 1, config.hidden_size)) * 0.02
        self.layers = [
            AceStepEncoderLayer(
                hidden_size=config.hidden_size,
                num_attention_heads=config.num_attention_heads,
                num_key_value_heads=config.num_key_value_heads,
                head_dim=config.head_dim,
                intermediate_size=config.intermediate_size,
                attention_bias=config.attention_bias,
                rms_norm_eps=config.rms_norm_eps,
                layer_idx=i,
                sliding_window=config.sliding_window,
                layer_type=config.layer_types[i],
            )
            for i in range(config.num_attention_pooler_hidden_layers)
        ]

    def __call__(self, x: mx.array) -> mx.array:
        """
        Args:
            x: (B, T, P, D) patch features

        Returns:
            (B, T, D) pooled features
        """
        B, T, P, D = x.shape
        x = self.embed_tokens(x)

        # Prepend CLS token to each patch sequence
        cls_tokens = mx.broadcast_to(self.special_token, (B, T, 1, D))
        x = mx.concatenate([cls_tokens, x], axis=2)  # (B, T, P+1, D)

        # Reshape for processing: (B*T, P+1, D)
        x = x.reshape(B * T, P + 1, D)

        seq_len = x.shape[1]
        cos, sin = self.rotary_emb(seq_len, dtype=x.dtype)

        hidden_states = x
        for layer in self.layers:
            hidden_states = layer(hidden_states, position_embeddings=(cos, sin))

        hidden_states = self.norm(hidden_states)

        # Extract CLS token (first position)
        cls_output = hidden_states[:, 0, :]  # (B*T, D)
        cls_output = cls_output.reshape(B, T, D)
        return cls_output


class AudioTokenDetokenizer(nn.Module):
    """Converts quantized 5Hz tokens back to 25Hz acoustic features.

    Expands each token into pool_window_size patches using learnable special tokens,
    processes through encoder layers, then projects to acoustic dimension.

    Input: (B, T, D) -> Output: (B, T*pool_window_size, audio_acoustic_hidden_dim)
    """

    def __init__(self, config: AceStepV15Config):
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Linear(config.hidden_size, config.hidden_size)
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = RotaryEmbedding(
            config.head_dim, config.max_position_embeddings, config.rope_theta
        )
        # Learnable tokens for expanding each quantized token into patches
        self.special_tokens = mx.random.normal(
            (1, config.pool_window_size, config.hidden_size)
        ) * 0.02
        self.layers = [
            AceStepEncoderLayer(
                hidden_size=config.hidden_size,
                num_attention_heads=config.num_attention_heads,
                num_key_value_heads=config.num_key_value_heads,
                head_dim=config.head_dim,
                intermediate_size=config.intermediate_size,
                attention_bias=config.attention_bias,
                rms_norm_eps=config.rms_norm_eps,
                layer_idx=i,
                sliding_window=config.sliding_window,
                layer_type=config.layer_types[i],
            )
            for i in range(config.num_attention_pooler_hidden_layers)
        ]
        self.proj_out = nn.Linear(config.hidden_size, config.audio_acoustic_hidden_dim)

    def __call__(self, x: mx.array) -> mx.array:
        """
        Args:
            x: (B, T, D) quantized token representations

        Returns:
            (B, T*pool_window_size, audio_acoustic_hidden_dim)
        """
        B, T, D = x.shape
        P = self.config.pool_window_size
        x = self.embed_tokens(x)

        # Expand: (B, T, D) -> (B, T, P, D) by repeating + adding special tokens
        x = mx.expand_dims(x, axis=2)  # (B, T, 1, D)
        x = mx.broadcast_to(x, (B, T, P, D))
        special = mx.broadcast_to(self.special_tokens, (B, T, P, D))
        x = x + special

        # Reshape for processing: (B*T, P, D)
        x = x.reshape(B * T, P, D)

        seq_len = x.shape[1]
        cos, sin = self.rotary_emb(seq_len, dtype=x.dtype)

        hidden_states = x
        for layer in self.layers:
            hidden_states = layer(hidden_states, position_embeddings=(cos, sin))

        hidden_states = self.norm(hidden_states)
        hidden_states = self.proj_out(hidden_states)

        # Reshape: (B*T, P, C) -> (B, T*P, C)
        hidden_states = hidden_states.reshape(B, T * P, -1)
        return hidden_states


class AceStepAudioTokenizer(nn.Module):
    """Audio tokenizer: project -> reshape -> pool -> quantize.

    Input: (B, T, audio_acoustic_hidden_dim=64) -> Output: (quantized, indices)
    """

    def __init__(self, config: AceStepV15Config):
        super().__init__()
        self.config = config
        self.pool_window_size = config.pool_window_size
        self.audio_acoustic_proj = nn.Linear(
            config.audio_acoustic_hidden_dim, config.hidden_size
        )
        self.attention_pooler = AttentionPooler(config)
        self.quantizer = ResidualFSQ(
            dim=config.fsq_dim,
            levels=config.fsq_input_levels,
            num_quantizers=config.fsq_input_num_quantizers,
        )

    def __call__(
        self, hidden_states: mx.array
    ) -> Tuple[mx.array, mx.array]:
        """
        Args:
            hidden_states: (B, T_patches, pool_window_size, D) patch features

        Returns:
            (quantized, indices) from FSQ
        """
        hidden_states = self.audio_acoustic_proj(hidden_states)
        hidden_states = self.attention_pooler(hidden_states)
        quantized, indices = self.quantizer(hidden_states)
        return quantized, indices

    def tokenize(self, x: mx.array) -> Tuple[mx.array, mx.array]:
        """Tokenize continuous audio features.

        Args:
            x: (B, T, D) where T is divisible by pool_window_size

        Returns:
            (quantized, indices)
        """
        P = self.pool_window_size
        B, T, D = x.shape
        x = x.reshape(B, T // P, P, D)
        return self(x)


__all__ = [
    "ResidualFSQ",
    "AttentionPooler",
    "AudioTokenDetokenizer",
    "AceStepAudioTokenizer",
]
