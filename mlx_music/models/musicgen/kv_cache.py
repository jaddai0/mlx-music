"""
Class-based KV Cache for MusicGen.

Provides pre-allocated, efficient KV caching for autoregressive generation.
Based on mlx-audio's KVCache implementation but specialized for MusicGen's
decoder architecture with self-attention and cross-attention caches.

Key optimizations:
- Pre-allocated buffers to avoid repeated allocations
- Step-based growth for memory efficiency
- Separate caches for self-attention and cross-attention
- Cross-attention cache reuse (encoder K/V doesn't change during generation)
"""

from dataclasses import dataclass
from typing import Optional, Tuple

import mlx.core as mx


class KVCache:
    """
    KV Cache for a single attention layer.

    Uses pre-allocated buffers with step-based growth to minimize
    memory allocations during autoregressive generation.

    Cache stores tensors in pre-reshape format (batch, seq, embed_dim)
    to match MusicGen's attention implementation.
    """

    def __init__(
        self,
        embed_dim: int,
        step: int = 256,
    ):
        """
        Initialize KV cache.

        Args:
            embed_dim: Embedding dimension (hidden_size)
            step: Growth step size for buffer allocation
        """
        self.embed_dim = embed_dim
        self.step = step
        self.keys: Optional[mx.array] = None
        self.values: Optional[mx.array] = None
        self.offset = 0

    def update_and_fetch(
        self,
        keys: mx.array,
        values: mx.array,
    ) -> Tuple[mx.array, mx.array]:
        """
        Update cache with new keys/values and return full cache.

        Args:
            keys: New keys (batch, new_seq, embed_dim)
            values: New values (batch, new_seq, embed_dim)

        Returns:
            Tuple of (all_keys, all_values) including cached and new
        """
        batch_size = keys.shape[0]
        new_seq_len = keys.shape[1]
        prev_offset = self.offset

        # Check if we need to grow the buffer
        if self.keys is None or (prev_offset + new_seq_len) > self.keys.shape[1]:
            # Calculate new size with step-based growth
            n_steps = (self.step + new_seq_len - 1) // self.step
            new_alloc_size = n_steps * self.step

            new_k = mx.zeros((batch_size, new_alloc_size, self.embed_dim), keys.dtype)
            new_v = mx.zeros((batch_size, new_alloc_size, self.embed_dim), values.dtype)

            if self.keys is not None:
                # Copy existing data and concatenate new allocation
                if prev_offset % self.step != 0:
                    # Trim to actual used size before concatenating
                    self.keys = self.keys[:, :prev_offset, :]
                    self.values = self.values[:, :prev_offset, :]
                self.keys = mx.concatenate([self.keys, new_k], axis=1)
                self.values = mx.concatenate([self.values, new_v], axis=1)
            else:
                self.keys = new_k
                self.values = new_v

        # Write new keys/values into buffer
        self.offset += new_seq_len
        self.keys = self.keys.at[:, prev_offset:self.offset, :].add(keys - self.keys[:, prev_offset:self.offset, :])
        self.values = self.values.at[:, prev_offset:self.offset, :].add(values - self.values[:, prev_offset:self.offset, :])

        # Return only the valid portion
        return self.keys[:, :self.offset, :], self.values[:, :self.offset, :]

    def reset(self):
        """Reset cache state."""
        self.keys = None
        self.values = None
        self.offset = 0

    @property
    def state(self) -> Tuple[Optional[mx.array], Optional[mx.array]]:
        """Get current cache state as tuple."""
        if self.keys is None:
            return None, None
        return self.keys[:, :self.offset, :], self.values[:, :self.offset, :]

    @property
    def seq_len(self) -> int:
        """Current cached sequence length."""
        return self.offset


class CrossAttentionCache:
    """
    Cache for cross-attention keys and values.

    Unlike self-attention, cross-attention K/V are computed once from
    encoder hidden states and reused for all generation steps.
    """

    def __init__(self):
        self.keys: Optional[mx.array] = None
        self.values: Optional[mx.array] = None
        self._is_set = False

    def set(self, keys: mx.array, values: mx.array):
        """
        Store cross-attention K/V (computed from encoder states).

        Args:
            keys: (batch, enc_seq, embed_dim)
            values: (batch, enc_seq, embed_dim)
        """
        self.keys = keys
        self.values = values
        self._is_set = True

    def get(self) -> Tuple[Optional[mx.array], Optional[mx.array]]:
        """Get cached cross-attention K/V."""
        return self.keys, self.values

    @property
    def is_set(self) -> bool:
        """Check if cache has been populated."""
        return self._is_set

    def reset(self):
        """Reset cache state."""
        self.keys = None
        self.values = None
        self._is_set = False


@dataclass
class MusicGenCacheState:
    """
    Complete cache state for MusicGen decoder.

    Manages all caches for a generation session:
    - Self-attention caches (one per layer, grows during generation)
    - Cross-attention caches (one per layer, computed once from encoder)
    """

    num_layers: int
    embed_dim: int
    step: int = 256

    def __post_init__(self):
        # Self-attention caches (grow during generation)
        self.self_attn_caches = [
            KVCache(self.embed_dim, self.step)
            for _ in range(self.num_layers)
        ]
        # Cross-attention caches (set once from encoder)
        self.cross_attn_caches = [
            CrossAttentionCache()
            for _ in range(self.num_layers)
        ]

    def get_self_attn_cache(self, layer_idx: int) -> KVCache:
        """Get self-attention cache for a layer."""
        return self.self_attn_caches[layer_idx]

    def get_cross_attn_cache(self, layer_idx: int) -> CrossAttentionCache:
        """Get cross-attention cache for a layer."""
        return self.cross_attn_caches[layer_idx]

    def reset(self):
        """Reset all caches."""
        for cache in self.self_attn_caches:
            cache.reset()
        for cache in self.cross_attn_caches:
            cache.reset()

    @property
    def past_seq_len(self) -> int:
        """Current cached sequence length (from self-attention)."""
        if self.self_attn_caches:
            return self.self_attn_caches[0].seq_len
        return 0

    @property
    def cross_attn_ready(self) -> bool:
        """Check if cross-attention caches are populated."""
        return all(cache.is_set for cache in self.cross_attn_caches)

    def to_legacy_format(self) -> Tuple[list, list]:
        """
        Convert to legacy tuple-based format for compatibility.

        Returns:
            (past_key_values, cross_attn_past_key_values) as lists of tuples
        """
        past_key_values = []
        cross_attn_past_key_values = []

        for i in range(self.num_layers):
            # Self-attention: (keys, values) tuple
            k, v = self.self_attn_caches[i].state
            if k is not None:
                past_key_values.append((k, v))
            else:
                past_key_values.append(None)

            # Cross-attention: (keys, values) tuple
            k, v = self.cross_attn_caches[i].get()
            if k is not None:
                cross_attn_past_key_values.append((k, v))
            else:
                cross_attn_past_key_values.append(None)

        return past_key_values, cross_attn_past_key_values

    @classmethod
    def from_legacy_format(
        cls,
        past_key_values: Optional[list],
        cross_attn_past_key_values: Optional[list],
        num_layers: int,
        embed_dim: int,
    ) -> "MusicGenCacheState":
        """
        Create cache state from legacy tuple-based format.

        Args:
            past_key_values: List of (k, v) tuples for self-attention
            cross_attn_past_key_values: List of (k, v) tuples for cross-attention
            num_layers: Number of decoder layers
            embed_dim: Embedding dimension

        Returns:
            MusicGenCacheState instance
        """
        state = cls(num_layers=num_layers, embed_dim=embed_dim)

        if past_key_values:
            for i, kv in enumerate(past_key_values):
                if kv is not None:
                    k, v = kv
                    state.self_attn_caches[i].keys = k
                    state.self_attn_caches[i].values = v
                    state.self_attn_caches[i].offset = k.shape[1]

        if cross_attn_past_key_values:
            for i, kv in enumerate(cross_attn_past_key_values):
                if kv is not None:
                    k, v = kv
                    state.cross_attn_caches[i].set(k, v)

        return state


def create_causal_mask(seq_len: int, past_len: int = 0) -> mx.array:
    """
    Create causal attention mask.

    Args:
        seq_len: Current sequence length (query positions)
        past_len: Cached sequence length (past key positions)

    Returns:
        Causal mask of shape (1, 1, seq_len, total_len)
    """
    total_len = past_len + seq_len

    # Create position indices
    row_indices = mx.arange(seq_len)[:, None]  # (seq_len, 1)
    col_indices = mx.arange(total_len)[None, :]  # (1, total_len)

    # Each position i can attend to positions 0 to past_len + i (inclusive)
    causal_mask = mx.where(
        col_indices <= (past_len + row_indices),
        0.0,
        float("-inf"),
    )

    return causal_mask[None, None, :, :]  # (1, 1, seq_len, total_len)


__all__ = [
    "KVCache",
    "CrossAttentionCache",
    "MusicGenCacheState",
    "create_causal_mask",
]
