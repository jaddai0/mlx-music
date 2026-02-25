"""
Conditioning encoders for ACE-Step v1.5.

Implements:
- AceStepLyricEncoder: 8-layer bidirectional encoder for lyrics
- AceStepTimbreEncoder: 4-layer encoder with CLS aggregation for timbre
- AceStepConditionEncoder: Combines text + lyrics + timbre via pack_sequences
- pack_sequences: Sort-based packing of variable-length sequences
"""

from typing import Dict, List, Optional, Tuple

import mlx.core as mx
import mlx.nn as nn

from mlx_music.models.ace_step_v15.attention import (
    RMSNorm,
    RotaryEmbedding,
    create_sliding_mask,
)
from mlx_music.models.ace_step_v15.config import AceStepV15Config
from mlx_music.models.ace_step_v15.dit import AceStepEncoderLayer


def pack_sequences(
    hidden1: mx.array,
    hidden2: mx.array,
    mask1: mx.array,
    mask2: mx.array,
) -> Tuple[mx.array, mx.array]:
    """Pack two sequences by sorting valid tokens first.

    Args:
        hidden1: (B, L1, D)
        hidden2: (B, L2, D)
        mask1: (B, L1) - 1 for valid, 0 for padding
        mask2: (B, L2) - 1 for valid, 0 for padding

    Returns:
        (packed_hidden, new_mask) where valid tokens come first.
    """
    hidden_cat = mx.concatenate([hidden1, hidden2], axis=1)  # (B, L, D)
    mask_cat = mx.concatenate([mask1, mask2], axis=1)  # (B, L)

    B, L, D = hidden_cat.shape

    # Sort so mask=1 comes before mask=0 (descending sort, stable)
    sort_idx = mx.argsort(-mask_cat, axis=1)  # descending: negate then argsort

    # Gather hidden states according to sorted indices
    sort_idx_expanded = mx.expand_dims(sort_idx, axis=-1)
    sort_idx_expanded = mx.broadcast_to(sort_idx_expanded, (B, L, D))
    hidden_packed = mx.take_along_axis(hidden_cat, sort_idx_expanded, axis=1)

    # Create new mask based on valid counts
    lengths = mask_cat.sum(axis=1)  # (B,)
    positions = mx.arange(L)[None, :]  # (1, L)
    new_mask = (positions < lengths[:, None]).astype(mx.int32)

    return hidden_packed, new_mask


class _EncoderBase(nn.Module):
    """Shared base for lyric/timbre/pooler/detokenizer encoders.

    All share the same pattern: project -> RoPE + encoder layers -> norm.
    """

    def _build_mask_mapping(self, seq_len: int, config: AceStepV15Config, dtype: mx.Dtype):
        """Build full and sliding attention masks."""
        full_mask = None  # Full bidirectional needs no mask
        sliding_mask = None
        if config.use_sliding_window:
            sliding_mask = create_sliding_mask(
                seq_len, config.sliding_window, is_sliding=True, dtype=dtype
            )
        return {"full_attention": full_mask, "sliding_attention": sliding_mask}


class AceStepLyricEncoder(_EncoderBase):
    """8-layer bidirectional encoder for lyric text embeddings.

    Input: (B, T, text_hidden_dim=1024) -> Output: (B, T, hidden_size=2048)
    """

    def __init__(self, config: AceStepV15Config):
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Linear(config.text_hidden_dim, config.hidden_size)
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = RotaryEmbedding(
            config.head_dim, config.max_position_embeddings, config.rope_theta
        )
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
            for i in range(config.num_lyric_encoder_hidden_layers)
        ]

    def __call__(
        self,
        inputs_embeds: mx.array,
        attention_mask: Optional[mx.array] = None,
    ) -> mx.array:
        """
        Args:
            inputs_embeds: (B, T, text_hidden_dim)
            attention_mask: (B, T) - 1 for valid, 0 for padding

        Returns:
            (B, T, hidden_size)
        """
        hidden_states = self.embed_tokens(inputs_embeds)
        seq_len = hidden_states.shape[1]
        cos, sin = self.rotary_emb(seq_len, dtype=hidden_states.dtype)
        mask_mapping = self._build_mask_mapping(seq_len, self.config, hidden_states.dtype)

        for layer in self.layers:
            attn_type = self.config.layer_types[layer.self_attn.layer_idx]
            hidden_states = layer(
                hidden_states,
                position_embeddings=(cos, sin),
                attention_mask=mask_mapping.get(attn_type),
            )

        hidden_states = self.norm(hidden_states)
        return hidden_states


class AceStepTimbreEncoder(_EncoderBase):
    """4-layer encoder for timbre extraction from reference audio.

    Processes packed reference audio, extracts CLS token per sample,
    then unpacks back to batch format.

    Input: packed (N, T, timbre_hidden_dim=64) -> Output: (B, max_count, hidden_size)
    """

    def __init__(self, config: AceStepV15Config):
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Linear(config.timbre_hidden_dim, config.hidden_size)
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = RotaryEmbedding(
            config.head_dim, config.max_position_embeddings, config.rope_theta
        )
        self.special_token = mx.random.normal((1, 1, config.hidden_size))
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
            for i in range(config.num_timbre_encoder_hidden_layers)
        ]

    def _unpack_timbre_embeddings(
        self,
        timbre_embs_packed: mx.array,
        refer_audio_order_mask: mx.array,
    ) -> Tuple[mx.array, mx.array]:
        """Unpack packed timbre embeddings into batch format.

        Args:
            timbre_embs_packed: (N, D) packed embeddings
            refer_audio_order_mask: (N,) batch index for each packed item

        Returns:
            (unpacked_embeddings, mask) where:
            - unpacked: (B, max_count, D)
            - mask: (B, max_count)
        """
        N, D = timbre_embs_packed.shape
        B = int(refer_audio_order_mask.max().item()) + 1

        # Count items per batch
        counts = mx.zeros((B,), dtype=mx.int32)
        for i in range(N):
            batch_idx = int(refer_audio_order_mask[i].item())
            counts = counts.at[batch_idx].add(1)
        max_count = int(counts.max().item())

        # Build output via scatter
        result = mx.zeros((B, max_count, D), dtype=timbre_embs_packed.dtype)
        mask = mx.zeros((B, max_count), dtype=mx.int32)
        pos_counts = mx.zeros((B,), dtype=mx.int32)

        for i in range(N):
            batch_idx = int(refer_audio_order_mask[i].item())
            pos = int(pos_counts[batch_idx].item())
            result = result.at[batch_idx, pos].add(timbre_embs_packed[i])
            mask = mask.at[batch_idx, pos].add(1)
            pos_counts = pos_counts.at[batch_idx].add(1)

        return result, mask

    def __call__(
        self,
        refer_audio_packed: mx.array,
        refer_audio_order_mask: mx.array,
        attention_mask: Optional[mx.array] = None,
    ) -> Tuple[mx.array, mx.array]:
        """
        Args:
            refer_audio_packed: (N, T, timbre_hidden_dim) packed reference audio
            refer_audio_order_mask: (N,) batch assignment

        Returns:
            (timbre_embeddings, timbre_mask):
            - timbre_embeddings: (B, max_count, hidden_size)
            - timbre_mask: (B, max_count)
        """
        hidden_states = self.embed_tokens(refer_audio_packed)
        seq_len = hidden_states.shape[1]
        cos, sin = self.rotary_emb(seq_len, dtype=hidden_states.dtype)
        mask_mapping = self._build_mask_mapping(seq_len, self.config, hidden_states.dtype)

        for layer in self.layers:
            attn_type = self.config.layer_types[layer.self_attn.layer_idx]
            hidden_states = layer(
                hidden_states,
                position_embeddings=(cos, sin),
                attention_mask=mask_mapping.get(attn_type),
            )

        hidden_states = self.norm(hidden_states)
        # Extract first position (CLS) as timbre embedding
        cls_output = hidden_states[:, 0, :]  # (N, D)

        # Unpack to batch format
        timbre_embs, timbre_mask = self._unpack_timbre_embeddings(
            cls_output, refer_audio_order_mask
        )
        return timbre_embs, timbre_mask


class AceStepConditionEncoder(nn.Module):
    """Combines text + lyrics + timbre into packed encoder hidden states.

    Packs: lyrics + timbre -> pack -> + text -> pack
    """

    def __init__(self, config: AceStepV15Config):
        super().__init__()
        self.config = config
        self.text_projector = nn.Linear(config.text_hidden_dim, config.hidden_size, bias=False)
        self.lyric_encoder = AceStepLyricEncoder(config)
        self.timbre_encoder = AceStepTimbreEncoder(config)

    def __call__(
        self,
        text_hidden_states: mx.array,
        text_attention_mask: mx.array,
        lyric_hidden_states: mx.array,
        lyric_attention_mask: mx.array,
        refer_audio_packed: mx.array,
        refer_audio_order_mask: mx.array,
    ) -> Tuple[mx.array, mx.array]:
        """
        Returns:
            (encoder_hidden_states, encoder_attention_mask)
        """
        # Project text
        text_hidden_states = self.text_projector(text_hidden_states)

        # Encode lyrics
        lyric_hidden_states = self.lyric_encoder(
            inputs_embeds=lyric_hidden_states,
            attention_mask=lyric_attention_mask,
        )

        # Encode timbre
        timbre_embs, timbre_mask = self.timbre_encoder(
            refer_audio_packed, refer_audio_order_mask
        )

        # Pack: lyrics + timbre, then + text
        enc_hidden, enc_mask = pack_sequences(
            lyric_hidden_states, timbre_embs, lyric_attention_mask, timbre_mask
        )
        enc_hidden, enc_mask = pack_sequences(
            enc_hidden, text_hidden_states, enc_mask, text_attention_mask
        )

        return enc_hidden, enc_mask


__all__ = [
    "pack_sequences",
    "AceStepLyricEncoder",
    "AceStepTimbreEncoder",
    "AceStepConditionEncoder",
]
