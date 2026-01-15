"""
MusicGen Decoder Transformer.

MLX implementation of the MusicGen language model decoder.
"""

import math
from typing import Optional, Tuple

import mlx.core as mx
import mlx.nn as nn

from .config import MusicGenDecoderConfig


class MusicGenSinusoidalPositionEmbedding(nn.Module):
    """Sinusoidal position embeddings (learned positions stored as weights)."""

    def __init__(self, num_positions: int, embedding_dim: int):
        super().__init__()
        self.num_positions = num_positions
        self.embedding_dim = embedding_dim
        # Note: HF stores these as "weights" parameter
        self.weights = mx.zeros((num_positions, embedding_dim))

    def __call__(self, position_ids: mx.array) -> mx.array:
        """Get position embeddings for given position IDs."""
        return self.weights[position_ids]


class MusicGenAttention(nn.Module):
    """Multi-head attention for MusicGen decoder."""

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        is_cross_attention: bool = False,
        bias: bool = False,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.is_cross_attention = is_cross_attention
        self.scale = self.head_dim**-0.5

        if embed_dim % num_heads != 0:
            raise ValueError(
                f"embed_dim ({embed_dim}) must be divisible by num_heads ({num_heads})"
            )

        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

    def __call__(
        self,
        hidden_states: mx.array,
        key_value_states: Optional[mx.array] = None,
        attention_mask: Optional[mx.array] = None,
        past_key_value: Optional[Tuple[mx.array, mx.array]] = None,
        cache: Optional[Tuple[mx.array, mx.array]] = None,
    ) -> Tuple[mx.array, Optional[Tuple[mx.array, mx.array]]]:
        """
        Forward pass for attention.

        Args:
            hidden_states: (batch, seq_len, embed_dim)
            key_value_states: For cross-attention, encoder hidden states
            attention_mask: Attention mask
            past_key_value: Cached key/value for incremental decoding
            cache: Alternative cache input (for cross-attention, can be precomputed K/V)

        Returns:
            output: (batch, seq_len, embed_dim)
            present_key_value: Updated cache
        """
        batch_size, seq_len, _ = hidden_states.shape

        # Use cache or past_key_value if provided
        kv_cache = cache if cache is not None else past_key_value

        # Query projection
        queries = self.q_proj(hidden_states)

        # Key/Value projection
        # IMPORTANT: Cache is stored in PRE-RESHAPE format (batch, seq, embed_dim)
        # This allows concatenation along axis=1 for incremental decoding
        if self.is_cross_attention:
            # Cross-attention: use cache if available, else compute from encoder states
            if kv_cache is not None:
                # Reuse cached K/V from encoder (huge performance win)
                # Cache is in pre-reshape format (batch, enc_seq, embed_dim)
                keys, values = kv_cache
            elif key_value_states is not None:
                # First call: compute K/V from encoder states
                keys = self.k_proj(key_value_states)
                values = self.v_proj(key_value_states)
            else:
                raise ValueError("Cross-attention requires either key_value_states or cache")
        elif kv_cache is not None:
            # Self-attention with cache (incremental decoding)
            # Cache is in pre-reshape format (batch, cached_seq, embed_dim)
            keys = self.k_proj(hidden_states)
            values = self.v_proj(hidden_states)
            # Concatenate with cached keys/values along sequence dimension
            keys = mx.concatenate([kv_cache[0], keys], axis=1)
            values = mx.concatenate([kv_cache[1], values], axis=1)
        else:
            # Regular self-attention (no cache)
            keys = self.k_proj(hidden_states)
            values = self.v_proj(hidden_states)

        # Store cache in PRE-RESHAPE format (batch, seq, embed_dim)
        # This is done BEFORE reshape so concatenation works correctly
        present_key_value = (keys, values)

        # Reshape for multi-head attention
        # (batch, seq, embed_dim) -> (batch, seq, heads, head_dim) -> (batch, heads, seq, head_dim)
        queries = queries.reshape(batch_size, -1, self.num_heads, self.head_dim)
        queries = queries.transpose(0, 2, 1, 3)

        keys = keys.reshape(batch_size, -1, self.num_heads, self.head_dim)
        keys = keys.transpose(0, 2, 1, 3)

        values = values.reshape(batch_size, -1, self.num_heads, self.head_dim)
        values = values.transpose(0, 2, 1, 3)

        # Scaled dot-product attention
        attn_weights = (queries @ keys.transpose(0, 1, 3, 2)) * self.scale

        # Apply attention mask if provided
        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask

        attn_weights = mx.softmax(attn_weights, axis=-1)

        # Apply attention to values
        attn_output = attn_weights @ values

        # Reshape back: (batch, heads, seq, head_dim) -> (batch, seq, embed_dim)
        attn_output = attn_output.transpose(0, 2, 1, 3)
        attn_output = attn_output.reshape(batch_size, -1, self.embed_dim)

        # Output projection
        attn_output = self.out_proj(attn_output)

        return attn_output, present_key_value


class MusicGenDecoderLayer(nn.Module):
    """Single decoder layer with self-attention, cross-attention, and FFN."""

    def __init__(self, config: MusicGenDecoderConfig):
        super().__init__()
        self.embed_dim = config.hidden_size

        # Self-attention
        self.self_attn = MusicGenAttention(
            embed_dim=config.hidden_size,
            num_heads=config.num_attention_heads,
            dropout=config.attention_dropout,
            is_cross_attention=False,
            bias=False,
        )
        self.self_attn_layer_norm = nn.LayerNorm(config.hidden_size)

        # Cross-attention to encoder (text encoder outputs)
        # Note: Encoder outputs are projected to decoder hidden size before cross-attention
        self.encoder_attn = MusicGenAttention(
            embed_dim=config.hidden_size,
            num_heads=config.num_attention_heads,
            dropout=config.attention_dropout,
            is_cross_attention=True,
            bias=False,
        )
        self.encoder_attn_layer_norm = nn.LayerNorm(config.hidden_size)

        # Feed-forward network
        self.fc1 = nn.Linear(config.hidden_size, config.ffn_dim, bias=False)
        self.fc2 = nn.Linear(config.ffn_dim, config.hidden_size, bias=False)
        self.final_layer_norm = nn.LayerNorm(config.hidden_size)

        self.activation_fn = nn.GELU()

    def __call__(
        self,
        hidden_states: mx.array,
        encoder_hidden_states: Optional[mx.array] = None,
        attention_mask: Optional[mx.array] = None,
        encoder_attention_mask: Optional[mx.array] = None,
        past_key_value: Optional[Tuple[mx.array, mx.array]] = None,
        cross_attn_past_key_value: Optional[Tuple[mx.array, mx.array]] = None,
    ) -> Tuple[mx.array, Optional[Tuple[mx.array, mx.array]], Optional[Tuple[mx.array, mx.array]]]:
        """
        Forward pass for decoder layer (POST-NORM architecture).

        MusicGen uses post-norm: LayerNorm is applied AFTER the residual addition,
        not before attention/FFN as in pre-norm architectures.

        Args:
            hidden_states: (batch, seq_len, hidden_size)
            encoder_hidden_states: Text encoder outputs for cross-attention
            attention_mask: Causal mask for self-attention
            encoder_attention_mask: Mask for cross-attention
            past_key_value: Cached self-attention key/value states
            cross_attn_past_key_value: Cached cross-attention key/value states

        Returns:
            hidden_states: Updated hidden states
            present_key_value: Updated self-attention cache
            cross_attn_present_key_value: Updated cross-attention cache
        """
        # Self-attention (POST-NORM: attention -> add residual -> layer norm)
        residual = hidden_states
        hidden_states, present_key_value = self.self_attn(
            hidden_states,
            attention_mask=attention_mask,
            past_key_value=past_key_value,
        )
        hidden_states = self.self_attn_layer_norm(residual + hidden_states)

        # Cross-attention (if encoder states provided OR we have cached cross-attn K/V)
        cross_attn_present_key_value = None
        if encoder_hidden_states is not None or cross_attn_past_key_value is not None:
            residual = hidden_states
            hidden_states, cross_attn_present_key_value = self.encoder_attn(
                hidden_states,
                key_value_states=encoder_hidden_states,
                attention_mask=encoder_attention_mask,
                cache=cross_attn_past_key_value,
            )
            hidden_states = self.encoder_attn_layer_norm(residual + hidden_states)

        # Feed-forward (POST-NORM: FFN -> add residual -> layer norm)
        residual = hidden_states
        hidden_states = self.fc1(hidden_states)
        hidden_states = self.activation_fn(hidden_states)
        hidden_states = self.fc2(hidden_states)
        hidden_states = self.final_layer_norm(residual + hidden_states)

        return hidden_states, present_key_value, cross_attn_present_key_value


class MusicGenDecoder(nn.Module):
    """
    MusicGen decoder transformer.

    Generates audio codes conditioned on text embeddings.
    """

    def __init__(self, config: MusicGenDecoderConfig):
        super().__init__()
        self.config = config
        self.num_codebooks = config.num_codebooks

        # Encoder to decoder projection
        # Projects T5 encoder outputs (768d) to decoder hidden size (1024d)
        self.enc_to_dec_proj = nn.Linear(
            config.encoder_hidden_size, config.hidden_size, bias=True
        )

        # Token embeddings - one per codebook
        # MLX expects integer keys in dicts for proper weight loading
        self.embed_tokens = {
            i: nn.Embedding(config.vocab_size + 1, config.hidden_size)
            for i in range(config.num_codebooks)
        }

        # Position embeddings
        self.embed_positions = MusicGenSinusoidalPositionEmbedding(
            config.max_position_embeddings, config.hidden_size
        )

        # Decoder layers
        self.layers = [
            MusicGenDecoderLayer(config) for _ in range(config.num_hidden_layers)
        ]

        # Final layer norm
        self.layer_norm = nn.LayerNorm(config.hidden_size)

        # Output heads - one per codebook (project to vocab)
        # MLX expects integer keys in dicts for proper weight loading
        self.lm_heads = {
            i: nn.Linear(config.hidden_size, config.vocab_size, bias=False)
            for i in range(config.num_codebooks)
        }

    def __call__(
        self,
        input_ids: mx.array,
        encoder_hidden_states: Optional[mx.array] = None,
        attention_mask: Optional[mx.array] = None,
        encoder_attention_mask: Optional[mx.array] = None,
        position_ids: Optional[mx.array] = None,
        past_key_values: Optional[list] = None,
        cross_attn_past_key_values: Optional[list] = None,
        use_cache: bool = False,
    ) -> Tuple[mx.array, Optional[list], Optional[list]]:
        """
        Forward pass for decoder.

        Args:
            input_ids: (batch, num_codebooks, seq_len) audio codes
            encoder_hidden_states: (batch, enc_seq_len, hidden_size) text embeddings
            attention_mask: Causal attention mask
            encoder_attention_mask: Cross-attention mask
            position_ids: Position indices
            past_key_values: Cached self-attention key/value states
            cross_attn_past_key_values: Cached cross-attention key/value states
            use_cache: Whether to return updated cache

        Returns:
            logits: (batch, num_codebooks, seq_len, vocab_size)
            present_key_values: Updated self-attention cache (if use_cache=True)
            cross_attn_present_key_values: Updated cross-attention cache (if use_cache=True)
        """
        batch_size, num_codebooks, seq_len = input_ids.shape

        # Embed tokens for each codebook and sum
        hidden_states = self.embed_tokens[0](input_ids[:, 0, :])
        for i in range(1, self.num_codebooks):
            hidden_states = hidden_states + self.embed_tokens[i](input_ids[:, i, :])

        # Add position embeddings
        if position_ids is None:
            # Safely get past sequence length from cache (handle None, empty list, and None elements)
            past_len = (
                past_key_values[0][0].shape[1]
                if past_key_values and len(past_key_values) > 0 and past_key_values[0] is not None
                else 0
            )
            position_ids = mx.arange(past_len, past_len + seq_len)
            # PERF: Only broadcast when needed (batch_size > 1)
            if batch_size > 1:
                position_ids = mx.broadcast_to(position_ids, (batch_size, seq_len))

        hidden_states = hidden_states + self.embed_positions(position_ids)

        # Create causal mask if not provided
        if attention_mask is None:
            # Safely get past sequence length from cache (handle None, empty list, and None elements)
            cache_len = (
                past_key_values[0][0].shape[1]
                if past_key_values and len(past_key_values) > 0 and past_key_values[0] is not None
                else 0
            )
            attention_mask = self._create_causal_mask(seq_len, cache_len)

        # Project encoder hidden states to decoder hidden size
        # This is done once and cached in cross_attn_past_key_values for incremental decoding
        if encoder_hidden_states is not None:
            encoder_hidden_states = self.enc_to_dec_proj(encoder_hidden_states)

        # Process through layers
        present_key_values = [] if use_cache else None
        cross_attn_present_key_values = [] if use_cache else None
        for i, layer in enumerate(self.layers):
            past_kv = past_key_values[i] if past_key_values else None
            cross_past_kv = cross_attn_past_key_values[i] if cross_attn_past_key_values else None
            hidden_states, present_kv, cross_present_kv = layer(
                hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                attention_mask=attention_mask,
                encoder_attention_mask=encoder_attention_mask,
                past_key_value=past_kv,
                cross_attn_past_key_value=cross_past_kv,
            )
            if use_cache:
                present_key_values.append(present_kv)
                cross_attn_present_key_values.append(cross_present_kv)

        # Final layer norm
        hidden_states = self.layer_norm(hidden_states)

        # Project to vocab for each codebook
        logits = mx.stack(
            [self.lm_heads[i](hidden_states) for i in range(self.num_codebooks)],
            axis=1,
        )  # (batch, num_codebooks, seq_len, vocab_size)

        return logits, present_key_values, cross_attn_present_key_values

    def _create_causal_mask(self, seq_len: int, past_len: int = 0) -> mx.array:
        """Create causal attention mask (vectorized, no Python loops)."""
        total_len = past_len + seq_len
        # Create position indices
        # row_indices: which query position (0 to seq_len-1)
        # col_indices: which key position (0 to total_len-1)
        row_indices = mx.arange(seq_len)[:, None]  # (seq_len, 1)
        col_indices = mx.arange(total_len)[None, :]  # (1, total_len)

        # Each position i can attend to positions 0 to past_len + i (inclusive)
        # So: col_indices <= past_len + row_indices
        # Mask positions where col > past_len + row
        # PERF: Use scalars instead of full tensor allocations - MLX broadcasts efficiently
        causal_mask = mx.where(
            col_indices <= (past_len + row_indices),
            0.0,
            float("-inf"),
        )
        return causal_mask[None, None, :, :]  # (1, 1, seq_len, total_len)


def load_musicgen_decoder_weights(
    model_path: str, dtype: mx.Dtype = mx.float32
) -> dict:
    """
    Load MusicGen decoder weights from pretrained model.

    Supports both SafeTensors and PyTorch .bin formats:
    1. model.safetensors.index.json (sharded safetensors)
    2. model.safetensors (single safetensors)
    3. pytorch_model.bin.index.json (sharded pytorch)
    4. pytorch_model.bin (single pytorch)

    Args:
        model_path: Path to model directory
        dtype: Target dtype for weights

    Returns:
        Dictionary of weight tensors
    """
    from pathlib import Path

    from mlx_music.weights.weight_loader import (
        load_pytorch_bin,
        load_sharded_pytorch,
        load_sharded_safetensors,
        load_single_safetensors,
    )

    model_path = Path(model_path)

    # Try formats in order of preference
    if (model_path / "model.safetensors.index.json").exists():
        all_weights = load_sharded_safetensors(model_path)
    elif (model_path / "model.safetensors").exists():
        all_weights = load_single_safetensors(model_path / "model.safetensors")
    elif (model_path / "pytorch_model.bin.index.json").exists():
        all_weights = load_sharded_pytorch(model_path)
    elif (model_path / "pytorch_model.bin").exists():
        all_weights = load_pytorch_bin(model_path / "pytorch_model.bin")
    else:
        raise FileNotFoundError(
            f"No weights found in {model_path}. "
            f"Expected model.safetensors, model.safetensors.index.json, "
            f"pytorch_model.bin, or pytorch_model.bin.index.json"
        )

    # Filter to decoder weights and remap keys
    decoder_weights = {}
    for key, value in all_weights.items():
        # Handle enc_to_dec_proj at root level (projects encoder to decoder hidden size)
        if key.startswith("enc_to_dec_proj."):
            decoder_weights[key] = value.astype(dtype)
            continue

        if not key.startswith("decoder."):
            continue

        # Remove "decoder." prefix and remap
        new_key = key[8:]  # Remove "decoder."

        # Remap model.decoder -> direct
        new_key = new_key.replace("model.decoder.", "")

        # Handle embed_tokens (list of embeddings)
        if new_key.startswith("embed_tokens."):
            # embed_tokens.0.weight -> embed_tokens.0.weight (keep as-is for list indexing)
            pass

        # Handle lm_heads (list of linear layers)
        # lm_heads.0.weight stays the same

        # Handle embed_positions
        if new_key == "embed_positions.weights":
            new_key = "embed_positions.weights"

        # Convert dtype
        decoder_weights[new_key] = value.astype(dtype)

    return decoder_weights
