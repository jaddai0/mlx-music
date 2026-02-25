"""
Full DiT model and condition generation model for ACE-Step v1.5.

Implements:
- AceStepDiTModel: patch_embed -> 24 DiT layers -> de-patch with AdaLN output
- AceStepConditionGenerationModel: wraps DiT + encoder + tokenizer + detokenizer
"""

import math
import time
from typing import Dict, List, Optional, Tuple, Union

import mlx.core as mx
import mlx.nn as nn

from mlx_music.models.ace_step_v15.attention import (
    RMSNorm,
    RotaryEmbedding,
    create_sliding_mask,
)
from mlx_music.models.ace_step_v15.config import AceStepV15Config
from mlx_music.models.ace_step_v15.dit import AceStepDiTLayer, TimestepEmbedding
from mlx_music.models.ace_step_v15.encoders import AceStepConditionEncoder
from mlx_music.models.ace_step_v15.tokenizer import (
    AceStepAudioTokenizer,
    AudioTokenDetokenizer,
)


# Pre-defined timestep schedules for each valid shift (fix_nfe=8, excluding final 0)
SHIFT_TIMESTEPS = {
    1.0: [1.0, 0.875, 0.75, 0.625, 0.5, 0.375, 0.25, 0.125],
    2.0: [1.0, 0.9333333333333333, 0.8571428571428571, 0.7692307692307693,
          0.6666666666666666, 0.5454545454545454, 0.4, 0.2222222222222222],
    3.0: [1.0, 0.9545454545454546, 0.9, 0.8333333333333334,
          0.75, 0.6428571428571429, 0.5, 0.3],
}

VALID_SHIFTS = [1.0, 2.0, 3.0]

VALID_TIMESTEPS = [
    1.0, 0.9545454545454546, 0.9333333333333333, 0.9, 0.875,
    0.8571428571428571, 0.8333333333333334, 0.7692307692307693, 0.75,
    0.6666666666666666, 0.6428571428571429, 0.625, 0.5454545454545454,
    0.5, 0.4, 0.375, 0.3, 0.25, 0.2222222222222222, 0.125,
]


class AceStepDiTModel(nn.Module):
    """DiT model for diffusion-based audio generation.

    Processes noisy latents conditioned on encoder states via cross-attention.
    Uses patch-based processing (Conv1d stride=2) and AdaLN for timestep conditioning.
    """

    def __init__(self, config: AceStepV15Config):
        super().__init__()
        self.config = config
        inner_dim = config.hidden_size
        in_channels = config.in_channels
        self.patch_size = config.patch_size

        # Rotary position embeddings
        self.rotary_emb = RotaryEmbedding(
            config.head_dim, config.max_position_embeddings, config.rope_theta
        )

        # Patch embedding: Conv1d with stride=patch_size
        # In MLX NLC format: (B, T, C) -> Conv1d -> (B, T/patch_size, inner_dim)
        self.proj_in = nn.Conv1d(
            in_channels=in_channels,
            out_channels=inner_dim,
            kernel_size=config.patch_size,
            stride=config.patch_size,
            padding=0,
        )

        # Two timestep embeddings: t and (t - r)
        self.time_embed = TimestepEmbedding(in_channels=256, time_embed_dim=inner_dim)
        self.time_embed_r = TimestepEmbedding(in_channels=256, time_embed_dim=inner_dim)

        # Project encoder hidden states
        self.condition_embedder = nn.Linear(inner_dim, inner_dim, bias=True)

        # DiT layers
        self.layers = [
            AceStepDiTLayer(
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
                use_cross_attention=True,
            )
            for i in range(config.num_hidden_layers)
        ]

        # Output: AdaLN norm + de-patchify via ConvTranspose1d
        self.norm_out = RMSNorm(inner_dim, eps=config.rms_norm_eps)
        self.proj_out = nn.ConvTranspose1d(
            in_channels=inner_dim,
            out_channels=config.audio_acoustic_hidden_dim,
            kernel_size=config.patch_size,
            stride=config.patch_size,
            padding=0,
        )
        # Scale-shift table for output AdaLN: [1, 2, inner_dim]
        self.scale_shift_table = mx.zeros((1, 2, inner_dim))

    def __call__(
        self,
        hidden_states: mx.array,
        timestep: mx.array,
        timestep_r: mx.array,
        encoder_hidden_states: mx.array,
        encoder_attention_mask: Optional[mx.array] = None,
        context_latents: mx.array = None,
        cross_kv_cache: Optional[Dict[int, Tuple[mx.array, mx.array]]] = None,
    ) -> Tuple[mx.array, Optional[Dict[int, Tuple[mx.array, mx.array]]]]:
        """
        Args:
            hidden_states: (B, T, audio_acoustic_hidden_dim) noisy latents
            timestep: (B,) current timestep t
            timestep_r: (B,) reference timestep r
            encoder_hidden_states: (B, enc_T, hidden_size) conditioning
            encoder_attention_mask: (B, enc_T) or None
            context_latents: (B, T, in_channels - audio_acoustic_hidden_dim) context
            cross_kv_cache: Dict mapping layer_idx -> (k, v)

        Returns:
            (predicted_flow, updated_cross_kv_cache)
        """
        # Concatenate context latents with hidden states
        if context_latents is not None:
            hidden_states = mx.concatenate([context_latents, hidden_states], axis=-1)

        # Record original seq len for cropping after de-patch
        original_seq_len = hidden_states.shape[1]

        # Pad if not divisible by patch_size
        pad_length = 0
        if hidden_states.shape[1] % self.patch_size != 0:
            pad_length = self.patch_size - (hidden_states.shape[1] % self.patch_size)
            padding = mx.zeros((hidden_states.shape[0], pad_length, hidden_states.shape[2]))
            hidden_states = mx.concatenate([hidden_states, padding], axis=1)

        # Compute timestep embeddings
        temb_t, proj_t = self.time_embed(timestep)
        temb_r, proj_r = self.time_embed_r(timestep - timestep_r)
        temb = temb_t + temb_r
        timestep_proj = proj_t + proj_r

        # Patch embedding (NLC Conv1d)
        hidden_states = self.proj_in(hidden_states)

        # Project encoder hidden states
        encoder_hidden_states = self.condition_embedder(encoder_hidden_states)

        # Build attention masks
        seq_len = hidden_states.shape[1]
        dtype = hidden_states.dtype

        mask_mapping = {"full_attention": None, "sliding_attention": None}
        if self.config.use_sliding_window:
            mask_mapping["sliding_attention"] = create_sliding_mask(
                seq_len, self.config.sliding_window, is_sliding=True, dtype=dtype
            )

        # Position embeddings
        cos, sin = self.rotary_emb(seq_len, dtype=dtype)

        # Process through DiT layers
        updated_cache = cross_kv_cache
        for i, layer in enumerate(self.layers):
            attn_type = self.config.layer_types[i]
            hidden_states, updated_cache = layer(
                hidden_states=hidden_states,
                position_embeddings=(cos, sin),
                temb=timestep_proj,
                attention_mask=mask_mapping.get(attn_type),
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=None,
                cross_kv_cache=updated_cache,
                layer_idx=i,
            )

        # Output AdaLN: norm(x) * (1 + scale) + shift
        combined = self.scale_shift_table + mx.expand_dims(temb, axis=1)
        shift = combined[:, 0:1, :]
        scale = combined[:, 1:2, :]
        hidden_states = self.norm_out(hidden_states) * (1 + scale) + shift

        # De-patchify via ConvTranspose1d
        hidden_states = self.proj_out(hidden_states)

        # Crop to original length
        hidden_states = hidden_states[:, :original_seq_len, :]

        return hidden_states, updated_cache


class AceStepConditionGenerationModel(nn.Module):
    """Full conditional generation model combining encoder + DiT + tokenizer.

    Handles:
    - Condition preparation (text + lyrics + timbre encoding)
    - Tokenization/detokenization for cover song LM hints
    - ODE-based flow matching inference
    """

    def __init__(self, config: AceStepV15Config):
        super().__init__()
        self.config = config
        self.decoder = AceStepDiTModel(config)
        self.encoder = AceStepConditionEncoder(config)
        self.tokenizer = AceStepAudioTokenizer(config)
        self.detokenizer = AudioTokenDetokenizer(config)
        # Null condition embedding for CFG
        self.null_condition_emb = mx.zeros((1, 1, config.hidden_size))

    def tokenize(
        self,
        x: mx.array,
        silence_latent: mx.array,
        attention_mask: mx.array,
    ) -> Tuple[mx.array, mx.array, mx.array]:
        """Tokenize latents with padding and mask pooling."""
        P = self.config.pool_window_size
        if x.shape[1] % P != 0:
            pad_len = P - (x.shape[1] % P)
            pad = mx.broadcast_to(
                silence_latent[:1, :pad_len], (x.shape[0], pad_len, x.shape[2])
            )
            x = mx.concatenate([x, pad], axis=1)
            mask_pad = mx.zeros((attention_mask.shape[0], pad_len))
            attention_mask = mx.concatenate([attention_mask, mask_pad], axis=1)

        B, T, D = x.shape
        x = x.reshape(B, T // P, P, D)
        seq_len = T // P

        # Pool the attention mask using max pooling equivalent
        chunk = math.ceil(attention_mask.shape[1] / seq_len)
        mask_len = attention_mask.shape[1]
        padded_mask_len = seq_len * chunk
        if mask_len < padded_mask_len:
            mask_pad = mx.zeros((B, padded_mask_len - mask_len))
            pooled_mask = mx.concatenate([attention_mask, mask_pad], axis=1)
        else:
            pooled_mask = attention_mask[:, :padded_mask_len]
        pooled_mask = pooled_mask.reshape(B, seq_len, chunk).max(axis=2)

        quantized, indices = self.tokenizer(x)
        return quantized, indices, pooled_mask

    def detokenize(self, quantized: mx.array) -> mx.array:
        """Detokenize quantized tokens back to continuous representations."""
        return self.detokenizer(quantized)

    def prepare_condition(
        self,
        text_hidden_states: mx.array,
        text_attention_mask: mx.array,
        lyric_hidden_states: mx.array,
        lyric_attention_mask: mx.array,
        refer_audio_packed: mx.array,
        refer_audio_order_mask: mx.array,
        hidden_states: mx.array,
        attention_mask: mx.array,
        silence_latent: mx.array,
        src_latents: mx.array,
        chunk_masks: mx.array,
        is_covers: mx.array,
        precomputed_lm_hints_25Hz: Optional[mx.array] = None,
        audio_codes: Optional[mx.array] = None,
    ) -> Tuple[mx.array, mx.array, mx.array]:
        """Prepare all conditioning inputs for the DiT model."""
        dtype = hidden_states.dtype

        encoder_hidden_states, encoder_attention_mask = self.encoder(
            text_hidden_states=text_hidden_states,
            text_attention_mask=text_attention_mask,
            lyric_hidden_states=lyric_hidden_states,
            lyric_attention_mask=lyric_attention_mask,
            refer_audio_packed=refer_audio_packed,
            refer_audio_order_mask=refer_audio_order_mask,
        )

        # Get LM hints for cover songs
        if precomputed_lm_hints_25Hz is not None:
            lm_hints_25Hz = precomputed_lm_hints_25Hz[:, :src_latents.shape[1], :]
        else:
            if audio_codes is not None:
                lm_hints_5Hz = self.tokenizer.quantizer.get_output_from_indices(audio_codes)
            else:
                lm_hints_5Hz, _, _ = self.tokenize(hidden_states, silence_latent, attention_mask)
            lm_hints_25Hz = self.detokenize(lm_hints_5Hz)
            lm_hints_25Hz = lm_hints_25Hz[:, :src_latents.shape[1], :]

        # Use LM hints for cover songs, src_latents for non-cover
        is_cover_mask = (is_covers > 0).astype(dtype)
        is_cover_mask = mx.expand_dims(mx.expand_dims(is_cover_mask, -1), -1)
        src_latents = mx.where(is_cover_mask > 0, lm_hints_25Hz, src_latents)

        # Concatenate source latents with chunk masks as context
        context_latents = mx.concatenate([src_latents, chunk_masks.astype(dtype)], axis=-1)

        return encoder_hidden_states, encoder_attention_mask, context_latents

    def prepare_noise(
        self,
        context_latents: mx.array,
        seed: Optional[Union[int, List[int]]] = None,
    ) -> mx.array:
        """Prepare noise tensor for generation."""
        shape = (
            context_latents.shape[0],
            context_latents.shape[1],
            context_latents.shape[-1] // 2,
        )
        if seed is not None:
            mx.random.seed(seed if isinstance(seed, int) else seed[0])
        return mx.random.normal(shape)

    def _materialize(self, *arrays):
        """Force evaluation of lazy MLX arrays."""
        mx.eval(*arrays)

    def generate_audio(
        self,
        text_hidden_states: mx.array,
        text_attention_mask: mx.array,
        lyric_hidden_states: mx.array,
        lyric_attention_mask: mx.array,
        refer_audio_packed: mx.array,
        refer_audio_order_mask: mx.array,
        src_latents: mx.array,
        chunk_masks: mx.array,
        is_covers: mx.array,
        silence_latent: Optional[mx.array] = None,
        attention_mask: Optional[mx.array] = None,
        seed: Optional[int] = None,
        fix_nfe: int = 8,
        shift: float = 3.0,
        timesteps: Optional[List[float]] = None,
        precomputed_lm_hints_25Hz: Optional[mx.array] = None,
        audio_codes: Optional[mx.array] = None,
        audio_cover_strength: float = 1.0,
    ) -> Dict:
        """Generate audio latents using ODE flow matching."""
        # Determine timestep schedule
        t_schedule_list = None
        if timesteps is not None:
            ts = list(timesteps)
            while ts and ts[-1] == 0:
                ts.pop()
            if ts:
                t_schedule_list = [
                    min(VALID_TIMESTEPS, key=lambda x, t=t: abs(x - t)) for t in ts
                ]

        if t_schedule_list is None:
            shift = min(VALID_SHIFTS, key=lambda x: abs(x - shift))
            t_schedule_list = SHIFT_TIMESTEPS[shift]

        if attention_mask is None:
            attention_mask = mx.ones((src_latents.shape[0], src_latents.shape[1]))

        time_costs = {}
        start_time = time.time()
        total_start_time = start_time

        # Prepare conditions
        encoder_hidden_states, encoder_attention_mask, context_latents = self.prepare_condition(
            text_hidden_states=text_hidden_states,
            text_attention_mask=text_attention_mask,
            lyric_hidden_states=lyric_hidden_states,
            lyric_attention_mask=lyric_attention_mask,
            refer_audio_packed=refer_audio_packed,
            refer_audio_order_mask=refer_audio_order_mask,
            hidden_states=src_latents,
            attention_mask=attention_mask,
            silence_latent=silence_latent if silence_latent is not None else mx.zeros_like(src_latents),
            src_latents=src_latents,
            chunk_masks=chunk_masks,
            is_covers=is_covers,
            precomputed_lm_hints_25Hz=precomputed_lm_hints_25Hz,
            audio_codes=audio_codes,
        )
        self._materialize(encoder_hidden_states, encoder_attention_mask, context_latents)

        end_time = time.time()
        time_costs["encoder_time_cost"] = end_time - start_time
        start_time = end_time

        # Prepare noise
        noise = self.prepare_noise(context_latents, seed)
        bsz = context_latents.shape[0]
        num_steps = len(t_schedule_list)

        # ODE loop
        xt = noise
        cross_kv_cache = None

        for step_idx in range(num_steps):
            current_t = t_schedule_list[step_idx]
            t_curr = mx.full((bsz,), current_t)

            vt, cross_kv_cache = self.decoder(
                hidden_states=xt,
                timestep=t_curr,
                timestep_r=t_curr,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_attention_mask,
                context_latents=context_latents,
                cross_kv_cache=cross_kv_cache,
            )

            if step_idx == num_steps - 1:
                # Final step: compute x0 directly
                t_expand = mx.expand_dims(mx.expand_dims(t_curr, -1), -1)
                xt = xt - vt * t_expand
                self._materialize(xt)
                break

            # Euler ODE step
            next_t = t_schedule_list[step_idx + 1]
            dt = current_t - next_t
            dt_tensor = mx.full((bsz, 1, 1), dt)
            xt = xt - vt * dt_tensor
            self._materialize(xt)

        end_time = time.time()
        time_costs["diffusion_time_cost"] = end_time - start_time
        time_costs["diffusion_per_step_time_cost"] = time_costs["diffusion_time_cost"] / num_steps
        time_costs["total_time_cost"] = end_time - total_start_time

        return {
            "target_latents": xt,
            "time_costs": time_costs,
        }


__all__ = [
    "AceStepDiTModel",
    "AceStepConditionGenerationModel",
    "SHIFT_TIMESTEPS",
    "VALID_SHIFTS",
    "VALID_TIMESTEPS",
]
