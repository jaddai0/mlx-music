"""Autoregressive generation loop for MOSS-SoundEffect (MLX)."""

from __future__ import annotations

from typing import Any

import mlx.core as mx
from mlx_lm.models.cache import make_prompt_cache

from .config import MossSoundEffectConfig
from .sampling import sample_token, repetition_penalty

# MLX lazy computation materializer (NOT Python's eval())
_materialize = mx.eval


def find_last_equal(tensor: mx.array, value: int) -> mx.array:
    """Find the last occurrence of `value` in each row.

    Args:
        tensor: (B, S) int array.
        value: Scalar to search for.

    Returns:
        indices: (B,) last index where tensor[b, idx] == value, or -1 if not found.
    """
    mask = (tensor == value).astype(mx.int32)
    flipped = mask[:, ::-1]
    flipped_idx = mx.argmax(flipped, axis=1)
    seq_len = tensor.shape[1]
    last_idx = seq_len - 1 - flipped_idx

    # Check if the value was actually found
    actual = tensor[mx.arange(tensor.shape[0]), last_idx]
    no_match = actual != value
    last_idx = mx.where(no_match, mx.array(-1), last_idx)
    return last_idx


def generate(
    model: Any,
    input_ids: mx.array,
    config: MossSoundEffectConfig,
    max_new_tokens: int = 1000,
    text_temperature: float = 1.5,
    text_top_p: float = 0.6,
    text_top_k: int = 50,
    audio_temperature: float = 1.5,
    audio_top_p: float = 0.6,
    audio_top_k: int = 50,
    audio_repetition_penalty: float = 1.2,
    callback: Any = None,
) -> list[tuple[int, mx.array]]:
    """Autoregressive generation for MOSS-SoundEffect.

    Args:
        model: MossSoundEffectModel instance.
        input_ids: (B, S, 1 + n_vq) input token IDs.
        config: Model config.
        max_new_tokens: Maximum tokens to generate.
        text_temperature: Temperature for text token sampling.
        text_top_p: Top-p for text tokens.
        text_top_k: Top-k for text tokens.
        audio_temperature: Temperature for audio token sampling.
        audio_top_p: Top-p for audio tokens.
        audio_top_k: Top-k for audio tokens.
        audio_repetition_penalty: Repetition penalty for audio tokens.
        callback: Optional callback(step, generation_ids) called each step.

    Returns:
        List of (start_length, generation_ids) tuples per batch sample.
    """
    text_do_sample = text_temperature > 0
    audio_do_sample = audio_temperature > 0
    if not text_do_sample:
        text_temperature = 1.0
    if not audio_do_sample:
        audio_temperature = 1.0

    batch_size, seq_len, total_vq = input_ids.shape
    n_vq = total_vq - 1  # Number of audio codebook channels

    generation_ids = input_ids.astype(mx.int32)
    cache = make_prompt_cache(model.language_model)
    current_input_ids = generation_ids

    is_stopping = mx.zeros((batch_size,), dtype=mx.bool_)
    audio_lengths = mx.zeros((batch_size,), dtype=mx.int32)
    INT_MAX = 2**30
    delayed_lengths = mx.full((batch_size,), INT_MAX, dtype=mx.int32)

    # Detect if input already in audio mode
    last_text_tokens = input_ids[:, -1, 0]
    is_continuation = (last_text_tokens == config.audio_start_token_id) | \
                      (last_text_tokens == config.audio_assistant_gen_slot_token_id)
    audio_start_indices = find_last_equal(input_ids[:, :, 0], config.audio_start_token_id)
    audio_start_mask = is_continuation & (audio_start_indices != -1)

    # Set initial audio lengths for continuations
    audio_lengths = mx.where(
        audio_start_mask,
        mx.array(seq_len) - audio_start_indices,
        audio_lengths,
    ).astype(mx.int32)
    is_audio = audio_start_mask

    for time_step in range(max_new_tokens):
        # Forward pass
        logits_list, cache = model(current_input_ids, cache=cache)

        # Extract last-token logits and apply temperature
        text_logits = logits_list[0][:, -1, :] / text_temperature
        audio_logits_list = [
            logits_list[i][:, -1, :] / audio_temperature
            for i in range(1, len(logits_list))
        ]

        # --- Text token sampling ---
        next_text = mx.full((batch_size,), config.pad_token_id, dtype=mx.int32)

        # Delay pattern handling
        in_delay = ~is_stopping & (delayed_lengths < n_vq)
        next_text = mx.where(
            in_delay,
            mx.array(config.audio_assistant_delay_slot_token_id),
            next_text,
        )

        is_audio_eos = ~is_stopping & (delayed_lengths == n_vq)
        next_text = mx.where(
            is_audio_eos,
            mx.array(config.audio_end_token_id),
            next_text,
        )
        is_audio = mx.where(is_audio_eos, mx.array(False), is_audio)

        # For tokens not in delay pattern, sample from text logits
        sampling_text_mask = ~is_stopping & (delayed_lengths > n_vq)

        # Apply masking to text logits
        # Non-audio: mask pad, gen_slot, delay_slot, audio_end
        exclude_tokens_noaudio = [
            config.pad_token_id,
            config.audio_assistant_gen_slot_token_id,
            config.audio_assistant_delay_slot_token_id,
            config.audio_end_token_id,
        ]
        masked_text_logits = text_logits
        for tok in exclude_tokens_noaudio:
            masked_text_logits = mx.where(
                mx.logical_and(~is_audio[:, None], mx.arange(masked_text_logits.shape[-1]) == tok),
                mx.array(float("-inf")),
                masked_text_logits,
            )

        # Audio mode: only allow gen_slot and delay_slot
        audio_allow = mx.zeros((masked_text_logits.shape[-1],), dtype=mx.bool_)
        audio_allow = audio_allow.at[config.audio_assistant_gen_slot_token_id].add(True)
        audio_allow = audio_allow.at[config.audio_assistant_delay_slot_token_id].add(True)
        masked_text_logits = mx.where(
            mx.logical_and(is_audio[:, None], ~audio_allow[None, :]),
            mx.array(float("-inf")),
            masked_text_logits,
        )

        # First step: mask delay token
        if time_step == 0:
            masked_text_logits = mx.where(
                mx.arange(masked_text_logits.shape[-1]) == config.audio_assistant_delay_slot_token_id,
                mx.array(float("-inf")),
                masked_text_logits,
            )

        # Early steps: mask im_end
        if time_step <= n_vq:
            masked_text_logits = mx.where(
                mx.arange(masked_text_logits.shape[-1]) == config.im_end_token_id,
                mx.array(float("-inf")),
                masked_text_logits,
            )

        if mx.any(sampling_text_mask):
            sampled = sample_token(
                masked_text_logits,
                temperature=1.0,  # Already applied
                top_p=text_top_p,
                top_k=text_top_k,
                do_sample=text_do_sample,
            ).astype(mx.int32)
            next_text = mx.where(sampling_text_mask, sampled, next_text)

        _materialize(next_text)

        # Update audio state based on text token
        is_audio = mx.where(next_text == config.audio_start_token_id, mx.array(True), is_audio)
        is_stopping = mx.where(next_text == config.im_end_token_id, mx.array(True), is_stopping)

        # --- Audio token sampling ---
        next_audio = mx.full((batch_size, n_vq), config.audio_pad_code, dtype=mx.int32)

        # Build masks for which audio channels to sample
        vq_range = mx.arange(n_vq)
        pre_audio_mask = audio_lengths[:, None] > vq_range[None, :]
        post_audio_mask = vq_range[None, :] > (delayed_lengths[:, None] - 1)
        # When delayed_lengths == INT_MAX, all channels are post-delay
        post_audio_mask = mx.where(
            delayed_lengths[:, None] == INT_MAX,
            mx.array(True),
            post_audio_mask,
        )
        sampling_audio_mask = pre_audio_mask & post_audio_mask

        if mx.any(sampling_audio_mask):
            # Stack audio logits: (B, n_vq, audio_vocab+1)
            stacked_audio = mx.stack(audio_logits_list, axis=1)

            # Mask padding token
            stacked_audio = mx.where(
                mx.arange(stacked_audio.shape[-1]) == config.audio_pad_code,
                mx.array(float("-inf")),
                stacked_audio,
            )

            # Apply repetition penalty using previous audio tokens
            if audio_repetition_penalty != 1.0 and generation_ids.shape[1] > 1:
                prev_audio = generation_ids[:, :, 1:]
                stacked_audio = repetition_penalty(
                    stacked_audio, prev_audio, audio_repetition_penalty,
                )

            # Sample all channels
            sampled_audio = sample_token(
                stacked_audio.reshape(-1, stacked_audio.shape[-1]),
                temperature=1.0,
                top_p=audio_top_p,
                top_k=audio_top_k,
                do_sample=audio_do_sample,
            ).reshape(batch_size, n_vq).astype(mx.int32)

            # Apply mask: only use sampled tokens where sampling_audio_mask is True
            next_audio = mx.where(sampling_audio_mask, sampled_audio, next_audio)

        # --- State updates ---
        audio_trigger = (
            (next_text == config.audio_start_token_id) |
            (next_text == config.audio_assistant_gen_slot_token_id) |
            (next_text == config.audio_assistant_delay_slot_token_id)
        )
        audio_lengths = mx.where(audio_trigger, audio_lengths + 1, audio_lengths)
        audio_lengths = mx.where(next_text == config.audio_end_token_id, mx.array(0), audio_lengths)

        # Delay pattern tracking
        start_delay = (delayed_lengths == INT_MAX) & (next_text == config.audio_assistant_delay_slot_token_id)
        delayed_lengths = mx.where(start_delay, mx.array(0), delayed_lengths)
        delayed_lengths = mx.where(delayed_lengths != INT_MAX, delayed_lengths + 1, delayed_lengths)
        delayed_lengths = mx.where(delayed_lengths > n_vq, mx.array(INT_MAX), delayed_lengths)

        # Assemble next input
        next_step = mx.concatenate([
            next_text[:, None, None],
            next_audio[:, None, :],
        ], axis=2).astype(mx.int32)  # (B, 1, 1+n_vq)

        generation_ids = mx.concatenate([generation_ids, next_step], axis=1)
        current_input_ids = next_step

        _materialize(generation_ids, cache)

        if callback is not None:
            callback(time_step, generation_ids)

        # Check termination
        if mx.all(is_stopping):
            break

        # Periodic memory cleanup for 8B model
        if time_step % 50 == 0:
            mx.clear_cache()

    # Extract outputs
    start_indices = find_last_equal(input_ids[:, :, 0], config.im_start_token_id) + 3
    output = []
    for b in range(batch_size):
        start_idx = int(start_indices[b].item())
        start_length = seq_len - start_idx
        output.append((start_length, generation_ids[b, start_idx:]))

    return output
