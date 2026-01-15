"""
MusicGen generation utilities.

Sampling strategies and autoregressive generation for MusicGen.
"""

from dataclasses import dataclass
from typing import Callable, Optional, Tuple

import mlx.core as mx
import numpy as np


@dataclass
class GenerationConfig:
    """Configuration for MusicGen audio generation."""

    # Duration
    duration: float = 10.0  # seconds
    max_duration: float = 30.0  # Maximum allowed duration

    # Sampling parameters
    temperature: float = 1.0
    top_k: int = 250
    top_p: float = 0.0  # 0.0 = disabled

    # Classifier-free guidance
    guidance_scale: float = 3.0

    # Generation
    seed: Optional[int] = None
    use_cache: bool = True


@dataclass
class GenerationOutput:
    """Output from MusicGen generation."""

    audio: np.ndarray  # (channels, samples) audio waveform
    sample_rate: int
    duration: float
    codes: Optional[mx.array] = None  # Raw audio codes if requested


def top_k_filtering(logits: mx.array, top_k: int) -> mx.array:
    """
    Filter logits to keep only top-k values.

    Args:
        logits: (batch, vocab_size) logits
        top_k: Number of top values to keep

    Returns:
        Filtered logits with -inf for non-top-k positions
    """
    if top_k <= 0:
        return logits

    # Get top-k values
    top_k = min(top_k, logits.shape[-1])
    values = mx.sort(logits, axis=-1)[:, -top_k:]
    # Use the value at position -top_k as threshold
    # NOTE: Use < (not <=) because we want to keep values >= threshold
    # But we need to handle duplicates at the threshold value
    # The k-th largest value should be included
    threshold = values[:, 0:1]  # (batch, 1) - this is the k-th largest value

    # Mask values strictly below threshold (keep values >= threshold)
    # This ensures all values equal to threshold are kept (may keep more than k if duplicates)
    mask = logits < threshold
    neg_inf = mx.full(logits.shape, float("-inf"), dtype=logits.dtype)
    return mx.where(mask, neg_inf, logits)


def top_p_filtering(logits: mx.array, top_p: float) -> mx.array:
    """
    Filter logits using nucleus (top-p) sampling.

    Keeps smallest set of tokens whose cumulative probability >= top_p.

    Args:
        logits: (batch, vocab_size) logits
        top_p: Cumulative probability threshold

    Returns:
        Filtered logits with -inf for excluded positions
    """
    if top_p >= 1.0 or top_p <= 0.0:
        return logits

    # Sort by descending probability
    sorted_indices = mx.argsort(-logits, axis=-1)
    sorted_logits = mx.take_along_axis(logits, sorted_indices, axis=-1)
    cumulative_probs = mx.cumsum(mx.softmax(sorted_logits, axis=-1), axis=-1)

    # Find cutoff
    sorted_mask = cumulative_probs > top_p
    # Shift mask right to include the first token that exceeds threshold
    sorted_mask = mx.concatenate(
        [mx.zeros_like(sorted_mask[:, :1]), sorted_mask[:, :-1]], axis=-1
    )

    # Create output logits - copy and mask based on sorted positions
    # Since top_p filters out low-probability tokens, we can process each batch
    output_logits = logits.astype(mx.float32)  # Make a copy

    # Use a vectorized approach with argsort to reverse the mapping
    batch_size, vocab_size = logits.shape
    reverse_indices = mx.argsort(sorted_indices, axis=-1)

    # Apply mask: for each position in original order, check if it should be masked
    sorted_mask_float = sorted_mask.astype(mx.float32)

    # Gather sorted_mask at reverse_indices to get mask in original order
    mask_original = mx.take_along_axis(sorted_mask_float, reverse_indices, axis=-1)

    neg_inf = mx.full(logits.shape, float("-inf"), dtype=logits.dtype)
    return mx.where(mask_original > 0.5, neg_inf, logits)


def sample_next_token(
    logits: mx.array,
    temperature: float = 1.0,
    top_k: int = 250,
    top_p: float = 0.0,
) -> mx.array:
    """
    Sample next token from logits.

    Args:
        logits: (batch, vocab_size) logits
        temperature: Sampling temperature (must be > 0, use very small for greedy)
        top_k: Top-k filtering (0 = disabled)
        top_p: Nucleus sampling threshold (0.0 = disabled)

    Returns:
        (batch, 1) sampled token indices
    """
    # Handle temperature = 0 or near-zero (greedy decoding)
    # Near-zero temperatures cause numerical overflow when dividing logits
    if temperature <= 1e-6:
        # Greedy: just take argmax
        next_token = mx.argmax(logits, axis=-1)
        return next_token[:, None]

    # Apply temperature
    if temperature != 1.0:
        logits = logits / temperature

    # Apply top-k filtering
    if top_k > 0:
        logits = top_k_filtering(logits, top_k)

    # Apply top-p filtering
    if top_p > 0.0:
        logits = top_p_filtering(logits, top_p)

    # Sample from logits using categorical
    # mx.random.categorical expects logits (unnormalized log-probabilities)
    # NOT log(softmax(logits)) which would double-transform
    next_token = mx.random.categorical(logits)

    return next_token[:, None]


def apply_classifier_free_guidance(
    cond_logits: mx.array,
    uncond_logits: mx.array,
    guidance_scale: float,
) -> mx.array:
    """
    Apply classifier-free guidance to logits.

    Args:
        cond_logits: Conditioned logits
        uncond_logits: Unconditioned logits
        guidance_scale: CFG scale (1.0 = no guidance)

    Returns:
        Guided logits
    """
    if guidance_scale == 1.0:
        return cond_logits

    return uncond_logits + guidance_scale * (cond_logits - uncond_logits)


class MusicGenGenerator:
    """
    Autoregressive generator for MusicGen.

    Handles the generation loop including:
    - Token sampling with top-k/top-p filtering
    - Classifier-free guidance
    - KV caching for efficient generation
    """

    def __init__(
        self,
        decoder,
        config,
        encodec,
        text_encoder,
    ):
        """
        Initialize generator.

        Args:
            decoder: MusicGen decoder transformer
            config: MusicGen configuration
            encodec: EnCodec audio codec
            text_encoder: Text encoder for conditioning
        """
        self.decoder = decoder
        self.config = config
        self.encodec = encodec
        self.text_encoder = text_encoder

    def generate(
        self,
        prompt: str,
        duration: float = 10.0,
        temperature: float = 1.0,
        top_k: int = 250,
        top_p: float = 0.0,
        guidance_scale: float = 3.0,
        seed: Optional[int] = None,
        callback: Optional[Callable[[int, int, mx.array], None]] = None,
        return_codes: bool = False,
    ) -> GenerationOutput:
        """
        Generate music from text prompt.

        Args:
            prompt: Text description of desired music
            duration: Target duration in seconds
            temperature: Sampling temperature
            top_k: Top-k filtering
            top_p: Nucleus sampling threshold
            guidance_scale: Classifier-free guidance scale
            seed: Random seed for reproducibility
            callback: Optional callback(step, total_steps, codes)
            return_codes: Whether to return raw audio codes

        Returns:
            GenerationOutput with audio and metadata
        """
        # Validate duration
        if duration <= 0:
            raise ValueError(f"duration must be positive, got {duration}")
        max_duration = 30.0  # Maximum allowed duration
        if duration > max_duration:
            raise ValueError(f"duration must be <= {max_duration} seconds, got {duration}")

        # Set seed
        if seed is not None:
            mx.random.seed(seed)

        # Calculate target length using config frame_rate
        frame_rate = self.config.frame_rate
        target_length = max(1, int(duration * frame_rate))  # At least 1 step

        # Encode text prompt
        encoder_hidden_states, encoder_attention_mask = self.text_encoder.encode(prompt)

        # For CFG, also need unconditional encoding (empty prompt)
        if guidance_scale > 1.0:
            uncond_hidden_states, uncond_mask = self.text_encoder.encode("")

        # Initialize with BOS tokens for all codebooks
        batch_size = 1
        num_codebooks = self.config.decoder.num_codebooks
        bos_token = self.config.decoder.bos_token_id

        # (batch, num_codebooks, seq_len)
        codes = mx.full((batch_size, num_codebooks, 1), bos_token, dtype=mx.int32)

        # KV caches for conditional path (self-attention and cross-attention)
        past_key_values = None
        cross_attn_past_key_values = None
        # KV caches for unconditional path (CFG)
        uncond_past_key_values = None
        uncond_cross_attn_past_key_values = None

        # Generation loop
        for step in range(target_length):
            # Get logits from decoder (conditional)
            # On first step: compute cross-attention K/V from encoder_hidden_states
            # On subsequent steps: reuse cached cross-attention K/V (huge perf win)
            logits, past_key_values, cross_attn_past_key_values = self.decoder(
                input_ids=codes if past_key_values is None else codes[:, :, -1:],
                encoder_hidden_states=encoder_hidden_states if cross_attn_past_key_values is None else None,
                encoder_attention_mask=encoder_attention_mask,
                past_key_values=past_key_values,
                cross_attn_past_key_values=cross_attn_past_key_values,
                use_cache=True,
            )

            # logits shape: (batch, num_codebooks, seq_len, vocab_size)
            # Take last position
            next_logits = logits[:, :, -1, :]  # (batch, num_codebooks, vocab_size)

            # Apply CFG with proper caching for unconditional path
            if guidance_scale > 1.0:
                # Get unconditional logits WITH caching (for efficiency)
                uncond_logits, uncond_past_key_values, uncond_cross_attn_past_key_values = self.decoder(
                    input_ids=codes if uncond_past_key_values is None else codes[:, :, -1:],
                    encoder_hidden_states=uncond_hidden_states if uncond_cross_attn_past_key_values is None else None,
                    encoder_attention_mask=uncond_mask,
                    past_key_values=uncond_past_key_values,
                    cross_attn_past_key_values=uncond_cross_attn_past_key_values,
                    use_cache=True,
                )
                uncond_next = uncond_logits[:, :, -1, :]

                next_logits = apply_classifier_free_guidance(
                    next_logits, uncond_next, guidance_scale
                )

            # Sample next tokens for each codebook
            next_tokens = []
            for cb in range(num_codebooks):
                cb_logits = next_logits[:, cb, :]
                next_token = sample_next_token(cb_logits, temperature, top_k, top_p)
                next_tokens.append(next_token)

            next_tokens = mx.stack(next_tokens, axis=1)  # (batch, num_codebooks, 1)
            codes = mx.concatenate([codes, next_tokens], axis=-1)

            # Evaluate to prevent memory buildup (codes AND all KV caches)
            mx.eval(codes)
            if past_key_values is not None:
                mx.eval(past_key_values)
            if cross_attn_past_key_values is not None:
                mx.eval(cross_attn_past_key_values)
            if uncond_past_key_values is not None:
                mx.eval(uncond_past_key_values)
            if uncond_cross_attn_past_key_values is not None:
                mx.eval(uncond_cross_attn_past_key_values)

            # Callback
            if callback is not None:
                callback(step, target_length, codes)

        # Remove BOS token
        codes = codes[:, :, 1:]

        # Decode to audio
        audio = self.encodec.decode(codes)
        audio_np = np.array(audio)

        # Remove batch dimension if present
        if audio_np.ndim == 3:
            audio_np = audio_np[0]

        return GenerationOutput(
            audio=audio_np,
            sample_rate=self.config.audio_encoder.sampling_rate,
            duration=duration,
            codes=codes if return_codes else None,
        )

    def generate_continuation(
        self,
        audio: mx.array,
        prompt: str,
        duration: float = 10.0,
        **kwargs,
    ) -> GenerationOutput:
        """
        Continue generation from existing audio.

        Args:
            audio: Existing audio to continue from
            prompt: Text prompt for continuation
            duration: Additional duration to generate
            **kwargs: Additional generation parameters

        Returns:
            GenerationOutput with continued audio
        """
        # Encode existing audio to codes
        existing_codes, _ = self.encodec.encode(audio)

        # Continue generation from those codes
        # ... (implementation would continue the generation loop)

        raise NotImplementedError("Continuation generation not yet implemented")


__all__ = [
    "GenerationConfig",
    "GenerationOutput",
    "top_k_filtering",
    "top_p_filtering",
    "sample_next_token",
    "apply_classifier_free_guidance",
    "MusicGenGenerator",
]
