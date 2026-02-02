"""
MusicGen generation utilities.

Sampling strategies and autoregressive generation for MusicGen.
"""

from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple, Union

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
    use_sampling: bool = True  # False = greedy decoding (always take argmax)

    # Classifier-free guidance
    guidance_scale: float = 3.0
    guidance_scale_beta: float = 0.0  # Secondary CFG scale (for melody), 0 = disabled

    # Extended generation (>30s)
    extend_stride: int = 750  # tokens per extension (~15s at 50fps)
    window_length: int = 1500  # max tokens per window (~30s at 50fps)
    fade_duration: float = 0.5  # crossfade duration in seconds

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

    # PERF: Use mx.topk (O(V)) instead of mx.sort (O(V log V))
    top_k = min(top_k, logits.shape[-1])
    top_values = mx.topk(logits, k=top_k, axis=-1)  # (batch, top_k) - not sorted
    # The threshold is the minimum of the top-k values
    threshold = mx.min(top_values, axis=-1, keepdims=True)  # (batch, 1)

    # Mask values strictly below threshold (keep values >= threshold)
    # This ensures all values equal to threshold are kept (may keep more than k if duplicates)
    mask = logits < threshold
    # PERF: Use scalars for mx.where instead of full tensor allocation
    return mx.where(mask, float("-inf"), logits)


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

    # Reverse mapping to original order using argsort
    reverse_indices = mx.argsort(sorted_indices, axis=-1)

    # Gather sorted_mask at reverse_indices to get mask in original order
    # PERF: Keep as boolean, use directly in mx.where (no float conversion needed)
    mask_original = mx.take_along_axis(sorted_mask, reverse_indices, axis=-1)

    # PERF: Use scalar for mx.where instead of full tensor allocation
    return mx.where(mask_original, float("-inf"), logits)


def sample_next_token(
    logits: mx.array,
    temperature: float = 1.0,
    top_k: int = 250,
    top_p: float = 0.0,
    use_sampling: bool = True,
) -> mx.array:
    """
    Sample next token from logits.

    Args:
        logits: (batch, vocab_size) logits
        temperature: Sampling temperature (must be > 0, use very small for greedy)
        top_k: Top-k filtering (0 = disabled)
        top_p: Nucleus sampling threshold (0.0 = disabled)
        use_sampling: If False, always use greedy decoding (argmax)

    Returns:
        (batch, 1) sampled token indices
    """
    # Greedy decoding: always take argmax
    if not use_sampling:
        next_token = mx.argmax(logits, axis=-1)
        return next_token[:, None]

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


def _compute_rms(x: mx.array, eps: float = 1e-8) -> mx.array:
    """
    Compute root mean square (standard deviation from zero) of a tensor.

    Args:
        x: Input tensor
        eps: Small epsilon to avoid division by zero

    Returns:
        RMS value with shape (..., 1) preserving all dims except last
    """
    return mx.sqrt(mx.mean(x ** 2, axis=-1, keepdims=True) + eps)


def apply_cfg_rescale(
    guided_output: mx.array,
    cond_output: mx.array,
    rescale_phi: float = 0.7,
) -> mx.array:
    """
    Apply CFG rescaling to prevent over-saturation at high guidance scales.

    High CFG values (>7) can cause the output to become over-saturated,
    leading to artifacts. This rescaling technique normalizes the guided
    output to match the standard deviation of the conditional output.

    Reference: "Common Diffusion Noise Schedules and Sample Steps are Flawed"
    (https://arxiv.org/abs/2305.08891)

    Args:
        guided_output: Output after CFG has been applied
        cond_output: Original conditional output (before CFG)
        rescale_phi: Interpolation factor between rescaled and original
                     (0.0 = no rescaling, 1.0 = full rescaling)
                     Recommended: 0.7

    Returns:
        Rescaled guided output
    """
    if rescale_phi <= 0.0:
        return guided_output

    # Calculate RMS (standard deviation from zero) for both outputs
    std_cond = _compute_rms(cond_output)
    std_guided = _compute_rms(guided_output)

    # Rescale guided output to match conditional std
    rescaled = guided_output * (std_cond / std_guided)

    # Interpolate between rescaled and original guided output
    return rescale_phi * rescaled + (1 - rescale_phi) * guided_output


def apply_classifier_free_guidance(
    cond_logits: mx.array,
    uncond_logits: mx.array,
    guidance_scale: float,
    cond_beta_logits: Optional[mx.array] = None,
    guidance_scale_beta: float = 0.0,
    rescale_phi: float = 0.0,
) -> mx.array:
    """
    Apply classifier-free guidance to logits.

    Supports both single and double CFG:
    - Single CFG: uncond + scale * (cond - uncond)
    - Double CFG: uncond + scale * (cond - uncond) + scale_beta * (cond_beta - uncond)

    Args:
        cond_logits: Primary conditioned logits (e.g., text + melody)
        uncond_logits: Unconditioned logits (empty text)
        guidance_scale: Primary CFG scale (1.0 = no guidance)
        cond_beta_logits: Secondary conditioned logits (e.g., text only, no melody)
        guidance_scale_beta: Secondary CFG scale (0.0 = disabled)
        rescale_phi: CFG rescale factor to prevent over-saturation (0.0 = disabled,
                     0.7 recommended for high guidance scales >7)

    Returns:
        Guided logits
    """
    if guidance_scale == 1.0 and guidance_scale_beta == 0.0:
        return cond_logits

    # Standard single CFG
    result = uncond_logits + guidance_scale * (cond_logits - uncond_logits)

    # Double CFG: add secondary guidance if enabled
    if cond_beta_logits is not None and guidance_scale_beta > 0.0:
        result = result + guidance_scale_beta * (cond_beta_logits - uncond_logits)

    # Apply rescaling if enabled (recommended for guidance_scale > 7)
    if rescale_phi > 0.0:
        result = apply_cfg_rescale(result, cond_logits, rescale_phi)

    return result


def blend_overlapping_audio(
    prev_audio: np.ndarray,
    curr_audio: np.ndarray,
    overlap_samples: int,
    fade_samples: int,
) -> np.ndarray:
    """
    Blend two overlapping audio segments with linear crossfade.

    Args:
        prev_audio: Previous audio segment (..., samples)
        curr_audio: Current audio segment (..., samples), with overlap_samples overlap
        overlap_samples: Number of samples that overlap between segments
        fade_samples: Number of samples for crossfade (within overlap region)

    Returns:
        Blended audio with smooth transition
    """
    if overlap_samples <= 0:
        # No overlap, just concatenate
        return np.concatenate([prev_audio, curr_audio], axis=-1)

    # Ensure fade_samples doesn't exceed overlap
    fade_samples = min(fade_samples, overlap_samples)

    # Get the overlapping regions
    prev_overlap = prev_audio[..., -overlap_samples:]
    curr_overlap = curr_audio[..., :overlap_samples]

    # Create linear fade weights
    fade_out = np.linspace(1.0, 0.0, fade_samples, dtype=np.float32)
    fade_in = np.linspace(0.0, 1.0, fade_samples, dtype=np.float32)

    # Apply crossfade in the fade region
    blended_overlap = prev_overlap.copy()
    blended_overlap[..., :fade_samples] = (
        prev_overlap[..., :fade_samples] * fade_out
        + curr_overlap[..., :fade_samples] * fade_in
    )
    # After fade region, use current audio for rest of overlap
    blended_overlap[..., fade_samples:] = curr_overlap[..., fade_samples:]

    # Combine: prev (without overlap) + blended overlap + curr (after overlap)
    result = np.concatenate([
        prev_audio[..., :-overlap_samples],
        blended_overlap,
        curr_audio[..., overlap_samples:],
    ], axis=-1)

    return result


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
        prompt: Union[str, List[str]],
        duration: float = 10.0,
        temperature: float = 1.0,
        top_k: int = 250,
        top_p: float = 0.0,
        guidance_scale: float = 3.0,
        use_sampling: bool = True,
        seed: Optional[int] = None,
        callback: Optional[Callable[[int, int, mx.array], None]] = None,
        return_codes: bool = False,
    ) -> Union[GenerationOutput, List[GenerationOutput]]:
        """
        Generate music from text prompt(s).

        Supports batch generation when prompt is a list.

        Args:
            prompt: Text description(s) of desired music (str or list of str)
            duration: Target duration in seconds
            temperature: Sampling temperature
            top_k: Top-k filtering
            top_p: Nucleus sampling threshold
            guidance_scale: Classifier-free guidance scale
            use_sampling: If False, use greedy decoding (deterministic)
            seed: Random seed for reproducibility
            callback: Optional callback(step, total_steps, codes)
            return_codes: Whether to return raw audio codes

        Returns:
            GenerationOutput (single) or List[GenerationOutput] (batch)
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

        # Handle batch vs single prompt
        if isinstance(prompt, list):
            if len(prompt) == 0:
                raise ValueError("prompt list cannot be empty")
            batch_size = len(prompt)
            encoder_hidden_states, encoder_attention_mask = self.text_encoder.encode_batch(prompt)
        else:
            batch_size = 1
            encoder_hidden_states, encoder_attention_mask = self.text_encoder.encode(prompt)

        # For CFG, also need unconditional encoding (empty prompt)
        if guidance_scale > 1.0:
            if batch_size > 1:
                uncond_hidden_states, uncond_mask = self.text_encoder.encode_batch([""] * batch_size)
            else:
                uncond_hidden_states, uncond_mask = self.text_encoder.encode("")

        # Initialize with BOS tokens for all codebooks
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
                next_token = sample_next_token(cb_logits, temperature, top_k, top_p, use_sampling)
                next_tokens.append(next_token)

            next_tokens = mx.stack(next_tokens, axis=1)  # (batch, num_codebooks, 1)
            codes = mx.concatenate([codes, next_tokens], axis=-1)

            # Evaluate to prevent memory buildup (codes AND all KV caches)
            # PERF: Single batched eval is much faster than multiple sequential evals
            to_eval = [codes]
            if past_key_values is not None:
                for kv in past_key_values:
                    if kv is not None:
                        to_eval.extend(x for x in kv if x is not None)
            if cross_attn_past_key_values is not None:
                for kv in cross_attn_past_key_values:
                    if kv is not None:
                        to_eval.extend(x for x in kv if x is not None)
            if uncond_past_key_values is not None:
                for kv in uncond_past_key_values:
                    if kv is not None:
                        to_eval.extend(x for x in kv if x is not None)
            if uncond_cross_attn_past_key_values is not None:
                for kv in uncond_cross_attn_past_key_values:
                    if kv is not None:
                        to_eval.extend(x for x in kv if x is not None)
            mx.eval(*to_eval)

            # Callback
            if callback is not None:
                callback(step, target_length, codes)

        # Remove BOS token
        codes = codes[:, :, 1:]

        # Decode to audio
        audio = self.encodec.decode(codes)
        audio_np = np.array(audio)

        # Handle batch vs single output
        sample_rate = self.config.audio_encoder.sampling_rate
        if batch_size == 1:
            # Single output: remove batch dimension
            if audio_np.ndim == 3:
                audio_np = audio_np[0]
            return GenerationOutput(
                audio=audio_np,
                sample_rate=sample_rate,
                duration=duration,
                codes=codes if return_codes else None,
            )
        else:
            # Batch output: return list of GenerationOutput
            # audio_np is always (batch, channels, samples) for batch generation
            outputs = []
            for i in range(batch_size):
                batch_audio = audio_np[i]  # (channels, samples)
                batch_codes = codes[i:i+1] if return_codes else None
                outputs.append(GenerationOutput(
                    audio=batch_audio,
                    sample_rate=sample_rate,
                    duration=duration,
                    codes=batch_codes,
                ))
            return outputs

    def generate_continuation(
        self,
        audio: mx.array,
        prompt: str,
        duration: float = 10.0,
        temperature: float = 1.0,
        top_k: int = 250,
        top_p: float = 0.0,
        guidance_scale: float = 3.0,
        use_sampling: bool = True,
        seed: Optional[int] = None,
        callback: Optional[Callable[[int, int, mx.array], None]] = None,
        return_codes: bool = False,
    ) -> GenerationOutput:
        """
        Continue generation from existing audio.

        Args:
            audio: Existing audio to continue from (channels, samples) or (samples,)
            prompt: Text prompt for continuation style
            duration: Additional duration to generate in seconds
            temperature: Sampling temperature
            top_k: Top-k filtering
            top_p: Nucleus sampling threshold
            guidance_scale: Classifier-free guidance scale
            seed: Random seed for reproducibility
            callback: Optional progress callback
            return_codes: Whether to return raw audio codes

        Returns:
            GenerationOutput with continuation (original + generated audio)
        """
        # Validate inputs
        if duration <= 0:
            raise ValueError(f"duration must be positive, got {duration}")
        max_duration = 30.0
        if duration > max_duration:
            raise ValueError(f"duration must be <= {max_duration} seconds, got {duration}")

        if seed is not None:
            mx.random.seed(seed)

        # Ensure audio has correct shape for encoding: (batch, channels, samples)
        if audio.ndim == 1:
            audio = audio[None, None, :]  # (1, 1, samples)
        elif audio.ndim == 2:
            audio = audio[None, :, :]  # (1, channels, samples)

        # Encode existing audio to codes
        existing_codes, _ = self.encodec.encode(audio)  # (batch, num_codebooks, num_frames)

        # Calculate target length for new generation
        frame_rate = self.config.frame_rate
        target_new_length = max(1, int(duration * frame_rate))

        # Validate total duration (existing + new) doesn't exceed limit
        existing_frames = existing_codes.shape[-1]
        existing_duration = existing_frames / frame_rate
        total_duration_estimate = existing_duration + duration
        if total_duration_estimate > max_duration:
            raise ValueError(
                f"Total duration ({existing_duration:.1f}s existing + {duration:.1f}s new = "
                f"{total_duration_estimate:.1f}s) exceeds max {max_duration}s. "
                f"Use shorter existing audio or request less new duration."
            )

        # Encode text prompt
        encoder_hidden_states, encoder_attention_mask = self.text_encoder.encode(prompt)

        # For CFG, also need unconditional encoding
        if guidance_scale > 1.0:
            uncond_hidden_states, uncond_mask = self.text_encoder.encode("")

        # Prepend BOS token to existing codes
        batch_size = 1
        num_codebooks = self.config.decoder.num_codebooks
        bos_token = self.config.decoder.bos_token_id
        bos_tokens = mx.full((batch_size, num_codebooks, 1), bos_token, dtype=mx.int32)

        # Combine: BOS + existing codes
        codes = mx.concatenate([bos_tokens, existing_codes], axis=-1)

        # Evaluate codes to prevent memory buildup from encodec computation graph
        mx.eval(codes)

        # KV caches - need to process existing sequence first
        past_key_values = None
        cross_attn_past_key_values = None
        uncond_past_key_values = None
        uncond_cross_attn_past_key_values = None

        # First pass: process existing codes to build KV cache
        _, past_key_values, cross_attn_past_key_values = self.decoder(
            input_ids=codes,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            past_key_values=None,
            cross_attn_past_key_values=None,
            use_cache=True,
        )

        if guidance_scale > 1.0:
            _, uncond_past_key_values, uncond_cross_attn_past_key_values = self.decoder(
                input_ids=codes,
                encoder_hidden_states=uncond_hidden_states,
                encoder_attention_mask=uncond_mask,
                past_key_values=None,
                cross_attn_past_key_values=None,
                use_cache=True,
            )

        # Evaluate KV caches from first pass to prevent memory buildup
        to_eval = []
        if past_key_values is not None:
            for kv in past_key_values:
                if kv is not None:
                    to_eval.extend(x for x in kv if x is not None)
        if cross_attn_past_key_values is not None:
            for kv in cross_attn_past_key_values:
                if kv is not None:
                    to_eval.extend(x for x in kv if x is not None)
        if uncond_past_key_values is not None:
            for kv in uncond_past_key_values:
                if kv is not None:
                    to_eval.extend(x for x in kv if x is not None)
        if uncond_cross_attn_past_key_values is not None:
            for kv in uncond_cross_attn_past_key_values:
                if kv is not None:
                    to_eval.extend(x for x in kv if x is not None)
        if to_eval:
            mx.eval(*to_eval)

        # Generation loop for new tokens
        for step in range(target_new_length):
            # Get logits from decoder (conditional)
            logits, past_key_values, cross_attn_past_key_values = self.decoder(
                input_ids=codes[:, :, -1:],
                encoder_hidden_states=None,  # Use cached
                encoder_attention_mask=encoder_attention_mask,
                past_key_values=past_key_values,
                cross_attn_past_key_values=cross_attn_past_key_values,
                use_cache=True,
            )

            next_logits = logits[:, :, -1, :]

            # Apply CFG
            if guidance_scale > 1.0:
                uncond_logits, uncond_past_key_values, uncond_cross_attn_past_key_values = self.decoder(
                    input_ids=codes[:, :, -1:],
                    encoder_hidden_states=None,
                    encoder_attention_mask=uncond_mask,
                    past_key_values=uncond_past_key_values,
                    cross_attn_past_key_values=uncond_cross_attn_past_key_values,
                    use_cache=True,
                )
                uncond_next = uncond_logits[:, :, -1, :]
                next_logits = apply_classifier_free_guidance(
                    next_logits, uncond_next, guidance_scale
                )

            # Sample next tokens
            next_tokens = []
            for cb in range(num_codebooks):
                cb_logits = next_logits[:, cb, :]
                next_token = sample_next_token(cb_logits, temperature, top_k, top_p, use_sampling)
                next_tokens.append(next_token)

            next_tokens = mx.stack(next_tokens, axis=1)
            codes = mx.concatenate([codes, next_tokens], axis=-1)

            # Evaluate to prevent memory buildup
            to_eval = [codes]
            if past_key_values is not None:
                for kv in past_key_values:
                    if kv is not None:
                        to_eval.extend(x for x in kv if x is not None)
            if cross_attn_past_key_values is not None:
                for kv in cross_attn_past_key_values:
                    if kv is not None:
                        to_eval.extend(x for x in kv if x is not None)
            if uncond_past_key_values is not None:
                for kv in uncond_past_key_values:
                    if kv is not None:
                        to_eval.extend(x for x in kv if x is not None)
            if uncond_cross_attn_past_key_values is not None:
                for kv in uncond_cross_attn_past_key_values:
                    if kv is not None:
                        to_eval.extend(x for x in kv if x is not None)
            mx.eval(*to_eval)

            if callback is not None:
                callback(step, target_new_length, codes)

        # Remove BOS token
        codes = codes[:, :, 1:]

        # Decode full sequence to audio
        audio = self.encodec.decode(codes)
        audio_np = np.array(audio)

        if audio_np.ndim == 3:
            audio_np = audio_np[0]

        # Calculate total duration
        existing_frames = existing_codes.shape[-1]
        total_frames = codes.shape[-1]
        total_duration = total_frames / frame_rate

        return GenerationOutput(
            audio=audio_np,
            sample_rate=self.config.audio_encoder.sampling_rate,
            duration=total_duration,
            codes=codes if return_codes else None,
        )

    def generate_with_melody(
        self,
        prompt: str,
        melody_audio: mx.array,
        melody_conditioner,
        duration: float = 10.0,
        temperature: float = 1.0,
        top_k: int = 250,
        top_p: float = 0.0,
        guidance_scale: float = 3.0,
        guidance_scale_beta: float = 0.0,
        use_sampling: bool = True,
        seed: Optional[int] = None,
        callback: Optional[Callable[[int, int, mx.array], None]] = None,
        return_codes: bool = False,
    ) -> GenerationOutput:
        """
        Generate music with melody conditioning.

        Uses chroma features extracted from reference audio to guide generation.
        The model attends to both text embeddings (for style) and chroma embeddings
        (for melody) simultaneously via cross-attention.

        Supports double CFG when guidance_scale_beta > 0:
        - Conditional path: text + melody
        - Beta path: text only (no melody)
        - Unconditional path: empty text

        Args:
            prompt: Text description of desired music
            melody_audio: Reference audio for melody extraction (samples,) or (channels, samples)
            melody_conditioner: MelodyConditioner instance with projection layer
            duration: Target duration in seconds
            temperature: Sampling temperature
            top_k: Top-k filtering
            top_p: Nucleus sampling threshold
            guidance_scale: Primary CFG scale (text+melody vs unconditional)
            guidance_scale_beta: Secondary CFG scale (text-only vs unconditional), 0 = disabled
            use_sampling: If False, use greedy decoding
            seed: Random seed for reproducibility
            callback: Optional progress callback
            return_codes: Whether to return raw audio codes

        Returns:
            GenerationOutput with melody-conditioned audio
        """
        # Validate inputs
        if duration <= 0:
            raise ValueError(f"duration must be positive, got {duration}")
        max_duration = 30.0
        if duration > max_duration:
            raise ValueError(f"duration must be <= {max_duration} seconds, got {duration}")

        if seed is not None:
            mx.random.seed(seed)

        # Normalize melody_audio shape (like generate_continuation does)
        if melody_audio.ndim == 1:
            melody_audio = melody_audio[None, :]  # (1, samples)
        elif melody_audio.ndim == 3:
            # Remove batch dimension if present
            melody_audio = melody_audio[0]  # (channels, samples)

        # Calculate target length
        frame_rate = self.config.frame_rate
        target_length = max(1, int(duration * frame_rate))

        # Extract and project chroma embeddings from melody audio
        # Pre-compute ALL chroma embeddings so model can attend to entire melody
        chroma_embeddings = melody_conditioner.get_chroma_embeddings(
            melody_audio, target_length
        )  # (1, target_length, hidden_size)

        # Evaluate chroma embeddings to prevent memory buildup
        mx.eval(chroma_embeddings)

        # Encode text prompt
        encoder_hidden_states, encoder_attention_mask = self.text_encoder.encode(prompt)

        # Combine text embeddings with ALL chroma embeddings for cross-attention
        # This allows the model to attend to relevant melody frames at each step
        combined_hidden_states = mx.concatenate(
            [encoder_hidden_states, chroma_embeddings], axis=1
        )
        text_len = encoder_attention_mask.shape[1]
        chroma_len = chroma_embeddings.shape[1]
        combined_mask = mx.concatenate(
            [encoder_attention_mask, mx.ones((1, chroma_len))], axis=1
        )

        # Evaluate combined embeddings to prevent memory buildup
        mx.eval(combined_hidden_states, combined_mask)

        # For CFG, need unconditional encoding (empty text)
        uncond_hidden_states = None
        uncond_mask = None
        if guidance_scale > 1.0 or guidance_scale_beta > 0.0:
            uncond_hidden_states, uncond_mask = self.text_encoder.encode("")

        # For double CFG (beta path), text-only conditioning (no melody)
        # This is just the original text encoder output without chroma
        beta_hidden_states = None
        beta_mask = None
        if guidance_scale_beta > 0.0:
            beta_hidden_states = encoder_hidden_states
            beta_mask = encoder_attention_mask

        # Initialize with BOS tokens
        batch_size = 1
        num_codebooks = self.config.decoder.num_codebooks
        bos_token = self.config.decoder.bos_token_id
        codes = mx.full((batch_size, num_codebooks, 1), bos_token, dtype=mx.int32)

        # KV caches
        past_key_values = None
        cross_attn_past_key_values = None
        uncond_past_key_values = None
        uncond_cross_attn_past_key_values = None
        beta_past_key_values = None
        beta_cross_attn_past_key_values = None

        # Generation loop with melody conditioning
        for step in range(target_length):
            # Decoder attends to combined text + melody embeddings
            logits, past_key_values, cross_attn_past_key_values = self.decoder(
                input_ids=codes if past_key_values is None else codes[:, :, -1:],
                encoder_hidden_states=combined_hidden_states if cross_attn_past_key_values is None else None,
                encoder_attention_mask=combined_mask,
                past_key_values=past_key_values,
                cross_attn_past_key_values=cross_attn_past_key_values,
                use_cache=True,
            )

            next_logits = logits[:, :, -1, :]

            # Apply CFG (unconditional path and optional beta path)
            if guidance_scale > 1.0 or guidance_scale_beta > 0.0:
                # Unconditional path (empty text)
                uncond_logits, uncond_past_key_values, uncond_cross_attn_past_key_values = self.decoder(
                    input_ids=codes if uncond_past_key_values is None else codes[:, :, -1:],
                    encoder_hidden_states=uncond_hidden_states if uncond_cross_attn_past_key_values is None else None,
                    encoder_attention_mask=uncond_mask,
                    past_key_values=uncond_past_key_values,
                    cross_attn_past_key_values=uncond_cross_attn_past_key_values,
                    use_cache=True,
                )
                uncond_next = uncond_logits[:, :, -1, :]

                # Beta path (text-only, no melody) for double CFG
                beta_next = None
                if guidance_scale_beta > 0.0:
                    beta_logits, beta_past_key_values, beta_cross_attn_past_key_values = self.decoder(
                        input_ids=codes if beta_past_key_values is None else codes[:, :, -1:],
                        encoder_hidden_states=beta_hidden_states if beta_cross_attn_past_key_values is None else None,
                        encoder_attention_mask=beta_mask,
                        past_key_values=beta_past_key_values,
                        cross_attn_past_key_values=beta_cross_attn_past_key_values,
                        use_cache=True,
                    )
                    beta_next = beta_logits[:, :, -1, :]

                next_logits = apply_classifier_free_guidance(
                    next_logits, uncond_next, guidance_scale,
                    cond_beta_logits=beta_next,
                    guidance_scale_beta=guidance_scale_beta,
                )

            # Sample next tokens
            next_tokens = []
            for cb in range(num_codebooks):
                cb_logits = next_logits[:, cb, :]
                next_token = sample_next_token(cb_logits, temperature, top_k, top_p, use_sampling)
                next_tokens.append(next_token)

            next_tokens = mx.stack(next_tokens, axis=1)
            codes = mx.concatenate([codes, next_tokens], axis=-1)

            # Evaluate to prevent memory buildup
            to_eval = [codes]
            if past_key_values is not None:
                for kv in past_key_values:
                    if kv is not None:
                        to_eval.extend(x for x in kv if x is not None)
            if cross_attn_past_key_values is not None:
                for kv in cross_attn_past_key_values:
                    if kv is not None:
                        to_eval.extend(x for x in kv if x is not None)
            if uncond_past_key_values is not None:
                for kv in uncond_past_key_values:
                    if kv is not None:
                        to_eval.extend(x for x in kv if x is not None)
            if uncond_cross_attn_past_key_values is not None:
                for kv in uncond_cross_attn_past_key_values:
                    if kv is not None:
                        to_eval.extend(x for x in kv if x is not None)
            if beta_past_key_values is not None:
                for kv in beta_past_key_values:
                    if kv is not None:
                        to_eval.extend(x for x in kv if x is not None)
            if beta_cross_attn_past_key_values is not None:
                for kv in beta_cross_attn_past_key_values:
                    if kv is not None:
                        to_eval.extend(x for x in kv if x is not None)
            mx.eval(*to_eval)

            if callback is not None:
                callback(step, target_length, codes)

        # Remove BOS token
        codes = codes[:, :, 1:]

        # Decode to audio
        audio = self.encodec.decode(codes)
        audio_np = np.array(audio)

        if audio_np.ndim == 3:
            audio_np = audio_np[0]

        return GenerationOutput(
            audio=audio_np,
            sample_rate=self.config.audio_encoder.sampling_rate,
            duration=duration,
            codes=codes if return_codes else None,
        )

    def generate_extended(
        self,
        prompt: str,
        duration: float,
        extend_stride: int = 750,
        window_length: int = 1500,
        fade_duration: float = 0.5,
        temperature: float = 1.0,
        top_k: int = 250,
        top_p: float = 0.0,
        guidance_scale: float = 3.0,
        use_sampling: bool = True,
        seed: Optional[int] = None,
        callback: Optional[Callable[[int, int, mx.array], None]] = None,
    ) -> GenerationOutput:
        """
        Generate music longer than 30 seconds using extend_stride pattern.

        Generates in overlapping windows with crossfade blending for seamless
        long-form audio. Each window after the first reuses the end of the
        previous window as context.

        Args:
            prompt: Text description of desired music
            duration: Target duration in seconds (can exceed 30s)
            extend_stride: Tokens to generate per extension (~15s = 750 at 50fps)
            window_length: Max tokens per window (~30s = 1500 at 50fps)
            fade_duration: Crossfade duration in seconds for blending
            temperature: Sampling temperature
            top_k: Top-k filtering
            top_p: Nucleus sampling threshold
            guidance_scale: Classifier-free guidance scale
            use_sampling: If False, use greedy decoding
            seed: Random seed for reproducibility
            callback: Optional progress callback(step, total_steps, codes)

        Returns:
            GenerationOutput with seamlessly blended long audio
        """
        if seed is not None:
            mx.random.seed(seed)

        # Validate parameters
        if duration <= 0:
            raise ValueError(f"duration must be positive, got {duration}")
        if extend_stride <= 0:
            raise ValueError(f"extend_stride must be positive, got {extend_stride}")
        if window_length <= extend_stride:
            raise ValueError(
                f"window_length ({window_length}) must be > extend_stride ({extend_stride})"
            )

        frame_rate = self.config.frame_rate
        sample_rate = self.config.audio_encoder.sampling_rate
        samples_per_frame = sample_rate // frame_rate  # typically 640 at 32kHz/50fps

        # For short durations (<= 30s), just use regular generate
        max_single_window = 30.0
        if duration <= max_single_window:
            return self.generate(
                prompt=prompt,
                duration=duration,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                guidance_scale=guidance_scale,
                use_sampling=use_sampling,
                seed=None,  # Already set above
                callback=callback,
            )

        # Calculate total tokens needed
        total_tokens = int(duration * frame_rate)
        overlap_tokens = window_length - extend_stride
        overlap_samples = overlap_tokens * samples_per_frame
        fade_samples = int(fade_duration * sample_rate)

        # Encode text prompt once (shared across all windows)
        encoder_hidden_states, encoder_attention_mask = self.text_encoder.encode(prompt)

        # For CFG
        uncond_hidden_states = None
        uncond_mask = None
        if guidance_scale > 1.0:
            uncond_hidden_states, uncond_mask = self.text_encoder.encode("")

        # Initialize tracking
        batch_size = 1
        num_codebooks = self.config.decoder.num_codebooks
        bos_token = self.config.decoder.bos_token_id

        all_audio = None
        tokens_generated = 0
        window_idx = 0
        total_callback_steps = total_tokens

        while tokens_generated < total_tokens:
            # Determine how many tokens to generate this window
            remaining = total_tokens - tokens_generated

            if window_idx == 0:
                # First window: generate up to window_length tokens
                window_tokens = min(window_length, remaining)
                # Start with BOS
                codes = mx.full((batch_size, num_codebooks, 1), bos_token, dtype=mx.int32)
                # Fresh caches
                past_key_values = None
                cross_attn_past_key_values = None
                uncond_past_key_values = None
                uncond_cross_attn_past_key_values = None
            else:
                # Subsequent windows: keep overlap, generate extend_stride tokens
                window_tokens = min(extend_stride, remaining)

                # Get the last overlap_tokens from previous codes (without BOS)
                # prev_codes is (batch, codebooks, seq) without BOS
                overlap_codes = prev_codes[:, :, -overlap_tokens:]

                # Prepend BOS and use as context
                codes = mx.concatenate([
                    mx.full((batch_size, num_codebooks, 1), bos_token, dtype=mx.int32),
                    overlap_codes,
                ], axis=-1)

                mx.eval(codes)

                # Build KV cache from overlap context
                _, past_key_values, cross_attn_past_key_values = self.decoder(
                    input_ids=codes,
                    encoder_hidden_states=encoder_hidden_states,
                    encoder_attention_mask=encoder_attention_mask,
                    past_key_values=None,
                    cross_attn_past_key_values=None,
                    use_cache=True,
                )

                if guidance_scale > 1.0:
                    _, uncond_past_key_values, uncond_cross_attn_past_key_values = self.decoder(
                        input_ids=codes,
                        encoder_hidden_states=uncond_hidden_states,
                        encoder_attention_mask=uncond_mask,
                        past_key_values=None,
                        cross_attn_past_key_values=None,
                        use_cache=True,
                    )

                # Evaluate caches
                to_eval = []
                for cache_list in [past_key_values, cross_attn_past_key_values,
                                   uncond_past_key_values, uncond_cross_attn_past_key_values]:
                    if cache_list is not None:
                        for kv in cache_list:
                            if kv is not None:
                                to_eval.extend(x for x in kv if x is not None)
                if to_eval:
                    mx.eval(*to_eval)

            # Generation loop for this window
            for step in range(window_tokens):
                # Conditional path
                logits, past_key_values, cross_attn_past_key_values = self.decoder(
                    input_ids=codes if past_key_values is None else codes[:, :, -1:],
                    encoder_hidden_states=encoder_hidden_states if cross_attn_past_key_values is None else None,
                    encoder_attention_mask=encoder_attention_mask,
                    past_key_values=past_key_values,
                    cross_attn_past_key_values=cross_attn_past_key_values,
                    use_cache=True,
                )

                next_logits = logits[:, :, -1, :]

                # CFG
                if guidance_scale > 1.0:
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

                # Sample next tokens
                next_tokens = []
                for cb in range(num_codebooks):
                    cb_logits = next_logits[:, cb, :]
                    next_token = sample_next_token(cb_logits, temperature, top_k, top_p, use_sampling)
                    next_tokens.append(next_token)

                next_tokens = mx.stack(next_tokens, axis=1)
                codes = mx.concatenate([codes, next_tokens], axis=-1)

                # Evaluate to prevent memory buildup
                to_eval = [codes]
                for cache_list in [past_key_values, cross_attn_past_key_values,
                                   uncond_past_key_values, uncond_cross_attn_past_key_values]:
                    if cache_list is not None:
                        for kv in cache_list:
                            if kv is not None:
                                to_eval.extend(x for x in kv if x is not None)
                mx.eval(*to_eval)

                # Callback with overall progress
                if callback is not None:
                    overall_step = tokens_generated + step
                    callback(overall_step, total_callback_steps, codes)

            # Remove BOS from this window's codes
            window_codes = codes[:, :, 1:]  # (batch, codebooks, window_tokens + overlap if not first)

            # Decode this window to audio
            window_audio = self.encodec.decode(window_codes)
            window_audio_np = np.array(window_audio)
            if window_audio_np.ndim == 3:
                window_audio_np = window_audio_np[0]  # (channels, samples)

            # Blend with previous audio
            if all_audio is None:
                all_audio = window_audio_np
            else:
                # Subsequent windows include overlap from previous
                all_audio = blend_overlapping_audio(
                    all_audio, window_audio_np, overlap_samples, fade_samples
                )

            # Save codes for next window's context (without BOS)
            prev_codes = window_codes

            # Update tracking - count new tokens only (window_tokens excludes overlap)
            tokens_generated += window_tokens
            window_idx += 1

            # Clear old caches to free memory
            past_key_values = None
            cross_attn_past_key_values = None
            uncond_past_key_values = None
            uncond_cross_attn_past_key_values = None

        # Calculate actual duration from audio length
        actual_duration = all_audio.shape[-1] / sample_rate

        return GenerationOutput(
            audio=all_audio,
            sample_rate=sample_rate,
            duration=actual_duration,
            codes=None,  # Not meaningful for extended generation
        )


__all__ = [
    "GenerationConfig",
    "GenerationOutput",
    "top_k_filtering",
    "top_p_filtering",
    "sample_next_token",
    "apply_cfg_rescale",
    "apply_classifier_free_guidance",
    "blend_overlapping_audio",
    "MusicGenGenerator",
]
