"""
ACE-Step main model class.

Provides a high-level interface for loading and generating
music with the ACE-Step model.
"""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import mlx.core as mx
import mlx.nn as nn
import numpy as np

logger = logging.getLogger(__name__)

from mlx_music.models.ace_step.dcae import DCAE, DCAEConfig
from mlx_music.models.ace_step.lyric_tokenizer import VoiceBpeTokenizer
from mlx_music.models.ace_step.scheduler import (
    FlowMatchEulerDiscreteScheduler,
    get_scheduler,
    retrieve_timesteps,
)
from mlx_music.models.ace_step.text_encoder import get_text_encoder
from mlx_music.models.ace_step.transformer import ACEStepConfig, ACEStepTransformer
from mlx_music.models.ace_step.vocoder import HiFiGANVocoder, MusicDCAEPipeline
from mlx_music.weights.weight_loader import (
    download_model,
    load_ace_step_weights,
)


@dataclass
class GenerationConfig:
    """Configuration for audio generation."""

    # Duration and quality
    duration: float = 30.0  # seconds
    sample_rate: int = 44100

    # Diffusion parameters
    num_inference_steps: int = 60
    guidance_scale: float = 15.0
    guidance_scale_text: float = 5.0
    guidance_scale_lyric: float = 2.5

    # Scheduler
    scheduler_type: str = "euler"
    shift: float = 3.0

    # Generation
    seed: Optional[int] = None
    batch_size: int = 1


@dataclass
class GenerationOutput:
    """Output from music generation."""

    audio: np.ndarray  # Shape: (channels, samples) or (samples,)
    sample_rate: int
    duration: float
    latents: Optional[mx.array] = None


class ACEStep:
    """
    ACE-Step Music Generation Model.

    High-level interface for loading and generating music.

    Example:
        >>> model = ACEStep.from_pretrained("ACE-Step/ACE-Step-v1-3.5B")
        >>> output = model.generate(
        ...     prompt="upbeat electronic dance music",
        ...     lyrics="Verse 1: Dancing through the night...",
        ...     duration=30.0
        ... )
        >>> # Save audio
        >>> import soundfile as sf
        >>> sf.write("output.wav", output.audio.T, output.sample_rate)
    """

    def __init__(
        self,
        transformer: ACEStepTransformer,
        config: ACEStepConfig,
        text_encoder: Optional[Any] = None,
        audio_pipeline: Optional[MusicDCAEPipeline] = None,
        lyric_tokenizer: Optional[VoiceBpeTokenizer] = None,
        dtype: mx.Dtype = mx.bfloat16,
    ):
        self.transformer = transformer
        self.config = config
        self.text_encoder = text_encoder
        self.audio_pipeline = audio_pipeline
        self.lyric_tokenizer = lyric_tokenizer
        self.dtype = dtype

        # Default scheduler
        self.scheduler = FlowMatchEulerDiscreteScheduler(
            num_train_timesteps=1000,
            shift=3.0,
        )

    @classmethod
    def from_pretrained(
        cls,
        model_path: Union[str, Path],
        dtype: mx.Dtype = mx.bfloat16,
        load_text_encoder: bool = True,
        load_audio_pipeline: bool = True,
        strict_components: bool = True,
    ) -> "ACEStep":
        """
        Load ACE-Step from pretrained weights.

        Supports both local paths and HuggingFace Hub repository IDs.

        Args:
            model_path: Path to model directory or HuggingFace repo ID
                Examples:
                - "/path/to/local/model" (local path)
                - "ACE-Step/ACE-Step-v1-3.5B" (HuggingFace repo)
            dtype: Data type for model weights
            load_text_encoder: Whether to load text encoder
            load_audio_pipeline: Whether to load DCAE + vocoder (needed for audio output)
            strict_components: If True (default), raise error if text encoder
                or audio pipeline fails to load. If False, fall back to placeholders.

        Returns:
            ACEStep instance

        Raises:
            RuntimeError: If strict_components=True and components fail to load
        """
        # Resolve path (handles both local paths and HuggingFace repo IDs)
        model_path = download_model(str(model_path))

        # Load transformer weights and config
        logger.info("Loading transformer...")
        weights, config_dict = load_ace_step_weights(
            model_path, component="transformer", dtype=dtype
        )

        # Create config
        config = ACEStepConfig.from_dict(config_dict)

        # Create transformer
        transformer = ACEStepTransformer(config)

        # Load weights into transformer (strict=False allows for extra weights like lyric_encoder)
        transformer.load_weights(list(weights.items()), strict=False)

        # Initialize additional components
        text_encoder = None
        audio_pipeline = None

        if load_text_encoder:
            logger.info("Loading text encoder (UMT5)...")
            try:
                # Use shared device detection utility
                from mlx_music.utils.device import get_default_torch_device
                device = get_default_torch_device()

                text_encoder = get_text_encoder(
                    model_path=model_path,
                    device=device,
                    use_fp16=(dtype == mx.float16),
                )
                # Check if we got a placeholder
                from mlx_music.models.ace_step.text_encoder import PlaceholderTextEncoder
                if isinstance(text_encoder, PlaceholderTextEncoder):
                    if strict_components:
                        raise RuntimeError(
                            "Text encoder failed to load (got placeholder). "
                            "Install with: pip install 'mlx-music[text-encoder]' "
                            "or use strict_components=False to allow placeholder."
                        )
                    logger.warning("Using placeholder encoder (generation will have limited quality)")
                else:
                    logger.info(f"Text encoder loaded successfully on {device}!")
            except (KeyboardInterrupt, SystemExit):
                raise
            except RuntimeError:
                # Re-raise RuntimeError from strict_components check
                raise
            except Exception as e:
                if strict_components:
                    raise RuntimeError(
                        f"Text encoder failed to load: {e}. "
                        "Use strict_components=False to allow placeholder."
                    ) from e
                logger.warning(f"Could not load text encoder: {e}")
                logger.warning("Using placeholder encoder (generation will have limited quality)")

        if load_audio_pipeline:
            logger.info("Loading audio pipeline (DCAE + vocoder)...")
            try:
                audio_pipeline = MusicDCAEPipeline.from_pretrained(str(model_path), dtype=dtype)
                logger.info("Audio pipeline loaded successfully!")
            except (KeyboardInterrupt, SystemExit):
                raise
            except Exception as e:
                if strict_components:
                    raise RuntimeError(
                        f"Audio pipeline failed to load: {e}. "
                        "Use strict_components=False to allow placeholder."
                    ) from e
                logger.warning(f"Could not load audio pipeline: {e}")
                logger.warning("Audio decoding will use placeholder (silence)")

        # Load lyric tokenizer
        lyric_tokenizer = None
        logger.info("Loading lyric tokenizer...")
        try:
            lyric_tokenizer = VoiceBpeTokenizer()
            logger.info(f"Lyric tokenizer loaded ({len(lyric_tokenizer)} tokens)")
        except Exception as e:
            logger.warning(f"Could not load lyric tokenizer: {e}")
            logger.warning("Lyric encoding will use placeholder tokens")

        return cls(
            transformer=transformer,
            config=config,
            text_encoder=text_encoder,
            audio_pipeline=audio_pipeline,
            lyric_tokenizer=lyric_tokenizer,
            dtype=dtype,
        )

    def encode_text(
        self,
        prompt: str,
    ) -> Tuple[mx.array, mx.array]:
        """
        Encode text prompt to embeddings.

        Args:
            prompt: Text prompt

        Returns:
            Tuple of (embeddings, attention_mask)
        """
        if self.text_encoder is None:
            # Return placeholder embeddings
            # Shape: (batch, seq_len, dim)
            embeddings = mx.zeros((1, 64, self.config.text_embedding_dim))
            mask = mx.ones((1, 64))
            return embeddings, mask

        # Encode using UMT5
        embeddings, mask = self.text_encoder.encode(prompt)

        # Cast to model dtype
        embeddings = embeddings.astype(self.dtype)

        return embeddings, mask

    def encode_lyrics(
        self,
        lyrics: str,
        lang: str = "en",
        max_length: int = 512,
    ) -> Tuple[mx.array, mx.array]:
        """
        Encode lyrics to token indices.

        Args:
            lyrics: Lyrics text
            lang: Language code (en, zh, ko, etc.)
            max_length: Maximum token length (will pad/truncate)

        Returns:
            Tuple of (token_indices, attention_mask)
        """
        if self.lyric_tokenizer is None:
            # Return placeholder tokens if tokenizer not available
            # Mask is all zeros to indicate no valid tokens
            tokens = mx.zeros((1, max_length), dtype=mx.int32)
            mask = mx.zeros((1, max_length))
            return tokens, mask

        # Tokenize lyrics
        token_ids = self.lyric_tokenizer.encode(lyrics, lang=lang)

        # Pad or truncate to max_length
        if len(token_ids) > max_length:
            logger.warning(
                f"Lyrics truncated from {len(token_ids)} to {max_length} tokens. "
                "Some lyrics may not be included in generation."
            )
            token_ids = token_ids[:max_length]
            actual_length = max_length
        else:
            actual_length = len(token_ids)
            # Pad with zeros
            token_ids = token_ids + [0] * (max_length - len(token_ids))

        # Create attention mask (1 for real tokens, 0 for padding)
        mask = [1.0] * actual_length + [0.0] * (max_length - actual_length)

        # Convert to MLX arrays
        tokens = mx.array([token_ids], dtype=mx.int32)
        mask = mx.array([mask])

        return tokens, mask

    def decode_latents(
        self,
        latents: mx.array,
    ) -> np.ndarray:
        """
        Decode latents to audio waveform.

        Args:
            latents: Audio latents from diffusion

        Returns:
            Audio waveform as numpy array
        """
        if self.audio_pipeline is None:
            # Return placeholder audio (silence)
            duration = latents.shape[-1] * 512 * 8 / 44100  # Approximate
            samples = int(duration * 44100)
            return np.zeros((2, samples), dtype=np.float32)

        # Decode using DCAE + vocoder pipeline
        audio = self.audio_pipeline.decode(latents)

        # Convert to numpy
        return np.array(audio, dtype=np.float32)

    def _transformer_forward(
        self,
        latents: mx.array,
        timestep: mx.array,
        text_embeds: mx.array,
        text_mask: mx.array,
        speaker_embeds: Optional[mx.array],
        lyric_tokens: Optional[mx.array],
        lyric_mask: Optional[mx.array],
    ) -> mx.array:
        """Transformer forward pass.

        Note: @mx.compile removed due to issues with Optional parameters.
        """
        return self.transformer(
            hidden_states=latents,
            timestep=timestep,
            encoder_text_hidden_states=text_embeds,
            text_attention_mask=text_mask,
            speaker_embeds=speaker_embeds,
            lyric_token_idx=lyric_tokens,
            lyric_mask=lyric_mask,
        )

    # Duration constraints (in seconds)
    MIN_DURATION = 1.0  # ~1 second minimum
    MAX_DURATION = 240.0  # ~4 minutes maximum

    # Latent dimension constraints
    MAX_LATENT_WIDTH = 3000  # Based on architecture limits

    def generate(
        self,
        prompt: str,
        lyrics: Optional[str] = None,
        duration: float = 30.0,
        num_inference_steps: int = 60,
        guidance_scale: float = 15.0,
        seed: Optional[int] = None,
        speaker_embeds: Optional[mx.array] = None,
        return_latents: bool = False,
        scheduler_type: str = "euler",
        callback: Optional[callable] = None,
    ) -> GenerationOutput:
        """
        Generate music from text prompt.

        Args:
            prompt: Text description of desired music
            lyrics: Optional lyrics for vocal generation
            duration: Duration in seconds (1-240 seconds, ~4 minutes max)
            num_inference_steps: Number of diffusion steps
            guidance_scale: CFG guidance scale
            seed: Random seed for reproducibility
            speaker_embeds: Optional speaker embedding for voice cloning
            return_latents: Whether to return intermediate latents
            scheduler_type: "euler" or "heun"
            callback: Optional callback(step, timestep, latents)

        Returns:
            GenerationOutput with audio and metadata

        Raises:
            ValueError: If duration or other parameters are outside valid range
        """
        # Validate num_inference_steps
        if num_inference_steps <= 0:
            raise ValueError(f"num_inference_steps must be positive, got {num_inference_steps}")
        if num_inference_steps > 1000:
            raise ValueError(
                f"num_inference_steps={num_inference_steps} is extremely high (max recommended: 1000). "
                "This will be very slow and may not improve quality."
            )

        # Validate guidance_scale
        if guidance_scale < 0:
            raise ValueError(f"guidance_scale must be >= 0, got {guidance_scale}")

        # Validate callback
        if callback is not None and not callable(callback):
            raise TypeError(f"callback must be callable, got {type(callback).__name__}")

        # Validate duration
        if duration < self.MIN_DURATION:
            raise ValueError(
                f"Duration {duration}s is too short. "
                f"Minimum duration is {self.MIN_DURATION}s"
            )
        if duration > self.MAX_DURATION:
            raise ValueError(
                f"Duration {duration}s is too long. "
                f"Maximum duration is {self.MAX_DURATION}s (~{self.MAX_DURATION / 60:.0f} minutes)"
            )

        # Set seed if provided
        if seed is not None:
            mx.random.seed(seed)

        # Calculate latent dimensions
        # Audio: duration * 44100 samples
        # Latent: 8 channels, 16 height, variable width
        # Width = duration * 44100 / (512 * 8) ≈ duration * 10.77
        latent_width = int(duration * 44100 / (512 * 8))

        # Validate latent dimensions
        if latent_width > self.MAX_LATENT_WIDTH:
            max_safe_duration = self.MAX_LATENT_WIDTH * 512 * 8 / 44100
            raise ValueError(
                f"Duration {duration}s produces latent width {latent_width}, "
                f"which exceeds maximum {self.MAX_LATENT_WIDTH}. "
                f"Maximum safe duration is ~{max_safe_duration:.1f}s"
            )

        latent_shape = (1, self.config.in_channels, self.config.max_height, latent_width)

        # Initialize noise
        latents = mx.random.normal(latent_shape).astype(self.dtype)

        # Encode text
        text_embeds, text_mask = self.encode_text(prompt)

        # Encode lyrics if provided
        if lyrics is not None:
            lyric_tokens, lyric_mask = self.encode_lyrics(lyrics)
        else:
            lyric_tokens, lyric_mask = None, None

        # Get scheduler
        scheduler = get_scheduler(scheduler_type, shift=3.0)
        timesteps, _ = retrieve_timesteps(scheduler, num_inference_steps)

        # Diffusion loop
        for i, t in enumerate(timesteps):
            # Expand timestep for batch
            timestep = mx.array([t] * latents.shape[0])

            # Model prediction (conditional)
            noise_pred = self._transformer_forward(
                latents,
                timestep,
                text_embeds,
                text_mask,
                speaker_embeds,
                lyric_tokens,
                lyric_mask,
            )

            # Classifier-free guidance (skip only when guidance_scale == 1.0)
            if guidance_scale != 1.0:
                # Evaluate conditional prediction first to free intermediate tensors
                mx.eval(noise_pred)
                # Get unconditional prediction
                noise_pred_uncond = self._transformer_forward(
                    latents,
                    timestep,
                    mx.zeros_like(text_embeds),
                    mx.zeros_like(text_mask),
                    None,
                    None,
                    None,
                )
                # Force evaluation to free intermediate tensors before CFG combination
                mx.eval(noise_pred_uncond)

                # CFG combination
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred - noise_pred_uncond)

                # Evaluate combined prediction to free memory
                mx.eval(noise_pred)

            # Scheduler step
            output = scheduler.step(noise_pred, t, latents)
            latents = output.prev_sample

            # Callback
            if callback is not None:
                callback(i, t, latents)

            # Evaluate for progress and memory management
            mx.eval(latents)

        # Decode to audio
        audio = self.decode_latents(latents)

        return GenerationOutput(
            audio=audio,
            sample_rate=44100,
            duration=duration,
            latents=latents if return_latents else None,
        )

    def __repr__(self) -> str:
        return (
            f"ACEStep(\n"
            f"  config={self.config},\n"
            f"  dtype={self.dtype},\n"
            f"  has_text_encoder={self.text_encoder is not None},\n"
            f"  has_audio_pipeline={self.audio_pipeline is not None},\n"
            f"  has_lyric_tokenizer={self.lyric_tokenizer is not None},\n"
            f")"
        )
