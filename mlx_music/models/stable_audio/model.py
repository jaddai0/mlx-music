"""
Stable Audio Open main model class.

Provides a high-level interface for loading and generating music
with the Stable Audio Open model.
"""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Tuple, Union

import mlx.core as mx
import numpy as np

from mlx_music.models.stable_audio.config import (
    DiTConfig,
    StableAudioConfig,
    VAEConfig,
    ProjectionConfig,
)
from mlx_music.models.stable_audio.conditioning import (
    ProjectionModel,
)
from mlx_music.models.stable_audio.scheduler import (
    EDMDPMSolverMultistepScheduler,
    retrieve_timesteps,
)
from mlx_music.models.stable_audio.transformer import StableAudioDiT
from mlx_music.models.stable_audio.vae import AutoencoderOobleck

logger = logging.getLogger(__name__)


@dataclass
class GenerationOutput:
    """Output from audio generation."""

    audio: np.ndarray  # Shape: (channels, samples)
    sample_rate: int
    duration: float
    latents: Optional[mx.array] = None


class StableAudio:
    """
    Stable Audio Open Model.

    High-level interface for loading and generating music with
    Stability AI's Stable Audio Open model.

    Example:
        >>> model = StableAudio.from_pretrained("stabilityai/stable-audio-open-1.0")
        >>> output = model.generate(
        ...     prompt="ambient electronic music with soft pads",
        ...     duration=30.0,
        ...     guidance_scale=7.0,
        ... )
        >>> # Save audio
        >>> import soundfile as sf
        >>> sf.write("output.wav", output.audio.T, output.sample_rate)
    """

    # Duration constraints (architectural limits from VAE latent sequence length)
    MIN_DURATION = 1.0  # Minimum practical duration in seconds
    MAX_DURATION = 47.0  # Maximum duration in seconds (limited by VAE training context)

    def __init__(
        self,
        transformer: StableAudioDiT,
        vae: AutoencoderOobleck,
        projection_model: ProjectionModel,
        config: StableAudioConfig,
        text_encoder: Optional[Any] = None,
        dtype: mx.Dtype = mx.float32,
    ):
        """
        Initialize Stable Audio model.

        Args:
            transformer: StableAudioDiT transformer
            vae: AutoencoderOobleck VAE
            projection_model: ProjectionModel for conditioning
            config: Model configuration
            text_encoder: Optional T5 text encoder
            dtype: Data type for computations
        """
        self.transformer = transformer
        self.vae = vae
        self.projection_model = projection_model
        self.config = config
        self.text_encoder = text_encoder
        self.dtype = dtype

        # Initialize scheduler
        self.scheduler = EDMDPMSolverMultistepScheduler()

    @classmethod
    def from_pretrained(
        cls,
        model_path: Union[str, Path],
        dtype: mx.Dtype = mx.float32,
        load_text_encoder: bool = True,
        strict_components: bool = True,
    ) -> "StableAudio":
        """
        Load Stable Audio from pretrained weights.

        Args:
            model_path: Path to model directory or HuggingFace repo ID
            dtype: Data type for model weights
            load_text_encoder: Whether to load T5 text encoder
            strict_components: If True (default), raise error if text encoder
                fails to load. If False, fall back to placeholder (unconditional).

        Returns:
            StableAudio instance

        Raises:
            RuntimeError: If strict_components=True and text encoder fails to load
        """
        from mlx_music.weights.weight_loader import (
            download_model,
            load_stable_audio_weights,
        )

        # Resolve path
        model_path = download_model(str(model_path))

        # Load configuration
        logger.info("Loading configuration...")
        config = StableAudioConfig.from_pretrained(model_path)

        # Load transformer
        logger.info("Loading transformer...")
        transformer = StableAudioDiT(config.transformer)
        transformer_weights = load_stable_audio_weights(
            model_path, component="transformer", dtype=dtype
        )
        transformer.load_weights(list(transformer_weights.items()), strict=False)

        # Load VAE
        logger.info("Loading VAE...")
        vae = AutoencoderOobleck(config.vae)
        vae_weights = load_stable_audio_weights(
            model_path, component="vae", dtype=dtype
        )
        vae.load_weights(list(vae_weights.items()), strict=False)

        # Load projection model
        logger.info("Loading projection model...")
        projection_model = ProjectionModel(
            text_encoder_dim=config.projection.text_encoder_dim,
            output_dim=config.projection.output_dim,
        )
        proj_weights = load_stable_audio_weights(
            model_path, component="projection_model", dtype=dtype
        )
        projection_model.load_weights(list(proj_weights.items()), strict=False)

        # Load text encoder
        text_encoder = None
        if load_text_encoder:
            logger.info("Loading text encoder (T5)...")
            try:
                from mlx_music.models.stable_audio.text_encoder import get_text_encoder
                text_encoder = get_text_encoder(model_path)

                # Check if we got a placeholder encoder
                from mlx_music.models.stable_audio.text_encoder import PlaceholderTextEncoder
                if isinstance(text_encoder, PlaceholderTextEncoder):
                    if strict_components:
                        raise RuntimeError(
                            "Text encoder not available. Text prompts will not condition generation. "
                            "Install with: pip install 'mlx-music[text-encoder]' "
                            "Or set strict_components=False to use unconditional generation."
                        )
                    else:
                        logger.warning(
                            "Using placeholder text encoder. "
                            "Text prompts will NOT condition generation."
                        )
                else:
                    logger.info("Text encoder loaded successfully!")
            except Exception as e:
                if strict_components:
                    raise RuntimeError(
                        f"Failed to load text encoder: {e}. "
                        "Install with: pip install 'mlx-music[text-encoder]' "
                        "Or set strict_components=False to use unconditional generation."
                    )
                else:
                    logger.warning(f"Could not load text encoder: {e}")
                    logger.warning("Using placeholder text encoder (unconditional generation)")

        logger.info("Model loaded successfully!")

        return cls(
            transformer=transformer,
            vae=vae,
            projection_model=projection_model,
            config=config,
            text_encoder=text_encoder,
            dtype=dtype,
        )

    def encode_text(
        self,
        prompt: str,
        negative_prompt: Optional[str] = None,
    ) -> Tuple[mx.array, mx.array, mx.array, mx.array]:
        """
        Encode text prompt.

        Args:
            prompt: Text description
            negative_prompt: Optional negative prompt for CFG

        Returns:
            Tuple of (text_embeds, pooled, neg_text_embeds, neg_pooled)
        """
        if self.text_encoder is None:
            # Placeholder embeddings using dimensions from config
            batch = 1
            seq_len = 64  # Default T5 sequence length
            dim = self.config.projection.text_encoder_dim

            text_embeds = mx.zeros((batch, seq_len, dim), dtype=self.dtype)
            pooled = mx.zeros((batch, dim), dtype=self.dtype)
            neg_text_embeds = mx.zeros((batch, seq_len, dim), dtype=self.dtype)
            neg_pooled = mx.zeros((batch, dim), dtype=self.dtype)

            return text_embeds, pooled, neg_text_embeds, neg_pooled

        # Encode using T5
        text_embeds, pooled = self.text_encoder.encode(prompt)

        if negative_prompt is not None:
            neg_text_embeds, neg_pooled = self.text_encoder.encode(negative_prompt)
        else:
            neg_text_embeds, neg_pooled = self.text_encoder.encode("")

        return (
            text_embeds.astype(self.dtype),
            pooled.astype(self.dtype),
            neg_text_embeds.astype(self.dtype),
            neg_pooled.astype(self.dtype),
        )

    def prepare_latents(
        self,
        duration: float,
        batch_size: int = 1,
    ) -> mx.array:
        """
        Prepare initial noise latents.

        Args:
            duration: Audio duration in seconds
            batch_size: Batch size

        Returns:
            Initial noise latents
        """
        # Calculate latent sequence length
        # Audio samples = duration * sample_rate
        # Latent length = audio_samples / compression_ratio
        audio_samples = int(duration * self.config.sample_rate)

        # Validate compression ratio
        if self.vae.compression_ratio <= 0:
            raise ValueError(
                f"Invalid VAE compression ratio: {self.vae.compression_ratio}"
            )

        latent_length = audio_samples // self.vae.compression_ratio

        # Validate minimum latent length
        if latent_length < 1:
            min_duration = self.vae.compression_ratio / self.config.sample_rate
            raise ValueError(
                f"Duration {duration}s too short (results in {latent_length} latent frames). "
                f"Minimum duration: {min_duration:.2f}s"
            )

        # Latent shape: (batch, length, channels)
        latent_shape = (batch_size, latent_length, self.vae.latent_channels)

        # Sample initial noise
        latents = mx.random.normal(latent_shape, dtype=self.dtype)

        return latents

    def decode_latents(self, latents: mx.array) -> np.ndarray:
        """
        Decode latents to audio waveform.

        Args:
            latents: Latent representation

        Returns:
            Audio waveform as numpy array (channels, samples)
        """
        # Decode through VAE
        audio = self.vae.decode(latents)

        # Convert to numpy and transpose to (channels, samples)
        audio_np = np.array(audio[0], dtype=np.float32)
        if audio_np.ndim == 2:
            # (samples, channels) -> (channels, samples)
            audio_np = audio_np.T

        return audio_np

    def generate(
        self,
        prompt: str,
        negative_prompt: Optional[str] = None,
        duration: float = 30.0,
        num_inference_steps: int = 100,
        guidance_scale: float = 7.0,
        seed: Optional[int] = None,
        return_latents: bool = False,
        callback: Optional[callable] = None,
    ) -> GenerationOutput:
        """
        Generate audio from text prompt.

        Args:
            prompt: Text description of desired audio
            negative_prompt: Optional negative prompt for CFG
            duration: Duration in seconds (1-47 seconds)
            num_inference_steps: Number of diffusion steps
            guidance_scale: Classifier-free guidance scale
            seed: Random seed for reproducibility
            return_latents: Whether to return intermediate latents
            callback: Optional callback(step, timestep, latents)

        Returns:
            GenerationOutput with audio and metadata

        Raises:
            ValueError: If duration is outside valid range
        """
        # Validate duration
        if duration < self.MIN_DURATION:
            raise ValueError(
                f"Duration {duration}s is too short. "
                f"Minimum duration is {self.MIN_DURATION}s"
            )
        if duration > self.MAX_DURATION:
            raise ValueError(
                f"Duration {duration}s is too long. "
                f"Maximum duration is {self.MAX_DURATION}s"
            )

        # Validate num_inference_steps
        if num_inference_steps <= 0:
            raise ValueError(
                f"num_inference_steps must be positive, got {num_inference_steps}"
            )
        if num_inference_steps > 1000:
            raise ValueError(
                f"num_inference_steps too large ({num_inference_steps}). Maximum is 1000."
            )

        # Validate guidance_scale (must be positive, 1.0 = no CFG, >1.0 = standard CFG)
        if guidance_scale <= 0:
            raise ValueError(
                f"guidance_scale must be > 0, got {guidance_scale}"
            )

        # Validate prompt
        if prompt is None:
            raise ValueError("prompt cannot be None")
        if not isinstance(prompt, str):
            raise TypeError(f"prompt must be a string, got {type(prompt).__name__}")
        if not prompt.strip():
            raise ValueError("prompt cannot be empty")

        # Set seed if provided
        if seed is not None:
            mx.random.seed(seed)

        # Encode text
        text_embeds, pooled, neg_text_embeds, neg_pooled = self.encode_text(
            prompt, negative_prompt
        )

        # Prepare global conditioning
        # Start time is 0, total time is duration
        seconds_start = mx.array([0.0])
        seconds_total = mx.array([duration])

        global_cond = self.projection_model(pooled, seconds_start, seconds_total)
        neg_global_cond = self.projection_model(neg_pooled, seconds_start, seconds_total)

        # Prepare latents
        latents = self.prepare_latents(duration)

        # Get timesteps
        timesteps, _ = retrieve_timesteps(self.scheduler, num_inference_steps)

        # Diffusion loop
        for i, t in enumerate(timesteps):
            # Expand timestep for batch
            timestep = mx.array([float(t)])

            # Scale input for EDM
            latent_model_input = self.scheduler.scale_model_input(latents, t)

            # Conditional prediction
            noise_pred = self.transformer(
                hidden_states=latent_model_input,
                timestep=timestep,
                encoder_hidden_states=text_embeds,
                global_embed=global_cond,
            )

            # Classifier-free guidance
            if guidance_scale > 1.0:
                # Unconditional prediction
                noise_pred_uncond = self.transformer(
                    hidden_states=latent_model_input,
                    timestep=timestep,
                    encoder_hidden_states=neg_text_embeds,
                    global_embed=neg_global_cond,
                )

                # CFG combination
                noise_pred = noise_pred_uncond + guidance_scale * (
                    noise_pred - noise_pred_uncond
                )

            # Scheduler step
            output = self.scheduler.step(noise_pred, t, latents)
            latents = output.prev_sample

            # Callback
            if callback is not None:
                callback(i, t, latents)

            # Single batched evaluation at end of iteration for better performance
            mx.eval(latents)

        # Decode to audio
        audio = self.decode_latents(latents)

        return GenerationOutput(
            audio=audio,
            sample_rate=self.config.sample_rate,
            duration=duration,
            latents=latents if return_latents else None,
        )

    def __repr__(self) -> str:
        return (
            f"StableAudio(\n"
            f"  sample_rate={self.config.sample_rate},\n"
            f"  max_duration={self.MAX_DURATION}s,\n"
            f"  dtype={self.dtype},\n"
            f"  has_text_encoder={self.text_encoder is not None},\n"
            f")"
        )


__all__ = [
    "StableAudio",
    "GenerationOutput",
]
