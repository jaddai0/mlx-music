"""
MusicGen main model class.

High-level interface for loading and generating music with MusicGen.
"""

import logging
from pathlib import Path
from typing import Callable, List, Optional, Union

import mlx.core as mx
import numpy as np

from .config import MusicGenConfig
from .conditioning import MelodyConditioner, get_text_encoder
from .generation import GenerationOutput, MusicGenGenerator
from .transformer import MusicGenDecoder, load_musicgen_decoder_weights

logger = logging.getLogger(__name__)


class MusicGen:
    """
    MusicGen Music Generation Model.

    High-level interface for loading and generating music.

    Example:
        >>> model = MusicGen.from_pretrained("/path/to/MusicGen-small")
        >>> output = model.generate(
        ...     prompt="upbeat electronic dance music",
        ...     duration=10.0
        ... )
        >>> # Save audio
        >>> import soundfile as sf
        >>> sf.write("output.wav", output.audio.T, output.sample_rate)
    """

    # Maximum duration for single generation (use generate_extended for longer)
    MAX_DURATION = 30.0

    def __init__(
        self,
        decoder: MusicGenDecoder,
        config: MusicGenConfig,
        text_encoder=None,
        encodec=None,
        melody_conditioner: Optional[MelodyConditioner] = None,
        dtype: mx.Dtype = mx.float32,
    ):
        """
        Initialize MusicGen model.

        Args:
            decoder: MusicGen decoder transformer
            config: Full MusicGen configuration
            text_encoder: Text encoder for conditioning
            encodec: EnCodec audio codec
            melody_conditioner: Optional melody conditioner (for melody variant)
            dtype: Data type for model
        """
        self.decoder = decoder
        self.config = config
        self.text_encoder = text_encoder
        self.encodec = encodec
        self.melody_conditioner = melody_conditioner
        self.dtype = dtype

        # Create generator
        self._generator = None

    @property
    def generator(self) -> MusicGenGenerator:
        """Get or create generator instance."""
        if self._generator is None:
            self._generator = MusicGenGenerator(
                decoder=self.decoder,
                config=self.config,
                encodec=self.encodec,
                text_encoder=self.text_encoder,
            )
        return self._generator

    @classmethod
    def from_pretrained(
        cls,
        model_path: Union[str, Path],
        dtype: mx.Dtype = mx.float32,
        load_text_encoder: bool = True,
        load_encodec: bool = True,
    ) -> "MusicGen":
        """
        Load MusicGen from pretrained weights.

        Supports both local paths and HuggingFace Hub repository IDs.

        Args:
            model_path: Path to model directory or HuggingFace repo ID
                Examples:
                - "/path/to/local/MusicGen-small" (local path)
                - "facebook/musicgen-small" (HuggingFace repo)
            dtype: Data type for model weights
            load_text_encoder: Whether to load T5 text encoder
            load_encodec: Whether to load EnCodec for audio decoding

        Returns:
            MusicGen instance
        """
        from mlx_music.weights.weight_loader import download_model

        # Resolve path (handles both local and HF Hub)
        model_path = download_model(str(model_path))

        # Load config
        logger.info("Loading configuration...")
        config = MusicGenConfig.from_pretrained(model_path)

        # Create and load decoder
        logger.info(f"Loading decoder ({config.decoder.num_hidden_layers} layers)...")
        decoder = MusicGenDecoder(config.decoder)
        weights = load_musicgen_decoder_weights(model_path, dtype)

        # Map weights to model parameters
        decoder.load_weights(list(weights.items()), strict=False)
        logger.info(f"Decoder loaded: {config.decoder.hidden_size}d, {config.decoder.num_attention_heads} heads")

        # Load text encoder
        text_encoder = None
        if load_text_encoder:
            logger.info("Loading text encoder (T5)...")
            try:
                text_encoder = get_text_encoder(
                    model_path=model_path,
                    use_fp16=(dtype == mx.float16),
                )
                logger.info("Text encoder loaded successfully!")
            except Exception as e:
                logger.warning(f"Could not load text encoder: {e}")
                logger.warning("Using placeholder encoder (generation will have limited quality)")

        # Load EnCodec
        encodec = None
        if load_encodec:
            logger.info("Loading EnCodec audio codec...")
            try:
                from mlx_music.codecs import get_encodec

                # Get audio_channels from config (2 for stereo models)
                audio_channels = config.decoder.audio_channels

                encodec = get_encodec(
                    model_id="facebook/encodec_32khz",
                    dtype=dtype,
                    audio_channels=audio_channels,
                )
                channels_str = "stereo" if audio_channels == 2 else "mono"
                logger.info(f"EnCodec loaded successfully! ({channels_str})")
            except Exception as e:
                logger.warning(f"Could not load EnCodec: {e}")
                logger.warning("Audio decoding will use placeholder (silence)")

        # Load melody conditioner for melody variant
        melody_conditioner = None
        if config.is_melody:
            melody_conditioner = MelodyConditioner(
                sample_rate=config.sampling_rate,
                num_chroma=config.num_chroma,
                hidden_size=config.hidden_size,
                frame_rate=config.frame_rate,
            )
            logger.info("Melody conditioner initialized")

        return cls(
            decoder=decoder,
            config=config,
            text_encoder=text_encoder,
            encodec=encodec,
            melody_conditioner=melody_conditioner,
            dtype=dtype,
        )

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
            duration: Target duration in seconds (max 30s recommended)
            temperature: Sampling temperature (higher = more random)
            top_k: Top-k filtering (higher = more diverse)
            top_p: Nucleus sampling threshold (0.0 = disabled)
            guidance_scale: Classifier-free guidance scale
            use_sampling: If False, use greedy decoding (deterministic)
            seed: Random seed for reproducibility
            callback: Optional progress callback(step, total_steps, codes)
            return_codes: Whether to return raw audio codes

        Returns:
            GenerationOutput (single) or List[GenerationOutput] (batch)
        """
        # Validate prompt
        if prompt is None:
            raise ValueError("prompt cannot be None")
        if isinstance(prompt, str):
            if not prompt.strip():
                raise ValueError("prompt cannot be empty")
        elif isinstance(prompt, list):
            if len(prompt) == 0:
                raise ValueError("prompt list cannot be empty")
            for i, p in enumerate(prompt):
                if not isinstance(p, str) or not p.strip():
                    raise ValueError(f"prompt[{i}] must be a non-empty string")
        else:
            raise TypeError(f"prompt must be str or list of str, got {type(prompt).__name__}")

        if self.text_encoder is None:
            raise RuntimeError(
                "Text encoder not available. Reload model with "
                "MusicGen.from_pretrained(..., load_text_encoder=True)"
            )

        if self.encodec is None:
            raise RuntimeError(
                "EnCodec not available. Reload model with "
                "MusicGen.from_pretrained(..., load_encodec=True)"
            )

        # Automatically route to extended generation for duration > MAX_DURATION
        if duration > self.MAX_DURATION:
            # Warn about batch limitation - extended generation requires sequential context
            if isinstance(prompt, list) and len(prompt) > 1:
                logger.warning(
                    f"Extended generation (>{self.MAX_DURATION}s) requires sequential context "
                    f"and doesn't support batch processing. Using only the first prompt "
                    f"(discarding {len(prompt) - 1} prompts)."
                )
            logger.info(
                f"Duration {duration}s exceeds {self.MAX_DURATION}s limit. "
                "Automatically using generate_extended() for seamless long-form audio."
            )
            return self.generate_extended(
                prompt=prompt[0] if isinstance(prompt, list) else prompt,
                duration=duration,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                guidance_scale=guidance_scale,
                use_sampling=use_sampling,
                seed=seed,
                callback=callback,
            )

        return self.generator.generate(
            prompt=prompt,
            duration=duration,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            guidance_scale=guidance_scale,
            use_sampling=use_sampling,
            seed=seed,
            callback=callback,
            return_codes=return_codes,
        )

    def generate_with_melody(
        self,
        prompt: str,
        melody_audio: mx.array,
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

        Only available for MusicGen-Melody variant. The melody from the reference
        audio guides the generation while the text prompt controls style/instrumentation.

        Supports double CFG when guidance_scale_beta > 0:
        - Primary: text+melody vs unconditional (controlled by guidance_scale)
        - Secondary: text-only vs unconditional (controlled by guidance_scale_beta)

        Args:
            prompt: Text description of desired style/instrumentation
            melody_audio: Reference audio for melody extraction (samples,) or (channels, samples)
            duration: Target duration in seconds
            temperature: Sampling temperature (higher = more random)
            top_k: Top-k filtering (higher = more diverse)
            top_p: Nucleus sampling threshold (0.0 = disabled)
            guidance_scale: Primary CFG scale (text+melody vs unconditional)
            guidance_scale_beta: Secondary CFG scale (text-only vs unconditional), 0 = disabled
            use_sampling: If False, use greedy decoding (deterministic)
            seed: Random seed for reproducibility
            callback: Optional progress callback(step, total_steps, codes)
            return_codes: Whether to return raw audio codes

        Returns:
            GenerationOutput with melody-conditioned audio
        """
        if not self.config.is_melody:
            raise RuntimeError(
                "Melody conditioning only available for MusicGen-Melody variant. "
                "Load a melody model like 'facebook/musicgen-melody'"
            )

        if self.melody_conditioner is None:
            raise RuntimeError("Melody conditioner not initialized")

        if self.text_encoder is None:
            raise RuntimeError("Text encoder not loaded. Load with load_text_encoder=True")

        if self.encodec is None:
            raise RuntimeError("EnCodec not loaded. Load with load_encodec=True")

        return self.generator.generate_with_melody(
            prompt=prompt,
            melody_audio=melody_audio,
            melody_conditioner=self.melody_conditioner,
            duration=duration,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            guidance_scale=guidance_scale,
            guidance_scale_beta=guidance_scale_beta,
            use_sampling=use_sampling,
            seed=seed,
            callback=callback,
            return_codes=return_codes,
        )

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

        Encodes the existing audio and continues generating in the same style,
        guided by the text prompt.

        Args:
            audio: Existing audio to continue from (samples,) or (channels, samples)
            prompt: Text prompt for continuation style
            duration: Additional duration to generate in seconds
            temperature: Sampling temperature (higher = more random)
            top_k: Top-k filtering (higher = more diverse)
            top_p: Nucleus sampling threshold (0.0 = disabled)
            guidance_scale: Classifier-free guidance scale
            use_sampling: If False, use greedy decoding (deterministic)
            seed: Random seed for reproducibility
            callback: Optional progress callback(step, total_steps, codes)
            return_codes: Whether to return raw audio codes

        Returns:
            GenerationOutput with continuation (original + generated audio)
        """
        if self.text_encoder is None:
            raise RuntimeError("Text encoder not loaded. Load with load_text_encoder=True")

        if self.encodec is None:
            raise RuntimeError("EnCodec not loaded. Load with load_encodec=True")

        return self.generator.generate_continuation(
            audio=audio,
            prompt=prompt,
            duration=duration,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            guidance_scale=guidance_scale,
            use_sampling=use_sampling,
            seed=seed,
            callback=callback,
            return_codes=return_codes,
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
            temperature: Sampling temperature (higher = more random)
            top_k: Top-k filtering (higher = more diverse)
            top_p: Nucleus sampling threshold (0.0 = disabled)
            guidance_scale: Classifier-free guidance scale
            use_sampling: If False, use greedy decoding (deterministic)
            seed: Random seed for reproducibility
            callback: Optional progress callback(step, total_steps, codes)

        Returns:
            GenerationOutput with seamlessly blended long audio

        Example:
            >>> output = model.generate_extended(
            ...     prompt="jazz piano improvisation",
            ...     duration=60.0,  # 1 minute
            ... )
            >>> # Audio is seamlessly blended from multiple windows
        """
        if self.text_encoder is None:
            raise RuntimeError(
                "Text encoder not loaded. Load with load_text_encoder=True"
            )

        if self.encodec is None:
            raise RuntimeError("EnCodec not loaded. Load with load_encodec=True")

        return self.generator.generate_extended(
            prompt=prompt,
            duration=duration,
            extend_stride=extend_stride,
            window_length=window_length,
            fade_duration=fade_duration,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            guidance_scale=guidance_scale,
            use_sampling=use_sampling,
            seed=seed,
            callback=callback,
        )

    @property
    def is_stereo(self) -> bool:
        """Check if this is a stereo model."""
        return self.config.decoder.audio_channels == 2

    @property
    def audio_channels(self) -> int:
        """Get the number of audio channels (1=mono, 2=stereo)."""
        return self.config.decoder.audio_channels

    def __repr__(self) -> str:
        audio_mode = "stereo" if self.is_stereo else "mono"
        return (
            f"MusicGen(\n"
            f"  model_type={self.config.model_type},\n"
            f"  hidden_size={self.config.hidden_size},\n"
            f"  num_layers={self.config.num_hidden_layers},\n"
            f"  num_heads={self.config.num_attention_heads},\n"
            f"  num_codebooks={self.config.num_codebooks},\n"
            f"  audio_channels={self.audio_channels} ({audio_mode}),\n"
            f"  dtype={self.dtype},\n"
            f"  has_text_encoder={self.text_encoder is not None},\n"
            f"  has_encodec={self.encodec is not None},\n"
            f")"
        )


__all__ = ["MusicGen"]
