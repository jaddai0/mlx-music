"""
MusicGen main model class.

High-level interface for loading and generating music with MusicGen.
"""

from pathlib import Path
from typing import Callable, Optional, Union

import mlx.core as mx
import numpy as np

from .config import MusicGenConfig
from .conditioning import MelodyConditioner, get_text_encoder
from .generation import GenerationOutput, MusicGenGenerator
from .transformer import MusicGenDecoder, load_musicgen_decoder_weights


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
        print("Loading configuration...")
        config = MusicGenConfig.from_pretrained(model_path)

        # Create and load decoder
        print(f"Loading decoder ({config.decoder.num_hidden_layers} layers)...")
        decoder = MusicGenDecoder(config.decoder)
        weights = load_musicgen_decoder_weights(model_path, dtype)

        # Map weights to model parameters
        decoder.load_weights(list(weights.items()), strict=False)
        print(f"Decoder loaded: {config.decoder.hidden_size}d, {config.decoder.num_attention_heads} heads")

        # Load text encoder
        text_encoder = None
        if load_text_encoder:
            print("Loading text encoder (T5)...")
            try:
                text_encoder = get_text_encoder(
                    model_path=model_path,
                    use_fp16=(dtype == mx.float16),
                )
                print("Text encoder loaded successfully!")
            except Exception as e:
                print(f"Warning: Could not load text encoder: {e}")
                print("Using placeholder encoder (generation will have limited quality)")

        # Load EnCodec
        encodec = None
        if load_encodec:
            print("Loading EnCodec audio codec...")
            try:
                from mlx_music.codecs import get_encodec

                encodec = get_encodec(
                    model_id="facebook/encodec_32khz",
                    dtype=dtype,
                )
                print("EnCodec loaded successfully!")
            except Exception as e:
                print(f"Warning: Could not load EnCodec: {e}")
                print("Audio decoding will use placeholder (silence)")

        # Load melody conditioner for melody variant
        melody_conditioner = None
        if config.is_melody:
            melody_conditioner = MelodyConditioner(
                sample_rate=config.sampling_rate,
                num_chroma=config.num_chroma,
            )
            print("Melody conditioner initialized")

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
            duration: Target duration in seconds (max 30s recommended)
            temperature: Sampling temperature (higher = more random)
            top_k: Top-k filtering (higher = more diverse)
            top_p: Nucleus sampling threshold (0.0 = disabled)
            guidance_scale: Classifier-free guidance scale
            seed: Random seed for reproducibility
            callback: Optional progress callback(step, total_steps, codes)
            return_codes: Whether to return raw audio codes

        Returns:
            GenerationOutput with audio and metadata
        """
        if self.text_encoder is None:
            raise RuntimeError(
                "Text encoder not loaded. Load with load_text_encoder=True"
            )

        if self.encodec is None:
            raise RuntimeError("EnCodec not loaded. Load with load_encodec=True")

        # Validate duration
        max_duration = 30.0  # MusicGen default max
        if duration > max_duration:
            print(
                f"Warning: Duration {duration}s exceeds recommended max {max_duration}s. "
                "Generation may be slow or fail."
            )

        return self.generator.generate(
            prompt=prompt,
            duration=duration,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            guidance_scale=guidance_scale,
            seed=seed,
            callback=callback,
            return_codes=return_codes,
        )

    def generate_with_melody(
        self,
        prompt: str,
        melody_audio: mx.array,
        duration: float = 10.0,
        **kwargs,
    ) -> GenerationOutput:
        """
        Generate music with melody conditioning.

        Only available for MusicGen-Melody variant.

        Args:
            prompt: Text description
            melody_audio: Reference audio for melody extraction
            duration: Target duration
            **kwargs: Additional generation parameters

        Returns:
            GenerationOutput with audio
        """
        if not self.config.is_melody:
            raise RuntimeError(
                "Melody conditioning only available for MusicGen-Melody variant"
            )

        if self.melody_conditioner is None:
            raise RuntimeError("Melody conditioner not initialized")

        # Extract chroma from melody audio
        chroma = self.melody_conditioner.extract_chroma(melody_audio)

        # TODO: Integrate chroma into generation
        raise NotImplementedError(
            "Melody-conditioned generation not yet fully implemented"
        )

    def __repr__(self) -> str:
        return (
            f"MusicGen(\n"
            f"  model_type={self.config.model_type},\n"
            f"  hidden_size={self.config.hidden_size},\n"
            f"  num_layers={self.config.num_hidden_layers},\n"
            f"  num_heads={self.config.num_attention_heads},\n"
            f"  num_codebooks={self.config.num_codebooks},\n"
            f"  dtype={self.dtype},\n"
            f"  has_text_encoder={self.text_encoder is not None},\n"
            f"  has_encodec={self.encodec is not None},\n"
            f")"
        )


__all__ = ["MusicGen"]
