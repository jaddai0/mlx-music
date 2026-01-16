"""
Audio codec integrations for mlx-music.

Provides wrappers around mlx-audio codec implementations.
Currently supports EnCodec for MusicGen compatibility.
"""

import logging
from typing import Optional, Tuple

import mlx.core as mx

logger = logging.getLogger(__name__)

# Lazy import flag
_encodec_available = None


def _check_encodec_available() -> bool:
    """Check if mlx-audio EnCodec is available."""
    global _encodec_available
    if _encodec_available is None:
        try:
            from mlx_audio.codec.models.encodec import Encodec

            _encodec_available = True
        except ImportError:
            _encodec_available = False
    return _encodec_available


class EnCodecWrapper:
    """
    Wrapper around mlx-audio's EnCodec implementation.

    Provides a simplified interface for MusicGen audio encoding/decoding.
    """

    def __init__(self, model, processor):
        """
        Initialize wrapper with mlx-audio EnCodec model.

        Args:
            model: Encodec model from mlx-audio
            processor: Audio preprocessing function
        """
        self.model = model
        self.processor = processor

    @classmethod
    def from_pretrained(
        cls,
        model_id: str = "facebook/encodec_32khz",
        dtype: mx.Dtype = mx.float32,
    ) -> "EnCodecWrapper":
        """
        Load EnCodec from pretrained weights.

        Args:
            model_id: HuggingFace model ID or local path
            dtype: Data type for model weights

        Returns:
            EnCodecWrapper instance
        """
        if not _check_encodec_available():
            raise ImportError(
                "mlx-audio is required for EnCodec support. "
                "Install with: pip install mlx-audio"
            )

        from mlx_audio.codec.models.encodec import Encodec

        # mlx-audio API may return model only or (model, processor) tuple
        # Handle both cases for compatibility
        result = Encodec.from_pretrained(model_id)
        if isinstance(result, tuple):
            model, processor = result
        else:
            model = result
            processor = None

        # Convert to specified dtype if needed
        if dtype != mx.float32:
            try:
                # Try to get parameters as dict
                params = model.parameters()
                if isinstance(params, dict):
                    new_weights = [(k, v.astype(dtype)) for k, v in params.items()]
                else:
                    # If it's a list of tuples or other format
                    new_weights = [(k, v.astype(dtype)) for k, v in params]
                model.load_weights(new_weights)
            except (AttributeError, TypeError, ValueError, RuntimeError) as e:
                logger.warning(f"Could not convert EnCodec to {dtype}: {e}")

        return cls(model, processor)

    def encode(
        self,
        audio: mx.array,
        bandwidth: float = 6.0,
    ) -> Tuple[mx.array, Optional[mx.array]]:
        """
        Encode audio waveform to discrete codes.

        Args:
            audio: Audio waveform (batch, channels, samples) or (channels, samples)
            bandwidth: Target bandwidth in kbps

        Returns:
            codes: Audio codes (batch, num_codebooks, num_frames)
            scales: Optional scale factors for reconstruction
        """
        # Ensure batch dimension
        if audio.ndim == 2:
            audio = audio[None]

        # Defensive unpacking - handle different mlx-audio API return formats
        result = self.model.encode(audio, bandwidth=bandwidth)
        if isinstance(result, tuple):
            codes = result[0]
            scales = result[1] if len(result) > 1 else None
        else:
            codes = result
            scales = None
        return codes, scales

    def decode(
        self,
        codes: mx.array,
        scales: Optional[mx.array] = None,
    ) -> mx.array:
        """
        Decode audio codes back to waveform.

        Args:
            codes: Audio codes (batch, num_codebooks, num_frames)
            scales: Optional scale factors from encoding

        Returns:
            audio: Reconstructed audio waveform
        """
        audio = self.model.decode(codes, scales)
        return audio

    @property
    def sample_rate(self) -> int:
        """Get the audio sample rate."""
        return self.model.config.sampling_rate

    @property
    def num_codebooks(self) -> int:
        """Get the number of codebooks."""
        # EnCodec uses 4 codebooks at 6kbps bandwidth
        return 4

    @property
    def frame_rate(self) -> int:
        """Get the frame rate (codes per second)."""
        # 32000 / (8*5*4*4) = 50 Hz
        import math

        hop_length = math.prod(self.model.config.upsampling_ratios)
        return self.model.config.sampling_rate // hop_length

    @property
    def audio_channels(self) -> int:
        """Get the number of audio channels (1=mono, 2=stereo)."""
        # Read from model config if available, default to 1 (mono)
        return getattr(self.model.config, "audio_channels", 1)


class PlaceholderEnCodec:
    """
    Placeholder EnCodec for when mlx-audio is not available.

    Returns zeros for encoding and silent audio for decoding.
    Used for testing model loading without full audio pipeline.
    """

    # Hop length = sample_rate / frame_rate = 32000 / 50 = 640
    # This matches EnCodec's upsampling ratios: 8 * 5 * 4 * 4 = 640
    HOP_LENGTH = 640
    DEFAULT_FRAME_RATE = 50

    def __init__(
        self,
        sample_rate: int = 32000,
        num_codebooks: int = 4,
        audio_channels: int = 1,
    ):
        self._sample_rate = sample_rate
        self._num_codebooks = num_codebooks
        self._audio_channels = audio_channels

    @classmethod
    def from_pretrained(
        cls,
        model_id: str = "facebook/encodec_32khz",
        dtype: mx.Dtype = mx.float32,
        audio_channels: int = 1,
    ) -> "PlaceholderEnCodec":
        """Create placeholder instance.

        Args:
            model_id: Model ID (used to detect stereo from model name)
            dtype: Data type (ignored for placeholder)
            audio_channels: Number of audio channels (1=mono, 2=stereo)

        Returns:
            PlaceholderEnCodec instance
        """
        # Auto-detect stereo from model name if not specified
        if audio_channels == 1 and "stereo" in model_id.lower():
            audio_channels = 2
        return cls(audio_channels=audio_channels)

    def encode(
        self,
        audio: mx.array,
        bandwidth: float = 6.0,
    ) -> Tuple[mx.array, Optional[mx.array]]:
        """Return placeholder codes."""
        if audio.ndim == 2:
            audio = audio[None]

        batch_size = audio.shape[0]
        num_samples = audio.shape[-1]
        num_frames = num_samples // self.HOP_LENGTH

        codes = mx.zeros((batch_size, self._num_codebooks, num_frames), dtype=mx.int32)
        return codes, None

    def decode(
        self,
        codes: mx.array,
        scales: Optional[mx.array] = None,
    ) -> mx.array:
        """Return silent audio with correct number of channels."""
        batch_size = codes.shape[0]
        num_frames = codes.shape[-1]
        num_samples = num_frames * self.HOP_LENGTH

        return mx.zeros(
            (batch_size, self._audio_channels, num_samples), dtype=mx.float32
        )

    @property
    def sample_rate(self) -> int:
        return self._sample_rate

    @property
    def num_codebooks(self) -> int:
        return self._num_codebooks

    @property
    def audio_channels(self) -> int:
        return self._audio_channels

    @property
    def frame_rate(self) -> int:
        return self.DEFAULT_FRAME_RATE  # 32000 / 640 = 50


def get_encodec(
    model_id: str = "facebook/encodec_32khz",
    dtype: mx.Dtype = mx.float32,
    use_placeholder: bool = False,
    audio_channels: int = 1,
) -> "EnCodecWrapper | PlaceholderEnCodec":
    """
    Get EnCodec model, with fallback to placeholder.

    Args:
        model_id: HuggingFace model ID
        dtype: Data type for model weights
        use_placeholder: Force use of placeholder
        audio_channels: Number of audio channels (1=mono, 2=stereo)
            Auto-detected from model_id if "stereo" is in the name.

    Returns:
        EnCodecWrapper or PlaceholderEnCodec
    """
    # Auto-detect stereo from model name
    if audio_channels == 1 and "stereo" in model_id.lower():
        audio_channels = 2

    if use_placeholder or not _check_encodec_available():
        if not use_placeholder:
            logger.warning(
                "mlx-audio not available, using placeholder EnCodec. "
                "Install with: pip install mlx-audio"
            )
        return PlaceholderEnCodec.from_pretrained(model_id, dtype, audio_channels)

    wrapper = EnCodecWrapper.from_pretrained(model_id, dtype)

    # Validate that loaded model has expected number of channels
    actual_channels = wrapper.audio_channels
    if actual_channels != audio_channels:
        logger.warning(
            f"EnCodec model has {actual_channels} channel(s) but {audio_channels} expected. "
            f"This may cause audio quality issues."
        )

    return wrapper


__all__ = [
    "EnCodecWrapper",
    "PlaceholderEnCodec",
    "get_encodec",
]
