"""
MusicGen conditioning modules.

Text encoder and optional melody conditioning for MusicGen models.
"""

import logging
from typing import Optional, Tuple

import mlx.core as mx
import mlx.nn as nn
import numpy as np

logger = logging.getLogger(__name__)

# Valid device options for PyTorch
VALID_DEVICES = {"cpu", "mps", "cuda"}


class MusicGenTextEncoder:
    """
    T5-based text encoder for MusicGen.

    Uses HuggingFace transformers T5 model for text encoding.
    The encoder outputs are used as cross-attention inputs to the decoder.
    """

    def __init__(
        self,
        model_name: str = "t5-base",
        device: str = "cpu",
        use_fp16: bool = False,
    ):
        """
        Initialize T5 text encoder.

        Args:
            model_name: T5 model name or path
            device: Device for PyTorch model ("cpu", "mps", "cuda")
            use_fp16: Whether to use FP16 precision

        Raises:
            ValueError: If device is not one of "cpu", "mps", "cuda"
        """
        if device not in VALID_DEVICES:
            raise ValueError(
                f"Invalid device '{device}'. Must be one of: {', '.join(sorted(VALID_DEVICES))}"
            )

        self.model_name = model_name
        self.device = device
        self.use_fp16 = use_fp16

        self._tokenizer = None
        self._encoder = None

    def _move_model_to_device(self, model, device: str):
        """
        Move model to device with CPU fallback on failure.

        Args:
            model: PyTorch model to move
            device: Target device

        Returns:
            Model on target device (or CPU if fallback occurred)
        """
        if device == "cpu":
            return model.to("cpu")

        try:
            if device == "cuda":
                return model.cuda()
            else:
                return model.to(device)
        except Exception as e:
            logger.warning(f"{device.upper()} failed ({e}), falling back to CPU")
            self.device = "cpu"
            return model.to("cpu")

    def _move_tensors_to_device(self, *tensors):
        """
        Move tensors to the current device.

        Args:
            *tensors: PyTorch tensors to move

        Returns:
            Tuple of tensors on the current device
        """
        if self.device == "cpu":
            return tensors if len(tensors) > 1 else tensors[0]

        moved = []
        for tensor in tensors:
            if self.device == "cuda":
                moved.append(tensor.cuda())
            else:
                moved.append(tensor.to(self.device))

        return tuple(moved) if len(moved) > 1 else moved[0]

    def _load_model(self):
        """Lazy load the T5 model and tokenizer."""
        if self._encoder is not None:
            return

        try:
            import torch
            from transformers import T5EncoderModel, T5Tokenizer
        except ImportError:
            raise ImportError(
                "transformers and torch are required for text encoding. "
                "Install with: pip install transformers torch"
            )

        self._tokenizer = T5Tokenizer.from_pretrained(self.model_name)
        self._encoder = T5EncoderModel.from_pretrained(self.model_name)

        # Move to device with fallback
        self._encoder = self._move_model_to_device(self._encoder, self.device)

        if self.use_fp16 and self.device != "cpu":
            self._encoder = self._encoder.half()

        self._encoder.eval()

    def encode(
        self,
        text: str,
        max_length: int = 256,
    ) -> Tuple[mx.array, mx.array]:
        """
        Encode text to embeddings.

        Args:
            text: Input text prompt
            max_length: Maximum sequence length

        Returns:
            embeddings: (1, seq_len, hidden_size) text embeddings as mx.array
            attention_mask: (1, seq_len) attention mask as mx.array
        """
        self._load_model()

        import torch

        # Tokenize
        inputs = self._tokenizer(
            text,
            return_tensors="pt",
            padding="max_length",
            max_length=max_length,
            truncation=True,
        )

        # Move to device
        input_ids, attention_mask = self._move_tensors_to_device(
            inputs["input_ids"], inputs["attention_mask"]
        )

        # Encode
        with torch.no_grad():
            outputs = self._encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )
            hidden_states = outputs.last_hidden_state

        # Convert to numpy then mx.array
        hidden_states_np = hidden_states.cpu().float().numpy()
        attention_mask_np = attention_mask.cpu().numpy()

        return mx.array(hidden_states_np), mx.array(attention_mask_np)

    def encode_batch(
        self,
        texts: list,
        max_length: int = 256,
    ) -> Tuple[mx.array, mx.array]:
        """
        Encode batch of texts.

        Args:
            texts: List of text prompts
            max_length: Maximum sequence length

        Returns:
            embeddings: (batch, seq_len, hidden_size) text embeddings
            attention_mask: (batch, seq_len) attention masks
        """
        self._load_model()

        import torch

        # Tokenize batch
        inputs = self._tokenizer(
            texts,
            return_tensors="pt",
            padding="max_length",
            max_length=max_length,
            truncation=True,
        )

        # Move to device
        input_ids, attention_mask = self._move_tensors_to_device(
            inputs["input_ids"], inputs["attention_mask"]
        )

        with torch.no_grad():
            outputs = self._encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )
            hidden_states = outputs.last_hidden_state

        hidden_states_np = hidden_states.cpu().float().numpy()
        attention_mask_np = attention_mask.cpu().numpy()

        return mx.array(hidden_states_np), mx.array(attention_mask_np)


class PlaceholderTextEncoder:
    """
    Placeholder text encoder for testing without transformers dependency.

    Returns zero embeddings of the correct shape.
    """

    def __init__(self, hidden_size: int = 768):
        self.hidden_size = hidden_size

    def encode(
        self,
        text: str,
        max_length: int = 256,
    ) -> Tuple[mx.array, mx.array]:
        """Return placeholder embeddings."""
        embeddings = mx.zeros((1, max_length, self.hidden_size))
        attention_mask = mx.ones((1, max_length))
        return embeddings, attention_mask

    def encode_batch(
        self,
        texts: list,
        max_length: int = 256,
    ) -> Tuple[mx.array, mx.array]:
        """Return placeholder embeddings for batch."""
        batch_size = len(texts)
        embeddings = mx.zeros((batch_size, max_length, self.hidden_size))
        attention_mask = mx.ones((batch_size, max_length))
        return embeddings, attention_mask


class MelodyConditioner:
    """
    Melody conditioning for MusicGen-Melody variant.

    Extracts chroma features from reference audio and projects them
    to embeddings for conditioning the generation.
    """

    def __init__(
        self,
        sample_rate: int = 32000,
        num_chroma: int = 12,
        hop_length: int = 4096,
        hidden_size: int = 1536,
        frame_rate: int = 50,
    ):
        """
        Initialize melody conditioner.

        Args:
            sample_rate: Audio sample rate
            num_chroma: Number of chroma bins (typically 12 for semitones)
            hop_length: Hop length for chroma extraction
            hidden_size: Target embedding dimension for decoder
            frame_rate: Target frame rate for resampling chroma
        """
        self.sample_rate = sample_rate
        self.num_chroma = num_chroma
        self.hop_length = hop_length
        self.hidden_size = hidden_size
        self.frame_rate = frame_rate

        # Chroma projection layer (num_chroma -> hidden_size)
        self.chroma_proj = nn.Linear(num_chroma, hidden_size)

    def extract_chroma(
        self,
        audio: mx.array,
    ) -> mx.array:
        """
        Extract chroma features from audio.

        Args:
            audio: Audio waveform (samples,) or (channels, samples)

        Returns:
            chroma: (num_frames, num_chroma) chroma features, always at least 1 frame
        """
        # Convert to numpy for librosa
        audio_np = np.array(audio)

        # Handle multi-channel by taking mean
        if audio_np.ndim > 1:
            audio_np = audio_np.mean(axis=0)

        # Validate audio has samples
        if len(audio_np) == 0:
            # Return single zero frame for empty audio
            return mx.zeros((1, self.num_chroma), dtype=mx.float32)

        try:
            import librosa

            chroma = librosa.feature.chroma_stft(
                y=audio_np,
                sr=self.sample_rate,
                hop_length=self.hop_length,
                n_chroma=self.num_chroma,
            )
            # Transpose to (num_frames, num_chroma)
            chroma = chroma.T

            # Handle edge case: librosa returns empty array for very short audio
            if chroma.shape[0] == 0:
                chroma = np.zeros((1, self.num_chroma), dtype=np.float32)

        except ImportError:
            # Fallback: return placeholder
            num_frames = max(1, len(audio_np) // self.hop_length)
            chroma = np.zeros((num_frames, self.num_chroma), dtype=np.float32)

        return mx.array(chroma, dtype=mx.float32)

    def get_chroma_embeddings(
        self,
        audio: mx.array,
        target_length: int,
    ) -> mx.array:
        """
        Extract chroma and project to embeddings, resampled to target length.

        Args:
            audio: Audio waveform (samples,) or (channels, samples)
            target_length: Target number of frames to generate

        Returns:
            chroma_embeddings: (1, target_length, hidden_size) embeddings
        """
        # Handle edge case: target_length <= 0
        if target_length <= 0:
            return mx.zeros((1, 0, self.hidden_size), dtype=mx.float32)

        # Extract raw chroma features
        chroma = self.extract_chroma(audio)  # (num_frames, num_chroma)

        # Resample chroma to target length using linear interpolation
        num_frames = chroma.shape[0]
        if num_frames != target_length:
            # Convert to numpy for interpolation
            chroma_np = np.array(chroma)
            indices = np.linspace(0, num_frames - 1, target_length)
            indices_floor = np.floor(indices).astype(int)
            indices_ceil = np.minimum(indices_floor + 1, num_frames - 1)
            weights = indices - indices_floor

            # Linear interpolation
            chroma_resampled = (
                chroma_np[indices_floor] * (1 - weights[:, None])
                + chroma_np[indices_ceil] * weights[:, None]
            )
            chroma = mx.array(chroma_resampled, dtype=mx.float32)

        # Project to hidden size: (target_length, num_chroma) -> (target_length, hidden_size)
        chroma_embeddings = self.chroma_proj(chroma)

        # Add batch dimension
        return chroma_embeddings[None, :, :]  # (1, target_length, hidden_size)


def get_text_encoder(
    model_path: str,
    device: Optional[str] = None,
    use_fp16: bool = False,
    use_placeholder: bool = False,
) -> "MusicGenTextEncoder | PlaceholderTextEncoder":
    """
    Get text encoder, with fallback to placeholder.

    Args:
        model_path: Path containing text encoder config
        device: Device for encoder ("cpu", "mps", "cuda", or None for auto)
        use_fp16: Whether to use FP16 precision
        use_placeholder: Force use of placeholder encoder

    Returns:
        Text encoder instance
    """
    if use_placeholder:
        return PlaceholderTextEncoder()

    # Auto-detect device using shared utility
    if device is None:
        from mlx_music.utils.device import get_default_torch_device
        device = get_default_torch_device()

    try:
        return MusicGenTextEncoder(
            model_name="t5-base",  # MusicGen uses t5-base
            device=device,
            use_fp16=use_fp16,
        )
    except Exception as e:
        logger.warning(f"Could not load text encoder: {e}")
        logger.warning("Using placeholder encoder (generation will have limited quality)")
        return PlaceholderTextEncoder()


__all__ = [
    "MusicGenTextEncoder",
    "PlaceholderTextEncoder",
    "MelodyConditioner",
    "get_text_encoder",
]
