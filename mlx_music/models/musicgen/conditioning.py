"""
MusicGen conditioning modules.

Text encoder and optional melody conditioning for MusicGen models.
"""

from typing import Optional, Tuple

import mlx.core as mx
import numpy as np


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
        """
        self.model_name = model_name
        self.device = device
        self.use_fp16 = use_fp16

        self._tokenizer = None
        self._encoder = None

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

        # Move to device
        if self.device == "mps":
            try:
                self._encoder = self._encoder.to("mps")
            except Exception as e:
                # Fallback to CPU if MPS fails - ensure model is actually on CPU
                print(f"Warning: MPS failed ({e}), falling back to CPU")
                self.device = "cpu"
                self._encoder = self._encoder.to("cpu")
        elif self.device == "cuda":
            try:
                self._encoder = self._encoder.cuda()
            except Exception as e:
                # Fallback to CPU if CUDA fails
                print(f"Warning: CUDA failed ({e}), falling back to CPU")
                self.device = "cpu"
                self._encoder = self._encoder.to("cpu")

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
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]

        if self.device == "mps":
            input_ids = input_ids.to("mps")
            attention_mask = attention_mask.to("mps")
        elif self.device == "cuda":
            input_ids = input_ids.cuda()
            attention_mask = attention_mask.cuda()

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

        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]

        if self.device == "mps":
            input_ids = input_ids.to("mps")
            attention_mask = attention_mask.to("mps")
        elif self.device == "cuda":
            input_ids = input_ids.cuda()
            attention_mask = attention_mask.cuda()

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

    Extracts chroma features from reference audio for melody guidance.
    """

    def __init__(
        self,
        sample_rate: int = 32000,
        num_chroma: int = 12,
        hop_length: int = 4096,
    ):
        """
        Initialize melody conditioner.

        Args:
            sample_rate: Audio sample rate
            num_chroma: Number of chroma bins (typically 12 for semitones)
            hop_length: Hop length for chroma extraction
        """
        self.sample_rate = sample_rate
        self.num_chroma = num_chroma
        self.hop_length = hop_length

    def extract_chroma(
        self,
        audio: mx.array,
    ) -> mx.array:
        """
        Extract chroma features from audio.

        Args:
            audio: Audio waveform (samples,) or (channels, samples)

        Returns:
            chroma: (num_frames, num_chroma) chroma features
        """
        # Convert to numpy for librosa
        audio_np = np.array(audio)

        # Handle multi-channel by taking mean
        if audio_np.ndim > 1:
            audio_np = audio_np.mean(axis=0)

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

        except ImportError:
            # Fallback: return placeholder
            num_frames = len(audio_np) // self.hop_length
            chroma = np.zeros((num_frames, self.num_chroma), dtype=np.float32)

        return mx.array(chroma, dtype=mx.float32)


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

    # Auto-detect device
    if device is None:
        import platform

        system = platform.system()
        if system == "Darwin":
            try:
                import torch

                if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                    device = "mps"
                else:
                    device = "cpu"
            except ImportError:
                device = "cpu"
        else:
            try:
                import torch

                if torch.cuda.is_available():
                    device = "cuda"
                else:
                    device = "cpu"
            except ImportError:
                device = "cpu"

    try:
        return MusicGenTextEncoder(
            model_name="t5-base",  # MusicGen uses t5-base
            device=device,
            use_fp16=use_fp16,
        )
    except Exception as e:
        print(f"Warning: Could not load text encoder: {e}")
        print("Using placeholder encoder (generation will have limited quality)")
        return PlaceholderTextEncoder()


__all__ = [
    "MusicGenTextEncoder",
    "PlaceholderTextEncoder",
    "MelodyConditioner",
    "get_text_encoder",
]
