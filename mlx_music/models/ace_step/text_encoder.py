"""
UMT5 Text Encoder for ACE-Step.

Encodes text prompts to embeddings for conditioning the diffusion model.
Uses HuggingFace transformers for the encoder, with MLX array outputs.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple, Union

import mlx.core as mx
import numpy as np

try:
    import torch
    from transformers import AutoTokenizer, UMT5EncoderModel

    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False


@dataclass
class TextEncoderConfig:
    """Configuration for UMT5 text encoder."""

    model_name: str = "google/umt5-base"
    max_length: int = 256
    hidden_size: int = 768
    use_fp16: bool = True


class UMT5TextEncoder:
    """
    UMT5 Text Encoder for ACE-Step.

    Encodes text prompts to embeddings using the UMT5-base model.
    Returns MLX arrays for integration with the diffusion model.

    Example:
        >>> encoder = UMT5TextEncoder.from_pretrained("ACE-Step/ACE-Step-v1-3.5B")
        >>> embeddings, mask = encoder.encode("upbeat electronic dance music")
        >>> print(embeddings.shape)  # (1, seq_len, 768)
    """

    def __init__(
        self,
        model: "UMT5EncoderModel",
        tokenizer: "AutoTokenizer",
        config: TextEncoderConfig,
        device: str = "cpu",
    ):
        if not HAS_TRANSFORMERS:
            raise ImportError(
                "transformers library required for text encoding. "
                "Install with: pip install transformers torch"
            )

        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        self.device = device

        # Move model to device
        self.model.to(device)
        self.model.eval()

    @classmethod
    def from_pretrained(
        cls,
        model_path: Union[str, Path],
        device: str = "cpu",
        use_fp16: bool = True,
    ) -> "UMT5TextEncoder":
        """
        Load UMT5 encoder from pretrained weights.

        Args:
            model_path: Path to ACE-Step model or direct path to umt5-base
            device: Device to load model on ("cpu", "cuda", "mps")
            use_fp16: Whether to use FP16 precision

        Returns:
            UMT5TextEncoder instance
        """
        if not HAS_TRANSFORMERS:
            raise ImportError(
                "transformers library required for text encoding. "
                "Install with: pip install transformers torch"
            )

        model_path = Path(model_path)

        # Check for umt5-base subdirectory (ACE-Step structure)
        umt5_path = model_path / "umt5-base"
        if umt5_path.exists():
            model_name = str(umt5_path)
        elif model_path.exists() and (model_path / "config.json").exists():
            model_name = str(model_path)
        else:
            # Fall back to HuggingFace model ID
            model_name = "google/umt5-base"
            print(f"Loading UMT5 from HuggingFace: {model_name}")

        config = TextEncoderConfig(model_name=model_name, use_fp16=use_fp16)

        # Load tokenizer and model
        print(f"Loading UMT5 tokenizer from {model_name}...")
        tokenizer = AutoTokenizer.from_pretrained(model_name)

        print(f"Loading UMT5 encoder from {model_name}...")
        dtype = torch.float16 if use_fp16 and device != "cpu" else torch.float32
        model = UMT5EncoderModel.from_pretrained(
            model_name,
            torch_dtype=dtype,
            low_cpu_mem_usage=True,
        )

        return cls(model, tokenizer, config, device)

    def encode(
        self,
        text: Union[str, List[str]],
        max_length: Optional[int] = None,
    ) -> Tuple[mx.array, mx.array]:
        """
        Encode text to embeddings.

        Args:
            text: Single text string or list of strings
            max_length: Maximum sequence length (defaults to config.max_length)

        Returns:
            Tuple of (embeddings, attention_mask) as MLX arrays
            - embeddings: Shape (batch, seq_len, 768)
            - attention_mask: Shape (batch, seq_len)
        """
        if isinstance(text, str):
            text = [text]

        max_length = max_length or self.config.max_length

        # Tokenize
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        )

        # Move to device
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # Encode
        with torch.no_grad():
            outputs = self.model(**inputs)
            last_hidden_state = outputs.last_hidden_state

        # Convert to MLX arrays
        embeddings = mx.array(last_hidden_state.cpu().numpy())
        attention_mask = mx.array(inputs["attention_mask"].cpu().numpy())

        return embeddings, attention_mask

    def encode_null(
        self,
        batch_size: int = 1,
        seq_length: int = 256,
    ) -> Tuple[mx.array, mx.array]:
        """
        Create null embeddings for classifier-free guidance.

        Args:
            batch_size: Number of null embeddings to create
            seq_length: Sequence length

        Returns:
            Tuple of (embeddings, attention_mask) with zeros
        """
        embeddings = mx.zeros((batch_size, seq_length, self.config.hidden_size))
        attention_mask = mx.zeros((batch_size, seq_length))
        return embeddings, attention_mask

    def __call__(
        self,
        text: Union[str, List[str]],
        max_length: Optional[int] = None,
    ) -> Tuple[mx.array, mx.array]:
        """Alias for encode()."""
        return self.encode(text, max_length)


class PlaceholderTextEncoder:
    """
    Placeholder text encoder when transformers is not available.

    Returns random embeddings for testing purposes only.
    """

    def __init__(self, hidden_size: int = 768):
        self.hidden_size = hidden_size

    def encode(
        self,
        text: Union[str, List[str]],
        max_length: int = 256,
    ) -> Tuple[mx.array, mx.array]:
        """Return placeholder embeddings."""
        if isinstance(text, str):
            batch_size = 1
        else:
            batch_size = len(text)

        # Create placeholder embeddings (not useful for generation)
        embeddings = mx.zeros((batch_size, 64, self.hidden_size))
        attention_mask = mx.ones((batch_size, 64))

        return embeddings, attention_mask

    def encode_null(
        self,
        batch_size: int = 1,
        seq_length: int = 256,
    ) -> Tuple[mx.array, mx.array]:
        """Create null embeddings."""
        embeddings = mx.zeros((batch_size, seq_length, self.hidden_size))
        attention_mask = mx.zeros((batch_size, seq_length))
        return embeddings, attention_mask

    def __call__(
        self,
        text: Union[str, List[str]],
        max_length: int = 256,
    ) -> Tuple[mx.array, mx.array]:
        """Alias for encode()."""
        return self.encode(text, max_length)


def get_text_encoder(
    model_path: Optional[Union[str, Path]] = None,
    device: str = "cpu",
    use_fp16: bool = True,
) -> Union[UMT5TextEncoder, PlaceholderTextEncoder]:
    """
    Get text encoder, with fallback to placeholder if transformers unavailable.

    Args:
        model_path: Path to model weights
        device: Device for PyTorch model
        use_fp16: Whether to use FP16 precision

    Returns:
        UMT5TextEncoder if available, else PlaceholderTextEncoder
    """
    if HAS_TRANSFORMERS and model_path is not None:
        try:
            return UMT5TextEncoder.from_pretrained(
                model_path, device=device, use_fp16=use_fp16
            )
        except Exception as e:
            print(f"Warning: Could not load UMT5 encoder: {e}")
            print("Using placeholder encoder (generation will not work properly)")
            return PlaceholderTextEncoder()
    else:
        if not HAS_TRANSFORMERS:
            print("Warning: transformers not installed, using placeholder encoder")
        return PlaceholderTextEncoder()
