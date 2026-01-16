"""
T5 Text Encoder for Stable Audio Open.

Encodes text prompts to embeddings for conditioning the diffusion model.
Uses HuggingFace transformers for the encoder, with MLX array outputs.
"""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple, Union

import mlx.core as mx
import numpy as np

logger = logging.getLogger(__name__)

try:
    import torch
    from transformers import AutoTokenizer, T5EncoderModel

    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False


@dataclass
class TextEncoderConfig:
    """Configuration for T5 text encoder."""

    model_name: str = "t5-base"
    max_length: int = 256
    hidden_size: int = 768
    use_fp16: bool = True


class T5TextEncoder:
    """
    T5 Text Encoder for Stable Audio Open.

    Encodes text prompts to embeddings using the T5 model.
    Returns both sequence embeddings (for cross-attention) and
    pooled embeddings (for global conditioning).

    Example:
        >>> encoder = T5TextEncoder.from_pretrained("stabilityai/stable-audio-open-1.0")
        >>> text_embeds, pooled = encoder.encode("ambient electronic music")
        >>> print(text_embeds.shape)  # (1, seq_len, 768)
        >>> print(pooled.shape)  # (1, 768)
    """

    def __init__(
        self,
        model: "T5EncoderModel",
        tokenizer: "AutoTokenizer",
        config: TextEncoderConfig,
        device: str = "cpu",
    ):
        if not HAS_TRANSFORMERS:
            raise ImportError(
                "transformers library required for text encoding. "
                "Install with: pip install 'mlx-music[text-encoder]'"
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
    ) -> "T5TextEncoder":
        """
        Load T5 encoder from pretrained weights.

        Args:
            model_path: Path to Stable Audio model or direct path to T5
            device: Device to load model on ("cpu", "cuda", "mps")
            use_fp16: Whether to use FP16 precision

        Returns:
            T5TextEncoder instance
        """
        if not HAS_TRANSFORMERS:
            raise ImportError(
                "transformers library required for text encoding. "
                "Install with: pip install 'mlx-music[text-encoder]'"
            )

        model_path = Path(model_path)

        # Check for text_encoder subdirectory (diffusers layout)
        text_encoder_path = model_path / "text_encoder"
        if text_encoder_path.exists():
            model_name = str(text_encoder_path)
        elif model_path.exists() and (model_path / "config.json").exists():
            model_name = str(model_path)
        else:
            # Fall back to HuggingFace model ID (t5-base is commonly used)
            model_name = "t5-base"
            logger.info(f"Loading T5 from HuggingFace: {model_name}")

        config = TextEncoderConfig(model_name=model_name, use_fp16=use_fp16)

        # Load tokenizer and model
        logger.info(f"Loading T5 tokenizer from {model_name}...")
        tokenizer = AutoTokenizer.from_pretrained(model_name)

        logger.info(f"Loading T5 encoder from {model_name}...")
        dtype = torch.float16 if use_fp16 and device != "cpu" else torch.float32
        model = T5EncoderModel.from_pretrained(
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
            Tuple of (text_embeds, pooled_embeds) as MLX arrays
            - text_embeds: Shape (batch, seq_len, hidden_size)
            - pooled_embeds: Shape (batch, hidden_size) - mean pooled
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

        # Create pooled embedding (mean over sequence, considering attention mask)
        attention_mask = inputs["attention_mask"]
        mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size())
        sum_embeddings = (last_hidden_state * mask_expanded).sum(dim=1)
        mask_sum = attention_mask.sum(dim=1, keepdim=True).clamp(min=1)
        pooled = sum_embeddings / mask_sum

        # Convert to MLX arrays
        text_embeds = mx.array(last_hidden_state.cpu().numpy())
        pooled_embeds = mx.array(pooled.cpu().numpy())

        return text_embeds, pooled_embeds

    def encode_null(
        self,
        batch_size: int = 1,
        seq_length: int = 64,
    ) -> Tuple[mx.array, mx.array]:
        """
        Create null embeddings for classifier-free guidance.

        Args:
            batch_size: Number of null embeddings to create
            seq_length: Sequence length

        Returns:
            Tuple of (text_embeds, pooled_embeds) with zeros
        """
        text_embeds = mx.zeros((batch_size, seq_length, self.config.hidden_size))
        pooled_embeds = mx.zeros((batch_size, self.config.hidden_size))
        return text_embeds, pooled_embeds

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

    Returns zero embeddings. Generation will not be conditioned on text.
    """

    def __init__(self, hidden_size: int = 768):
        self.hidden_size = hidden_size
        logger.warning(
            "Using placeholder text encoder. "
            "Text prompts will NOT condition generation. "
            "Install with: pip install 'mlx-music[text-encoder]'"
        )

    def encode(
        self,
        text: Union[str, List[str]],
        max_length: int = 64,
    ) -> Tuple[mx.array, mx.array]:
        """Return placeholder embeddings."""
        if isinstance(text, str):
            batch_size = 1
        else:
            batch_size = len(text)

        # Create placeholder embeddings
        text_embeds = mx.zeros((batch_size, max_length, self.hidden_size))
        pooled_embeds = mx.zeros((batch_size, self.hidden_size))

        return text_embeds, pooled_embeds

    def encode_null(
        self,
        batch_size: int = 1,
        seq_length: int = 64,
    ) -> Tuple[mx.array, mx.array]:
        """Create null embeddings."""
        text_embeds = mx.zeros((batch_size, seq_length, self.hidden_size))
        pooled_embeds = mx.zeros((batch_size, self.hidden_size))
        return text_embeds, pooled_embeds

    def __call__(
        self,
        text: Union[str, List[str]],
        max_length: int = 64,
    ) -> Tuple[mx.array, mx.array]:
        """Alias for encode()."""
        return self.encode(text, max_length)


def get_text_encoder(
    model_path: Optional[Union[str, Path]] = None,
    device: Optional[str] = None,
    use_fp16: bool = True,
) -> Union[T5TextEncoder, PlaceholderTextEncoder]:
    """
    Get text encoder, with fallback to placeholder if transformers unavailable.

    Args:
        model_path: Path to model weights
        device: Device for PyTorch model (auto-detected if None)
        use_fp16: Whether to use FP16 precision

    Returns:
        T5TextEncoder if available, else PlaceholderTextEncoder
    """
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

    if HAS_TRANSFORMERS and model_path is not None:
        try:
            return T5TextEncoder.from_pretrained(
                model_path, device=device, use_fp16=use_fp16
            )
        except Exception as e:
            logger.warning(f"Could not load T5 encoder: {e}")
            logger.warning("Using placeholder encoder (text will not condition generation)")
            return PlaceholderTextEncoder()
    else:
        if not HAS_TRANSFORMERS:
            logger.warning(
                "transformers not installed. Using placeholder encoder. "
                "Install with: pip install 'mlx-music[text-encoder]'"
            )
        return PlaceholderTextEncoder()


__all__ = [
    "T5TextEncoder",
    "PlaceholderTextEncoder",
    "TextEncoderConfig",
    "get_text_encoder",
]
