"""Device detection utilities for mlx-music."""

import logging
import platform
from functools import lru_cache
from typing import Optional

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def get_default_torch_device() -> str:
    """
    Get the default PyTorch device for the current system.

    Checks for available accelerators in order: CUDA, MPS, CPU.
    Results are cached to avoid repeated checks.

    Returns:
        Device string: "cuda", "mps", or "cpu"
    """
    system = platform.system()

    if system == "Darwin":
        # macOS - check for MPS (Apple Silicon)
        try:
            import torch

            if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                return "mps"
        except ImportError:
            pass
        return "cpu"
    else:
        # Linux/Windows - check for CUDA
        try:
            import torch

            if torch.cuda.is_available():
                return "cuda"
        except ImportError:
            pass
        return "cpu"


def get_device(device: Optional[str] = None) -> str:
    """
    Get the device to use, with auto-detection if not specified.

    Args:
        device: Explicit device ("cpu", "cuda", "mps") or None for auto-detect

    Returns:
        Device string to use
    """
    if device is not None:
        return device
    return get_default_torch_device()


__all__ = ["get_default_torch_device", "get_device"]
