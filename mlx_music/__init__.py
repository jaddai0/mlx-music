"""
MLX Music - Native music generation library for Apple Silicon.

This library provides MLX-native implementations of music generation models,
optimized for Apple Silicon (M1/M2/M3/M4) hardware.

Supported Models:
- ACE-Step: Text-to-music generation with lyrics support (diffusion-based)
- MusicGen: Text-to-music generation (autoregressive LM)
- StableAudio: High-quality text-to-audio generation (diffusion transformer)

Example:
    >>> from mlx_music import ACEStep, MusicGen, StableAudio
    >>>
    >>> # ACE-Step example
    >>> model = ACEStep.from_pretrained("ACE-Step/ACE-Step-v1-3.5B")
    >>> audio = model.generate(
    ...     prompt="upbeat electronic dance music",
    ...     lyrics="Verse 1: Dancing through the night...",
    ...     duration=30.0
    ... )
    >>>
    >>> # MusicGen example
    >>> model = MusicGen.from_pretrained("/path/to/MusicGen-small")
    >>> audio = model.generate(
    ...     prompt="jazz piano with drums",
    ...     duration=10.0
    ... )
    >>>
    >>> # StableAudio example
    >>> model = StableAudio.from_pretrained("stabilityai/stable-audio-open-1.0")
    >>> output = model.generate(
    ...     prompt="ambient electronic music with soft pads",
    ...     duration=30.0,
    ... )
"""

__version__ = "0.1.0"

# Lazy-loaded modules - improves import time significantly
_ace_step = None
_musicgen = None
_stable_audio = None


def _get_ace_step():
    """Lazy load ACEStep model class."""
    global _ace_step
    if _ace_step is None:
        from mlx_music.models.ace_step import ACEStep

        _ace_step = ACEStep
    return _ace_step


def _get_musicgen():
    """Lazy load MusicGen model class."""
    global _musicgen
    if _musicgen is None:
        from mlx_music.models.musicgen import MusicGen

        _musicgen = MusicGen
    return _musicgen


def _get_stable_audio():
    """Lazy load StableAudio model class."""
    global _stable_audio
    if _stable_audio is None:
        from mlx_music.models.stable_audio import StableAudio

        _stable_audio = StableAudio
    return _stable_audio


def __getattr__(name: str):
    """Module-level lazy attribute access."""
    if name == "ACEStep":
        return _get_ace_step()
    if name == "MusicGen":
        return _get_musicgen()
    if name == "StableAudio":
        return _get_stable_audio()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    """List available module attributes."""
    return ["ACEStep", "MusicGen", "StableAudio", "__version__"]


__all__ = ["ACEStep", "MusicGen", "StableAudio", "__version__"]
