"""Model implementations for mlx-music.

Models are lazily loaded to improve import time.
"""

# Lazy-loaded model classes
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
    return ["ACEStep", "MusicGen", "StableAudio"]


__all__ = ["ACEStep", "MusicGen", "StableAudio"]
