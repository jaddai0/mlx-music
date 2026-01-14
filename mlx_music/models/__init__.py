"""Model implementations for mlx-music.

Models are lazily loaded to improve import time.
"""

# Lazy-loaded model classes
_ace_step = None


def _get_ace_step():
    """Lazy load ACEStep model class."""
    global _ace_step
    if _ace_step is None:
        from mlx_music.models.ace_step import ACEStep

        _ace_step = ACEStep
    return _ace_step


def __getattr__(name: str):
    """Module-level lazy attribute access."""
    if name == "ACEStep":
        return _get_ace_step()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    """List available module attributes."""
    return ["ACEStep"]


__all__ = ["ACEStep"]
