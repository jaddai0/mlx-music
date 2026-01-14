"""Tests for lazy module loading."""

import sys
import time


def test_lazy_import_speed():
    """Test that package import is fast (lazy loading working)."""
    # Clear any cached imports
    modules_to_clear = [m for m in sys.modules if m.startswith("mlx_music")]
    for m in modules_to_clear:
        del sys.modules[m]

    start = time.time()
    import mlx_music

    import_time = time.time() - start

    # Package import should be very fast (< 100ms) with lazy loading
    assert import_time < 0.5, f"Package import took {import_time:.2f}s, expected < 0.5s"


def test_lazy_version_access():
    """Test that __version__ is accessible without loading models."""
    import mlx_music

    assert hasattr(mlx_music, "__version__")
    assert mlx_music.__version__ == "0.1.0"


def test_lazy_ace_step_access():
    """Test that ACEStep is lazily loaded on first access."""
    import mlx_music

    # Access ACEStep - this triggers lazy load
    ACEStep = mlx_music.ACEStep

    assert ACEStep is not None
    assert hasattr(ACEStep, "from_pretrained")
    assert hasattr(ACEStep, "generate")


def test_lazy_models_module():
    """Test that models module also has lazy loading."""
    from mlx_music import models

    # Access ACEStep from models
    ACEStep = models.ACEStep

    assert ACEStep is not None
    assert hasattr(ACEStep, "from_pretrained")


def test_module_dir():
    """Test that __dir__ returns expected attributes."""
    import mlx_music

    attrs = dir(mlx_music)
    assert "ACEStep" in attrs
    assert "__version__" in attrs
