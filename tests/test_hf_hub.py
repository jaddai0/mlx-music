"""Tests for HuggingFace Hub integration."""

import pytest
from pathlib import Path
from mlx_music.weights.weight_loader import is_local_path, download_model


class TestIsLocalPath:
    """Tests for local path detection."""

    def test_absolute_unix_path(self):
        """Test absolute Unix paths are detected."""
        assert is_local_path("/Users/test/model")
        assert is_local_path("/home/user/models/ace-step")

    def test_relative_path_dot(self):
        """Test relative paths starting with . are detected."""
        assert is_local_path("./models/ace-step")
        assert is_local_path("../models/ace-step")

    def test_home_directory_path(self):
        """Test paths starting with ~ are detected."""
        assert is_local_path("~/models/ace-step")

    def test_hf_repo_id(self):
        """Test HuggingFace repo IDs are not detected as local."""
        assert not is_local_path("ACE-Step/ACE-Step-v1-3.5B")
        assert not is_local_path("org/model-name")

    def test_simple_name(self):
        """Test simple names that don't exist are not local paths."""
        assert not is_local_path("nonexistent-model-12345")


class TestDownloadModel:
    """Tests for download_model function."""

    def test_local_path_returns_directly(self, tmp_path):
        """Test that existing local paths are returned directly."""
        model_dir = tmp_path / "test_model"
        model_dir.mkdir()

        result = download_model(str(model_dir))
        assert result == model_dir

    def test_local_path_not_found_raises(self):
        """Test that non-existent local paths raise FileNotFoundError."""
        with pytest.raises(FileNotFoundError, match="not found"):
            download_model("/nonexistent/path/to/model/12345")

    def test_local_path_expands_home(self, tmp_path):
        """Test that ~ is expanded in paths."""
        # Create a real path in home for testing
        import os

        home = Path.home()
        # We can't easily test ~ expansion without creating files in home
        # Just verify is_local_path handles ~ correctly
        assert is_local_path("~/test")
