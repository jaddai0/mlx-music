"""Tests for HuggingFace Hub integration and security."""

import json
import pytest
from pathlib import Path
from mlx_music.weights.weight_loader import (
    is_local_path,
    download_model,
    _validate_safe_path,
    PathTraversalError,
    load_sharded_safetensors,
)


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

    def test_empty_model_id_raises(self):
        """Test that empty model_id raises ValueError."""
        with pytest.raises(ValueError, match="non-empty string"):
            download_model("")

        with pytest.raises(ValueError, match="empty or whitespace"):
            download_model("   ")

    def test_none_model_id_raises(self):
        """Test that None model_id raises ValueError."""
        with pytest.raises(ValueError, match="non-empty string"):
            download_model(None)


class TestPathTraversalSecurity:
    """Tests for path traversal security."""

    def test_valid_filename(self, tmp_path):
        """Test that valid filenames pass validation."""
        safe_path = _validate_safe_path(tmp_path, "model.safetensors")
        assert safe_path == tmp_path / "model.safetensors"

    def test_valid_nested_filename(self, tmp_path):
        """Test that valid nested filenames pass validation."""
        subdir = tmp_path / "subdir"
        subdir.mkdir()
        safe_path = _validate_safe_path(tmp_path, "subdir/model.safetensors")
        assert safe_path.parent == subdir

    def test_path_traversal_dotdot(self, tmp_path):
        """Test that .. path traversal is blocked."""
        with pytest.raises(PathTraversalError, match="path traversal"):
            _validate_safe_path(tmp_path, "../escape.txt")

        with pytest.raises(PathTraversalError, match="path traversal"):
            _validate_safe_path(tmp_path, "subdir/../../../etc/passwd")

    def test_path_traversal_absolute(self, tmp_path):
        """Test that absolute paths are blocked."""
        with pytest.raises(PathTraversalError, match="path traversal"):
            _validate_safe_path(tmp_path, "/etc/passwd")

        with pytest.raises(PathTraversalError, match="path traversal"):
            _validate_safe_path(tmp_path, "\\etc\\passwd")

    def test_path_traversal_windows_drive(self, tmp_path):
        """Test that Windows drive letters are blocked."""
        with pytest.raises(PathTraversalError, match="Windows path"):
            _validate_safe_path(tmp_path, "C:\\Windows\\System32")

    def test_sharded_loading_validates_paths(self, tmp_path):
        """Test that load_sharded_safetensors validates index file paths."""
        # Create a malicious index file with path traversal
        index_file = tmp_path / "model.safetensors.index.json"
        malicious_index = {
            "weight_map": {
                "layer.weight": "../../../etc/passwd"
            }
        }
        with open(index_file, "w") as f:
            json.dump(malicious_index, f)

        with pytest.raises(PathTraversalError, match="path traversal"):
            load_sharded_safetensors(tmp_path)

    def test_sharded_loading_validates_absolute_paths(self, tmp_path):
        """Test that load_sharded_safetensors blocks absolute paths in index."""
        index_file = tmp_path / "model.safetensors.index.json"
        malicious_index = {
            "weight_map": {
                "layer.weight": "/etc/passwd"
            }
        }
        with open(index_file, "w") as f:
            json.dump(malicious_index, f)

        with pytest.raises(PathTraversalError, match="path traversal"):
            load_sharded_safetensors(tmp_path)

    def test_sharded_loading_validates_index_structure(self, tmp_path):
        """Test that load_sharded_safetensors validates index structure."""
        index_file = tmp_path / "model.safetensors.index.json"

        # Test invalid weight_map type
        with open(index_file, "w") as f:
            json.dump({"weight_map": "not_a_dict"}, f)
        with pytest.raises(ValueError, match="weight_map must be a dict"):
            load_sharded_safetensors(tmp_path)

        # Test invalid shard filename type
        with open(index_file, "w") as f:
            json.dump({"weight_map": {"layer.weight": 12345}}, f)
        with pytest.raises(ValueError, match="shard filename must be string"):
            load_sharded_safetensors(tmp_path)
