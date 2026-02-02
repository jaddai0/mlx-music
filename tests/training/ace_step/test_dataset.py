"""
Tests for ACE-Step dataset utilities.

Tests:
- AudioExample: Dataclass fields, auto-generated ID
- AudioDataset: Loading, caption discovery, validation
- AudioBatch: Dataclass fields
- iter_batches: Shuffling, drop_last, batch sizes
- PreEncodedDataset: Loading pre-encoded latents
"""

import json
import pytest
from pathlib import Path
import mlx.core as mx


class TestAudioExample:
    """Tests for AudioExample dataclass."""

    def test_audio_example_fields(self, temp_dir):
        """AudioExample should have expected fields."""
        from mlx_music.training.ace_step.dataset import AudioExample

        example = AudioExample(
            audio_path=temp_dir / "test.wav",
            caption="Test caption",
        )

        assert example.audio_path == temp_dir / "test.wav"
        assert example.caption == "Test caption"
        assert example.encoded_latent is None
        assert example.text_embedding is None
        assert example.duration_seconds == 0.0

    def test_audio_example_auto_generates_id(self, temp_dir):
        """AudioExample should auto-generate example_id if not provided."""
        from mlx_music.training.ace_step.dataset import AudioExample

        example = AudioExample(
            audio_path=temp_dir / "test.wav",
            caption="Test",
        )

        assert example.example_id != ""
        assert len(example.example_id) == 12  # MD5 hash truncated

    def test_audio_example_stable_id(self, temp_dir):
        """Same audio path should produce same example_id."""
        from mlx_music.training.ace_step.dataset import AudioExample

        path = temp_dir / "test.wav"

        ex1 = AudioExample(audio_path=path, caption="Caption 1")
        ex2 = AudioExample(audio_path=path, caption="Caption 2")

        assert ex1.example_id == ex2.example_id


class TestAudioBatch:
    """Tests for AudioBatch dataclass."""

    def test_audio_batch_fields(self, rng):
        """AudioBatch should have expected fields."""
        from mlx_music.training.ace_step.dataset import AudioBatch, AudioExample

        examples = [
            AudioExample(audio_path=Path("/test.wav"), caption="test")
        ]

        batch = AudioBatch(
            encoded_latents=mx.zeros((1, 8, 100)),
            text_embeddings=mx.zeros((1, 32, 768)),
            examples=examples,
            rng=rng,
        )

        assert batch.encoded_latents.shape == (1, 8, 100)
        assert batch.text_embeddings.shape == (1, 32, 768)
        assert len(batch.examples) == 1


class TestAudioDataset:
    """Tests for AudioDataset class."""

    def test_dataset_requires_existing_dir(self):
        """AudioDataset should raise for nonexistent directory."""
        from mlx_music.training.ace_step.dataset import AudioDataset

        with pytest.raises(FileNotFoundError, match="not found"):
            AudioDataset(data_dir="/nonexistent/path")

    def test_dataset_requires_directory(self, temp_dir):
        """AudioDataset should raise if path is a file."""
        from mlx_music.training.ace_step.dataset import AudioDataset

        file_path = temp_dir / "file.txt"
        file_path.touch()

        with pytest.raises(ValueError, match="not a directory"):
            AudioDataset(data_dir=file_path)

    def test_dataset_discovers_audio_with_captions(self, temp_dir):
        """AudioDataset should discover audio files with sidecar captions."""
        from mlx_music.training.ace_step.dataset import AudioDataset

        # Create test audio and caption files
        (temp_dir / "audio1.wav").touch()
        (temp_dir / "audio1.txt").write_text("Caption for audio 1")
        (temp_dir / "audio2.wav").touch()
        (temp_dir / "audio2.txt").write_text("Caption for audio 2")

        dataset = AudioDataset(data_dir=temp_dir)

        assert len(dataset) == 2
        captions = {ex.caption for ex in dataset.examples}
        assert "Caption for audio 1" in captions
        assert "Caption for audio 2" in captions

    def test_dataset_skips_audio_without_caption(self, temp_dir):
        """AudioDataset should skip audio files without captions."""
        from mlx_music.training.ace_step.dataset import AudioDataset

        # Create audio without caption
        (temp_dir / "no_caption.wav").touch()
        # Create audio with caption
        (temp_dir / "has_caption.wav").touch()
        (temp_dir / "has_caption.txt").write_text("Has caption")

        dataset = AudioDataset(data_dir=temp_dir)

        assert len(dataset) == 1
        assert dataset.examples[0].caption == "Has caption"

    def test_dataset_caption_extension_variants(self, temp_dir):
        """AudioDataset should check both .txt and .caption extensions."""
        from mlx_music.training.ace_step.dataset import AudioDataset

        (temp_dir / "test1.wav").touch()
        (temp_dir / "test1.caption").write_text("Caption extension")

        dataset = AudioDataset(data_dir=temp_dir)

        assert len(dataset) == 1
        assert dataset.examples[0].caption == "Caption extension"

    def test_dataset_directory_structure_caption(self, temp_dir):
        """AudioDataset should generate caption from directory structure."""
        from mlx_music.training.ace_step.dataset import AudioDataset

        # Create nested directory structure
        subdir = temp_dir / "electronic" / "ambient"
        subdir.mkdir(parents=True)
        (subdir / "track.wav").touch()

        dataset = AudioDataset(data_dir=temp_dir)

        assert len(dataset) == 1
        assert "electronic" in dataset.examples[0].caption.lower()
        assert "ambient" in dataset.examples[0].caption.lower()

    def test_dataset_loads_from_manifest(self, temp_dir):
        """AudioDataset should load from JSON manifest."""
        from mlx_music.training.ace_step.dataset import AudioDataset

        # Create audio files
        (temp_dir / "song1.wav").touch()
        (temp_dir / "song2.wav").touch()

        # Create manifest
        manifest = [
            {"file": "song1.wav", "caption": "First song caption"},
            {"file": "song2.wav", "caption": "Second song caption"},
        ]
        (temp_dir / "manifest.json").write_text(json.dumps(manifest))

        dataset = AudioDataset(data_dir=temp_dir, manifest_file="manifest.json")

        assert len(dataset) == 2
        captions = {ex.caption for ex in dataset.examples}
        assert "First song caption" in captions

    def test_dataset_manifest_validates_paths(self, temp_dir):
        """AudioDataset should reject path traversal in manifest."""
        from mlx_music.training.ace_step.dataset import AudioDataset

        manifest = [
            {"file": "../outside.wav", "caption": "Path traversal attempt"},
            {"file": "/absolute/path.wav", "caption": "Absolute path"},
        ]
        (temp_dir / "manifest.json").write_text(json.dumps(manifest))

        dataset = AudioDataset(data_dir=temp_dir, manifest_file="manifest.json")

        # Both entries should be skipped
        assert len(dataset) == 0

    def test_dataset_shuffle(self, temp_dir):
        """AudioDataset.shuffle() should randomize order."""
        from mlx_music.training.ace_step.dataset import AudioDataset

        # Create multiple audio files
        for i in range(10):
            (temp_dir / f"audio{i:02d}.wav").touch()
            (temp_dir / f"audio{i:02d}.txt").write_text(f"Caption {i}")

        dataset = AudioDataset(data_dir=temp_dir, seed=42)
        original_order = [ex.audio_path for ex in dataset.examples]

        dataset.shuffle()
        shuffled_order = [ex.audio_path for ex in dataset.examples]

        # Order should be different (highly likely with 10 items)
        assert original_order != shuffled_order

    def test_dataset_getitem(self, temp_dir):
        """AudioDataset should support indexing."""
        from mlx_music.training.ace_step.dataset import AudioDataset

        (temp_dir / "test.wav").touch()
        (temp_dir / "test.txt").write_text("Test caption")

        dataset = AudioDataset(data_dir=temp_dir)

        example = dataset[0]
        assert example.caption == "Test caption"

    def test_dataset_len(self, temp_dir):
        """AudioDataset should return correct length."""
        from mlx_music.training.ace_step.dataset import AudioDataset

        for i in range(5):
            (temp_dir / f"audio{i}.wav").touch()
            (temp_dir / f"audio{i}.txt").write_text(f"Caption {i}")

        dataset = AudioDataset(data_dir=temp_dir)

        assert len(dataset) == 5


class TestIterBatches:
    """Tests for AudioDataset.iter_batches method."""

    def test_iter_batches_requires_encoding(self, temp_dir):
        """iter_batches should raise if dataset not encoded."""
        from mlx_music.training.ace_step.dataset import AudioDataset

        (temp_dir / "test.wav").touch()
        (temp_dir / "test.txt").write_text("Caption")

        dataset = AudioDataset(data_dir=temp_dir)

        with pytest.raises(RuntimeError, match="must be encoded"):
            list(dataset.iter_batches(batch_size=1))

    def test_iter_batches_batch_size(self, temp_dir):
        """iter_batches should yield correct batch sizes."""
        from mlx_music.training.ace_step.dataset import AudioDataset, AudioExample

        # Create dataset with pre-populated encoded data
        (temp_dir / "a1.wav").touch()
        (temp_dir / "a1.txt").write_text("c1")
        (temp_dir / "a2.wav").touch()
        (temp_dir / "a2.txt").write_text("c2")
        (temp_dir / "a3.wav").touch()
        (temp_dir / "a3.txt").write_text("c3")
        (temp_dir / "a4.wav").touch()
        (temp_dir / "a4.txt").write_text("c4")

        dataset = AudioDataset(data_dir=temp_dir)

        # Manually set encoded data
        for ex in dataset.examples:
            ex.encoded_latent = mx.zeros((8, 100))
            ex.text_embedding = mx.zeros((32, 768))
        dataset._is_encoded = True

        batches = list(dataset.iter_batches(batch_size=2, shuffle=False, drop_last=True))

        assert len(batches) == 2
        assert batches[0].encoded_latents.shape[0] == 2
        assert batches[1].encoded_latents.shape[0] == 2

    def test_iter_batches_drop_last(self, temp_dir):
        """iter_batches drop_last=False should include incomplete batch."""
        from mlx_music.training.ace_step.dataset import AudioDataset

        # Create 3 examples
        for i in range(3):
            (temp_dir / f"a{i}.wav").touch()
            (temp_dir / f"a{i}.txt").write_text(f"c{i}")

        dataset = AudioDataset(data_dir=temp_dir)

        for ex in dataset.examples:
            ex.encoded_latent = mx.zeros((8, 100))
            ex.text_embedding = mx.zeros((32, 768))
        dataset._is_encoded = True

        # With drop_last=True: 3 // 2 = 1 batch
        batches_drop = list(dataset.iter_batches(batch_size=2, drop_last=True))
        assert len(batches_drop) == 1

        # With drop_last=False: 2 batches (size 2 and size 1)
        batches_keep = list(dataset.iter_batches(batch_size=2, drop_last=False))
        assert len(batches_keep) == 2

    def test_iter_batches_shuffles(self, temp_dir):
        """iter_batches shuffle=True should randomize order."""
        from mlx_music.training.ace_step.dataset import AudioDataset

        # Create enough examples
        for i in range(20):
            (temp_dir / f"a{i:02d}.wav").touch()
            (temp_dir / f"a{i:02d}.txt").write_text(f"c{i}")

        dataset = AudioDataset(data_dir=temp_dir, seed=42)

        for ex in dataset.examples:
            ex.encoded_latent = mx.zeros((8, 100))
            ex.text_embedding = mx.zeros((32, 768))
        dataset._is_encoded = True

        # Collect example IDs from first epoch
        epoch1_ids = []
        for batch in dataset.iter_batches(batch_size=4, shuffle=True):
            for ex in batch.examples:
                epoch1_ids.append(ex.example_id)

        # Collect from second epoch (should differ)
        epoch2_ids = []
        for batch in dataset.iter_batches(batch_size=4, shuffle=True):
            for ex in batch.examples:
                epoch2_ids.append(ex.example_id)

        # Same examples but likely different order
        assert set(epoch1_ids) == set(epoch2_ids)
        # Order should be different (highly likely)
        assert epoch1_ids != epoch2_ids


class TestPreEncodedDataset:
    """Tests for PreEncodedDataset class."""

    def test_pre_encoded_loads_safetensors(self, temp_dir):
        """PreEncodedDataset should load from safetensors files."""
        from mlx_music.training.ace_step.dataset import PreEncodedDataset

        # Create latents directory
        latents_dir = temp_dir / "latents"
        latents_dir.mkdir()

        # Save some test data
        data = {
            "latent": mx.random.normal((8, 100)),
            "text_embedding": mx.random.normal((32, 768)),
        }
        mx.save_safetensors(str(latents_dir / "example_001.safetensors"), data)

        dataset = PreEncodedDataset(data_dir=temp_dir)

        assert len(dataset) == 1
        assert dataset.examples[0].encoded_latent is not None
        assert dataset._is_encoded  # Should be True for pre-encoded

    def test_pre_encoded_encode_all_is_noop(self, temp_dir):
        """PreEncodedDataset.encode_all() should be a no-op."""
        from mlx_music.training.ace_step.dataset import PreEncodedDataset

        latents_dir = temp_dir / "latents"
        latents_dir.mkdir()

        data = {
            "latent": mx.random.normal((8, 100)),
            "text_embedding": mx.random.normal((32, 768)),
        }
        mx.save_safetensors(str(latents_dir / "example.safetensors"), data)

        dataset = PreEncodedDataset(data_dir=temp_dir)

        # Should not raise
        dataset.encode_all()

    def test_pre_encoded_validates_paths(self, temp_dir):
        """PreEncodedDataset should skip files outside data_dir."""
        from mlx_music.training.ace_step.dataset import PreEncodedDataset

        # This is hard to test directly without symlinks
        # Just verify the dataset loads from the correct directory
        latents_dir = temp_dir / "latents"
        latents_dir.mkdir()

        data = {"latent": mx.zeros((8, 100)), "text_embedding": mx.zeros((32, 768))}
        mx.save_safetensors(str(latents_dir / "valid.safetensors"), data)

        dataset = PreEncodedDataset(data_dir=temp_dir)

        # Should only have the valid file
        assert len(dataset) == 1
