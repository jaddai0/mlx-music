"""
Dataset utilities for ACE-Step training.

Provides audio + text caption dataset loading with:
- Audio file discovery (wav, flac, mp3, ogg)
- Caption/prompt loading from sidecar files or directory structure
- VAE encoding for latent space training
- Caching of encoded latents for faster subsequent runs
"""

import hashlib
import json
import logging
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

import mlx.core as mx

logger = logging.getLogger(__name__)

# Supported audio formats
AUDIO_EXTENSIONS = {".wav", ".flac", ".mp3", ".ogg", ".m4a"}


@dataclass
class AudioExample:
    """A single training example with audio and text caption."""

    audio_path: Path
    """Path to the audio file."""

    caption: str
    """Text description/prompt for the audio."""

    encoded_latent: Optional[mx.array] = None
    """VAE-encoded audio latent (populated after encoding)."""

    text_embedding: Optional[mx.array] = None
    """Text encoder embedding (populated after encoding)."""

    example_id: str = ""
    """Unique identifier for caching."""

    duration_seconds: float = 0.0
    """Audio duration in seconds."""

    def __post_init__(self):
        if not self.example_id:
            # Generate stable ID from path
            self.example_id = hashlib.md5(str(self.audio_path).encode()).hexdigest()[:12]


@dataclass
class AudioBatch:
    """A batch of audio examples for training."""

    encoded_latents: mx.array
    """Stacked VAE-encoded latents [B, C, T]."""

    text_embeddings: mx.array
    """Stacked text embeddings [B, seq, dim]."""

    examples: List[AudioExample]
    """Original examples for reference."""

    rng: random.Random
    """Random generator for reproducibility."""


class AudioDataset:
    """
    Dataset for audio + caption training.

    Supports multiple ways to specify captions:
    1. Sidecar files: audio.wav + audio.txt (or audio.caption)
    2. JSON manifest: dataset.json with {"file": "path", "caption": "text"}
    3. Directory structure: genre/subgenre/file.wav with auto-generated captions

    Example:
        dataset = AudioDataset(
            data_dir=Path("./training_audio"),
            sample_rate=44100,
            max_duration=30.0,
        )
        print(f"Found {len(dataset)} examples")

        # Encode all examples with VAE and text encoder
        dataset.encode_all(vae=vae, text_encoder=text_encoder)

        # Get batches for training
        for batch in dataset.iter_batches(batch_size=4):
            loss = train_step(batch.encoded_latents, batch.text_embeddings)
    """

    def __init__(
        self,
        data_dir: Path | str,
        sample_rate: int = 44100,
        max_duration: float = 30.0,
        caption_extension: str = ".txt",
        manifest_file: Optional[str] = None,
        seed: int = 42,
    ):
        """
        Initialize dataset from a directory.

        Args:
            data_dir: Root directory containing audio files. Must exist.
            sample_rate: Target sample rate for audio
            max_duration: Maximum audio duration in seconds
            caption_extension: Extension for caption sidecar files
            manifest_file: Optional JSON manifest file name
            seed: Random seed for shuffling

        Raises:
            FileNotFoundError: If data_dir doesn't exist.
            ValueError: If data_dir is not a directory.
        """
        self.data_dir = Path(data_dir).resolve()

        # Validate data directory exists
        if not self.data_dir.exists():
            raise FileNotFoundError(f"Data directory not found: {self.data_dir}")
        if not self.data_dir.is_dir():
            raise ValueError(f"Data path is not a directory: {self.data_dir}")

        self.sample_rate = sample_rate
        self.max_duration = max_duration
        self.caption_extension = caption_extension
        self.seed = seed
        self.rng = random.Random(seed)

        self.examples: List[AudioExample] = []
        self._is_encoded = False

        # Load examples
        if manifest_file and (self.data_dir / manifest_file).exists():
            self._load_from_manifest(self.data_dir / manifest_file)
        else:
            self._discover_audio_files()

        logger.info(f"Loaded {len(self.examples)} audio examples from {self.data_dir}")

    def _discover_audio_files(self) -> None:
        """Discover audio files and their captions."""
        for ext in AUDIO_EXTENSIONS:
            for audio_path in self.data_dir.rglob(f"*{ext}"):
                caption = self._find_caption(audio_path)
                if caption:
                    self.examples.append(
                        AudioExample(
                            audio_path=audio_path,
                            caption=caption,
                        )
                    )
                else:
                    logger.warning(f"No caption found for {audio_path}, skipping")

    def _find_caption(self, audio_path: Path) -> Optional[str]:
        """Find caption for an audio file."""
        # Try sidecar file
        caption_path = audio_path.with_suffix(self.caption_extension)
        if caption_path.exists():
            return caption_path.read_text().strip()

        # Try .caption extension
        caption_path = audio_path.with_suffix(".caption")
        if caption_path.exists():
            return caption_path.read_text().strip()

        # Generate from directory structure
        # e.g., "electronic/ambient/track01.wav" -> "electronic ambient music"
        rel_path = audio_path.relative_to(self.data_dir)
        if len(rel_path.parts) > 1:
            tags = [p.replace("_", " ").replace("-", " ") for p in rel_path.parts[:-1]]
            return f"{' '.join(tags)} music"

        return None

    def _load_from_manifest(self, manifest_path: Path) -> None:
        """
        Load examples from a JSON manifest.

        Security: Validates that all file paths are within data_dir
        to prevent path traversal attacks.
        """
        with open(manifest_path) as f:
            data = json.load(f)

        for entry in data:
            # Security: Validate manifest entries have required fields
            if "file" not in entry or "caption" not in entry:
                logger.warning(f"Manifest entry missing required fields: {entry}")
                continue

            # Security: Prevent path traversal via manifest file paths
            file_path = entry["file"]
            if ".." in str(file_path) or str(file_path).startswith("/"):
                logger.warning(f"Skipping invalid file path in manifest: {file_path}")
                continue

            audio_path = (self.data_dir / file_path).resolve()

            # Security: Verify resolved path is within data_dir
            try:
                audio_path.relative_to(self.data_dir)
            except ValueError:
                logger.warning(
                    f"Skipping file outside data directory: {file_path} -> {audio_path}"
                )
                continue

            if audio_path.exists():
                self.examples.append(
                    AudioExample(
                        audio_path=audio_path,
                        caption=str(entry["caption"])[:10000],  # Limit caption length
                        duration_seconds=float(entry.get("duration", 0.0)),
                    )
                )
            else:
                logger.warning(f"Audio file not found: {audio_path}")

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> AudioExample:
        return self.examples[idx]

    def shuffle(self) -> None:
        """Shuffle examples in-place."""
        self.rng.shuffle(self.examples)

    def encode_all(
        self,
        vae: Any,  # ACE-Step VAE/DCAE
        text_encoder: Any,  # ACE-Step text encoder
        cache_dir: Optional[Path] = None,
        show_progress: bool = True,
    ) -> None:
        """
        Encode all examples with VAE and text encoder.

        Args:
            vae: VAE model for audio encoding
            text_encoder: Text encoder for caption embedding
            cache_dir: Optional directory for caching encoded latents
            show_progress: Show progress bar
        """
        # Try to load from cache
        if cache_dir is not None:
            cache_key = self._compute_cache_key()
            if self._load_cache(cache_dir, cache_key):
                logger.info("Loaded encoded dataset from cache")
                self._is_encoded = True
                return

        # Encode examples
        try:
            from tqdm import tqdm
            iterator = tqdm(self.examples, desc="Encoding dataset") if show_progress else self.examples
        except ImportError:
            iterator = self.examples

        for example in iterator:
            # Load and encode audio
            # NOTE: Actual implementation depends on ACE-Step's audio loading
            # This is a placeholder for the encoding pipeline
            example.encoded_latent = self._encode_audio(vae, example.audio_path)
            example.text_embedding = self._encode_text(text_encoder, example.caption)

        self._is_encoded = True

        # Save to cache
        if cache_dir is not None:
            self._save_cache(cache_dir, cache_key)

    def _encode_audio(self, vae: Any, audio_path: Path) -> mx.array:
        """
        Encode audio file to latent space.

        NOTE: This is a placeholder. Actual implementation depends on
        ACE-Step's audio loading and VAE encoding pipeline.
        """
        # Placeholder - actual implementation would:
        # 1. Load audio with librosa/soundfile
        # 2. Resample to target sample rate
        # 3. Normalize and pad/trim to max_duration
        # 4. Pass through VAE encoder
        raise NotImplementedError(
            "Audio encoding requires ACE-Step VAE. "
            "Override this method or use pre-encoded latents."
        )

    def _encode_text(self, text_encoder: Any, caption: str) -> mx.array:
        """
        Encode caption to text embedding.

        NOTE: This is a placeholder. Actual implementation depends on
        ACE-Step's text encoder.
        """
        # Placeholder - actual implementation would call text encoder
        raise NotImplementedError(
            "Text encoding requires ACE-Step text encoder. "
            "Override this method or use pre-encoded embeddings."
        )

    def _compute_cache_key(self) -> str:
        """Compute unique cache key for this dataset configuration."""
        key_parts = [
            f"sr:{self.sample_rate}",
            f"maxdur:{self.max_duration}",
            f"n:{len(self.examples)}",
        ]
        for ex in sorted(self.examples, key=lambda x: str(x.audio_path)):
            key_parts.append(f"{ex.audio_path.name}:{ex.caption[:50]}")

        return hashlib.sha256("|".join(key_parts).encode()).hexdigest()[:16]

    def _save_cache(self, cache_dir: Path, cache_key: str) -> None:
        """Save encoded latents to cache."""
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_path = cache_dir / f"audio_cache_{cache_key}.safetensors"
        meta_path = cache_dir / f"audio_cache_{cache_key}.json"

        arrays = {}
        metadata = {"examples": []}

        for i, ex in enumerate(self.examples):
            if ex.encoded_latent is not None:
                arrays[f"latent_{i}"] = ex.encoded_latent
            if ex.text_embedding is not None:
                arrays[f"text_{i}"] = ex.text_embedding

            metadata["examples"].append({
                "id": ex.example_id,
                "audio_path": str(ex.audio_path),
                "caption": ex.caption,
            })

        mx.save_safetensors(str(cache_path), arrays)
        with open(meta_path, "w") as f:
            json.dump(metadata, f, indent=2)

        logger.info(f"Saved dataset cache: {cache_path}")

    def _load_cache(self, cache_dir: Path, cache_key: str) -> bool:
        """
        Load encoded latents from cache. Returns True if successful.

        Security: Only loads from .safetensors files within the cache directory.
        """
        # Security: Validate cache_key to prevent path injection
        if not cache_key.isalnum():
            logger.warning(f"Invalid cache key format: {cache_key}")
            return False

        cache_dir = Path(cache_dir).resolve()
        cache_path = cache_dir / f"audio_cache_{cache_key}.safetensors"
        meta_path = cache_dir / f"audio_cache_{cache_key}.json"

        if not cache_path.exists() or not meta_path.exists():
            return False

        # Security: Verify paths are within cache_dir
        try:
            cache_path.resolve().relative_to(cache_dir)
            meta_path.resolve().relative_to(cache_dir)
        except ValueError:
            logger.warning("Cache path traversal attempt detected")
            return False

        try:
            # safetensors format is safe - no arbitrary code execution
            arrays = mx.load(str(cache_path))
            with open(meta_path) as f:
                metadata = json.load(f)

            # Restore encoded data to examples
            for i, ex in enumerate(self.examples):
                latent_key = f"latent_{i}"
                text_key = f"text_{i}"
                if latent_key in arrays:
                    ex.encoded_latent = arrays[latent_key]
                if text_key in arrays:
                    ex.text_embedding = arrays[text_key]

            return True
        except Exception as e:
            logger.warning(f"Failed to load cache: {e}")
            return False

    def iter_batches(
        self,
        batch_size: int,
        shuffle: bool = True,
        drop_last: bool = True,
    ) -> Iterator[AudioBatch]:
        """
        Iterate over batches of examples.

        Args:
            batch_size: Number of examples per batch
            shuffle: Whether to shuffle before each epoch
            drop_last: Whether to drop incomplete final batch

        Yields:
            AudioBatch objects with stacked tensors

        Raises:
            RuntimeError: If dataset is not encoded or examples have None tensors
        """
        if not self._is_encoded:
            raise RuntimeError("Dataset must be encoded first. Call encode_all().")

        if len(self.examples) == 0:
            return  # Empty dataset yields nothing

        indices = list(range(len(self.examples)))
        if shuffle:
            self.rng.shuffle(indices)

        for start in range(0, len(indices), batch_size):
            end = start + batch_size
            if end > len(indices) and drop_last:
                break

            batch_indices = indices[start:end]
            batch_examples = [self.examples[i] for i in batch_indices]

            # Validate tensors are encoded before stacking
            for ex in batch_examples:
                if ex.encoded_latent is None:
                    raise RuntimeError(
                        f"Example {ex.example_id} has no encoded_latent. "
                        "Ensure encode_all() completed successfully."
                    )
                if ex.text_embedding is None:
                    raise RuntimeError(
                        f"Example {ex.example_id} has no text_embedding. "
                        "Ensure encode_all() completed successfully."
                    )

            # Stack tensors
            latents = mx.stack([ex.encoded_latent for ex in batch_examples])
            embeddings = mx.stack([ex.text_embedding for ex in batch_examples])

            yield AudioBatch(
                encoded_latents=latents,
                text_embeddings=embeddings,
                examples=batch_examples,
                rng=self.rng,
            )


class PreEncodedDataset(AudioDataset):
    """
    Dataset loading pre-encoded latents and embeddings.

    Use this when you've pre-computed VAE latents and text embeddings
    to avoid re-encoding on each training run.

    Expected directory structure:
        data_dir/
            latents/
                example_001.safetensors  # Contains "latent" and "text_embedding"
                example_002.safetensors
                ...
            manifest.json  # Optional metadata

    Or:
        data_dir/
            latents/
                *.safetensors
            embeddings/
                *.safetensors (matching names)

    Security Note:
        Only .safetensors files are loaded. This format is safe because it
        stores only tensor data without arbitrary Python objects (unlike pickle).
    """

    def __init__(
        self,
        data_dir: Path | str,
        seed: int = 42,
    ):
        self.data_dir = Path(data_dir).resolve()

        # Validate data directory exists
        if not self.data_dir.exists():
            raise FileNotFoundError(f"Data directory not found: {self.data_dir}")
        if not self.data_dir.is_dir():
            raise ValueError(f"Data path is not a directory: {self.data_dir}")

        self.seed = seed
        self.rng = random.Random(seed)
        self.examples: List[AudioExample] = []
        self._is_encoded = True  # Pre-encoded by definition

        self._load_pre_encoded()
        logger.info(f"Loaded {len(self.examples)} pre-encoded examples")

    def _load_pre_encoded(self) -> None:
        """
        Load pre-encoded latents and embeddings from safetensors files.

        Security Note:
            - Only loads .safetensors files (safe tensor serialization format)
            - Validates all paths are within data_dir to prevent symlink attacks
            - safetensors format does NOT use Python's pickle module
        """
        latents_dir = self.data_dir / "latents"

        if not latents_dir.exists():
            # Try flat structure
            latents_dir = self.data_dir

        for latent_path in sorted(latents_dir.glob("*.safetensors")):
            # Security: Verify path is within data_dir (prevent symlink attacks)
            resolved_path = latent_path.resolve()
            try:
                resolved_path.relative_to(self.data_dir)
            except ValueError:
                logger.warning(f"Skipping file outside data directory: {latent_path}")
                continue

            try:
                # Load from safetensors (safe format - stores tensors only)
                data = mx.load(str(resolved_path))
                example = AudioExample(
                    audio_path=resolved_path,  # Use latent path as identifier
                    caption="",  # Caption not needed for pre-encoded
                    encoded_latent=data.get("latent", data.get("encoded_latent")),
                    text_embedding=data.get("text_embedding", data.get("embedding")),
                    example_id=latent_path.stem,
                )
                self.examples.append(example)
            except Exception as e:
                logger.warning(f"Failed to load {latent_path}: {e}")

    def encode_all(self, *args, **kwargs) -> None:
        """No-op for pre-encoded datasets."""
        pass
