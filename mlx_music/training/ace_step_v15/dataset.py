"""
Dataset utilities for ACE-Step v1.5 training.

Provides audio + text caption dataset loading with:
- 48kHz stereo audio loading (Oobleck VAE target format)
- VAE encoding to 25Hz, 64-dim latent space
- Text and lyric embedding support
- Caching of encoded latents for faster subsequent runs

Key differences from v1:
- 48kHz stereo instead of 44.1kHz
- Oobleck VAE with 64-dim latents at 25Hz
- Separate text and lyric embeddings
- Lyrics support via sidecar .lyrics files
"""

import hashlib
import json
import logging
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional

import mlx.core as mx

logger = logging.getLogger(__name__)

AUDIO_EXTENSIONS = {".wav", ".flac", ".mp3", ".ogg", ".m4a"}

# Oobleck VAE parameters
SAMPLE_RATE = 48000
LATENT_HZ = 25
VAE_HOP = 1920  # 48000 / 25
LATENT_DIM = 64


def _force_evaluate(*args) -> None:
    """Force MLX lazy evaluation. This is mlx.core.eval, NOT Python's eval()."""
    mx.eval(*args)


@dataclass
class AudioExampleV15:
    """A single v1.5 training example."""

    audio_path: Path
    caption: str
    lyrics: str = ""
    encoded_latent: Optional[mx.array] = None
    text_embedding: Optional[mx.array] = None
    lyric_embedding: Optional[mx.array] = None
    example_id: str = ""
    duration_seconds: float = 0.0

    def __post_init__(self):
        if not self.example_id:
            self.example_id = hashlib.md5(str(self.audio_path).encode()).hexdigest()[:12]


@dataclass
class AudioBatchV15:
    """A batch of v1.5 training examples."""

    encoded_latents: mx.array
    """Clean VAE latents [B, T, 64]."""

    text_embeddings: mx.array
    """Text encoder embeddings [B, text_T, 1024]."""

    lyric_embeddings: mx.array
    """Lyric encoder embeddings [B, lyric_T, 1024]."""

    attention_masks: mx.array
    """Latent attention masks [B, T]."""

    examples: List[AudioExampleV15]
    rng: random.Random


class AudioDatasetV15:
    """Dataset for ACE-Step v1.5 training.

    Supports:
    1. Sidecar files: audio.wav + audio.txt (caption) + audio.lyrics (lyrics)
    2. JSON manifest with {"file", "caption", "lyrics", "duration"}
    3. Directory structure for auto-generated captions

    Example:
        dataset = AudioDatasetV15(data_dir="./training_audio")
        dataset.encode_all(vae=vae, text_encoder=text_encoder)
        for batch in dataset.iter_batches(batch_size=1):
            loss = train_step(batch)
    """

    def __init__(
        self,
        data_dir: Path | str,
        max_duration: float = 30.0,
        caption_extension: str = ".txt",
        lyrics_extension: str = ".lyrics",
        manifest_file: Optional[str] = None,
        seed: int = 42,
    ):
        self.data_dir = Path(data_dir).resolve()
        if not self.data_dir.exists():
            raise FileNotFoundError(f"Data directory not found: {self.data_dir}")
        if not self.data_dir.is_dir():
            raise ValueError(f"Data path is not a directory: {self.data_dir}")

        self.max_duration = max_duration
        self.caption_extension = caption_extension
        self.lyrics_extension = lyrics_extension
        self.seed = seed
        self.rng = random.Random(seed)
        self.examples: List[AudioExampleV15] = []
        self._is_encoded = False

        if manifest_file and (self.data_dir / manifest_file).exists():
            self._load_from_manifest(self.data_dir / manifest_file)
        else:
            self._discover_audio_files()

        logger.info(f"Loaded {len(self.examples)} audio examples from {self.data_dir}")

    def _discover_audio_files(self) -> None:
        """Discover audio files with captions and optional lyrics."""
        for ext in AUDIO_EXTENSIONS:
            for audio_path in self.data_dir.rglob(f"*{ext}"):
                caption = self._find_caption(audio_path)
                if caption:
                    lyrics = self._find_lyrics(audio_path)
                    self.examples.append(
                        AudioExampleV15(
                            audio_path=audio_path,
                            caption=caption,
                            lyrics=lyrics,
                        )
                    )
                else:
                    logger.warning(f"No caption found for {audio_path}, skipping")

    def _find_caption(self, audio_path: Path) -> Optional[str]:
        """Find caption for an audio file."""
        caption_path = audio_path.with_suffix(self.caption_extension)
        if caption_path.exists():
            return caption_path.read_text().strip()

        caption_path = audio_path.with_suffix(".caption")
        if caption_path.exists():
            return caption_path.read_text().strip()

        rel_path = audio_path.relative_to(self.data_dir)
        if len(rel_path.parts) > 1:
            tags = [p.replace("_", " ").replace("-", " ") for p in rel_path.parts[:-1]]
            return f"{' '.join(tags)} music"

        return None

    def _find_lyrics(self, audio_path: Path) -> str:
        """Find lyrics for an audio file."""
        lyrics_path = audio_path.with_suffix(self.lyrics_extension)
        if lyrics_path.exists():
            return lyrics_path.read_text().strip()

        lyrics_path = audio_path.with_suffix(".lrc")
        if lyrics_path.exists():
            return lyrics_path.read_text().strip()

        return ""

    def _load_from_manifest(self, manifest_path: Path) -> None:
        """Load examples from JSON manifest with path traversal protection."""
        with open(manifest_path) as f:
            data = json.load(f)

        for entry in data:
            if "file" not in entry or "caption" not in entry:
                logger.warning(f"Manifest entry missing required fields: {entry}")
                continue

            file_path = entry["file"]
            if ".." in str(file_path) or str(file_path).startswith("/"):
                logger.warning(f"Skipping invalid file path in manifest: {file_path}")
                continue

            audio_path = (self.data_dir / file_path).resolve()
            try:
                audio_path.relative_to(self.data_dir)
            except ValueError:
                logger.warning(f"Skipping file outside data directory: {file_path}")
                continue

            if audio_path.exists():
                self.examples.append(
                    AudioExampleV15(
                        audio_path=audio_path,
                        caption=str(entry["caption"])[:10000],
                        lyrics=str(entry.get("lyrics", ""))[:50000],
                        duration_seconds=float(entry.get("duration", 0.0)),
                    )
                )
            else:
                logger.warning(f"Audio file not found: {audio_path}")

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> AudioExampleV15:
        return self.examples[idx]

    def shuffle(self) -> None:
        self.rng.shuffle(self.examples)

    def encode_all(
        self,
        vae: Any,
        text_encoder: Any = None,
        cache_dir: Optional[Path] = None,
        show_progress: bool = True,
    ) -> None:
        """Encode all examples with Oobleck VAE and text encoder.

        Args:
            vae: AutoencoderOobleck for audio -> 25Hz latent encoding
            text_encoder: Optional text encoder for caption/lyric embeddings
            cache_dir: Optional directory for caching
            show_progress: Show progress bar
        """
        if cache_dir is not None:
            cache_key = self._compute_cache_key()
            if self._load_cache(cache_dir, cache_key):
                logger.info("Loaded encoded dataset from cache")
                self._is_encoded = True
                return

        try:
            from tqdm import tqdm
            iterator = tqdm(self.examples, desc="Encoding v1.5 dataset") if show_progress else self.examples
        except ImportError:
            iterator = self.examples

        for example in iterator:
            example.encoded_latent = self._encode_audio(vae, example.audio_path)

            if text_encoder is not None:
                example.text_embedding = self._encode_text(text_encoder, example.caption)
                if example.lyrics:
                    example.lyric_embedding = self._encode_text(text_encoder, example.lyrics)
                else:
                    example.lyric_embedding = mx.zeros((1, 1024), dtype=mx.bfloat16)
            else:
                # Placeholder embeddings
                example.text_embedding = mx.zeros((1, 1024), dtype=mx.bfloat16)
                example.lyric_embedding = mx.zeros((1, 1024), dtype=mx.bfloat16)

        self._is_encoded = True

        if cache_dir is not None:
            self._save_cache(cache_dir, cache_key)

    def _encode_audio(self, vae: Any, audio_path: Path) -> mx.array:
        """Encode audio to Oobleck VAE latents.

        Loads 48kHz stereo audio, encodes to 25Hz, 64-dim latents.
        """
        try:
            import numpy as np
            import soundfile as sf

            audio, sr = sf.read(str(audio_path), dtype="float32")

            # Convert to stereo if mono
            if audio.ndim == 1:
                audio = np.stack([audio, audio], axis=-1)

            # Resample to 48kHz if needed
            if sr != SAMPLE_RATE:
                try:
                    import librosa
                    left = librosa.resample(audio[:, 0], orig_sr=sr, target_sr=SAMPLE_RATE)
                    right = librosa.resample(audio[:, 1], orig_sr=sr, target_sr=SAMPLE_RATE)
                    audio = np.stack([left, right], axis=-1)
                except ImportError:
                    raise RuntimeError(
                        f"librosa required for resampling from {sr}Hz to {SAMPLE_RATE}Hz. "
                        "Install with: pip install librosa"
                    )

            # Trim to max duration
            max_samples = int(self.max_duration * SAMPLE_RATE)
            if audio.shape[0] > max_samples:
                audio = audio[:max_samples]

            # Convert to MLX: [samples, 2] -> [1, samples, 2]
            audio_mx = mx.array(audio, dtype=mx.float32)[None]

            # Encode with VAE
            latents = vae.encode(audio_mx)
            _force_evaluate(latents)

            return latents[0]  # Remove batch dim -> [T, 64]

        except Exception as e:
            logger.error(f"Failed to encode {audio_path}: {e}")
            fallback_length = int(self.max_duration * LATENT_HZ)
            return mx.zeros((fallback_length, LATENT_DIM), dtype=mx.bfloat16)

    def _encode_text(self, text_encoder: Any, text: str) -> mx.array:
        """Encode text with the text encoder."""
        try:
            embedding = text_encoder.encode(text)
            _force_evaluate(embedding)
            return embedding[0] if embedding.ndim == 3 else embedding
        except Exception as e:
            logger.error(f"Failed to encode text: {e}")
            return mx.zeros((1, 1024), dtype=mx.bfloat16)

    def _compute_cache_key(self) -> str:
        key_parts = [
            f"v15:sr:{SAMPLE_RATE}",
            f"maxdur:{self.max_duration}",
            f"n:{len(self.examples)}",
        ]
        for ex in sorted(self.examples, key=lambda x: str(x.audio_path)):
            key_parts.append(f"{ex.audio_path.name}:{ex.caption[:50]}")
        return hashlib.sha256("|".join(key_parts).encode()).hexdigest()[:16]

    def _save_cache(self, cache_dir: Path, cache_key: str) -> None:
        cache_dir = Path(cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_path = cache_dir / f"v15_cache_{cache_key}.safetensors"
        meta_path = cache_dir / f"v15_cache_{cache_key}.json"

        arrays = {}
        metadata = {"examples": []}

        for i, ex in enumerate(self.examples):
            if ex.encoded_latent is not None:
                arrays[f"latent_{i}"] = ex.encoded_latent
            if ex.text_embedding is not None:
                arrays[f"text_{i}"] = ex.text_embedding
            if ex.lyric_embedding is not None:
                arrays[f"lyric_{i}"] = ex.lyric_embedding

            metadata["examples"].append({
                "id": ex.example_id,
                "audio_path": str(ex.audio_path),
                "caption": ex.caption,
                "lyrics": ex.lyrics[:200],
            })

        mx.save_safetensors(str(cache_path), arrays)
        with open(meta_path, "w") as f:
            json.dump(metadata, f, indent=2)

        logger.info(f"Saved v1.5 dataset cache: {cache_path}")

    def _load_cache(self, cache_dir: Path, cache_key: str) -> bool:
        if not cache_key.isalnum():
            return False

        cache_dir = Path(cache_dir).resolve()
        cache_path = cache_dir / f"v15_cache_{cache_key}.safetensors"
        meta_path = cache_dir / f"v15_cache_{cache_key}.json"

        if not cache_path.exists() or not meta_path.exists():
            return False

        try:
            cache_path.resolve().relative_to(cache_dir)
            meta_path.resolve().relative_to(cache_dir)
        except ValueError:
            logger.warning("Cache path traversal attempt detected")
            return False

        try:
            arrays = mx.load(str(cache_path))

            for i, ex in enumerate(self.examples):
                if f"latent_{i}" in arrays:
                    ex.encoded_latent = arrays[f"latent_{i}"]
                if f"text_{i}" in arrays:
                    ex.text_embedding = arrays[f"text_{i}"]
                if f"lyric_{i}" in arrays:
                    ex.lyric_embedding = arrays[f"lyric_{i}"]

            return True
        except Exception as e:
            logger.warning(f"Failed to load cache: {e}")
            return False

    def iter_batches(
        self,
        batch_size: int,
        shuffle: bool = True,
        drop_last: bool = True,
    ) -> Iterator[AudioBatchV15]:
        """Iterate over batches with padded and stacked tensors."""
        if not self._is_encoded:
            raise RuntimeError("Dataset must be encoded first. Call encode_all().")

        if len(self.examples) == 0:
            return

        indices = list(range(len(self.examples)))
        if shuffle:
            self.rng.shuffle(indices)

        for start in range(0, len(indices), batch_size):
            end = start + batch_size
            if end > len(indices) and drop_last:
                break

            batch_indices = indices[start:end]
            batch_examples = [self.examples[i] for i in batch_indices]

            for ex in batch_examples:
                if ex.encoded_latent is None:
                    raise RuntimeError(
                        f"Example {ex.example_id} has no encoded_latent."
                    )

            # Pad latents to same length
            max_len = max(ex.encoded_latent.shape[0] for ex in batch_examples)
            padded_latents = []
            attention_masks = []
            for ex in batch_examples:
                latent = ex.encoded_latent
                seq_len = latent.shape[0]
                if seq_len < max_len:
                    pad = mx.zeros((max_len - seq_len, LATENT_DIM), dtype=latent.dtype)
                    latent = mx.concatenate([latent, pad], axis=0)
                padded_latents.append(latent)

                mask = mx.concatenate([
                    mx.ones((seq_len,)),
                    mx.zeros((max_len - seq_len,)),
                ])
                attention_masks.append(mask)

            # Pad text embeddings to same length
            max_text_len = max(
                ex.text_embedding.shape[0] if ex.text_embedding is not None else 1
                for ex in batch_examples
            )
            padded_text = []
            for ex in batch_examples:
                emb = ex.text_embedding if ex.text_embedding is not None else mx.zeros((1, 1024))
                if emb.shape[0] < max_text_len:
                    pad = mx.zeros((max_text_len - emb.shape[0], emb.shape[-1]), dtype=emb.dtype)
                    emb = mx.concatenate([emb, pad], axis=0)
                padded_text.append(emb)

            # Pad lyric embeddings
            max_lyric_len = max(
                ex.lyric_embedding.shape[0] if ex.lyric_embedding is not None else 1
                for ex in batch_examples
            )
            padded_lyrics = []
            for ex in batch_examples:
                emb = ex.lyric_embedding if ex.lyric_embedding is not None else mx.zeros((1, 1024))
                if emb.shape[0] < max_lyric_len:
                    pad = mx.zeros((max_lyric_len - emb.shape[0], emb.shape[-1]), dtype=emb.dtype)
                    emb = mx.concatenate([emb, pad], axis=0)
                padded_lyrics.append(emb)

            yield AudioBatchV15(
                encoded_latents=mx.stack(padded_latents),
                text_embeddings=mx.stack(padded_text),
                lyric_embeddings=mx.stack(padded_lyrics),
                attention_masks=mx.stack(attention_masks),
                examples=batch_examples,
                rng=self.rng,
            )


class PreEncodedDatasetV15(AudioDatasetV15):
    """Dataset loading pre-encoded v1.5 latents and embeddings.

    Expected directory structure:
        data_dir/
            latents/
                example_001.safetensors  # "latent", "text_embedding", "lyric_embedding"
                ...

    Security: Only .safetensors files are loaded (no pickle).
    """

    def __init__(self, data_dir: Path | str, seed: int = 42):
        self.data_dir = Path(data_dir).resolve()
        if not self.data_dir.exists():
            raise FileNotFoundError(f"Data directory not found: {self.data_dir}")
        if not self.data_dir.is_dir():
            raise ValueError(f"Data path is not a directory: {self.data_dir}")

        self.max_duration = 240.0
        self.seed = seed
        self.rng = random.Random(seed)
        self.examples: List[AudioExampleV15] = []
        self._is_encoded = True

        self._load_pre_encoded()
        logger.info(f"Loaded {len(self.examples)} pre-encoded v1.5 examples")

    def _load_pre_encoded(self) -> None:
        latents_dir = self.data_dir / "latents"
        if not latents_dir.exists():
            latents_dir = self.data_dir

        for path in sorted(latents_dir.glob("*.safetensors")):
            resolved = path.resolve()
            try:
                resolved.relative_to(self.data_dir)
            except ValueError:
                logger.warning(f"Skipping file outside data directory: {path}")
                continue

            try:
                data = mx.load(str(resolved))
                example = AudioExampleV15(
                    audio_path=resolved,
                    caption="",
                    encoded_latent=data.get("latent", data.get("encoded_latent")),
                    text_embedding=data.get("text_embedding", data.get("embedding")),
                    lyric_embedding=data.get("lyric_embedding"),
                    example_id=path.stem,
                )
                if example.lyric_embedding is None:
                    example.lyric_embedding = mx.zeros((1, 1024), dtype=mx.bfloat16)
                self.examples.append(example)
            except Exception as e:
                logger.warning(f"Failed to load {path}: {e}")

    def encode_all(self, *args, **kwargs) -> None:
        pass
