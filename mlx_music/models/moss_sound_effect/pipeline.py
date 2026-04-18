"""End-to-end inference pipeline for MOSS-SoundEffect (MLX).

Usage:
    from mlx_music.models.moss_sound_effect.pipeline import SoundEffectPipeline

    pipe = SoundEffectPipeline.from_pretrained(
        model_path="/path/to/mlx-moss-sound-effect",
        tokenizer_path="/path/to/mlx-moss-tokenizer",
    )
    waveform = pipe.generate("A deep guttural growl from a large beast", duration=5.0)
    pipe.save_wav(waveform, "growl.wav")
"""

from __future__ import annotations

import json
import struct
from pathlib import Path
from typing import Any

import mlx.core as mx

from .config import MossSoundEffectConfig
from .generation import generate
from .model import MossSoundEffectModel
from .processor import apply_de_delay_pattern, build_prompt

# MLX lazy computation materializer (NOT Python's built-in)
_materialize = mx.eval


class SoundEffectPipeline:
    """End-to-end sound effect generation pipeline."""

    def __init__(
        self,
        model: MossSoundEffectModel,
        tokenizer: Any,
        audio_tokenizer: Any,
        config: MossSoundEffectConfig,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.audio_tokenizer = audio_tokenizer
        self.config = config

    @classmethod
    def from_pretrained(
        cls,
        model_path: str | Path,
        tokenizer_path: str | Path | None = None,
        dtype: mx.Dtype = mx.bfloat16,
    ) -> "SoundEffectPipeline":
        """Load pipeline from pretrained MLX weights.

        Args:
            model_path: Path to converted MOSS-SoundEffect MLX weights.
            tokenizer_path: Path to converted MOSS Audio Tokenizer MLX weights.
                If None, looks for tokenizer_weights.safetensors in model_path.
            dtype: Model dtype (default: bfloat16).
        """
        model_path = Path(model_path)
        if tokenizer_path is not None:
            tokenizer_path = Path(tokenizer_path)

        # Load config
        config_path = model_path / "config.json"
        with open(config_path) as f:
            config_dict = json.load(f)
        if "language_config" in config_dict:
            config = MossSoundEffectConfig.from_hf_config(config_dict)
        else:
            config = MossSoundEffectConfig.from_dict(config_dict)

        # Load text tokenizer
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(str(model_path), trust_remote_code=True)

        # Load generation model
        print("Loading MOSS-SoundEffect model...")
        model = MossSoundEffectModel.from_pretrained(model_path, config=config, dtype=dtype)
        print(f"  Loaded model ({config.num_hidden_layers} layers, dtype={dtype})")

        # Load audio tokenizer
        audio_tokenizer = None
        if tokenizer_path is not None:
            from ...codecs.moss_audio_tokenizer import MossAudioTokenizerModel, MossAudioTokenizerConfig
            print("Loading MOSS Audio Tokenizer...")
            audio_tokenizer = MossAudioTokenizerModel.from_pretrained(
                tokenizer_path, dtype=mx.float32,
            )
            print("  Loaded audio tokenizer")

        return cls(model, tokenizer, audio_tokenizer, config)

    def generate(
        self,
        prompt: str,
        duration: float = 5.0,
        quality: str = "high",
        max_new_tokens: int = 2000,
        text_temperature: float = 1.5,
        text_top_p: float = 0.6,
        text_top_k: int = 50,
        audio_temperature: float = 1.5,
        audio_top_p: float = 0.6,
        audio_top_k: int = 50,
        audio_repetition_penalty: float = 1.2,
    ) -> mx.array | None:
        """Generate a sound effect from a text prompt.

        Args:
            prompt: Text description of the desired sound effect.
            duration: Target duration in seconds.
            quality: Audio quality descriptor.
            max_new_tokens: Maximum generation tokens.
            text_temperature: Temperature for text token sampling.
            text_top_p: Top-p for text tokens.
            text_top_k: Top-k for text tokens.
            audio_temperature: Temperature for audio token sampling.
            audio_top_p: Top-p for audio tokens.
            audio_top_k: Top-k for audio tokens.
            audio_repetition_penalty: Repetition penalty for audio tokens.

        Returns:
            waveform: (T_samples,) audio at 24kHz, or None if decoding unavailable.
        """
        # 1. Build prompt -> input_ids
        input_ids = build_prompt(
            self.tokenizer, self.config, prompt, duration, quality,
        )
        print(f"Prompt: {prompt!r} ({input_ids.shape[1]} tokens)")

        # 2. Generate audio codes
        print("Generating...")
        outputs = generate(
            self.model,
            input_ids,
            self.config,
            max_new_tokens=max_new_tokens,
            text_temperature=text_temperature,
            text_top_p=text_top_p,
            text_top_k=text_top_k,
            audio_temperature=audio_temperature,
            audio_top_p=audio_top_p,
            audio_top_k=audio_top_k,
            audio_repetition_penalty=audio_repetition_penalty,
        )

        if not outputs:
            print("No output generated")
            return None

        start_length, gen_ids = outputs[0]

        # 3. Extract and de-delay audio codes
        # gen_ids: (T, 1 + n_vq)
        audio_codes_delayed = gen_ids[:, 1:]  # (T, n_vq)
        audio_codes = apply_de_delay_pattern(
            audio_codes_delayed.astype(mx.int32), self.config.n_vq, self.config.audio_pad_code,
        )
        print(f"Generated {audio_codes.shape[0]} audio frames ({audio_codes.shape[0] / 12.5:.1f}s)")

        # 4. Decode to waveform
        if self.audio_tokenizer is None:
            print("No audio tokenizer loaded - returning raw codes")
            return audio_codes

        # Reshape for tokenizer: (T, n_vq) -> (n_vq, 1, T)
        codes = audio_codes.transpose(1, 0)[:, None, :]

        waveform = self.audio_tokenizer.decode(codes.astype(mx.int32))
        # waveform: (1, 1, T_samples) -> (T_samples,)
        waveform = waveform.reshape(-1)
        _materialize(waveform)

        actual_duration = waveform.shape[0] / self.config.sampling_rate
        print(f"Decoded to {waveform.shape[0]} samples ({actual_duration:.1f}s at {self.config.sampling_rate}Hz)")

        return waveform

    @staticmethod
    def save_wav(
        waveform: mx.array,
        path: str | Path,
        sample_rate: int = 24000,
    ):
        """Save waveform as 16-bit PCM WAV file.

        Args:
            waveform: (T,) audio samples in [-1, 1].
            path: Output file path.
            sample_rate: Sample rate in Hz.
        """
        import numpy as np

        audio = np.array(waveform)
        # Normalize to [-1, 1] and convert to int16
        audio = np.clip(audio, -1.0, 1.0)
        audio_int16 = (audio * 32767).astype(np.int16)

        # Write WAV header + data
        num_samples = len(audio_int16)
        data_size = num_samples * 2  # 16-bit = 2 bytes per sample
        file_size = 36 + data_size

        with open(path, "wb") as f:
            # RIFF header
            f.write(b"RIFF")
            f.write(struct.pack("<I", file_size))
            f.write(b"WAVE")
            # fmt chunk
            f.write(b"fmt ")
            f.write(struct.pack("<I", 16))  # chunk size
            f.write(struct.pack("<H", 1))   # PCM format
            f.write(struct.pack("<H", 1))   # mono
            f.write(struct.pack("<I", sample_rate))
            f.write(struct.pack("<I", sample_rate * 2))  # byte rate
            f.write(struct.pack("<H", 2))   # block align
            f.write(struct.pack("<H", 16))  # bits per sample
            # data chunk
            f.write(b"data")
            f.write(struct.pack("<I", data_size))
            f.write(audio_int16.tobytes())

        print(f"Saved WAV to {path}")
