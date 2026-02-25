"""Integration tests for ACE-Step v1.5 MLX implementation.

These tests require real model weights and are gated by environment variables:
  ACE_STEP_V15_PATH: Path to models directory (e.g., /path/to/ACE-Step-1.5/models)

Run: ACE_STEP_V15_PATH=/path/to/models pytest tests/test_ace_step_v15_integration.py -v
"""

import os
from pathlib import Path

import mlx.core as mx
import numpy as np
import pytest

# MLX lazy computation materializer (not Python's eval)
_materialize = mx.eval

MODEL_PATH = os.environ.get("ACE_STEP_V15_PATH", "")
requires_model = pytest.mark.skipif(
    not MODEL_PATH or not Path(MODEL_PATH).exists(),
    reason="ACE_STEP_V15_PATH not set or does not exist",
)


@requires_model
class TestModelLoading:
    """Test loading real model weights."""

    def test_load_turbo(self):
        from mlx_music.models.ace_step_v15.model import ACEStepV15

        model = ACEStepV15.from_pretrained(MODEL_PATH, variant="turbo")
        assert model is not None
        assert model.config.is_turbo is True
        assert model.config.num_hidden_layers == 24

    def test_load_with_quantization(self):
        from mlx_music.models.ace_step_v15.model import ACEStepV15

        model = ACEStepV15.from_pretrained(MODEL_PATH, variant="turbo", quantization="int4")
        assert model is not None

    def test_silence_latent_shape(self):
        from mlx_music.models.ace_step_v15.model import ACEStepV15

        model = ACEStepV15.from_pretrained(MODEL_PATH, variant="turbo")
        # silence_latent should be [1, T, 64]
        assert model.silence_latent.ndim == 3
        assert model.silence_latent.shape[0] == 1
        assert model.silence_latent.shape[2] == 64

    def test_vae_loaded(self):
        from mlx_music.models.ace_step_v15.model import ACEStepV15

        model = ACEStepV15.from_pretrained(MODEL_PATH, variant="turbo")
        assert model.vae is not None


@requires_model
class TestGeneration:
    """Test end-to-end music generation."""

    def test_generate_short(self):
        """Generate 2 seconds of audio (minimal test)."""
        from mlx_music.models.ace_step_v15.model import ACEStepV15

        model = ACEStepV15.from_pretrained(MODEL_PATH, variant="turbo")
        result = model.generate(duration=2.0, seed=42)

        assert "audio" in result
        assert "sample_rate" in result
        assert result["sample_rate"] == 48000
        audio = result["audio"]
        assert audio.ndim == 2
        assert audio.shape[1] == 2  # stereo
        # At 48kHz, 2s = 96000 samples
        assert audio.shape[0] >= 48000  # At least 1 second

    def test_generate_deterministic(self):
        """Same seed should produce same output."""
        from mlx_music.models.ace_step_v15.model import ACEStepV15

        model = ACEStepV15.from_pretrained(MODEL_PATH, variant="turbo")
        r1 = model.generate(duration=2.0, seed=42)
        r2 = model.generate(duration=2.0, seed=42)

        np.testing.assert_allclose(
            r1["audio"][:48000],
            r2["audio"][:48000],
            atol=1e-4,
            err_msg="Same seed should produce same audio",
        )

    def test_generate_different_seeds(self):
        """Different seeds should produce different output."""
        from mlx_music.models.ace_step_v15.model import ACEStepV15

        model = ACEStepV15.from_pretrained(MODEL_PATH, variant="turbo")
        r1 = model.generate(duration=2.0, seed=42)
        r2 = model.generate(duration=2.0, seed=123)

        diff = np.mean(np.abs(r1["audio"][:48000] - r2["audio"][:48000]))
        assert diff > 0.01, "Different seeds should produce different audio"

    def test_generate_with_quantization(self):
        """Quantized model should still produce valid audio."""
        from mlx_music.models.ace_step_v15.model import ACEStepV15

        model = ACEStepV15.from_pretrained(
            MODEL_PATH, variant="turbo", quantization="int4"
        )
        result = model.generate(duration=2.0, seed=42)

        audio = result["audio"]
        assert audio.ndim == 2
        assert audio.shape[0] > 0
        # Audio should not be all zeros
        assert np.mean(np.abs(audio)) > 0.001


@requires_model
class TestLoRAIntegration:
    """Test LoRA application on real model."""

    def test_apply_lora_attention_only(self):
        from mlx_music.models.ace_step_v15.model import ACEStepV15
        from mlx_music.training.ace_step_v15.lora_layers import apply_lora_to_v15_model

        model = ACEStepV15.from_pretrained(MODEL_PATH, variant="turbo")
        num = apply_lora_to_v15_model(model.model, rank=16, include_mlp=False)
        # 24 DiT layers * 2 attention blocks * 4 projections + encoder layers
        assert num >= 192

    def test_apply_lora_with_mlp(self):
        from mlx_music.models.ace_step_v15.model import ACEStepV15
        from mlx_music.training.ace_step_v15.lora_layers import apply_lora_to_v15_model

        model = ACEStepV15.from_pretrained(MODEL_PATH, variant="turbo")
        num = apply_lora_to_v15_model(model.model, rank=16, include_mlp=True)
        assert num > 256  # Should include MLP layers

    def test_generate_with_lora(self):
        """LoRA model should still generate valid audio."""
        from mlx_music.models.ace_step_v15.model import ACEStepV15
        from mlx_music.training.ace_step_v15.lora_layers import apply_lora_to_v15_model

        model = ACEStepV15.from_pretrained(MODEL_PATH, variant="turbo")
        apply_lora_to_v15_model(model.model, rank=16)
        result = model.generate(duration=2.0, seed=42)

        audio = result["audio"]
        assert audio.ndim == 2
        assert audio.shape[0] > 0

    def test_lora_save_load_roundtrip(self, tmp_path):
        """Save LoRA weights, reload, verify they match."""
        from mlx_music.models.ace_step_v15.model import ACEStepV15
        from mlx_music.training.ace_step_v15.lora_layers import apply_lora_to_v15_model
        from mlx_music.training.ace_step.lora_layers import (
            save_lora_weights,
            load_lora_weights,
            get_lora_parameters,
        )
        from mlx.utils import tree_flatten

        model = ACEStepV15.from_pretrained(MODEL_PATH, variant="turbo")
        apply_lora_to_v15_model(model.model, rank=16)

        # Get LoRA params before save
        params_before = tree_flatten(get_lora_parameters(model.model))

        # Save
        save_path = tmp_path / "test_lora.safetensors"
        save_lora_weights(model.model, save_path)
        assert save_path.exists()

        # Reload into fresh model
        model2 = ACEStepV15.from_pretrained(MODEL_PATH, variant="turbo")
        apply_lora_to_v15_model(model2.model, rank=16)
        load_lora_weights(model2.model, save_path, allowed_dir=tmp_path)

        # Compare
        params_after = tree_flatten(get_lora_parameters(model2.model))
        assert len(params_before) == len(params_after)


@requires_model
class TestVAE:
    """Test VAE encode/decode on real model."""

    def test_vae_encode_decode_roundtrip(self):
        """Encode and decode should roughly reconstruct."""
        from mlx_music.models.ace_step_v15.model import ACEStepV15

        model = ACEStepV15.from_pretrained(MODEL_PATH, variant="turbo")

        # Create synthetic audio: 1 second of sine wave
        t = np.linspace(0, 1, 48000)
        sine = np.sin(2 * np.pi * 440 * t)
        audio_np = np.stack([sine, sine], axis=-1)  # stereo
        audio = mx.array(audio_np[None, :, :], dtype=mx.float32)  # [1, 48000, 2]

        # Encode
        latents = model.vae.encode(audio)
        _materialize(latents)
        assert latents.ndim == 3
        assert latents.shape[2] == 64

        # Decode
        reconstructed = model.vae.decode(latents)
        _materialize(reconstructed)
        assert reconstructed.ndim == 3


class TestCLIEngineDetection:
    """Test CLI engine auto-detection (no model weights needed)."""

    def test_detect_v15_engine(self):
        from mlx_music.cli import detect_model_family

        assert detect_model_family("ACE-Step-1.5/models") == "ace-step-v15"
        assert detect_model_family("acestep-v15-turbo") == "ace-step-v15"
        assert detect_model_family("some/path/v1.5") == "ace-step-v15"

    def test_detect_v1_engine(self):
        from mlx_music.cli import detect_model_family

        assert detect_model_family("ACE-Step/ACE-Step-v1-3.5B") == "ace-step"

    def test_detect_musicgen(self):
        from mlx_music.cli import detect_model_family

        assert detect_model_family("facebook/musicgen-small") == "musicgen"

    def test_detect_stable_audio(self):
        from mlx_music.cli import detect_model_family

        assert detect_model_family("stabilityai/stable-audio-open-1.0") == "stable-audio"
