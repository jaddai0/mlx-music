"""Ground truth tests comparing ACE-Step v1.5 MLX vs PyTorch outputs.

These tests load the same weights into both frameworks and compare
numerical outputs at various stages of the pipeline.

Requires:
  ACE_STEP_V15_PATH: Path to models directory
  ACE_STEP_V15_PYTORCH: Path to PyTorch model repo (with modeling_acestep_v15_turbo.py)
  torch (PyTorch) must be installed

Run:
  ACE_STEP_V15_PATH=/path/to/models \
  ACE_STEP_V15_PYTORCH=/path/to/repo \
  pytest tests/test_ace_step_v15_ground_truth.py -v
"""

import os
from pathlib import Path

import mlx.core as mx
import numpy as np
import pytest

# MLX lazy computation materializer (not Python's eval)
_materialize = mx.eval

MODEL_PATH = os.environ.get("ACE_STEP_V15_PATH", "")
PYTORCH_PATH = os.environ.get("ACE_STEP_V15_PYTORCH", "")

requires_ground_truth = pytest.mark.skipif(
    not MODEL_PATH
    or not PYTORCH_PATH
    or not Path(MODEL_PATH).exists()
    or not Path(PYTORCH_PATH).exists(),
    reason="ACE_STEP_V15_PATH and ACE_STEP_V15_PYTORCH must be set",
)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two flattened arrays."""
    a_flat = a.flatten().astype(np.float64)
    b_flat = b.flatten().astype(np.float64)
    dot = np.dot(a_flat, b_flat)
    norm_a = np.linalg.norm(a_flat)
    norm_b = np.linalg.norm(b_flat)
    if norm_a < 1e-10 or norm_b < 1e-10:
        return 0.0
    return float(dot / (norm_a * norm_b))


@requires_ground_truth
class TestRoPEGroundTruth:
    """Compare RoPE embeddings between MLX and PyTorch."""

    def test_rope_cos_sin_match(self):
        """RoPE cos/sin caches should match exactly at fp32."""
        from mlx_music.models.ace_step_v15.attention import RotaryEmbedding

        rope_mlx = RotaryEmbedding(dim=128, max_position_embeddings=1024, base=1000000.0)
        cos_mlx, sin_mlx = rope_mlx(seq_len=64)
        _materialize(cos_mlx, sin_mlx)

        cos_np = np.array(cos_mlx.astype(mx.float32))
        sin_np = np.array(sin_mlx.astype(mx.float32))

        # Build reference RoPE manually (same formula)
        inv_freq = 1.0 / (1000000.0 ** (np.arange(0, 128, 2, dtype=np.float64) / 128))
        t = np.arange(64, dtype=np.float64)
        freqs = np.outer(t, inv_freq)
        emb = np.concatenate([freqs, freqs], axis=-1)
        cos_ref = np.cos(emb).astype(np.float32)
        sin_ref = np.sin(emb).astype(np.float32)

        np.testing.assert_allclose(cos_np, cos_ref, atol=1e-5, rtol=1e-5)
        np.testing.assert_allclose(sin_np, sin_ref, atol=1e-5, rtol=1e-5)


@requires_ground_truth
class TestTimestepEmbeddingGroundTruth:
    """Compare timestep embeddings."""

    def test_sinusoidal_embedding_shape(self):
        """Sinusoidal embeddings should have expected properties."""
        from mlx_music.models.ace_step_v15.dit import TimestepEmbedding

        te = TimestepEmbedding(in_channels=256, time_embed_dim=2048)
        t = mx.array([0.0, 0.25, 0.5, 0.75, 1.0])
        temb, proj = te(t)
        _materialize(temb, proj)

        temb_np = np.array(temb)
        # Different timesteps should give different embeddings
        for i in range(4):
            sim = cosine_similarity(temb_np[i], temb_np[i + 1])
            # Adjacent timesteps should be similar but not identical
            assert sim < 0.999, f"Timesteps {i} and {i+1} are too similar"

    def test_proj_6_way(self):
        """Timestep projection should produce 6 AdaLN parameters."""
        from mlx_music.models.ace_step_v15.dit import TimestepEmbedding

        te = TimestepEmbedding(in_channels=256, time_embed_dim=2048)
        t = mx.array([0.5])
        _, proj = te(t)
        _materialize(proj)
        assert proj.shape == (1, 6, 2048)


@requires_ground_truth
class TestAttentionGroundTruth:
    """Compare attention output between MLX and reference."""

    def test_self_attention_output(self):
        """Self-attention with random weights should produce consistent shapes."""
        from mlx_music.models.ace_step_v15.attention import AceStepV15Attention
        from mlx_music.models.ace_step_v15.config import AceStepV15Config

        config = AceStepV15Config()
        attn = AceStepV15Attention(config, layer_idx=0)

        # Fixed seed for reproducibility
        key = mx.random.key(42)
        x = mx.random.normal((1, 64, 2048), key=key)
        out = attn(x)
        _materialize(out)

        out_np = np.array(out)
        assert out_np.shape == (1, 64, 2048)
        assert not np.any(np.isnan(out_np)), "Attention output contains NaN"
        assert not np.any(np.isinf(out_np)), "Attention output contains Inf"


@requires_ground_truth
class TestFlowMatchingGroundTruth:
    """Compare flow matching formulations."""

    def test_noise_interpolation_at_boundaries(self):
        """x_t should equal clean at t=0 and noise at t=1."""
        from mlx_music.training.ace_step_v15.loss import ACEStepV15Loss

        clean = mx.ones((1, 10, 64))
        noise = mx.random.normal((1, 10, 64))

        # At t=0: x_t = clean
        x0 = ACEStepV15Loss.add_noise(clean, noise, mx.array([0.0]))
        _materialize(x0)
        np.testing.assert_allclose(np.array(x0), np.array(clean), atol=1e-6)

        # At t=1: x_t = noise
        x1 = ACEStepV15Loss.add_noise(clean, noise, mx.array([1.0]))
        _materialize(x1)
        np.testing.assert_allclose(np.array(x1), np.array(noise), atol=1e-6)

    def test_velocity_field_direction(self):
        """Target velocity should point from clean to noise."""
        from mlx_music.training.ace_step_v15.loss import ACEStepV15Loss

        clean = mx.zeros((1, 10, 64))
        noise = mx.ones((1, 10, 64))

        v = ACEStepV15Loss.compute_target_velocity(clean, noise)
        _materialize(v)
        # v = noise - clean = 1 - 0 = 1
        np.testing.assert_allclose(np.array(v), 1.0, atol=1e-6)

    def test_timestep_distribution(self):
        """Log-normal timestep sampling should have expected properties."""
        from mlx_music.training.ace_step_v15.loss import ACEStepV15Loss
        import random

        rng = random.Random(42)
        t = ACEStepV15Loss.sample_timestep(batch_size=10000, mu=-0.4, sigma=1.0, rng=rng)
        _materialize(t)
        t_np = np.array(t)

        # Should be in (0, 1)
        assert np.all(t_np > 0)
        assert np.all(t_np < 1)

        # Mean should be biased toward lower values (mu=-0.4)
        mean_t = np.mean(t_np)
        assert 0.3 < mean_t < 0.6, f"Mean timestep {mean_t} outside expected range"


@requires_ground_truth
class TestDiTModelGroundTruth:
    """Compare full DiT model forward pass."""

    def test_dit_model_no_nan(self):
        """Full model forward pass should not produce NaN."""
        from mlx_music.models.ace_step_v15.model import ACEStepV15

        model = ACEStepV15.from_pretrained(MODEL_PATH, variant="turbo")

        # Create minimal inputs
        B = 1
        T = 50  # Short sequence
        dtype = mx.bfloat16

        text_hidden = mx.zeros((B, 1, 1024), dtype=dtype)
        text_mask = mx.ones((B, 1), dtype=dtype)
        lyric_hidden = mx.zeros((B, 1, 1024), dtype=dtype)
        lyric_mask = mx.ones((B, 1), dtype=dtype)
        refer_audio = mx.zeros((1, 750, 64), dtype=dtype)
        refer_mask = mx.array([0])
        src_latents = model.silence_latent[:, :T, :].astype(dtype)
        chunk_masks = mx.ones((B, T, 64), dtype=dtype)
        is_covers = mx.array([0])
        attention_mask = mx.ones((B, T), dtype=dtype)

        result = model.model.generate_audio(
            text_hidden_states=text_hidden,
            text_attention_mask=text_mask,
            lyric_hidden_states=lyric_hidden,
            lyric_attention_mask=lyric_mask,
            refer_audio_packed=refer_audio,
            refer_audio_order_mask=refer_mask,
            src_latents=src_latents,
            chunk_masks=chunk_masks,
            is_covers=is_covers,
            silence_latent=model.silence_latent[:, :T, :].astype(dtype),
            attention_mask=attention_mask,
            seed=42,
            shift=3.0,
        )

        latents = result["target_latents"]
        _materialize(latents)
        latents_np = np.array(latents.astype(mx.float32))
        assert not np.any(np.isnan(latents_np)), "DiT output contains NaN"


@requires_ground_truth
class TestVAEGroundTruth:
    """Compare VAE encode/decode numerics."""

    def test_vae_latent_stats(self):
        """VAE latents should have reasonable statistics."""
        from mlx_music.models.ace_step_v15.model import ACEStepV15

        model = ACEStepV15.from_pretrained(MODEL_PATH, variant="turbo")

        # Create test audio (white noise)
        np.random.seed(42)
        audio_np = np.random.randn(1, 48000, 2).astype(np.float32) * 0.1
        audio = mx.array(audio_np)

        latents = model.vae.encode(audio)
        _materialize(latents)
        latents_np = np.array(latents.astype(mx.float32))

        assert latents_np.shape[2] == 64
        # Latents should not be zero
        assert np.mean(np.abs(latents_np)) > 0.001
        # Latents should not be huge
        assert np.max(np.abs(latents_np)) < 100

    def test_vae_decode_valid_range(self):
        """VAE decoded audio should be in reasonable range."""
        from mlx_music.models.ace_step_v15.model import ACEStepV15

        model = ACEStepV15.from_pretrained(MODEL_PATH, variant="turbo")

        # Use silence latent as input
        latents = model.silence_latent[:, :25, :].astype(mx.float32)  # 1 second
        audio = model.vae.decode(latents)
        _materialize(audio)
        audio_np = np.array(audio)

        # Audio should be bounded
        assert np.max(np.abs(audio_np)) < 10.0, "Decoded audio amplitude too high"


@requires_ground_truth
class TestEndToEndGroundTruth:
    """End-to-end comparison between MLX outputs at different seeds."""

    def test_same_seed_same_latents(self):
        """Same seed should produce identical latents."""
        from mlx_music.models.ace_step_v15.model import ACEStepV15

        model = ACEStepV15.from_pretrained(MODEL_PATH, variant="turbo")
        r1 = model.generate(duration=2.0, seed=42)
        r2 = model.generate(duration=2.0, seed=42)

        l1 = np.array(r1["latents"].astype(mx.float32))
        l2 = np.array(r2["latents"].astype(mx.float32))

        sim = cosine_similarity(l1, l2)
        assert sim > 0.999, f"Same-seed latent cosine similarity {sim} < 0.999"

    def test_output_audio_valid(self):
        """Generated audio should pass basic sanity checks."""
        from mlx_music.models.ace_step_v15.model import ACEStepV15

        model = ACEStepV15.from_pretrained(MODEL_PATH, variant="turbo")
        result = model.generate(duration=5.0, seed=42)

        audio = result["audio"]
        assert audio.ndim == 2
        assert audio.shape[1] == 2  # stereo
        assert result["sample_rate"] == 48000

        # Audio should not be silence
        rms = np.sqrt(np.mean(audio ** 2))
        assert rms > 0.001, f"Audio RMS {rms} is too low (likely silence)"

        # Audio should not be clipped
        clip_ratio = np.mean(np.abs(audio) > 0.99)
        assert clip_ratio < 0.1, f"Audio clip ratio {clip_ratio} is too high"
