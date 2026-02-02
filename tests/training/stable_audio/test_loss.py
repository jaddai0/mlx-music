"""
Tests for Stable Audio loss functions.

Tests:
- StableAudioLoss: sample_sigma, add_noise_edm, compute_target_v, compute_loss
"""

import random
import pytest
import mlx.core as mx


class TestStableAudioLoss:
    """Tests for StableAudioLoss class."""

    def test_sample_sigma_shape(self):
        """sample_sigma should return correct shape."""
        from mlx_music.training.stable_audio.loss import StableAudioLoss

        sigmas = StableAudioLoss.sample_sigma(batch_size=4)

        assert sigmas.shape == (4,)

    def test_sample_sigma_range(self):
        """sample_sigma should return values within [sigma_min, sigma_max]."""
        from mlx_music.training.stable_audio.loss import StableAudioLoss

        sigma_min = 0.002
        sigma_max = 80.0

        # Sample many to test range
        sigmas = StableAudioLoss.sample_sigma(
            batch_size=1000,
            sigma_min=sigma_min,
            sigma_max=sigma_max,
        )

        assert mx.all(sigmas >= sigma_min)
        assert mx.all(sigmas <= sigma_max)

    def test_sample_sigma_reproducible_with_rng(self):
        """sample_sigma should be reproducible with same RNG."""
        from mlx_music.training.stable_audio.loss import StableAudioLoss

        rng1 = random.Random(42)
        rng2 = random.Random(42)

        s1 = StableAudioLoss.sample_sigma(batch_size=10, rng=rng1)
        s2 = StableAudioLoss.sample_sigma(batch_size=10, rng=rng2)

        assert mx.allclose(s1, s2)

    def test_add_noise_edm_formula(self):
        """add_noise_edm should implement: noised = clean + sigma * noise."""
        from mlx_music.training.stable_audio.loss import StableAudioLoss

        clean = mx.ones((2, 4, 10)) * 2.0
        noise = mx.ones((2, 4, 10)) * 3.0
        sigma = 0.5

        # Expected: 2 + 0.5 * 3 = 2 + 1.5 = 3.5
        result = StableAudioLoss.add_noise_edm(clean, noise, sigma)

        expected = mx.full((2, 4, 10), 3.5)
        assert mx.allclose(result, expected)

    def test_add_noise_edm_sigma_zero(self):
        """add_noise_edm with sigma=0 should return clean."""
        from mlx_music.training.stable_audio.loss import StableAudioLoss

        clean = mx.random.normal((2, 4, 10))
        noise = mx.random.normal((2, 4, 10))

        result = StableAudioLoss.add_noise_edm(clean, noise, sigma=0.0)

        assert mx.allclose(result, clean)

    def test_add_noise_edm_array_sigma(self):
        """add_noise_edm should work with per-sample sigma array."""
        from mlx_music.training.stable_audio.loss import StableAudioLoss

        batch_size = 4
        clean = mx.ones((batch_size, 4, 10))  # All 1s
        noise = mx.ones((batch_size, 4, 10))  # All 1s
        sigma = mx.array([0.0, 1.0, 2.0, 3.0])

        result = StableAudioLoss.add_noise_edm(clean, noise, sigma)

        # Sample 0: clean + 0*noise = 1
        assert mx.allclose(result[0], mx.ones((4, 10)))
        # Sample 1: clean + 1*noise = 2
        assert mx.allclose(result[1], mx.full((4, 10), 2.0))
        # Sample 3: clean + 3*noise = 4
        assert mx.allclose(result[3], mx.full((4, 10), 4.0))

    def test_compute_target_v_formula(self):
        """compute_target_v should implement: v = sigma * clean - (1 - sigma) * noise."""
        from mlx_music.training.stable_audio.loss import StableAudioLoss

        clean = mx.array([[[1.0, 2.0]]])
        noise = mx.array([[[3.0, 4.0]]])
        sigma = 0.5

        # v = 0.5 * [1, 2] - 0.5 * [3, 4] = [0.5, 1] - [1.5, 2] = [-1, -1]
        result = StableAudioLoss.compute_target_v(clean, noise, sigma)

        expected = mx.array([[[-1.0, -1.0]]])
        assert mx.allclose(result, expected)

    def test_compute_target_v_sigma_zero(self):
        """compute_target_v with sigma=0 should return -noise."""
        from mlx_music.training.stable_audio.loss import StableAudioLoss

        clean = mx.random.normal((2, 4, 10))
        noise = mx.random.normal((2, 4, 10))

        result = StableAudioLoss.compute_target_v(clean, noise, sigma=0.0)

        # v = 0*clean - 1*noise = -noise
        assert mx.allclose(result, -noise)

    def test_compute_target_v_sigma_one(self):
        """compute_target_v with sigma=1 should return clean."""
        from mlx_music.training.stable_audio.loss import StableAudioLoss

        clean = mx.random.normal((2, 4, 10))
        noise = mx.random.normal((2, 4, 10))

        result = StableAudioLoss.compute_target_v(clean, noise, sigma=1.0)

        # v = 1*clean - 0*noise = clean
        assert mx.allclose(result, clean)

    def test_compute_loss_mse(self):
        """compute_loss should compute MSE."""
        from mlx_music.training.stable_audio.loss import StableAudioLoss

        predicted = mx.zeros((2, 4, 10))
        target = mx.ones((2, 4, 10))

        loss = StableAudioLoss.compute_loss(predicted, target, reduction="mean")

        # MSE of 0 vs 1 = 1
        assert float(loss) == pytest.approx(1.0)

    def test_compute_loss_zero_residual(self):
        """compute_loss should be zero for perfect prediction."""
        from mlx_music.training.stable_audio.loss import StableAudioLoss

        predicted = mx.random.normal((2, 4, 10))
        target = predicted  # Same as prediction

        loss = StableAudioLoss.compute_loss(predicted, target)

        assert float(loss) == pytest.approx(0.0, abs=1e-6)

    def test_compute_loss_with_snr_weighting(self):
        """compute_loss with SNR weighting should weight by sigma."""
        from mlx_music.training.stable_audio.loss import StableAudioLoss

        predicted = mx.zeros((4, 4, 10))
        target = mx.ones((4, 4, 10))
        sigma = mx.array([0.1, 0.5, 1.0, 2.0])

        loss_no_snr = StableAudioLoss.compute_loss(
            predicted, target, use_snr_weighting=False
        )
        loss_snr = StableAudioLoss.compute_loss(
            predicted, target, sigma=sigma, use_snr_weighting=True
        )

        # Losses should be different with SNR weighting
        assert float(loss_no_snr) != float(loss_snr)

    def test_compute_loss_reduction_none(self):
        """compute_loss with reduction='none' returns per-element loss."""
        from mlx_music.training.stable_audio.loss import StableAudioLoss

        predicted = mx.zeros((2, 4, 10))
        target = mx.ones((2, 4, 10))

        loss = StableAudioLoss.compute_loss(predicted, target, reduction="none")

        assert loss.shape == (2, 4, 10)


class TestStableAudioLossForward:
    """Tests for StableAudioLoss.forward method."""

    def test_forward_requires_model(self):
        """forward() requires an actual model - skip without one."""
        pytest.skip("Requires Stable Audio model - tested via integration tests")
