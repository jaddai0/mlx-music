"""
Tests for ACE-Step loss functions.

Tests:
- ACEStepLoss: sample_timestep, sample_noise, add_noise, compute_target_velocity
- ACEStepLoss.forward: Complete training forward pass
- create_train_step: Training step function factory
"""

import random
import pytest
import mlx.core as mx


class TestACEStepLoss:
    """Tests for ACEStepLoss class."""

    def test_sample_timestep_shape(self):
        """sample_timestep should return correct shape."""
        from mlx_music.training.ace_step.loss import ACEStepLoss

        batch_size = 4
        num_timesteps = 100

        timesteps = ACEStepLoss.sample_timestep(batch_size, num_timesteps)

        assert timesteps.shape == (batch_size,)
        assert timesteps.dtype == mx.int32

    def test_sample_timestep_range(self):
        """sample_timestep should return values in [0, num_timesteps)."""
        from mlx_music.training.ace_step.loss import ACEStepLoss

        batch_size = 100
        num_timesteps = 50

        timesteps = ACEStepLoss.sample_timestep(batch_size, num_timesteps)

        # All values should be in range
        assert mx.all(timesteps >= 0)
        assert mx.all(timesteps < num_timesteps)

    def test_sample_timestep_reproducible_with_rng(self):
        """sample_timestep should be reproducible with same RNG."""
        from mlx_music.training.ace_step.loss import ACEStepLoss

        rng1 = random.Random(42)
        rng2 = random.Random(42)

        t1 = ACEStepLoss.sample_timestep(10, 100, rng=rng1)
        t2 = ACEStepLoss.sample_timestep(10, 100, rng=rng2)

        assert mx.array_equal(t1, t2)

    def test_sample_noise_shape(self):
        """sample_noise should return correct shape."""
        from mlx_music.training.ace_step.loss import ACEStepLoss

        shape = (2, 8, 100)
        noise = ACEStepLoss.sample_noise(shape)

        assert noise.shape == shape

    def test_sample_noise_dtype(self):
        """sample_noise should respect dtype parameter."""
        from mlx_music.training.ace_step.loss import ACEStepLoss

        noise_bf16 = ACEStepLoss.sample_noise((2, 8, 100), dtype=mx.bfloat16)
        noise_f32 = ACEStepLoss.sample_noise((2, 8, 100), dtype=mx.float32)

        assert noise_bf16.dtype == mx.bfloat16
        assert noise_f32.dtype == mx.float32

    def test_sample_noise_is_gaussian(self):
        """sample_noise should produce approximately Gaussian distribution."""
        from mlx_music.training.ace_step.loss import ACEStepLoss

        # Large sample for statistical test
        noise = ACEStepLoss.sample_noise((1000, 100), dtype=mx.float32)

        # Mean should be ~0, std should be ~1
        mean = float(noise.mean())
        std = float(noise.std())

        assert abs(mean) < 0.1
        assert abs(std - 1.0) < 0.1

    def test_add_noise_formula(self):
        """add_noise should implement: latents_t = (1 - sigma) * clean + sigma * noise."""
        from mlx_music.training.ace_step.loss import ACEStepLoss

        clean = mx.ones((2, 4, 10)) * 2.0
        noise = mx.ones((2, 4, 10)) * 8.0
        sigma = 0.25

        # Expected: (1 - 0.25) * 2 + 0.25 * 8 = 1.5 + 2 = 3.5
        result = ACEStepLoss.add_noise(clean, noise, sigma)

        expected = mx.full((2, 4, 10), 3.5)
        assert mx.allclose(result, expected)

    def test_add_noise_sigma_zero(self):
        """add_noise with sigma=0 should return clean."""
        from mlx_music.training.ace_step.loss import ACEStepLoss

        clean = mx.random.normal((2, 4, 10))
        noise = mx.random.normal((2, 4, 10))

        result = ACEStepLoss.add_noise(clean, noise, sigma=0.0)

        assert mx.allclose(result, clean)

    def test_add_noise_sigma_one(self):
        """add_noise with sigma=1 should return noise."""
        from mlx_music.training.ace_step.loss import ACEStepLoss

        clean = mx.random.normal((2, 4, 10))
        noise = mx.random.normal((2, 4, 10))

        result = ACEStepLoss.add_noise(clean, noise, sigma=1.0)

        assert mx.allclose(result, noise)

    def test_add_noise_array_sigma(self):
        """add_noise should work with per-sample sigma array."""
        from mlx_music.training.ace_step.loss import ACEStepLoss

        batch_size = 4
        clean = mx.ones((batch_size, 4, 10))
        noise = mx.zeros((batch_size, 4, 10))
        sigma = mx.array([0.0, 0.5, 0.75, 1.0])

        result = ACEStepLoss.add_noise(clean, noise, sigma)

        # Sample 0: sigma=0 -> result=clean=1
        assert mx.allclose(result[0], mx.ones((4, 10)))
        # Sample 3: sigma=1 -> result=noise=0
        assert mx.allclose(result[3], mx.zeros((4, 10)))

    def test_compute_target_velocity_formula(self):
        """compute_target_velocity should return noise - clean."""
        from mlx_music.training.ace_step.loss import ACEStepLoss

        clean = mx.array([[1.0, 2.0, 3.0]])
        noise = mx.array([[4.0, 5.0, 6.0]])

        target = ACEStepLoss.compute_target_velocity(clean, noise)

        expected = mx.array([[3.0, 3.0, 3.0]])  # noise - clean
        assert mx.allclose(target, expected)

    def test_compute_loss_mse(self):
        """compute_loss should compute MSE."""
        from mlx_music.training.ace_step.loss import ACEStepLoss

        predicted = mx.zeros((2, 4, 10))
        target = mx.ones((2, 4, 10))

        loss = ACEStepLoss.compute_loss(predicted, target, reduction="mean")

        # MSE of 0 vs 1 = 1
        assert float(loss) == pytest.approx(1.0)

    def test_compute_loss_with_mask(self):
        """compute_loss should apply mask correctly."""
        from mlx_music.training.ace_step.loss import ACEStepLoss

        predicted = mx.zeros((2, 4, 10))
        target = mx.ones((2, 4, 10))

        # Mask out half the values - first sample is valid (1s), second is masked (0s)
        mask = mx.concatenate([
            mx.ones((1, 4, 10)),
            mx.zeros((1, 4, 10))
        ], axis=0)

        loss = ACEStepLoss.compute_loss(predicted, target, mask=mask, reduction="mean")

        # Only first sample contributes, MSE = 1
        assert float(loss) == pytest.approx(1.0, rel=0.1)

    def test_compute_loss_reduction_none(self):
        """compute_loss with reduction='none' returns per-element loss."""
        from mlx_music.training.ace_step.loss import ACEStepLoss

        predicted = mx.zeros((2, 4, 10))
        target = mx.ones((2, 4, 10))

        loss = ACEStepLoss.compute_loss(predicted, target, reduction="none")

        assert loss.shape == (2, 4, 10)
        assert mx.allclose(loss, mx.ones((2, 4, 10)))

    def test_compute_loss_numerical_stability(self):
        """compute_loss should be stable with large values."""
        from mlx_music.training.ace_step.loss import ACEStepLoss

        predicted = mx.ones((2, 4, 10)) * 1000.0
        target = mx.ones((2, 4, 10)) * 1001.0

        loss = ACEStepLoss.compute_loss(predicted, target, reduction="mean")

        assert not mx.isnan(loss)
        assert not mx.isinf(loss)


class TestACEStepLossForward:
    """Tests for ACEStepLoss.forward method."""

    def test_forward_returns_loss_and_prediction(self):
        """forward() should return (loss, predicted_velocity) tuple."""
        # This test requires a mock model since we don't have ACE-Step loaded
        pytest.skip("Requires ACE-Step model - tested via integration tests")

    def test_forward_loss_is_scalar(self):
        """forward() should return scalar loss with default reduction."""
        pytest.skip("Requires ACE-Step model - tested via integration tests")


class TestCreateTrainStep:
    """Tests for create_train_step factory function."""

    def test_create_train_step_returns_callable(self):
        """create_train_step should return a callable."""
        pytest.skip("Requires ACE-Step model - tested via integration tests")
