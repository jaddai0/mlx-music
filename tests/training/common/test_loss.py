"""
Tests for common loss functions.

Tests:
- flow_matching_loss: Interpolation, velocity target, MSE, shapes
- v_prediction_loss: V-prediction formulation
- snr_weighted_loss: SNR weighting for intermediate noise levels
- min_snr_weighted_loss: Min-SNR clipped weighting
"""

import math
import pytest
import mlx.core as mx


class TestFlowMatchingLoss:
    """Tests for flow_matching_loss function."""

    def test_flow_matching_zero_residual_zero_loss(self):
        """Perfect prediction (residual=0) should give zero loss."""
        from mlx_music.training.common.loss import flow_matching_loss

        clean = mx.random.normal((2, 8, 100))
        noise = mx.random.normal((2, 8, 100))

        # Target velocity = noise - clean
        target = noise - clean

        # Perfect prediction
        model_output = target

        loss = flow_matching_loss(model_output, clean, noise)

        assert float(loss) == pytest.approx(0.0, abs=1e-6)

    def test_flow_matching_loss_positive(self):
        """Non-zero residual should give positive loss."""
        from mlx_music.training.common.loss import flow_matching_loss

        mx.random.seed(42)
        clean = mx.random.normal((2, 8, 100))
        noise = mx.random.normal((2, 8, 100))

        # Bad prediction (zeros)
        model_output = mx.zeros_like(clean)

        loss = flow_matching_loss(model_output, clean, noise)

        assert float(loss) > 0.0

    def test_flow_matching_loss_mean_reduction(self):
        """Mean reduction should average over all elements."""
        from mlx_music.training.common.loss import flow_matching_loss

        clean = mx.ones((2, 4, 10))
        noise = mx.zeros((2, 4, 10))  # Target = noise - clean = -ones

        # Prediction off by 1
        model_output = mx.zeros((2, 4, 10))

        loss = flow_matching_loss(model_output, clean, noise, reduction="mean")

        # Residual = 0 - (-1) = 1, squared = 1
        # Mean of all 1s = 1
        assert float(loss) == pytest.approx(1.0, abs=1e-5)

    def test_flow_matching_loss_sum_reduction(self):
        """Sum reduction should sum all squared errors."""
        from mlx_music.training.common.loss import flow_matching_loss

        clean = mx.ones((2, 4, 10))  # 80 elements
        noise = mx.zeros((2, 4, 10))

        model_output = mx.zeros((2, 4, 10))

        loss = flow_matching_loss(model_output, clean, noise, reduction="sum")

        # Sum of 80 ones = 80
        assert float(loss) == pytest.approx(80.0, abs=1e-4)

    def test_flow_matching_loss_none_reduction(self):
        """None reduction should return per-element loss."""
        from mlx_music.training.common.loss import flow_matching_loss

        clean = mx.ones((2, 4, 10))
        noise = mx.zeros((2, 4, 10))

        model_output = mx.zeros((2, 4, 10))

        loss = flow_matching_loss(model_output, clean, noise, reduction="none")

        assert loss.shape == clean.shape
        assert mx.allclose(loss, mx.ones_like(loss))

    def test_flow_matching_loss_invalid_reduction_raises(self):
        """Invalid reduction should raise ValueError."""
        from mlx_music.training.common.loss import flow_matching_loss

        clean = mx.ones((2, 4, 10))
        noise = mx.zeros((2, 4, 10))
        model_output = mx.zeros((2, 4, 10))

        with pytest.raises(ValueError, match="Unknown reduction"):
            flow_matching_loss(model_output, clean, noise, reduction="invalid")

    def test_flow_matching_loss_target_formula(self):
        """Verify target velocity = noise - clean."""
        from mlx_music.training.common.loss import flow_matching_loss

        clean = mx.array([[1.0, 2.0, 3.0]])
        noise = mx.array([[4.0, 5.0, 6.0]])

        # Target = noise - clean = [3, 3, 3]
        # If model predicts [3, 3, 3], loss should be 0
        model_output = mx.array([[3.0, 3.0, 3.0]])

        loss = flow_matching_loss(model_output, clean, noise)

        assert float(loss) == pytest.approx(0.0, abs=1e-6)

    def test_flow_matching_loss_broadcasts_correctly(self):
        """Loss should work with different batch sizes."""
        from mlx_music.training.common.loss import flow_matching_loss

        for batch_size in [1, 2, 4, 8]:
            clean = mx.random.normal((batch_size, 8, 50))
            noise = mx.random.normal((batch_size, 8, 50))
            model_output = mx.random.normal((batch_size, 8, 50))

            loss = flow_matching_loss(model_output, clean, noise)

            assert loss.ndim == 0  # Scalar
            assert not mx.isnan(loss)
            assert not mx.isinf(loss)

    def test_flow_matching_loss_numerical_stability(self):
        """Loss should be numerically stable with large values."""
        from mlx_music.training.common.loss import flow_matching_loss

        # Large values
        clean = mx.ones((2, 8, 100)) * 1000.0
        noise = mx.ones((2, 8, 100)) * 1000.0
        model_output = mx.ones((2, 8, 100)) * 1000.0

        # Target = noise - clean = 0, prediction = 1000
        # Loss = (1000 - 0)^2 = 1e6

        loss = flow_matching_loss(model_output, clean, noise)

        assert not mx.isnan(loss)
        assert not mx.isinf(loss)

    def test_flow_matching_loss_2d_input(self):
        """Loss should work with 2D input (no batch dim)."""
        from mlx_music.training.common.loss import flow_matching_loss

        clean = mx.random.normal((8, 100))
        noise = mx.random.normal((8, 100))
        model_output = mx.random.normal((8, 100))

        loss = flow_matching_loss(model_output, clean, noise)

        assert loss.ndim == 0


class TestVPredictionLoss:
    """Tests for v_prediction_loss function."""

    def test_v_prediction_zero_residual_zero_loss(self):
        """Perfect prediction should give zero loss."""
        from mlx_music.training.common.loss import v_prediction_loss

        clean = mx.random.normal((2, 8, 100))
        noise = mx.random.normal((2, 8, 100))
        sigma = 0.5

        # V-prediction target: v = sigma * clean - (1 - sigma) * noise
        target = sigma * clean - (1 - sigma) * noise
        model_output = target

        loss = v_prediction_loss(model_output, clean, noise, sigma)

        assert float(loss) == pytest.approx(0.0, abs=1e-5)

    def test_v_prediction_scalar_sigma(self):
        """v_prediction_loss should work with scalar sigma."""
        from mlx_music.training.common.loss import v_prediction_loss

        clean = mx.random.normal((2, 8, 100))
        noise = mx.random.normal((2, 8, 100))
        model_output = mx.zeros_like(clean)

        loss = v_prediction_loss(model_output, clean, noise, sigma=0.5)

        assert not mx.isnan(loss)

    def test_v_prediction_array_sigma(self):
        """v_prediction_loss should work with array sigma."""
        from mlx_music.training.common.loss import v_prediction_loss

        batch_size = 4
        clean = mx.random.normal((batch_size, 8, 100))
        noise = mx.random.normal((batch_size, 8, 100))
        model_output = mx.zeros_like(clean)

        # Different sigma per sample
        sigma = mx.array([0.1, 0.3, 0.5, 0.9])

        loss = v_prediction_loss(model_output, clean, noise, sigma)

        assert not mx.isnan(loss)

    def test_v_prediction_target_formula(self):
        """Verify v-prediction target = sigma * clean - (1 - sigma) * noise."""
        from mlx_music.training.common.loss import v_prediction_loss

        clean = mx.array([[[1.0, 2.0]]])  # [1, 1, 2]
        noise = mx.array([[[3.0, 4.0]]])
        sigma = 0.5

        # target = 0.5 * [1, 2] - 0.5 * [3, 4] = [0.5, 1] - [1.5, 2] = [-1, -1]
        expected_target = sigma * clean - (1 - sigma) * noise
        model_output = expected_target

        loss = v_prediction_loss(model_output, clean, noise, sigma)

        assert float(loss) == pytest.approx(0.0, abs=1e-6)

    def test_v_prediction_sigma_extremes(self):
        """v_prediction_loss should handle sigma near 0 and 1."""
        from mlx_music.training.common.loss import v_prediction_loss

        clean = mx.random.normal((2, 8, 100))
        noise = mx.random.normal((2, 8, 100))
        model_output = mx.random.normal((2, 8, 100))

        # Near 0
        loss_low = v_prediction_loss(model_output, clean, noise, sigma=0.001)
        assert not mx.isnan(loss_low)

        # Near 1
        loss_high = v_prediction_loss(model_output, clean, noise, sigma=0.999)
        assert not mx.isnan(loss_high)


class TestSNRWeightedLoss:
    """Tests for snr_weighted_loss function."""

    def test_snr_weighting_high_sigma_low_weight(self):
        """SNR weighting: high SNR (low noise) gets lower weight."""
        from mlx_music.training.common.loss import snr_weighted_loss

        loss = mx.array([1.0])  # Unit loss

        # Formula: weight = (1 + SNR)^(-gamma) where SNR = 1/sigma^2
        # High sigma → low SNR → larger (1 + low)^(-gamma) → higher weight
        # Low sigma → high SNR → smaller (1 + high)^(-gamma) → lower weight
        weighted_high_sigma = snr_weighted_loss(loss, sigma=1.0, snr_gamma=5.0)
        weighted_low_sigma = snr_weighted_loss(loss, sigma=0.1, snr_gamma=5.0)

        # Lower sigma (higher SNR) results in lower weighted loss
        # because (1 + high_SNR)^(-gamma) is smaller
        assert float(weighted_low_sigma) < float(weighted_high_sigma)

    def test_snr_weighting_formula(self):
        """Verify SNR weight = (1 + SNR)^(-gamma)."""
        from mlx_music.training.common.loss import snr_weighted_loss

        loss = mx.array([1.0])
        sigma = 0.5
        gamma = 5.0

        # SNR = 1 / sigma^2 = 1 / 0.25 = 4
        # weight = (1 + 4)^(-5) = 5^(-5) = 0.00032
        snr = 1.0 / (sigma ** 2)
        expected_weight = (1.0 + snr) ** (-gamma)

        weighted = snr_weighted_loss(loss, sigma=sigma, snr_gamma=gamma)

        assert float(weighted) == pytest.approx(expected_weight, rel=0.01)

    def test_snr_weighting_array_sigma(self):
        """snr_weighted_loss should work with array sigma."""
        from mlx_music.training.common.loss import snr_weighted_loss

        loss = mx.array([1.0, 1.0, 1.0, 1.0])
        sigma = mx.array([0.1, 0.3, 0.5, 0.9])

        weighted = snr_weighted_loss(loss, sigma)

        assert weighted.shape == loss.shape
        assert not mx.any(mx.isnan(weighted))


class TestMinSNRWeightedLoss:
    """Tests for min_snr_weighted_loss function."""

    def test_min_snr_clips_weight(self):
        """min_snr_weighted_loss should clip SNR weight."""
        from mlx_music.training.common.loss import min_snr_weighted_loss

        loss = mx.array([1.0])

        # Very low sigma = very high SNR
        # Without clipping, this would have very low weight
        # With clipping at gamma=5, weight = min(SNR, 5) / SNR
        weighted = min_snr_weighted_loss(loss, sigma=0.01, min_snr_gamma=5.0)

        # Weight should be clipped, resulting in reasonable value
        assert 0 < float(weighted) <= 1.0

    def test_min_snr_formula(self):
        """Verify min-SNR weight = min(SNR, gamma) / SNR."""
        from mlx_music.training.common.loss import min_snr_weighted_loss

        loss = mx.array([1.0])
        sigma = 0.5
        gamma = 5.0

        # SNR = 1 / 0.25 = 4
        # min(4, 5) / 4 = 4/4 = 1.0
        snr = 1.0 / (sigma ** 2)
        expected_weight = min(snr, gamma) / snr

        weighted = min_snr_weighted_loss(loss, sigma=sigma, min_snr_gamma=gamma)

        assert float(weighted) == pytest.approx(expected_weight, rel=0.01)

    def test_min_snr_high_snr_clipped(self):
        """Very high SNR should be clipped to gamma."""
        from mlx_music.training.common.loss import min_snr_weighted_loss

        loss = mx.array([1.0])
        gamma = 5.0

        # Very low sigma = very high SNR (e.g., 10000)
        sigma = 0.01
        snr = 1.0 / (sigma ** 2)  # = 10000

        # weight = min(10000, 5) / 10000 = 5/10000 = 0.0005
        expected_weight = gamma / snr

        weighted = min_snr_weighted_loss(loss, sigma=sigma, min_snr_gamma=gamma)

        assert float(weighted) == pytest.approx(expected_weight, rel=0.01)

    def test_min_snr_low_snr_unchanged(self):
        """Low SNR (high sigma) should pass through unchanged."""
        from mlx_music.training.common.loss import min_snr_weighted_loss

        loss = mx.array([1.0])
        gamma = 5.0

        # High sigma = low SNR (e.g., 1.0)
        sigma = 1.0
        snr = 1.0 / (sigma ** 2)  # = 1.0

        # weight = min(1, 5) / 1 = 1/1 = 1.0
        expected_weight = min(snr, gamma) / snr

        weighted = min_snr_weighted_loss(loss, sigma=sigma, min_snr_gamma=gamma)

        assert float(weighted) == pytest.approx(expected_weight, rel=0.01)

    def test_min_snr_numerical_stability(self):
        """min_snr_weighted_loss should be stable near zero sigma."""
        from mlx_music.training.common.loss import min_snr_weighted_loss

        loss = mx.array([1.0])

        # Very small sigma (would cause division by near-zero without epsilon)
        weighted = min_snr_weighted_loss(loss, sigma=1e-6, min_snr_gamma=5.0)

        assert not mx.isnan(weighted)
        assert not mx.isinf(weighted)
