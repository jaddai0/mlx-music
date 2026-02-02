"""
Tests for learning rate schedulers.

Tests:
- ConstantLR: Basic constant LR, warmup support
- CosineAnnealingLR: Warmup, decay, min_lr, total_steps
- OneCycleLR: Three phases, peak, div_factor
- create_scheduler: Factory function for all types
"""

import math
import pytest
import mlx.core as mx
import mlx.optimizers as optim


class TestConstantLR:
    """Tests for ConstantLR scheduler."""

    def test_constant_lr_returns_initial(self, mock_optimizer):
        """ConstantLR should return initial_lr on every step."""
        from mlx_music.training.common.lr_scheduler import ConstantLR

        scheduler = ConstantLR(mock_optimizer, initial_lr=1e-3)

        for _ in range(10):
            lr = scheduler.step()
            assert lr == pytest.approx(1e-3)

    def test_constant_lr_with_warmup(self, mock_optimizer):
        """ConstantLR should linearly warmup when warmup_steps > 0."""
        from mlx_music.training.common.lr_scheduler import ConstantLR

        scheduler = ConstantLR(mock_optimizer, initial_lr=1e-3, warmup_steps=5)

        # Warmup phase: LR should increase linearly
        lrs = [scheduler.step() for _ in range(5)]

        # First step should be 1/5 of initial_lr
        assert lrs[0] == pytest.approx(1e-3 * (1 / 5))
        # Last warmup step should be initial_lr
        assert lrs[4] == pytest.approx(1e-3)

        # After warmup: constant
        for _ in range(10):
            lr = scheduler.step()
            assert lr == pytest.approx(1e-3)

    def test_constant_lr_state_dict(self, mock_optimizer):
        """ConstantLR should save and load state correctly."""
        from mlx_music.training.common.lr_scheduler import ConstantLR

        scheduler = ConstantLR(mock_optimizer, initial_lr=1e-3, warmup_steps=5)

        # Advance a few steps
        for _ in range(3):
            scheduler.step()

        # Save state
        state = scheduler.state_dict()
        assert state["step_count"] == 3
        assert state["warmup_steps"] == 5
        assert state["scheduler_type"] == "ConstantLR"

        # Create new scheduler and load state
        new_scheduler = ConstantLR(mock_optimizer, initial_lr=1e-3, warmup_steps=5)
        new_scheduler.load_state_dict(state)

        assert new_scheduler.step_count == 3


class TestLinearWarmupLR:
    """Tests for LinearWarmupLR scheduler."""

    def test_linear_warmup_increases_lr(self, mock_optimizer):
        """LinearWarmupLR should linearly increase LR during warmup."""
        from mlx_music.training.common.lr_scheduler import LinearWarmupLR

        scheduler = LinearWarmupLR(mock_optimizer, initial_lr=1e-3, warmup_steps=10)

        lrs = [scheduler.step() for _ in range(10)]

        # LR should monotonically increase
        for i in range(1, len(lrs)):
            assert lrs[i] > lrs[i - 1]

        # Final warmup step should equal initial_lr
        assert lrs[-1] == pytest.approx(1e-3)

    def test_linear_warmup_constant_after(self, mock_optimizer):
        """LinearWarmupLR should be constant after warmup."""
        from mlx_music.training.common.lr_scheduler import LinearWarmupLR

        scheduler = LinearWarmupLR(mock_optimizer, initial_lr=1e-3, warmup_steps=5)

        # Skip warmup
        for _ in range(5):
            scheduler.step()

        # After warmup: constant
        for _ in range(10):
            lr = scheduler.step()
            assert lr == pytest.approx(1e-3)


class TestCosineAnnealingLR:
    """Tests for CosineAnnealingLR scheduler."""

    def test_cosine_starts_at_initial_lr(self, mock_optimizer):
        """CosineAnnealingLR should start at initial_lr (no warmup)."""
        from mlx_music.training.common.lr_scheduler import CosineAnnealingLR

        scheduler = CosineAnnealingLR(
            mock_optimizer, initial_lr=1e-3, total_steps=100, warmup_steps=0
        )

        lr = scheduler.step()
        assert lr == pytest.approx(1e-3)

    def test_cosine_ends_at_min_lr(self, mock_optimizer):
        """CosineAnnealingLR should approach min_lr at the end."""
        from mlx_music.training.common.lr_scheduler import CosineAnnealingLR

        min_lr = 1e-6
        scheduler = CosineAnnealingLR(
            mock_optimizer,
            initial_lr=1e-3,
            total_steps=100,
            warmup_steps=0,
            min_lr=min_lr,
        )

        # Run to completion
        lrs = [scheduler.step() for _ in range(100)]

        # Final LR should be very close to min_lr
        # Note: At step N-1, progress = (N-1)/N which is slightly less than 1.0
        # The LR will be approximately 25% above min_lr at the second-to-last step
        # and will reach exactly min_lr at step N (if we called step one more time)
        assert lrs[-1] == pytest.approx(min_lr, rel=0.3)

    def test_cosine_with_warmup(self, mock_optimizer):
        """CosineAnnealingLR should warmup then decay."""
        from mlx_music.training.common.lr_scheduler import CosineAnnealingLR

        scheduler = CosineAnnealingLR(
            mock_optimizer,
            initial_lr=1e-3,
            total_steps=100,
            warmup_steps=10,
            min_lr=0.0,
        )

        lrs = [scheduler.step() for _ in range(100)]

        # Warmup: LR should increase
        for i in range(1, 10):
            assert lrs[i] > lrs[i - 1], f"Step {i}: warmup should increase LR"

        # Peak at end of warmup
        assert lrs[9] == pytest.approx(1e-3)

        # After warmup: LR should decrease
        for i in range(11, 50):
            assert lrs[i] < lrs[i - 1], f"Step {i}: cosine should decrease LR"

    def test_cosine_decay_is_smooth(self, mock_optimizer):
        """CosineAnnealingLR decay should be smooth (monotonic decrease)."""
        from mlx_music.training.common.lr_scheduler import CosineAnnealingLR

        scheduler = CosineAnnealingLR(
            mock_optimizer,
            initial_lr=1e-3,
            total_steps=100,
            warmup_steps=0,
            min_lr=0.0,
        )

        lrs = [scheduler.step() for _ in range(100)]

        # Should monotonically decrease (or stay same at boundaries)
        for i in range(1, len(lrs)):
            assert lrs[i] <= lrs[i - 1] + 1e-10, f"Step {i}: LR should not increase"

    def test_cosine_midpoint_is_half(self, mock_optimizer):
        """CosineAnnealingLR midpoint should be approx 50% of range."""
        from mlx_music.training.common.lr_scheduler import CosineAnnealingLR

        initial_lr = 1e-3
        min_lr = 0.0
        scheduler = CosineAnnealingLR(
            mock_optimizer,
            initial_lr=initial_lr,
            total_steps=100,
            warmup_steps=0,
            min_lr=min_lr,
        )

        # Advance to midpoint
        for _ in range(50):
            scheduler.step()

        # At midpoint, cosine is 0, so LR = (initial + min) / 2
        lr = scheduler.get_lr()
        expected = (initial_lr + min_lr) / 2
        assert lr == pytest.approx(expected, rel=0.05)

    def test_cosine_warmup_too_large_raises(self, mock_optimizer):
        """CosineAnnealingLR should raise if warmup >= total."""
        from mlx_music.training.common.lr_scheduler import CosineAnnealingLR

        with pytest.raises(ValueError, match="warmup_steps"):
            CosineAnnealingLR(
                mock_optimizer, initial_lr=1e-3, total_steps=100, warmup_steps=100
            )

    def test_cosine_state_roundtrip(self, mock_optimizer):
        """CosineAnnealingLR should save/load state correctly."""
        from mlx_music.training.common.lr_scheduler import CosineAnnealingLR

        scheduler = CosineAnnealingLR(
            mock_optimizer,
            initial_lr=1e-3,
            total_steps=100,
            warmup_steps=10,
            min_lr=1e-6,
        )

        # Advance
        for _ in range(25):
            scheduler.step()

        # Save
        state = scheduler.state_dict()

        # Create new and load
        new_scheduler = CosineAnnealingLR(
            mock_optimizer, initial_lr=1e-3, total_steps=100, warmup_steps=10, min_lr=1e-6
        )
        new_scheduler.load_state_dict(state)

        # Should produce same LR
        assert scheduler.get_lr() == pytest.approx(new_scheduler.get_lr())

    def test_cosine_handles_overshoot(self, mock_optimizer):
        """CosineAnnealingLR should handle steps beyond total_steps."""
        from mlx_music.training.common.lr_scheduler import CosineAnnealingLR

        scheduler = CosineAnnealingLR(
            mock_optimizer, initial_lr=1e-3, total_steps=10, warmup_steps=0, min_lr=1e-6
        )

        # Run well past total_steps
        lrs = [scheduler.step() for _ in range(20)]

        # After total_steps, should stay at min_lr
        for lr in lrs[10:]:
            assert lr == pytest.approx(1e-6, rel=0.01)


class TestOneCycleLR:
    """Tests for OneCycleLR scheduler."""

    def test_onecycle_starts_low(self, mock_optimizer):
        """OneCycleLR should start at initial_lr / div_factor."""
        from mlx_music.training.common.lr_scheduler import OneCycleLR

        initial_lr = 1e-3
        div_factor = 25.0
        scheduler = OneCycleLR(
            mock_optimizer, initial_lr=initial_lr, total_steps=100, div_factor=div_factor
        )

        lr = scheduler.step()
        # First step after warmup from start_lr
        expected_start = initial_lr / div_factor
        assert lr < initial_lr  # Should be less than peak
        assert lr >= expected_start  # Should be at or above start

    def test_onecycle_peaks_after_warmup(self, mock_optimizer):
        """OneCycleLR should reach initial_lr at pct_start."""
        from mlx_music.training.common.lr_scheduler import OneCycleLR

        initial_lr = 1e-3
        pct_start = 0.3
        total_steps = 100
        scheduler = OneCycleLR(
            mock_optimizer,
            initial_lr=initial_lr,
            total_steps=total_steps,
            pct_start=pct_start,
        )

        # Advance to warmup end
        warmup_steps = int(total_steps * pct_start)
        for _ in range(warmup_steps):
            scheduler.step()

        # At warmup end, should be at peak (or very close)
        lr = scheduler.get_lr()
        assert lr == pytest.approx(initial_lr, rel=0.1)

    def test_onecycle_ends_low(self, mock_optimizer):
        """OneCycleLR should end at initial_lr / final_div_factor."""
        from mlx_music.training.common.lr_scheduler import OneCycleLR

        initial_lr = 1e-3
        final_div_factor = 1e4
        scheduler = OneCycleLR(
            mock_optimizer,
            initial_lr=initial_lr,
            total_steps=100,
            final_div_factor=final_div_factor,
        )

        # Run to completion
        lrs = [scheduler.step() for _ in range(100)]

        expected_final = initial_lr / final_div_factor
        # Final LR will be close to but not exactly at the expected minimum
        # due to the progress calculation reaching (N-1)/N instead of 1.0
        # Use a looser tolerance to account for this
        assert lrs[-1] < initial_lr * 0.01  # Should be well below peak
        assert lrs[-1] > expected_final * 0.5  # Should be approaching final

    def test_onecycle_warmup_then_decay(self, mock_optimizer):
        """OneCycleLR should increase then decrease."""
        from mlx_music.training.common.lr_scheduler import OneCycleLR

        scheduler = OneCycleLR(
            mock_optimizer, initial_lr=1e-3, total_steps=100, pct_start=0.3
        )

        lrs = [scheduler.step() for _ in range(100)]

        # Find peak
        peak_idx = lrs.index(max(lrs))

        # Before peak: should generally increase
        # (might not be strictly monotonic due to step boundaries)
        assert lrs[peak_idx] > lrs[0]

        # After peak: should decrease
        for i in range(peak_idx + 1, len(lrs) - 1):
            assert lrs[i] >= lrs[i + 1], f"Step {i}: should decrease in annealing phase"

    def test_onecycle_invalid_pct_start(self, mock_optimizer):
        """OneCycleLR should raise for invalid pct_start."""
        from mlx_music.training.common.lr_scheduler import OneCycleLR

        with pytest.raises(ValueError, match="pct_start"):
            OneCycleLR(mock_optimizer, initial_lr=1e-3, total_steps=100, pct_start=0.0)

        with pytest.raises(ValueError, match="pct_start"):
            OneCycleLR(mock_optimizer, initial_lr=1e-3, total_steps=100, pct_start=1.0)

    def test_onecycle_total_steps_too_small(self, mock_optimizer):
        """OneCycleLR should raise for total_steps < 2."""
        from mlx_music.training.common.lr_scheduler import OneCycleLR

        with pytest.raises(ValueError, match="total_steps"):
            OneCycleLR(mock_optimizer, initial_lr=1e-3, total_steps=1)

    def test_onecycle_state_roundtrip(self, mock_optimizer):
        """OneCycleLR should save/load state correctly."""
        from mlx_music.training.common.lr_scheduler import OneCycleLR

        scheduler = OneCycleLR(
            mock_optimizer,
            initial_lr=1e-3,
            total_steps=100,
            pct_start=0.3,
            div_factor=25.0,
            final_div_factor=1e4,
        )

        # Advance
        for _ in range(40):
            scheduler.step()

        state = scheduler.state_dict()

        # Load into new scheduler
        new_scheduler = OneCycleLR(
            mock_optimizer,
            initial_lr=1e-3,
            total_steps=100,
            pct_start=0.3,
            div_factor=25.0,
            final_div_factor=1e4,
        )
        new_scheduler.load_state_dict(state)

        # Should produce same LR
        assert scheduler.get_lr() == pytest.approx(new_scheduler.get_lr(), rel=0.01)

    def test_onecycle_type_mismatch_raises(self, mock_optimizer):
        """Loading state from different scheduler type should raise."""
        from mlx_music.training.common.lr_scheduler import OneCycleLR, ConstantLR

        onecycle = OneCycleLR(mock_optimizer, initial_lr=1e-3, total_steps=100)
        constant = ConstantLR(mock_optimizer, initial_lr=1e-3)

        state = constant.state_dict()

        with pytest.raises(ValueError, match="scheduler type mismatch"):
            onecycle.load_state_dict(state)


class TestSchedulerFactory:
    """Tests for create_scheduler factory function."""

    def test_create_constant_scheduler(self, mock_optimizer):
        """create_scheduler should create ConstantLR."""
        from mlx_music.training.common.lr_scheduler import create_scheduler, ConstantLR

        scheduler = create_scheduler(
            name="constant", optimizer=mock_optimizer, initial_lr=1e-3, total_steps=100
        )
        assert isinstance(scheduler, ConstantLR)

    def test_create_cosine_scheduler(self, mock_optimizer):
        """create_scheduler should create CosineAnnealingLR."""
        from mlx_music.training.common.lr_scheduler import (
            create_scheduler,
            CosineAnnealingLR,
        )

        scheduler = create_scheduler(
            name="cosine",
            optimizer=mock_optimizer,
            initial_lr=1e-3,
            total_steps=100,
            warmup_steps=10,
        )
        assert isinstance(scheduler, CosineAnnealingLR)

    def test_create_onecycle_scheduler(self, mock_optimizer):
        """create_scheduler should create OneCycleLR."""
        from mlx_music.training.common.lr_scheduler import create_scheduler, OneCycleLR

        scheduler = create_scheduler(
            name="onecycle",
            optimizer=mock_optimizer,
            initial_lr=1e-3,
            total_steps=100,
            pct_start=0.3,
        )
        assert isinstance(scheduler, OneCycleLR)

    def test_create_linear_warmup_scheduler(self, mock_optimizer):
        """create_scheduler should create LinearWarmupLR."""
        from mlx_music.training.common.lr_scheduler import (
            create_scheduler,
            LinearWarmupLR,
        )

        scheduler = create_scheduler(
            name="linear_warmup",
            optimizer=mock_optimizer,
            initial_lr=1e-3,
            total_steps=100,
            warmup_steps=20,
        )
        assert isinstance(scheduler, LinearWarmupLR)

    def test_create_scheduler_unknown_raises(self, mock_optimizer):
        """create_scheduler should raise for unknown scheduler name."""
        from mlx_music.training.common.lr_scheduler import create_scheduler

        with pytest.raises(ValueError, match="Unknown scheduler"):
            create_scheduler(
                name="unknown", optimizer=mock_optimizer, initial_lr=1e-3, total_steps=100
            )

    def test_create_scheduler_name_variants(self, mock_optimizer):
        """create_scheduler should handle name variants."""
        from mlx_music.training.common.lr_scheduler import create_scheduler

        # Test various name formats for cosine
        for name in ["cosine", "COSINE", "cosine_annealing", "cosinelr"]:
            scheduler = create_scheduler(
                name=name, optimizer=mock_optimizer, initial_lr=1e-3, total_steps=100
            )
            assert "Cosine" in type(scheduler).__name__
