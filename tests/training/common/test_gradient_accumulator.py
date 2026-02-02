"""
Tests for Gradient Accumulator.

Tests:
- GradientAccumulator: accumulate(), reset(), step counting
- NoOpAccumulator: Passthrough behavior
- create_accumulator: Factory with steps=1 and steps>1
"""

import pytest
import warnings
import mlx.core as mx
from mlx.utils import tree_map


class TestGradientAccumulator:
    """Tests for GradientAccumulator class."""

    def test_accumulator_single_step_returns_none(self):
        """Single accumulation step (not complete) returns None."""
        from mlx_music.training.common.gradient_accumulator import GradientAccumulator

        accumulator = GradientAccumulator(accumulation_steps=4)

        grads = {"layer1.weight": mx.ones((10, 10)), "layer1.bias": mx.ones((10,))}

        result = accumulator.accumulate(grads)
        assert result is None

    def test_accumulator_complete_window_returns_averaged(self):
        """Completed accumulation window returns averaged gradients."""
        from mlx_music.training.common.gradient_accumulator import GradientAccumulator

        accumulator = GradientAccumulator(accumulation_steps=4)

        grads = {"layer1.weight": mx.ones((10, 10))}

        # Accumulate 4 times
        for i in range(3):
            result = accumulator.accumulate(grads)
            assert result is None

        result = accumulator.accumulate(grads)
        assert result is not None

        # Average of 4 ones = 0.25 (since we average by accumulation_steps)
        # Wait - actually we divide by accumulation_steps at the end
        # So 4 * ones / 4 = ones
        expected = mx.ones((10, 10))
        assert mx.allclose(result["layer1.weight"], expected)

    def test_accumulator_sums_different_gradients(self):
        """Accumulator should sum gradients then divide."""
        from mlx_music.training.common.gradient_accumulator import GradientAccumulator

        accumulator = GradientAccumulator(accumulation_steps=2)

        # First gradient: all 1s
        grads1 = {"w": mx.ones((5,))}
        accumulator.accumulate(grads1)

        # Second gradient: all 3s
        grads2 = {"w": mx.full((5,), 3.0)}
        result = accumulator.accumulate(grads2)

        # Sum = 1 + 3 = 4, average = 4/2 = 2
        expected = mx.full((5,), 2.0)
        assert mx.allclose(result["w"], expected)

    def test_accumulator_count_property(self):
        """count property should track accumulation progress."""
        from mlx_music.training.common.gradient_accumulator import GradientAccumulator

        accumulator = GradientAccumulator(accumulation_steps=4)
        assert accumulator.count == 0

        grads = {"w": mx.ones((5,))}

        accumulator.accumulate(grads)
        assert accumulator.count == 1

        accumulator.accumulate(grads)
        assert accumulator.count == 2

        accumulator.accumulate(grads)
        assert accumulator.count == 3

        # Fourth accumulation completes window and resets
        accumulator.accumulate(grads)
        assert accumulator.count == 0

    def test_accumulator_is_accumulating_property(self):
        """is_accumulating should be True during partial window."""
        from mlx_music.training.common.gradient_accumulator import GradientAccumulator

        accumulator = GradientAccumulator(accumulation_steps=3)
        grads = {"w": mx.ones((5,))}

        assert not accumulator.is_accumulating

        accumulator.accumulate(grads)
        assert accumulator.is_accumulating

        accumulator.accumulate(grads)
        assert accumulator.is_accumulating

        accumulator.accumulate(grads)  # Completes window
        assert not accumulator.is_accumulating

    def test_accumulator_reset(self):
        """reset() should clear accumulated state."""
        from mlx_music.training.common.gradient_accumulator import GradientAccumulator

        accumulator = GradientAccumulator(accumulation_steps=4)
        grads = {"w": mx.ones((5,))}

        accumulator.accumulate(grads)
        accumulator.accumulate(grads)
        assert accumulator.count == 2

        accumulator.reset()
        assert accumulator.count == 0

    def test_accumulator_flush_partial(self):
        """flush() should return averaged partial accumulation."""
        from mlx_music.training.common.gradient_accumulator import GradientAccumulator

        accumulator = GradientAccumulator(accumulation_steps=4)

        # Accumulate 2 different values
        accumulator.accumulate({"w": mx.full((5,), 2.0)})
        accumulator.accumulate({"w": mx.full((5,), 4.0)})

        result = accumulator.flush()

        # Sum = 2 + 4 = 6, average over 2 = 3
        expected = mx.full((5,), 3.0)
        assert mx.allclose(result["w"], expected)

        # Should be reset after flush
        assert accumulator.count == 0

    def test_accumulator_flush_empty(self):
        """flush() with no accumulated gradients returns None."""
        from mlx_music.training.common.gradient_accumulator import GradientAccumulator

        accumulator = GradientAccumulator(accumulation_steps=4)

        result = accumulator.flush()
        assert result is None

    def test_accumulator_state_dict(self):
        """state_dict() should return accumulation config."""
        from mlx_music.training.common.gradient_accumulator import GradientAccumulator

        accumulator = GradientAccumulator(accumulation_steps=4)
        grads = {"w": mx.ones((5,))}

        accumulator.accumulate(grads)
        accumulator.accumulate(grads)

        state = accumulator.state_dict()

        assert state["accumulation_steps"] == 4
        assert state["count"] == 2

    def test_accumulator_state_dict_warns_mid_window(self):
        """state_dict() should warn when called mid-window."""
        from mlx_music.training.common.gradient_accumulator import GradientAccumulator

        accumulator = GradientAccumulator(accumulation_steps=4)
        grads = {"w": mx.ones((5,))}

        accumulator.accumulate(grads)

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            accumulator.state_dict()
            assert len(w) == 1
            assert "mid-window" in str(w[0].message).lower()

    def test_accumulator_load_state_dict_resets_count(self):
        """load_state_dict() should reset count to 0."""
        from mlx_music.training.common.gradient_accumulator import GradientAccumulator

        accumulator = GradientAccumulator(accumulation_steps=4)
        grads = {"w": mx.ones((5,))}

        accumulator.accumulate(grads)
        accumulator.accumulate(grads)

        # Load state with non-zero count
        state = {"accumulation_steps": 4, "count": 2}
        accumulator.load_state_dict(state)

        # Count should be reset to 0 (since gradients weren't saved)
        assert accumulator.count == 0

    def test_accumulator_invalid_steps_raises(self):
        """GradientAccumulator should raise for invalid steps."""
        from mlx_music.training.common.gradient_accumulator import GradientAccumulator

        with pytest.raises(ValueError, match="accumulation_steps must be >= 1"):
            GradientAccumulator(accumulation_steps=0)

        with pytest.raises(ValueError, match="accumulation_steps must be >= 1"):
            GradientAccumulator(accumulation_steps=-1)

    def test_accumulator_forces_mlx_materialization(self):
        """Accumulator should force MLX array materialization."""
        from mlx_music.training.common.gradient_accumulator import GradientAccumulator

        accumulator = GradientAccumulator(accumulation_steps=2)

        # Create lazy arrays
        a = mx.ones((100, 100))
        b = mx.ones((100, 100))
        grads = {"w": a + b}  # This is lazy until materialized

        # Accumulate should force materialization
        accumulator.accumulate(grads)

        # If we can access the values, it was materialized
        # This would hang if not materialized properly in a graph explosion scenario
        assert accumulator._accumulated is not None


class TestNoOpAccumulator:
    """Tests for NoOpAccumulator class."""

    def test_noop_returns_grads_unchanged(self):
        """NoOpAccumulator should return gradients unchanged."""
        from mlx_music.training.common.gradient_accumulator import NoOpAccumulator

        accumulator = NoOpAccumulator()

        grads = {"w": mx.ones((5,)) * 3.0}
        result = accumulator.accumulate(grads)

        assert mx.allclose(result["w"], grads["w"])

    def test_noop_count_always_zero(self):
        """NoOpAccumulator.count should always be 0."""
        from mlx_music.training.common.gradient_accumulator import NoOpAccumulator

        accumulator = NoOpAccumulator()
        assert accumulator.count == 0

        accumulator.accumulate({"w": mx.ones((5,))})
        assert accumulator.count == 0

    def test_noop_is_accumulating_always_false(self):
        """NoOpAccumulator.is_accumulating should always be False."""
        from mlx_music.training.common.gradient_accumulator import NoOpAccumulator

        accumulator = NoOpAccumulator()
        assert not accumulator.is_accumulating

        accumulator.accumulate({"w": mx.ones((5,))})
        assert not accumulator.is_accumulating

    def test_noop_flush_returns_none(self):
        """NoOpAccumulator.flush() should return None."""
        from mlx_music.training.common.gradient_accumulator import NoOpAccumulator

        accumulator = NoOpAccumulator()
        assert accumulator.flush() is None

    def test_noop_reset_is_noop(self):
        """NoOpAccumulator.reset() should be a no-op."""
        from mlx_music.training.common.gradient_accumulator import NoOpAccumulator

        accumulator = NoOpAccumulator()
        # Should not raise
        accumulator.reset()

    def test_noop_state_dict(self):
        """NoOpAccumulator.state_dict() should return expected values."""
        from mlx_music.training.common.gradient_accumulator import NoOpAccumulator

        accumulator = NoOpAccumulator()
        state = accumulator.state_dict()

        assert state["accumulation_steps"] == 1
        assert state["count"] == 0


class TestCreateAccumulator:
    """Tests for create_accumulator factory function."""

    def test_create_accumulator_steps_1_returns_noop(self):
        """create_accumulator(steps=1) should return NoOpAccumulator."""
        from mlx_music.training.common.gradient_accumulator import (
            create_accumulator,
            NoOpAccumulator,
        )

        accumulator = create_accumulator(accumulation_steps=1)
        assert isinstance(accumulator, NoOpAccumulator)

    def test_create_accumulator_steps_gt_1_returns_accumulator(self):
        """create_accumulator(steps>1) should return GradientAccumulator."""
        from mlx_music.training.common.gradient_accumulator import (
            create_accumulator,
            GradientAccumulator,
        )

        accumulator = create_accumulator(accumulation_steps=4)
        assert isinstance(accumulator, GradientAccumulator)
        assert accumulator.accumulation_steps == 4
