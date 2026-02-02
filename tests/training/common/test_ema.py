"""
Tests for Exponential Moving Average (EMA) model.

Tests:
- EMAModel: update(), apply_shadow(), restore(), copy_to(), save/load
- NoOpEMA: All methods are no-ops
- create_ema: Factory function for enabled/disabled
"""

import pytest
import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten


class TestEMAModel:
    """Tests for EMAModel class."""

    def test_ema_initializes_shadow_from_model(self, simple_model):
        """EMAModel should initialize shadow weights from model."""
        from mlx_music.training.common.ema import EMAModel

        ema = EMAModel(simple_model, decay=0.9999)

        # Shadow should match model params initially
        model_flat = dict(tree_flatten(simple_model.parameters()))
        shadow_flat = dict(tree_flatten(ema.shadow))

        for name, param in model_flat.items():
            assert name in shadow_flat
            assert mx.allclose(param, shadow_flat[name])

    def test_ema_update_moves_toward_current(self, simple_model):
        """EMAModel.update() should move shadow toward current weights."""
        from mlx_music.training.common.ema import EMAModel

        decay = 0.9
        ema = EMAModel(simple_model, decay=decay)

        # Get initial shadow
        initial_shadow = dict(tree_flatten(ema.shadow))

        # Modify model weights using MLX's children() method
        for name, child in simple_model.children().items():
            if isinstance(child, nn.Linear):
                child.weight = child.weight + 1.0
        mx.eval(simple_model.parameters())

        # Update EMA
        ema.update(simple_model)

        # Shadow should move toward new weights
        # shadow_new = decay * shadow_old + (1 - decay) * current
        model_flat = dict(tree_flatten(simple_model.parameters()))
        shadow_flat = dict(tree_flatten(ema.shadow))

        for name in initial_shadow:
            if name in model_flat:
                expected = decay * initial_shadow[name] + (1 - decay) * model_flat[name]
                assert mx.allclose(shadow_flat[name], expected, atol=1e-5)

    def test_ema_apply_shadow_swaps_weights(self, simple_model, sample_input):
        """apply_shadow() should swap shadow weights into model."""
        from mlx_music.training.common.ema import EMAModel

        ema = EMAModel(simple_model, decay=0.9999)

        # Modify model weights using MLX's children() method
        for name, child in simple_model.children().items():
            if isinstance(child, nn.Linear):
                child.weight = child.weight + 10.0
        mx.eval(simple_model.parameters())

        # Update EMA a few times so shadow differs from model
        for _ in range(5):
            ema.update(simple_model)

        # Store output with modified weights
        output_modified = simple_model(sample_input)
        mx.eval(output_modified)

        # Apply shadow
        ema.apply_shadow(simple_model)

        # Output should now be different (using shadow weights)
        output_shadow = simple_model(sample_input)
        mx.eval(output_shadow)

        # They should be different (shadow is behind the modified weights)
        assert not mx.allclose(output_modified, output_shadow)

        # Restore
        ema.restore(simple_model)

        # Output should match modified again
        output_restored = simple_model(sample_input)
        mx.eval(output_restored)
        assert mx.allclose(output_modified, output_restored)

    def test_ema_restore_without_apply_raises(self, simple_model):
        """restore() without prior apply_shadow() should raise."""
        from mlx_music.training.common.ema import EMAModel

        ema = EMAModel(simple_model, decay=0.9999)

        with pytest.raises(RuntimeError, match="restore.*without prior apply_shadow"):
            ema.restore(simple_model)

    def test_ema_nested_apply_raises(self, simple_model):
        """Nested apply_shadow() calls should raise."""
        from mlx_music.training.common.ema import EMAModel

        ema = EMAModel(simple_model, decay=0.9999)

        ema.apply_shadow(simple_model)

        with pytest.raises(RuntimeError, match="backup exists"):
            ema.apply_shadow(simple_model)

        # Cleanup
        ema.restore(simple_model)

    def test_ema_copy_to_permanently_updates_model(self, simple_model, sample_input):
        """copy_to() should permanently copy shadow to model."""
        from mlx_music.training.common.ema import EMAModel

        ema = EMAModel(simple_model, decay=0.9)

        # Modify and update a few times
        for _ in range(10):
            for name, child in simple_model.children().items():
                if isinstance(child, nn.Linear):
                    child.weight = child.weight + 0.1
            mx.eval(simple_model.parameters())
            ema.update(simple_model)

        # Get shadow output
        ema.apply_shadow(simple_model)
        shadow_output = simple_model(sample_input)
        mx.eval(shadow_output)
        ema.restore(simple_model)

        # Copy shadow permanently
        ema.copy_to(simple_model)

        # Model output should now match shadow output
        model_output = simple_model(sample_input)
        mx.eval(model_output)
        assert mx.allclose(shadow_output, model_output)

    def test_ema_save_and_load(self, simple_model, temp_dir):
        """EMAModel should save and load correctly."""
        from mlx_music.training.common.ema import EMAModel

        ema = EMAModel(simple_model, decay=0.99)

        # Update a few times
        for _ in range(5):
            for name, child in simple_model.children().items():
                if isinstance(child, nn.Linear):
                    child.weight = child.weight + 0.5
            mx.eval(simple_model.parameters())
            ema.update(simple_model)

        # Save
        save_path = temp_dir / "ema_weights.safetensors"
        ema.save(save_path)

        assert save_path.exists()

        # Load into new EMA
        new_ema = EMAModel.load(save_path, simple_model, decay=0.99)

        # Shadows should match
        old_flat = dict(tree_flatten(ema.shadow))
        new_flat = dict(tree_flatten(new_ema.shadow))

        for name in old_flat:
            assert mx.allclose(old_flat[name], new_flat[name])

    def test_ema_load_rejects_non_safetensors(self, simple_model, temp_dir):
        """load() should reject non-safetensors files."""
        from mlx_music.training.common.ema import EMAModel

        # Create a fake .pkl file
        pkl_path = temp_dir / "ema.pkl"
        pkl_path.touch()

        with pytest.raises(ValueError, match="safetensors"):
            EMAModel.load(pkl_path, simple_model, decay=0.99)

    def test_ema_load_validates_path(self, simple_model, temp_dir):
        """load() should validate path is within allowed_dir."""
        from mlx_music.training.common.ema import EMAModel

        # Try to load from outside allowed_dir
        outside_path = temp_dir.parent / "outside.safetensors"

        with pytest.raises(ValueError, match="within allowed directory"):
            EMAModel.load(outside_path, simple_model, decay=0.99, allowed_dir=temp_dir)

    def test_ema_state_dict_roundtrip(self, simple_model):
        """state_dict/load_state_dict should preserve state."""
        from mlx_music.training.common.ema import EMAModel

        ema = EMAModel(simple_model, decay=0.95)

        # Update a few times
        for _ in range(3):
            ema.update(simple_model)

        state = ema.state_dict()

        # Create new EMA and load state
        new_ema = EMAModel(simple_model, decay=0.9999)  # Different decay
        new_ema.load_state_dict(state)

        assert new_ema.decay == 0.95  # Should load decay from state

    def test_ema_invalid_decay_raises(self, simple_model):
        """EMAModel should raise for invalid decay values."""
        from mlx_music.training.common.ema import EMAModel

        with pytest.raises(ValueError, match="decay must be in"):
            EMAModel(simple_model, decay=-0.1)

        with pytest.raises(ValueError, match="decay must be in"):
            EMAModel(simple_model, decay=1.5)

    def test_ema_high_decay_slow_update(self, simple_model):
        """High decay (0.9999) should result in slow shadow updates."""
        from mlx_music.training.common.ema import EMAModel

        ema = EMAModel(simple_model, decay=0.9999)
        initial_shadow = dict(tree_flatten(ema.shadow))

        # Large modification to model using MLX's children() method
        for name, child in simple_model.children().items():
            if isinstance(child, nn.Linear):
                child.weight = child.weight + 100.0
        mx.eval(simple_model.parameters())

        # Single update
        ema.update(simple_model)

        # Shadow should barely move (high decay = slow update)
        shadow_flat = dict(tree_flatten(ema.shadow))
        for name in initial_shadow:
            diff = mx.abs(shadow_flat[name] - initial_shadow[name]).mean()
            assert float(diff) < 1.0  # Should be small despite large model change


class TestNoOpEMA:
    """Tests for NoOpEMA class."""

    def test_noop_ema_update_does_nothing(self, simple_model):
        """NoOpEMA.update() should be a no-op."""
        from mlx_music.training.common.ema import NoOpEMA

        ema = NoOpEMA(simple_model, decay=0.9999)

        # Get initial params and force evaluation to capture actual values
        initial = dict(tree_flatten(simple_model.parameters()))
        mx.eval(*initial.values())

        # Directly modify model weights (more reliable than iterating children)
        simple_model.linear1.weight = simple_model.linear1.weight + 1.0
        simple_model.linear2.weight = simple_model.linear2.weight + 1.0
        mx.eval(simple_model.parameters())

        # Update EMA (should be no-op)
        ema.update(simple_model)

        # Model should still have modified params (EMA didn't touch it)
        current = dict(tree_flatten(simple_model.parameters()))
        for name in initial:
            if 'weight' in name:  # Only check weights we modified
                assert not mx.allclose(initial[name], current[name])

    def test_noop_ema_apply_restore_are_noops(self, simple_model, sample_input):
        """NoOpEMA apply_shadow/restore should be no-ops."""
        from mlx_music.training.common.ema import NoOpEMA

        ema = NoOpEMA(simple_model, decay=0.9999)

        # Get output before
        output_before = simple_model(sample_input)
        mx.eval(output_before)

        # Apply shadow (no-op)
        ema.apply_shadow(simple_model)

        # Output should be same
        output_during = simple_model(sample_input)
        mx.eval(output_during)
        assert mx.allclose(output_before, output_during)

        # Restore (no-op)
        ema.restore(simple_model)

        # Output should still be same
        output_after = simple_model(sample_input)
        mx.eval(output_after)
        assert mx.allclose(output_before, output_after)

    def test_noop_ema_state_dict_is_empty(self, simple_model):
        """NoOpEMA.state_dict() should return empty dict."""
        from mlx_music.training.common.ema import NoOpEMA

        ema = NoOpEMA(simple_model)
        state = ema.state_dict()
        assert state == {}

    def test_noop_ema_load_state_is_noop(self, simple_model):
        """NoOpEMA.load_state_dict() should be a no-op."""
        from mlx_music.training.common.ema import NoOpEMA

        ema = NoOpEMA(simple_model)
        # Should not raise
        ema.load_state_dict({"some": "data"})


class TestCreateEMA:
    """Tests for create_ema factory function."""

    def test_create_ema_enabled_returns_emamodel(self, simple_model):
        """create_ema(enabled=True) should return EMAModel."""
        from mlx_music.training.common.ema import create_ema, EMAModel

        ema = create_ema(simple_model, enabled=True, decay=0.999)
        assert isinstance(ema, EMAModel)

    def test_create_ema_disabled_returns_noop(self, simple_model):
        """create_ema(enabled=False) should return NoOpEMA."""
        from mlx_music.training.common.ema import create_ema, NoOpEMA

        ema = create_ema(simple_model, enabled=False, decay=0.999)
        assert isinstance(ema, NoOpEMA)

    def test_create_ema_default_decay(self, simple_model):
        """create_ema should use default decay of 0.9999."""
        from mlx_music.training.common.ema import create_ema, EMAModel

        ema = create_ema(simple_model, enabled=True)
        assert isinstance(ema, EMAModel)
        assert ema.decay == 0.9999

    def test_create_ema_custom_decay(self, simple_model):
        """create_ema should respect custom decay."""
        from mlx_music.training.common.ema import create_ema

        ema = create_ema(simple_model, enabled=True, decay=0.95)
        assert ema.decay == 0.95
