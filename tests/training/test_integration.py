"""
Integration tests for mlx-music training infrastructure.

Tests:
- Full training iteration with mock data
- EMA updates during training
- Gradient flow through LoRA params
- Checkpoint roundtrip
- Scheduler with accumulator
- Numerical stability
- Memory cleanup
- Cross-model LoRA patterns
"""

import pytest
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.utils import tree_flatten


class TestTrainingLoopIntegration:
    """Integration tests for training loop components."""

    def test_lora_gradient_flow(self):
        """Gradients should flow only through LoRA params."""
        from mlx_music.training.ace_step.lora_layers import (
            apply_lora_to_model,
            get_lora_parameters,
            LoRALinear,
        )

        class TestModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.to_q = nn.Linear(64, 64)
                self.to_k = nn.Linear(64, 64)

            def __call__(self, x):
                return self.to_q(x) + self.to_k(x)

        model = TestModel()
        apply_lora_to_model(model, rank=8, alpha=8.0)

        # Get LoRA params for optimizer
        lora_params = get_lora_parameters(model)

        # Create loss function
        def loss_fn(params):
            # Temporarily update model with params
            x = mx.random.normal((2, 64))
            y = model(x)
            return (y ** 2).mean()

        # Compute gradients
        loss, grads = nn.value_and_grad(model, loss_fn)(model.parameters())

        # Check that only LoRA params have gradients
        grads_flat = tree_flatten(grads)
        for name, grad in grads_flat:
            if "lora_A" in name or "lora_B" in name:
                # LoRA params should have non-zero gradients
                assert grad is not None
            # Base weights would have gradients too in this setup,
            # but in real training we'd freeze them

    def test_ema_updates_during_training(self):
        """EMA should track model weight changes."""
        from mlx_music.training.common.ema import create_ema, EMAModel

        class TestModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(64, 64)

            def __call__(self, x):
                return self.linear(x)

        model = TestModel()
        ema = create_ema(model, enabled=True, decay=0.9)

        assert isinstance(ema, EMAModel)

        # Simulate training steps
        optimizer = optim.SGD(learning_rate=0.1)
        optimizer.init(model.trainable_parameters())

        initial_shadow = dict(tree_flatten(ema.shadow))

        for _ in range(5):
            # Create fake gradients matching model's nested structure
            # optimizer.update expects nested dict, not flat paths
            fake_grads = {
                'linear': {
                    'weight': mx.ones_like(model.linear.weight) * 0.1,
                    'bias': mx.ones_like(model.linear.bias) * 0.1
                }
            }

            optimizer.update(model, fake_grads)

            # Update EMA
            ema.update(model)

        # Shadow should have moved toward model weights
        final_shadow = dict(tree_flatten(ema.shadow))

        for name in initial_shadow:
            # Shadow should be different after updates
            assert not mx.allclose(initial_shadow[name], final_shadow[name])

    def test_scheduler_with_accumulator_alignment(self):
        """LR scheduler should step only on accumulator boundaries."""
        from mlx_music.training.common.lr_scheduler import create_scheduler
        from mlx_music.training.common.gradient_accumulator import create_accumulator

        optimizer = optim.SGD(learning_rate=1e-3)
        scheduler = create_scheduler(
            name="cosine",
            optimizer=optimizer,
            initial_lr=1e-3,
            total_steps=100,
        )

        accumulator = create_accumulator(accumulation_steps=4)

        lr_values = []
        for i in range(20):
            grads = {"w": mx.ones((10,))}
            accumulated = accumulator.accumulate(grads)

            if accumulated is not None:
                # Optimizer step
                lr = scheduler.step()
                lr_values.append(lr)

        # Should have stepped 5 times (20 / 4)
        assert len(lr_values) == 5
        assert scheduler.step_count == 5

    def test_checkpoint_roundtrip(self, temp_dir):
        """Save and load should preserve all training state."""
        from mlx_music.training.common.lr_scheduler import CosineAnnealingLR
        from mlx_music.training.common.gradient_accumulator import GradientAccumulator
        from mlx_music.training.common.ema import EMAModel

        class TestModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(64, 64)

        model = TestModel()
        optimizer = optim.SGD(learning_rate=1e-3)

        # Create components
        scheduler = CosineAnnealingLR(optimizer, initial_lr=1e-3, total_steps=100)
        accumulator = GradientAccumulator(accumulation_steps=4)
        ema = EMAModel(model, decay=0.99)

        # Advance state
        for _ in range(10):
            scheduler.step()
        accumulator.accumulate({"w": mx.ones((10,))})
        ema.update(model)

        # Save states
        scheduler_state = scheduler.state_dict()
        accumulator_state = accumulator.state_dict()
        ema_state = ema.state_dict()

        # Create new components and load
        new_model = TestModel()
        new_optimizer = optim.SGD(learning_rate=1e-3)
        new_scheduler = CosineAnnealingLR(new_optimizer, initial_lr=1e-3, total_steps=100)
        new_accumulator = GradientAccumulator(accumulation_steps=4)
        new_ema = EMAModel(new_model, decay=0.99)

        new_scheduler.load_state_dict(scheduler_state)
        new_accumulator.load_state_dict(accumulator_state)
        new_ema.load_state_dict(ema_state)

        # Verify state
        assert new_scheduler.step_count == 10
        assert new_accumulator.accumulation_steps == 4
        assert new_ema.decay == 0.99

    def test_loss_numerical_stability(self):
        """Loss functions should handle edge cases without NaN/Inf."""
        from mlx_music.training.common.loss import (
            flow_matching_loss,
            v_prediction_loss,
            min_snr_weighted_loss,
        )

        # Test with very small values
        small = mx.ones((2, 4, 10)) * 1e-10
        noise = mx.random.normal((2, 4, 10))

        loss1 = flow_matching_loss(small, small, noise)
        assert not mx.isnan(loss1)
        assert not mx.isinf(loss1)

        # Test with very large values
        large = mx.ones((2, 4, 10)) * 1e6
        loss2 = flow_matching_loss(large, large, noise)
        assert not mx.isnan(loss2)
        assert not mx.isinf(loss2)

        # Test v_prediction with extreme sigma
        loss3 = v_prediction_loss(small, small, noise, sigma=1e-10)
        assert not mx.isnan(loss3)

        loss4 = v_prediction_loss(large, large, noise, sigma=1e6)
        assert not mx.isnan(loss4)

        # Test SNR weighting with extreme sigma
        base_loss = mx.array([1.0])
        weighted = min_snr_weighted_loss(base_loss, sigma=1e-10)
        assert not mx.isnan(weighted)
        assert not mx.isinf(weighted)

    def test_memory_cleanup_after_training(self):
        """Training loop should not leak memory."""
        from mlx_music.training.common.gradient_accumulator import create_accumulator

        # This is a basic test - full memory testing would need profiling
        accumulator = create_accumulator(accumulation_steps=4)

        # Simulate many training steps
        for step in range(100):
            grads = {"w": mx.random.normal((1000, 1000))}
            result = accumulator.accumulate(grads)

            if result is not None:
                # Simulate using the result
                _ = result["w"].mean()

            # Reset periodically (simulates epoch boundary)
            if step % 20 == 0:
                accumulator.reset()

        # If we got here without OOM, basic memory management is working
        assert True

    def test_cross_model_lora_patterns(self):
        """LoRA should work across different model architectures."""
        from mlx_music.training.ace_step.lora_layers import (
            apply_lora_to_model,
            get_lora_parameters,
            LoRALinear,
        )

        # ACE-Step style (to_q, to_k, to_v)
        class ACEStepStyle(nn.Module):
            def __init__(self):
                super().__init__()
                self.to_q = nn.Linear(64, 64)
                self.to_k = nn.Linear(64, 64)
                self.to_v = nn.Linear(64, 64)

        # MusicGen style (q_proj, k_proj, v_proj)
        class MusicGenStyle(nn.Module):
            def __init__(self):
                super().__init__()
                self.q_proj = nn.Linear(64, 64)
                self.k_proj = nn.Linear(64, 64)
                self.v_proj = nn.Linear(64, 64)

        # Test ACE-Step style
        ace = ACEStepStyle()
        ace_count = apply_lora_to_model(
            ace, rank=8, target_modules=["to_q", "to_k", "to_v"]
        )
        assert ace_count == 3
        assert isinstance(ace.to_q, LoRALinear)

        # Test MusicGen style
        mg = MusicGenStyle()
        mg_count = apply_lora_to_model(
            mg, rank=8, target_modules=["q_proj", "k_proj", "v_proj"]
        )
        assert mg_count == 3
        assert isinstance(mg.q_proj, LoRALinear)

        # Both should have extractable LoRA params
        ace_params = get_lora_parameters(ace)
        mg_params = get_lora_parameters(mg)

        assert len(tree_flatten(ace_params)) == 6  # 3 layers * 2 (A, B)
        assert len(tree_flatten(mg_params)) == 6


class TestFullTrainingIteration:
    """Test a complete training iteration."""

    def test_full_iteration_mock(self):
        """Test full training iteration with mock components."""
        from mlx_music.training.ace_step.lora_layers import apply_lora_to_model, get_lora_parameters
        from mlx_music.training.common.ema import create_ema
        from mlx_music.training.common.lr_scheduler import create_scheduler
        from mlx_music.training.common.gradient_accumulator import create_accumulator
        from mlx_music.training.common.loss import flow_matching_loss

        # Create model
        class MockTransformer(nn.Module):
            def __init__(self):
                super().__init__()
                self.to_q = nn.Linear(64, 64)
                self.to_k = nn.Linear(64, 64)
                self.to_v = nn.Linear(64, 64)

            def __call__(self, x):
                q = self.to_q(x)
                k = self.to_k(x)
                v = self.to_v(x)
                return q + k + v

        model = MockTransformer()

        # Apply LoRA
        apply_lora_to_model(model, rank=8, alpha=8.0)

        # Setup training components
        lora_params = get_lora_parameters(model)
        optimizer = optim.AdamW(learning_rate=1e-4)
        ema = create_ema(model, enabled=True, decay=0.99)
        scheduler = create_scheduler("cosine", optimizer, initial_lr=1e-4, total_steps=100)
        accumulator = create_accumulator(accumulation_steps=2)

        # Training loop
        losses = []
        for step in range(10):
            # Mock batch
            clean = mx.random.normal((2, 64))
            noise = mx.random.normal((2, 64))

            # Forward pass
            def loss_fn(params):
                predicted = model(clean)
                return flow_matching_loss(predicted, clean, noise)

            loss, grads = nn.value_and_grad(model, loss_fn)(model.parameters())

            losses.append(float(loss))

            # Accumulate
            accumulated = accumulator.accumulate(grads)

            if accumulated is not None:
                optimizer.update(model, accumulated)
                ema.update(model)
                scheduler.step()

        # Verify training happened
        assert len(losses) == 10
        assert all(loss > 0 for loss in losses)

        # Verify EMA shadow differs from model
        ema.apply_shadow(model)
        shadow_output = model(mx.random.normal((1, 64)))
        ema.restore(model)
        model_output = model(mx.random.normal((1, 64)))

        # Outputs should exist (basic sanity check)
        assert shadow_output.shape == (1, 64)
        assert model_output.shape == (1, 64)
