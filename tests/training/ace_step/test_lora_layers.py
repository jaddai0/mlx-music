"""
Tests for ACE-Step LoRA layers.

Tests:
- LoRALinear: Forward pass, scaling, merge_weights, param validation
- apply_lora_to_model: Target modules, recursion depth
- get_lora_parameters: Extracts only LoRA params
- save/load_lora_weights: Path validation, safetensors only
"""

import pytest
import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten


class TestLoRALinear:
    """Tests for LoRALinear class."""

    def test_lora_linear_forward_shape(self):
        """LoRALinear should produce correct output shape."""
        from mlx_music.training.ace_step.lora_layers import LoRALinear

        layer = LoRALinear(in_features=64, out_features=128, rank=16, alpha=16.0)

        x = mx.random.normal((2, 64))
        output = layer(x)

        assert output.shape == (2, 128)

    def test_lora_linear_initialization_is_identity(self):
        """LoRALinear B=0 means initial LoRA contribution is zero."""
        from mlx_music.training.ace_step.lora_layers import LoRALinear

        layer = LoRALinear(in_features=64, out_features=128, rank=16, alpha=16.0)

        # B is initialized to zeros, so LoRA contribution should be zero initially
        assert mx.allclose(layer.lora_B, mx.zeros((128, 16)))

    def test_lora_linear_scaling(self):
        """LoRALinear scaling should be alpha/rank."""
        from mlx_music.training.ace_step.lora_layers import LoRALinear

        rank = 16
        alpha = 32.0
        layer = LoRALinear(in_features=64, out_features=128, rank=rank, alpha=alpha)

        expected_scaling = alpha / rank
        assert layer.scaling == pytest.approx(expected_scaling)

    def test_lora_linear_with_original_layer(self):
        """LoRALinear should wrap existing nn.Linear."""
        from mlx_music.training.ace_step.lora_layers import LoRALinear

        original = nn.Linear(64, 128)
        lora = LoRALinear(
            in_features=64,
            out_features=128,
            rank=16,
            alpha=16.0,
            original_layer=original,
        )

        # Weights should be shared (same array)
        assert mx.array_equal(lora.weight, original.weight)

    def test_lora_linear_merge_weights(self):
        """merge_weights should combine base and LoRA weights."""
        from mlx_music.training.ace_step.lora_layers import LoRALinear

        layer = LoRALinear(in_features=64, out_features=128, rank=16, alpha=16.0)

        # Set non-zero LoRA weights
        layer.lora_A = mx.ones((16, 64)) * 0.1
        layer.lora_B = mx.ones((128, 16)) * 0.1

        original_weight = layer.weight * 1.0  # Copy
        merged = layer.merge_weights()

        # Merged should differ from original (LoRA added)
        assert not mx.allclose(merged, original_weight)

        # Check formula: merged = weight + scaling * B @ A
        expected = original_weight + layer.scaling * mx.matmul(layer.lora_B, layer.lora_A)
        assert mx.allclose(merged, expected)

    def test_lora_linear_get_lora_params(self):
        """get_lora_params should return only LoRA parameters."""
        from mlx_music.training.ace_step.lora_layers import LoRALinear

        layer = LoRALinear(in_features=64, out_features=128, rank=16, alpha=16.0)

        params = layer.get_lora_params()

        assert "lora_A" in params
        assert "lora_B" in params
        assert len(params) == 2

    def test_lora_linear_invalid_rank_raises(self):
        """LoRALinear should raise for invalid rank."""
        from mlx_music.training.ace_step.lora_layers import LoRALinear

        with pytest.raises(ValueError, match="rank must be positive"):
            LoRALinear(in_features=64, out_features=128, rank=0, alpha=16.0)

        with pytest.raises(ValueError, match="rank must be positive"):
            LoRALinear(in_features=64, out_features=128, rank=-1, alpha=16.0)

    def test_lora_linear_invalid_alpha_raises(self):
        """LoRALinear should raise for invalid alpha."""
        from mlx_music.training.ace_step.lora_layers import LoRALinear

        with pytest.raises(ValueError, match="alpha must be positive"):
            LoRALinear(in_features=64, out_features=128, rank=16, alpha=0)

        with pytest.raises(ValueError, match="alpha must be positive"):
            LoRALinear(in_features=64, out_features=128, rank=16, alpha=-1.0)

    def test_lora_linear_invalid_dropout_raises(self):
        """LoRALinear should raise for invalid dropout."""
        from mlx_music.training.ace_step.lora_layers import LoRALinear

        with pytest.raises(ValueError, match="dropout must be in"):
            LoRALinear(in_features=64, out_features=128, rank=16, alpha=16.0, dropout=-0.1)

        with pytest.raises(ValueError, match="dropout must be in"):
            LoRALinear(in_features=64, out_features=128, rank=16, alpha=16.0, dropout=1.0)

    def test_lora_linear_with_dropout(self):
        """LoRALinear should apply dropout to LoRA path."""
        from mlx_music.training.ace_step.lora_layers import LoRALinear

        layer = LoRALinear(in_features=64, out_features=128, rank=16, alpha=16.0, dropout=0.5)

        assert layer.dropout is not None

        x = mx.random.normal((2, 64))
        output = layer(x)  # Should not raise
        assert output.shape == (2, 128)

    def test_lora_linear_without_bias(self):
        """LoRALinear should work without bias."""
        from mlx_music.training.ace_step.lora_layers import LoRALinear

        layer = LoRALinear(in_features=64, out_features=128, rank=16, alpha=16.0, bias=False)

        assert layer.bias is None

        x = mx.random.normal((2, 64))
        output = layer(x)
        assert output.shape == (2, 128)

    def test_lora_linear_3d_input(self):
        """LoRALinear should handle 3D input (batch, seq, features)."""
        from mlx_music.training.ace_step.lora_layers import LoRALinear

        layer = LoRALinear(in_features=64, out_features=128, rank=16, alpha=16.0)

        x = mx.random.normal((2, 10, 64))  # [batch, seq, features]
        output = layer(x)

        assert output.shape == (2, 10, 128)


class TestApplyLoRA:
    """Tests for apply_lora_to_model function."""

    def test_apply_lora_replaces_target_layers(self, attention_model):
        """apply_lora_to_model should replace specified layers."""
        from mlx_music.training.ace_step.lora_layers import apply_lora_to_model, LoRALinear

        # Before: should be nn.Linear
        assert isinstance(attention_model.to_q, nn.Linear)

        num_replaced = apply_lora_to_model(
            attention_model,
            rank=16,
            alpha=16.0,
            target_modules=["to_q", "to_k", "to_v", "to_out"],
        )

        # Should replace 4 layers
        assert num_replaced == 4

        # After: should be LoRALinear
        assert isinstance(attention_model.to_q, LoRALinear)
        assert isinstance(attention_model.to_k, LoRALinear)
        assert isinstance(attention_model.to_v, LoRALinear)
        assert isinstance(attention_model.to_out, LoRALinear)

    def test_apply_lora_preserves_weights(self, attention_model, sample_sequence_input):
        """apply_lora_to_model should preserve original weights."""
        from mlx_music.training.ace_step.lora_layers import apply_lora_to_model

        # Get output before LoRA
        output_before = attention_model(sample_sequence_input)

        apply_lora_to_model(
            attention_model,
            rank=16,
            alpha=16.0,
            target_modules=["to_q", "to_k", "to_v", "to_out"],
        )

        # Get output after LoRA (with B=0, should be same)
        output_after = attention_model(sample_sequence_input)

        # Should be identical (LoRA B=0 initially)
        assert mx.allclose(output_before, output_after, atol=1e-5)

    def test_apply_lora_with_custom_targets(self):
        """apply_lora_to_model should only replace specified targets."""
        from mlx_music.training.ace_step.lora_layers import apply_lora_to_model, LoRALinear

        class CustomModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear1 = nn.Linear(64, 64)
                self.linear2 = nn.Linear(64, 64)
                self.keep_this = nn.Linear(64, 64)

            def __call__(self, x):
                return self.keep_this(self.linear2(self.linear1(x)))

        model = CustomModel()

        apply_lora_to_model(model, rank=8, alpha=8.0, target_modules=["linear1", "linear2"])

        # Should replace only specified
        assert isinstance(model.linear1, LoRALinear)
        assert isinstance(model.linear2, LoRALinear)
        assert isinstance(model.keep_this, nn.Linear)  # Not LoRALinear

    def test_apply_lora_returns_count(self, simple_model):
        """apply_lora_to_model should return number of replaced layers."""
        from mlx_music.training.ace_step.lora_layers import apply_lora_to_model

        count = apply_lora_to_model(
            simple_model,
            rank=8,
            alpha=8.0,
            target_modules=["linear1", "linear2"],
        )

        assert count == 2

    def test_apply_lora_no_matching_layers(self, simple_model):
        """apply_lora_to_model with no matches should return 0."""
        from mlx_music.training.ace_step.lora_layers import apply_lora_to_model

        count = apply_lora_to_model(
            simple_model,
            rank=8,
            alpha=8.0,
            target_modules=["nonexistent_layer"],
        )

        assert count == 0

    def test_apply_lora_nested_modules(self):
        """apply_lora_to_model should handle nested modules."""
        from mlx_music.training.ace_step.lora_layers import apply_lora_to_model, LoRALinear

        class Attention(nn.Module):
            def __init__(self):
                super().__init__()
                self.to_q = nn.Linear(64, 64)
                self.to_k = nn.Linear(64, 64)

        class NestedModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.attn1 = Attention()
                self.attn2 = Attention()

        model = NestedModel()

        count = apply_lora_to_model(
            model, rank=8, alpha=8.0, target_modules=["to_q", "to_k"]
        )

        # Should find and replace all 4 (2 per Attention, 2 Attentions)
        assert count == 4
        assert isinstance(model.attn1.to_q, LoRALinear)
        assert isinstance(model.attn2.to_k, LoRALinear)


class TestGetLoRAParameters:
    """Tests for get_lora_parameters function."""

    def test_get_lora_params_returns_only_lora(self, attention_model):
        """get_lora_parameters should return only LoRA params."""
        from mlx_music.training.ace_step.lora_layers import (
            apply_lora_to_model,
            get_lora_parameters,
        )

        apply_lora_to_model(
            attention_model,
            rank=8,
            alpha=8.0,
            target_modules=["to_q", "to_k", "to_v", "to_out"],
        )

        lora_params = get_lora_parameters(attention_model)
        flat = tree_flatten(lora_params)

        # Should only contain lora_A and lora_B
        for name, _ in flat:
            assert "lora_A" in name or "lora_B" in name

    def test_get_lora_params_excludes_base_weights(self, attention_model):
        """get_lora_parameters should exclude base weights."""
        from mlx_music.training.ace_step.lora_layers import (
            apply_lora_to_model,
            get_lora_parameters,
        )

        apply_lora_to_model(attention_model, rank=8, alpha=8.0)

        lora_params = get_lora_parameters(attention_model)
        flat = tree_flatten(lora_params)
        names = [name for name, _ in flat]

        # Should not contain "weight" or "bias" (base layer params)
        for name in names:
            assert "weight" not in name or "lora" in name
            assert "bias" not in name

    def test_get_lora_params_empty_without_lora(self, simple_model):
        """get_lora_parameters on model without LoRA should return empty."""
        from mlx_music.training.ace_step.lora_layers import get_lora_parameters

        lora_params = get_lora_parameters(simple_model)
        flat = tree_flatten(lora_params)

        assert len(flat) == 0


class TestLoRASaveLoad:
    """Tests for save_lora_weights and load_lora_weights."""

    def test_save_load_roundtrip(self, attention_model, temp_dir):
        """Saved LoRA weights should load correctly."""
        from mlx_music.training.ace_step.lora_layers import (
            apply_lora_to_model,
            save_lora_weights,
            load_lora_weights,
        )

        apply_lora_to_model(attention_model, rank=8, alpha=8.0)

        # Modify LoRA weights
        attention_model.to_q.lora_A = mx.ones((8, 64)) * 0.5
        attention_model.to_q.lora_B = mx.ones((64, 8)) * 0.3

        save_path = temp_dir / "lora.safetensors"
        save_lora_weights(attention_model, save_path)

        assert save_path.exists()

        # Create fresh model
        from tests.training.conftest import AttentionModel
        new_model = AttentionModel()
        apply_lora_to_model(new_model, rank=8, alpha=8.0)

        # Load weights
        count = load_lora_weights(new_model, save_path)

        assert count > 0
        assert mx.allclose(new_model.to_q.lora_A, mx.ones((8, 64)) * 0.5)

    def test_load_rejects_non_safetensors(self, attention_model, temp_dir):
        """load_lora_weights should reject non-safetensors files."""
        from mlx_music.training.ace_step.lora_layers import (
            apply_lora_to_model,
            load_lora_weights,
        )

        apply_lora_to_model(attention_model, rank=8, alpha=8.0)

        pkl_path = temp_dir / "weights.pkl"
        pkl_path.touch()

        with pytest.raises(ValueError, match="safetensors"):
            load_lora_weights(attention_model, pkl_path)

    def test_load_validates_allowed_dir(self, attention_model, temp_dir):
        """load_lora_weights should validate path is within allowed_dir."""
        from mlx_music.training.ace_step.lora_layers import (
            apply_lora_to_model,
            load_lora_weights,
        )

        apply_lora_to_model(attention_model, rank=8, alpha=8.0)

        outside_path = temp_dir.parent / "outside.safetensors"

        with pytest.raises(ValueError, match="within allowed directory"):
            load_lora_weights(attention_model, outside_path, allowed_dir=temp_dir)

    def test_load_nonexistent_raises(self, attention_model, temp_dir):
        """load_lora_weights should raise for nonexistent file."""
        from mlx_music.training.ace_step.lora_layers import (
            apply_lora_to_model,
            load_lora_weights,
        )

        apply_lora_to_model(attention_model, rank=8, alpha=8.0)

        with pytest.raises(FileNotFoundError):
            load_lora_weights(attention_model, temp_dir / "nonexistent.safetensors")

    def test_load_updates_only_matching_params(self, attention_model, temp_dir):
        """load_lora_weights should only update params that exist in checkpoint."""
        from mlx_music.training.ace_step.lora_layers import (
            apply_lora_to_model,
            save_lora_weights,
            load_lora_weights,
        )

        # Create model with LoRA
        apply_lora_to_model(
            attention_model, rank=8, alpha=8.0, target_modules=["to_q", "to_k"]
        )

        # Save only to_q, to_k
        save_path = temp_dir / "partial.safetensors"
        save_lora_weights(attention_model, save_path)

        # Create new model with more LoRA layers
        from tests.training.conftest import AttentionModel
        new_model = AttentionModel()
        apply_lora_to_model(
            new_model, rank=8, alpha=8.0, target_modules=["to_q", "to_k", "to_v", "to_out"]
        )

        # Load should only update matching params
        count = load_lora_weights(new_model, save_path)

        # Should load the matching params
        assert count > 0


class TestMergeLoRAWeights:
    """Tests for merge_lora_weights function."""

    def test_merge_lora_resets_to_zero(self, attention_model):
        """After merge, LoRA matrices should be zero."""
        from mlx_music.training.ace_step.lora_layers import (
            apply_lora_to_model,
            merge_lora_weights,
        )

        apply_lora_to_model(attention_model, rank=8, alpha=8.0)

        # Set non-zero LoRA
        attention_model.to_q.lora_A = mx.ones((8, 64))
        attention_model.to_q.lora_B = mx.ones((64, 8))

        merge_lora_weights(attention_model)

        # LoRA matrices should be zero after merge
        assert mx.allclose(attention_model.to_q.lora_A, mx.zeros((8, 64)))
        assert mx.allclose(attention_model.to_q.lora_B, mx.zeros((64, 8)))

    def test_merge_preserves_functionality(self, attention_model, sample_sequence_input):
        """After merge, output should be same as with LoRA."""
        from mlx_music.training.ace_step.lora_layers import (
            apply_lora_to_model,
            merge_lora_weights,
        )

        apply_lora_to_model(attention_model, rank=8, alpha=8.0)

        # Set non-zero LoRA
        attention_model.to_q.lora_A = mx.random.normal((8, 64)) * 0.1
        attention_model.to_q.lora_B = mx.random.normal((64, 8)) * 0.1

        # Output before merge
        output_before = attention_model(sample_sequence_input)

        # Merge
        merge_lora_weights(attention_model)

        # Output after merge should be same
        output_after = attention_model(sample_sequence_input)

        assert mx.allclose(output_before, output_after, atol=1e-4)
