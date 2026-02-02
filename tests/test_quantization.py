"""
Tests for mlx_music.weights.quantization module.

Tests:
- QuantizationMode: Enum values
- QuantizationConfig: Factory methods, defaults, exclude layers
- quantize_model: Model quantization with INT4/INT8/Mixed
- get_model_size: Parameter and memory counting
- save/load: Quantization config persistence
- QuantizedConv1d: Conv1d to quantized linear conversion
- quantize_conv1d_layers: Batch Conv1d quantization
- QuantizationStats: Statistics and compression ratio
- get_metal_memory_info: Metal memory tracking
"""

import pytest
import mlx.core as mx
import mlx.nn as nn
import tempfile
from pathlib import Path


class TestQuantizationMode:
    """Tests for QuantizationMode enum."""

    def test_mode_values(self):
        """QuantizationMode should have correct string values."""
        from mlx_music.weights.quantization import QuantizationMode

        assert QuantizationMode.NONE.value == "none"
        assert QuantizationMode.INT4.value == "int4"
        assert QuantizationMode.INT8.value == "int8"
        assert QuantizationMode.MIXED.value == "mixed"

    def test_mode_from_string(self):
        """QuantizationMode should be constructable from strings."""
        from mlx_music.weights.quantization import QuantizationMode

        assert QuantizationMode("none") == QuantizationMode.NONE
        assert QuantizationMode("int4") == QuantizationMode.INT4
        assert QuantizationMode("int8") == QuantizationMode.INT8
        assert QuantizationMode("mixed") == QuantizationMode.MIXED


class TestQuantizationConfig:
    """Tests for QuantizationConfig class."""

    def test_default_config(self):
        """Default config should have no quantization."""
        from mlx_music.weights.quantization import QuantizationConfig, QuantizationMode

        config = QuantizationConfig()
        assert config.mode == QuantizationMode.NONE
        assert config.attention_bits == 8
        assert config.ffn_bits == 4
        assert config.group_size == 64

    def test_for_quality_config(self):
        """for_quality() should return INT8 config."""
        from mlx_music.weights.quantization import QuantizationConfig, QuantizationMode

        config = QuantizationConfig.for_quality()
        assert config.mode == QuantizationMode.INT8
        assert config.attention_bits == 8
        assert config.ffn_bits == 8

    def test_for_speed_config(self):
        """for_speed() should return INT4 config."""
        from mlx_music.weights.quantization import QuantizationConfig, QuantizationMode

        config = QuantizationConfig.for_speed()
        assert config.mode == QuantizationMode.INT4
        assert config.attention_bits == 4
        assert config.ffn_bits == 4

    def test_for_balanced_config(self):
        """for_balanced() should return MIXED config."""
        from mlx_music.weights.quantization import QuantizationConfig, QuantizationMode

        config = QuantizationConfig.for_balanced()
        assert config.mode == QuantizationMode.MIXED
        assert config.attention_bits == 8
        assert config.ffn_bits == 4

    def test_exclude_layers_default(self):
        """Default exclude_layers should contain common layer names."""
        from mlx_music.weights.quantization import QuantizationConfig

        config = QuantizationConfig()
        assert "norm" in config.exclude_layers
        assert "rotary_emb" in config.exclude_layers

    def test_custom_exclude_layers(self):
        """Custom exclude_layers should override defaults."""
        from mlx_music.weights.quantization import QuantizationConfig

        config = QuantizationConfig(exclude_layers=["custom_layer"])
        assert config.exclude_layers == ["custom_layer"]


class TestShouldExclude:
    """Tests for _should_exclude helper."""

    def test_exclude_matching_layer(self):
        """Should exclude layers matching exclude patterns."""
        from mlx_music.weights.quantization import _should_exclude, QuantizationConfig

        config = QuantizationConfig(exclude_layers=["norm", "embed"])

        assert _should_exclude("layer_norm", config) is True
        assert _should_exclude("embedding_layer", config) is True

    def test_include_non_matching_layer(self):
        """Should include layers not matching patterns."""
        from mlx_music.weights.quantization import _should_exclude, QuantizationConfig

        config = QuantizationConfig(exclude_layers=["norm"])

        assert _should_exclude("linear_proj", config) is False
        assert _should_exclude("attention", config) is False


class TestGetBitsForLayer:
    """Tests for _get_bits_for_layer helper."""

    def test_none_mode_returns_16(self):
        """NONE mode should return 16 bits."""
        from mlx_music.weights.quantization import _get_bits_for_layer, QuantizationConfig, QuantizationMode

        config = QuantizationConfig(mode=QuantizationMode.NONE)
        assert _get_bits_for_layer("any_layer", config) == 16

    def test_int4_mode_returns_4(self):
        """INT4 mode should return 4 bits."""
        from mlx_music.weights.quantization import _get_bits_for_layer, QuantizationConfig, QuantizationMode

        config = QuantizationConfig(mode=QuantizationMode.INT4)
        assert _get_bits_for_layer("any_layer", config) == 4

    def test_int8_mode_returns_8(self):
        """INT8 mode should return 8 bits."""
        from mlx_music.weights.quantization import _get_bits_for_layer, QuantizationConfig, QuantizationMode

        config = QuantizationConfig(mode=QuantizationMode.INT8)
        assert _get_bits_for_layer("any_layer", config) == 8

    def test_mixed_mode_attention_layers(self):
        """MIXED mode should use attention_bits for attention layers."""
        from mlx_music.weights.quantization import _get_bits_for_layer, QuantizationConfig, QuantizationMode

        config = QuantizationConfig(
            mode=QuantizationMode.MIXED,
            attention_bits=8,
            ffn_bits=4,
        )

        assert _get_bits_for_layer("self_attn.q_proj", config) == 8
        assert _get_bits_for_layer("attention.v_proj", config) == 8

    def test_mixed_mode_ffn_layers(self):
        """MIXED mode should use ffn_bits for FFN layers."""
        from mlx_music.weights.quantization import _get_bits_for_layer, QuantizationConfig, QuantizationMode

        config = QuantizationConfig(
            mode=QuantizationMode.MIXED,
            attention_bits=8,
            ffn_bits=4,
        )

        assert _get_bits_for_layer("ff.up_proj", config) == 4
        assert _get_bits_for_layer("mlp.fc1", config) == 4


class TestGetModelSize:
    """Tests for get_model_size function."""

    def test_model_size_simple(self):
        """get_model_size should return correct parameter count."""
        from mlx_music.weights.quantization import get_model_size

        class SimpleModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(64, 32)

            def __call__(self, x):
                return self.linear(x)

        model = SimpleModel()
        num_params, size_mb = get_model_size(model)

        # Linear(64, 32) = 64*32 weight + 32 bias = 2048 + 32 = 2080 params
        assert num_params == 2080

    def test_model_size_nested(self):
        """get_model_size should count nested module parameters."""
        from mlx_music.weights.quantization import get_model_size

        class NestedModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.layer1 = nn.Linear(32, 32)
                self.layer2 = nn.Linear(32, 32)

            def __call__(self, x):
                return self.layer2(self.layer1(x))

        model = NestedModel()
        num_params, size_mb = get_model_size(model)

        # 2 x (32*32 + 32) = 2 * 1056 = 2112 params
        assert num_params == 2112
        assert size_mb > 0


class TestQuantizeModel:
    """Tests for quantize_model function."""

    def test_quantize_model_returns_same_object(self):
        """quantize_model should modify model in-place."""
        from mlx_music.weights.quantization import quantize_model

        class SimpleModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(64, 64)

            def __call__(self, x):
                return self.linear(x)

        model = SimpleModel()
        result = quantize_model(model, bits=4)

        assert result is model

    def test_quantize_model_none_mode_unchanged(self):
        """NONE mode should not modify the model."""
        from mlx_music.weights.quantization import quantize_model, QuantizationConfig, QuantizationMode

        class SimpleModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(64, 64)

            def __call__(self, x):
                return self.linear(x)

        model = SimpleModel()
        config = QuantizationConfig(mode=QuantizationMode.NONE)

        # Should return immediately without modification
        result = quantize_model(model, config=config)
        assert result is model

    def test_quantize_model_produces_output(self):
        """Quantized model should still produce valid output."""
        from mlx_music.weights.quantization import quantize_model

        class SimpleModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(64, 64)

            def __call__(self, x):
                return self.linear(x)

        model = SimpleModel()
        quantize_model(model, bits=4)

        x = mx.random.normal((2, 64))
        output = model(x)
        mx.eval(output)

        assert output.shape == (2, 64)
        assert not mx.any(mx.isnan(output))


class TestQuantizedConv1d:
    """Tests for QuantizedConv1d class."""

    def test_quantized_conv1d_init(self):
        """QuantizedConv1d should initialize correctly."""
        from mlx_music.weights.quantization import QuantizedConv1d

        qconv = QuantizedConv1d(in_channels=64, out_channels=128, bits=4)

        assert qconv.in_channels == 64
        assert qconv.out_channels == 128
        assert qconv.bits == 4

    def test_quantized_conv1d_forward(self):
        """QuantizedConv1d should produce valid output."""
        from mlx_music.weights.quantization import QuantizedConv1d

        qconv = QuantizedConv1d(in_channels=64, out_channels=128, bits=4)

        x = mx.random.normal((2, 10, 64))  # (batch, seq, channels)
        output = qconv(x)
        mx.eval(output)

        assert output.shape == (2, 10, 128)
        assert not mx.any(mx.isnan(output))

    def test_quantized_conv1d_from_conv1d(self):
        """from_conv1d should convert kernel_size=1 Conv1d."""
        from mlx_music.weights.quantization import QuantizedConv1d

        # Create Conv1d with kernel_size=1
        conv = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=1)

        qconv = QuantizedConv1d.from_conv1d(conv, bits=4)

        assert qconv.in_channels == 64
        assert qconv.out_channels == 128

    def test_quantized_conv1d_rejects_large_kernel(self):
        """from_conv1d should reject kernel_size > 1."""
        from mlx_music.weights.quantization import QuantizedConv1d

        # Create Conv1d with kernel_size=3
        conv = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3)

        with pytest.raises(ValueError, match="kernel_size=1"):
            QuantizedConv1d.from_conv1d(conv, bits=4)


class TestQuantizationStats:
    """Tests for QuantizationStats dataclass."""

    def test_memory_saved_calculation(self):
        """memory_saved_mb should be initial - final."""
        from mlx_music.weights.quantization import QuantizationStats

        stats = QuantizationStats(
            initial_params=1000,
            final_params=1000,
            initial_memory_mb=100.0,
            final_memory_mb=25.0,
            linear_layers_quantized=10,
            conv1d_layers_quantized=2,
            layers_skipped=3,
        )

        assert stats.memory_saved_mb == 75.0

    def test_compression_ratio_calculation(self):
        """compression_ratio should be initial / final."""
        from mlx_music.weights.quantization import QuantizationStats

        stats = QuantizationStats(
            initial_params=1000,
            final_params=1000,
            initial_memory_mb=100.0,
            final_memory_mb=25.0,
            linear_layers_quantized=10,
            conv1d_layers_quantized=2,
            layers_skipped=3,
        )

        assert stats.compression_ratio == 4.0

    def test_compression_ratio_zero_final(self):
        """compression_ratio should handle zero final memory."""
        from mlx_music.weights.quantization import QuantizationStats

        stats = QuantizationStats(
            initial_params=1000,
            final_params=0,
            initial_memory_mb=100.0,
            final_memory_mb=0.0,
            linear_layers_quantized=0,
            conv1d_layers_quantized=0,
            layers_skipped=0,
        )

        assert stats.compression_ratio == 0.0


class TestGetMetalMemoryInfo:
    """Tests for get_metal_memory_info function."""

    def test_returns_dict_with_keys(self):
        """get_metal_memory_info should return dict with expected keys."""
        from mlx_music.weights.quantization import get_metal_memory_info

        info = get_metal_memory_info()

        assert isinstance(info, dict)
        assert "active_mb" in info
        assert "peak_mb" in info
        assert "cache_mb" in info

    def test_returns_non_negative_values(self):
        """Memory values should be non-negative."""
        from mlx_music.weights.quantization import get_metal_memory_info

        info = get_metal_memory_info()

        assert info["active_mb"] >= 0
        assert info["peak_mb"] >= 0
        assert info["cache_mb"] >= 0


class TestSaveLoadQuantizationConfig:
    """Tests for save_quantized_model and load_quantization_config."""

    def test_save_load_config_roundtrip(self, tmp_path):
        """Config should survive save/load roundtrip."""
        from mlx_music.weights.quantization import (
            QuantizationConfig,
            QuantizationMode,
            save_quantized_model,
            load_quantization_config,
        )

        class SimpleModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(32, 32)

            def __call__(self, x):
                return self.linear(x)

        model = SimpleModel()
        config = QuantizationConfig(
            mode=QuantizationMode.INT8,
            attention_bits=8,
            ffn_bits=4,
            group_size=32,
        )

        # Save
        save_quantized_model(model, tmp_path, config=config)

        # Load
        loaded_config = load_quantization_config(tmp_path)

        assert loaded_config is not None
        assert loaded_config.mode == QuantizationMode.INT8
        assert loaded_config.attention_bits == 8
        assert loaded_config.ffn_bits == 4
        assert loaded_config.group_size == 32

    def test_load_nonexistent_config_returns_none(self, tmp_path):
        """load_quantization_config should return None for missing file."""
        from mlx_music.weights.quantization import load_quantization_config

        result = load_quantization_config(tmp_path / "nonexistent")
        assert result is None

    def test_save_creates_safetensors_file(self, tmp_path):
        """save_quantized_model should create .safetensors file."""
        from mlx_music.weights.quantization import save_quantized_model

        class SimpleModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(32, 32)

            def __call__(self, x):
                return self.linear(x)

        model = SimpleModel()
        output_path = save_quantized_model(model, tmp_path)

        assert output_path.exists()
        assert output_path.suffix == ".safetensors"


class TestQuantizeModelWithStats:
    """Tests for quantize_model_with_stats function."""

    def test_returns_model_and_stats(self):
        """quantize_model_with_stats should return (model, stats) tuple."""
        from mlx_music.weights.quantization import (
            quantize_model_with_stats,
            QuantizationStats,
        )

        class SimpleModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(64, 64)

            def __call__(self, x):
                return self.linear(x)

        model = SimpleModel()
        result_model, stats = quantize_model_with_stats(model, bits=4, verbose=False)

        assert result_model is model
        assert isinstance(stats, QuantizationStats)

    def test_stats_has_positive_values(self):
        """Stats should have positive initial memory values."""
        from mlx_music.weights.quantization import quantize_model_with_stats

        class SimpleModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(64, 64)

            def __call__(self, x):
                return self.linear(x)

        model = SimpleModel()
        _, stats = quantize_model_with_stats(model, bits=4, verbose=False)

        assert stats.initial_params > 0
        assert stats.initial_memory_mb > 0


class TestIterModules:
    """Tests for _iter_modules helper."""

    def test_iter_modules_simple(self):
        """_iter_modules should yield all modules."""
        from mlx_music.weights.quantization import _iter_modules

        class SimpleModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(32, 32)

            def __call__(self, x):
                return self.linear(x)

        model = SimpleModel()
        modules = list(_iter_modules(model))

        # Should yield at least the model itself and the linear layer
        paths = [path for path, _ in modules]
        assert "" in paths  # Root module

    def test_iter_modules_nested(self):
        """_iter_modules should traverse nested modules."""
        from mlx_music.weights.quantization import _iter_modules

        class Inner(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc = nn.Linear(32, 32)

            def __call__(self, x):
                return self.fc(x)

        class Outer(nn.Module):
            def __init__(self):
                super().__init__()
                self.inner = Inner()

            def __call__(self, x):
                return self.inner(x)

        model = Outer()
        modules = list(_iter_modules(model))

        # Should find nested modules
        paths = [path for path, _ in modules]
        assert any("inner" in p for p in paths)


@pytest.fixture
def tmp_path():
    """Create temporary directory for tests."""
    with tempfile.TemporaryDirectory() as d:
        yield Path(d)
