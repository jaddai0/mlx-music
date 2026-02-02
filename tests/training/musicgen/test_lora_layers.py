"""
Tests for MusicGen LoRA layers.

Tests:
- apply_lora_to_musicgen: Target modules, decoder-only mode
- MUSICGEN_LORA_TARGETS: Default target layers
"""

import pytest
import mlx.core as mx
import mlx.nn as nn


class TestApplyLoRAToMusicGen:
    """Tests for apply_lora_to_musicgen function."""

    def test_musicgen_lora_targets_defined(self):
        """MUSICGEN_LORA_TARGETS should define default targets."""
        from mlx_music.training.musicgen.lora_layers import MUSICGEN_LORA_TARGETS

        assert "q_proj" in MUSICGEN_LORA_TARGETS
        assert "k_proj" in MUSICGEN_LORA_TARGETS
        assert "v_proj" in MUSICGEN_LORA_TARGETS
        assert "out_proj" in MUSICGEN_LORA_TARGETS

    def test_apply_lora_to_decoder(self):
        """apply_lora_to_musicgen should apply LoRA to decoder."""
        from mlx_music.training.musicgen.lora_layers import (
            apply_lora_to_musicgen,
            LoRALinear,
        )

        class MockDecoder(nn.Module):
            def __init__(self):
                super().__init__()
                self.q_proj = nn.Linear(64, 64)
                self.k_proj = nn.Linear(64, 64)
                self.v_proj = nn.Linear(64, 64)
                self.out_proj = nn.Linear(64, 64)

        class MockMusicGen(nn.Module):
            def __init__(self):
                super().__init__()
                self.decoder = MockDecoder()

        model = MockMusicGen()
        count = apply_lora_to_musicgen(model, rank=32, alpha=32.0)

        assert count == 4
        assert isinstance(model.decoder.q_proj, LoRALinear)
        assert isinstance(model.decoder.k_proj, LoRALinear)

    def test_apply_lora_decoder_only(self):
        """finetune_decoder_only=True should skip text encoder."""
        from mlx_music.training.musicgen.lora_layers import (
            apply_lora_to_musicgen,
            LoRALinear,
        )

        class MockEncoder(nn.Module):
            def __init__(self):
                super().__init__()
                self.q_proj = nn.Linear(64, 64)

        class MockDecoder(nn.Module):
            def __init__(self):
                super().__init__()
                self.q_proj = nn.Linear(64, 64)

        class MockMusicGen(nn.Module):
            def __init__(self):
                super().__init__()
                self.text_encoder = MockEncoder()
                self.decoder = MockDecoder()

        model = MockMusicGen()
        count = apply_lora_to_musicgen(model, rank=32, finetune_decoder_only=True)

        # Only decoder should have LoRA
        assert isinstance(model.decoder.q_proj, LoRALinear)
        assert isinstance(model.text_encoder.q_proj, nn.Linear)  # Not LoRALinear

    def test_apply_lora_with_encoder(self):
        """finetune_decoder_only=False should also apply to text encoder."""
        from mlx_music.training.musicgen.lora_layers import (
            apply_lora_to_musicgen,
            LoRALinear,
        )

        class MockEncoder(nn.Module):
            def __init__(self):
                super().__init__()
                self.q_proj = nn.Linear(64, 64)

        class MockDecoder(nn.Module):
            def __init__(self):
                super().__init__()
                self.q_proj = nn.Linear(64, 64)

        class MockMusicGen(nn.Module):
            def __init__(self):
                super().__init__()
                self.text_encoder = MockEncoder()
                self.decoder = MockDecoder()

        model = MockMusicGen()
        count = apply_lora_to_musicgen(model, rank=32, finetune_decoder_only=False)

        # Both should have LoRA
        assert isinstance(model.decoder.q_proj, LoRALinear)
        assert isinstance(model.text_encoder.q_proj, LoRALinear)

    def test_apply_lora_custom_targets(self):
        """apply_lora_to_musicgen should respect custom targets."""
        from mlx_music.training.musicgen.lora_layers import (
            apply_lora_to_musicgen,
            LoRALinear,
        )

        class MockDecoder(nn.Module):
            def __init__(self):
                super().__init__()
                self.q_proj = nn.Linear(64, 64)
                self.k_proj = nn.Linear(64, 64)
                self.other_proj = nn.Linear(64, 64)

        class MockMusicGen(nn.Module):
            def __init__(self):
                super().__init__()
                self.decoder = MockDecoder()

        model = MockMusicGen()
        count = apply_lora_to_musicgen(
            model, rank=32, target_modules=["q_proj"]  # Only q_proj
        )

        assert isinstance(model.decoder.q_proj, LoRALinear)
        assert isinstance(model.decoder.k_proj, nn.Linear)  # Not LoRA
        assert isinstance(model.decoder.other_proj, nn.Linear)  # Not LoRA

    def test_apply_lora_fallback_to_whole_model(self):
        """apply_lora_to_musicgen should fall back to whole model if no decoder."""
        from mlx_music.training.musicgen.lora_layers import (
            apply_lora_to_musicgen,
            LoRALinear,
        )

        class FlatModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.q_proj = nn.Linear(64, 64)
                self.k_proj = nn.Linear(64, 64)

        model = FlatModel()
        count = apply_lora_to_musicgen(model, rank=32)

        # Should find targets in the model itself
        assert count >= 2
        assert isinstance(model.q_proj, LoRALinear)

    def test_musicgen_lora_exports(self):
        """musicgen lora_layers should export common utilities."""
        from mlx_music.training.musicgen.lora_layers import (
            LoRALinear,
            apply_lora_to_musicgen,
            get_lora_parameters,
            merge_lora_weights,
            save_lora_weights,
            load_lora_weights,
            MUSICGEN_LORA_TARGETS,
        )

        # All should be importable
        assert LoRALinear is not None
        assert apply_lora_to_musicgen is not None
        assert get_lora_parameters is not None
