"""
Tests for Stable Audio LoRA layers.

Tests:
- apply_lora_to_stable_audio: Target modules, GQA-aware config
- apply_gqa_aware_lora: Different ranks for Q vs KV
- STABLE_AUDIO_LORA_TARGETS: Default target layers
"""

import pytest
import mlx.core as mx
import mlx.nn as nn


class TestApplyLoRAToStableAudio:
    """Tests for apply_lora_to_stable_audio function."""

    def test_stable_audio_lora_targets_defined(self):
        """STABLE_AUDIO_LORA_TARGETS should define default targets."""
        from mlx_music.training.stable_audio.lora_layers import STABLE_AUDIO_LORA_TARGETS

        assert "q_proj" in STABLE_AUDIO_LORA_TARGETS
        assert "k_proj" in STABLE_AUDIO_LORA_TARGETS
        assert "v_proj" in STABLE_AUDIO_LORA_TARGETS
        assert "out_proj" in STABLE_AUDIO_LORA_TARGETS

    def test_apply_lora_to_dit(self):
        """apply_lora_to_stable_audio should apply LoRA to DiT model."""
        from mlx_music.training.stable_audio.lora_layers import (
            apply_lora_to_stable_audio,
            LoRALinear,
        )

        class MockDiT(nn.Module):
            def __init__(self):
                super().__init__()
                self.q_proj = nn.Linear(64, 64)
                self.k_proj = nn.Linear(64, 64)
                self.v_proj = nn.Linear(64, 64)
                self.out_proj = nn.Linear(64, 64)

        model = MockDiT()
        count = apply_lora_to_stable_audio(model, q_rank=64, kv_rank=32)

        assert count == 4
        assert isinstance(model.q_proj, LoRALinear)
        assert isinstance(model.k_proj, LoRALinear)
        assert isinstance(model.v_proj, LoRALinear)
        assert isinstance(model.out_proj, LoRALinear)

    def test_apply_lora_uses_max_rank(self):
        """apply_lora_to_stable_audio should use max(q_rank, kv_rank) for all."""
        from mlx_music.training.stable_audio.lora_layers import (
            apply_lora_to_stable_audio,
            LoRALinear,
        )

        class MockDiT(nn.Module):
            def __init__(self):
                super().__init__()
                self.q_proj = nn.Linear(64, 64)
                self.k_proj = nn.Linear(64, 64)

        model = MockDiT()
        apply_lora_to_stable_audio(model, q_rank=64, kv_rank=32)

        # All should use max rank (64)
        assert model.q_proj.rank == 64
        assert model.k_proj.rank == 64

    def test_apply_lora_custom_alpha(self):
        """apply_lora_to_stable_audio should use custom alpha when provided."""
        from mlx_music.training.stable_audio.lora_layers import (
            apply_lora_to_stable_audio,
        )

        class MockDiT(nn.Module):
            def __init__(self):
                super().__init__()
                self.q_proj = nn.Linear(64, 64)

        model = MockDiT()
        apply_lora_to_stable_audio(model, q_rank=64, kv_rank=32, alpha=128.0)

        assert model.q_proj.alpha == 128.0

    def test_apply_lora_default_alpha_equals_rank(self):
        """apply_lora_to_stable_audio default alpha should equal rank."""
        from mlx_music.training.stable_audio.lora_layers import (
            apply_lora_to_stable_audio,
        )

        class MockDiT(nn.Module):
            def __init__(self):
                super().__init__()
                self.q_proj = nn.Linear(64, 64)

        model = MockDiT()
        apply_lora_to_stable_audio(model, q_rank=64, kv_rank=32)

        # Default alpha = max(q_rank, kv_rank)
        assert model.q_proj.alpha == 64.0


class TestApplyGQAAwareLoRA:
    """Tests for apply_gqa_aware_lora function."""

    def test_gqa_aware_different_ranks(self):
        """apply_gqa_aware_lora should use different ranks for Q vs KV."""
        from mlx_music.training.stable_audio.lora_layers import (
            apply_gqa_aware_lora,
            LoRALinear,
        )

        class MockGQA(nn.Module):
            def __init__(self):
                super().__init__()
                self.q_proj = nn.Linear(64, 64)
                self.k_proj = nn.Linear(64, 32)  # Fewer KV heads
                self.v_proj = nn.Linear(64, 32)
                self.out_proj = nn.Linear(64, 64)

        model = MockGQA()
        count = apply_gqa_aware_lora(model, q_rank=64, kv_rank=32)

        # Should replace all 4
        assert count == 4

        # Q and out should have q_rank
        assert model.q_proj.rank == 64
        assert model.out_proj.rank == 64

        # K and V should have kv_rank
        assert model.k_proj.rank == 32
        assert model.v_proj.rank == 32

    def test_gqa_aware_alpha_equals_rank(self):
        """apply_gqa_aware_lora should set alpha = rank for each layer."""
        from mlx_music.training.stable_audio.lora_layers import apply_gqa_aware_lora

        class MockGQA(nn.Module):
            def __init__(self):
                super().__init__()
                self.q_proj = nn.Linear(64, 64)
                self.k_proj = nn.Linear(64, 32)

        model = MockGQA()
        apply_gqa_aware_lora(model, q_rank=64, kv_rank=32)

        # Alpha should match rank for each
        assert model.q_proj.alpha == 64.0
        assert model.k_proj.alpha == 32.0

    def test_gqa_aware_nested_modules(self):
        """apply_gqa_aware_lora should handle nested modules."""
        from mlx_music.training.stable_audio.lora_layers import (
            apply_gqa_aware_lora,
            LoRALinear,
        )

        class Attention(nn.Module):
            def __init__(self):
                super().__init__()
                self.q_proj = nn.Linear(64, 64)
                self.k_proj = nn.Linear(64, 32)

        class DiTBlock(nn.Module):
            def __init__(self):
                super().__init__()
                self.self_attn = Attention()
                self.cross_attn = Attention()

        model = DiTBlock()
        count = apply_gqa_aware_lora(model, q_rank=64, kv_rank=32)

        # Should find all 4 projections (2 per attention, 2 attentions)
        assert count == 4
        assert isinstance(model.self_attn.q_proj, LoRALinear)
        assert isinstance(model.cross_attn.k_proj, LoRALinear)

    def test_gqa_aware_skips_non_targets(self):
        """apply_gqa_aware_lora should skip non-target layers."""
        from mlx_music.training.stable_audio.lora_layers import apply_gqa_aware_lora

        class MockDiT(nn.Module):
            def __init__(self):
                super().__init__()
                self.q_proj = nn.Linear(64, 64)
                self.other_linear = nn.Linear(64, 64)
                self.mlp = nn.Linear(64, 256)

        model = MockDiT()
        count = apply_gqa_aware_lora(model, q_rank=64, kv_rank=32)

        # Only q_proj should be replaced
        assert count == 1
        assert isinstance(model.other_linear, nn.Linear)  # Not LoRA


class TestStableAudioLoRAExports:
    """Tests for module exports."""

    def test_exports(self):
        """stable_audio lora_layers should export common utilities."""
        from mlx_music.training.stable_audio.lora_layers import (
            LoRALinear,
            apply_lora_to_stable_audio,
            apply_gqa_aware_lora,
            get_lora_parameters,
            merge_lora_weights,
            save_lora_weights,
            load_lora_weights,
            STABLE_AUDIO_LORA_TARGETS,
        )

        # All should be importable
        assert LoRALinear is not None
        assert apply_lora_to_stable_audio is not None
        assert apply_gqa_aware_lora is not None
