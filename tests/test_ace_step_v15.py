"""Unit tests for ACE-Step v1.5 MLX implementation."""

import os
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import numpy as np
import pytest


# ─────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────

class TestAceStepV15Config:
    """Test v1.5 configuration."""

    def test_config_defaults(self):
        from mlx_music.models.ace_step_v15.config import AceStepV15Config

        config = AceStepV15Config()
        assert config.hidden_size == 2048
        assert config.num_attention_heads == 16
        assert config.num_key_value_heads == 8
        assert config.head_dim == 128
        assert config.num_hidden_layers == 24
        assert config.intermediate_size == 6144
        assert config.sliding_window == 128

    def test_config_gqa_groups(self):
        from mlx_music.models.ace_step_v15.config import AceStepV15Config

        config = AceStepV15Config()
        assert config.num_kv_groups == 2  # 16 / 8

    def test_config_fsq_codebook_size(self):
        from mlx_music.models.ace_step_v15.config import AceStepV15Config

        config = AceStepV15Config()
        # 8 * 8 * 8 * 5 * 5 * 5 = 64000
        assert config.fsq_codebook_size == 64000

    def test_config_latent_rates(self):
        from mlx_music.models.ace_step_v15.config import AceStepV15Config

        config = AceStepV15Config()
        assert config.latent_hz == 25.0
        assert config.audio_code_hz == 5.0  # 25 / pool_window_size=5

    def test_config_from_dict(self):
        from mlx_music.models.ace_step_v15.config import AceStepV15Config

        d = {"hidden_size": 1024, "num_attention_heads": 8, "unknown_key": 42}
        config = AceStepV15Config.from_dict(d)
        assert config.hidden_size == 1024
        assert config.num_attention_heads == 8

    def test_config_layer_types_auto(self):
        from mlx_music.models.ace_step_v15.config import AceStepV15Config

        config = AceStepV15Config(num_hidden_layers=4)
        assert len(config.layer_types) == 4
        # Odd layers (1-indexed): sliding, even: full
        assert config.layer_types[0] == "sliding_attention"  # layer 1 (odd)
        assert config.layer_types[1] == "full_attention"      # layer 2 (even)
        assert config.layer_types[2] == "sliding_attention"  # layer 3
        assert config.layer_types[3] == "full_attention"      # layer 4

    def test_variant_defaults(self):
        from mlx_music.models.ace_step_v15.config import VARIANT_DEFAULTS

        assert "turbo" in VARIANT_DEFAULTS
        assert "base" in VARIANT_DEFAULTS
        turbo = VARIANT_DEFAULTS["turbo"]
        assert turbo["default_inference_steps"] == 8
        assert turbo["supports_cfg"] is False

        base = VARIANT_DEFAULTS["base"]
        assert base["default_inference_steps"] == 50
        assert base["supports_cfg"] is True

    def test_config_to_dict_roundtrip(self):
        from mlx_music.models.ace_step_v15.config import AceStepV15Config

        config = AceStepV15Config(hidden_size=512, num_attention_heads=4)
        d = config.to_dict()
        config2 = AceStepV15Config.from_dict(d)
        assert config2.hidden_size == 512
        assert config2.num_attention_heads == 4


# ─────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────

class TestConstants:
    """Test v1.5 constants."""

    def test_valid_languages(self):
        from mlx_music.models.ace_step_v15.constants import VALID_LANGUAGES

        assert "en" in VALID_LANGUAGES
        assert "zh" in VALID_LANGUAGES
        assert "ja" in VALID_LANGUAGES
        assert "unknown" in VALID_LANGUAGES
        assert len(VALID_LANGUAGES) >= 45

    def test_valid_keyscales(self):
        from mlx_music.models.ace_step_v15.constants import VALID_KEYSCALES

        assert "C major" in VALID_KEYSCALES
        assert "A minor" in VALID_KEYSCALES
        assert "C# minor" in VALID_KEYSCALES
        assert len(VALID_KEYSCALES) == 70  # 7 * 5 * 2

    def test_task_types(self):
        from mlx_music.models.ace_step_v15.constants import (
            TASK_TYPES,
            TASK_TYPES_TURBO,
            TASK_TYPES_BASE,
        )

        assert "text2music" in TASK_TYPES
        assert "text2music" in TASK_TYPES_TURBO
        assert "extract" in TASK_TYPES_BASE
        assert "extract" not in TASK_TYPES_TURBO

    def test_metadata_ranges(self):
        from mlx_music.models.ace_step_v15.constants import (
            BPM_MIN, BPM_MAX, DURATION_MIN, DURATION_MAX, VALID_TIME_SIGNATURES,
        )

        assert BPM_MIN == 30
        assert BPM_MAX == 300
        assert DURATION_MIN == 10
        assert DURATION_MAX == 600
        assert 4 in VALID_TIME_SIGNATURES


# ─────────────────────────────────────────────────────────
# Attention
# ─────────────────────────────────────────────────────────

class TestRMSNorm:
    """Test RMSNorm layer."""

    def test_output_shape(self):
        from mlx_music.models.ace_step_v15.attention import RMSNorm

        norm = RMSNorm(dim=128)
        x = mx.random.normal((2, 10, 128))
        out = norm(x)
        assert out.shape == (2, 10, 128)

    def test_normalization(self):
        from mlx_music.models.ace_step_v15.attention import RMSNorm

        norm = RMSNorm(dim=64)
        x = mx.random.normal((1, 5, 64)) * 10.0
        out = norm(x)
        mx.eval(out)
        # RMS should be close to 1 (before weight scaling)
        rms = float(mx.sqrt(mx.mean(out * out, axis=-1)).mean())
        assert 0.5 < rms < 2.0


class TestRotaryEmbedding:
    """Test RoPE implementation."""

    def test_cache_shape(self):
        from mlx_music.models.ace_step_v15.attention import RotaryEmbedding

        rope = RotaryEmbedding(dim=128, max_position_embeddings=1024)
        cos, sin = rope(seq_len=64)
        assert cos.shape == (64, 128)
        assert sin.shape == (64, 128)

    def test_cache_extension(self):
        from mlx_music.models.ace_step_v15.attention import RotaryEmbedding

        rope = RotaryEmbedding(dim=128, max_position_embeddings=64)
        cos, sin = rope(seq_len=128)  # Exceeds initial max
        assert cos.shape == (128, 128)


class TestApplyRotaryPosEmb:
    """Test RoPE application."""

    def test_shape_preserved(self):
        from mlx_music.models.ace_step_v15.attention import (
            RotaryEmbedding,
            apply_rotary_pos_emb,
        )

        rope = RotaryEmbedding(dim=128)
        q = mx.random.normal((1, 16, 32, 128))
        k = mx.random.normal((1, 8, 32, 128))
        cos, sin = rope(seq_len=32)
        q_out, k_out = apply_rotary_pos_emb(q, k, cos, sin)
        assert q_out.shape == q.shape
        assert k_out.shape == k.shape


class TestSlidingMask:
    """Test sliding window mask creation."""

    def test_mask_shape(self):
        from mlx_music.models.ace_step_v15.attention import create_sliding_mask

        mask = create_sliding_mask(seq_len=32, sliding_window=8, is_sliding=True)
        # Shape is (1, 1, seq, seq) for broadcasting with attention
        assert mask.shape == (1, 1, 32, 32)

    def test_mask_values(self):
        from mlx_music.models.ace_step_v15.attention import create_sliding_mask

        mask = create_sliding_mask(seq_len=16, sliding_window=4, is_sliding=True)
        mx.eval(mask)
        mask_np = np.array(mask[0, 0])  # Extract (seq, seq) from (1, 1, seq, seq)
        # Diagonal should be 0 (attended)
        for i in range(16):
            assert mask_np[i, i] == 0.0
        # Far off-diagonal should be -inf
        assert mask_np[0, 15] == float("-inf") or mask_np[0, 15] < -1e6

    def test_full_attention_mask_returns_none(self):
        from mlx_music.models.ace_step_v15.attention import create_sliding_mask

        mask = create_sliding_mask(seq_len=16, sliding_window=None, is_sliding=False)
        # Full attention with no sliding window returns None (no mask needed)
        assert mask is None


class TestAttention:
    """Test GQA attention module."""

    def test_attention_creation(self):
        from mlx_music.models.ace_step_v15.attention import AceStepV15Attention

        attn = AceStepV15Attention(
            hidden_size=2048,
            num_attention_heads=16,
            num_key_value_heads=8,
            head_dim=128,
            layer_idx=0,
        )
        assert attn.num_heads == 16
        assert attn.num_kv_heads == 8

    def test_attention_forward_shape(self):
        from mlx_music.models.ace_step_v15.attention import (
            AceStepV15Attention,
            RotaryEmbedding,
        )

        attn = AceStepV15Attention(
            hidden_size=2048,
            num_attention_heads=16,
            num_key_value_heads=8,
            head_dim=128,
            layer_idx=0,
        )
        rope = RotaryEmbedding(dim=128)
        cos, sin = rope(seq_len=32)
        x = mx.random.normal((1, 32, 2048))
        out, kv_cache = attn(
            hidden_states=x,
            position_embeddings=(cos, sin),
        )
        assert out.shape == (1, 32, 2048)


class TestQwen3MLP:
    """Test SiLU gated MLP."""

    def test_mlp_forward_shape(self):
        from mlx_music.models.ace_step_v15.attention import Qwen3MLP

        mlp = Qwen3MLP(hidden_size=2048, intermediate_size=6144)
        x = mx.random.normal((1, 32, 2048))
        out = mlp(x)
        assert out.shape == (1, 32, 2048)


# ─────────────────────────────────────────────────────────
# DiT layers
# ─────────────────────────────────────────────────────────

class TestTimestepEmbedding:
    """Test timestep embedding."""

    def test_output_shapes(self):
        from mlx_music.models.ace_step_v15.dit import TimestepEmbedding

        te = TimestepEmbedding(in_channels=256, time_embed_dim=2048)
        t = mx.array([0.5, 0.8])
        temb, proj = te(t)
        assert temb.shape == (2, 2048)
        assert proj.shape == (2, 6, 2048)

    def test_different_timesteps_give_different_embeddings(self):
        from mlx_music.models.ace_step_v15.dit import TimestepEmbedding

        te = TimestepEmbedding(in_channels=256, time_embed_dim=512)
        t = mx.array([0.1, 0.9])
        temb, _ = te(t)
        mx.eval(temb)
        # Different timesteps should give different embeddings
        diff = float(mx.sum(mx.abs(temb[0] - temb[1])))
        assert diff > 0.1


class TestAceStepEncoderLayer:
    """Test bidirectional encoder layer."""

    def test_encoder_layer_forward(self):
        from mlx_music.models.ace_step_v15.dit import AceStepEncoderLayer
        from mlx_music.models.ace_step_v15.attention import RotaryEmbedding

        layer = AceStepEncoderLayer(
            hidden_size=2048,
            num_attention_heads=16,
            num_key_value_heads=8,
            head_dim=128,
            intermediate_size=6144,
        )
        rope = RotaryEmbedding(dim=128)
        cos, sin = rope(seq_len=32)
        x = mx.random.normal((1, 32, 2048))
        out = layer(x, position_embeddings=(cos, sin))
        assert out.shape == (1, 32, 2048)


class TestAceStepDiTLayer:
    """Test DiT layer with AdaLN."""

    def test_dit_layer_forward(self):
        from mlx_music.models.ace_step_v15.dit import AceStepDiTLayer
        from mlx_music.models.ace_step_v15.attention import RotaryEmbedding

        layer = AceStepDiTLayer(
            hidden_size=2048,
            num_attention_heads=16,
            num_key_value_heads=8,
            head_dim=128,
            intermediate_size=6144,
            layer_idx=0,
        )
        rope = RotaryEmbedding(dim=128)
        cos, sin = rope(seq_len=32)
        x = mx.random.normal((1, 32, 2048))
        # temb is the 6-way AdaLN projection: (batch, 6, hidden_size)
        temb = mx.random.normal((1, 6, 2048))
        encoder_hidden = mx.random.normal((1, 16, 2048))

        out, _ = layer(
            hidden_states=x,
            position_embeddings=(cos, sin),
            temb=temb,
            encoder_hidden_states=encoder_hidden,
        )
        assert out.shape == (1, 32, 2048)


# ─────────────────────────────────────────────────────────
# Constrained decoding
# ─────────────────────────────────────────────────────────

class TestConstrainedDecoding:
    """Test FSM-based constrained decoding components."""

    def test_parse_lm_output_basic(self):
        from mlx_music.models.ace_step_v15.constrained_decoding import parse_lm_output

        raw = "<think>bpm: 120\nkeyscale: C major\nduration: 30</think>some audio codes"
        metadata, audio_codes_str = parse_lm_output(raw)
        assert isinstance(metadata, dict)
        assert metadata["bpm"] == "120"
        assert metadata["keyscale"] == "C major"
        assert isinstance(audio_codes_str, str)

    def test_extract_audio_code_indices(self):
        from mlx_music.models.ace_step_v15.constrained_decoding import extract_audio_code_indices

        # Uses <|audio_code_N|> format
        raw_codes = "<|audio_code_100|><|audio_code_200|><|audio_code_300|><|audio_code_400|>"
        indices = extract_audio_code_indices(raw_codes)
        assert isinstance(indices, list)
        assert len(indices) == 4
        assert indices == [100, 200, 300, 400]

    def test_extract_audio_code_empty(self):
        from mlx_music.models.ace_step_v15.constrained_decoding import extract_audio_code_indices

        indices = extract_audio_code_indices("no audio codes here")
        assert indices == []


# ─────────────────────────────────────────────────────────
# Model API
# ─────────────────────────────────────────────────────────

class _DummyV15GenerationModel:
    """Minimal stub that records inputs and returns zero latents."""

    def __init__(self):
        self.last_kwargs = None

    def generate_audio(self, **kwargs):
        self.last_kwargs = kwargs
        batch, frames, _ = kwargs["src_latents"].shape
        return {
            "target_latents": mx.zeros((batch, frames, 64), dtype=mx.float32),
            "time_costs": {"total_time_cost": 0.0},
        }


class _DummyV15VAE:
    """Minimal VAE stub for model API tests."""

    def decode(self, latents):
        batch, frames, _ = latents.shape
        # 1920 = VAE hop size (48k / 25Hz)
        return mx.zeros((batch, frames * 1920, 2), dtype=mx.float32)


class TestV15ModelAPI:
    """Test high-level ACEStepV15 API behavior."""

    def _create_model(self):
        from mlx_music.models.ace_step_v15.config import AceStepV15Config
        from mlx_music.models.ace_step_v15.model import ACEStepV15

        dummy_model = _DummyV15GenerationModel()
        api = ACEStepV15(
            model=dummy_model,
            vae=_DummyV15VAE(),
            silence_latent=mx.zeros((1, 25 * 10, 64), dtype=mx.float32),
            config=AceStepV15Config(),
        )
        return api, dummy_model

    def test_build_timestep_schedule(self):
        from mlx_music.models.ace_step_v15.model import ACEStepV15

        schedule = ACEStepV15._build_timestep_schedule(steps=8, shift=3.0)
        assert len(schedule) == 8
        assert schedule[0] == pytest.approx(1.0)
        assert schedule[1] == pytest.approx(0.9545454545, rel=1e-6)
        assert schedule[-1] == pytest.approx(0.3, rel=1e-6)
        assert all(schedule[i] > schedule[i + 1] for i in range(len(schedule) - 1))

    def test_generate_uses_auto_conditioning_and_steps(self, monkeypatch):
        api, dummy_model = self._create_model()

        text_hidden = mx.ones((1, 3, api.config.text_hidden_dim), dtype=mx.bfloat16)
        text_mask = mx.ones((1, 3), dtype=mx.bfloat16)
        lyric_hidden = mx.ones((1, 5, api.config.text_hidden_dim), dtype=mx.bfloat16)
        lyric_mask = mx.ones((1, 5), dtype=mx.bfloat16)

        monkeypatch.setattr(
            api,
            "_build_text_conditioning_from_strings",
            lambda **_: (text_hidden, text_mask, lyric_hidden, lyric_mask),
        )

        result = api.generate(
            prompt="dark industrial synthwave",
            lyrics="",
            duration=1.0,
            steps=5,
            shift=2.0,
            seed=42,
        )

        assert dummy_model.last_kwargs is not None
        assert dummy_model.last_kwargs["text_hidden_states"].shape == (1, 3, api.config.text_hidden_dim)
        assert dummy_model.last_kwargs["lyric_hidden_states"].shape == (1, 5, api.config.text_hidden_dim)
        assert dummy_model.last_kwargs["timesteps"] is not None
        assert len(dummy_model.last_kwargs["timesteps"]) == 5
        assert result["audio"].shape[1] == 2

    def test_generate_falls_back_to_zero_conditioning(self, monkeypatch):
        api, dummy_model = self._create_model()

        monkeypatch.setattr(
            api,
            "_build_text_conditioning_from_strings",
            lambda **_: None,
        )

        api.generate(prompt="test", lyrics="", duration=1.0)
        assert dummy_model.last_kwargs is not None
        text_hidden = dummy_model.last_kwargs["text_hidden_states"].astype(mx.float32)
        mx.eval(text_hidden)
        text_hidden_np = np.array(text_hidden)
        assert text_hidden_np.shape == (1, 1, api.config.text_hidden_dim)
        assert np.allclose(text_hidden_np, 0.0)


# ─────────────────────────────────────────────────────────
# Training: LoRA
# ─────────────────────────────────────────────────────────

class TestLoRAV15:
    """Test v1.5 LoRA configuration."""

    def test_lora_target_constants(self):
        from mlx_music.training.ace_step_v15.lora_layers import (
            V15_ATTENTION_TARGETS,
            V15_MLP_TARGETS,
            V15_ALL_TARGETS,
        )

        assert "q_proj" in V15_ATTENTION_TARGETS
        assert "k_proj" in V15_ATTENTION_TARGETS
        assert "v_proj" in V15_ATTENTION_TARGETS
        assert "o_proj" in V15_ATTENTION_TARGETS
        assert "gate_proj" in V15_MLP_TARGETS
        assert "up_proj" in V15_MLP_TARGETS
        assert "down_proj" in V15_MLP_TARGETS
        assert len(V15_ALL_TARGETS) == 7

    def test_lora_linear_forward(self):
        from mlx_music.training.ace_step.lora_layers import LoRALinear

        lora = LoRALinear(in_features=128, out_features=256, rank=8, alpha=8.0, bias=False)
        x = mx.random.normal((2, 10, 128))
        out = lora(x)
        assert out.shape == (2, 10, 256)

    def test_lora_starts_as_identity(self):
        from mlx_music.training.ace_step.lora_layers import LoRALinear

        lora = LoRALinear(in_features=64, out_features=64, rank=8, alpha=8.0, bias=False)
        x = mx.random.normal((1, 5, 64))

        # Since lora_B is initialized to zeros, LoRA contribution should be 0
        # So output should equal just Wx
        base_out = mx.matmul(x, lora.weight.T)
        lora_out = lora(x)
        mx.eval(base_out, lora_out)

        diff = float(mx.max(mx.abs(base_out - lora_out)))
        assert diff < 1e-5

    def test_lora_merge_weights(self):
        from mlx_music.training.ace_step.lora_layers import LoRALinear

        lora = LoRALinear(in_features=64, out_features=64, rank=8, alpha=8.0, bias=False)
        # Set non-zero lora_B
        lora.lora_B = mx.random.normal((64, 8)) * 0.1
        merged = lora.merge_weights()
        assert merged.shape == (64, 64)
        # Merged should differ from original
        diff = float(mx.sum(mx.abs(merged - lora.weight)))
        assert diff > 0.01


# ─────────────────────────────────────────────────────────
# Training: Loss
# ─────────────────────────────────────────────────────────

class TestFlowMatchingLoss:
    """Test v1.5 flow matching loss."""

    def test_timestep_sampling(self):
        from mlx_music.training.ace_step_v15.loss import ACEStepV15Loss

        t = ACEStepV15Loss.sample_timestep(batch_size=100)
        mx.eval(t)
        assert t.shape == (100,)
        # All values should be in (0, 1) since sigmoid output
        t_np = np.array(t)
        assert np.all(t_np > 0)
        assert np.all(t_np < 1)

    def test_reference_frame_sampling(self):
        from mlx_music.training.ace_step_v15.loss import ACEStepV15Loss

        t = mx.array([0.5, 0.8, 0.3])
        r = ACEStepV15Loss.sample_reference_frame(t, data_proportion=0.5)
        mx.eval(r)
        assert r.shape == (3,)
        r_np = np.array(r)
        t_np = np.array(t)
        # r = t - offset, where offset in [0, 0.5], so r <= t
        assert np.all(r_np <= t_np + 1e-6)

    def test_add_noise(self):
        from mlx_music.training.ace_step_v15.loss import ACEStepV15Loss

        clean = mx.ones((2, 10, 64))
        noise = mx.zeros((2, 10, 64))
        t = mx.array([0.0, 1.0])

        noised = ACEStepV15Loss.add_noise(clean, noise, t)
        mx.eval(noised)
        noised_np = np.array(noised)

        # t=0: x_t = (1-0)*clean + 0*noise = clean = 1
        np.testing.assert_allclose(noised_np[0], 1.0, atol=1e-5)
        # t=1: x_t = (1-1)*clean + 1*noise = noise = 0
        np.testing.assert_allclose(noised_np[1], 0.0, atol=1e-5)

    def test_target_velocity(self):
        from mlx_music.training.ace_step_v15.loss import ACEStepV15Loss

        clean = mx.ones((1, 5, 64))
        noise = mx.zeros((1, 5, 64))
        v = ACEStepV15Loss.compute_target_velocity(clean, noise)
        mx.eval(v)
        # v = noise - clean = 0 - 1 = -1
        np.testing.assert_allclose(np.array(v), -1.0, atol=1e-5)

    def test_compute_loss_mean(self):
        from mlx_music.training.ace_step_v15.loss import ACEStepV15Loss

        predicted = mx.ones((2, 10, 64))
        target = mx.zeros((2, 10, 64))
        loss = ACEStepV15Loss.compute_loss(predicted, target, reduction="mean")
        mx.eval(loss)
        # MSE of (1 - 0)^2 = 1
        assert abs(float(loss) - 1.0) < 1e-5

    def test_compute_loss_with_mask(self):
        from mlx_music.training.ace_step_v15.loss import ACEStepV15Loss

        predicted = mx.ones((1, 10, 64))
        target = mx.zeros((1, 10, 64))
        # Mask out half the sequence
        mask = mx.concatenate([
            mx.ones((1, 5, 64)),
            mx.zeros((1, 5, 64)),
        ], axis=1)

        loss = ACEStepV15Loss.compute_loss(predicted, target, mask=mask, reduction="mean")
        mx.eval(loss)
        # Only masked region contributes: MSE still = 1
        assert abs(float(loss) - 1.0) < 1e-5


# ─────────────────────────────────────────────────────────
# Training: Config
# ─────────────────────────────────────────────────────────

class TestTrainingConfig:
    """Test training configuration."""

    def test_training_config_defaults(self):
        from mlx_music.training.ace_step_v15.trainer import TrainingConfigV15

        config = TrainingConfigV15()
        assert config.learning_rate == 1e-4
        assert config.lora_rank == 64
        assert config.timestep_mu == -0.4
        assert config.timestep_sigma == 1.0
        assert config.data_proportion == 0.5
        assert config.use_ema is True
        assert config.use_lora is True

    def test_training_state_roundtrip(self):
        from mlx_music.training.ace_step_v15.trainer import TrainingStateV15

        state = TrainingStateV15(global_step=100, epoch=2, best_loss=0.5)
        d = state.state_dict()
        state2 = TrainingStateV15()
        state2.load_state_dict(d)
        assert state2.global_step == 100
        assert state2.epoch == 2
        assert state2.best_loss == 0.5


# ─────────────────────────────────────────────────────────
# Imports
# ─────────────────────────────────────────────────────────

class TestV15Imports:
    """Test v1.5 module imports."""

    def test_all_exports_accessible(self):
        from mlx_music.models.ace_step_v15 import __all__
        import mlx_music.models.ace_step_v15 as v15

        for name in __all__:
            assert hasattr(v15, name), f"Missing export: {name}"

    def test_main_classes_importable(self):
        from mlx_music.models.ace_step_v15 import (
            ACEStepV15,
            AceStepV15Config,
            AceStepV15Attention,
            AceStepDiTLayer,
            AceStepEncoderLayer,
            AceStepConditionGenerationModel,
            AceStepDiTModel,
            AutoencoderOobleck,
            TimestepEmbedding,
            RMSNorm,
            RotaryEmbedding,
            Qwen3MLP,
            LMPlanner,
        )

        assert ACEStepV15 is not None
        assert AceStepV15Config is not None
        assert AceStepV15Attention is not None

    def test_training_imports(self):
        from mlx_music.training.ace_step_v15 import (
            ACEStepV15Trainer,
            TrainingConfigV15,
            ACEStepV15Loss,
            apply_lora_to_v15_model,
        )

        assert ACEStepV15Trainer is not None
        assert TrainingConfigV15 is not None
        assert ACEStepV15Loss is not None

    def test_v15_in_lazy_module(self):
        """Test that ACEStepV15 is accessible from top-level."""
        import mlx_music

        attrs = dir(mlx_music)
        assert "ACEStepV15" in attrs
