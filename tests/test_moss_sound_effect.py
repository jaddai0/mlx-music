"""Unit tests for MOSS-SoundEffect MLX port."""

from __future__ import annotations

import math
import os
from pathlib import Path

import mlx.core as mx
import pytest

# MLX lazy computation materializer (NOT Python's built-in)
_materialize = mx.eval
REAL_MOSS_MODEL_PATH = os.environ.get("MOSS_SOUND_EFFECT_MLX_PATH")
REAL_MOSS_TOKENIZER_PATH = os.environ.get("MOSS_AUDIO_TOKENIZER_MLX_PATH")


# =============================================================================
# Config Tests
# =============================================================================


class TestMossSoundEffectConfig:
    def test_default_config(self):
        from mlx_music.models.moss_sound_effect.config import MossSoundEffectConfig

        cfg = MossSoundEffectConfig()
        assert cfg.hidden_size == 4096
        assert cfg.num_hidden_layers == 36
        assert cfg.n_vq == 16
        assert cfg.audio_vocab_size == 1024
        assert cfg.audio_pad_code == 1024
        assert cfg.sampling_rate == 24000
        assert cfg.vocab_size == 155648

    def test_from_hf_config(self):
        from mlx_music.models.moss_sound_effect.config import MossSoundEffectConfig

        hf = {
            "language_config": {
                "hidden_size": 2048,
                "num_hidden_layers": 12,
                "num_attention_heads": 16,
                "num_key_value_heads": 4,
                "intermediate_size": 6144,
                "vocab_size": 32000,
            },
            "n_vq": 8,
            "audio_vocab_size": 512,
        }
        cfg = MossSoundEffectConfig.from_hf_config(hf)
        assert cfg.hidden_size == 2048
        assert cfg.n_vq == 8
        assert cfg.audio_vocab_size == 512

    def test_roundtrip_dict(self):
        from mlx_music.models.moss_sound_effect.config import MossSoundEffectConfig

        cfg = MossSoundEffectConfig(n_vq=8, audio_vocab_size=512)
        d = cfg.to_dict()
        cfg2 = MossSoundEffectConfig.from_dict(d)
        assert cfg2.n_vq == 8
        assert cfg2.audio_vocab_size == 512


# =============================================================================
# Delay Pattern Tests
# =============================================================================


class TestDelayPattern:
    def test_basic_delay(self):
        from mlx_music.models.moss_sound_effect.processor import (
            apply_delay_pattern,
            apply_de_delay_pattern,
        )

        T, n_vq = 3, 4
        codes = mx.arange(T * n_vq).reshape(T, n_vq)
        pad = 9999

        delayed = apply_delay_pattern(codes, n_vq, pad)
        assert delayed.shape == (T + n_vq - 1, n_vq)

        # Check padding pattern
        _materialize(delayed)
        # First row should have code in col 0, pad in rest
        assert int(delayed[0, 0].item()) == int(codes[0, 0].item())
        assert int(delayed[0, 1].item()) == pad
        assert int(delayed[0, 2].item()) == pad

    def test_roundtrip(self):
        from mlx_music.models.moss_sound_effect.processor import (
            apply_delay_pattern,
            apply_de_delay_pattern,
        )

        T, n_vq = 5, 4
        codes = mx.arange(T * n_vq).reshape(T, n_vq)
        pad = 9999

        delayed = apply_delay_pattern(codes, n_vq, pad)
        recovered = apply_de_delay_pattern(delayed, n_vq, pad)

        _materialize(recovered)
        assert recovered.shape == (T, n_vq)
        # Verify values match
        for t in range(T):
            for v in range(n_vq):
                assert int(recovered[t, v].item()) == int(codes[t, v].item()), \
                    f"Mismatch at ({t}, {v}): got {recovered[t, v].item()}, expected {codes[t, v].item()}"

    def test_empty(self):
        from mlx_music.models.moss_sound_effect.processor import apply_de_delay_pattern

        delayed = mx.zeros((2, 4))
        result = apply_de_delay_pattern(delayed, 4, 0)
        _materialize(result)
        # T' - n_vq + 1 = 2 - 4 + 1 = -1, should return empty
        assert result.shape[0] == 0


# =============================================================================
# Sampling Tests
# =============================================================================


class TestSampling:
    def test_top_k_filter(self):
        from mlx_music.models.moss_sound_effect.sampling import top_k_filter

        logits = mx.array([[1.0, 2.0, 3.0, 4.0, 5.0]])
        filtered = top_k_filter(logits, k=3)
        _materialize(filtered)
        # Only top 3 should remain
        assert float(filtered[0, 0].item()) == float("-inf")
        assert float(filtered[0, 1].item()) == float("-inf")
        assert float(filtered[0, 2].item()) == 3.0
        assert float(filtered[0, 3].item()) == 4.0
        assert float(filtered[0, 4].item()) == 5.0

    def test_top_p_filter(self):
        from mlx_music.models.moss_sound_effect.sampling import top_p_filter

        # Create logits where softmax gives roughly [0.01, 0.04, 0.15, 0.3, 0.5]
        logits = mx.array([[0.0, 1.0, 2.5, 3.5, 4.5]])
        filtered = top_p_filter(logits, p=0.8)
        _materialize(filtered)
        # Most of the probability mass is in the last 2-3 tokens
        # Low-probability tokens should be -inf
        assert float(filtered[0, 0].item()) == float("-inf")

    def test_sample_greedy(self):
        from mlx_music.models.moss_sound_effect.sampling import sample_token

        logits = mx.array([[1.0, 5.0, 2.0, 3.0]])
        tok = sample_token(logits, do_sample=False)
        _materialize(tok)
        assert int(tok[0].item()) == 1  # argmax

    def test_sample_stochastic(self):
        from mlx_music.models.moss_sound_effect.sampling import sample_token

        logits = mx.array([[0.0, 0.0, 100.0, 0.0]])  # Very peaked
        tok = sample_token(logits, temperature=1.0, do_sample=True)
        _materialize(tok)
        assert int(tok[0].item()) == 2  # Should almost always pick index 2

    def test_repetition_penalty_2d(self):
        from mlx_music.models.moss_sound_effect.sampling import repetition_penalty

        logits = mx.array([[1.0, 2.0, 3.0, 4.0]])
        prev = mx.array([[1, 3]])  # Penalize tokens 1 and 3
        result = repetition_penalty(logits, prev, penalty=2.0)
        _materialize(result)
        # Token 1: logit 2.0 > 0, so 2.0 / 2.0 = 1.0
        assert abs(float(result[0, 1].item()) - 1.0) < 1e-5
        # Token 3: logit 4.0 > 0, so 4.0 / 2.0 = 2.0
        assert abs(float(result[0, 3].item()) - 2.0) < 1e-5


# =============================================================================
# RoPE Tests
# =============================================================================


class TestRoPE:
    def test_rope_shape(self):
        from mlx_music.codecs.moss_audio_tokenizer.transformer import apply_rope

        B, H, T, D = 2, 4, 8, 64
        q = mx.random.normal((B, H, T, D))
        k = mx.random.normal((B, H, T, D))

        qr, kr = apply_rope(q, k, offset=0)
        _materialize(qr, kr)
        assert qr.shape == (B, H, T, D)
        assert kr.shape == (B, H, T, D)

    def test_rope_offset_invariance(self):
        from mlx_music.codecs.moss_audio_tokenizer.transformer import apply_rope

        B, H, T, D = 1, 2, 4, 32
        q = mx.random.normal((B, H, T, D))
        k = mx.random.normal((B, H, T, D))

        # Applying with offset=0 then with offset=T should give same
        # positions for the second set
        q1, k1 = apply_rope(q, k, offset=0)
        q2, k2 = apply_rope(q[:, :, :1, :], k[:, :, :1, :], offset=0)
        _materialize(q1, q2)

        # First position should match
        diff = mx.abs(q1[:, :, 0, :] - q2[:, :, 0, :])
        _materialize(diff)
        assert float(mx.max(diff).item()) < 1e-5


# =============================================================================
# Patched Pretransform Tests
# =============================================================================


class TestPatchedPretransform:
    def test_encode_decode_roundtrip(self):
        from mlx_music.codecs.moss_audio_tokenizer.patched_pretransform import PatchedPretransform

        B, D, T = 2, 4, 240
        x = mx.random.normal((B, D, T))

        encoder = PatchedPretransform(patch_size=4, is_downsample=True)
        decoder = PatchedPretransform(patch_size=4, is_downsample=False)

        encoded = encoder(x)
        assert encoded.shape == (B, D * 4, T // 4)

        decoded = decoder(encoded)
        _materialize(decoded)
        assert decoded.shape == (B, D, T)

        diff = mx.max(mx.abs(x - decoded))
        _materialize(diff)
        assert float(diff.item()) < 1e-5

    def test_patch_240(self):
        from mlx_music.codecs.moss_audio_tokenizer.patched_pretransform import PatchedPretransform

        B, D = 1, 1
        T = 480  # Must be divisible by 240
        x = mx.random.normal((B, D, T))

        enc = PatchedPretransform(patch_size=240, is_downsample=True)
        encoded = enc(x)
        assert encoded.shape == (B, D * 240, T // 240)

        dec = PatchedPretransform(patch_size=240, is_downsample=False)
        decoded = dec(encoded)
        _materialize(decoded)
        assert decoded.shape == (B, D, T)


# =============================================================================
# LFQ Quantizer Tests
# =============================================================================


class TestQuantizer:
    def test_lfq_decode_shape(self):
        from mlx_music.codecs.moss_audio_tokenizer.quantizer import LFQ

        lfq = LFQ(input_dim=8, codebook_size=32, codebook_dim=8)
        indices = mx.zeros((2, 10), dtype=mx.int32)
        result = lfq.decode_code(indices)
        _materialize(result)
        assert result.shape == (2, 8, 10)

    def test_residual_lfq_decode_shape(self):
        from mlx_music.codecs.moss_audio_tokenizer.quantizer import ResidualLFQ

        rlfq = ResidualLFQ(
            input_dim=16, rvq_dim=8, output_dim=16,
            num_quantizers=4, codebook_size=32, codebook_dim=8,
        )
        codes = mx.zeros((4, 2, 10), dtype=mx.int32)
        result = rlfq.decode_codes(codes)
        _materialize(result)
        assert result.shape == (2, 16, 10)

    def test_wn_conv1d(self):
        from mlx_music.codecs.moss_audio_tokenizer.quantizer import WNConv1d

        wn = WNConv1d(in_channels=8, out_channels=16)
        x = mx.random.normal((2, 8, 10))
        y = wn(x)
        _materialize(y)
        assert y.shape == (2, 16, 10)


# =============================================================================
# Tokenizer Config Tests
# =============================================================================


class TestTokenizerConfig:
    def test_default_config(self):
        from mlx_music.codecs.moss_audio_tokenizer.config import MossAudioTokenizerConfig

        cfg = MossAudioTokenizerConfig()
        assert cfg.sampling_rate == 24000
        assert cfg.downsample_rate == 1920
        assert cfg.frame_rate == 12.5
        assert cfg.num_quantizers == 32
        assert cfg.codebook_size == 1024

    def test_encoder_decoder_symmetry(self):
        from mlx_music.codecs.moss_audio_tokenizer.config import MossAudioTokenizerConfig

        cfg = MossAudioTokenizerConfig()
        # Encoder has 8 stages (4 PatchedPretransform + 4 Transformer)
        assert len(cfg.encoder_kwargs) == 8
        # Decoder has 10 stages (5 Transformer + 5 PatchedPretransform)
        assert len(cfg.decoder_kwargs) == 10


# =============================================================================
# Weight Mapping Tests
# =============================================================================


class TestWeightMapping:
    def test_backbone_key_mapping(self):
        from mlx_music.models.moss_sound_effect.weight_mapping import map_weights

        weights = {
            "language_model.embed_tokens.weight": mx.zeros((100, 64)),
            "language_model.layers.0.self_attn.q_proj.weight": mx.zeros((64, 64)),
            "emb_ext.0.weight": mx.zeros((1025, 64)),
            "lm_heads.0.weight": mx.zeros((100, 64)),
        }
        mapped = map_weights(weights)

        assert "language_model.model.embed_tokens.weight" in mapped
        assert "language_model.model.layers.0.self_attn.q_proj.weight" in mapped
        assert "emb_ext.0.weight" in mapped
        assert "lm_heads.0.weight" in mapped

    def test_should_quantize(self):
        from mlx_music.models.moss_sound_effect.weight_mapping import should_quantize

        # Should quantize transformer layers
        assert should_quantize("language_model.model.layers.0.self_attn.q_proj.weight")
        assert should_quantize("language_model.model.layers.10.mlp.gate_proj.weight")
        assert should_quantize("lm_heads.0.weight")  # Text head (large)

        # Should NOT quantize
        assert not should_quantize("emb_ext.0.weight")
        assert not should_quantize("language_model.model.embed_tokens.weight")
        assert not should_quantize("language_model.model.norm.weight")
        assert not should_quantize("lm_heads.1.weight")  # Audio head (small)
        assert not should_quantize("lm_heads.16.weight")


# =============================================================================
# Transformer Tests
# =============================================================================


class TestTransformer:
    def test_layer_forward(self):
        from mlx_music.codecs.moss_audio_tokenizer.transformer import TransformerLayer

        layer = TransformerLayer(
            d_model=64, num_heads=4, dim_feedforward=128,
            causal=True, use_rope=True, norm="layer_norm",
            layer_scale=0.01,
        )
        x = mx.random.normal((2, 8, 64))
        y = layer(x)
        _materialize(y)
        assert y.shape == (2, 8, 64)

    def test_projected_transformer(self):
        from mlx_music.codecs.moss_audio_tokenizer.transformer import ProjectedTransformer

        pt = ProjectedTransformer(
            input_dimension=32, output_dimension=48,
            d_model=64, num_heads=4, num_layers=2,
            dim_feedforward=128, causal=True,
            positional_embedding="rope",
        )
        x = mx.random.normal((2, 32, 10))  # (B, D, T) conv_layout
        y = pt(x)
        _materialize(y)
        assert y.shape == (2, 48, 10)


# =============================================================================
# find_last_equal Tests
# =============================================================================


class TestFindLastEqual:
    def test_basic(self):
        from mlx_music.models.moss_sound_effect.generation import find_last_equal

        tensor = mx.array([[1, 2, 3, 2, 5], [2, 2, 2, 2, 2]])
        result = find_last_equal(tensor, 2)
        _materialize(result)
        assert int(result[0].item()) == 3  # Last 2 at index 3
        assert int(result[1].item()) == 4  # Last 2 at index 4

    def test_not_found(self):
        from mlx_music.models.moss_sound_effect.generation import find_last_equal

        tensor = mx.array([[1, 3, 5, 7]])
        result = find_last_equal(tensor, 2)
        _materialize(result)
        assert int(result[0].item()) == -1


# =============================================================================
# Build Prompt Tests
# =============================================================================


class TestBuildPrompt:
    def test_user_message_format(self):
        from mlx_music.models.moss_sound_effect.processor import UserMessage

        msg = UserMessage(
            sound_event="thunder rumble",
            tokens=63,
            quality="high",
        )
        content = msg.format_content()
        assert "thunder rumble" in content
        assert "63" in content
        assert "high" in content
        assert "<user_inst>" in content
        assert "</user_inst>" in content


@pytest.mark.skipif(
    not REAL_MOSS_MODEL_PATH or not Path(REAL_MOSS_MODEL_PATH).exists(),
    reason="MOSS_SOUND_EFFECT_MLX_PATH not set or converted model not found",
)
@pytest.mark.skipif(
    not REAL_MOSS_TOKENIZER_PATH or not Path(REAL_MOSS_TOKENIZER_PATH).exists(),
    reason="MOSS_AUDIO_TOKENIZER_MLX_PATH not set or converted tokenizer not found",
)
def test_real_pipeline_smoke():
    from mlx_music.models.moss_sound_effect.pipeline import SoundEffectPipeline

    pipe = SoundEffectPipeline.from_pretrained(
        model_path=REAL_MOSS_MODEL_PATH,
        tokenizer_path=REAL_MOSS_TOKENIZER_PATH,
    )
    waveform = pipe.generate(
        "fresh snow crunching under footsteps",
        duration=0.5,
        max_new_tokens=20,
    )

    assert waveform is not None
    _materialize(waveform)
    assert waveform.shape[-1] > 0
