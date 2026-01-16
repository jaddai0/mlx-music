"""Tests for MusicGen MLX implementation."""

import os
import sys
from pathlib import Path

import mlx.core as mx
import numpy as np
import pytest


# Test model paths from environment variables (for CI-friendly testing)
# Set these environment variables to run integration tests with real models:
#   export MUSICGEN_SMALL_PATH=/path/to/MusicGen-small
#   export MUSICGEN_MELODY_PATH=/path/to/MusicGen-melody
MUSICGEN_SMALL_PATH = os.environ.get("MUSICGEN_SMALL_PATH", "")
MUSICGEN_MELODY_PATH = os.environ.get("MUSICGEN_MELODY_PATH", "")


class TestMusicGenLazyLoading:
    """Test lazy loading of MusicGen module."""

    def test_musicgen_in_module_dir(self):
        """Test that MusicGen is listed in module directory."""
        import mlx_music

        attrs = dir(mlx_music)
        assert "MusicGen" in attrs

    def test_musicgen_in_all(self):
        """Test that MusicGen is in __all__."""
        import mlx_music

        assert "MusicGen" in mlx_music.__all__

    def test_musicgen_lazy_access(self):
        """Test that MusicGen is lazily loaded on first access."""
        import mlx_music

        MusicGen = mlx_music.MusicGen

        assert MusicGen is not None
        assert hasattr(MusicGen, "from_pretrained")
        assert hasattr(MusicGen, "generate")


class TestMusicGenConfig:
    """Test MusicGen configuration classes."""

    def test_decoder_config_defaults(self):
        """Test MusicGenDecoderConfig has sensible defaults."""
        from mlx_music.models.musicgen.config import MusicGenDecoderConfig

        config = MusicGenDecoderConfig()

        assert config.vocab_size == 2048
        assert config.hidden_size == 1024
        assert config.num_hidden_layers == 24
        assert config.num_attention_heads == 16
        assert config.num_codebooks == 4

    def test_full_config_defaults(self):
        """Test MusicGenConfig assembles sub-configs correctly."""
        from mlx_music.models.musicgen.config import MusicGenConfig

        config = MusicGenConfig()

        assert config.model_type == "musicgen"
        assert hasattr(config, "decoder")
        assert hasattr(config, "audio_encoder")
        assert hasattr(config, "text_encoder")

    @pytest.mark.skipif(
        not MUSICGEN_SMALL_PATH or not Path(MUSICGEN_SMALL_PATH).exists(),
        reason="MUSICGEN_SMALL_PATH not set or model not found",
    )
    def test_config_from_pretrained_small(self):
        """Test loading config from MusicGen-small checkpoint."""
        from mlx_music.models.musicgen.config import MusicGenConfig

        config = MusicGenConfig.from_pretrained(MUSICGEN_SMALL_PATH)

        # MusicGen-small specs
        assert config.decoder.hidden_size == 1024
        assert config.decoder.num_hidden_layers == 24
        assert config.decoder.num_attention_heads == 16
        assert config.decoder.num_codebooks == 4

    @pytest.mark.skipif(
        not MUSICGEN_MELODY_PATH or not Path(MUSICGEN_MELODY_PATH).exists(),
        reason="MUSICGEN_MELODY_PATH not set or model not found",
    )
    def test_config_from_pretrained_melody(self):
        """Test loading config from MusicGen-melody checkpoint."""
        from mlx_music.models.musicgen.config import MusicGenConfig

        config = MusicGenConfig.from_pretrained(MUSICGEN_MELODY_PATH)

        # MusicGen-melody specs (larger than small)
        assert config.decoder.hidden_size == 1536
        assert config.decoder.num_hidden_layers == 48
        assert config.decoder.num_attention_heads == 24
        assert config.is_melody


class TestMusicGenSampling:
    """Test MusicGen sampling utilities."""

    def test_top_k_filtering(self):
        """Test top-k filtering on logits."""
        from mlx_music.models.musicgen.generation import top_k_filtering

        # Create logits with known distribution
        logits = mx.array([[1.0, 2.0, 3.0, 4.0, 5.0]])

        # Keep only top 2
        filtered = top_k_filtering(logits, top_k=2)
        filtered_np = np.array(filtered)

        # Top 2 values (5.0 and 4.0) should be preserved
        assert filtered_np[0, 4] == 5.0
        assert filtered_np[0, 3] == 4.0
        # Others should be -inf
        assert np.isneginf(filtered_np[0, 0])
        assert np.isneginf(filtered_np[0, 1])
        assert np.isneginf(filtered_np[0, 2])

    def test_top_k_filtering_disabled(self):
        """Test top-k filtering is disabled when top_k <= 0."""
        from mlx_music.models.musicgen.generation import top_k_filtering

        logits = mx.array([[1.0, 2.0, 3.0]])
        filtered = top_k_filtering(logits, top_k=0)

        # Should be unchanged
        np.testing.assert_array_equal(np.array(filtered), np.array(logits))

    def test_top_p_filtering(self):
        """Test nucleus (top-p) sampling filter."""
        from mlx_music.models.musicgen.generation import top_p_filtering

        # Create logits
        logits = mx.array([[1.0, 2.0, 3.0, 10.0]])  # Last value dominates

        # With high top_p, should keep most
        filtered = top_p_filtering(logits, top_p=0.99)
        filtered_np = np.array(filtered)

        # The dominant value should always be preserved
        assert not np.isneginf(filtered_np[0, 3])

    def test_top_p_filtering_disabled(self):
        """Test top-p filtering disabled with p >= 1.0."""
        from mlx_music.models.musicgen.generation import top_p_filtering

        logits = mx.array([[1.0, 2.0, 3.0]])
        filtered = top_p_filtering(logits, top_p=1.0)

        # Should be unchanged
        np.testing.assert_array_equal(np.array(filtered), np.array(logits))

    def test_sample_next_token_deterministic(self):
        """Test sampling with temperature=0-like behavior."""
        from mlx_music.models.musicgen.generation import sample_next_token

        # Set seed for reproducibility
        mx.random.seed(42)

        # Create logits with one dominant value
        logits = mx.array([[0.0, 0.0, 0.0, 100.0]])  # Index 3 is very high

        # Sample with top_k=1 to make it deterministic
        token = sample_next_token(logits, temperature=1.0, top_k=1, top_p=0.0)

        assert np.array(token)[0, 0] == 3

    def test_classifier_free_guidance(self):
        """Test CFG combines conditional and unconditional logits."""
        from mlx_music.models.musicgen.generation import apply_classifier_free_guidance

        cond = mx.array([[2.0, 4.0]])
        uncond = mx.array([[1.0, 2.0]])

        # With scale=1.0, should just return cond
        result = apply_classifier_free_guidance(cond, uncond, guidance_scale=1.0)
        np.testing.assert_array_equal(np.array(result), np.array(cond))

        # With scale=2.0: uncond + 2 * (cond - uncond) = uncond + 2*diff
        # [1, 2] + 2 * ([2, 4] - [1, 2]) = [1, 2] + 2 * [1, 2] = [3, 6]
        result = apply_classifier_free_guidance(cond, uncond, guidance_scale=2.0)
        expected = mx.array([[3.0, 6.0]])
        np.testing.assert_array_almost_equal(np.array(result), np.array(expected))


class TestMusicGenGeneration:
    """Test MusicGen generation configuration."""

    def test_generation_config_defaults(self):
        """Test GenerationConfig has sensible defaults."""
        from mlx_music.models.musicgen.generation import GenerationConfig

        config = GenerationConfig()

        assert config.duration == 10.0
        assert config.max_duration == 30.0
        assert config.temperature == 1.0
        assert config.top_k == 250
        assert config.guidance_scale == 3.0

    def test_generation_output_structure(self):
        """Test GenerationOutput holds expected fields."""
        from mlx_music.models.musicgen.generation import GenerationOutput

        output = GenerationOutput(
            audio=np.zeros((2, 32000)),
            sample_rate=32000,
            duration=1.0,
        )

        assert output.audio.shape == (2, 32000)
        assert output.sample_rate == 32000
        assert output.duration == 1.0
        assert output.codes is None


class TestMusicGenConditioning:
    """Test MusicGen conditioning modules."""

    def test_placeholder_encoder(self):
        """Test placeholder encoder returns correct shapes."""
        from mlx_music.models.musicgen.conditioning import PlaceholderTextEncoder

        encoder = PlaceholderTextEncoder(hidden_size=768)
        embeddings, mask = encoder.encode("test prompt", max_length=256)

        assert embeddings.shape == (1, 256, 768)
        assert mask.shape == (1, 256)

    def test_placeholder_encoder_batch(self):
        """Test placeholder encoder handles batches."""
        from mlx_music.models.musicgen.conditioning import PlaceholderTextEncoder

        encoder = PlaceholderTextEncoder(hidden_size=512)
        texts = ["prompt 1", "prompt 2", "prompt 3"]
        embeddings, mask = encoder.encode_batch(texts, max_length=128)

        assert embeddings.shape == (3, 128, 512)
        assert mask.shape == (3, 128)


class TestMusicGenTransformer:
    """Test MusicGen transformer components."""

    def test_decoder_config_head_dim(self):
        """Test head dimension calculation."""
        from mlx_music.models.musicgen.config import MusicGenDecoderConfig

        config = MusicGenDecoderConfig(hidden_size=1024, num_attention_heads=16)
        assert config.head_dim == 64

    def test_sinusoidal_embeddings_shape(self):
        """Test sinusoidal position embeddings have correct shape."""
        from mlx_music.models.musicgen.transformer import (
            MusicGenSinusoidalPositionEmbedding,
        )

        embed = MusicGenSinusoidalPositionEmbedding(
            num_positions=2048, embedding_dim=1024
        )
        positions = mx.arange(100)
        embeddings = embed(positions)

        assert embeddings.shape == (100, 1024)

    def test_attention_layer_forward(self):
        """Test attention layer forward pass."""
        from mlx_music.models.musicgen.transformer import MusicGenAttention

        attn = MusicGenAttention(
            embed_dim=256,
            num_heads=4,
            dropout=0.0,
            is_cross_attention=False,
        )

        # Create input
        hidden_states = mx.random.normal((1, 10, 256))
        output, _ = attn(hidden_states)

        assert output.shape == hidden_states.shape

    def test_decoder_layer_forward(self):
        """Test decoder layer forward pass."""
        from mlx_music.models.musicgen.config import MusicGenDecoderConfig
        from mlx_music.models.musicgen.transformer import MusicGenDecoderLayer

        config = MusicGenDecoderConfig(
            hidden_size=256, num_attention_heads=4, ffn_dim=512
        )
        layer = MusicGenDecoderLayer(config)

        hidden_states = mx.random.normal((1, 10, 256))
        output, self_attn_cache, cross_attn_cache = layer(hidden_states)

        assert output.shape == hidden_states.shape
        # Self-attention cache should be populated
        assert self_attn_cache is not None
        # Cross-attention cache is None when no encoder states provided
        assert cross_attn_cache is None


class TestMusicGenCodecs:
    """Test EnCodec integration."""

    def test_placeholder_encodec(self):
        """Test placeholder EnCodec returns correct shapes."""
        from mlx_music.codecs import PlaceholderEnCodec

        codec = PlaceholderEnCodec(num_codebooks=4, sample_rate=32000)

        # Test decode
        codes = mx.zeros((1, 4, 50), dtype=mx.int32)  # 50 frames = 1 sec of codes
        audio = codec.decode(codes)

        # Shape is (batch, channels, samples)
        assert audio.shape[0] == 1  # batch
        assert audio.shape[1] == 1  # mono channel
        assert audio.shape[2] == 32000  # 50 frames * 640 = 32000 samples = 1 sec at 32kHz

    def test_placeholder_encodec_stereo(self):
        """Test placeholder EnCodec returns correct stereo shapes."""
        from mlx_music.codecs import PlaceholderEnCodec

        codec = PlaceholderEnCodec(num_codebooks=4, sample_rate=32000, audio_channels=2)

        # Test decode
        codes = mx.zeros((1, 4, 50), dtype=mx.int32)
        audio = codec.decode(codes)

        # Shape is (batch, channels, samples)
        assert audio.shape[0] == 1  # batch
        assert audio.shape[1] == 2  # stereo channels
        assert audio.shape[2] == 32000

        # Verify audio_channels property
        assert codec.audio_channels == 2

    def test_placeholder_encodec_stereo_from_model_id(self):
        """Test that stereo is auto-detected from model name."""
        from mlx_music.codecs import PlaceholderEnCodec

        # Test stereo model name detection
        codec = PlaceholderEnCodec.from_pretrained("facebook/musicgen-stereo-small")
        assert codec.audio_channels == 2

        # Test mono model name (should stay mono)
        codec_mono = PlaceholderEnCodec.from_pretrained("facebook/musicgen-small")
        assert codec_mono.audio_channels == 1

    def test_get_encodec_stereo(self):
        """Test get_encodec returns stereo codec with correct channels."""
        from mlx_music.codecs import get_encodec

        # Force placeholder to test stereo support
        codec = get_encodec(
            model_id="facebook/encodec_32khz",
            use_placeholder=True,
            audio_channels=2,
        )
        assert codec.audio_channels == 2

        codes = mx.zeros((1, 4, 50), dtype=mx.int32)
        audio = codec.decode(codes)
        assert audio.shape[1] == 2  # stereo

    def test_get_encodec_stereo_auto_detect(self):
        """Test get_encodec auto-detects stereo from model name."""
        from mlx_music.codecs import get_encodec

        # Should auto-detect stereo from model name
        codec = get_encodec(
            model_id="facebook/musicgen-stereo-medium",
            use_placeholder=True,
        )
        assert codec.audio_channels == 2


@pytest.mark.skipif(
    not MUSICGEN_SMALL_PATH or not Path(MUSICGEN_SMALL_PATH).exists(),
    reason="MUSICGEN_SMALL_PATH not set or model not found",
)
class TestMusicGenModelLoading:
    """Test MusicGen model loading (requires local models)."""

    def test_model_loads_without_encoders(self):
        """Test model loads with only decoder (fast test)."""
        from mlx_music import MusicGen

        model = MusicGen.from_pretrained(
            MUSICGEN_SMALL_PATH,
            load_text_encoder=False,
            load_encodec=False,
        )

        assert model is not None
        assert model.config.decoder.hidden_size == 1024
        assert model.text_encoder is None
        assert model.encodec is None

    def test_model_repr(self):
        """Test model string representation."""
        from mlx_music import MusicGen

        model = MusicGen.from_pretrained(
            MUSICGEN_SMALL_PATH,
            load_text_encoder=False,
            load_encodec=False,
        )

        repr_str = repr(model)
        assert "MusicGen" in repr_str
        assert "hidden_size=1024" in repr_str
        assert "num_codebooks=4" in repr_str
