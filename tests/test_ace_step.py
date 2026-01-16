"""Tests for ACE-Step MLX implementation."""

import os
import sys
from pathlib import Path

import mlx.core as mx
import numpy as np
import pytest


# Test model path from environment variable (for CI-friendly testing)
# Set this environment variable to run integration tests with real models:
#   export ACE_STEP_PATH=/path/to/ACE-Step
ACE_STEP_PATH = os.environ.get("ACE_STEP_PATH", "")


class TestACEStepLazyLoading:
    """Test lazy loading of ACE-Step module."""

    def test_acestep_in_module_dir(self):
        """Test that ACEStep is listed in module directory."""
        import mlx_music

        attrs = dir(mlx_music)
        assert "ACEStep" in attrs

    def test_acestep_lazy_access(self):
        """Test that ACEStep is lazily loaded on first access."""
        import mlx_music

        ACEStep = mlx_music.ACEStep

        assert ACEStep is not None
        assert hasattr(ACEStep, "from_pretrained")
        assert hasattr(ACEStep, "generate")


class TestACEStepConfig:
    """Test ACE-Step configuration classes."""

    def test_transformer_config_defaults(self):
        """Test ACEStepConfig has sensible defaults."""
        from mlx_music.models.ace_step.transformer import ACEStepConfig

        config = ACEStepConfig()

        assert config.inner_dim == 2560
        assert config.num_attention_heads == 20
        assert config.num_layers == 24
        assert config.in_channels == 8
        assert config.max_height == 16

    def test_config_from_dict(self):
        """Test creating config from dictionary."""
        from mlx_music.models.ace_step.transformer import ACEStepConfig

        config_dict = {
            "inner_dim": 1024,
            "num_attention_heads": 16,
            "num_layers": 12,
        }
        config = ACEStepConfig.from_dict(config_dict)

        assert config.inner_dim == 1024
        assert config.num_attention_heads == 16
        assert config.num_layers == 12

    def test_vocoder_config_defaults(self):
        """Test VocoderConfig has sensible defaults."""
        from mlx_music.models.ace_step.vocoder import VocoderConfig

        config = VocoderConfig()

        assert config.sampling_rate == 44100
        assert config.n_mels == 128
        assert config.upsample_initial_channel == 1024

    def test_dcae_config_defaults(self):
        """Test DCAEConfig has sensible defaults."""
        from mlx_music.models.ace_step.dcae import DCAEConfig

        config = DCAEConfig()

        assert config.latent_channels == 8
        assert config.base_channels == 128


class TestACEStepLyricTokenizer:
    """Test ACE-Step lyric tokenizer."""

    def test_tokenizer_initialization(self):
        """Test tokenizer initializes correctly."""
        from mlx_music.models.ace_step.lyric_tokenizer import VoiceBpeTokenizer

        tokenizer = VoiceBpeTokenizer()

        assert tokenizer is not None
        assert len(tokenizer) > 0
        assert tokenizer.tokenizer is not None

    def test_tokenizer_encode_decode(self):
        """Test tokenizer encode/decode roundtrip."""
        from mlx_music.models.ace_step.lyric_tokenizer import VoiceBpeTokenizer

        tokenizer = VoiceBpeTokenizer()

        text = "hello world"
        tokens = tokenizer.encode(text, lang="en")
        decoded = tokenizer.decode(tokens)

        # Decoded should contain the original text (may have language tag)
        assert "hello" in decoded.lower()
        assert "world" in decoded.lower()

    def test_tokenizer_english_normalization(self):
        """Test English text normalization."""
        from mlx_music.models.ace_step.lyric_tokenizer import VoiceBpeTokenizer

        tokenizer = VoiceBpeTokenizer()

        # Test abbreviation expansion
        tokens = tokenizer.encode("Hello Mr. Smith", lang="en")
        decoded = tokenizer.decode(tokens)
        # Verify abbreviation was expanded (case-insensitive check)
        assert "mister" in decoded.lower()
        # Verify original abbreviation is NOT present
        assert "mr." not in decoded.lower()

        # Test symbol expansion
        tokens = tokenizer.encode("I have 50%", lang="en")
        decoded = tokenizer.decode(tokens)
        assert "percent" in decoded.lower()
        # Verify original symbol is NOT present
        assert "%" not in decoded

        # Test number expansion
        tokens = tokenizer.encode("This is 1st", lang="en")
        decoded = tokenizer.decode(tokens)
        assert "first" in decoded.lower()
        # Verify original ordinal is NOT present
        assert "1st" not in decoded.lower()

    def test_tokenizer_vocab_size(self):
        """Test tokenizer reports correct vocab size."""
        from mlx_music.models.ace_step.lyric_tokenizer import VoiceBpeTokenizer

        tokenizer = VoiceBpeTokenizer()

        # ACE-Step vocab should be a few thousand tokens
        assert len(tokenizer) > 1000
        assert len(tokenizer) < 100000

    def test_tokenizer_batch_decode(self):
        """Test batch decoding."""
        from mlx_music.models.ace_step.lyric_tokenizer import VoiceBpeTokenizer

        tokenizer = VoiceBpeTokenizer()

        texts = ["hello", "world"]
        sequences = [tokenizer.encode(t, lang="en") for t in texts]
        decoded = tokenizer.batch_decode(sequences)

        assert len(decoded) == 2
        assert "hello" in decoded[0].lower()
        assert "world" in decoded[1].lower()

    def test_get_tokenizer_singleton(self):
        """Test get_tokenizer returns singleton."""
        from mlx_music.models.ace_step.lyric_tokenizer import get_tokenizer

        tokenizer1 = get_tokenizer()
        tokenizer2 = get_tokenizer()

        assert tokenizer1 is tokenizer2


class TestACEStepTextNormalization:
    """Test text normalization functions."""

    def test_expand_abbreviations(self):
        """Test abbreviation expansion."""
        from mlx_music.models.ace_step.lyric_tokenizer import expand_abbreviations

        result = expand_abbreviations("Hello Mr. Smith", "en")
        assert "mister" in result

        result = expand_abbreviations("Dr. Jones is here", "en")
        assert "doctor" in result

    def test_expand_symbols(self):
        """Test symbol expansion."""
        from mlx_music.models.ace_step.lyric_tokenizer import expand_symbols

        result = expand_symbols("I have $20", "en")
        assert "dollar" in result

        result = expand_symbols("50%", "en")
        assert "percent" in result

    def test_expand_numbers(self):
        """Test number expansion."""
        from mlx_music.models.ace_step.lyric_tokenizer import expand_numbers

        result = expand_numbers("I have 5 apples", "en")
        assert "five" in result

        result = expand_numbers("This is the 1st time", "en")
        assert "first" in result

    def test_preprocess_text(self):
        """Test full text preprocessing."""
        from mlx_music.models.ace_step.lyric_tokenizer import preprocess_text

        result = preprocess_text("Hello Mr. Smith! I have $20.", "en")

        assert result == result.lower()  # Should be lowercased
        assert "mister" in result
        assert "dollar" in result


class TestACEStepScheduler:
    """Test ACE-Step diffusion schedulers."""

    def test_euler_scheduler_creation(self):
        """Test Euler scheduler creation."""
        from mlx_music.models.ace_step.scheduler import FlowMatchEulerDiscreteScheduler

        scheduler = FlowMatchEulerDiscreteScheduler(
            num_train_timesteps=1000,
            shift=3.0,
        )

        assert scheduler is not None
        assert scheduler.num_train_timesteps == 1000
        assert scheduler.shift == 3.0

    def test_heun_scheduler_creation(self):
        """Test Heun scheduler creation."""
        from mlx_music.models.ace_step.scheduler import FlowMatchHeunDiscreteScheduler

        scheduler = FlowMatchHeunDiscreteScheduler(
            num_train_timesteps=1000,
            shift=3.0,
        )

        assert scheduler is not None

    def test_get_scheduler(self):
        """Test scheduler factory function."""
        from mlx_music.models.ace_step.scheduler import get_scheduler

        euler = get_scheduler("euler", shift=3.0)
        assert euler is not None

        heun = get_scheduler("heun", shift=3.0)
        assert heun is not None

    def test_retrieve_timesteps(self):
        """Test timestep retrieval."""
        from mlx_music.models.ace_step.scheduler import (
            FlowMatchEulerDiscreteScheduler,
            retrieve_timesteps,
        )

        scheduler = FlowMatchEulerDiscreteScheduler(num_train_timesteps=1000, shift=3.0)
        timesteps, num_steps = retrieve_timesteps(scheduler, num_inference_steps=50)

        assert len(timesteps) == 50
        assert num_steps == 50


class TestACEStepTransformer:
    """Test ACE-Step transformer components."""

    def test_transformer_creation(self):
        """Test transformer model creation."""
        from mlx_music.models.ace_step.transformer import ACEStepConfig, ACEStepTransformer

        # Use smaller config for testing
        config = ACEStepConfig(
            inner_dim=256,
            num_attention_heads=4,
            attention_head_dim=64,
            num_layers=2,
            in_channels=8,
            max_height=16,
        )
        transformer = ACEStepTransformer(config)

        assert transformer is not None
        # Check that the transformer has the expected structure
        assert hasattr(transformer, "transformer_blocks")
        assert transformer.config.num_layers == 2

    def test_transformer_forward_shape(self):
        """Test transformer forward pass produces correct shapes."""
        from mlx_music.models.ace_step.transformer import ACEStepConfig, ACEStepTransformer

        config = ACEStepConfig(
            inner_dim=256,
            num_attention_heads=4,
            attention_head_dim=64,
            num_layers=2,
            in_channels=8,
            max_height=16,
        )
        transformer = ACEStepTransformer(config)

        # Create test input
        batch_size = 1
        latent_width = 10
        hidden_states = mx.random.normal((batch_size, 8, 16, latent_width))
        timestep = mx.array([500.0])
        text_embeds = mx.random.normal((batch_size, 32, config.text_embedding_dim))
        text_mask = mx.ones((batch_size, 32))

        # Forward pass
        output = transformer(
            hidden_states=hidden_states,
            timestep=timestep,
            encoder_text_hidden_states=text_embeds,
            text_attention_mask=text_mask,
        )

        assert output.shape == hidden_states.shape


class TestACEStepDCAE:
    """Test ACE-Step DCAE components."""

    def test_dcae_config_creation(self):
        """Test DCAEConfig creation."""
        from mlx_music.models.ace_step.dcae import DCAEConfig

        config = DCAEConfig(
            latent_channels=8,
            base_channels=128,
        )

        assert config.latent_channels == 8
        assert config.base_channels == 128


class TestACEStepVocoder:
    """Test ACE-Step vocoder components."""

    def test_vocoder_config_creation(self):
        """Test VocoderConfig creation."""
        from mlx_music.models.ace_step.vocoder import VocoderConfig

        config = VocoderConfig(
            sampling_rate=44100,
            n_mels=128,
        )

        assert config.sampling_rate == 44100
        assert config.n_mels == 128


class TestACEStepModel:
    """Test ACE-Step main model class."""

    def test_model_encode_text_placeholder(self):
        """Test text encoding with placeholder encoder."""
        from mlx_music.models.ace_step.model import ACEStep
        from mlx_music.models.ace_step.transformer import ACEStepConfig, ACEStepTransformer

        # Create minimal model without loading weights
        config = ACEStepConfig(inner_dim=256, num_attention_heads=4, attention_head_dim=64, num_layers=2)
        transformer = ACEStepTransformer(config)

        model = ACEStep(
            transformer=transformer,
            config=config,
            text_encoder=None,
            audio_pipeline=None,
            lyric_tokenizer=None,
        )

        # Test encoding (should return placeholder)
        embeddings, mask = model.encode_text("test prompt")

        assert embeddings.shape == (1, 64, config.text_embedding_dim)
        assert mask.shape == (1, 64)

    def test_model_encode_lyrics_with_tokenizer(self):
        """Test lyric encoding with real tokenizer."""
        from mlx_music.models.ace_step.model import ACEStep
        from mlx_music.models.ace_step.lyric_tokenizer import VoiceBpeTokenizer
        from mlx_music.models.ace_step.transformer import ACEStepConfig, ACEStepTransformer

        config = ACEStepConfig(inner_dim=256, num_attention_heads=4, attention_head_dim=64, num_layers=2)
        transformer = ACEStepTransformer(config)
        tokenizer = VoiceBpeTokenizer()

        model = ACEStep(
            transformer=transformer,
            config=config,
            text_encoder=None,
            audio_pipeline=None,
            lyric_tokenizer=tokenizer,
        )

        # Test encoding
        tokens, mask = model.encode_lyrics("Dancing through the night")

        assert tokens.shape[0] == 1  # batch size
        assert tokens.shape[1] == 512  # default max_length
        assert mask.shape == tokens.shape

        # Check that mask has correct pattern (1s then 0s)
        mask_np = np.array(mask[0])
        first_zero = np.where(mask_np == 0)[0]
        if len(first_zero) > 0:
            # All values before first zero should be 1
            assert np.all(mask_np[:first_zero[0]] == 1)
            # All values after first zero should be 0
            assert np.all(mask_np[first_zero[0]:] == 0)
        else:
            # No padding - verify all values are 1
            assert np.all(mask_np == 1), "Mask should be all 1s when no padding"

    def test_model_encode_lyrics_without_tokenizer(self):
        """Test lyric encoding fallback without tokenizer."""
        from mlx_music.models.ace_step.model import ACEStep
        from mlx_music.models.ace_step.transformer import ACEStepConfig, ACEStepTransformer

        config = ACEStepConfig(inner_dim=256, num_attention_heads=4, attention_head_dim=64, num_layers=2)
        transformer = ACEStepTransformer(config)

        model = ACEStep(
            transformer=transformer,
            config=config,
            text_encoder=None,
            audio_pipeline=None,
            lyric_tokenizer=None,
        )

        # Test encoding (should return placeholder zeros)
        tokens, mask = model.encode_lyrics("test lyrics")

        assert tokens.shape == (1, 512)
        assert mask.shape == (1, 512)
        # Should be all zeros for placeholder tokens
        assert np.sum(np.array(tokens)) == 0
        # Mask should also be all zeros to indicate no valid tokens
        assert np.sum(np.array(mask)) == 0

    def test_model_repr(self):
        """Test model string representation."""
        from mlx_music.models.ace_step.model import ACEStep
        from mlx_music.models.ace_step.transformer import ACEStepConfig, ACEStepTransformer

        config = ACEStepConfig(inner_dim=256, num_attention_heads=4, attention_head_dim=64, num_layers=2)
        transformer = ACEStepTransformer(config)

        model = ACEStep(
            transformer=transformer,
            config=config,
            text_encoder=None,
            audio_pipeline=None,
            lyric_tokenizer=None,
        )

        repr_str = repr(model)
        assert "ACEStep" in repr_str
        assert "has_text_encoder=False" in repr_str
        assert "has_audio_pipeline=False" in repr_str
        assert "has_lyric_tokenizer=False" in repr_str

    def test_generation_output_structure(self):
        """Test GenerationOutput holds expected fields."""
        from mlx_music.models.ace_step.model import GenerationOutput

        output = GenerationOutput(
            audio=np.zeros((2, 44100)),
            sample_rate=44100,
            duration=1.0,
        )

        assert output.audio.shape == (2, 44100)
        assert output.sample_rate == 44100
        assert output.duration == 1.0
        assert output.latents is None


class TestACEStepImports:
    """Test ACE-Step module imports work correctly."""

    def test_all_exports_accessible(self):
        """Test all __all__ exports are accessible."""
        from mlx_music.models.ace_step import __all__

        import mlx_music.models.ace_step as ace_step

        for name in __all__:
            assert hasattr(ace_step, name), f"Missing export: {name}"

    def test_main_classes_importable(self):
        """Test main classes can be imported."""
        from mlx_music.models.ace_step import (
            ACEStep,
            ACEStepConfig,
            ACEStepTransformer,
            DCAE,
            DCAEConfig,
            FlowMatchEulerDiscreteScheduler,
            FlowMatchHeunDiscreteScheduler,
            GenerationConfig,
            GenerationOutput,
            HiFiGANVocoder,
            MusicDCAEPipeline,
            VoiceBpeTokenizer,
            get_tokenizer,
        )

        # Just verify they're all importable
        assert ACEStep is not None
        assert ACEStepConfig is not None
        assert VoiceBpeTokenizer is not None
