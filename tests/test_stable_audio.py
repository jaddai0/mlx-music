"""Tests for Stable Audio Open MLX implementation."""

import mlx.core as mx
import numpy as np
import pytest


class TestStableAudioLazyLoading:
    """Test lazy loading of Stable Audio module."""

    def test_stable_audio_in_module_dir(self):
        """Test that StableAudio is listed in module directory."""
        import mlx_music

        attrs = dir(mlx_music)
        assert "StableAudio" in attrs

    def test_stable_audio_lazy_access(self):
        """Test that StableAudio is lazily loaded on first access."""
        import mlx_music

        StableAudio = mlx_music.StableAudio

        assert StableAudio is not None
        assert hasattr(StableAudio, "from_pretrained")
        assert hasattr(StableAudio, "generate")


class TestStableAudioConfig:
    """Test Stable Audio configuration classes."""

    def test_dit_config_defaults(self):
        """Test DiTConfig has sensible defaults."""
        from mlx_music.models.stable_audio.config import DiTConfig

        config = DiTConfig()

        assert config.num_layers == 24
        assert config.num_attention_heads == 24
        assert config.num_key_value_heads == 12
        assert config.attention_head_dim == 64
        assert config.cross_attention_dim == 768

    def test_dit_config_from_dict(self):
        """Test creating DiTConfig from dictionary."""
        from mlx_music.models.stable_audio.config import DiTConfig

        config_dict = {
            "num_layers": 12,
            "num_attention_heads": 16,
            "attention_head_dim": 64,
        }
        config = DiTConfig.from_dict(config_dict)

        assert config.num_layers == 12
        assert config.num_attention_heads == 16
        assert config.attention_head_dim == 64

    def test_vae_config_defaults(self):
        """Test VAEConfig has sensible defaults."""
        from mlx_music.models.stable_audio.config import VAEConfig

        config = VAEConfig()

        assert config.latent_channels == 64
        assert config.audio_channels == 2
        assert config.encoder_hidden_size == 128

    def test_stable_audio_config_defaults(self):
        """Test StableAudioConfig has sensible defaults."""
        from mlx_music.models.stable_audio.config import StableAudioConfig

        config = StableAudioConfig()

        assert config.sample_rate == 44100
        assert config.max_duration_seconds == 47.0
        assert config.transformer is not None
        assert config.vae is not None


class TestEDMScheduler:
    """Test EDM DPM-Solver scheduler."""

    def test_scheduler_creation(self):
        """Test scheduler creation."""
        from mlx_music.models.stable_audio.scheduler import EDMDPMSolverMultistepScheduler

        scheduler = EDMDPMSolverMultistepScheduler(
            sigma_min=0.3,
            sigma_max=500.0,
            sigma_data=1.0,
        )

        assert scheduler is not None
        assert scheduler.sigma_min == 0.3
        assert scheduler.sigma_max == 500.0

    def test_scheduler_set_timesteps(self):
        """Test setting timesteps."""
        from mlx_music.models.stable_audio.scheduler import EDMDPMSolverMultistepScheduler

        scheduler = EDMDPMSolverMultistepScheduler()
        scheduler.set_timesteps(50)

        assert scheduler.num_inference_steps == 50
        assert scheduler.timesteps is not None
        assert len(scheduler.timesteps) == 50
        assert scheduler.sigmas is not None
        assert len(scheduler.sigmas) == 51  # n+1 sigmas

    def test_karras_sigmas_decrease(self):
        """Test that Karras sigmas decrease from max to min."""
        from mlx_music.models.stable_audio.scheduler import EDMDPMSolverMultistepScheduler

        scheduler = EDMDPMSolverMultistepScheduler()
        scheduler.set_timesteps(10)

        sigmas = np.array(scheduler.sigmas)
        # Sigmas should be decreasing
        for i in range(len(sigmas) - 1):
            assert sigmas[i] > sigmas[i + 1]

    def test_scheduler_step(self):
        """Test scheduler step produces valid output."""
        from mlx_music.models.stable_audio.scheduler import EDMDPMSolverMultistepScheduler

        scheduler = EDMDPMSolverMultistepScheduler()
        scheduler.set_timesteps(10)

        # Create dummy inputs
        batch_size = 1
        seq_len = 100
        channels = 64
        sample = mx.random.normal((batch_size, seq_len, channels))
        model_output = mx.random.normal((batch_size, seq_len, channels))

        # Take a step
        timestep = scheduler.timesteps[0]
        output = scheduler.step(model_output, timestep, sample)

        assert output.prev_sample is not None
        assert output.prev_sample.shape == sample.shape

    def test_retrieve_timesteps(self):
        """Test retrieve_timesteps utility."""
        from mlx_music.models.stable_audio.scheduler import (
            EDMDPMSolverMultistepScheduler,
            retrieve_timesteps,
        )

        scheduler = EDMDPMSolverMultistepScheduler()
        timesteps, num_steps = retrieve_timesteps(scheduler, num_inference_steps=50)

        assert len(timesteps) == 50
        assert num_steps == 50


class TestConditioning:
    """Test conditioning modules."""

    def test_timestep_embedding(self):
        """Test timestep embedding."""
        from mlx_music.models.stable_audio.conditioning import TimestepEmbedding

        emb = TimestepEmbedding(dim=256, time_embed_dim=512)
        timesteps = mx.array([0.5, 1.0, 10.0])

        output = emb(timesteps)

        assert output.shape == (3, 512)

    def test_number_embedding(self):
        """Test number embedding for timing."""
        from mlx_music.models.stable_audio.conditioning import NumberEmbedding

        emb = NumberEmbedding(fourier_dim=256, output_dim=768)
        values = mx.array([0.0, 10.0, 30.0])

        output = emb(values)

        assert output.shape == (3, 768)

    def test_projection_model(self):
        """Test projection model."""
        from mlx_music.models.stable_audio.conditioning import ProjectionModel

        proj = ProjectionModel(
            text_encoder_dim=768,
            output_dim=1536,
        )

        batch_size = 2
        text_embeds = mx.random.normal((batch_size, 768))
        seconds_start = mx.array([0.0, 0.0])
        seconds_total = mx.array([30.0, 45.0])

        output = proj(text_embeds, seconds_start, seconds_total)

        assert output.shape == (batch_size, 1536)


class TestVAE:
    """Test AutoencoderOobleck VAE."""

    def test_snake_activation(self):
        """Test Snake activation function."""
        from mlx_music.models.stable_audio.vae import snake

        x = mx.random.normal((2, 100, 64))
        alpha = mx.ones((64,))

        output = snake(x, alpha[None, None, :])

        assert output.shape == x.shape
        # Snake(x) = x + (1/alpha) * sin^2(alpha * x)
        # Output should be different from input
        assert not mx.allclose(output, x)

    def test_snake1d_module(self):
        """Test Snake1d module."""
        from mlx_music.models.stable_audio.vae import Snake1d

        snake = Snake1d(channels=64)
        x = mx.random.normal((2, 100, 64))

        output = snake(x)

        assert output.shape == x.shape

    def test_vae_config_compression_ratio(self):
        """Test VAE compression ratio calculation."""
        from mlx_music.models.stable_audio.config import VAEConfig

        config = VAEConfig()
        compression = 1
        for r in config.downsampling_ratios:
            compression *= r

        # 2 * 4 * 4 * 8 * 8 = 2048
        assert compression == 2048


class TestTransformer:
    """Test StableAudioDiT transformer."""

    def test_rms_norm(self):
        """Test RMSNorm layer."""
        from mlx_music.models.stable_audio.transformer import RMSNorm

        norm = RMSNorm(dim=256)
        x = mx.random.normal((2, 100, 256))

        output = norm(x)

        assert output.shape == x.shape

    def test_rotary_embedding(self):
        """Test rotary position embeddings."""
        from mlx_music.models.stable_audio.transformer import RotaryEmbedding

        rope = RotaryEmbedding(dim=64, max_seq_len=1024)
        cos, sin = rope(seq_len=100)

        # rope_dim = dim // 2 = 32
        # inv_freq size = rope_dim // 2 = 16 (arange(0, 32, 2) = 16 elements)
        assert cos.shape == (100, 16)
        assert sin.shape == (100, 16)

    def test_gq_attention_shapes(self):
        """Test GQA attention produces correct shapes."""
        from mlx_music.models.stable_audio.transformer import GQAttention

        attn = GQAttention(
            dim=256,
            num_attention_heads=8,
            num_key_value_heads=4,
            head_dim=32,
        )

        batch_size = 2
        seq_len = 50
        x = mx.random.normal((batch_size, seq_len, 256))

        output = attn(x)

        assert output.shape == (batch_size, seq_len, 256)

    def test_cross_attention_shapes(self):
        """Test cross-attention produces correct shapes."""
        from mlx_music.models.stable_audio.transformer import CrossAttention

        cross_attn = CrossAttention(
            dim=256,
            cross_attention_dim=768,
            num_attention_heads=8,
            head_dim=32,
        )

        batch_size = 2
        seq_len = 50
        enc_seq_len = 64
        x = mx.random.normal((batch_size, seq_len, 256))
        encoder_hidden = mx.random.normal((batch_size, enc_seq_len, 768))

        output = cross_attn(x, encoder_hidden)

        assert output.shape == (batch_size, seq_len, 256)

    def test_dit_block_forward(self):
        """Test DiT block forward pass."""
        from mlx_music.models.stable_audio.config import DiTConfig
        from mlx_music.models.stable_audio.transformer import DiTBlock, RotaryEmbedding

        config = DiTConfig(
            num_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            attention_head_dim=32,
            cross_attention_dim=768,
            global_states_input_dim=256,
            ff_mult=2.0,
        )
        block = DiTBlock(config)

        batch_size = 1
        seq_len = 20
        hidden_dim = config.num_attention_heads * config.attention_head_dim  # 4 * 32 = 128

        x = mx.random.normal((batch_size, seq_len, hidden_dim))
        encoder_hidden = mx.random.normal((batch_size, 64, 768))
        global_cond = mx.random.normal((batch_size, 256))

        rope = RotaryEmbedding(dim=config.attention_head_dim)
        cos, sin = rope(seq_len)

        output = block(x, encoder_hidden, global_cond, cos, sin)

        assert output.shape == x.shape


class TestStableAudioModel:
    """Test StableAudio main model class."""

    def test_model_creation(self):
        """Test model can be created with minimal components."""
        from mlx_music.models.stable_audio.config import DiTConfig, StableAudioConfig, VAEConfig
        from mlx_music.models.stable_audio.conditioning import ProjectionModel
        from mlx_music.models.stable_audio.model import StableAudio
        from mlx_music.models.stable_audio.transformer import StableAudioDiT
        from mlx_music.models.stable_audio.vae import AutoencoderOobleck

        # Create minimal configs
        dit_config = DiTConfig(
            num_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            attention_head_dim=32,
        )
        vae_config = VAEConfig()
        config = StableAudioConfig(transformer=dit_config, vae=vae_config)

        # Create components
        transformer = StableAudioDiT(dit_config)
        vae = AutoencoderOobleck(vae_config)
        projection = ProjectionModel()

        # Create model
        model = StableAudio(
            transformer=transformer,
            vae=vae,
            projection_model=projection,
            config=config,
        )

        assert model is not None
        assert model.transformer is not None
        assert model.vae is not None

    def test_generation_output_structure(self):
        """Test GenerationOutput holds expected fields."""
        from mlx_music.models.stable_audio.model import GenerationOutput

        output = GenerationOutput(
            audio=np.zeros((2, 44100)),
            sample_rate=44100,
            duration=1.0,
        )

        assert output.audio.shape == (2, 44100)
        assert output.sample_rate == 44100
        assert output.duration == 1.0
        assert output.latents is None

    def test_model_repr(self):
        """Test model string representation."""
        from mlx_music.models.stable_audio.config import DiTConfig, StableAudioConfig, VAEConfig
        from mlx_music.models.stable_audio.conditioning import ProjectionModel
        from mlx_music.models.stable_audio.model import StableAudio
        from mlx_music.models.stable_audio.transformer import StableAudioDiT
        from mlx_music.models.stable_audio.vae import AutoencoderOobleck

        dit_config = DiTConfig(num_layers=2, num_attention_heads=4, num_key_value_heads=2, attention_head_dim=32)
        config = StableAudioConfig(transformer=dit_config)

        transformer = StableAudioDiT(dit_config)
        vae = AutoencoderOobleck(VAEConfig())
        projection = ProjectionModel()

        model = StableAudio(
            transformer=transformer,
            vae=vae,
            projection_model=projection,
            config=config,
        )

        repr_str = repr(model)
        assert "StableAudio" in repr_str
        assert "sample_rate=44100" in repr_str


class TestStableAudioImports:
    """Test Stable Audio module imports work correctly."""

    def test_all_exports_accessible(self):
        """Test all __all__ exports are accessible."""
        from mlx_music.models.stable_audio import __all__

        import mlx_music.models.stable_audio as stable_audio

        for name in __all__:
            assert hasattr(stable_audio, name), f"Missing export: {name}"

    def test_main_classes_importable(self):
        """Test main classes can be imported."""
        from mlx_music.models.stable_audio import (
            AutoencoderOobleck,
            DiTConfig,
            EDMDPMSolverMultistepScheduler,
            GenerationOutput,
            GQAttention,
            ProjectionModel,
            RMSNorm,
            RotaryEmbedding,
            StableAudio,
            StableAudioConfig,
            StableAudioDiT,
            VAEConfig,
        )

        # Just verify they're all importable
        assert StableAudio is not None
        assert StableAudioConfig is not None
        assert StableAudioDiT is not None
        assert AutoencoderOobleck is not None


class TestValidation:
    """Test input validation and numerical stability."""

    def test_snake_activation_with_negative_alpha(self):
        """Test snake activation handles negative alpha safely."""
        from mlx_music.models.stable_audio.vae import snake

        x = mx.random.normal((2, 100, 64))
        # Test with negative alpha (should not crash or produce NaN)
        alpha = -0.5 * mx.ones((64,))[None, None, :]

        output = snake(x, alpha)

        assert output.shape == x.shape
        assert not mx.any(mx.isnan(output))
        assert not mx.any(mx.isinf(output))

    def test_snake_activation_with_near_zero_alpha(self):
        """Test snake activation handles near-zero alpha safely."""
        from mlx_music.models.stable_audio.vae import snake

        x = mx.random.normal((2, 100, 64))
        # Test with very small alpha
        alpha = 1e-12 * mx.ones((64,))[None, None, :]

        output = snake(x, alpha)

        assert output.shape == x.shape
        assert not mx.any(mx.isnan(output))
        assert not mx.any(mx.isinf(output))

    def test_vae_encode_input_validation(self):
        """Test VAE encode validates input dimensions."""
        from mlx_music.models.stable_audio.config import VAEConfig
        from mlx_music.models.stable_audio.vae import AutoencoderOobleck

        config = VAEConfig()
        vae = AutoencoderOobleck(config)

        # Test with wrong number of dimensions (should raise)
        x_2d = mx.random.normal((100, 2))

        with pytest.raises(ValueError, match="Expected 3D input"):
            vae.encode(x_2d)

    def test_scheduler_numerical_stability(self):
        """Test scheduler handles edge cases safely."""
        from mlx_music.models.stable_audio.scheduler import EDMDPMSolverMultistepScheduler

        scheduler = EDMDPMSolverMultistepScheduler()
        scheduler.set_timesteps(10)

        # Full iteration through all timesteps should not produce NaN
        sample = mx.random.normal((1, 100, 64))
        model_output = mx.random.normal((1, 100, 64))

        for timestep in scheduler.timesteps:
            output = scheduler.step(model_output, timestep, sample)
            sample = output.prev_sample

            assert not mx.any(mx.isnan(sample))
            assert not mx.any(mx.isinf(sample))


class TestWeightLoader:
    """Test weight loader functions for Stable Audio."""

    def test_load_stable_audio_weights_function_exists(self):
        """Test load_stable_audio_weights function is available."""
        from mlx_music.weights.weight_loader import load_stable_audio_weights

        assert callable(load_stable_audio_weights)

    def test_load_stable_audio_weights_invalid_component(self):
        """Test invalid component raises error."""
        from mlx_music.weights.weight_loader import load_stable_audio_weights

        with pytest.raises(ValueError, match="Unknown component"):
            load_stable_audio_weights("/nonexistent/path", component="invalid")
