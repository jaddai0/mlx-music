"""
Tests for mlx_music.models.ace_step.scheduler module.

Tests:
- SchedulerOutput: dataclass fields
- FlowMatchEulerDiscreteScheduler: set_timesteps, scale_noise, step
- FlowMatchHeunDiscreteScheduler: set_timesteps, scale_noise, step
- DPMPlusPlus2MKarrasScheduler: Karras sigmas, multistep correction, reset
- retrieve_timesteps: utility function
- get_scheduler: factory function
"""

import pytest
import mlx.core as mx


class TestSchedulerOutput:
    """Tests for SchedulerOutput dataclass."""

    def test_scheduler_output_fields(self):
        """SchedulerOutput should have expected fields."""
        from mlx_music.models.ace_step.scheduler import SchedulerOutput

        sample = mx.random.normal((2, 8, 100))
        output = SchedulerOutput(prev_sample=sample)

        assert output.prev_sample is not None
        assert output.pred_original_sample is None

    def test_scheduler_output_with_pred_original(self):
        """SchedulerOutput should accept pred_original_sample."""
        from mlx_music.models.ace_step.scheduler import SchedulerOutput

        sample = mx.random.normal((2, 8, 100))
        pred_orig = mx.random.normal((2, 8, 100))
        output = SchedulerOutput(prev_sample=sample, pred_original_sample=pred_orig)

        assert output.prev_sample is not None
        assert output.pred_original_sample is not None


class TestFlowMatchEulerDiscreteScheduler:
    """Tests for FlowMatchEulerDiscreteScheduler."""

    def test_init_defaults(self):
        """Scheduler should initialize with sensible defaults."""
        from mlx_music.models.ace_step.scheduler import FlowMatchEulerDiscreteScheduler

        scheduler = FlowMatchEulerDiscreteScheduler()

        assert scheduler.num_train_timesteps == 1000
        assert scheduler.shift == 3.0
        assert scheduler.timesteps is not None
        assert scheduler.sigmas is not None

    def test_set_timesteps(self):
        """set_timesteps should configure timesteps and sigmas."""
        from mlx_music.models.ace_step.scheduler import FlowMatchEulerDiscreteScheduler

        scheduler = FlowMatchEulerDiscreteScheduler()
        scheduler.set_timesteps(20)

        assert scheduler.num_inference_steps == 20
        assert scheduler.timesteps.shape == (20,)
        assert scheduler.sigmas.shape == (20,)

    def test_timesteps_decreasing(self):
        """Timesteps should be decreasing (noise to clean)."""
        from mlx_music.models.ace_step.scheduler import FlowMatchEulerDiscreteScheduler

        scheduler = FlowMatchEulerDiscreteScheduler()
        scheduler.set_timesteps(10)
        mx.synchronize()

        timesteps = scheduler.timesteps.tolist()
        for i in range(len(timesteps) - 1):
            assert timesteps[i] > timesteps[i + 1]

    def test_sigmas_decreasing(self):
        """Sigmas should be decreasing."""
        from mlx_music.models.ace_step.scheduler import FlowMatchEulerDiscreteScheduler

        scheduler = FlowMatchEulerDiscreteScheduler()
        scheduler.set_timesteps(10)
        mx.synchronize()

        sigmas = scheduler.sigmas.tolist()
        for i in range(len(sigmas) - 1):
            assert sigmas[i] > sigmas[i + 1]

    def test_scale_noise_formula(self):
        """scale_noise should implement flow matching interpolation."""
        from mlx_music.models.ace_step.scheduler import FlowMatchEulerDiscreteScheduler

        scheduler = FlowMatchEulerDiscreteScheduler()
        scheduler.set_timesteps(10)

        sample = mx.ones((2, 8, 100))
        noise = mx.zeros((2, 8, 100))

        # Get first timestep
        timestep = scheduler.timesteps[0]
        noisy = scheduler.scale_noise(sample, timestep, noise)
        mx.synchronize()

        # Result should be between sample and noise
        assert noisy.shape == sample.shape
        assert not mx.any(mx.isnan(noisy))

    def test_scale_noise_sigma_zero(self):
        """scale_noise with sigma=0 should return sample."""
        from mlx_music.models.ace_step.scheduler import FlowMatchEulerDiscreteScheduler

        scheduler = FlowMatchEulerDiscreteScheduler()
        scheduler.set_timesteps(10)

        sample = mx.random.normal((2, 8, 100))
        noise = mx.random.normal((2, 8, 100))

        # Last timestep should have lowest sigma
        timestep = scheduler.timesteps[-1]
        noisy = scheduler.scale_noise(sample, timestep, noise)
        mx.synchronize()

        # Should be close to sample (low sigma means low noise)
        assert noisy.shape == sample.shape

    def test_step_returns_scheduler_output(self):
        """step() should return SchedulerOutput."""
        from mlx_music.models.ace_step.scheduler import FlowMatchEulerDiscreteScheduler, SchedulerOutput

        scheduler = FlowMatchEulerDiscreteScheduler()
        scheduler.set_timesteps(10)

        model_output = mx.random.normal((2, 8, 100))
        sample = mx.random.normal((2, 8, 100))
        timestep = scheduler.timesteps[0]

        output = scheduler.step(model_output, timestep, sample)

        assert isinstance(output, SchedulerOutput)
        assert output.prev_sample.shape == sample.shape

    def test_step_return_dict_false(self):
        """step(return_dict=False) should return tuple."""
        from mlx_music.models.ace_step.scheduler import FlowMatchEulerDiscreteScheduler

        scheduler = FlowMatchEulerDiscreteScheduler()
        scheduler.set_timesteps(10)

        model_output = mx.random.normal((2, 8, 100))
        sample = mx.random.normal((2, 8, 100))
        timestep = scheduler.timesteps[0]

        output = scheduler.step(model_output, timestep, sample, return_dict=False)

        assert isinstance(output, tuple)
        assert output[0].shape == sample.shape

    def test_full_denoising_loop(self):
        """Complete denoising loop should produce valid output."""
        from mlx_music.models.ace_step.scheduler import FlowMatchEulerDiscreteScheduler

        scheduler = FlowMatchEulerDiscreteScheduler()
        scheduler.set_timesteps(10)

        # Start with noise
        sample = mx.random.normal((1, 8, 100))

        for t in scheduler.timesteps:
            # Mock model output (velocity prediction)
            model_output = mx.random.normal(sample.shape)
            output = scheduler.step(model_output, t, sample)
            sample = output.prev_sample

        mx.synchronize()
        assert sample.shape == (1, 8, 100)
        assert not mx.any(mx.isnan(sample))


class TestFlowMatchHeunDiscreteScheduler:
    """Tests for FlowMatchHeunDiscreteScheduler."""

    def test_init_defaults(self):
        """Scheduler should initialize with sensible defaults."""
        from mlx_music.models.ace_step.scheduler import FlowMatchHeunDiscreteScheduler

        scheduler = FlowMatchHeunDiscreteScheduler()

        assert scheduler.num_train_timesteps == 1000
        assert scheduler.shift == 3.0

    def test_set_timesteps(self):
        """set_timesteps should configure timesteps."""
        from mlx_music.models.ace_step.scheduler import FlowMatchHeunDiscreteScheduler

        scheduler = FlowMatchHeunDiscreteScheduler()
        scheduler.set_timesteps(20)

        assert scheduler.num_inference_steps == 20
        assert scheduler.timesteps.shape == (20,)

    def test_step_without_model_fn_uses_euler(self):
        """step() without model_fn should fall back to Euler."""
        from mlx_music.models.ace_step.scheduler import FlowMatchHeunDiscreteScheduler

        scheduler = FlowMatchHeunDiscreteScheduler()
        scheduler.set_timesteps(10)

        model_output = mx.random.normal((2, 8, 100))
        sample = mx.random.normal((2, 8, 100))
        timestep = scheduler.timesteps[0]

        # Without model_fn, should use Euler
        output = scheduler.step(model_output, timestep, sample, model_fn=None)

        assert output.prev_sample.shape == sample.shape

    def test_step_with_model_fn(self):
        """step() with model_fn should use Heun correction."""
        from mlx_music.models.ace_step.scheduler import FlowMatchHeunDiscreteScheduler

        scheduler = FlowMatchHeunDiscreteScheduler()
        scheduler.set_timesteps(10)

        model_output = mx.random.normal((2, 8, 100))
        sample = mx.random.normal((2, 8, 100))
        timestep = scheduler.timesteps[0]

        # Mock model function
        def mock_model_fn(x, t):
            return mx.random.normal(x.shape)

        output = scheduler.step(model_output, timestep, sample, model_fn=mock_model_fn)

        assert output.prev_sample.shape == sample.shape


class TestDPMPlusPlus2MKarrasScheduler:
    """Tests for DPMPlusPlus2MKarrasScheduler."""

    def test_init_defaults(self):
        """Scheduler should initialize with sensible defaults."""
        from mlx_music.models.ace_step.scheduler import DPMPlusPlus2MKarrasScheduler

        scheduler = DPMPlusPlus2MKarrasScheduler()

        assert scheduler.num_train_timesteps == 1000
        assert scheduler.sigma_min == 0.002
        assert scheduler.sigma_max == 80.0
        assert scheduler.rho == 7.0

    def test_set_timesteps(self):
        """set_timesteps should configure Karras sigmas."""
        from mlx_music.models.ace_step.scheduler import DPMPlusPlus2MKarrasScheduler

        scheduler = DPMPlusPlus2MKarrasScheduler()
        scheduler.set_timesteps(20)

        assert scheduler.num_inference_steps == 20
        assert scheduler.timesteps.shape == (20,)
        # Sigmas include final 0, so length is num_steps + 1
        assert scheduler.sigmas.shape == (21,)

    def test_karras_sigmas_decreasing(self):
        """Karras sigmas should be decreasing."""
        from mlx_music.models.ace_step.scheduler import DPMPlusPlus2MKarrasScheduler

        scheduler = DPMPlusPlus2MKarrasScheduler()
        scheduler.set_timesteps(10)
        mx.synchronize()

        sigmas = scheduler.sigmas.tolist()
        for i in range(len(sigmas) - 1):
            assert sigmas[i] >= sigmas[i + 1]

    def test_karras_sigmas_range(self):
        """Karras sigmas should be within min/max range."""
        from mlx_music.models.ace_step.scheduler import DPMPlusPlus2MKarrasScheduler

        scheduler = DPMPlusPlus2MKarrasScheduler(sigma_min=0.01, sigma_max=100.0)
        scheduler.set_timesteps(10)
        mx.synchronize()

        sigmas = scheduler.sigmas.tolist()
        # First sigma should be near sigma_max
        assert sigmas[0] <= 100.0
        # Last non-zero sigma should be near sigma_min (allow floating point tolerance)
        assert sigmas[-2] >= 0.01 - 1e-6
        # Final sigma should be 0
        assert sigmas[-1] == 0.0

    def test_step_first_uses_euler(self):
        """First step should use Euler (no previous derivative)."""
        from mlx_music.models.ace_step.scheduler import DPMPlusPlus2MKarrasScheduler

        scheduler = DPMPlusPlus2MKarrasScheduler()
        scheduler.set_timesteps(10)

        model_output = mx.random.normal((2, 8, 100))
        sample = mx.random.normal((2, 8, 100))
        timestep = scheduler.timesteps[0]

        output = scheduler.step(model_output, timestep, sample)

        assert output.prev_sample.shape == sample.shape
        assert scheduler._prev_derivative is not None

    def test_step_subsequent_uses_multistep(self):
        """Subsequent steps should use 2nd order correction."""
        from mlx_music.models.ace_step.scheduler import DPMPlusPlus2MKarrasScheduler

        scheduler = DPMPlusPlus2MKarrasScheduler()
        scheduler.set_timesteps(10)

        sample = mx.random.normal((2, 8, 100))

        # First step
        model_output1 = mx.random.normal((2, 8, 100))
        output1 = scheduler.step(model_output1, scheduler.timesteps[0], sample)

        # Second step should use multistep
        model_output2 = mx.random.normal((2, 8, 100))
        output2 = scheduler.step(model_output2, scheduler.timesteps[1], output1.prev_sample)

        assert output2.prev_sample.shape == sample.shape

    def test_reset_clears_state(self):
        """reset() should clear previous derivative."""
        from mlx_music.models.ace_step.scheduler import DPMPlusPlus2MKarrasScheduler

        scheduler = DPMPlusPlus2MKarrasScheduler()
        scheduler.set_timesteps(10)

        # Do a step to set state
        model_output = mx.random.normal((2, 8, 100))
        sample = mx.random.normal((2, 8, 100))
        scheduler.step(model_output, scheduler.timesteps[0], sample)

        assert scheduler._prev_derivative is not None

        # Reset
        scheduler.reset()

        assert scheduler._prev_derivative is None
        assert scheduler._step_index == 0

    def test_pred_original_sample_returned(self):
        """DPM++ should return predicted original sample."""
        from mlx_music.models.ace_step.scheduler import DPMPlusPlus2MKarrasScheduler

        scheduler = DPMPlusPlus2MKarrasScheduler()
        scheduler.set_timesteps(10)

        model_output = mx.random.normal((2, 8, 100))
        sample = mx.random.normal((2, 8, 100))
        timestep = scheduler.timesteps[0]

        output = scheduler.step(model_output, timestep, sample)

        assert output.pred_original_sample is not None
        assert output.pred_original_sample.shape == sample.shape

    def test_full_denoising_loop(self):
        """Complete denoising loop should produce valid output."""
        from mlx_music.models.ace_step.scheduler import DPMPlusPlus2MKarrasScheduler

        scheduler = DPMPlusPlus2MKarrasScheduler()
        scheduler.set_timesteps(20)

        # Start with noise
        sample = mx.random.normal((1, 8, 100))

        for t in scheduler.timesteps:
            model_output = mx.random.normal(sample.shape)
            output = scheduler.step(model_output, t, sample)
            sample = output.prev_sample

        mx.synchronize()
        assert sample.shape == (1, 8, 100)
        assert not mx.any(mx.isnan(sample))


class TestRetrieveTimesteps:
    """Tests for retrieve_timesteps function."""

    def test_retrieve_timesteps_euler(self):
        """retrieve_timesteps should work with Euler scheduler."""
        from mlx_music.models.ace_step.scheduler import (
            FlowMatchEulerDiscreteScheduler,
            retrieve_timesteps,
        )

        scheduler = FlowMatchEulerDiscreteScheduler()
        timesteps, num_steps = retrieve_timesteps(scheduler, 15)

        assert num_steps == 15
        assert timesteps.shape == (15,)

    def test_retrieve_timesteps_dpm(self):
        """retrieve_timesteps should work with DPM++ scheduler."""
        from mlx_music.models.ace_step.scheduler import (
            DPMPlusPlus2MKarrasScheduler,
            retrieve_timesteps,
        )

        scheduler = DPMPlusPlus2MKarrasScheduler()
        timesteps, num_steps = retrieve_timesteps(scheduler, 25)

        assert num_steps == 25
        assert timesteps.shape == (25,)


class TestGetScheduler:
    """Tests for get_scheduler factory function."""

    def test_get_scheduler_euler(self):
        """get_scheduler('euler') should return Euler scheduler."""
        from mlx_music.models.ace_step.scheduler import (
            get_scheduler,
            FlowMatchEulerDiscreteScheduler,
        )

        scheduler = get_scheduler("euler")
        assert isinstance(scheduler, FlowMatchEulerDiscreteScheduler)

    def test_get_scheduler_heun(self):
        """get_scheduler('heun') should return Heun scheduler."""
        from mlx_music.models.ace_step.scheduler import (
            get_scheduler,
            FlowMatchHeunDiscreteScheduler,
        )

        scheduler = get_scheduler("heun")
        assert isinstance(scheduler, FlowMatchHeunDiscreteScheduler)

    def test_get_scheduler_dpm(self):
        """get_scheduler('dpm++') should return DPM++ scheduler."""
        from mlx_music.models.ace_step.scheduler import (
            get_scheduler,
            DPMPlusPlus2MKarrasScheduler,
        )

        scheduler = get_scheduler("dpm++")
        assert isinstance(scheduler, DPMPlusPlus2MKarrasScheduler)

    def test_get_scheduler_dpm_karras(self):
        """get_scheduler('dpm++_karras') should return DPM++ scheduler."""
        from mlx_music.models.ace_step.scheduler import (
            get_scheduler,
            DPMPlusPlus2MKarrasScheduler,
        )

        scheduler = get_scheduler("dpm++_karras")
        assert isinstance(scheduler, DPMPlusPlus2MKarrasScheduler)

    def test_get_scheduler_case_insensitive(self):
        """get_scheduler should be case-insensitive."""
        from mlx_music.models.ace_step.scheduler import (
            get_scheduler,
            FlowMatchEulerDiscreteScheduler,
        )

        scheduler = get_scheduler("EULER")
        assert isinstance(scheduler, FlowMatchEulerDiscreteScheduler)

    def test_get_scheduler_invalid_raises(self):
        """get_scheduler with invalid type should raise ValueError."""
        from mlx_music.models.ace_step.scheduler import get_scheduler

        with pytest.raises(ValueError, match="Unknown scheduler"):
            get_scheduler("invalid_scheduler")

    def test_get_scheduler_custom_params(self):
        """get_scheduler should accept custom parameters."""
        from mlx_music.models.ace_step.scheduler import get_scheduler

        scheduler = get_scheduler(
            "dpm++",
            num_train_timesteps=500,
            sigma_min=0.01,
            sigma_max=50.0,
            rho=5.0,
        )

        assert scheduler.num_train_timesteps == 500
        assert scheduler.sigma_min == 0.01
        assert scheduler.sigma_max == 50.0
        assert scheduler.rho == 5.0


class TestSchedulerNumericalStability:
    """Tests for numerical stability of schedulers."""

    def test_euler_no_nan_in_loop(self):
        """Euler scheduler should not produce NaN in loop."""
        from mlx_music.models.ace_step.scheduler import FlowMatchEulerDiscreteScheduler

        scheduler = FlowMatchEulerDiscreteScheduler()
        scheduler.set_timesteps(50)

        sample = mx.random.normal((1, 8, 100))

        for t in scheduler.timesteps:
            model_output = mx.random.normal(sample.shape)
            output = scheduler.step(model_output, t, sample)
            sample = output.prev_sample

            assert not mx.any(mx.isnan(sample)), f"NaN at timestep {t}"

    def test_dpm_no_nan_in_loop(self):
        """DPM++ scheduler should not produce NaN in loop."""
        from mlx_music.models.ace_step.scheduler import DPMPlusPlus2MKarrasScheduler

        scheduler = DPMPlusPlus2MKarrasScheduler()
        scheduler.set_timesteps(50)

        sample = mx.random.normal((1, 8, 100))

        for t in scheduler.timesteps:
            model_output = mx.random.normal(sample.shape)
            output = scheduler.step(model_output, t, sample)
            sample = output.prev_sample

            assert not mx.any(mx.isnan(sample)), f"NaN at timestep {t}"

    def test_dpm_no_inf_in_coefficients(self):
        """DPM++ should not have infinite coefficients."""
        from mlx_music.models.ace_step.scheduler import DPMPlusPlus2MKarrasScheduler

        scheduler = DPMPlusPlus2MKarrasScheduler()
        scheduler.set_timesteps(20)

        sample = mx.random.normal((1, 8, 100))

        for i, t in enumerate(scheduler.timesteps):
            model_output = mx.random.normal(sample.shape)
            output = scheduler.step(model_output, t, sample)
            sample = output.prev_sample

            assert not mx.any(mx.isinf(sample)), f"Inf at step {i}"
