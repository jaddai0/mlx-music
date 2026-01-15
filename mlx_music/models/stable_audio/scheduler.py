"""
EDM DPM-Solver Multistep Scheduler for Stable Audio Open.

Implements the EDM (Elucidating the Design Space of Diffusion Models)
formulation with DPM-Solver++ as the solver.
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

import mlx.core as mx
import numpy as np

from mlx_music.models.stable_audio.config import EDMSchedulerConfig


@dataclass
class SchedulerOutput:
    """Output from a scheduler step."""

    prev_sample: mx.array
    pred_original_sample: Optional[mx.array] = None


class EDMDPMSolverMultistepScheduler:
    """
    EDM DPM-Solver Multistep Scheduler.

    Implements DPM-Solver++ with EDM formulation for Stable Audio Open.
    Uses Karras sigmas and v-prediction parameterization.

    Reference: https://arxiv.org/abs/2206.00364 (DPM-Solver++)
    Reference: https://arxiv.org/abs/2206.00364 (EDM)
    """

    def __init__(
        self,
        sigma_min: float = 0.3,
        sigma_max: float = 500.0,
        sigma_data: float = 1.0,
        rho: float = 7.0,
        solver_order: int = 2,
        prediction_type: str = "v_prediction",
        algorithm_type: str = "dpmsolver++",
    ):
        """
        Initialize scheduler.

        Args:
            sigma_min: Minimum sigma (noise level)
            sigma_max: Maximum sigma (noise level)
            sigma_data: Data standard deviation for EDM scaling
            rho: Karras rho parameter for sigma schedule
            solver_order: DPM-Solver order (1 or 2)
            prediction_type: Model prediction type ("epsilon", "v_prediction")
            algorithm_type: Solver algorithm ("dpmsolver++")
        """
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.sigma_data = sigma_data
        self.rho = rho
        self.solver_order = solver_order
        self.prediction_type = prediction_type
        self.algorithm_type = algorithm_type

        # Will be set by set_timesteps
        self.sigmas: Optional[mx.array] = None
        self.timesteps: Optional[mx.array] = None
        self.num_inference_steps: int = 0

        # For multistep solvers, store previous outputs
        self.model_outputs: List[mx.array] = []
        self.lower_order_nums: int = 0

        # Initialize with default steps
        self.set_timesteps(100)

    @classmethod
    def from_config(cls, config: EDMSchedulerConfig) -> "EDMDPMSolverMultistepScheduler":
        """Create scheduler from config."""
        return cls(
            sigma_min=config.sigma_min,
            sigma_max=config.sigma_max,
            sigma_data=config.sigma_data,
            rho=config.rho,
            solver_order=config.solver_order,
            prediction_type=config.prediction_type,
            algorithm_type=config.algorithm_type,
        )

    def _get_karras_sigmas(self, num_steps: int) -> mx.array:
        """
        Compute Karras sigmas schedule.

        Formula: sigma_i = (sigma_max^(1/rho) + i/(n-1) * (sigma_min^(1/rho) - sigma_max^(1/rho)))^rho

        This creates a schedule where sigmas decrease faster at high noise levels
        and slower at low noise levels, which is more efficient for sampling.
        """
        ramp = mx.linspace(0, 1, num_steps + 1)

        min_inv_rho = self.sigma_min ** (1 / self.rho)
        max_inv_rho = self.sigma_max ** (1 / self.rho)

        sigmas = (max_inv_rho + ramp * (min_inv_rho - max_inv_rho)) ** self.rho

        return sigmas

    def set_timesteps(self, num_inference_steps: int, device: Optional[str] = None):
        """
        Set the timesteps/sigmas for inference.

        Args:
            num_inference_steps: Number of denoising steps
            device: Ignored (MLX handles device automatically)
        """
        self.num_inference_steps = num_inference_steps

        # Get Karras sigmas
        sigmas = self._get_karras_sigmas(num_inference_steps)

        # For EDM, timesteps are the same as sigmas
        self.timesteps = sigmas[:-1]  # Exclude final sigma=0
        self.sigmas = sigmas

        # Reset solver state
        self.model_outputs = []
        self.lower_order_nums = 0

    def _sigma_to_t(self, sigma: mx.array) -> mx.array:
        """
        Convert sigma to timestep.

        For EDM, t = sigma (they are the same)
        """
        return sigma

    def _get_scalings(self, sigma: mx.array) -> Tuple[mx.array, mx.array, mx.array]:
        """
        Get EDM scalings for a given sigma.

        Returns c_skip, c_out, c_in based on EDM formulation.
        """
        sigma_data = self.sigma_data

        c_skip = sigma_data**2 / (sigma**2 + sigma_data**2)
        c_out = sigma * sigma_data / mx.sqrt(sigma**2 + sigma_data**2)
        c_in = 1 / mx.sqrt(sigma**2 + sigma_data**2)

        return c_skip, c_out, c_in

    def scale_model_input(
        self,
        sample: mx.array,
        timestep: mx.array,
    ) -> mx.array:
        """
        Scale the model input based on current sigma.

        For EDM: x_in = x / sqrt(sigma^2 + sigma_data^2)
        """
        sigma = timestep
        _, _, c_in = self._get_scalings(sigma)

        # Expand for broadcasting
        while c_in.ndim < sample.ndim:
            c_in = c_in[..., None]

        return sample * c_in

    def _convert_model_output(
        self,
        model_output: mx.array,
        sample: mx.array,
        sigma: mx.array,
    ) -> mx.array:
        """
        Convert model output to predicted original sample (x_0).

        For v-prediction: x_0 = c_skip * x + c_out * v
        For epsilon: x_0 = (x - sigma * epsilon) / sigma_data
        """
        c_skip, c_out, _ = self._get_scalings(sigma)

        # Expand for broadcasting
        while c_skip.ndim < sample.ndim:
            c_skip = c_skip[..., None]
            c_out = c_out[..., None]

        if self.prediction_type == "v_prediction":
            # v-prediction: output is the velocity
            pred_original_sample = c_skip * sample + c_out * model_output
        elif self.prediction_type == "epsilon":
            # epsilon prediction: output is the noise
            sigma_expanded = sigma
            while sigma_expanded.ndim < sample.ndim:
                sigma_expanded = sigma_expanded[..., None]
            pred_original_sample = sample - sigma_expanded * model_output
        else:
            raise ValueError(f"Unknown prediction type: {self.prediction_type}")

        return pred_original_sample

    def step(
        self,
        model_output: mx.array,
        timestep: mx.array,
        sample: mx.array,
        return_dict: bool = True,
    ) -> SchedulerOutput:
        """
        Perform one DPM-Solver++ step.

        Args:
            model_output: Direct model output (v or epsilon)
            timestep: Current sigma/timestep
            sample: Current noisy sample
            return_dict: Whether to return SchedulerOutput

        Returns:
            SchedulerOutput with denoised sample
        """
        # Find current step index
        if self.timesteps is None or len(self.timesteps) == 0:
            raise RuntimeError(
                "Scheduler timesteps not initialized. Call set_timesteps() first."
            )
        step_idx = int(mx.argmin(mx.abs(self.timesteps - timestep)).item())

        # Get current and next sigma
        sigma = self.sigmas[step_idx]
        sigma_next = self.sigmas[step_idx + 1]

        # Convert model output to x_0 prediction
        pred_original_sample = self._convert_model_output(
            model_output, sample, sigma
        )

        # Store for multistep
        self.model_outputs.append(pred_original_sample)
        if len(self.model_outputs) > self.solver_order:
            self.model_outputs.pop(0)

        # Perform DPM-Solver++ step
        if self.algorithm_type == "dpmsolver++":
            prev_sample = self._dpm_solver_step(
                sample, sigma, sigma_next, pred_original_sample
            )
        else:
            raise ValueError(f"Unknown algorithm: {self.algorithm_type}")

        # Update lower order counter
        if self.lower_order_nums < self.solver_order:
            self.lower_order_nums += 1

        if return_dict:
            return SchedulerOutput(
                prev_sample=prev_sample,
                pred_original_sample=pred_original_sample,
            )
        return (prev_sample,)

    def _dpm_solver_step(
        self,
        sample: mx.array,
        sigma: mx.array,
        sigma_next: mx.array,
        pred_original_sample: mx.array,
    ) -> mx.array:
        """
        Perform DPM-Solver++ first or second order step.

        DPM-Solver++ in continuous time:
        - First order: x_{t-1} = x_0 + sigma_{t-1} * (x_t - x_0) / sigma_t
        - Second order uses linear extrapolation of denoised samples
        """
        # Compute log sigma ratio for DPM-Solver
        # Add epsilon for numerical stability when sigma approaches zero
        lambda_t = mx.log(mx.maximum(sigma, 1e-10))
        lambda_s = mx.log(mx.maximum(sigma_next, 1e-10))
        h = lambda_s - lambda_t

        # Expand for broadcasting
        while sigma.ndim < sample.ndim:
            sigma = sigma[..., None]
        while sigma_next.ndim < sample.ndim:
            sigma_next = sigma_next[..., None]
        while h.ndim < sample.ndim:
            h = h[..., None]

        if self.solver_order == 1 or len(self.model_outputs) == 1:
            # First order DPM-Solver++
            # x_{s} = sigma_s/sigma_t * x_t + (1 - sigma_s/sigma_t) * x_0
            ratio = sigma_next / sigma
            prev_sample = ratio * sample + (1 - ratio) * pred_original_sample
        else:
            # Second order DPM-Solver++
            # Use linear extrapolation of x_0 predictions
            x0_prev = self.model_outputs[-2]

            # Compute coefficients
            ratio = sigma_next / sigma
            # Linear extrapolation coefficient
            prev_sigma = self.sigmas[len(self.model_outputs) - 2]
            r = h / (lambda_t - mx.log(mx.maximum(prev_sigma, 1e-10)))
            while r.ndim < sample.ndim:
                r = r[..., None]

            # Second order correction
            D1 = pred_original_sample - x0_prev

            prev_sample = (
                ratio * sample
                + (1 - ratio) * pred_original_sample
                + (0.5 * (1 - ratio) * r) * D1
            )

        return prev_sample

    def add_noise(
        self,
        original_samples: mx.array,
        noise: mx.array,
        timesteps: mx.array,
    ) -> mx.array:
        """
        Add noise to samples for a given timestep.

        For EDM: x_t = x_0 + sigma * noise
        """
        sigma = timesteps
        while sigma.ndim < original_samples.ndim:
            sigma = sigma[..., None]

        noisy_samples = original_samples + sigma * noise
        return noisy_samples


def retrieve_timesteps(
    scheduler: EDMDPMSolverMultistepScheduler,
    num_inference_steps: int,
) -> Tuple[mx.array, int]:
    """
    Retrieve timesteps from scheduler.

    Args:
        scheduler: The scheduler instance
        num_inference_steps: Number of denoising steps

    Returns:
        Tuple of (timesteps array, num_inference_steps)
    """
    scheduler.set_timesteps(num_inference_steps)
    return scheduler.timesteps, num_inference_steps


__all__ = [
    "EDMDPMSolverMultistepScheduler",
    "SchedulerOutput",
    "retrieve_timesteps",
]
