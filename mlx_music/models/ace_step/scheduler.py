"""
Flow Matching Scheduler for ACE-Step.

Implements the Euler, Heun, and PingPong schedulers
for flow matching diffusion.
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple

import mlx.core as mx


@dataclass
class SchedulerOutput:
    """Output from a scheduler step."""

    prev_sample: mx.array
    pred_original_sample: Optional[mx.array] = None


class FlowMatchEulerDiscreteScheduler:
    """
    Flow Match Euler Discrete Scheduler.

    Implements the flow matching scheduler with Euler method
    for solving the ODE: dx/dt = v(x, t)
    """

    def __init__(
        self,
        num_train_timesteps: int = 1000,
        shift: float = 3.0,
        sigma_max: float = 1.0,
    ):
        self.num_train_timesteps = num_train_timesteps
        self.shift = shift
        self.sigma_max = sigma_max

        # Initialize timesteps
        self.timesteps: Optional[mx.array] = None
        self.sigmas: Optional[mx.array] = None

        # Set default
        self.set_timesteps(50)

    def set_timesteps(
        self,
        num_inference_steps: int,
        device: Optional[str] = None,
    ):
        """
        Set the timesteps for inference.

        Args:
            num_inference_steps: Number of denoising steps
            device: Ignored (MLX handles device automatically)
        """
        # Linear timesteps from 1 to near 0
        timesteps = mx.linspace(1.0, 1.0 / self.num_train_timesteps, num_inference_steps)

        # Apply shift transformation
        sigmas = self.shift * timesteps / (1 + (self.shift - 1) * timesteps)

        self.timesteps = timesteps * self.num_train_timesteps
        self.sigmas = sigmas
        self.num_inference_steps = num_inference_steps

    def scale_noise(
        self,
        sample: mx.array,
        timestep: mx.array,
        noise: mx.array,
    ) -> mx.array:
        """
        Scale and add noise to sample based on timestep.

        For flow matching: x_t = (1 - sigma) * x_0 + sigma * noise
        """
        # Get sigma for this timestep
        step_idx = mx.argmin(mx.abs(self.timesteps - timestep))
        sigma = self.sigmas[step_idx]

        # Expand sigma for broadcasting
        while sigma.ndim < sample.ndim:
            sigma = sigma[..., None]

        noisy_sample = (1 - sigma) * sample + sigma * noise
        return noisy_sample

    def step(
        self,
        model_output: mx.array,
        timestep: mx.array,
        sample: mx.array,
        return_dict: bool = True,
    ) -> SchedulerOutput:
        """
        Perform one denoising step.

        For flow matching with Euler:
        x_{t-1} = x_t + (sigma_{t-1} - sigma_t) * v(x_t, t)

        Args:
            model_output: Predicted velocity from the model
            timestep: Current timestep
            sample: Current noisy sample
            return_dict: Whether to return SchedulerOutput

        Returns:
            SchedulerOutput with denoised sample
        """
        # Find current step index
        step_idx = mx.argmin(mx.abs(self.timesteps - timestep))

        # Get current and previous sigma
        sigma = self.sigmas[step_idx]

        if step_idx + 1 < self.num_inference_steps:
            sigma_prev = self.sigmas[step_idx + 1]
        else:
            sigma_prev = mx.array(0.0)

        # Expand for broadcasting
        while sigma.ndim < sample.ndim:
            sigma = sigma[..., None]
            sigma_prev = sigma_prev[..., None]

        # Euler step: x_{t-1} = x_t + (sigma_{t-1} - sigma_t) * velocity
        dt = sigma_prev - sigma
        prev_sample = sample + dt * model_output

        if return_dict:
            return SchedulerOutput(prev_sample=prev_sample)
        return (prev_sample,)


class FlowMatchHeunDiscreteScheduler:
    """
    Flow Match Heun Discrete Scheduler.

    Uses Heun's method (improved Euler) for more accurate
    ODE solving at the cost of 2x model evaluations.
    """

    def __init__(
        self,
        num_train_timesteps: int = 1000,
        shift: float = 3.0,
        sigma_max: float = 1.0,
    ):
        self.num_train_timesteps = num_train_timesteps
        self.shift = shift
        self.sigma_max = sigma_max

        self.timesteps: Optional[mx.array] = None
        self.sigmas: Optional[mx.array] = None

        self.set_timesteps(50)

    def set_timesteps(self, num_inference_steps: int, device: Optional[str] = None):
        """Set timesteps for inference."""
        timesteps = mx.linspace(1.0, 1.0 / self.num_train_timesteps, num_inference_steps)
        sigmas = self.shift * timesteps / (1 + (self.shift - 1) * timesteps)

        self.timesteps = timesteps * self.num_train_timesteps
        self.sigmas = sigmas
        self.num_inference_steps = num_inference_steps

    def scale_noise(
        self,
        sample: mx.array,
        timestep: mx.array,
        noise: mx.array,
    ) -> mx.array:
        """Scale and add noise to sample."""
        step_idx = mx.argmin(mx.abs(self.timesteps - timestep))
        sigma = self.sigmas[step_idx]

        while sigma.ndim < sample.ndim:
            sigma = sigma[..., None]

        return (1 - sigma) * sample + sigma * noise

    def step(
        self,
        model_output: mx.array,
        timestep: mx.array,
        sample: mx.array,
        model_fn: Optional[callable] = None,
        return_dict: bool = True,
    ) -> SchedulerOutput:
        """
        Perform one Heun step.

        Heun's method:
        1. k1 = v(x_t, t)
        2. x_euler = x_t + dt * k1
        3. k2 = v(x_euler, t-1)
        4. x_{t-1} = x_t + dt * (k1 + k2) / 2

        Note: Requires model_fn for the second evaluation.
        If model_fn is None, falls back to Euler.
        """
        step_idx = mx.argmin(mx.abs(self.timesteps - timestep))

        sigma = self.sigmas[step_idx]
        if step_idx + 1 < self.num_inference_steps:
            sigma_prev = self.sigmas[step_idx + 1]
        else:
            sigma_prev = mx.array(0.0)

        while sigma.ndim < sample.ndim:
            sigma = sigma[..., None]
            sigma_prev = sigma_prev[..., None]

        dt = sigma_prev - sigma

        # First step (Euler prediction)
        k1 = model_output
        x_euler = sample + dt * k1

        # If no model function provided, use Euler
        if model_fn is None:
            if return_dict:
                return SchedulerOutput(prev_sample=x_euler)
            return (x_euler,)

        # Second evaluation at predicted point
        if step_idx + 1 < self.num_inference_steps:
            next_timestep = self.timesteps[step_idx + 1]
        else:
            next_timestep = mx.array(0.0)

        k2 = model_fn(x_euler, next_timestep)

        # Heun combination
        prev_sample = sample + dt * (k1 + k2) / 2

        if return_dict:
            return SchedulerOutput(prev_sample=prev_sample)
        return (prev_sample,)


class DPMPlusPlus2MKarrasScheduler:
    """
    DPM++ 2M Scheduler with Karras noise schedule.

    DPM-Solver++ is a second-order multistep solver that achieves
    better quality with fewer steps than Euler methods. Combined with
    the Karras noise schedule, it provides excellent results at 20-30 steps.

    Reference:
    - DPM-Solver++: Lu et al. "DPM-Solver++: Fast Solver for Guided Sampling
      of Diffusion Probabilistic Models"
    - Karras schedule: Karras et al. "Elucidating the Design Space of
      Diffusion-Based Generative Models"
    """

    # Epsilon for numerical stability in coefficient calculations
    EPSILON: float = 1e-8

    def __init__(
        self,
        num_train_timesteps: int = 1000,
        sigma_min: float = 0.002,
        sigma_max: float = 80.0,
        rho: float = 7.0,
        shift: float = 3.0,
    ):
        """
        Initialize DPM++ 2M Karras scheduler.

        Args:
            num_train_timesteps: Number of training timesteps
            sigma_min: Minimum sigma value
            sigma_max: Maximum sigma value
            rho: Karras schedule exponent (higher = more steps at low noise)
            shift: Flow matching shift parameter
        """
        self.num_train_timesteps = num_train_timesteps
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.rho = rho
        self.shift = shift

        self.timesteps: Optional[mx.array] = None
        self.sigmas: Optional[mx.array] = None
        self.num_inference_steps: int = 0

        # State for multistep
        self._prev_derivative: Optional[mx.array] = None
        self._step_index: int = 0

        # Set default
        self.set_timesteps(50)

    def _get_karras_sigmas(self, n_steps: int) -> mx.array:
        """
        Generate Karras noise schedule.

        The Karras schedule concentrates more steps at lower noise levels
        where the model needs more precision.
        """
        # Karras schedule: sigma_i = (sigma_max^(1/rho) + i/(n-1) * (sigma_min^(1/rho) - sigma_max^(1/rho)))^rho
        ramp = mx.linspace(0, 1, n_steps)
        min_inv_rho = self.sigma_min ** (1 / self.rho)
        max_inv_rho = self.sigma_max ** (1 / self.rho)

        sigmas = (max_inv_rho + ramp * (min_inv_rho - max_inv_rho)) ** self.rho

        # Append 0 for the final step
        sigmas = mx.concatenate([sigmas, mx.array([0.0])])

        return sigmas

    def set_timesteps(
        self,
        num_inference_steps: int,
        device: Optional[str] = None,
    ):
        """
        Set the timesteps for inference.

        Args:
            num_inference_steps: Number of denoising steps
            device: Ignored (MLX handles device automatically)
        """
        self.num_inference_steps = num_inference_steps

        # Generate Karras sigmas
        self.sigmas = self._get_karras_sigmas(num_inference_steps)

        # Convert sigmas to timesteps (for compatibility)
        # Map sigma range to timestep range
        timesteps = (self.sigmas[:-1] / self.sigma_max) * self.num_train_timesteps
        self.timesteps = timesteps

        # Reset state
        self._prev_derivative = None
        self._step_index = 0

    def scale_noise(
        self,
        sample: mx.array,
        timestep: mx.array,
        noise: mx.array,
    ) -> mx.array:
        """
        Scale and add noise to sample based on timestep.

        Uses the continuous formulation: x_t = sample + sigma * noise

        Note: This method uses O(n) argmin lookup because it's called with
        arbitrary timesteps during training, not sequential steps like step().
        For inference, step() uses O(1) index tracking instead.
        """
        # O(n) lookup is acceptable here - called once per training sample,
        # not in the tight denoising loop like step()
        step_idx = mx.argmin(mx.abs(self.timesteps - timestep))
        sigma = self.sigmas[step_idx]

        while sigma.ndim < sample.ndim:
            sigma = sigma[..., None]

        # For Karras-style: x_t = x_0 + sigma * noise
        noisy_sample = sample + sigma * noise
        return noisy_sample

    def step(
        self,
        model_output: mx.array,
        timestep: mx.array,
        sample: mx.array,
        return_dict: bool = True,
    ) -> SchedulerOutput:
        """
        Perform one DPM++ 2M denoising step.

        DPM++ 2M uses a second-order correction based on the previous
        derivative estimate for more accurate ODE solving.

        Args:
            model_output: Predicted velocity/noise from the model
            timestep: Current timestep
            sample: Current noisy sample
            return_dict: Whether to return SchedulerOutput

        Returns:
            SchedulerOutput with denoised sample
        """
        # Use tracked step index for O(1) lookup instead of O(n) argmin search
        step_idx = self._step_index

        sigma = self.sigmas[step_idx]
        sigma_next = self.sigmas[step_idx + 1]

        # Expand for broadcasting
        while sigma.ndim < sample.ndim:
            sigma = sigma[..., None]
            sigma_next = sigma_next[..., None]

        # Convert model output to derivative estimate
        # For flow matching velocity prediction: derivative = model_output
        derivative = model_output

        # DPM++ 2M update
        if self._prev_derivative is None or step_idx == 0:
            # First step: use Euler method
            # x_{t+1} = x_t + (sigma_{t+1} - sigma_t) * derivative
            prev_sample = sample + (sigma_next - sigma) * derivative
        else:
            # Second-order correction using previous derivative
            # This is the "2M" (multistep) part of DPM++ 2M

            # Get previous sigma for coefficient calculation
            sigma_prev = self.sigmas[step_idx - 1]
            while sigma_prev.ndim < sample.ndim:
                sigma_prev = sigma_prev[..., None]

            # Calculate coefficients for 2nd order correction
            # h = sigma_next - sigma (current step size)
            # h_prev = sigma - sigma_prev (previous step size)
            h = sigma_next - sigma
            h_prev = sigma - sigma_prev

            # Coefficient for 2nd order correction
            # r = h / h_prev
            # Use mx.maximum to ensure denominator is at least EPSILON (prevents div by zero)
            h_prev_safe = mx.maximum(mx.abs(h_prev), self.EPSILON) * mx.sign(h_prev + self.EPSILON)
            r = h / h_prev_safe

            # DPM++ 2M formula:
            # x_{t+1} = x_t + h * ((1 + 0.5/r) * derivative - 0.5/r * prev_derivative)
            r_safe = mx.maximum(mx.abs(r), self.EPSILON) * mx.sign(r + self.EPSILON)
            coeff_cur = 1.0 + 0.5 / r_safe
            coeff_prev = 0.5 / r_safe

            corrected_derivative = coeff_cur * derivative - coeff_prev * self._prev_derivative
            prev_sample = sample + h * corrected_derivative

        # Store derivative and increment step index for next step
        self._prev_derivative = derivative
        self._step_index = step_idx + 1

        # Optionally compute predicted original sample (x0 prediction)
        # For flow matching: x0 = sample - sigma * derivative
        pred_original_sample = sample - sigma * derivative

        if return_dict:
            return SchedulerOutput(
                prev_sample=prev_sample,
                pred_original_sample=pred_original_sample,
            )
        return (prev_sample,)

    def reset(self):
        """Reset scheduler state for a new generation."""
        self._prev_derivative = None
        self._step_index = 0


def retrieve_timesteps(
    scheduler,
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


def get_scheduler(
    scheduler_type: str = "euler",
    num_train_timesteps: int = 1000,
    shift: float = 3.0,
    sigma_min: float = 0.002,
    sigma_max: float = 80.0,
    rho: float = 7.0,
):
    """
    Get scheduler by type.

    Args:
        scheduler_type: "euler", "heun", "dpm++", or "dpm++_karras"
        num_train_timesteps: Number of training timesteps
        shift: Shift parameter for flow matching
        sigma_min: Minimum sigma for DPM++ (default: 0.002)
        sigma_max: Maximum sigma for DPM++ (default: 80.0)
        rho: Karras schedule exponent for DPM++ (default: 7.0)

    Returns:
        Scheduler instance

    Scheduler comparison:
        - euler: Fast, simple, good for 50+ steps
        - heun: Higher quality, 2x model evals per step
        - dpm++/dpm++_karras: Best quality at 20-30 steps, recommended
    """
    scheduler_type = scheduler_type.lower()

    if scheduler_type == "euler":
        return FlowMatchEulerDiscreteScheduler(
            num_train_timesteps=num_train_timesteps,
            shift=shift,
        )
    elif scheduler_type == "heun":
        return FlowMatchHeunDiscreteScheduler(
            num_train_timesteps=num_train_timesteps,
            shift=shift,
        )
    elif scheduler_type in ("dpm++", "dpm++_karras", "dpmpp", "dpmpp_karras"):
        return DPMPlusPlus2MKarrasScheduler(
            num_train_timesteps=num_train_timesteps,
            sigma_min=sigma_min,
            sigma_max=sigma_max,
            rho=rho,
            shift=shift,
        )
    else:
        raise ValueError(
            f"Unknown scheduler type: {scheduler_type}. "
            f"Available: euler, heun, dpm++, dpm++_karras"
        )
