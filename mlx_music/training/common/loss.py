"""
Loss Functions for mlx-music Training.

Provides flow matching and v-prediction loss functions for diffusion model training.

Flow Matching Loss (used by ACE-Step):
- Interpolates between clean audio and noise at timestep t
- Trains model to predict velocity (direction from noise to clean)
- Loss = ||predicted_velocity - target_velocity||²

V-Prediction Loss (used by Stable Audio, EDM-style):
- Uses sigma-weighted v-prediction parameterization
- Target: v = sigma * clean - (1 - sigma) * noise
- Better gradient scaling across noise levels
"""

import mlx.core as mx


def flow_matching_loss(
    model_output: mx.array,
    clean: mx.array,
    noise: mx.array,
    reduction: str = "mean",
) -> mx.array:
    """
    Compute flow matching loss for diffusion training.

    Flow matching trains the model to predict the velocity field from
    noise to clean audio. The target velocity is simply: v = noise - clean

    This formulation is used by ACE-Step and similar rectified flow models.

    Args:
        model_output: Predicted velocity from the model [B, C, T] or [C, T]
        clean: Clean (denoised) audio latents [B, C, T] or [C, T]
        noise: Pure noise [B, C, T] or [C, T]
        reduction: How to reduce the loss ("mean", "sum", or "none")

    Returns:
        Loss value (scalar if reduction != "none")

    Example:
        # In training loop:
        sigma = scheduler.sigmas[t]
        latents_t = (1 - sigma) * clean + sigma * noise  # Interpolate
        predicted_v = model(latents_t, t, ...)  # Forward pass
        loss = flow_matching_loss(
            model_output=predicted_v,
            clean=clean,
            noise=noise,
        )
    """
    # Target velocity: direction from clean to noise (or noise - clean depending on formulation)
    # For rectified flow: target_v = noise - clean (model learns to predict noise direction)
    target_velocity = noise - clean

    # Compute squared error
    residual = model_output - target_velocity
    loss = residual.square()

    if reduction == "mean":
        return loss.mean()
    elif reduction == "sum":
        return loss.sum()
    elif reduction == "none":
        return loss
    else:
        raise ValueError(f"Unknown reduction: {reduction}. Use 'mean', 'sum', or 'none'.")


def v_prediction_loss(
    model_output: mx.array,
    clean: mx.array,
    noise: mx.array,
    sigma: mx.array | float,
    reduction: str = "mean",
) -> mx.array:
    """
    Compute v-prediction loss for EDM-style diffusion training.

    V-prediction parameterization provides better gradient scaling across
    noise levels compared to epsilon-prediction. The target is:
        v = sigma * clean - (1 - sigma) * noise

    This formulation is used by Stable Audio and similar EDM-style models.

    Args:
        model_output: Predicted v from the model [B, C, T] or [C, T]
        clean: Clean (denoised) audio latents [B, C, T] or [C, T]
        noise: Pure noise [B, C, T] or [C, T]
        sigma: Noise level (scalar or broadcastable array)
        reduction: How to reduce the loss ("mean", "sum", or "none")

    Returns:
        Loss value (scalar if reduction != "none")

    Example:
        # In training loop:
        sigma = scheduler.get_sigma(t)
        latents_t = clean + sigma * noise  # EDM noise schedule
        predicted_v = model(latents_t, sigma, ...)
        loss = v_prediction_loss(
            model_output=predicted_v,
            clean=clean,
            noise=noise,
            sigma=sigma,
        )
    """
    # Handle sigma shape for broadcasting
    if isinstance(sigma, (int, float)):
        sigma = mx.array(sigma)

    # Ensure sigma can broadcast with audio shape
    while sigma.ndim < clean.ndim:
        sigma = sigma[..., None]

    # V-prediction target: v = sigma * clean - (1 - sigma) * noise
    # This parameterization provides better gradient scaling
    target_v = sigma * clean - (1 - sigma) * noise

    # Compute squared error
    residual = model_output - target_v
    loss = residual.square()

    if reduction == "mean":
        return loss.mean()
    elif reduction == "sum":
        return loss.sum()
    elif reduction == "none":
        return loss
    else:
        raise ValueError(f"Unknown reduction: {reduction}. Use 'mean', 'sum', or 'none'.")


def snr_weighted_loss(
    loss: mx.array,
    sigma: mx.array | float,
    snr_gamma: float = 5.0,
) -> mx.array:
    """
    Apply signal-to-noise ratio (SNR) weighting to loss.

    SNR weighting reduces the contribution of very noisy or very clean
    samples, focusing learning on intermediate noise levels where the
    model can learn most effectively.

    Weight = (1 + SNR)^(-gamma) where SNR = 1/sigma^2

    Args:
        loss: Per-sample loss values [B] or scalar
        sigma: Noise level(s) matching loss shape
        snr_gamma: Weighting exponent (default: 5.0)
            Higher values = more aggressive downweighting of extreme sigmas

    Returns:
        Weighted loss (same shape as input loss)

    Reference:
        "Perception Prioritized Training of Diffusion Models" (Choi et al., 2022)
    """
    if isinstance(sigma, (int, float)):
        sigma = mx.array(sigma)

    # SNR = 1 / sigma^2
    snr = 1.0 / (sigma.square() + 1e-8)

    # Weight = (1 + SNR)^(-gamma)
    weight = (1.0 + snr) ** (-snr_gamma)

    return loss * weight


def min_snr_weighted_loss(
    loss: mx.array,
    sigma: mx.array | float,
    min_snr_gamma: float = 5.0,
) -> mx.array:
    """
    Apply min-SNR weighting to loss.

    Min-SNR weighting clips the SNR weight to prevent very high weights
    at low noise levels. This provides more stable training.

    Weight = min(SNR, gamma) / SNR

    Args:
        loss: Per-sample loss values [B] or scalar
        sigma: Noise level(s) matching loss shape
        min_snr_gamma: Minimum SNR clipping value (default: 5.0)

    Returns:
        Weighted loss (same shape as input loss)

    Reference:
        "Efficient Diffusion Training via Min-SNR Weighting Strategy" (Hang et al., 2023)
    """
    if isinstance(sigma, (int, float)):
        sigma = mx.array(sigma)

    # SNR = 1 / sigma^2
    snr = 1.0 / (sigma.square() + 1e-8)

    # Weight = min(SNR, gamma) / SNR
    weight = mx.minimum(snr, min_snr_gamma) / snr

    return loss * weight
