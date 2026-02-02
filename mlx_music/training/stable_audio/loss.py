"""
V-Prediction Loss for Stable Audio Training.

Stable Audio uses EDM-style v-prediction training where:
- v = sigma * clean - (1 - sigma) * noise
- The model predicts v given noised input and sigma

V-prediction provides better gradient scaling across noise levels
compared to epsilon-prediction, especially for high-resolution audio.

Reference:
    "EDM: Elucidating the Design Space of Diffusion-Based Generative Models"
    (Karras et al., 2022)
"""

import random
from typing import Optional, Tuple

import mlx.core as mx

from mlx_music.training.common.loss import v_prediction_loss, min_snr_weighted_loss


class StableAudioLoss:
    """
    V-prediction loss computation for Stable Audio training.

    Uses EDM-style parameterization:
    - noised = clean + sigma * noise (different from flow matching!)
    - target_v = sigma * clean - (1 - sigma) * noise
    - Loss = ||predicted_v - target_v||²

    Optionally applies min-SNR weighting for stable training.
    """

    @staticmethod
    def sample_sigma(
        batch_size: int,
        sigma_min: float = 0.002,
        sigma_max: float = 80.0,
        rho: float = 7.0,
        rng: Optional[random.Random] = None,
    ) -> mx.array:
        """
        Sample sigma values from Karras noise schedule.

        Uses the inverse CDF of the Karras schedule for uniform sampling
        in the warped space.

        Args:
            batch_size: Number of samples
            sigma_min: Minimum sigma
            sigma_max: Maximum sigma
            rho: Schedule curvature (higher = more samples near sigma_min)
            rng: Random generator

        Returns:
            Sigma values [batch_size]
        """
        if rng is not None:
            seed = rng.randint(0, 2**31 - 1)
        else:
            seed = random.randint(0, 2**31 - 1)

        key = mx.random.key(seed)

        # Sample uniform in [0, 1]
        u = mx.random.uniform(shape=(batch_size,), key=key)

        # Apply inverse Karras schedule
        sigma_min_inv = sigma_min ** (1 / rho)
        sigma_max_inv = sigma_max ** (1 / rho)

        sigmas = (sigma_max_inv + u * (sigma_min_inv - sigma_max_inv)) ** rho

        return sigmas

    @staticmethod
    def add_noise_edm(
        clean: mx.array,
        noise: mx.array,
        sigma: mx.array | float,
    ) -> mx.array:
        """
        Add noise using EDM parameterization.

        Formula: noised = clean + sigma * noise

        This is different from flow matching which uses interpolation!

        Args:
            clean: Clean audio latents
            noise: Gaussian noise
            sigma: Noise level(s)

        Returns:
            Noised latents
        """
        if isinstance(sigma, (int, float)):
            sigma = mx.array(sigma)

        while sigma.ndim < clean.ndim:
            sigma = sigma[..., None]

        return clean + sigma * noise

    @staticmethod
    def compute_target_v(
        clean: mx.array,
        noise: mx.array,
        sigma: mx.array | float,
    ) -> mx.array:
        """
        Compute v-prediction target.

        Formula: v = sigma * clean - (1 - sigma) * noise

        This parameterization provides better gradient scaling.

        Args:
            clean: Clean audio latents
            noise: Gaussian noise
            sigma: Noise level(s)

        Returns:
            Target v for training
        """
        if isinstance(sigma, (int, float)):
            sigma = mx.array(sigma)

        while sigma.ndim < clean.ndim:
            sigma = sigma[..., None]

        return sigma * clean - (1 - sigma) * noise

    @staticmethod
    def compute_loss(
        predicted: mx.array,
        target: mx.array,
        sigma: Optional[mx.array] = None,
        use_snr_weighting: bool = False,
        snr_gamma: float = 5.0,
        reduction: str = "mean",
    ) -> mx.array:
        """
        Compute v-prediction loss with optional SNR weighting.

        Args:
            predicted: Model's predicted v
            target: Target v
            sigma: Noise levels (required for SNR weighting)
            use_snr_weighting: Whether to apply min-SNR weighting
            snr_gamma: SNR weighting parameter
            reduction: Loss reduction ("mean", "sum", "none")

        Returns:
            Loss value
        """
        loss = v_prediction_loss(
            model_output=predicted,
            clean=mx.zeros_like(predicted),  # Not used in basic MSE
            noise=mx.zeros_like(predicted),  # Not used in basic MSE
            sigma=1.0,  # Not used when we compute target separately
            reduction="none",
        )

        # Actually compute MSE directly since we already have target
        loss = (predicted - target).square()

        if use_snr_weighting and sigma is not None:
            # Apply min-SNR weighting per sample
            loss_per_sample = loss.mean(axis=tuple(range(1, loss.ndim)))
            loss_per_sample = min_snr_weighted_loss(
                loss_per_sample,
                sigma,
                min_snr_gamma=snr_gamma,
            )
            if reduction == "mean":
                return loss_per_sample.mean()
            elif reduction == "sum":
                return loss_per_sample.sum()
            else:
                return loss_per_sample

        if reduction == "mean":
            return loss.mean()
        elif reduction == "sum":
            return loss.sum()
        else:
            return loss

    @classmethod
    def forward(
        cls,
        model,  # Stable Audio DiT model
        clean_latents: mx.array,
        encoder_hidden_states: mx.array,
        sigma: Optional[mx.array] = None,
        noise: Optional[mx.array] = None,
        rng: Optional[random.Random] = None,
        use_snr_weighting: bool = True,
        reduction: str = "mean",
    ) -> Tuple[mx.array, mx.array]:
        """
        Complete forward pass for training.

        Args:
            model: Stable Audio DiT model
            clean_latents: Clean VAE-encoded audio [B, T, C]
            encoder_hidden_states: Text conditioning [B, seq, dim]
            sigma: Optional pre-sampled sigma values [B]
            noise: Optional pre-sampled noise [B, T, C]
            rng: Random generator
            use_snr_weighting: Whether to apply min-SNR weighting
            reduction: Loss reduction method

        Returns:
            Tuple of (loss, predicted_v)
        """
        batch_size = clean_latents.shape[0]

        # Sample sigma if not provided
        if sigma is None:
            sigma = cls.sample_sigma(batch_size, rng=rng)

        # Sample noise if not provided
        if noise is None:
            seed = rng.randint(0, 2**31 - 1) if rng else random.randint(0, 2**31 - 1)
            key = mx.random.key(seed)
            noise = mx.random.normal(
                shape=clean_latents.shape,
                dtype=clean_latents.dtype,
                key=key,
            )

        # Add noise using EDM parameterization
        noised_latents = cls.add_noise_edm(clean_latents, noise, sigma)

        # Compute target v
        target_v = cls.compute_target_v(clean_latents, noise, sigma)

        # Forward through model
        # Stable Audio expects: (latents, sigma, encoder_hidden_states)
        predicted_v = model.transformer(
            x=noised_latents,
            sigma=sigma,
            encoder_hidden_states=encoder_hidden_states,
        )

        # Compute loss
        loss = cls.compute_loss(
            predicted=predicted_v,
            target=target_v,
            sigma=sigma,
            use_snr_weighting=use_snr_weighting,
            reduction=reduction,
        )

        return loss, predicted_v
