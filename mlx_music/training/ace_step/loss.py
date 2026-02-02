"""
Flow Matching Loss for ACE-Step Training.

ACE-Step uses rectified flow (flow matching) for training:
- Sample random timestep t
- Interpolate between clean audio latent and noise at t
- Predict velocity (direction from noise to clean)
- Loss = ||predicted_velocity - target_velocity||²

This is mathematically equivalent to training the model to predict
the velocity field that transports samples from noise to data.

Reference:
    "Flow Matching for Generative Modeling" (Lipman et al., 2022)
    "Rectified Flow" (Liu et al., 2022)
"""

import random
from typing import TYPE_CHECKING, Optional, Tuple

import mlx.core as mx

if TYPE_CHECKING:
    from mlx_music.models.ace_step import ACEStepModel


class ACEStepLoss:
    """
    Flow matching loss computation for ACE-Step training.

    ACE-Step uses rectified flow where:
    - latents_t = (1 - sigma) * clean + sigma * noise
    - target_velocity = noise - clean
    - model predicts velocity at latents_t, timestep t

    The loss is MSE between predicted and target velocity.

    Optimizations:
    - Vectorized batch processing for tensor operations
    - MLX key splitting for reproducible random generation
    """

    @staticmethod
    def sample_timestep(
        batch_size: int,
        num_timesteps: int,
        rng: Optional[random.Random] = None,
    ) -> mx.array:
        """
        Sample random timesteps for the batch.

        Args:
            batch_size: Number of samples in batch
            num_timesteps: Total number of timesteps (scheduler steps)
            rng: Optional random generator for reproducibility

        Returns:
            Timestep indices [batch_size] as int32
        """
        if rng is not None:
            seed = rng.randint(0, 2**31 - 1)
        else:
            seed = random.randint(0, 2**31 - 1)

        key = mx.random.key(seed)
        timesteps = mx.random.randint(
            low=0,
            high=num_timesteps,
            shape=(batch_size,),
            key=key,
        )
        return timesteps.astype(mx.int32)

    @staticmethod
    def sample_noise(
        shape: Tuple[int, ...],
        dtype: mx.Dtype = mx.bfloat16,
        rng: Optional[random.Random] = None,
    ) -> mx.array:
        """
        Sample Gaussian noise.

        Args:
            shape: Shape of noise tensor
            dtype: Data type for noise
            rng: Optional random generator for reproducibility

        Returns:
            Gaussian noise tensor
        """
        if rng is not None:
            seed = rng.randint(0, 2**31 - 1)
        else:
            seed = random.randint(0, 2**31 - 1)

        key = mx.random.key(seed)
        return mx.random.normal(shape=shape, dtype=dtype, key=key)

    @staticmethod
    def add_noise(
        clean: mx.array,
        noise: mx.array,
        sigma: mx.array | float,
    ) -> mx.array:
        """
        Interpolate between clean and noise at sigma level.

        Formula: latents_t = (1 - sigma) * clean + sigma * noise

        Args:
            clean: Clean audio latents
            noise: Gaussian noise
            sigma: Noise level(s) - scalar or broadcastable array

        Returns:
            Noised latents at sigma level
        """
        if isinstance(sigma, (int, float)):
            sigma = mx.array(sigma)

        # Ensure sigma broadcasts correctly
        while sigma.ndim < clean.ndim:
            sigma = sigma[..., None]

        return (1 - sigma) * clean + sigma * noise

    @staticmethod
    def compute_target_velocity(clean: mx.array, noise: mx.array) -> mx.array:
        """
        Compute target velocity for flow matching.

        For rectified flow: target = noise - clean
        This represents the direction from clean to noise (or vice versa
        depending on formulation).

        Args:
            clean: Clean audio latents
            noise: Gaussian noise

        Returns:
            Target velocity
        """
        return noise - clean

    @staticmethod
    def compute_loss(
        predicted: mx.array,
        target: mx.array,
        mask: Optional[mx.array] = None,
        reduction: str = "mean",
    ) -> mx.array:
        """
        Compute MSE loss between predicted and target velocity.

        Args:
            predicted: Model's predicted velocity
            target: Target velocity (noise - clean)
            mask: Optional mask for valid regions
            reduction: How to reduce loss ("mean", "sum", "none")

        Returns:
            Loss value
        """
        loss = (predicted - target).square()

        if mask is not None:
            # Apply mask and normalize
            loss = loss * mask
            if reduction == "mean":
                return loss.sum() / (mask.sum() + 1e-8)
            elif reduction == "sum":
                return loss.sum()
            else:
                return loss

        if reduction == "mean":
            return loss.mean()
        elif reduction == "sum":
            return loss.sum()
        else:
            return loss

    @classmethod
    def forward(
        cls,
        model: "ACEStepModel",
        clean_latents: mx.array,
        text_embeddings: mx.array,
        sigmas: mx.array,
        timesteps: Optional[mx.array] = None,
        noise: Optional[mx.array] = None,
        rng: Optional[random.Random] = None,
        reduction: str = "mean",
    ) -> Tuple[mx.array, mx.array]:
        """
        Complete forward pass for training.

        Args:
            model: ACE-Step model
            clean_latents: Clean VAE-encoded audio [B, C, T]
            text_embeddings: Text conditioning embeddings
            sigmas: Scheduler sigma values (from scheduler.sigmas)
            timesteps: Optional pre-sampled timesteps [B]
            noise: Optional pre-sampled noise [B, C, T]
            rng: Random generator for reproducibility
            reduction: Loss reduction method

        Returns:
            Tuple of (loss, predicted_velocity)

        Example:
            # In training loop:
            loss, pred = ACEStepLoss.forward(
                model=model,
                clean_latents=batch.latents,
                text_embeddings=batch.text_embeds,
                sigmas=scheduler.sigmas,
            )
        """
        batch_size = clean_latents.shape[0]
        num_timesteps = len(sigmas)

        # Sample timesteps if not provided
        if timesteps is None:
            timesteps = cls.sample_timestep(batch_size, num_timesteps, rng)

        # Sample noise if not provided
        if noise is None:
            noise = cls.sample_noise(clean_latents.shape, clean_latents.dtype, rng)

        # Get sigma for each sample
        sigma_t = sigmas[timesteps]  # [B]

        # Add noise to clean latents
        noised_latents = cls.add_noise(clean_latents, noise, sigma_t)

        # Get target velocity
        target_velocity = cls.compute_target_velocity(clean_latents, noise)

        # Forward through model
        # ACE-Step transformer expects: (latents, timestep, sigmas, text_embeddings)
        predicted_velocity = model.transformer(
            x=noised_latents,
            t=timesteps,
            sigmas=sigmas,
            cap_feats=text_embeddings,
        )

        # Compute loss
        loss = cls.compute_loss(predicted_velocity, target_velocity, reduction=reduction)

        return loss, predicted_velocity


def create_train_step(model: "ACEStepModel", sigmas: mx.array):
    """
    Create a training step function for gradient computation.

    This returns a function suitable for use with nn.value_and_grad().

    Args:
        model: ACE-Step model
        sigmas: Scheduler sigma values

    Returns:
        Training step function that takes (clean_latents, text_embeddings)
        and returns loss.

    Example:
        train_fn = create_train_step(model, scheduler.sigmas)
        loss_and_grad_fn = nn.value_and_grad(model, train_fn)

        for batch in dataloader:
            loss, grads = loss_and_grad_fn(batch.latents, batch.text_embeds)
            optimizer.update(model=model, gradients=grads)
    """

    def train_step(clean_latents: mx.array, text_embeddings: mx.array) -> mx.array:
        loss, _ = ACEStepLoss.forward(
            model=model,
            clean_latents=clean_latents,
            text_embeddings=text_embeddings,
            sigmas=sigmas,
        )
        return loss

    return train_step
