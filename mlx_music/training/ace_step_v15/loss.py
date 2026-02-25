"""
Flow Matching Loss for ACE-Step v1.5 Training.

V1.5 uses a modified flow matching formulation with:
- Log-normal timestep sampling: t = sigmoid(mu + sigma * N(0,1))
- Reference frame: r = t - uniform(0, data_proportion)
- Interpolation: x_t = (1 - t) * x_0 + t * noise
- Target velocity: v = noise - x_0
- Model predicts: decoder(x_t, t, r, conditions)

Key differences from v1:
- Two timestep inputs (t and r) for temporal reference
- Log-normal timestep distribution instead of uniform
- data_proportion parameter controls reference frame range
"""

import random
from typing import TYPE_CHECKING, Dict, Optional, Tuple

import mlx.core as mx

if TYPE_CHECKING:
    from mlx_music.models.ace_step_v15.dit_model import AceStepConditionGenerationModel


class ACEStepV15Loss:
    """Flow matching loss for ACE-Step v1.5 training.

    Uses the v1.5 formulation with log-normal timestep sampling and
    a reference frame parameter r for temporal conditioning.
    """

    @staticmethod
    def sample_timestep(
        batch_size: int,
        mu: float = -0.4,
        sigma: float = 1.0,
        rng: Optional[random.Random] = None,
    ) -> mx.array:
        """Sample timesteps from log-normal distribution.

        t = sigmoid(mu + sigma * N(0,1))

        This biases timesteps toward the middle range, which is where
        most of the useful learning signal exists.

        Args:
            batch_size: Number of samples
            mu: Log-normal mean (default: -0.4, biases toward lower t)
            sigma: Log-normal std (default: 1.0)
            rng: Optional random generator

        Returns:
            Timesteps [batch_size] in (0, 1)
        """
        if rng is not None:
            seed = rng.randint(0, 2**31 - 1)
        else:
            seed = random.randint(0, 2**31 - 1)

        key = mx.random.key(seed)
        z = mx.random.normal(shape=(batch_size,), key=key)
        t = mx.sigmoid(mu + sigma * z)
        return t

    @staticmethod
    def sample_reference_frame(
        t: mx.array,
        data_proportion: float = 0.5,
        rng: Optional[random.Random] = None,
    ) -> mx.array:
        """Sample reference frame r relative to timestep t.

        r = t - uniform(0, data_proportion)

        The reference frame tells the model about its temporal context
        within the diffusion process.

        Args:
            t: Current timesteps [batch_size]
            data_proportion: Maximum offset for r (default: 0.5)
            rng: Optional random generator

        Returns:
            Reference frames [batch_size]
        """
        if rng is not None:
            seed = rng.randint(0, 2**31 - 1)
        else:
            seed = random.randint(0, 2**31 - 1)

        key = mx.random.key(seed)
        offset = mx.random.uniform(shape=t.shape, key=key) * data_proportion
        r = t - offset
        return r

    @staticmethod
    def sample_noise(
        shape: Tuple[int, ...],
        dtype: mx.Dtype = mx.bfloat16,
        rng: Optional[random.Random] = None,
    ) -> mx.array:
        """Sample Gaussian noise."""
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
        t: mx.array,
    ) -> mx.array:
        """Interpolate between clean latents and noise at timestep t.

        x_t = (1 - t) * clean + t * noise

        Args:
            clean: Clean audio latents [B, T, D]
            noise: Gaussian noise [B, T, D]
            t: Timesteps [B] in (0, 1)

        Returns:
            Noised latents at timestep t
        """
        # Expand t for broadcasting: [B] -> [B, 1, 1]
        while t.ndim < clean.ndim:
            t = t[..., None]
        return (1 - t) * clean + t * noise

    @staticmethod
    def compute_target_velocity(clean: mx.array, noise: mx.array) -> mx.array:
        """Compute target velocity for flow matching.

        v = noise - clean

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
        """Compute MSE loss between predicted and target velocity.

        Args:
            predicted: Model's predicted velocity
            target: Target velocity (noise - clean)
            mask: Optional mask for valid regions
            reduction: "mean", "sum", or "none"

        Returns:
            Loss value
        """
        loss = (predicted - target).square()

        if mask is not None:
            loss = loss * mask
            if reduction == "mean":
                return loss.sum() / (mask.sum() + 1e-8)
            elif reduction == "sum":
                return loss.sum()
            return loss

        if reduction == "mean":
            return loss.mean()
        elif reduction == "sum":
            return loss.sum()
        return loss

    @classmethod
    def forward(
        cls,
        model: "AceStepConditionGenerationModel",
        clean_latents: mx.array,
        encoder_hidden_states: mx.array,
        encoder_attention_mask: mx.array,
        context_latents: mx.array,
        mu: float = -0.4,
        sigma: float = 1.0,
        data_proportion: float = 0.5,
        t: Optional[mx.array] = None,
        noise: Optional[mx.array] = None,
        rng: Optional[random.Random] = None,
        reduction: str = "mean",
    ) -> Tuple[mx.array, mx.array]:
        """Complete forward pass for v1.5 training.

        Args:
            model: AceStepConditionGenerationModel (or its decoder)
            clean_latents: Clean VAE-encoded audio [B, T, D]
            encoder_hidden_states: Condition embeddings [B, enc_T, hidden]
            encoder_attention_mask: Condition mask [B, enc_T]
            context_latents: Concatenated src_latents + chunk_masks [B, T, C]
            mu: Timestep sampling mean
            sigma: Timestep sampling std
            data_proportion: Reference frame range
            t: Optional pre-sampled timesteps [B]
            noise: Optional pre-sampled noise [B, T, D]
            rng: Random generator
            reduction: Loss reduction method

        Returns:
            (loss, predicted_velocity)
        """
        batch_size = clean_latents.shape[0]

        # Sample timesteps if not provided
        if t is None:
            t = cls.sample_timestep(batch_size, mu=mu, sigma=sigma, rng=rng)

        # Sample reference frame
        r = cls.sample_reference_frame(t, data_proportion=data_proportion, rng=rng)

        # Sample noise if not provided
        if noise is None:
            noise = cls.sample_noise(clean_latents.shape, clean_latents.dtype, rng)

        # Add noise
        noised_latents = cls.add_noise(clean_latents, noise, t)

        # Target velocity
        target_velocity = cls.compute_target_velocity(clean_latents, noise)

        # Forward through DiT decoder
        predicted_velocity, _ = model.decoder(
            hidden_states=noised_latents,
            timestep=t,
            timestep_r=r,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            context_latents=context_latents,
        )

        # Compute loss
        loss = cls.compute_loss(predicted_velocity, target_velocity, reduction=reduction)

        return loss, predicted_velocity


def create_v15_train_step(
    model: "AceStepConditionGenerationModel",
    mu: float = -0.4,
    sigma: float = 1.0,
    data_proportion: float = 0.5,
):
    """Create a training step function for gradient computation.

    Returns a function suitable for use with nn.value_and_grad().

    Args:
        model: ACE-Step v1.5 model
        mu: Timestep sampling mean
        sigma: Timestep sampling std
        data_proportion: Reference frame range

    Returns:
        Training step function
    """

    def train_step(
        clean_latents: mx.array,
        encoder_hidden_states: mx.array,
        encoder_attention_mask: mx.array,
        context_latents: mx.array,
    ) -> mx.array:
        loss, _ = ACEStepV15Loss.forward(
            model=model,
            clean_latents=clean_latents,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            context_latents=context_latents,
            mu=mu,
            sigma=sigma,
            data_proportion=data_proportion,
        )
        return loss

    return train_step
