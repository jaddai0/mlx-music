"""
Conditioning modules for Stable Audio Open.

Contains the projection model for combining text and timing embeddings,
as well as timestep embedding utilities.
"""

import math
from typing import Optional, Tuple

import mlx.core as mx
import mlx.nn as nn


class TimestepEmbedding(nn.Module):
    """
    Sinusoidal timestep embeddings for diffusion models.

    Projects scalar timesteps to high-dimensional vectors using
    sinusoidal positional encoding followed by MLP.
    """

    def __init__(
        self,
        dim: int,
        time_embed_dim: int,
        act_fn: str = "silu",
    ):
        """
        Initialize timestep embedding.

        Args:
            dim: Input dimension for sinusoidal encoding
            time_embed_dim: Output dimension after MLP
            act_fn: Activation function ("silu", "gelu")
        """
        super().__init__()
        self.dim = dim
        self.time_embed_dim = time_embed_dim

        # MLP layers
        self.linear_1 = nn.Linear(dim, time_embed_dim)
        self.linear_2 = nn.Linear(time_embed_dim, time_embed_dim)

        if act_fn == "silu":
            self.act = nn.SiLU()
        elif act_fn == "gelu":
            self.act = nn.GELU()
        else:
            raise ValueError(f"Unknown activation: {act_fn}")

    def __call__(self, timesteps: mx.array) -> mx.array:
        """
        Embed timesteps.

        Args:
            timesteps: Scalar timesteps (batch,) or (batch, 1)

        Returns:
            Embeddings of shape (batch, time_embed_dim)
        """
        # Ensure 1D
        if timesteps.ndim == 2:
            timesteps = timesteps.squeeze(-1)

        # Sinusoidal embedding
        half_dim = self.dim // 2
        freqs = mx.exp(
            -math.log(10000.0) * mx.arange(0, half_dim) / half_dim
        )

        # Outer product: (batch,) x (half_dim,) -> (batch, half_dim)
        args = timesteps[:, None] * freqs[None, :]

        # Concatenate sin and cos
        embedding = mx.concatenate([mx.cos(args), mx.sin(args)], axis=-1)

        # MLP
        embedding = self.linear_1(embedding)
        embedding = self.act(embedding)
        embedding = self.linear_2(embedding)

        return embedding


class NumberEmbedding(nn.Module):
    """
    Embedding for scalar number inputs (like timing values).

    Uses Fourier features followed by MLP for rich representations.
    """

    def __init__(
        self,
        fourier_dim: int = 256,
        output_dim: int = 768,
    ):
        """
        Initialize number embedding.

        Args:
            fourier_dim: Dimension of Fourier features
            output_dim: Output embedding dimension
        """
        super().__init__()
        self.fourier_dim = fourier_dim
        self.output_dim = output_dim

        # Learnable Fourier features
        self.weight = mx.random.normal((fourier_dim,))

        # MLP projection
        self.mlp = nn.Sequential(
            nn.Linear(fourier_dim * 2, output_dim),
            nn.SiLU(),
            nn.Linear(output_dim, output_dim),
        )

    def __call__(self, x: mx.array) -> mx.array:
        """
        Embed scalar values.

        Args:
            x: Scalar values of shape (batch,) or (batch, 1)

        Returns:
            Embeddings of shape (batch, output_dim)
        """
        # Ensure 2D for broadcasting
        if x.ndim == 1:
            x = x[:, None]

        # Fourier features: x * weight
        # (batch, 1) * (fourier_dim,) -> (batch, fourier_dim)
        fourier = x * self.weight[None, :] * 2 * math.pi

        # Concatenate sin and cos
        fourier_features = mx.concatenate(
            [mx.sin(fourier), mx.cos(fourier)], axis=-1
        )

        # MLP
        return self.mlp(fourier_features)


class ProjectionModel(nn.Module):
    """
    Projection model for Stable Audio conditioning.

    Combines text embeddings with timing information (start/end seconds)
    to create global conditioning for the diffusion model.
    """

    def __init__(
        self,
        text_encoder_dim: int = 768,
        output_dim: int = 1536,
        num_timing_features: int = 2,
    ):
        """
        Initialize projection model.

        Args:
            text_encoder_dim: Dimension of text encoder outputs
            output_dim: Output conditioning dimension
            num_timing_features: Number of timing features (start, total)
        """
        super().__init__()
        self.text_encoder_dim = text_encoder_dim
        self.output_dim = output_dim

        # Text projection
        self.text_proj = nn.Linear(text_encoder_dim, output_dim)

        # Timing embeddings
        self.seconds_start_embed = NumberEmbedding(
            fourier_dim=256, output_dim=output_dim
        )
        self.seconds_total_embed = NumberEmbedding(
            fourier_dim=256, output_dim=output_dim
        )

    def __call__(
        self,
        text_embeds: mx.array,
        seconds_start: mx.array,
        seconds_total: mx.array,
    ) -> mx.array:
        """
        Compute global conditioning.

        Args:
            text_embeds: Pooled text embeddings (batch, text_encoder_dim)
            seconds_start: Start time in seconds (batch,)
            seconds_total: Total duration in seconds (batch,)

        Returns:
            Global conditioning (batch, output_dim)
        """
        # Project text
        text_cond = self.text_proj(text_embeds)

        # Embed timing
        start_cond = self.seconds_start_embed(seconds_start)
        total_cond = self.seconds_total_embed(seconds_total)

        # Combine
        global_cond = text_cond + start_cond + total_cond

        return global_cond


class ConditioningManager:
    """
    Manager for preparing conditioning inputs for Stable Audio.

    Handles text encoding, timing embedding, and CFG (classifier-free guidance).
    """

    def __init__(
        self,
        projection_model: ProjectionModel,
        text_encoder: Optional[object] = None,
    ):
        """
        Initialize conditioning manager.

        Args:
            projection_model: ProjectionModel instance
            text_encoder: Optional T5 text encoder
        """
        self.projection_model = projection_model
        self.text_encoder = text_encoder

    def encode_text(
        self,
        prompt: str,
        negative_prompt: Optional[str] = None,
    ) -> Tuple[mx.array, mx.array, mx.array, mx.array]:
        """
        Encode text prompts.

        Args:
            prompt: Text prompt
            negative_prompt: Optional negative prompt for CFG

        Returns:
            Tuple of (text_embeds, pooled_embeds, neg_text_embeds, neg_pooled_embeds)
        """
        if self.text_encoder is None:
            # Return placeholder embeddings
            batch_size = 1
            seq_len = 64
            dim = 768

            text_embeds = mx.zeros((batch_size, seq_len, dim))
            pooled = mx.zeros((batch_size, dim))

            neg_text_embeds = mx.zeros((batch_size, seq_len, dim))
            neg_pooled = mx.zeros((batch_size, dim))

            return text_embeds, pooled, neg_text_embeds, neg_pooled

        # Encode positive prompt
        text_embeds, pooled = self.text_encoder.encode(prompt)

        # Encode negative prompt
        if negative_prompt is not None:
            neg_text_embeds, neg_pooled = self.text_encoder.encode(negative_prompt)
        else:
            # Use empty string for unconditional
            neg_text_embeds, neg_pooled = self.text_encoder.encode("")

        return text_embeds, pooled, neg_text_embeds, neg_pooled

    def prepare_global_conditioning(
        self,
        pooled_embeds: mx.array,
        seconds_start: float,
        seconds_total: float,
        batch_size: int = 1,
    ) -> mx.array:
        """
        Prepare global conditioning from pooled embeddings and timing.

        Args:
            pooled_embeds: Pooled text embeddings (batch, dim)
            seconds_start: Start time in seconds
            seconds_total: Total duration in seconds
            batch_size: Batch size for expanding timing

        Returns:
            Global conditioning (batch, output_dim)
        """
        # Convert timing to arrays
        start_array = mx.array([seconds_start] * batch_size)
        total_array = mx.array([seconds_total] * batch_size)

        # Compute global conditioning
        return self.projection_model(pooled_embeds, start_array, total_array)


__all__ = [
    "TimestepEmbedding",
    "NumberEmbedding",
    "ProjectionModel",
    "ConditioningManager",
]
