"""
Shared fixtures for mlx-music training tests.

Provides reusable test fixtures for:
- Temporary directories
- Sample models and batches
- MLX random state management
"""

import random
import tempfile
from pathlib import Path
from typing import Generator

import pytest
import mlx.core as mx
import mlx.nn as nn


@pytest.fixture
def temp_dir() -> Generator[Path, None, None]:
    """Temporary directory that is cleaned up after the test."""
    with tempfile.TemporaryDirectory() as d:
        yield Path(d)


@pytest.fixture
def rng() -> random.Random:
    """Seeded random generator for reproducibility."""
    return random.Random(42)


@pytest.fixture
def mlx_seed():
    """Set MLX random seed for reproducibility."""
    mx.random.seed(42)
    return 42


class SimpleLinearModel(nn.Module):
    """Simple model for testing LoRA and training utilities."""

    def __init__(self, in_features: int = 64, hidden_features: int = 128, out_features: int = 64):
        super().__init__()
        self.linear1 = nn.Linear(in_features, hidden_features)
        self.linear2 = nn.Linear(hidden_features, out_features)

    def __call__(self, x: mx.array) -> mx.array:
        x = self.linear1(x)
        x = nn.relu(x)
        x = self.linear2(x)
        return x


class AttentionModel(nn.Module):
    """Simple attention model for testing LoRA with attention patterns."""

    def __init__(self, dim: int = 64, num_heads: int = 4):
        super().__init__()
        self.to_q = nn.Linear(dim, dim)
        self.to_k = nn.Linear(dim, dim)
        self.to_v = nn.Linear(dim, dim)
        self.to_out = nn.Linear(dim, dim)
        self.num_heads = num_heads
        self.dim = dim

    def __call__(self, x: mx.array) -> mx.array:
        q = self.to_q(x)
        k = self.to_k(x)
        v = self.to_v(x)

        # Simplified attention
        head_dim = self.dim // self.num_heads
        batch, seq, _ = x.shape

        # Reshape for multi-head attention
        q = q.reshape(batch, seq, self.num_heads, head_dim).transpose(0, 2, 1, 3)
        k = k.reshape(batch, seq, self.num_heads, head_dim).transpose(0, 2, 1, 3)
        v = v.reshape(batch, seq, self.num_heads, head_dim).transpose(0, 2, 1, 3)

        # Attention scores
        scale = head_dim ** -0.5
        attn = mx.softmax((q @ k.transpose(0, 1, 3, 2)) * scale, axis=-1)

        # Apply attention
        out = attn @ v
        out = out.transpose(0, 2, 1, 3).reshape(batch, seq, self.dim)

        return self.to_out(out)


@pytest.fixture
def simple_model() -> SimpleLinearModel:
    """Create a simple linear model for testing."""
    return SimpleLinearModel()


@pytest.fixture
def attention_model() -> AttentionModel:
    """Create a simple attention model for testing LoRA patterns."""
    return AttentionModel()


@pytest.fixture
def sample_input() -> mx.array:
    """Sample input tensor [batch=2, features=64]."""
    mx.random.seed(42)
    return mx.random.normal((2, 64))


@pytest.fixture
def sample_sequence_input() -> mx.array:
    """Sample sequence input tensor [batch=2, seq=10, features=64]."""
    mx.random.seed(42)
    return mx.random.normal((2, 10, 64))


@pytest.fixture
def sample_latents() -> mx.array:
    """Sample audio latents [batch=2, channels=8, time=100]."""
    mx.random.seed(42)
    return mx.random.normal((2, 8, 100))


@pytest.fixture
def sample_text_embeddings() -> mx.array:
    """Sample text embeddings [batch=2, seq=32, dim=768]."""
    mx.random.seed(42)
    return mx.random.normal((2, 32, 768))


@pytest.fixture
def sample_batch_dict(sample_input) -> dict:
    """Sample batch dictionary with input and target."""
    return {
        "input": sample_input,
        "target": mx.random.normal((2, 64)),
    }


@pytest.fixture
def mock_optimizer():
    """Create a mock optimizer for scheduler testing."""
    import mlx.optimizers as optim
    return optim.SGD(learning_rate=1e-3)


@pytest.fixture
def adamw_optimizer():
    """Create an AdamW optimizer for training tests."""
    import mlx.optimizers as optim
    return optim.AdamW(learning_rate=1e-4, weight_decay=0.01)
