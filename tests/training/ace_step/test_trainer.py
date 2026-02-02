"""
Tests for ACE-Step trainer.

Tests:
- TrainingConfig: Defaults, __post_init__, validation
- TrainingState: state_dict(), load_state_dict()
- ACEStepTrainer: Training loop, checkpointing, load_checkpoint security
"""

import json
import pytest
from pathlib import Path
import mlx.core as mx
import mlx.nn as nn


class TestTrainingConfig:
    """Tests for TrainingConfig dataclass."""

    def test_training_config_defaults(self):
        """TrainingConfig should have sensible defaults."""
        from mlx_music.training.ace_step.trainer import TrainingConfig

        config = TrainingConfig()

        assert config.learning_rate == 1e-4
        assert config.num_epochs == 10
        assert config.batch_size == 1
        assert config.gradient_accumulation_steps == 4
        assert config.use_lora is True
        assert config.lora_rank == 64
        assert config.use_ema is True
        assert config.lr_scheduler == "cosine"

    def test_training_config_post_init_converts_path(self):
        """TrainingConfig.__post_init__ should convert output_dir to Path."""
        from mlx_music.training.ace_step.trainer import TrainingConfig

        config = TrainingConfig(output_dir="./my_checkpoints")

        assert isinstance(config.output_dir, Path)
        assert config.output_dir == Path("./my_checkpoints")

    def test_training_config_custom_values(self):
        """TrainingConfig should accept custom values."""
        from mlx_music.training.ace_step.trainer import TrainingConfig

        config = TrainingConfig(
            learning_rate=5e-5,
            num_epochs=20,
            lora_rank=128,
            use_ema=False,
        )

        assert config.learning_rate == 5e-5
        assert config.num_epochs == 20
        assert config.lora_rank == 128
        assert config.use_ema is False


class TestTrainingState:
    """Tests for TrainingState dataclass."""

    def test_training_state_defaults(self):
        """TrainingState should initialize with zeros/inf."""
        from mlx_music.training.ace_step.trainer import TrainingState

        state = TrainingState()

        assert state.global_step == 0
        assert state.epoch == 0
        assert state.best_loss == float("inf")
        assert state.total_tokens == 0

    def test_training_state_state_dict(self):
        """TrainingState.state_dict() should return serializable dict."""
        from mlx_music.training.ace_step.trainer import TrainingState

        state = TrainingState()
        state.global_step = 100
        state.epoch = 5
        state.best_loss = 0.123

        d = state.state_dict()

        assert d["global_step"] == 100
        assert d["epoch"] == 5
        assert d["best_loss"] == 0.123

        # Should be JSON serializable
        json_str = json.dumps(d)
        assert "global_step" in json_str

    def test_training_state_load_state_dict(self):
        """TrainingState.load_state_dict() should restore state."""
        from mlx_music.training.ace_step.trainer import TrainingState

        state = TrainingState()

        d = {
            "global_step": 200,
            "epoch": 10,
            "best_loss": 0.05,
            "total_tokens": 50000,
        }
        state.load_state_dict(d)

        assert state.global_step == 200
        assert state.epoch == 10
        assert state.best_loss == 0.05
        assert state.total_tokens == 50000


class TestACEStepTrainer:
    """Tests for ACEStepTrainer class."""

    def test_trainer_load_checkpoint_validates_path(self, temp_dir):
        """load_checkpoint should validate path is within output_dir."""
        from mlx_music.training.ace_step.trainer import ACEStepTrainer, TrainingConfig

        # Create minimal mock model
        class MockModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(64, 64)

            def __call__(self, x):
                return self.linear(x)

        model = MockModel()
        config = TrainingConfig(output_dir=temp_dir)
        sigmas = mx.linspace(0.001, 1.0, 100)

        trainer = ACEStepTrainer(model=model, config=config, sigmas=sigmas)

        # Try to load from outside output_dir
        outside_dir = temp_dir.parent / "outside_checkpoint"
        outside_dir.mkdir(exist_ok=True)

        with pytest.raises(ValueError, match="within output directory"):
            trainer.load_checkpoint(outside_dir)

    def test_trainer_load_checkpoint_validates_exists(self, temp_dir):
        """load_checkpoint should raise for nonexistent directory."""
        from mlx_music.training.ace_step.trainer import ACEStepTrainer, TrainingConfig

        class MockModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(64, 64)

            def __call__(self, x):
                return self.linear(x)

        model = MockModel()
        config = TrainingConfig(output_dir=temp_dir)
        sigmas = mx.linspace(0.001, 1.0, 100)

        trainer = ACEStepTrainer(model=model, config=config, sigmas=sigmas)

        nonexistent = temp_dir / "nonexistent"

        with pytest.raises(FileNotFoundError, match="not found"):
            trainer.load_checkpoint(nonexistent)

    def test_trainer_save_checkpoint_creates_files(self, temp_dir):
        """_save_checkpoint should create expected files."""
        from mlx_music.training.ace_step.trainer import ACEStepTrainer, TrainingConfig

        class MockModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.to_q = nn.Linear(64, 64)  # Target for LoRA
                self.to_k = nn.Linear(64, 64)

            def __call__(self, x):
                return self.to_q(x) + self.to_k(x)

        model = MockModel()
        config = TrainingConfig(
            output_dir=temp_dir,
            experiment_name="test_exp",
            use_lora=True,
            use_ema=True,
        )
        sigmas = mx.linspace(0.001, 1.0, 100)

        trainer = ACEStepTrainer(model=model, config=config, sigmas=sigmas)
        trainer._save_checkpoint("test_checkpoint")

        checkpoint_dir = temp_dir / "test_exp" / "test_checkpoint"

        assert checkpoint_dir.exists()
        assert (checkpoint_dir / "lora_weights.safetensors").exists()
        assert (checkpoint_dir / "training_state.json").exists()
        assert (checkpoint_dir / "optimizer.safetensors").exists()
        assert (checkpoint_dir / "ema_weights.safetensors").exists()

    def test_trainer_checkpoint_roundtrip(self, temp_dir):
        """Save then load checkpoint should restore state."""
        from mlx_music.training.ace_step.trainer import ACEStepTrainer, TrainingConfig

        class MockModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.to_q = nn.Linear(64, 64)

            def __call__(self, x):
                return self.to_q(x)

        model = MockModel()
        config = TrainingConfig(
            output_dir=temp_dir,
            experiment_name="roundtrip",
            use_lora=True,
            use_ema=False,  # Simplify test
        )
        sigmas = mx.linspace(0.001, 1.0, 100)

        trainer = ACEStepTrainer(model=model, config=config, sigmas=sigmas)

        # Modify state
        trainer.state.global_step = 500
        trainer.state.epoch = 3
        trainer.state.best_loss = 0.05

        trainer._save_checkpoint("state_test")

        # Create new trainer and load
        new_model = MockModel()
        new_trainer = ACEStepTrainer(model=new_model, config=config, sigmas=sigmas)
        new_trainer.load_checkpoint(temp_dir / "roundtrip" / "state_test")

        assert new_trainer.state.global_step == 500
        assert new_trainer.state.epoch == 3
        assert new_trainer.state.best_loss == 0.05

    def test_trainer_cleanup_checkpoints(self, temp_dir):
        """_cleanup_checkpoints should remove old checkpoints."""
        from mlx_music.training.ace_step.trainer import ACEStepTrainer, TrainingConfig

        class MockModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.to_q = nn.Linear(64, 64)

            def __call__(self, x):
                return self.to_q(x)

        model = MockModel()
        config = TrainingConfig(
            output_dir=temp_dir,
            experiment_name="cleanup_test",
            max_checkpoints=2,
            use_ema=False,
        )
        sigmas = mx.linspace(0.001, 1.0, 100)

        trainer = ACEStepTrainer(model=model, config=config, sigmas=sigmas)

        # Create several checkpoints
        for i in range(5):
            trainer.state.global_step = (i + 1) * 100
            trainer._save_checkpoint(f"step_{(i + 1) * 100}")

        # Should only keep max_checkpoints step checkpoints
        exp_dir = temp_dir / "cleanup_test"
        step_dirs = [d for d in exp_dir.iterdir() if d.name.startswith("step_")]

        assert len(step_dirs) <= config.max_checkpoints

    def test_trainer_applies_lora(self, temp_dir):
        """Trainer should apply LoRA when use_lora=True."""
        from mlx_music.training.ace_step.trainer import ACEStepTrainer, TrainingConfig
        from mlx_music.training.ace_step.lora_layers import LoRALinear

        class MockModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.to_q = nn.Linear(64, 64)
                self.to_k = nn.Linear(64, 64)

            def __call__(self, x):
                return self.to_q(x) + self.to_k(x)

        model = MockModel()
        config = TrainingConfig(
            output_dir=temp_dir,
            use_lora=True,
            lora_rank=16,
        )
        sigmas = mx.linspace(0.001, 1.0, 100)

        # After trainer init, model should have LoRA layers
        trainer = ACEStepTrainer(model=model, config=config, sigmas=sigmas)

        assert isinstance(trainer.model.to_q, LoRALinear)
        assert isinstance(trainer.model.to_k, LoRALinear)
        assert trainer.model.to_q.rank == 16

    def test_trainer_creates_ema_when_enabled(self, temp_dir):
        """Trainer should create EMA when use_ema=True."""
        from mlx_music.training.ace_step.trainer import ACEStepTrainer, TrainingConfig
        from mlx_music.training.common.ema import EMAModel

        class MockModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.to_q = nn.Linear(64, 64)

            def __call__(self, x):
                return self.to_q(x)

        model = MockModel()
        config = TrainingConfig(output_dir=temp_dir, use_ema=True)
        sigmas = mx.linspace(0.001, 1.0, 100)

        trainer = ACEStepTrainer(model=model, config=config, sigmas=sigmas)

        assert isinstance(trainer.ema, EMAModel)

    def test_trainer_creates_noop_ema_when_disabled(self, temp_dir):
        """Trainer should create NoOpEMA when use_ema=False."""
        from mlx_music.training.ace_step.trainer import ACEStepTrainer, TrainingConfig
        from mlx_music.training.common.ema import NoOpEMA

        class MockModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.to_q = nn.Linear(64, 64)

            def __call__(self, x):
                return self.to_q(x)

        model = MockModel()
        config = TrainingConfig(output_dir=temp_dir, use_ema=False)
        sigmas = mx.linspace(0.001, 1.0, 100)

        trainer = ACEStepTrainer(model=model, config=config, sigmas=sigmas)

        assert isinstance(trainer.ema, NoOpEMA)
