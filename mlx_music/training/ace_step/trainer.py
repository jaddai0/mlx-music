"""
ACE-Step LoRA Trainer.

Provides a complete training loop for ACE-Step with:
- LoRA fine-tuning support
- Flow matching loss
- EMA weight averaging
- Learning rate scheduling
- Gradient accumulation
- Checkpointing

Example:
    from mlx_music.training.ace_step import ACEStepTrainer, TrainingConfig

    config = TrainingConfig(
        output_dir="./checkpoints",
        learning_rate=1e-4,
        num_epochs=10,
        lora_rank=64,
    )

    trainer = ACEStepTrainer(model=model, config=config)
    trainer.train(dataset)
"""

import json
import logging
import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Optional

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim

from mlx_music.training.common import (
    EMAModel,
    GradientAccumulator,
    create_accumulator,
    create_ema,
    create_scheduler,
    flow_matching_loss,
)

from .dataset import AudioBatch, AudioDataset
from .lora_layers import apply_lora_to_model, get_lora_parameters, save_lora_weights
from .loss import ACEStepLoss

logger = logging.getLogger(__name__)


def _force_evaluate(*args) -> None:
    """Force MLX lazy evaluation. This is mlx.core.eval, NOT Python's eval()."""
    mx.eval(*args)


@dataclass
class TrainingConfig:
    """Configuration for ACE-Step training."""

    # Output
    output_dir: Path = field(default_factory=lambda: Path("./ace_step_checkpoints"))
    experiment_name: str = "ace_step_lora"

    # Training parameters
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    num_epochs: int = 10
    batch_size: int = 1
    gradient_accumulation_steps: int = 4

    # LoRA configuration
    use_lora: bool = True
    lora_rank: int = 64
    lora_alpha: float = 64.0
    lora_dropout: float = 0.0

    # EMA configuration
    use_ema: bool = True
    ema_decay: float = 0.9999

    # Learning rate schedule
    lr_scheduler: str = "cosine"  # "constant", "cosine", "onecycle"
    warmup_steps: int = 100
    min_lr: float = 1e-6

    # Checkpointing
    save_every_n_steps: int = 500
    validate_every_n_steps: int = 100
    max_checkpoints: int = 5

    # Logging
    log_every_n_steps: int = 10

    # Mixed precision
    use_bf16: bool = True

    def __post_init__(self):
        self.output_dir = Path(self.output_dir)


@dataclass
class TrainingState:
    """Mutable training state."""

    global_step: int = 0
    epoch: int = 0
    best_loss: float = float("inf")
    total_tokens: int = 0

    def state_dict(self) -> Dict[str, Any]:
        return {
            "global_step": self.global_step,
            "epoch": self.epoch,
            "best_loss": self.best_loss,
            "total_tokens": self.total_tokens,
        }

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        self.global_step = state["global_step"]
        self.epoch = state["epoch"]
        self.best_loss = state["best_loss"]
        self.total_tokens = state.get("total_tokens", 0)


class ACEStepTrainer:
    """
    Trainer for ACE-Step LoRA fine-tuning.

    Handles the complete training loop including:
    - LoRA layer injection
    - Optimizer and scheduler setup
    - Training step with gradient accumulation
    - EMA weight updates
    - Checkpointing and logging
    """

    def __init__(
        self,
        model: nn.Module,
        config: TrainingConfig,
        sigmas: mx.array,
    ):
        """
        Initialize trainer.

        Args:
            model: ACE-Step model to train
            config: Training configuration
            sigmas: Scheduler sigmas from ACE-Step scheduler
        """
        self.model = model
        self.config = config
        self.sigmas = sigmas
        self.state = TrainingState()

        # Setup output directory
        self.config.output_dir.mkdir(parents=True, exist_ok=True)

        # Apply LoRA if configured
        if config.use_lora:
            num_lora = apply_lora_to_model(
                model,
                rank=config.lora_rank,
                alpha=config.lora_alpha,
                dropout=config.lora_dropout,
            )
            logger.info(f"Applied LoRA to {num_lora} layers (rank={config.lora_rank})")

        # Setup optimizer (only LoRA params if using LoRA)
        if config.use_lora:
            self.trainable_params = get_lora_parameters(model)
        else:
            self.trainable_params = model.trainable_parameters()

        self.optimizer = optim.AdamW(
            learning_rate=config.learning_rate,
            weight_decay=config.weight_decay,
        )

        # Setup EMA
        self.ema = create_ema(model, enabled=config.use_ema, decay=config.ema_decay)

        # Setup gradient accumulator
        self.accumulator = create_accumulator(config.gradient_accumulation_steps)

        # Scheduler will be initialized when we know total steps
        self.scheduler = None

    def _setup_scheduler(self, total_steps: int) -> None:
        """Setup learning rate scheduler."""
        self.scheduler = create_scheduler(
            name=self.config.lr_scheduler,
            optimizer=self.optimizer,
            initial_lr=self.config.learning_rate,
            total_steps=total_steps,
            warmup_steps=self.config.warmup_steps,
            min_lr=self.config.min_lr,
        )

    def _compute_loss(
        self,
        batch: AudioBatch,
    ) -> mx.array:
        """Compute flow matching loss for a batch."""
        loss, _ = ACEStepLoss.forward(
            model=self.model,
            clean_latents=batch.encoded_latents,
            text_embeddings=batch.text_embeddings,
            sigmas=self.sigmas,
            rng=batch.rng,
        )
        return loss

    def _train_step(
        self,
        batch: AudioBatch,
    ) -> Dict[str, float]:
        """
        Execute a single training step.

        Returns metrics dict with loss and learning rate.
        """
        # Create loss function for value_and_grad
        def loss_fn(model_params):
            # Update model with current params
            self.model.update(model_params)
            return self._compute_loss(batch)

        # Get loss and gradients
        loss, grads = nn.value_and_grad(self.model, loss_fn)(self.model.parameters())

        # Force evaluation of loss and grads (mlx.core.eval, not Python eval)
        _force_evaluate(loss, grads)

        # Accumulate gradients
        accumulated = self.accumulator.accumulate(grads)

        metrics = {"loss": float(loss)}

        # If accumulation complete, update weights
        if accumulated is not None:
            # Apply optimizer
            self.optimizer.update(self.model, accumulated)
            _force_evaluate(self.model.parameters())

            # Update EMA
            self.ema.update(self.model)

            # Step scheduler
            if self.scheduler is not None:
                lr = self.scheduler.step()
                metrics["lr"] = lr

            self.state.global_step += 1

        return metrics

    def train(
        self,
        dataset: AudioDataset,
        val_dataset: Optional[AudioDataset] = None,
        callbacks: Optional[Dict[str, Callable]] = None,
    ) -> None:
        """
        Run the training loop.

        Args:
            dataset: Training dataset
            val_dataset: Optional validation dataset
            callbacks: Optional dict of callback functions:
                - "on_step_end": called after each step with (trainer, metrics)
                - "on_epoch_end": called after each epoch with (trainer, metrics)
                - "on_validation": called after validation with (trainer, val_loss)
        """
        callbacks = callbacks or {}

        # Calculate total steps
        steps_per_epoch = len(dataset) // self.config.batch_size
        total_steps = steps_per_epoch * self.config.num_epochs
        total_steps = total_steps // self.config.gradient_accumulation_steps

        logger.info(f"Starting training: {total_steps} steps, {self.config.num_epochs} epochs")
        logger.info(f"Effective batch size: {self.config.batch_size * self.config.gradient_accumulation_steps}")

        # Setup scheduler
        self._setup_scheduler(total_steps)

        # Training loop
        for epoch in range(self.config.num_epochs):
            self.state.epoch = epoch
            epoch_start = time.time()
            epoch_losses = []

            # Iterate over batches
            for batch in dataset.iter_batches(
                batch_size=self.config.batch_size,
                shuffle=True,
            ):
                step_start = time.time()

                # Train step
                metrics = self._train_step(batch)
                epoch_losses.append(metrics["loss"])

                # Logging
                if self.state.global_step % self.config.log_every_n_steps == 0:
                    avg_loss = sum(epoch_losses[-100:]) / len(epoch_losses[-100:])
                    step_time = time.time() - step_start
                    logger.info(
                        f"Step {self.state.global_step} | "
                        f"Loss: {metrics['loss']:.4f} | "
                        f"Avg: {avg_loss:.4f} | "
                        f"LR: {metrics.get('lr', self.config.learning_rate):.2e} | "
                        f"Time: {step_time:.2f}s"
                    )

                # Callbacks
                if "on_step_end" in callbacks:
                    callbacks["on_step_end"](self, metrics)

                # Validation
                if (
                    val_dataset is not None
                    and self.state.global_step % self.config.validate_every_n_steps == 0
                ):
                    val_loss = self._validate(val_dataset)
                    logger.info(f"Validation loss: {val_loss:.4f}")

                    if "on_validation" in callbacks:
                        callbacks["on_validation"](self, val_loss)

                    # Save best model
                    if val_loss < self.state.best_loss:
                        self.state.best_loss = val_loss
                        self._save_checkpoint("best")

                # Checkpointing
                if self.state.global_step % self.config.save_every_n_steps == 0:
                    self._save_checkpoint(f"step_{self.state.global_step}")

            # Epoch end
            epoch_time = time.time() - epoch_start
            epoch_loss = sum(epoch_losses) / len(epoch_losses)
            logger.info(
                f"Epoch {epoch + 1}/{self.config.num_epochs} | "
                f"Loss: {epoch_loss:.4f} | "
                f"Time: {epoch_time:.1f}s"
            )

            if "on_epoch_end" in callbacks:
                callbacks["on_epoch_end"](self, {"epoch_loss": epoch_loss})

        # Final checkpoint
        self._save_checkpoint("final")
        logger.info("Training complete!")

    def _validate(self, dataset: AudioDataset) -> float:
        """Run validation and return average loss."""
        losses = []

        # Use EMA weights for validation
        self.ema.apply_shadow(self.model)

        for batch in dataset.iter_batches(
            batch_size=self.config.batch_size,
            shuffle=False,
            drop_last=False,
        ):
            loss = self._compute_loss(batch)
            _force_evaluate(loss)
            losses.append(float(loss))

        # Restore original weights
        self.ema.restore(self.model)

        return sum(losses) / len(losses) if losses else float("inf")

    def _save_checkpoint(self, name: str) -> None:
        """Save a checkpoint."""
        checkpoint_dir = self.config.output_dir / self.config.experiment_name / name
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        try:
            # Save LoRA weights
            if self.config.use_lora:
                save_lora_weights(self.model, checkpoint_dir / "lora_weights.safetensors")

            # Save training state
            state_path = checkpoint_dir / "training_state.json"
            with open(state_path, "w") as f:
                json.dump(self.state.state_dict(), f, indent=2)

            # Save optimizer state
            opt_path = checkpoint_dir / "optimizer.safetensors"
            opt_state = {"learning_rate": mx.array([self.optimizer.learning_rate])}
            mx.save_safetensors(str(opt_path), opt_state)

            # Save EMA weights if enabled
            if self.config.use_ema:
                self.ema.save(checkpoint_dir / "ema_weights.safetensors")

            logger.info(f"Saved checkpoint: {checkpoint_dir}")

        except Exception as e:
            logger.error(f"Failed to save checkpoint {name}: {e}")
            raise

        # Cleanup old checkpoints (non-critical, log errors but don't raise)
        try:
            self._cleanup_checkpoints()
        except Exception as e:
            logger.warning(f"Failed to cleanup old checkpoints: {e}")

    def _cleanup_checkpoints(self) -> None:
        """Remove old checkpoints, keeping only max_checkpoints."""
        exp_dir = self.config.output_dir / self.config.experiment_name
        if not exp_dir.exists():
            return

        # List step checkpoints (not 'best' or 'final')
        checkpoints = []
        for d in exp_dir.iterdir():
            if d.is_dir() and d.name.startswith("step_"):
                try:
                    step = int(d.name.split("_")[1])
                    checkpoints.append((step, d))
                except (ValueError, IndexError):
                    pass

        # Sort by step and remove old ones
        checkpoints.sort(key=lambda x: x[0])
        while len(checkpoints) > self.config.max_checkpoints:
            _, old_dir = checkpoints.pop(0)
            shutil.rmtree(old_dir)
            logger.info(f"Removed old checkpoint: {old_dir}")

    def load_checkpoint(self, checkpoint_dir: Path | str) -> None:
        """
        Load a checkpoint to resume training.

        Args:
            checkpoint_dir: Path to checkpoint directory. Must be within output_dir
                           to prevent path traversal attacks.

        Raises:
            ValueError: If checkpoint_dir is outside allowed paths.
            FileNotFoundError: If checkpoint directory doesn't exist.
        """
        checkpoint_dir = Path(checkpoint_dir).resolve()

        # Security: Validate checkpoint path is within allowed directory
        # Resolve symlinks and canonicalize to prevent path traversal
        allowed_base = self.config.output_dir.resolve()
        try:
            # Check that checkpoint is within output_dir
            checkpoint_dir.relative_to(allowed_base)
        except ValueError:
            raise ValueError(
                f"Checkpoint path must be within output directory: {allowed_base}. "
                f"Got: {checkpoint_dir}"
            )

        if not checkpoint_dir.exists():
            raise FileNotFoundError(f"Checkpoint directory not found: {checkpoint_dir}")

        if not checkpoint_dir.is_dir():
            raise ValueError(f"Checkpoint path is not a directory: {checkpoint_dir}")

        # Load LoRA weights
        lora_path = checkpoint_dir / "lora_weights.safetensors"
        if lora_path.exists():
            from .lora_layers import load_lora_weights

            load_lora_weights(self.model, lora_path, allowed_dir=allowed_base)

        # Load training state
        state_path = checkpoint_dir / "training_state.json"
        if state_path.exists():
            import json

            with open(state_path) as f:
                self.state.load_state_dict(json.load(f))

        # Load EMA weights
        ema_path = checkpoint_dir / "ema_weights.safetensors"
        if ema_path.exists() and self.config.use_ema:
            from mlx_music.training.common.ema import EMAModel

            self.ema = EMAModel.load(
                ema_path, self.model, self.config.ema_decay, allowed_dir=allowed_base
            )

        logger.info(f"Loaded checkpoint: {checkpoint_dir}")
