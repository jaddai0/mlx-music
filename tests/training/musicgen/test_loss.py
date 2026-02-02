"""
Tests for MusicGen loss functions.

Tests:
- MusicGenLoss: compute_loss, compute_accuracy
- shift_codes_right: Teacher forcing input preparation
- create_causal_mask: Causal attention masking
"""

import pytest
import mlx.core as mx


class TestMusicGenLoss:
    """Tests for MusicGenLoss class."""

    def test_compute_loss_cross_entropy(self):
        """compute_loss should compute cross-entropy."""
        from mlx_music.training.musicgen.loss import MusicGenLoss

        batch_size = 2
        seq_len = 10
        vocab_size = 100

        # Create logits and targets
        logits = mx.random.normal((batch_size, seq_len, vocab_size))
        targets = mx.random.randint(0, vocab_size, (batch_size, seq_len))

        loss = MusicGenLoss.compute_loss(logits, targets)

        assert loss.ndim == 0  # Scalar
        assert not mx.isnan(loss)
        assert float(loss) > 0  # CE loss is always positive

    def test_compute_loss_perfect_prediction(self):
        """Perfect prediction should have low loss."""
        from mlx_music.training.musicgen.loss import MusicGenLoss

        batch_size = 2
        seq_len = 5
        vocab_size = 10

        # Create perfect logits (very high at target position)
        targets = mx.array([[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]])

        # Create logits with high values at target positions using one-hot
        # Start with low values, then add 20 at target positions
        logits = mx.full((batch_size, seq_len, vocab_size), -10.0)
        one_hot = mx.eye(vocab_size)[targets]  # [batch, seq, vocab]
        logits = logits + 20.0 * one_hot  # -10 + 20 = 10 at target positions

        loss = MusicGenLoss.compute_loss(logits, targets)

        # Loss should be very low (near 0)
        assert float(loss) < 0.1

    def test_compute_loss_with_mask(self):
        """compute_loss should apply mask correctly."""
        from mlx_music.training.musicgen.loss import MusicGenLoss

        batch_size = 2
        seq_len = 10
        vocab_size = 100

        logits = mx.random.normal((batch_size, seq_len, vocab_size))
        targets = mx.random.randint(0, vocab_size, (batch_size, seq_len))

        # Mask out second half - create mask with 1s in first half
        mask = mx.concatenate([
            mx.ones((batch_size, 5)),
            mx.zeros((batch_size, 5))
        ], axis=1)

        loss_masked = MusicGenLoss.compute_loss(logits, targets, mask=mask)
        loss_full = MusicGenLoss.compute_loss(logits, targets)

        # Losses should differ
        assert float(loss_masked) != float(loss_full)

    def test_compute_loss_label_smoothing(self):
        """compute_loss with label_smoothing should smooth targets."""
        from mlx_music.training.musicgen.loss import MusicGenLoss

        logits = mx.random.normal((2, 10, 100))
        targets = mx.random.randint(0, 100, (2, 10))

        loss_no_smooth = MusicGenLoss.compute_loss(logits, targets, label_smoothing=0.0)
        loss_smooth = MusicGenLoss.compute_loss(logits, targets, label_smoothing=0.1)

        # Smoothed loss should be different
        assert float(loss_no_smooth) != float(loss_smooth)

    def test_compute_loss_reduction_none(self):
        """compute_loss with reduction='none' returns per-element loss."""
        from mlx_music.training.musicgen.loss import MusicGenLoss

        batch_size = 2
        seq_len = 10
        vocab_size = 100

        logits = mx.random.normal((batch_size, seq_len, vocab_size))
        targets = mx.random.randint(0, vocab_size, (batch_size, seq_len))

        loss = MusicGenLoss.compute_loss(logits, targets, reduction="none")

        assert loss.shape == (batch_size, seq_len)

    def test_compute_loss_multi_codebook(self):
        """compute_loss should handle multi-codebook format."""
        from mlx_music.training.musicgen.loss import MusicGenLoss

        batch_size = 2
        seq_len = 10
        num_codebooks = 4
        vocab_size = 2048

        # Multi-codebook format: [B, seq, num_codebooks, vocab_size]
        logits = mx.random.normal((batch_size, seq_len, num_codebooks, vocab_size))
        targets = mx.random.randint(0, vocab_size, (batch_size, seq_len, num_codebooks))

        loss = MusicGenLoss.compute_loss(logits, targets)

        assert loss.ndim == 0  # Scalar
        assert not mx.isnan(loss)

    def test_compute_accuracy(self):
        """compute_accuracy should compute prediction accuracy."""
        from mlx_music.training.musicgen.loss import MusicGenLoss

        # Perfect predictions: all predict class 0
        # Create logits with class 0 having highest value
        logits = mx.zeros((2, 5, 10))
        # Add 10 to class 0 for all positions
        class_0_boost = mx.zeros((10,))
        class_0_boost = mx.concatenate([mx.array([10.0]), mx.zeros((9,))])
        logits = logits + class_0_boost  # Broadcast to all batch/seq positions

        targets = mx.zeros((2, 5), dtype=mx.int32)  # Target is class 0

        accuracy = MusicGenLoss.compute_accuracy(logits, targets)

        assert float(accuracy) == pytest.approx(1.0)  # 100% accuracy

    def test_compute_accuracy_with_mask(self):
        """compute_accuracy should respect mask."""
        from mlx_music.training.musicgen.loss import MusicGenLoss

        # Half correct, half wrong
        # Positions 0,1: predict class 0 (correct)
        # Positions 2,3: predict class 5 (wrong, target is 0)
        logits = mx.zeros((1, 4, 10))

        # Create correct predictions for positions 0, 1
        correct_row = mx.zeros((10,))
        correct_row = mx.concatenate([mx.array([10.0]), mx.zeros((9,))])

        # Create wrong predictions for positions 2, 3
        wrong_row = mx.zeros((10,))
        wrong_row = mx.concatenate([mx.zeros((5,)), mx.array([10.0]), mx.zeros((4,))])

        # Stack: [1, 4, 10]
        logits = mx.stack([
            mx.stack([correct_row, correct_row, wrong_row, wrong_row])
        ])

        targets = mx.zeros((1, 4), dtype=mx.int32)

        # Only consider first two positions
        mask = mx.array([[1.0, 1.0, 0.0, 0.0]])

        accuracy = MusicGenLoss.compute_accuracy(logits, targets, mask=mask)

        assert float(accuracy) == pytest.approx(1.0)  # 100% on masked positions


class TestShiftCodesRight:
    """Tests for shift_codes_right function."""

    def test_shift_codes_2d(self):
        """shift_codes_right should shift 2D codes."""
        from mlx_music.training.musicgen.loss import shift_codes_right

        codes = mx.array([[1, 2, 3, 4, 5]])

        shifted = shift_codes_right(codes, pad_token_id=0)

        expected = mx.array([[0, 1, 2, 3, 4]])
        assert mx.array_equal(shifted, expected)

    def test_shift_codes_3d(self):
        """shift_codes_right should shift 3D codes (multi-codebook)."""
        from mlx_music.training.musicgen.loss import shift_codes_right

        # [batch=1, seq=4, codebooks=2]
        codes = mx.array([[[1, 2], [3, 4], [5, 6], [7, 8]]])

        shifted = shift_codes_right(codes, pad_token_id=0)

        # First position should be pad, others shifted right
        assert shifted.shape == (1, 4, 2)
        assert mx.array_equal(shifted[0, 0], mx.array([0, 0]))  # Pad
        assert mx.array_equal(shifted[0, 1], mx.array([1, 2]))  # Original pos 0

    def test_shift_codes_preserves_shape(self):
        """shift_codes_right should preserve shape."""
        from mlx_music.training.musicgen.loss import shift_codes_right

        for shape in [(2, 10), (4, 20, 3)]:
            codes = mx.random.randint(0, 100, shape)
            shifted = shift_codes_right(codes)
            assert shifted.shape == codes.shape


class TestCreateCausalMask:
    """Tests for create_causal_mask function."""

    def test_causal_mask_shape(self):
        """create_causal_mask should return correct shape."""
        from mlx_music.training.musicgen.loss import create_causal_mask

        mask = create_causal_mask(seq_len=10)

        assert mask.shape == (10, 10)

    def test_causal_mask_lower_triangular(self):
        """create_causal_mask should be lower triangular."""
        from mlx_music.training.musicgen.loss import create_causal_mask

        mask = create_causal_mask(seq_len=5)

        # Check lower triangular structure
        # Position (i, j) should be 1 if j <= i, else 0
        for i in range(5):
            for j in range(5):
                expected = 1.0 if j <= i else 0.0
                assert float(mask[i, j]) == expected

    def test_causal_mask_dtype(self):
        """create_causal_mask should respect dtype parameter."""
        from mlx_music.training.musicgen.loss import create_causal_mask

        mask_f32 = create_causal_mask(seq_len=5, dtype=mx.float32)
        mask_f16 = create_causal_mask(seq_len=5, dtype=mx.float16)

        assert mask_f32.dtype == mx.float32
        assert mask_f16.dtype == mx.float16
