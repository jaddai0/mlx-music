"""
Cross-Entropy Loss for MusicGen Training.

MusicGen is an autoregressive model that predicts audio codebook tokens.
Training uses:
- Cross-entropy loss between predicted and target codebooks
- Teacher forcing: model sees ground truth tokens during training
- Causal masking: model can only attend to past tokens
- Multiple codebook streams (4 codebooks for 32kHz audio)

The model predicts logits over the codebook vocabulary at each position,
and we compute cross-entropy against the target codebook indices.

Reference:
    "Simple and Controllable Music Generation" (Copet et al., 2023)
"""

import random
from typing import Optional, Tuple

import mlx.core as mx


class MusicGenLoss:
    """
    Cross-entropy loss for MusicGen codebook prediction.

    MusicGen uses 4 parallel codebook streams (for EnCodec at 32kHz).
    Each stream predicts indices into a 2048-token vocabulary.

    The loss is computed per-codebook and averaged:
        loss = mean([CE(pred_i, target_i) for i in codebooks])
    """

    @staticmethod
    def compute_loss(
        logits: mx.array,
        targets: mx.array,
        mask: Optional[mx.array] = None,
        label_smoothing: float = 0.0,
        reduction: str = "mean",
    ) -> mx.array:
        """
        Compute cross-entropy loss for codebook predictions.

        Args:
            logits: Model predictions [B, seq, vocab_size] or [B, seq, num_codebooks, vocab_size]
            targets: Target codebook indices [B, seq] or [B, seq, num_codebooks]
            mask: Optional mask for valid positions [B, seq]
            label_smoothing: Label smoothing factor (0.0 = no smoothing)
            reduction: How to reduce loss ("mean", "sum", "none")

        Returns:
            Loss value
        """
        # Handle multi-codebook format
        if logits.ndim == 4:
            # [B, seq, num_codebooks, vocab_size]
            batch, seq_len, num_codebooks, vocab_size = logits.shape
            # Reshape to [B * num_codebooks, seq, vocab_size]
            logits = logits.transpose(0, 2, 1, 3).reshape(-1, seq_len, vocab_size)
            targets = targets.transpose(0, 2, 1).reshape(-1, seq_len)
            if mask is not None:
                # Expand mask for all codebooks
                mask = mx.repeat(mask, num_codebooks, axis=0)

        # Compute log softmax for numerical stability
        log_probs = mx.softmax(logits, axis=-1)
        log_probs = mx.log(log_probs + 1e-10)

        # Gather log probs for target indices
        # targets: [B, seq] -> one-hot: [B, seq, vocab_size]
        batch_size, seq_len, vocab_size = logits.shape

        # Create indices for gathering
        batch_idx = mx.arange(batch_size)[:, None]
        seq_idx = mx.arange(seq_len)[None, :]

        # Gather log probs at target positions
        # This is equivalent to: log_probs[batch_idx, seq_idx, targets]
        target_log_probs = mx.take_along_axis(
            log_probs, targets[..., None], axis=-1
        ).squeeze(-1)

        # Negative log likelihood
        loss = -target_log_probs

        # Label smoothing
        if label_smoothing > 0.0:
            # Smooth loss = (1 - eps) * nll + eps * uniform_loss
            uniform_loss = -log_probs.mean(axis=-1)
            loss = (1 - label_smoothing) * loss + label_smoothing * uniform_loss

        # Apply mask if provided
        if mask is not None:
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

    @staticmethod
    def compute_accuracy(
        logits: mx.array,
        targets: mx.array,
        mask: Optional[mx.array] = None,
    ) -> mx.array:
        """
        Compute prediction accuracy for monitoring.

        Args:
            logits: Model predictions [B, seq, vocab_size]
            targets: Target indices [B, seq]
            mask: Optional mask for valid positions

        Returns:
            Accuracy (0-1 range)
        """
        predictions = mx.argmax(logits, axis=-1)
        correct = (predictions == targets).astype(mx.float32)

        if mask is not None:
            return (correct * mask).sum() / (mask.sum() + 1e-8)

        return correct.mean()

    @classmethod
    def forward(
        cls,
        model,  # MusicGen decoder
        input_codes: mx.array,
        encoder_hidden_states: mx.array,
        target_codes: Optional[mx.array] = None,
        attention_mask: Optional[mx.array] = None,
        label_smoothing: float = 0.1,
        reduction: str = "mean",
    ) -> Tuple[mx.array, mx.array]:
        """
        Complete forward pass for training.

        Uses teacher forcing: the model receives ground truth input_codes
        and predicts the next token at each position.

        Args:
            model: MusicGen decoder model
            input_codes: Input codebook tokens [B, seq, num_codebooks]
                For teacher forcing, this is target_codes shifted right
            encoder_hidden_states: Text encoder output [B, enc_seq, dim]
            target_codes: Target codebook tokens [B, seq, num_codebooks]
                If None, uses input_codes shifted left
            attention_mask: Mask for valid positions [B, seq]
            label_smoothing: Label smoothing factor
            reduction: Loss reduction method

        Returns:
            Tuple of (loss, logits)

        Example:
            # Prepare input (shift target right by 1)
            input_codes = shift_codes_right(target_codes)

            loss, logits = MusicGenLoss.forward(
                model=decoder,
                input_codes=input_codes,
                encoder_hidden_states=text_embeds,
                target_codes=target_codes,
            )
        """
        # Default: target is input shifted left (next token prediction)
        if target_codes is None:
            target_codes = input_codes  # Model predicts current position

        # Forward through model
        # MusicGen decoder expects: (input_codes, encoder_hidden_states, ...)
        logits = model(
            input_ids=input_codes,
            encoder_hidden_states=encoder_hidden_states,
            attention_mask=attention_mask,
        )

        # Compute loss
        loss = cls.compute_loss(
            logits=logits,
            targets=target_codes,
            mask=attention_mask,
            label_smoothing=label_smoothing,
            reduction=reduction,
        )

        return loss, logits


def shift_codes_right(codes: mx.array, pad_token_id: int = 0) -> mx.array:
    """
    Shift codebook tokens right for teacher forcing.

    Prepends a pad token and removes the last token.

    Args:
        codes: Token codes [B, seq, num_codebooks] or [B, seq]
        pad_token_id: Token to prepend

    Returns:
        Shifted codes (same shape)
    """
    if codes.ndim == 2:
        # [B, seq]
        pad = mx.full((codes.shape[0], 1), pad_token_id, dtype=codes.dtype)
        return mx.concatenate([pad, codes[:, :-1]], axis=1)
    elif codes.ndim == 3:
        # [B, seq, num_codebooks]
        pad = mx.full((codes.shape[0], 1, codes.shape[2]), pad_token_id, dtype=codes.dtype)
        return mx.concatenate([pad, codes[:, :-1, :]], axis=1)
    else:
        raise ValueError(f"Expected 2D or 3D codes, got {codes.ndim}D")


def create_causal_mask(seq_len: int, dtype: mx.Dtype = mx.float32) -> mx.array:
    """
    Create causal attention mask.

    Args:
        seq_len: Sequence length
        dtype: Data type

    Returns:
        Causal mask [seq_len, seq_len] where True = attend, False = mask
    """
    # Lower triangular mask (attend to past and current)
    mask = mx.tril(mx.ones((seq_len, seq_len), dtype=dtype))
    return mask
