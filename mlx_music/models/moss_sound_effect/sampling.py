"""Token sampling utilities for MOSS-SoundEffect (MLX)."""

from __future__ import annotations

import mlx.core as mx


def top_k_filter(logits: mx.array, k: int) -> mx.array:
    """Keep only top-k logits, set rest to -inf.

    Args:
        logits: (..., V) logits array.
        k: Number of top values to keep.
    """
    if k <= 0:
        return logits

    V = logits.shape[-1]
    k = min(k, V)

    # Get top-k values (mx.topk returns ascending order)
    top_vals = mx.topk(logits, k, axis=-1)
    # Threshold = minimum of top-k (first element in ascending order)
    threshold = top_vals[..., :1]
    return mx.where(logits >= threshold, logits, mx.array(float("-inf")))


def top_p_filter(logits: mx.array, p: float) -> mx.array:
    """Nucleus sampling: keep smallest set of tokens with cumulative prob >= p.

    Args:
        logits: (..., V) logits array.
        p: Cumulative probability threshold.
    """
    if p >= 1.0:
        return logits

    probs = mx.softmax(logits, axis=-1)
    sorted_indices = mx.argsort(-probs, axis=-1)
    sorted_probs = mx.take_along_axis(probs, sorted_indices, axis=-1)

    cumulative_probs = mx.cumsum(sorted_probs, axis=-1)

    # Create mask: remove tokens after cumulative prob exceeds p
    # Shift right by 1 to keep the token that crosses the threshold
    sorted_mask = cumulative_probs - sorted_probs > p

    # Scatter mask back to original positions
    # Create -inf mask in sorted order, then unsort
    sorted_logits = mx.take_along_axis(logits, sorted_indices, axis=-1)
    sorted_logits = mx.where(sorted_mask, mx.array(float("-inf")), sorted_logits)

    # Unsort: create inverse permutation
    # Use argsort of sorted_indices to get inverse mapping
    inv_indices = mx.argsort(sorted_indices, axis=-1)
    filtered = mx.take_along_axis(sorted_logits, inv_indices, axis=-1)

    return filtered


def repetition_penalty(
    logits: mx.array,
    prev_tokens: mx.array,
    penalty: float,
) -> mx.array:
    """Apply repetition penalty to logits.

    For tokens that appeared in prev_tokens:
    - If logit > 0: divide by penalty (reduce repeated high-prob tokens)
    - If logit <= 0: multiply by penalty (make repeated low-prob tokens even less likely)

    Args:
        logits: (N, V) or (B, H, V) logits.
        prev_tokens: Previous token IDs to penalize.
        penalty: Penalty factor (1.0 = no penalty).
    """
    if penalty == 1.0:
        return logits

    if logits.ndim == 2:
        # Text case: (N, V)
        prev_flat = prev_tokens.reshape(-1)
        unique_tokens = mx.array(list(set(prev_flat.tolist())))

        if unique_tokens.size == 0:
            return logits

        for tok in unique_tokens.tolist():
            tok_int = int(tok)
            if tok_int < 0 or tok_int >= logits.shape[-1]:
                continue
            col = logits[:, tok_int]
            col = mx.where(col > 0, col / penalty, col * penalty)
            logits[:, tok_int] = col
        return logits

    elif logits.ndim == 3:
        # Audio delay pattern: (B, H, V)
        B, H, V = logits.shape
        for h in range(H):
            prev_h = prev_tokens[..., h].reshape(-1)
            unique_tokens = mx.array(list(set(prev_h.tolist())))

            for tok in unique_tokens.tolist():
                tok_int = int(tok)
                if tok_int < 0 or tok_int >= V:
                    continue
                col = logits[:, h, tok_int]
                col = mx.where(col > 0, col / penalty, col * penalty)
                logits[:, h, tok_int] = col
        return logits

    return logits


def sample_token(
    logits: mx.array,
    temperature: float = 1.0,
    top_k: int = 0,
    top_p: float = 1.0,
    do_sample: bool = True,
) -> mx.array:
    """Sample a token from logits.

    Args:
        logits: (..., V) logits.
        temperature: Sampling temperature.
        top_k: Top-k filtering (0 = disabled).
        top_p: Nucleus sampling threshold (1.0 = disabled).
        do_sample: If False, use greedy argmax.

    Returns:
        tokens: (...) sampled token indices.
    """
    if not do_sample:
        return mx.argmax(logits, axis=-1)

    V = logits.shape[-1]
    original_shape = logits.shape[:-1]

    # Flatten for filtering
    flat = logits.reshape(-1, V)

    if top_k > 0:
        flat = top_k_filter(flat, top_k)
    if top_p < 1.0:
        flat = top_p_filter(flat, top_p)

    # Sample
    probs = mx.softmax(flat, axis=-1)
    tokens = mx.random.categorical(mx.log(probs + 1e-10))

    return tokens.reshape(original_shape)
