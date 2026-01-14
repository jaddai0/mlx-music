"""
Cached DSP utilities for mlx-music.

Window functions, mel filterbanks, and STFT helpers with automatic caching
to avoid redundant computation. Following mlx-audio patterns for efficiency.
"""

import math
from functools import lru_cache
from typing import Optional, Literal

import mlx.core as mx
import numpy as np

__all__ = [
    # Window functions
    "hanning",
    "hamming",
    "blackman",
    "bartlett",
    "get_window",
    "STR_TO_WINDOW_FN",
    # Mel filterbank
    "mel_filterbank",
    "hz_to_mel",
    "mel_to_hz",
    # STFT helpers
    "get_stft_window",
    # Cache management
    "clear_dsp_cache",
    "dsp_cache_info",
]


# =============================================================================
# Cached Window Functions
# =============================================================================


@lru_cache(maxsize=None)
def hanning(size: int, periodic: bool = False) -> mx.array:
    """
    Hanning (Hann) window with caching.

    Args:
        size: Window length
        periodic: If True, use periodic window (for spectral analysis)

    Returns:
        Hann window array
    """
    denom = size if periodic else size - 1
    return mx.array(
        [0.5 * (1 - math.cos(2 * math.pi * n / denom)) for n in range(size)],
        dtype=mx.float32,
    )


@lru_cache(maxsize=None)
def hamming(size: int, periodic: bool = False) -> mx.array:
    """
    Hamming window with caching.

    Args:
        size: Window length
        periodic: If True, use periodic window (for spectral analysis)

    Returns:
        Hamming window array
    """
    denom = size if periodic else size - 1
    return mx.array(
        [0.54 - 0.46 * math.cos(2 * math.pi * n / denom) for n in range(size)],
        dtype=mx.float32,
    )


@lru_cache(maxsize=None)
def blackman(size: int, periodic: bool = False) -> mx.array:
    """
    Blackman window with caching.

    Args:
        size: Window length
        periodic: If True, use periodic window (for spectral analysis)

    Returns:
        Blackman window array
    """
    denom = size if periodic else size - 1
    return mx.array(
        [
            0.42
            - 0.5 * math.cos(2 * math.pi * n / denom)
            + 0.08 * math.cos(4 * math.pi * n / denom)
            for n in range(size)
        ],
        dtype=mx.float32,
    )


@lru_cache(maxsize=None)
def bartlett(size: int, periodic: bool = False) -> mx.array:
    """
    Bartlett (triangular) window with caching.

    Args:
        size: Window length
        periodic: If True, use periodic window

    Returns:
        Bartlett window array
    """
    denom = size if periodic else size - 1
    return mx.array(
        [1 - 2 * abs(n - denom / 2) / denom for n in range(size)],
        dtype=mx.float32,
    )


# Window function lookup
STR_TO_WINDOW_FN = {
    "hann": hanning,
    "hanning": hanning,
    "hamming": hamming,
    "blackman": blackman,
    "bartlett": bartlett,
}


def get_window(
    window: str | mx.array,
    size: int,
    periodic: bool = False,
) -> mx.array:
    """
    Get a window function by name or return the provided array.

    Args:
        window: Window name ("hann", "hamming", etc.) or mx.array
        size: Window size (ignored if window is an array)
        periodic: Whether to use periodic window

    Returns:
        Window function array
    """
    if isinstance(window, mx.array):
        return window

    window_fn = STR_TO_WINDOW_FN.get(window.lower())
    if window_fn is None:
        raise ValueError(
            f"Unknown window function: {window}. "
            f"Available: {list(STR_TO_WINDOW_FN.keys())}"
        )
    return window_fn(size, periodic)


@lru_cache(maxsize=32)
def get_stft_window(
    win_length: int,
    n_fft: int,
    window_type: str = "hann",
) -> mx.array:
    """
    Get a padded STFT window with caching.

    Args:
        win_length: Window length
        n_fft: FFT size
        window_type: Window function name

    Returns:
        Padded window ready for STFT
    """
    window = get_window(window_type, win_length)

    if win_length < n_fft:
        pad_amount = n_fft - win_length
        pad_left = pad_amount // 2
        pad_right = pad_amount - pad_left
        window = mx.pad(window, [(pad_left, pad_right)])

    return window


# =============================================================================
# Mel Scale Utilities
# =============================================================================


def hz_to_mel(freq: float, scale: Literal["htk", "slaney"] = "htk") -> float:
    """
    Convert frequency in Hz to mel scale.

    Args:
        freq: Frequency in Hz
        scale: Mel scale formula ("htk" or "slaney")

    Returns:
        Mel frequency
    """
    if scale == "htk":
        return 2595.0 * math.log10(1.0 + freq / 700.0)

    # Slaney scale
    f_min, f_sp = 0.0, 200.0 / 3
    mels = (freq - f_min) / f_sp

    min_log_hz = 1000.0
    min_log_mel = (min_log_hz - f_min) / f_sp
    logstep = math.log(6.4) / 27.0

    if freq >= min_log_hz:
        mels = min_log_mel + math.log(freq / min_log_hz) / logstep

    return mels


def mel_to_hz(mels: float, scale: Literal["htk", "slaney"] = "htk") -> float:
    """
    Convert mel frequency to Hz.

    Args:
        mels: Mel frequency
        scale: Mel scale formula ("htk" or "slaney")

    Returns:
        Frequency in Hz
    """
    if scale == "htk":
        return 700.0 * (10.0 ** (mels / 2595.0) - 1.0)

    # Slaney scale
    f_min, f_sp = 0.0, 200.0 / 3
    freqs = f_min + f_sp * mels

    min_log_hz = 1000.0
    min_log_mel = (min_log_hz - f_min) / f_sp
    logstep = math.log(6.4) / 27.0

    if mels >= min_log_mel:
        freqs = min_log_hz * math.exp(logstep * (mels - min_log_mel))

    return freqs


# =============================================================================
# Cached Mel Filterbank
# =============================================================================


@lru_cache(maxsize=32)
def mel_filterbank(
    n_mels: int = 128,
    n_fft: int = 2048,
    sample_rate: int = 44100,
    f_min: float = 0.0,
    f_max: Optional[float] = None,
    norm: Optional[Literal["slaney"]] = None,
    mel_scale: Literal["htk", "slaney"] = "htk",
) -> mx.array:
    """
    Create a mel filterbank matrix with caching.

    Args:
        n_mels: Number of mel bands
        n_fft: FFT size
        sample_rate: Audio sample rate
        f_min: Minimum frequency
        f_max: Maximum frequency (default: sample_rate / 2)
        norm: Normalization mode ("slaney" or None)
        mel_scale: Mel scale formula ("htk" or "slaney")

    Returns:
        Filterbank matrix of shape (n_mels, n_fft // 2 + 1)
    """
    if f_max is None:
        f_max = sample_rate / 2.0

    n_freqs = n_fft // 2 + 1
    all_freqs = np.linspace(0, sample_rate / 2, n_freqs)

    # Convert to mel scale
    m_min = hz_to_mel(f_min, mel_scale)
    m_max = hz_to_mel(f_max, mel_scale)
    m_pts = np.linspace(m_min, m_max, n_mels + 2)

    # Convert back to Hz
    f_pts = np.array([mel_to_hz(m, mel_scale) for m in m_pts])

    # Compute slopes
    f_diff = f_pts[1:] - f_pts[:-1]
    slopes = np.expand_dims(f_pts, 0) - np.expand_dims(all_freqs, 1)

    # Calculate overlapping triangular filters
    down_slopes = (-slopes[:, :-2]) / (f_diff[:-1] + 1e-10)
    up_slopes = slopes[:, 2:] / (f_diff[1:] + 1e-10)
    filterbank = np.maximum(0.0, np.minimum(down_slopes, up_slopes))

    # Apply normalization
    if norm == "slaney":
        enorm = 2.0 / (f_pts[2 : n_mels + 2] - f_pts[:n_mels] + 1e-10)
        filterbank *= np.expand_dims(enorm, 0)

    # Transpose to (n_mels, n_freqs)
    filterbank = filterbank.T

    return mx.array(filterbank, dtype=mx.float32)


# =============================================================================
# ISTFT Cache (for efficient overlap-add)
# =============================================================================


class ISTFTCache:
    """
    Advanced caching for iSTFT operations.

    Caches normalization buffers and position indices for efficient
    vectorized overlap-add reconstruction.
    """

    def __init__(self):
        self.norm_buffer_cache = {}
        self.position_cache = {}

    def get_positions(
        self,
        num_frames: int,
        frame_length: int,
        hop_length: int,
    ) -> mx.array:
        """Get cached position indices or create new ones."""
        key = (num_frames, frame_length, hop_length)

        if key not in self.position_cache:
            positions = (
                mx.arange(num_frames)[:, None] * hop_length
                + mx.arange(frame_length)[None, :]
            )
            self.position_cache[key] = positions.reshape(-1)

        return self.position_cache[key]

    def get_norm_buffer(
        self,
        n_fft: int,
        hop_length: int,
        window: mx.array,
        num_frames: int,
    ) -> mx.array:
        """Get cached normalization buffer or create new one."""
        # Use window content hash for cache key
        window_hash = hash(tuple(window.tolist()))
        key = (n_fft, hop_length, window_hash, num_frames)

        if key not in self.norm_buffer_cache:
            frame_length = window.shape[0]
            ola_len = (num_frames - 1) * hop_length + frame_length
            positions_flat = self.get_positions(num_frames, frame_length, hop_length)

            window_squared = window**2
            norm_buffer = mx.zeros(ola_len, dtype=mx.float32)
            window_sq_tiled = mx.tile(window_squared, (num_frames,))
            norm_buffer = norm_buffer.at[positions_flat].add(window_sq_tiled)
            norm_buffer = mx.maximum(norm_buffer, 1e-10)

            self.norm_buffer_cache[key] = norm_buffer

        return self.norm_buffer_cache[key]

    def clear(self):
        """Clear all cached data to free memory."""
        self.norm_buffer_cache.clear()
        self.position_cache.clear()

    def info(self) -> dict:
        """Get information about cached items."""
        return {
            "norm_buffers": len(self.norm_buffer_cache),
            "position_indices": len(self.position_cache),
            "total_cached_items": (
                len(self.norm_buffer_cache) + len(self.position_cache)
            ),
        }


# Global ISTFT cache instance
_istft_cache = ISTFTCache()


def get_istft_cache() -> ISTFTCache:
    """Get the global ISTFT cache instance."""
    return _istft_cache


# =============================================================================
# Cache Management
# =============================================================================


def clear_dsp_cache():
    """Clear all DSP caches to free memory."""
    hanning.cache_clear()
    hamming.cache_clear()
    blackman.cache_clear()
    bartlett.cache_clear()
    get_stft_window.cache_clear()
    mel_filterbank.cache_clear()
    _istft_cache.clear()


def dsp_cache_info() -> dict:
    """Get information about all DSP caches."""
    return {
        "hanning": hanning.cache_info()._asdict(),
        "hamming": hamming.cache_info()._asdict(),
        "blackman": blackman.cache_info()._asdict(),
        "bartlett": bartlett.cache_info()._asdict(),
        "stft_window": get_stft_window.cache_info()._asdict(),
        "mel_filterbank": mel_filterbank.cache_info()._asdict(),
        "istft_cache": _istft_cache.info(),
    }
