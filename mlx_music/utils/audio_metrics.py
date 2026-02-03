"""
Audio quality metrics for mlx-music.

Provides perceptual quality metrics and audio normalization utilities:
- RMS energy and dynamic range
- Spectral analysis (centroid, bandwidth, rolloff)
- Clipping detection
- Loudness normalization (EBU R128 style)
- SI-SNR (scale-invariant signal-to-noise ratio)
"""

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np


@dataclass
class AudioQualityMetrics:
    """Container for audio quality metrics."""

    # Basic metrics
    rms_energy: float
    peak_amplitude: float
    dynamic_range_db: float

    # Spectral metrics
    spectral_centroid_hz: float
    spectral_bandwidth_hz: float
    spectral_rolloff_hz: float

    # Quality flags
    is_silent: bool
    is_clipped: bool
    clipped_sample_count: int

    # Duration info
    duration_sec: float
    sample_rate: int

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "rms_energy": self.rms_energy,
            "peak_amplitude": self.peak_amplitude,
            "dynamic_range_db": self.dynamic_range_db,
            "spectral_centroid_hz": self.spectral_centroid_hz,
            "spectral_bandwidth_hz": self.spectral_bandwidth_hz,
            "spectral_rolloff_hz": self.spectral_rolloff_hz,
            "is_silent": self.is_silent,
            "is_clipped": self.is_clipped,
            "clipped_sample_count": self.clipped_sample_count,
            "duration_sec": self.duration_sec,
            "sample_rate": self.sample_rate,
        }


# Maximum audio size to prevent memory exhaustion (10 minutes at 48kHz stereo)
MAX_AUDIO_SAMPLES = 48000 * 60 * 10 * 2

# Minimum threshold to prevent division by zero in audio calculations
EPSILON = 1e-10

# Valid sample rate range (Hz) - covers common audio formats
MIN_SAMPLE_RATE = 8000    # Telephone quality
MAX_SAMPLE_RATE = 192000  # High-resolution audio


def _ensure_mono(audio: np.ndarray) -> np.ndarray:
    """
    Ensure audio is mono (1D array).

    Handles various input shapes:
    - (samples,) -> unchanged
    - (channels, samples) -> mean across channels
    - (samples, channels) -> transpose then mean

    Raises:
        ValueError: If audio has more than 2 dimensions or exceeds size limit
    """
    audio = np.asarray(audio)

    # Validate dimensions
    if audio.ndim > 2:
        raise ValueError(f"Audio must be 1D or 2D array, got {audio.ndim}D")

    # Validate size to prevent memory exhaustion
    if audio.size > MAX_AUDIO_SAMPLES:
        raise ValueError(f"Audio too large ({audio.size} samples, max {MAX_AUDIO_SAMPLES})")

    if audio.ndim == 1:
        return audio
    elif audio.ndim == 2:
        # Determine which axis is channels vs samples
        if audio.shape[0] <= 2 and audio.shape[1] > 2:
            # (channels, samples) format
            return audio.mean(axis=0)
        elif audio.shape[1] <= 2 and audio.shape[0] > 2:
            # (samples, channels) format
            return audio.mean(axis=1)
        else:
            # Ambiguous, assume first axis is channels
            return audio.mean(axis=0)
    else:
        # This shouldn't be reached due to ndim check above
        return audio.flatten()


def _compute_rms_mono(audio_mono: np.ndarray) -> float:
    """Compute RMS on pre-converted mono audio (internal use)."""
    return float(np.sqrt(np.mean(audio_mono**2)))


def compute_rms(audio: np.ndarray) -> float:
    """Compute RMS (root mean square) energy."""
    return _compute_rms_mono(_ensure_mono(audio))


def _compute_dynamic_range_mono(audio_mono: np.ndarray, floor_percentile: float = 0.1) -> float:
    """Compute dynamic range on pre-converted mono audio (internal use)."""
    audio_abs = np.abs(audio_mono)

    max_val = audio_abs.max()
    non_zero = audio_abs[audio_abs > 0]

    if len(non_zero) == 0 or max_val == 0:
        return 0.0

    min_val = np.percentile(non_zero, floor_percentile)

    if min_val <= 0:
        min_val = EPSILON

    return float(20 * np.log10(max_val / min_val))


def compute_dynamic_range(audio: np.ndarray, floor_percentile: float = 0.1) -> float:
    """
    Compute dynamic range in dB.

    Uses percentile for floor to avoid silence affecting the measurement.

    Args:
        audio: Audio waveform
        floor_percentile: Percentile for minimum level (default: 0.1%)

    Returns:
        Dynamic range in dB
    """
    return _compute_dynamic_range_mono(_ensure_mono(audio), floor_percentile)


def _compute_fft_once(
    audio_mono: np.ndarray,
    sample_rate: int,
    frame_length: int = 2048,
) -> tuple:
    """
    Compute FFT and frequency bins once for all spectral metrics.

    Returns:
        Tuple of (fft_magnitudes, frequencies, magnitude_sum, audio_padded)
        Returns (None, None, 0.0, audio_mono) if magnitude_sum too small.
    """
    # Pad audio if needed (do once for all spectral functions)
    if len(audio_mono) < frame_length:
        audio_mono = np.pad(audio_mono, (0, frame_length - len(audio_mono)))

    fft = np.abs(np.fft.rfft(audio_mono[:frame_length]))
    freqs = np.fft.rfftfreq(frame_length, 1 / sample_rate)
    magnitude_sum = np.sum(fft)

    if magnitude_sum < EPSILON:
        return None, None, 0.0, audio_mono

    return fft, freqs, magnitude_sum, audio_mono


def _spectral_centroid_from_fft(
    fft: np.ndarray,
    freqs: np.ndarray,
    magnitude_sum: float,
) -> float:
    """Compute spectral centroid from pre-computed FFT (internal use)."""
    if fft is None or magnitude_sum < EPSILON:
        return 0.0
    return float(np.sum(freqs * fft) / magnitude_sum)


def _spectral_bandwidth_from_fft(
    fft: np.ndarray,
    freqs: np.ndarray,
    magnitude_sum: float,
    centroid: float,
) -> float:
    """Compute spectral bandwidth from pre-computed FFT and centroid (internal use)."""
    if fft is None or magnitude_sum < EPSILON:
        return 0.0
    variance = np.sum(((freqs - centroid) ** 2) * fft) / magnitude_sum
    return float(np.sqrt(variance))


def _spectral_rolloff_from_fft(
    fft: np.ndarray,
    freqs: np.ndarray,
    rolloff_percent: float = 0.85,
) -> float:
    """Compute spectral rolloff from pre-computed FFT (internal use)."""
    if fft is None:
        return 0.0

    total_energy = np.sum(fft)
    if total_energy < EPSILON:
        return 0.0

    cumulative_energy = np.cumsum(fft)
    threshold = rolloff_percent * total_energy
    rolloff_idx = np.searchsorted(cumulative_energy, threshold)

    if rolloff_idx >= len(freqs):
        rolloff_idx = len(freqs) - 1

    return float(freqs[rolloff_idx])


def _compute_spectral_centroid_mono(
    audio_mono: np.ndarray,
    sample_rate: int,
    frame_length: int = 2048,
) -> float:
    """Compute spectral centroid on pre-converted mono audio (internal use)."""
    fft, freqs, magnitude_sum, _ = _compute_fft_once(audio_mono, sample_rate, frame_length)
    return _spectral_centroid_from_fft(fft, freqs, magnitude_sum)


def compute_spectral_centroid(
    audio: np.ndarray,
    sample_rate: int,
    frame_length: int = 2048,
) -> float:
    """
    Compute spectral centroid (brightness measure).

    The spectral centroid is the "center of mass" of the spectrum,
    indicating where most of the energy is concentrated.

    Note:
        For efficiency when computing multiple spectral metrics, use
        `compute_audio_metrics()` which computes FFT once and derives
        all metrics from it.

    Args:
        audio: Audio waveform
        sample_rate: Sample rate in Hz
        frame_length: FFT frame length

    Returns:
        Spectral centroid in Hz
    """
    return _compute_spectral_centroid_mono(_ensure_mono(audio), sample_rate, frame_length)


def _compute_spectral_bandwidth_mono(
    audio_mono: np.ndarray,
    sample_rate: int,
    frame_length: int = 2048,
) -> float:
    """Compute spectral bandwidth on pre-converted mono audio (internal use)."""
    fft, freqs, magnitude_sum, _ = _compute_fft_once(audio_mono, sample_rate, frame_length)
    centroid = _spectral_centroid_from_fft(fft, freqs, magnitude_sum)
    return _spectral_bandwidth_from_fft(fft, freqs, magnitude_sum, centroid)


def compute_spectral_bandwidth(
    audio: np.ndarray,
    sample_rate: int,
    frame_length: int = 2048,
) -> float:
    """
    Compute spectral bandwidth (spread around centroid).

    Note:
        For efficiency when computing multiple spectral metrics, use
        `compute_audio_metrics()` which computes FFT once and derives
        all metrics from it.

    Args:
        audio: Audio waveform
        sample_rate: Sample rate in Hz
        frame_length: FFT frame length

    Returns:
        Spectral bandwidth in Hz
    """
    return _compute_spectral_bandwidth_mono(_ensure_mono(audio), sample_rate, frame_length)


def _compute_spectral_rolloff_mono(
    audio_mono: np.ndarray,
    sample_rate: int,
    frame_length: int = 2048,
    rolloff_percent: float = 0.85,
) -> float:
    """Compute spectral rolloff on pre-converted mono audio (internal use)."""
    fft, freqs, _, _ = _compute_fft_once(audio_mono, sample_rate, frame_length)
    return _spectral_rolloff_from_fft(fft, freqs, rolloff_percent)


def compute_spectral_rolloff(
    audio: np.ndarray,
    sample_rate: int,
    frame_length: int = 2048,
    rolloff_percent: float = 0.85,
) -> float:
    """
    Compute spectral rolloff frequency.

    The frequency below which a specified percentage of the total
    spectral energy is contained.

    Note:
        For efficiency when computing multiple spectral metrics, use
        `compute_audio_metrics()` which computes FFT once and derives
        all metrics from it.

    Args:
        audio: Audio waveform
        sample_rate: Sample rate in Hz
        frame_length: FFT frame length
        rolloff_percent: Percentage threshold (default: 85%)

    Returns:
        Rolloff frequency in Hz
    """
    return _compute_spectral_rolloff_mono(_ensure_mono(audio), sample_rate, frame_length, rolloff_percent)


def detect_clipping(
    audio: np.ndarray,
    threshold: float = 0.999,
    min_samples: int = 10,
) -> Tuple[bool, int]:
    """
    Detect hard clipping in audio.

    True clipping occurs when multiple consecutive samples hit the
    hard limit of +/- 1.0, creating flat-topped waveforms.

    Args:
        audio: Audio waveform
        threshold: Amplitude threshold (default: 0.999)
        min_samples: Minimum clipped samples to flag (default: 10)

    Returns:
        Tuple of (is_clipped, clipped_sample_count)
    """
    audio_mono = _ensure_mono(audio)
    clipped_count = int(np.sum(np.abs(audio_mono) >= threshold))
    return clipped_count >= min_samples, clipped_count


def detect_silence(
    audio: np.ndarray,
    rms_threshold: float = 0.001,
) -> bool:
    """
    Detect if audio is essentially silent.

    Args:
        audio: Audio waveform
        rms_threshold: RMS threshold for silence (default: 0.001)

    Returns:
        True if audio is silent
    """
    return compute_rms(audio) < rms_threshold


def compute_audio_metrics(
    audio: np.ndarray,
    sample_rate: int,
) -> AudioQualityMetrics:
    """
    Compute comprehensive audio quality metrics.

    Args:
        audio: Audio waveform, shape (channels, samples) or (samples,)
        sample_rate: Sample rate in Hz (must be between 8000 and 192000)

    Returns:
        AudioQualityMetrics with all computed metrics

    Raises:
        ValueError: If sample_rate is outside valid range
    """
    # Validate sample rate
    if not MIN_SAMPLE_RATE <= sample_rate <= MAX_SAMPLE_RATE:
        raise ValueError(
            f"Sample rate must be between {MIN_SAMPLE_RATE} and {MAX_SAMPLE_RATE} Hz, "
            f"got {sample_rate}"
        )

    # Single mono conversion for all metrics
    audio_mono = _ensure_mono(audio)

    # Ensure we have enough samples
    if len(audio_mono) < sample_rate:
        audio_mono = np.pad(audio_mono, (0, sample_rate - len(audio_mono)))

    # Basic metrics (use internal _mono functions to avoid redundant conversion)
    rms = _compute_rms_mono(audio_mono)
    peak = float(np.abs(audio_mono).max())
    dynamic_range = _compute_dynamic_range_mono(audio_mono)

    # Spectral metrics - compute FFT ONCE and derive all metrics from it
    fft, freqs, magnitude_sum, _ = _compute_fft_once(audio_mono, sample_rate)
    centroid = _spectral_centroid_from_fft(fft, freqs, magnitude_sum)
    bandwidth = _spectral_bandwidth_from_fft(fft, freqs, magnitude_sum, centroid)
    rolloff = _spectral_rolloff_from_fft(fft, freqs)

    # Quality flags (inline to avoid redundant conversion)
    is_silent = rms < 0.001  # Use already-computed RMS
    clipped_count = int(np.sum(np.abs(audio_mono) >= 0.999))
    is_clipped = clipped_count >= 10

    # Duration
    duration = len(audio_mono) / sample_rate

    return AudioQualityMetrics(
        rms_energy=rms,
        peak_amplitude=peak,
        dynamic_range_db=dynamic_range,
        spectral_centroid_hz=centroid,
        spectral_bandwidth_hz=bandwidth,
        spectral_rolloff_hz=rolloff,
        is_silent=is_silent,
        is_clipped=is_clipped,
        clipped_sample_count=clipped_count,
        duration_sec=duration,
        sample_rate=sample_rate,
    )


# =============================================================================
# Audio Normalization
# =============================================================================


def normalize_loudness(
    audio: np.ndarray,
    target_db: float = -14.0,
    peak_ceiling_db: float = -1.0,
) -> np.ndarray:
    """
    Normalize audio to target loudness (EBU R128 inspired).

    Uses RMS-based normalization with a peak ceiling to prevent clipping.

    Args:
        audio: Audio waveform
        target_db: Target RMS level in dB (default: -14 dBFS)
        peak_ceiling_db: Maximum peak level (default: -1 dBFS)

    Returns:
        Normalized audio
    """
    audio = np.asarray(audio, dtype=np.float32)

    # Calculate current RMS
    rms = np.sqrt(np.mean(audio**2))
    if rms < EPSILON:
        return audio  # Silent audio, nothing to normalize

    # Calculate target RMS from dB
    target_rms = 10 ** (target_db / 20)

    # Calculate gain needed
    gain = target_rms / rms

    # Apply gain
    normalized = audio * gain

    # Check if we exceed peak ceiling (guard against division by zero)
    peak = np.abs(normalized).max()
    peak_ceiling = 10 ** (peak_ceiling_db / 20)

    if peak > peak_ceiling and peak > 0:
        # Reduce gain to meet peak ceiling
        normalized = normalized * (peak_ceiling / peak)

    return normalized


def normalize_peak(
    audio: np.ndarray,
    target_db: float = -1.0,
) -> np.ndarray:
    """
    Normalize audio to target peak level.

    Args:
        audio: Audio waveform
        target_db: Target peak level in dB (default: -1 dBFS)

    Returns:
        Peak-normalized audio
    """
    audio = np.asarray(audio, dtype=np.float32)

    peak = np.abs(audio).max()
    if peak < EPSILON:
        return audio

    target_peak = 10 ** (target_db / 20)
    gain = target_peak / peak

    return audio * gain


# =============================================================================
# Signal Quality Metrics
# =============================================================================


def compute_si_snr(
    reference: np.ndarray,
    estimate: np.ndarray,
) -> float:
    """
    Compute Scale-Invariant Signal-to-Noise Ratio (SI-SNR).

    SI-SNR is useful for comparing audio quality where the scale
    might differ between reference and estimate.

    Args:
        reference: Reference (clean) signal
        estimate: Estimated (noisy/processed) signal

    Returns:
        SI-SNR in dB (higher is better). Returns -inf if reference is
        silent (no signal to compare), or +inf if estimate perfectly
        matches reference (no noise).

    Note:
        Callers should check for inf/-inf when reference or estimate
        may be silent. Use `np.isfinite(result)` to filter these cases.
    """
    reference = _ensure_mono(np.asarray(reference))
    estimate = _ensure_mono(np.asarray(estimate))

    # Ensure same length
    min_len = min(len(reference), len(estimate))
    reference = reference[:min_len]
    estimate = estimate[:min_len]

    # Zero-mean
    reference = reference - np.mean(reference)
    estimate = estimate - np.mean(estimate)

    # Compute SI-SNR
    # s_target = <s', s> / ||s||^2 * s
    # e_noise = s' - s_target
    # SI-SNR = 10 * log10(||s_target||^2 / ||e_noise||^2)

    ref_energy = np.sum(reference**2)
    if ref_energy < EPSILON:
        return float("-inf")

    dot_product = np.sum(reference * estimate)
    s_target = (dot_product / ref_energy) * reference
    e_noise = estimate - s_target

    target_energy = np.sum(s_target**2)
    noise_energy = np.sum(e_noise**2)

    if noise_energy < EPSILON:
        return float("inf")  # Perfect match

    return float(10 * np.log10(target_energy / noise_energy))


def compute_correlation(
    audio1: np.ndarray,
    audio2: np.ndarray,
) -> float:
    """
    Compute Pearson correlation between two audio signals.

    Args:
        audio1: First audio signal
        audio2: Second audio signal

    Returns:
        Correlation coefficient (-1 to 1)
    """
    audio1 = _ensure_mono(np.asarray(audio1))
    audio2 = _ensure_mono(np.asarray(audio2))

    min_len = min(len(audio1), len(audio2))
    audio1 = audio1[:min_len]
    audio2 = audio2[:min_len]

    # Zero-mean
    audio1 = audio1 - np.mean(audio1)
    audio2 = audio2 - np.mean(audio2)

    # Correlation
    numerator = np.sum(audio1 * audio2)
    denominator = np.sqrt(np.sum(audio1**2) * np.sum(audio2**2))

    if denominator < EPSILON:
        return 0.0

    return float(numerator / denominator)
