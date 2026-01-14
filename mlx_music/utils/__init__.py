"""Utility functions for mlx-music."""

from mlx_music.utils.audio_io import load_audio, save_audio
from mlx_music.utils.mel import LogMelSpectrogram, stft, istft
from mlx_music.utils.dsp import (
    # Window functions
    hanning,
    hamming,
    blackman,
    bartlett,
    get_window,
    # Mel utilities
    mel_filterbank,
    hz_to_mel,
    mel_to_hz,
    # STFT helpers
    get_stft_window,
    ISTFTCache,
    get_istft_cache,
    # Cache management
    clear_dsp_cache,
    dsp_cache_info,
)

__all__ = [
    # Audio I/O
    "load_audio",
    "save_audio",
    # Mel spectrogram
    "LogMelSpectrogram",
    "stft",
    "istft",
    # Window functions
    "hanning",
    "hamming",
    "blackman",
    "bartlett",
    "get_window",
    # Mel utilities
    "mel_filterbank",
    "hz_to_mel",
    "mel_to_hz",
    # STFT helpers
    "get_stft_window",
    "ISTFTCache",
    "get_istft_cache",
    # Cache management
    "clear_dsp_cache",
    "dsp_cache_info",
]
