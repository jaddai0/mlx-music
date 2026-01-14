"""Tests for DSP caching utilities."""

import time
import mlx.core as mx
from mlx_music.utils.dsp import (
    hanning,
    hamming,
    blackman,
    bartlett,
    get_window,
    mel_filterbank,
    get_stft_window,
    clear_dsp_cache,
    dsp_cache_info,
    hz_to_mel,
    mel_to_hz,
    ISTFTCache,
)


class TestWindowFunctions:
    """Tests for cached window functions."""

    def setup_method(self):
        """Clear cache before each test."""
        clear_dsp_cache()

    def test_hanning_caching(self):
        """Test that hanning window is cached."""
        # First call
        start = time.time()
        win1 = hanning(2048)
        first_time = time.time() - start

        # Second call (should be cached)
        start = time.time()
        win2 = hanning(2048)
        second_time = time.time() - start

        # Cached call should be much faster
        assert second_time < first_time
        # Verify same data
        assert mx.allclose(win1, win2)

    def test_hamming_caching(self):
        """Test that hamming window is cached."""
        clear_dsp_cache()

        win1 = hamming(1024)
        win2 = hamming(1024)

        # Same object should be returned
        info = dsp_cache_info()
        assert info["hamming"]["hits"] == 1

    def test_blackman_caching(self):
        """Test that blackman window is cached."""
        win1 = blackman(512)
        win2 = blackman(512)

        info = dsp_cache_info()
        assert info["blackman"]["hits"] == 1

    def test_bartlett_caching(self):
        """Test that bartlett window is cached."""
        win1 = bartlett(256)
        win2 = bartlett(256)

        info = dsp_cache_info()
        assert info["bartlett"]["hits"] == 1

    def test_different_sizes_cached_separately(self):
        """Test that different window sizes are cached separately."""
        clear_dsp_cache()

        hanning(1024)
        hanning(2048)
        hanning(1024)  # Should be cache hit

        info = dsp_cache_info()
        assert info["hanning"]["misses"] == 2  # Two different sizes
        assert info["hanning"]["hits"] == 1  # One repeated size

    def test_get_window_by_name(self):
        """Test get_window function."""
        win = get_window("hann", 1024)
        assert win.shape == (1024,)

        win = get_window("hamming", 512)
        assert win.shape == (512,)

    def test_get_window_invalid_name(self):
        """Test get_window raises error for invalid name."""
        import pytest

        with pytest.raises(ValueError, match="Unknown window function"):
            get_window("invalid_window", 1024)


class TestMelFilterbank:
    """Tests for mel filterbank caching."""

    def setup_method(self):
        """Clear cache before each test."""
        clear_dsp_cache()

    def test_mel_filterbank_shape(self):
        """Test mel filterbank output shape."""
        fb = mel_filterbank(n_mels=128, n_fft=2048, sample_rate=44100)
        assert fb.shape == (128, 1025)  # (n_mels, n_fft // 2 + 1)

    def test_mel_filterbank_caching(self):
        """Test that mel filterbank is cached."""
        fb1 = mel_filterbank(n_mels=128, n_fft=2048, sample_rate=44100)
        fb2 = mel_filterbank(n_mels=128, n_fft=2048, sample_rate=44100)

        info = dsp_cache_info()
        assert info["mel_filterbank"]["hits"] == 1

    def test_different_params_cached_separately(self):
        """Test different parameters create separate cache entries."""
        clear_dsp_cache()

        mel_filterbank(n_mels=128, n_fft=2048, sample_rate=44100)
        mel_filterbank(n_mels=80, n_fft=1024, sample_rate=22050)

        info = dsp_cache_info()
        assert info["mel_filterbank"]["misses"] == 2


class TestMelScale:
    """Tests for mel scale conversion functions."""

    def test_hz_to_mel_htk(self):
        """Test Hz to mel conversion with HTK formula."""
        # At 1000 Hz, mel ~= 999.98 in HTK scale (2595 * log10(1 + 1000/700))
        mel = hz_to_mel(1000, scale="htk")
        assert 990 < mel < 1010  # Should be close to 1000

    def test_mel_to_hz_htk(self):
        """Test mel to Hz conversion with HTK formula."""
        # Round trip: hz -> mel -> hz
        original = 1000
        mel = hz_to_mel(original, scale="htk")
        recovered = mel_to_hz(mel, scale="htk")
        assert abs(original - recovered) < 1e-6


class TestSTFTWindow:
    """Tests for STFT window helper."""

    def setup_method(self):
        """Clear cache before each test."""
        clear_dsp_cache()

    def test_stft_window_padding(self):
        """Test STFT window is padded correctly."""
        # Window smaller than n_fft should be padded
        win = get_stft_window(win_length=1024, n_fft=2048, window_type="hann")
        assert win.shape == (2048,)

    def test_stft_window_no_padding(self):
        """Test STFT window with matching sizes."""
        win = get_stft_window(win_length=2048, n_fft=2048, window_type="hann")
        assert win.shape == (2048,)

    def test_stft_window_caching(self):
        """Test STFT window is cached."""
        win1 = get_stft_window(1024, 2048, "hann")
        win2 = get_stft_window(1024, 2048, "hann")

        info = dsp_cache_info()
        assert info["stft_window"]["hits"] == 1


class TestISTFTCache:
    """Tests for ISTFT caching."""

    def test_istft_cache_positions(self):
        """Test position cache in ISTFTCache."""
        cache = ISTFTCache()

        pos1 = cache.get_positions(num_frames=10, frame_length=256, hop_length=128)
        pos2 = cache.get_positions(num_frames=10, frame_length=256, hop_length=128)

        # Should be cached
        assert cache.info()["position_indices"] == 1

    def test_istft_cache_clear(self):
        """Test cache clearing."""
        cache = ISTFTCache()

        cache.get_positions(10, 256, 128)
        assert cache.info()["total_cached_items"] > 0

        cache.clear()
        assert cache.info()["total_cached_items"] == 0


class TestCacheManagement:
    """Tests for cache management functions."""

    def test_clear_dsp_cache(self):
        """Test clearing all caches."""
        # Populate caches
        hanning(1024)
        mel_filterbank(n_mels=128, n_fft=2048, sample_rate=44100)

        # Clear
        clear_dsp_cache()

        # Re-access should be cache miss
        hanning(1024)
        info = dsp_cache_info()
        assert info["hanning"]["misses"] == 1
        assert info["hanning"]["hits"] == 0

    def test_cache_info_structure(self):
        """Test cache info returns correct structure."""
        info = dsp_cache_info()

        expected_keys = [
            "hanning",
            "hamming",
            "blackman",
            "bartlett",
            "stft_window",
            "mel_filterbank",
            "istft_cache",
        ]

        for key in expected_keys:
            assert key in info
