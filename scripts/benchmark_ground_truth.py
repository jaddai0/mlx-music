#!/usr/bin/env python3
"""
Ground Truth Benchmark for mlx-music.

Runs real models with real weights to get actual performance and quality numbers.
Supports ACE-Step, MusicGen, and StableAudio models with quantization comparisons.

Usage:
    # Full benchmark (30-60 minutes)
    python scripts/benchmark_ground_truth.py \
        --models ace-step musicgen stable-audio \
        --quantizations bfloat16 int8 int4 \
        --duration 10.0 \
        --runs 3

    # Quick smoke test (5 minutes)
    python scripts/benchmark_ground_truth.py \
        --models ace-step \
        --quantizations bfloat16 \
        --duration 5.0 \
        --runs 1

    # Scheduler comparison (ACE-Step)
    python scripts/benchmark_ground_truth.py \
        --models ace-step \
        --schedulers euler dpm++ \
        --steps 20 30 50
"""

import argparse
import gc
import json
import sys
import time
import warnings
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

# Add mlx_music to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import mlx.core as mx

# Import quantization utilities
from mlx_music.weights.quantization import (
    QuantizationConfig,
    get_metal_memory_info,
    get_model_size,
    quantize_model,
)


# =============================================================================
# Configuration and Results Data Structures
# =============================================================================


@dataclass
class BenchmarkConfig:
    """Configuration for benchmark run."""

    # Model paths
    ace_step_path: str = "/Users/dustinpainter/Dev-Projects/audio-models/ACEStep-models/ACE-Step-v1-3.5B"
    musicgen_path: str = "/Users/dustinpainter/Dev-Projects/audio-models/MusicGen-small"
    stable_audio_path: str = "/Users/dustinpainter/Dev-Projects/audio-models/StableAudio-models/stable-audio-open-1.0"

    # Generation parameters
    prompt: str = "electronic dance music with synthesizers and drums"
    duration: float = 10.0  # seconds
    seed: int = 42
    runs: int = 3

    # ACE-Step specific
    ace_step_steps: int = 60
    ace_step_cfg_scale: float = 15.0

    # MusicGen specific
    musicgen_top_k: int = 250
    musicgen_temperature: float = 1.0
    musicgen_cfg_scale: float = 3.0

    # StableAudio specific
    stable_audio_steps: int = 100
    stable_audio_cfg_scale: float = 7.0


@dataclass
class BenchmarkResult:
    """Single benchmark run result."""

    model_name: str
    quantization: str
    scheduler: Optional[str]
    steps: Optional[int]
    duration_requested: float

    # Timing (seconds)
    load_time_sec: float
    encode_time_sec: float
    denoise_time_sec: float
    decode_time_sec: float
    total_time_sec: float
    time_per_step_ms: Optional[float]

    # Memory (MB)
    model_memory_mb: float
    peak_memory_mb: float

    # Audio info
    audio_duration_sec: float
    sample_rate: int
    channels: int

    # Throughput
    realtime_factor: float  # audio_sec / gen_sec (>1 = faster than realtime)

    # Quality metrics
    rms_energy: float
    dynamic_range_db: float
    spectral_centroid_hz: float
    is_silent: bool
    is_clipped: bool
    clipped_sample_count: int = 0

    # Optional CLAP score
    clap_score: Optional[float] = None

    # Audio file path if saved
    audio_file: Optional[str] = None


@dataclass
class SchedulerComparisonResult:
    """Results from scheduler comparison."""

    model_name: str
    scheduler: str
    steps: int
    total_time_sec: float
    realtime_factor: float
    quality_score: float  # Relative to baseline


# =============================================================================
# Audio Quality Metrics
# =============================================================================


def measure_audio_quality(audio: np.ndarray, sample_rate: int) -> Dict[str, Any]:
    """
    Compute audio quality metrics.

    Args:
        audio: Audio waveform, shape (channels, samples) or (samples,)
        sample_rate: Sample rate in Hz

    Returns:
        Dict with quality metrics
    """
    # Ensure audio is properly shaped
    audio = np.asarray(audio)

    # Handle various input shapes
    if audio.ndim == 1:
        audio_mono = audio
    elif audio.ndim == 2:
        # Determine which axis is channels vs samples
        # Channels are typically 1-2, samples are typically >> 2
        if audio.shape[0] <= 2 and audio.shape[1] > 2:
            # Shape is (channels, samples), typical format
            audio_mono = audio.mean(axis=0)
        elif audio.shape[1] <= 2 and audio.shape[0] > 2:
            # Shape is (samples, channels), transpose first
            audio_mono = audio.mean(axis=1)
        else:
            # Ambiguous, assume first axis is channels
            audio_mono = audio.mean(axis=0)
    else:
        # Higher dimensional, flatten
        audio_mono = audio.flatten()

    # Ensure 1D
    audio_mono = audio_mono.flatten()

    # Ensure we have enough samples
    if len(audio_mono) < sample_rate:
        # Pad if needed
        audio_mono = np.pad(audio_mono, (0, sample_rate - len(audio_mono)))

    # RMS energy
    rms = float(np.sqrt(np.mean(audio_mono**2)))

    # Dynamic range (dB)
    audio_abs = np.abs(audio_mono)
    max_val = audio_abs.max()
    # Use 0.1% percentile as floor to avoid silence
    min_val = np.percentile(audio_abs[audio_abs > 0], 0.1) if np.any(audio_abs > 0) else 1e-10
    if max_val > 0 and min_val > 0:
        dynamic_range_db = float(20 * np.log10(max_val / min_val))
    else:
        dynamic_range_db = 0.0

    # Spectral centroid (simplified - first second only)
    analysis_samples = min(len(audio_mono), sample_rate)
    fft = np.abs(np.fft.rfft(audio_mono[:analysis_samples]))
    freqs = np.fft.rfftfreq(analysis_samples, 1 / sample_rate)
    spectral_centroid = float(np.sum(freqs * fft) / (np.sum(fft) + 1e-10))

    # Quality flags
    is_silent = rms < 0.001

    # Clipping detection: True clipping is when MULTIPLE samples hit the hard limit
    # A single sample at 0.99 is normal for loud audio; true clipping produces
    # flat-topped waveforms with many consecutive samples at ±1.0
    peak_threshold = 0.999  # Very close to hard limit
    min_clipped_samples = 10  # Require multiple samples to flag as clipped
    clipped_sample_count = int(np.sum(np.abs(audio_mono) >= peak_threshold))
    is_clipped = clipped_sample_count >= min_clipped_samples

    return {
        "rms_energy": rms,
        "dynamic_range_db": dynamic_range_db,
        "spectral_centroid_hz": spectral_centroid,
        "is_silent": is_silent,
        "is_clipped": is_clipped,
        "clipped_sample_count": clipped_sample_count,
    }


# =============================================================================
# CLAP Audio-Text Similarity (Optional)
# =============================================================================

_clap_model = None


def get_clap_model():
    """Lazy-load CLAP model for audio-text similarity."""
    global _clap_model
    if _clap_model is None:
        try:
            import laion_clap

            _clap_model = laion_clap.CLAP_Module(enable_fusion=False)
            _clap_model.load_ckpt()
            print("CLAP model loaded successfully")
        except ImportError:
            warnings.warn(
                "laion-clap not installed. CLAP scoring disabled. "
                "Install with: pip install laion-clap"
            )
            return None
        except Exception as e:
            warnings.warn(f"Failed to load CLAP model: {e}")
            return None
    return _clap_model


def compute_clap_score(audio: np.ndarray, sample_rate: int, prompt: str) -> Optional[float]:
    """
    Compute CLAP similarity between audio and text prompt.

    Args:
        audio: Audio waveform (channels, samples) or (samples,)
        sample_rate: Sample rate in Hz
        prompt: Text prompt

    Returns:
        Cosine similarity score (0-1) or None if CLAP unavailable
    """
    clap = get_clap_model()
    if clap is None:
        return None

    try:
        import tempfile

        import soundfile as sf
        import torch

        # Ensure mono and correct shape
        if audio.ndim > 1:
            audio_mono = audio.mean(axis=0)
        else:
            audio_mono = audio

        # Save to temp file (CLAP requires file path)
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            sf.write(f.name, audio_mono, sample_rate)
            audio_path = f.name

        # Get embeddings
        audio_embed = clap.get_audio_embedding_from_filelist([audio_path], use_tensor=True)
        text_embed = clap.get_text_embedding([prompt], use_tensor=True)

        # Cosine similarity
        similarity = torch.nn.functional.cosine_similarity(audio_embed, text_embed)

        # Cleanup
        Path(audio_path).unlink(missing_ok=True)

        return float(similarity.item())
    except Exception as e:
        warnings.warn(f"CLAP scoring failed: {e}")
        return None


# =============================================================================
# Memory Management
# =============================================================================


def clear_memory():
    """Clear MLX memory cache and run garbage collection."""
    gc.collect()
    try:
        # Use new API if available, fall back to deprecated
        if hasattr(mx, 'clear_cache'):
            mx.clear_cache()
        elif hasattr(mx, 'metal') and hasattr(mx.metal, 'clear_cache'):
            mx.metal.clear_cache()
    except Exception:
        pass


def reset_peak_memory():
    """Reset peak memory tracking."""
    try:
        # Use new API if available, fall back to deprecated
        if hasattr(mx, 'reset_peak_memory'):
            mx.reset_peak_memory()
        elif hasattr(mx, 'metal') and hasattr(mx.metal, 'reset_peak_memory'):
            mx.metal.reset_peak_memory()
    except Exception:
        pass


# =============================================================================
# Model Benchmarking Functions
# =============================================================================


def benchmark_ace_step(
    model_path: str,
    prompt: str,
    duration: float,
    quantization: str,
    scheduler_type: str = "euler",
    steps: int = 60,
    cfg_scale: float = 15.0,
    seed: int = 42,
    save_audio: bool = False,
    output_dir: Optional[Path] = None,
    compute_clap: bool = False,
) -> BenchmarkResult:
    """
    Benchmark ACE-Step model.

    Args:
        model_path: Path to ACE-Step model
        prompt: Text prompt for generation
        duration: Audio duration in seconds
        quantization: "bfloat16", "int8", or "int4"
        scheduler_type: "euler" or "dpm++" (or "heun")
        steps: Number of diffusion steps
        cfg_scale: Classifier-free guidance scale
        seed: Random seed
        save_audio: Whether to save generated audio
        output_dir: Directory for saved audio
        compute_clap: Whether to compute CLAP score

    Returns:
        BenchmarkResult with all metrics
    """
    from mlx_music import ACEStep

    clear_memory()
    reset_peak_memory()
    mem_before = get_metal_memory_info()

    # Load model
    print(f"  Loading ACE-Step ({quantization})...")
    t0 = time.perf_counter()
    model = ACEStep.from_pretrained(model_path)
    load_time = time.perf_counter() - t0
    print(f"    Load time: {load_time:.2f}s")

    # Apply quantization if needed
    if quantization != "bfloat16":
        print(f"  Applying {quantization} quantization...")
        if quantization == "int8":
            config = QuantizationConfig.for_quality()
        elif quantization == "int4":
            config = QuantizationConfig.for_speed()
        else:
            config = QuantizationConfig.for_balanced()
        quantize_model(model.transformer, config=config)

    mem_after_load = get_metal_memory_info()
    model_memory = mem_after_load["active_mb"] - mem_before["active_mb"]

    # Get model size
    num_params, size_mb = get_model_size(model.transformer)
    print(f"    Model size: {size_mb:.1f} MB, {num_params:,} params")

    # Map scheduler type - dpm++ is directly supported by the scheduler module
    # No mapping needed - the scheduler module accepts "dpm++", "dpm++_karras", etc.

    # Generate
    print(f"  Generating ({steps} steps, {scheduler_type})...")
    mx.random.seed(seed)

    t0 = time.perf_counter()
    output = model.generate(
        prompt=prompt,
        duration=duration,
        num_inference_steps=steps,
        guidance_scale=cfg_scale,
        seed=seed,
        scheduler_type=scheduler_type,
    )
    total_time = time.perf_counter() - t0

    mem_peak = get_metal_memory_info()

    # Get audio
    audio = output.audio
    if isinstance(audio, mx.array):
        audio = np.array(audio)

    # Ensure proper shape (channels, samples)
    if audio.ndim == 1:
        audio = audio.reshape(1, -1)
    elif audio.shape[0] > audio.shape[1]:
        audio = audio.T

    sample_rate = output.sample_rate
    actual_duration = audio.shape[-1] / sample_rate

    # Quality metrics
    quality = measure_audio_quality(audio, sample_rate)

    # CLAP score
    clap_score = None
    if compute_clap:
        print("  Computing CLAP score...")
        clap_score = compute_clap_score(audio, sample_rate, prompt)
        if clap_score is not None:
            print(f"    CLAP score: {clap_score:.3f}")

    # Save audio if requested
    audio_file = None
    if save_audio and output_dir is not None:
        import soundfile as sf

        filename = f"ace_step_{quantization}_{scheduler_type}_{steps}steps.wav"
        audio_file = str(output_dir / filename)
        # Ensure audio is 2D (samples, channels) for soundfile
        audio_for_save = audio.squeeze()  # Remove extra dimensions
        if audio_for_save.ndim == 1:
            audio_for_save = audio_for_save.reshape(-1, 1)  # (samples, 1)
        elif audio_for_save.ndim == 2:
            # Ensure (samples, channels) format
            if audio_for_save.shape[0] <= 2 and audio_for_save.shape[1] > 2:
                audio_for_save = audio_for_save.T  # (channels, samples) -> (samples, channels)
        sf.write(audio_file, audio_for_save, sample_rate)
        print(f"    Saved: {audio_file}")

    # Extract timing from output if available
    if hasattr(output, 'timing') and output.timing is not None:
        encode_time = output.timing.encode_sec
        denoise_time = output.timing.denoise_sec
        decode_time = output.timing.decode_sec
        generation_time = output.timing.total_sec
    else:
        # Fallback: estimate generation time as total - load
        generation_time = total_time - load_time
        encode_time = 0.0
        denoise_time = generation_time
        decode_time = 0.0

    # Calculate realtime factor (based on generation time, not total with load)
    realtime_factor = actual_duration / generation_time if generation_time > 0 else 0

    # Clean up
    del model
    clear_memory()

    return BenchmarkResult(
        model_name="ACE-Step",
        quantization=quantization,
        scheduler=scheduler_type,
        steps=steps,
        duration_requested=duration,
        load_time_sec=load_time,
        encode_time_sec=encode_time,
        denoise_time_sec=denoise_time,
        decode_time_sec=decode_time,
        total_time_sec=total_time,
        time_per_step_ms=(denoise_time / steps * 1000) if steps > 0 else None,
        model_memory_mb=model_memory,
        peak_memory_mb=mem_peak["peak_mb"],
        audio_duration_sec=actual_duration,
        sample_rate=sample_rate,
        channels=audio.shape[0],
        realtime_factor=realtime_factor,
        clap_score=clap_score,
        audio_file=audio_file,
        **quality,
    )


def benchmark_musicgen(
    model_path: str,
    prompt: str,
    duration: float,
    quantization: str,
    temperature: float = 1.0,
    top_k: int = 250,
    cfg_scale: float = 3.0,
    seed: int = 42,
    save_audio: bool = False,
    output_dir: Optional[Path] = None,
    compute_clap: bool = False,
) -> BenchmarkResult:
    """
    Benchmark MusicGen model.

    Args:
        model_path: Path to MusicGen model
        prompt: Text prompt for generation
        duration: Audio duration in seconds
        quantization: "bfloat16", "int8", or "int4"
        temperature: Sampling temperature
        top_k: Top-k filtering
        cfg_scale: Classifier-free guidance scale
        seed: Random seed
        save_audio: Whether to save generated audio
        output_dir: Directory for saved audio
        compute_clap: Whether to compute CLAP score

    Returns:
        BenchmarkResult with all metrics
    """
    from mlx_music import MusicGen

    clear_memory()
    reset_peak_memory()
    mem_before = get_metal_memory_info()

    # Load model
    print(f"  Loading MusicGen ({quantization})...")
    t0 = time.perf_counter()
    model = MusicGen.from_pretrained(model_path)
    load_time = time.perf_counter() - t0
    print(f"    Load time: {load_time:.2f}s")

    # Apply quantization if needed
    if quantization != "bfloat16":
        print(f"  Applying {quantization} quantization...")
        if quantization == "int8":
            config = QuantizationConfig.for_quality()
        elif quantization == "int4":
            config = QuantizationConfig.for_speed()
        else:
            config = QuantizationConfig.for_balanced()
        quantize_model(model.decoder, config=config)

    mem_after_load = get_metal_memory_info()
    model_memory = mem_after_load["active_mb"] - mem_before["active_mb"]

    # Get model size
    num_params, size_mb = get_model_size(model.decoder)
    print(f"    Model size: {size_mb:.1f} MB, {num_params:,} params")

    # Generate
    print(f"  Generating ({duration}s audio)...")
    mx.random.seed(seed)

    t0 = time.perf_counter()
    output = model.generate(
        prompt=prompt,
        duration=duration,
        temperature=temperature,
        top_k=top_k,
        guidance_scale=cfg_scale,
        seed=seed,
    )
    total_time = time.perf_counter() - t0

    mem_peak = get_metal_memory_info()

    # Get audio
    audio = output.audio
    if isinstance(audio, mx.array):
        audio = np.array(audio)

    # Ensure proper shape
    if audio.ndim == 1:
        audio = audio.reshape(1, -1)
    elif audio.shape[0] > audio.shape[1]:
        audio = audio.T

    sample_rate = output.sample_rate
    actual_duration = audio.shape[-1] / sample_rate

    # Quality metrics
    quality = measure_audio_quality(audio, sample_rate)

    # CLAP score
    clap_score = None
    if compute_clap:
        print("  Computing CLAP score...")
        clap_score = compute_clap_score(audio, sample_rate, prompt)
        if clap_score is not None:
            print(f"    CLAP score: {clap_score:.3f}")

    # Save audio if requested
    audio_file = None
    if save_audio and output_dir is not None:
        import soundfile as sf

        filename = f"musicgen_{quantization}.wav"
        audio_file = str(output_dir / filename)
        # Ensure audio is 2D (samples, channels) for soundfile
        audio_for_save = audio.squeeze()
        if audio_for_save.ndim == 1:
            audio_for_save = audio_for_save.reshape(-1, 1)
        elif audio_for_save.ndim == 2:
            if audio_for_save.shape[0] <= 2 and audio_for_save.shape[1] > 2:
                audio_for_save = audio_for_save.T
        sf.write(audio_file, audio_for_save, sample_rate)
        print(f"    Saved: {audio_file}")

    # Calculate realtime factor
    generation_time = total_time - load_time
    realtime_factor = actual_duration / generation_time if generation_time > 0 else 0

    # Clean up
    del model
    clear_memory()

    return BenchmarkResult(
        model_name="MusicGen",
        quantization=quantization,
        scheduler=None,  # MusicGen is autoregressive, no scheduler
        steps=None,
        duration_requested=duration,
        load_time_sec=load_time,
        encode_time_sec=0.0,
        denoise_time_sec=generation_time,
        decode_time_sec=0.0,
        total_time_sec=total_time,
        time_per_step_ms=None,
        model_memory_mb=model_memory,
        peak_memory_mb=mem_peak["peak_mb"],
        audio_duration_sec=actual_duration,
        sample_rate=sample_rate,
        channels=audio.shape[0],
        realtime_factor=realtime_factor,
        clap_score=clap_score,
        audio_file=audio_file,
        **quality,
    )


def benchmark_stable_audio(
    model_path: str,
    prompt: str,
    duration: float,
    quantization: str,
    steps: int = 100,
    cfg_scale: float = 7.0,
    seed: int = 42,
    save_audio: bool = False,
    output_dir: Optional[Path] = None,
    compute_clap: bool = False,
) -> BenchmarkResult:
    """
    Benchmark StableAudio model.

    Args:
        model_path: Path to StableAudio model
        prompt: Text prompt for generation
        duration: Audio duration in seconds (max 47s)
        quantization: "bfloat16", "int8", or "int4"
        steps: Number of diffusion steps
        cfg_scale: Classifier-free guidance scale
        seed: Random seed
        save_audio: Whether to save generated audio
        output_dir: Directory for saved audio
        compute_clap: Whether to compute CLAP score

    Returns:
        BenchmarkResult with all metrics
    """
    from mlx_music import StableAudio

    # Clamp duration to StableAudio's max
    if duration > 47.0:
        print(f"  Warning: StableAudio max duration is 47s, clamping from {duration}s")
        duration = 47.0

    clear_memory()
    reset_peak_memory()
    mem_before = get_metal_memory_info()

    # Load model
    print(f"  Loading StableAudio ({quantization})...")
    t0 = time.perf_counter()
    model = StableAudio.from_pretrained(model_path, strict_components=False)
    load_time = time.perf_counter() - t0
    print(f"    Load time: {load_time:.2f}s")

    # Apply quantization if needed
    if quantization != "bfloat16":
        print(f"  Applying {quantization} quantization...")
        if quantization == "int8":
            config = QuantizationConfig.for_quality()
        elif quantization == "int4":
            config = QuantizationConfig.for_speed()
        else:
            config = QuantizationConfig.for_balanced()
        quantize_model(model.transformer, config=config)

    mem_after_load = get_metal_memory_info()
    model_memory = mem_after_load["active_mb"] - mem_before["active_mb"]

    # Get model size
    num_params, size_mb = get_model_size(model.transformer)
    print(f"    Model size: {size_mb:.1f} MB, {num_params:,} params")

    # Generate
    print(f"  Generating ({steps} steps)...")
    mx.random.seed(seed)

    t0 = time.perf_counter()
    output = model.generate(
        prompt=prompt,
        duration=duration,
        num_inference_steps=steps,
        guidance_scale=cfg_scale,
        seed=seed,
    )
    total_time = time.perf_counter() - t0

    mem_peak = get_metal_memory_info()

    # Get audio
    audio = output.audio
    if isinstance(audio, mx.array):
        audio = np.array(audio)

    # Ensure proper shape
    if audio.ndim == 1:
        audio = audio.reshape(1, -1)
    elif audio.shape[0] > audio.shape[1]:
        audio = audio.T

    sample_rate = output.sample_rate
    actual_duration = audio.shape[-1] / sample_rate

    # Quality metrics
    quality = measure_audio_quality(audio, sample_rate)

    # CLAP score
    clap_score = None
    if compute_clap:
        print("  Computing CLAP score...")
        clap_score = compute_clap_score(audio, sample_rate, prompt)
        if clap_score is not None:
            print(f"    CLAP score: {clap_score:.3f}")

    # Save audio if requested
    audio_file = None
    if save_audio and output_dir is not None:
        import soundfile as sf

        filename = f"stable_audio_{quantization}_{steps}steps.wav"
        audio_file = str(output_dir / filename)
        # Ensure audio is 2D (samples, channels) for soundfile
        audio_for_save = audio.squeeze()
        if audio_for_save.ndim == 1:
            audio_for_save = audio_for_save.reshape(-1, 1)
        elif audio_for_save.ndim == 2:
            if audio_for_save.shape[0] <= 2 and audio_for_save.shape[1] > 2:
                audio_for_save = audio_for_save.T
        sf.write(audio_file, audio_for_save, sample_rate)
        print(f"    Saved: {audio_file}")

    # Calculate realtime factor
    generation_time = total_time - load_time
    realtime_factor = actual_duration / generation_time if generation_time > 0 else 0

    # Clean up
    del model
    clear_memory()

    return BenchmarkResult(
        model_name="StableAudio",
        quantization=quantization,
        scheduler="EDM-DPM++",
        steps=steps,
        duration_requested=duration,
        load_time_sec=load_time,
        encode_time_sec=0.0,
        denoise_time_sec=generation_time,
        decode_time_sec=0.0,
        total_time_sec=total_time,
        time_per_step_ms=(generation_time / steps * 1000) if steps > 0 else None,
        model_memory_mb=model_memory,
        peak_memory_mb=mem_peak["peak_mb"],
        audio_duration_sec=actual_duration,
        sample_rate=sample_rate,
        channels=audio.shape[0],
        realtime_factor=realtime_factor,
        clap_score=clap_score,
        audio_file=audio_file,
        **quality,
    )


# =============================================================================
# Scheduler Comparison
# =============================================================================


def run_scheduler_comparison(
    model_path: str,
    prompt: str,
    duration: float,
    quantization: str = "bfloat16",
    schedulers: List[str] = None,
    step_counts: List[int] = None,
    seed: int = 42,
    compute_clap: bool = True,
) -> List[SchedulerComparisonResult]:
    """
    Compare different schedulers and step counts for ACE-Step.

    Args:
        model_path: Path to ACE-Step model
        prompt: Text prompt
        duration: Audio duration
        quantization: Quantization level
        schedulers: List of schedulers to compare
        step_counts: List of step counts to test
        seed: Random seed
        compute_clap: Whether to compute CLAP score

    Returns:
        List of SchedulerComparisonResult
    """
    if schedulers is None:
        schedulers = ["euler", "dpm++"]
    if step_counts is None:
        step_counts = [20, 30, 50]

    results = []
    baseline_clap = None

    for scheduler in schedulers:
        for steps in step_counts:
            print(f"\n  Testing {scheduler} with {steps} steps...")

            result = benchmark_ace_step(
                model_path=model_path,
                prompt=prompt,
                duration=duration,
                quantization=quantization,
                scheduler_type=scheduler,
                steps=steps,
                seed=seed,
                compute_clap=compute_clap,
            )

            # Use first result as baseline for quality comparison
            if baseline_clap is None and result.clap_score is not None:
                baseline_clap = result.clap_score

            quality_score = 1.0
            if result.clap_score is not None and baseline_clap is not None:
                quality_score = result.clap_score / baseline_clap

            results.append(
                SchedulerComparisonResult(
                    model_name="ACE-Step",
                    scheduler=scheduler,
                    steps=steps,
                    total_time_sec=result.total_time_sec,
                    realtime_factor=result.realtime_factor,
                    quality_score=quality_score,
                )
            )

    return results


# =============================================================================
# Results Display and Saving
# =============================================================================


def print_results_table(results: List[BenchmarkResult]):
    """Print formatted results table."""
    print("\n" + "=" * 100)
    print("BENCHMARK RESULTS")
    print("=" * 100)

    # Header
    header = (
        f"{'Model':<12} {'Quant':<10} {'Steps':<8} {'Time (s)':<10} "
        f"{'RTF':<8} {'Memory (MB)':<12} {'CLAP':<8} {'Quality':<10}"
    )
    print(header)
    print("-" * 100)

    for r in results:
        steps_str = str(r.steps) if r.steps else "N/A"
        clap_str = f"{r.clap_score:.3f}" if r.clap_score else "N/A"
        quality_str = "PASS" if not r.is_silent and not r.is_clipped else "FAIL"
        if r.is_silent:
            quality_str = "SILENT"
        if r.is_clipped:
            quality_str = "CLIPPED"

        row = (
            f"{r.model_name:<12} {r.quantization:<10} {steps_str:<8} "
            f"{r.total_time_sec:<10.2f} {r.realtime_factor:<8.2f} "
            f"{r.peak_memory_mb:<12.0f} {clap_str:<8} {quality_str:<10}"
        )
        print(row)

    print("\nRTF = Realtime Factor (audio duration / generation time)")
    print("      >1.0 means faster than realtime")


def print_scheduler_comparison(results: List[SchedulerComparisonResult]):
    """Print scheduler comparison table."""
    print("\n" + "=" * 80)
    print("SCHEDULER COMPARISON (ACE-Step)")
    print("=" * 80)

    header = f"{'Scheduler':<12} {'Steps':<8} {'Time (s)':<10} {'RTF':<8} {'Quality':<10}"
    print(header)
    print("-" * 80)

    for r in results:
        quality_str = f"{r.quality_score:.2f}"
        row = (
            f"{r.scheduler:<12} {r.steps:<8} {r.total_time_sec:<10.2f} "
            f"{r.realtime_factor:<8.2f} {quality_str:<10}"
        )
        print(row)


def save_results(
    results: List[BenchmarkResult],
    scheduler_results: Optional[List[SchedulerComparisonResult]],
    config: BenchmarkConfig,
    output_path: Path,
):
    """Save benchmark results to JSON."""
    output = {
        "timestamp": datetime.now().isoformat(),
        "config": asdict(config),
        "results": [asdict(r) for r in results],
    }

    if scheduler_results:
        output["scheduler_comparison"] = [asdict(r) for r in scheduler_results]

    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\nResults saved to: {output_path}")


# =============================================================================
# Main Entry Point
# =============================================================================


def main():
    parser = argparse.ArgumentParser(
        description="Ground Truth Benchmark for mlx-music",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # Model selection
    parser.add_argument(
        "--models",
        nargs="+",
        default=["ace-step"],
        choices=["ace-step", "musicgen", "stable-audio"],
        help="Models to benchmark",
    )

    # Quantization
    parser.add_argument(
        "--quantizations",
        nargs="+",
        default=["bfloat16"],
        choices=["bfloat16", "int8", "int4"],
        help="Quantization levels to test",
    )

    # Generation parameters
    parser.add_argument("--prompt", type=str, default="electronic dance music with synthesizers and drums")
    parser.add_argument("--duration", type=float, default=10.0, help="Audio duration in seconds")
    parser.add_argument("--runs", type=int, default=1, help="Number of runs per configuration")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    # ACE-Step specific
    parser.add_argument("--ace-steps", type=int, default=60, help="ACE-Step diffusion steps")
    parser.add_argument("--ace-cfg", type=float, default=15.0, help="ACE-Step CFG scale")

    # MusicGen specific
    parser.add_argument("--musicgen-temperature", type=float, default=1.0)
    parser.add_argument("--musicgen-top-k", type=int, default=250)
    parser.add_argument("--musicgen-cfg", type=float, default=3.0)

    # StableAudio specific
    parser.add_argument("--stable-steps", type=int, default=100)
    parser.add_argument("--stable-cfg", type=float, default=7.0)

    # Scheduler comparison
    parser.add_argument(
        "--schedulers",
        nargs="+",
        default=None,
        choices=["euler", "dpm++", "heun"],
        help="Schedulers to compare (ACE-Step only)",
    )
    parser.add_argument(
        "--steps",
        nargs="+",
        type=int,
        default=None,
        help="Step counts for scheduler comparison",
    )

    # Output options
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("benchmark_results"),
        help="Output directory for results",
    )
    parser.add_argument("--save-audio", action="store_true", help="Save generated audio files")
    parser.add_argument("--compute-clap", action="store_true", help="Compute CLAP audio-text similarity scores")

    # Model paths
    parser.add_argument(
        "--ace-step-path",
        type=str,
        default="/Users/dustinpainter/Dev-Projects/audio-models/ACEStep-models/ACE-Step-v1-3.5B",
    )
    parser.add_argument(
        "--musicgen-path",
        type=str,
        default="/Users/dustinpainter/Dev-Projects/audio-models/MusicGen-small",
    )
    parser.add_argument(
        "--stable-audio-path",
        type=str,
        default="/Users/dustinpainter/Dev-Projects/audio-models/StableAudio-models/stable-audio-open-1.0",
    )

    args = parser.parse_args()

    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)
    audio_output_dir = args.output_dir / "audio_samples" if args.save_audio else None
    if audio_output_dir:
        audio_output_dir.mkdir(parents=True, exist_ok=True)

    # Create config
    config = BenchmarkConfig(
        ace_step_path=args.ace_step_path,
        musicgen_path=args.musicgen_path,
        stable_audio_path=args.stable_audio_path,
        prompt=args.prompt,
        duration=args.duration,
        seed=args.seed,
        runs=args.runs,
        ace_step_steps=args.ace_steps,
        ace_step_cfg_scale=args.ace_cfg,
        musicgen_temperature=args.musicgen_temperature,
        musicgen_top_k=args.musicgen_top_k,
        musicgen_cfg_scale=args.musicgen_cfg,
        stable_audio_steps=args.stable_steps,
        stable_audio_cfg_scale=args.stable_cfg,
    )

    print("=" * 80)
    print("MLX-MUSIC GROUND TRUTH BENCHMARK")
    print("=" * 80)
    print(f"Models: {', '.join(args.models)}")
    print(f"Quantizations: {', '.join(args.quantizations)}")
    print(f"Duration: {args.duration}s")
    print(f"Runs: {args.runs}")
    print(f"Prompt: \"{args.prompt}\"")
    print(f"Output: {args.output_dir}")

    results = []
    scheduler_results = None

    # Run benchmarks for each model and quantization
    for model_name in args.models:
        for quant in args.quantizations:
            for run in range(args.runs):
                print(f"\n{'='*60}")
                print(f"Model: {model_name.upper()}, Quantization: {quant}, Run: {run + 1}/{args.runs}")
                print(f"{'='*60}")

                try:
                    if model_name == "ace-step":
                        result = benchmark_ace_step(
                            model_path=config.ace_step_path,
                            prompt=config.prompt,
                            duration=config.duration,
                            quantization=quant,
                            steps=config.ace_step_steps,
                            cfg_scale=config.ace_step_cfg_scale,
                            seed=config.seed + run,
                            save_audio=args.save_audio,
                            output_dir=audio_output_dir,
                            compute_clap=args.compute_clap,
                        )
                    elif model_name == "musicgen":
                        result = benchmark_musicgen(
                            model_path=config.musicgen_path,
                            prompt=config.prompt,
                            duration=config.duration,
                            quantization=quant,
                            temperature=config.musicgen_temperature,
                            top_k=config.musicgen_top_k,
                            cfg_scale=config.musicgen_cfg_scale,
                            seed=config.seed + run,
                            save_audio=args.save_audio,
                            output_dir=audio_output_dir,
                            compute_clap=args.compute_clap,
                        )
                    elif model_name == "stable-audio":
                        result = benchmark_stable_audio(
                            model_path=config.stable_audio_path,
                            prompt=config.prompt,
                            duration=min(config.duration, 47.0),  # StableAudio max
                            quantization=quant,
                            steps=config.stable_audio_steps,
                            cfg_scale=config.stable_audio_cfg_scale,
                            seed=config.seed + run,
                            save_audio=args.save_audio,
                            output_dir=audio_output_dir,
                            compute_clap=args.compute_clap,
                        )
                    else:
                        print(f"  Unknown model: {model_name}")
                        continue

                    results.append(result)
                    print(f"  RTF: {result.realtime_factor:.2f}x, Peak Memory: {result.peak_memory_mb:.0f} MB")

                except Exception as e:
                    print(f"  ERROR: {e}")
                    import traceback
                    traceback.print_exc()

    # Run scheduler comparison if requested
    if args.schedulers and "ace-step" in args.models:
        print(f"\n{'='*60}")
        print("SCHEDULER COMPARISON")
        print(f"{'='*60}")

        scheduler_results = run_scheduler_comparison(
            model_path=config.ace_step_path,
            prompt=config.prompt,
            duration=config.duration,
            quantization=args.quantizations[0],  # Use first quantization
            schedulers=args.schedulers,
            step_counts=args.steps,
            seed=config.seed,
            compute_clap=args.compute_clap,
        )

    # Print results
    if results:
        print_results_table(results)

    if scheduler_results:
        print_scheduler_comparison(scheduler_results)

    # Save results
    if results:
        output_path = args.output_dir / "ground_truth_results.json"
        save_results(results, scheduler_results, config, output_path)

    print("\nBenchmark complete!")


if __name__ == "__main__":
    main()
