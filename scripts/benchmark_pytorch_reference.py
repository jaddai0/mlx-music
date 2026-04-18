#!/usr/bin/env python3
"""
PyTorch Reference Benchmark for comparison with MLX.

Benchmarks MusicGen using PyTorch with MPS backend on Apple Silicon.
Results can be compared with mlx-music benchmarks to measure MLX performance gains.

Usage:
    python scripts/benchmark_pytorch_reference.py
    python scripts/benchmark_pytorch_reference.py --duration 10 --runs 3
"""

import argparse
import gc
import json
import sys
import time
import warnings
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np


@dataclass
class PyTorchBenchmarkResult:
    """Results from PyTorch benchmark."""

    framework: str  # "pytorch"
    device: str  # "mps", "cpu", "cuda"
    model_name: str
    model_variant: str  # "small", "medium", "large"

    # Timing (seconds)
    load_time_sec: float
    generation_time_sec: float
    total_time_sec: float

    # Audio
    audio_duration_sec: float
    sample_rate: int
    channels: int

    # Throughput
    realtime_factor: float

    # Memory (if available)
    peak_memory_mb: Optional[float] = None

    # Quality
    rms_energy: Optional[float] = None
    is_silent: Optional[bool] = None

    # CLAP score
    clap_score: Optional[float] = None


def get_device():
    """Get the best available PyTorch device."""
    try:
        import torch

        if torch.backends.mps.is_available():
            return "mps"
        elif torch.cuda.is_available():
            return "cuda"
        else:
            return "cpu"
    except ImportError:
        return "cpu"


def clear_memory(device: str):
    """Clear memory for the given device."""
    gc.collect()
    try:
        import torch

        if device == "mps":
            torch.mps.empty_cache()
        elif device == "cuda":
            torch.cuda.empty_cache()
    except Exception:
        pass


def get_memory_usage(device: str) -> Optional[float]:
    """Get current memory usage in MB."""
    try:
        import torch

        if device == "mps":
            # MPS doesn't have direct memory query, estimate from process
            import psutil

            process = psutil.Process()
            return process.memory_info().rss / (1024 * 1024)
        elif device == "cuda":
            return torch.cuda.max_memory_allocated() / (1024 * 1024)
        else:
            return None
    except Exception:
        return None


def measure_audio_quality(audio: np.ndarray, sample_rate: int) -> Dict:
    """Compute basic audio quality metrics."""
    if audio.ndim > 1:
        audio_mono = audio.mean(axis=0)
    else:
        audio_mono = audio

    rms = float(np.sqrt(np.mean(audio_mono**2)))
    is_silent = rms < 0.001

    return {"rms_energy": rms, "is_silent": is_silent}


def benchmark_musicgen_pytorch(
    model_name: str,
    prompt: str,
    duration: float,
    device: str = "mps",
    seed: Optional[int] = None,
    compute_clap: bool = False,
) -> PyTorchBenchmarkResult:
    """
    Benchmark MusicGen with PyTorch.

    Args:
        model_name: MusicGen variant ("small", "medium", "large", "melody")
        prompt: Text prompt for generation
        duration: Audio duration in seconds
        device: Device to use ("mps", "cuda", "cpu")
        seed: Random seed
        compute_clap: Whether to compute CLAP score

    Returns:
        PyTorchBenchmarkResult with benchmark metrics
    """
    try:
        import torch
        import torchaudio
        from audiocraft.models import MusicGen
    except ImportError as e:
        raise RuntimeError(
            f"Required packages not installed: {e}. "
            "Install with: pip install torch torchaudio audiocraft"
        )

    clear_memory(device)

    # Set seed
    if seed is not None:
        torch.manual_seed(seed)
        if device == "cuda":
            torch.cuda.manual_seed(seed)

    # Load model
    print(f"  Loading MusicGen-{model_name} on {device}...")
    t0 = time.perf_counter()

    model = MusicGen.get_pretrained(f"facebook/musicgen-{model_name}")

    # Move to device if supported
    if device == "mps":
        # audiocraft may not fully support MPS, try anyway
        try:
            model = model.to(device)
        except Exception as e:
            print(f"    Warning: Could not move to MPS: {e}")
            device = "cpu"
    elif device == "cuda":
        model = model.to(device)

    load_time = time.perf_counter() - t0
    print(f"    Load time: {load_time:.2f}s")

    # Set generation parameters
    model.set_generation_params(duration=duration)

    # Generate
    print(f"  Generating ({duration}s audio)...")
    t0 = time.perf_counter()

    with torch.no_grad():
        wav = model.generate([prompt])

    generation_time = time.perf_counter() - t0
    total_time = load_time + generation_time
    print(f"    Generation time: {generation_time:.2f}s")

    # Get audio
    audio = wav[0].cpu().numpy()
    sample_rate = model.sample_rate

    # Ensure shape is (channels, samples)
    if audio.ndim == 1:
        audio = audio.reshape(1, -1)

    actual_duration = audio.shape[-1] / sample_rate
    channels = audio.shape[0]

    # Calculate realtime factor
    realtime_factor = actual_duration / generation_time if generation_time > 0 else 0

    # Memory usage
    peak_memory = get_memory_usage(device)

    # Quality metrics
    quality = measure_audio_quality(audio, sample_rate)

    # CLAP score (optional)
    clap_score = None
    if compute_clap:
        try:
            # Import CLAP utilities from the other benchmark script
            sys.path.insert(0, str(Path(__file__).parent.parent))
            from scripts.benchmark_ground_truth import compute_clap_score as _compute_clap

            clap_score = _compute_clap(audio, sample_rate, prompt)
            if clap_score is not None:
                print(f"    CLAP score: {clap_score:.3f}")
        except Exception as e:
            print(f"    CLAP scoring failed: {e}")

    # Clean up
    del model
    clear_memory(device)

    return PyTorchBenchmarkResult(
        framework="pytorch",
        device=device,
        model_name="MusicGen",
        model_variant=model_name,
        load_time_sec=load_time,
        generation_time_sec=generation_time,
        total_time_sec=total_time,
        audio_duration_sec=actual_duration,
        sample_rate=sample_rate,
        channels=channels,
        realtime_factor=realtime_factor,
        peak_memory_mb=peak_memory,
        clap_score=clap_score,
        **quality,
    )


def benchmark_stable_audio_pytorch(
    model_path: str,
    prompt: str,
    duration: float,
    steps: int = 100,
    cfg_scale: float = 7.0,
    device: str = "mps",
    seed: Optional[int] = None,
) -> Optional[PyTorchBenchmarkResult]:
    """
    Benchmark Stable Audio with PyTorch (if available).

    Note: stable-audio-tools may not be available or may require specific setup.
    This function attempts to benchmark but gracefully handles failures.
    """
    try:
        import torch

        # Try to import stable-audio-tools
        from stable_audio_tools import get_pretrained_model
        from stable_audio_tools.inference.generation import generate_diffusion_cond
    except ImportError:
        print("  stable-audio-tools not installed, skipping PyTorch StableAudio benchmark")
        return None

    clear_memory(device)

    if seed is not None:
        torch.manual_seed(seed)

    print(f"  Loading StableAudio on {device}...")
    t0 = time.perf_counter()

    try:
        model, model_config = get_pretrained_model("stabilityai/stable-audio-open-1.0")
        model = model.to(device)
        load_time = time.perf_counter() - t0
        print(f"    Load time: {load_time:.2f}s")
    except Exception as e:
        print(f"  Failed to load StableAudio: {e}")
        return None

    # Generate
    print(f"  Generating ({steps} steps)...")
    t0 = time.perf_counter()

    try:
        with torch.no_grad():
            output = generate_diffusion_cond(
                model,
                steps=steps,
                cfg_scale=cfg_scale,
                conditioning={"prompt": prompt, "seconds_start": 0, "seconds_total": duration},
                sample_size=int(duration * model_config.get("sample_rate", 44100)),
                seed=seed or -1,
            )
        generation_time = time.perf_counter() - t0
    except Exception as e:
        print(f"  Generation failed: {e}")
        return None

    total_time = load_time + generation_time

    # Get audio
    audio = output.cpu().numpy()
    sample_rate = model_config.get("sample_rate", 44100)

    if audio.ndim == 1:
        audio = audio.reshape(1, -1)

    actual_duration = audio.shape[-1] / sample_rate
    realtime_factor = actual_duration / generation_time if generation_time > 0 else 0

    peak_memory = get_memory_usage(device)
    quality = measure_audio_quality(audio, sample_rate)

    del model
    clear_memory(device)

    return PyTorchBenchmarkResult(
        framework="pytorch",
        device=device,
        model_name="StableAudio",
        model_variant="open-1.0",
        load_time_sec=load_time,
        generation_time_sec=generation_time,
        total_time_sec=total_time,
        audio_duration_sec=actual_duration,
        sample_rate=sample_rate,
        channels=audio.shape[0],
        realtime_factor=realtime_factor,
        peak_memory_mb=peak_memory,
        **quality,
    )


def print_results(results: List[PyTorchBenchmarkResult]):
    """Print formatted results table."""
    print("\n" + "=" * 90)
    print("PYTORCH REFERENCE BENCHMARK RESULTS")
    print("=" * 90)

    header = f"{'Model':<12} {'Variant':<10} {'Device':<8} {'Gen (s)':<10} {'RTF':<8} {'Memory (MB)':<12}"
    print(header)
    print("-" * 90)

    for r in results:
        memory_str = f"{r.peak_memory_mb:.0f}" if r.peak_memory_mb else "N/A"
        row = (
            f"{r.model_name:<12} {r.model_variant:<10} {r.device:<8} "
            f"{r.generation_time_sec:<10.2f} {r.realtime_factor:<8.2f} {memory_str:<12}"
        )
        print(row)

    print("\nRTF = Realtime Factor (audio duration / generation time)")


def save_results(results: List[PyTorchBenchmarkResult], output_path: Path):
    """Save results to JSON."""
    output = {
        "timestamp": datetime.now().isoformat(),
        "framework": "pytorch",
        "results": [asdict(r) for r in results],
    }

    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\nResults saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="PyTorch Reference Benchmark")

    parser.add_argument(
        "--models",
        nargs="+",
        default=["musicgen"],
        choices=["musicgen", "stable-audio"],
        help="Models to benchmark",
    )
    parser.add_argument(
        "--musicgen-variants",
        nargs="+",
        default=["small"],
        choices=["small", "medium", "large", "melody"],
        help="MusicGen variants to test",
    )
    parser.add_argument("--prompt", type=str, default="electronic dance music with synthesizers and drums")
    parser.add_argument("--duration", type=float, default=10.0, help="Audio duration in seconds")
    parser.add_argument("--runs", type=int, default=1, help="Number of runs")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--compute-clap", action="store_true", help="Compute CLAP scores")
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        choices=["mps", "cuda", "cpu"],
        help="Device to use (auto-detect if not specified)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("benchmark_results"),
        help="Output directory",
    )

    args = parser.parse_args()

    # Auto-detect device
    device = args.device or get_device()

    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("PYTORCH REFERENCE BENCHMARK")
    print("=" * 80)
    print(f"Device: {device}")
    print(f"Models: {', '.join(args.models)}")
    print(f"Duration: {args.duration}s")
    print(f"Runs: {args.runs}")
    print(f"Prompt: \"{args.prompt}\"")

    # Check PyTorch is available
    try:
        import torch

        print(f"PyTorch version: {torch.__version__}")
        if device == "mps":
            print(f"MPS available: {torch.backends.mps.is_available()}")
        elif device == "cuda":
            print(f"CUDA available: {torch.cuda.is_available()}")
    except ImportError:
        print("ERROR: PyTorch not installed. Install with: pip install torch")
        return

    results = []

    # Benchmark MusicGen
    if "musicgen" in args.models:
        for variant in args.musicgen_variants:
            for run in range(args.runs):
                print(f"\n{'='*60}")
                print(f"MusicGen-{variant}, Run {run + 1}/{args.runs}")
                print(f"{'='*60}")

                try:
                    result = benchmark_musicgen_pytorch(
                        model_name=variant,
                        prompt=args.prompt,
                        duration=args.duration,
                        device=device,
                        seed=args.seed + run,
                        compute_clap=args.compute_clap,
                    )
                    results.append(result)
                    print(f"  RTF: {result.realtime_factor:.2f}x")
                except Exception as e:
                    print(f"  ERROR: {e}")
                    import traceback

                    traceback.print_exc()

    # Benchmark Stable Audio
    if "stable-audio" in args.models:
        for run in range(args.runs):
            print(f"\n{'='*60}")
            print(f"StableAudio, Run {run + 1}/{args.runs}")
            print(f"{'='*60}")

            try:
                result = benchmark_stable_audio_pytorch(
                    model_path="stabilityai/stable-audio-open-1.0",
                    prompt=args.prompt,
                    duration=min(args.duration, 47.0),
                    device=device,
                    seed=args.seed + run,
                )
                if result:
                    results.append(result)
                    print(f"  RTF: {result.realtime_factor:.2f}x")
            except Exception as e:
                print(f"  ERROR: {e}")

    # Print and save results
    if results:
        print_results(results)
        output_path = args.output_dir / "pytorch_reference_results.json"
        save_results(results, output_path)

    print("\nPyTorch benchmark complete!")


if __name__ == "__main__":
    main()
