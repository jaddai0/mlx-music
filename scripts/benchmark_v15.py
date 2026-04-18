#!/usr/bin/env python3
"""
Benchmark suite for ACE-Step v1.5 on MLX.

Measures speed, memory, and real-time factor across:
- Quantization modes (BF16, INT4, INT8, MIXED)
- Generation durations (2s, 10s, 30s, 60s)
- Model variants (turbo)

Usage:
    python scripts/benchmark_v15.py
    python scripts/benchmark_v15.py --iterations 3 --durations 10 30
    python scripts/benchmark_v15.py --quantization int4 --duration 30
"""

import argparse
import gc
import json
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

sys.path.insert(0, str(Path(__file__).parent.parent))

import mlx.core as mx


@dataclass
class BenchmarkResult:
    variant: str
    quantization: str
    duration_target: float
    generation_time: float
    peak_memory_gb: float
    audio_samples: int
    sample_rate: int
    real_time_factor: float
    iteration: int

    def __repr__(self):
        return (
            f"{self.variant}/{self.quantization} "
            f"{self.duration_target}s: {self.generation_time:.2f}s "
            f"(RTF={self.real_time_factor:.1f}x, "
            f"peak={self.peak_memory_gb:.2f}GB)"
        )


def reset_memory():
    gc.collect()
    mx.clear_cache()
    if hasattr(mx, "reset_peak_memory"):
        mx.reset_peak_memory()
    elif hasattr(mx, "metal") and hasattr(mx.metal, "reset_peak_memory"):
        mx.metal.reset_peak_memory()


def get_peak_memory_gb():
    return mx.get_peak_memory() / (1024**3)


def run_benchmark(
    models_dir: str,
    variant: str = "turbo",
    quantization: Optional[str] = None,
    durations: List[float] = None,
    iterations: int = 3,
    seed: int = 42,
    warmup: bool = True,
) -> List[BenchmarkResult]:
    from mlx_music.models.ace_step_v15.model import ACEStepV15

    if durations is None:
        durations = [2.0, 10.0, 30.0]

    quant_label = quantization or "bf16"
    print(f"\n{'='*60}")
    print(f"Loading {variant} model (quantization={quant_label})")
    print(f"{'='*60}")

    t0 = time.time()
    model = ACEStepV15.from_pretrained(
        models_dir, variant=variant, quantization=quantization
    )
    load_time = time.time() - t0
    print(f"Model loaded in {load_time:.2f}s")

    # Warmup run
    if warmup:
        print("Warmup run...")
        reset_memory()
        model.generate(duration=1.0, seed=seed)

    results = []
    for dur in durations:
        for i in range(iterations):
            reset_memory()

            t0 = time.time()
            result = model.generate(duration=dur, seed=seed + i)
            gen_time = time.time() - t0
            peak_mem = get_peak_memory_gb()

            audio = result["audio"]
            actual_duration = audio.shape[0] / result["sample_rate"]
            rtf = actual_duration / gen_time

            bench = BenchmarkResult(
                variant=variant,
                quantization=quant_label,
                duration_target=dur,
                generation_time=gen_time,
                peak_memory_gb=peak_mem,
                audio_samples=audio.shape[0],
                sample_rate=result["sample_rate"],
                real_time_factor=rtf,
                iteration=i,
            )
            results.append(bench)
            print(f"  {bench}")

    del model
    gc.collect()
    mx.clear_cache()

    return results


def print_summary(results: List[BenchmarkResult]):
    print(f"\n{'='*60}")
    print("BENCHMARK SUMMARY")
    print(f"{'='*60}")

    # Group by (variant, quantization, duration)
    groups: Dict[tuple, List[BenchmarkResult]] = {}
    for r in results:
        key = (r.variant, r.quantization, r.duration_target)
        groups.setdefault(key, []).append(r)

    header = f"{'Config':<20} {'Duration':>8} {'Time (avg)':>12} {'RTF':>8} {'Peak Mem':>10}"
    print(header)
    print("-" * len(header))

    for (variant, quant, dur), runs in sorted(groups.items()):
        avg_time = sum(r.generation_time for r in runs) / len(runs)
        avg_rtf = sum(r.real_time_factor for r in runs) / len(runs)
        avg_mem = sum(r.peak_memory_gb for r in runs) / len(runs)
        label = f"{variant}/{quant}"
        print(
            f"{label:<20} {dur:>6.0f}s {avg_time:>10.2f}s {avg_rtf:>7.1f}x {avg_mem:>8.2f}GB"
        )


def main():
    parser = argparse.ArgumentParser(description="Benchmark ACE-Step v1.5")
    parser.add_argument(
        "--models-dir",
        default="/Users/dustinpainter/Dev-Projects/audio-models/ACE-Step-1.5/models",
        help="Path to ACE-Step v1.5 models directory",
    )
    parser.add_argument(
        "--variant", default="turbo", help="Model variant (default: turbo)"
    )
    parser.add_argument(
        "--quantization",
        nargs="*",
        default=None,
        help="Quantization modes to test (bf16 int4 int8 mixed)",
    )
    parser.add_argument(
        "--durations",
        type=float,
        nargs="+",
        default=[2.0, 10.0, 30.0],
        help="Durations to benchmark",
    )
    parser.add_argument(
        "--iterations", type=int, default=3, help="Iterations per config"
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--output", type=str, default=None, help="Save results to JSON file"
    )
    args = parser.parse_args()

    quant_modes = args.quantization or [None, "int4", "int8", "mixed"]
    # Convert "bf16" to None
    quant_modes = [None if q == "bf16" else q for q in quant_modes]

    all_results = []
    for quant in quant_modes:
        results = run_benchmark(
            models_dir=args.models_dir,
            variant=args.variant,
            quantization=quant,
            durations=args.durations,
            iterations=args.iterations,
            seed=args.seed,
        )
        all_results.extend(results)

    print_summary(all_results)

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "timestamp": datetime.now().isoformat(),
            "system": {
                "chip": "Apple Silicon",
                "mlx_version": str(mx.__version__) if hasattr(mx, "__version__") else "unknown",
            },
            "results": [asdict(r) for r in all_results],
        }
        with open(output_path, "w") as f:
            json.dump(data, f, indent=2)
        print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
