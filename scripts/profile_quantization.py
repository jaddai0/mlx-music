#!/usr/bin/env python3
"""
Profile quantization performance to investigate INT4 vs INT8 performance parity.

This script benchmarks quantized operations to understand why INT4 is not
significantly faster than INT8 (and may even be slower).

Hypotheses to test:
1. Memory bandwidth is NOT the bottleneck (compute-bound operations)
2. Dequantization overhead for INT4 is higher than INT8
3. MLX INT4 kernels may be less optimized than INT8 kernels

Usage:
    python scripts/profile_quantization.py
    python scripts/profile_quantization.py --sizes 1024 2048 4096
    python scripts/profile_quantization.py --iterations 100
"""

import argparse
import time
from dataclasses import dataclass
from typing import List, Tuple

import mlx.core as mx
import mlx.nn as nn

# Warmup iterations to ensure JIT compilation
WARMUP_ITERATIONS = 5


@dataclass
class BenchmarkResult:
    """Result from a single benchmark."""

    size: Tuple[int, int]
    dtype: str
    bits: int
    mean_ms: float
    std_ms: float
    throughput_gflops: float
    memory_bandwidth_gbps: float


def benchmark_matmul(
    a: mx.array,
    b: mx.array,
    iterations: int = 50,
) -> Tuple[float, float]:
    """
    Benchmark matrix multiplication.

    Returns:
        Tuple of (mean_ms, std_ms)
    """
    # Warmup
    for _ in range(WARMUP_ITERATIONS):
        c = a @ b
        mx.synchronize()

    # Benchmark
    times = []
    for _ in range(iterations):
        start = time.perf_counter()
        c = a @ b
        mx.synchronize()
        end = time.perf_counter()
        times.append((end - start) * 1000)  # Convert to ms

    mean_ms = sum(times) / len(times)
    variance = sum((t - mean_ms) ** 2 for t in times) / len(times)
    std_ms = variance ** 0.5

    return mean_ms, std_ms


def benchmark_quantized_matmul(
    a: mx.array,
    linear: nn.Linear,
    iterations: int = 50,
) -> Tuple[float, float]:
    """
    Benchmark quantized linear layer (matmul with dequantization).

    Returns:
        Tuple of (mean_ms, std_ms)
    """
    # Warmup
    for _ in range(WARMUP_ITERATIONS):
        c = linear(a)
        mx.synchronize()

    # Benchmark
    times = []
    for _ in range(iterations):
        start = time.perf_counter()
        c = linear(a)
        mx.synchronize()
        end = time.perf_counter()
        times.append((end - start) * 1000)

    mean_ms = sum(times) / len(times)
    variance = sum((t - mean_ms) ** 2 for t in times) / len(times)
    std_ms = variance ** 0.5

    return mean_ms, std_ms


def calculate_metrics(
    m: int,
    n: int,
    k: int,
    time_ms: float,
    bits: int,
) -> Tuple[float, float]:
    """
    Calculate throughput and memory bandwidth.

    Args:
        m, n, k: Matrix dimensions (a: m*k, b: k*n, c: m*n)
        time_ms: Time in milliseconds
        bits: Quantization bits (4, 8, or 16)

    Returns:
        Tuple of (throughput_gflops, bandwidth_gbps)
    """
    # FLOPS: 2 * M * N * K (multiply-add for each output element)
    flops = 2 * m * n * k
    gflops = flops / (time_ms * 1e-3) / 1e9

    # Memory bandwidth (read A, read B, write C)
    # A: m*k elements, B: k*n elements, C: m*n elements
    # For quantized: B is bits/8 bytes per element, plus scales/biases overhead
    bytes_a = m * k * 2  # Always bfloat16 input
    bytes_b = k * n * (bits / 8) if bits < 16 else k * n * 2
    # Add overhead for scales/biases (~3% for group_size=64)
    if bits < 16:
        bytes_b *= 1.03
    bytes_c = m * n * 2  # bfloat16 output

    total_bytes = bytes_a + bytes_b + bytes_c
    gbps = total_bytes / (time_ms * 1e-3) / 1e9

    return gflops, gbps


def run_profiling(
    sizes: List[int],
    iterations: int = 50,
    batch_sizes: List[int] = None,
) -> List[BenchmarkResult]:
    """
    Run comprehensive quantization profiling.

    Args:
        sizes: List of matrix sizes to test (square matrices n*n)
        iterations: Number of iterations per test
        batch_sizes: Batch sizes to test (default: [1, 4, 16])

    Returns:
        List of BenchmarkResult
    """
    if batch_sizes is None:
        batch_sizes = [1, 4, 16]

    results = []

    for n in sizes:
        for batch in batch_sizes:
            print(f"\nProfiling {batch}x{n} @ {n}x{n} matrix multiply...")

            # Create input matrix (batch x n)
            a = mx.random.normal((batch, n)).astype(mx.bfloat16)

            # --- FP16 Baseline ---
            print("  FP16 baseline...")
            linear_fp16 = nn.Linear(n, n, bias=False)
            linear_fp16.weight = mx.random.normal((n, n)).astype(mx.bfloat16)
            mx.synchronize()

            mean_ms, std_ms = benchmark_quantized_matmul(a, linear_fp16, iterations)
            gflops, gbps = calculate_metrics(batch, n, n, mean_ms, 16)
            results.append(BenchmarkResult(
                size=(batch, n),
                dtype="bfloat16",
                bits=16,
                mean_ms=mean_ms,
                std_ms=std_ms,
                throughput_gflops=gflops,
                memory_bandwidth_gbps=gbps,
            ))
            print(f"    Time: {mean_ms:.3f} +/- {std_ms:.3f} ms")
            print(f"    Throughput: {gflops:.1f} GFLOPS, Bandwidth: {gbps:.1f} GB/s")

            # --- INT8 ---
            print("  INT8 quantized...")
            linear_int8 = nn.Linear(n, n, bias=False)
            linear_int8.weight = mx.random.normal((n, n)).astype(mx.bfloat16)
            mx.synchronize()
            nn.quantize(linear_int8, bits=8, group_size=64)
            mx.synchronize()

            mean_ms, std_ms = benchmark_quantized_matmul(a, linear_int8, iterations)
            gflops, gbps = calculate_metrics(batch, n, n, mean_ms, 8)
            results.append(BenchmarkResult(
                size=(batch, n),
                dtype="int8",
                bits=8,
                mean_ms=mean_ms,
                std_ms=std_ms,
                throughput_gflops=gflops,
                memory_bandwidth_gbps=gbps,
            ))
            print(f"    Time: {mean_ms:.3f} +/- {std_ms:.3f} ms")
            print(f"    Throughput: {gflops:.1f} GFLOPS, Bandwidth: {gbps:.1f} GB/s")

            # --- INT4 ---
            print("  INT4 quantized...")
            linear_int4 = nn.Linear(n, n, bias=False)
            linear_int4.weight = mx.random.normal((n, n)).astype(mx.bfloat16)
            mx.synchronize()
            nn.quantize(linear_int4, bits=4, group_size=64)
            mx.synchronize()

            mean_ms, std_ms = benchmark_quantized_matmul(a, linear_int4, iterations)
            gflops, gbps = calculate_metrics(batch, n, n, mean_ms, 4)
            results.append(BenchmarkResult(
                size=(batch, n),
                dtype="int4",
                bits=4,
                mean_ms=mean_ms,
                std_ms=std_ms,
                throughput_gflops=gflops,
                memory_bandwidth_gbps=gbps,
            ))
            print(f"    Time: {mean_ms:.3f} +/- {std_ms:.3f} ms")
            print(f"    Throughput: {gflops:.1f} GFLOPS, Bandwidth: {gbps:.1f} GB/s")

            # Clean up
            del linear_fp16, linear_int8, linear_int4, a

    return results


def print_summary_table(results: List[BenchmarkResult]):
    """Print summary table of results."""
    print("\n" + "=" * 100)
    print("QUANTIZATION PROFILING SUMMARY")
    print("=" * 100)

    print(f"\n{'Size':<15} {'Type':<10} {'Time (ms)':<15} {'GFLOPS':<12} {'GB/s':<12} {'Speedup':<10}")
    print("-" * 100)

    # Group by size for comparison
    sizes = sorted(set(r.size for r in results))

    for size in sizes:
        size_results = [r for r in results if r.size == size]

        # Find baseline (bfloat16)
        baseline = next((r for r in size_results if r.bits == 16), None)
        baseline_time = baseline.mean_ms if baseline else 1.0

        for r in sorted(size_results, key=lambda x: x.bits, reverse=True):
            speedup = baseline_time / r.mean_ms if r.mean_ms > 0 else 0
            print(
                f"{str(r.size):<15} {r.dtype:<10} "
                f"{r.mean_ms:.3f} +/- {r.std_ms:.3f}    "
                f"{r.throughput_gflops:<12.1f} {r.memory_bandwidth_gbps:<12.1f} "
                f"{speedup:.2f}x"
            )

        print()  # Blank line between sizes


def analyze_results(results: List[BenchmarkResult]):
    """Analyze results and print insights."""
    print("\n" + "=" * 100)
    print("ANALYSIS")
    print("=" * 100)

    # Group by size
    sizes = sorted(set(r.size for r in results))

    for size in sizes:
        size_results = [r for r in results if r.size == size]
        fp16 = next((r for r in size_results if r.bits == 16), None)
        int8 = next((r for r in size_results if r.bits == 8), None)
        int4 = next((r for r in size_results if r.bits == 4), None)

        if not all([fp16, int8, int4]):
            continue

        print(f"\nSize {size}:")

        # INT8 vs FP16
        int8_speedup = fp16.mean_ms / int8.mean_ms
        int8_expected = 2.0  # 2x memory reduction should give ~2x speedup if memory-bound
        print(f"  INT8 speedup: {int8_speedup:.2f}x (expected ~{int8_expected:.1f}x if memory-bound)")

        # INT4 vs FP16
        int4_speedup = fp16.mean_ms / int4.mean_ms
        int4_expected = 4.0  # 4x memory reduction should give ~4x speedup if memory-bound
        print(f"  INT4 speedup: {int4_speedup:.2f}x (expected ~{int4_expected:.1f}x if memory-bound)")

        # INT4 vs INT8
        int4_vs_int8 = int8.mean_ms / int4.mean_ms
        print(f"  INT4 vs INT8: {int4_vs_int8:.2f}x (expected ~2x if memory-bound)")

        # Diagnosis
        if int4_speedup < 1.5:
            print("  DIAGNOSIS: Operation is likely COMPUTE-BOUND, not memory-bound.")
            print("             Quantization saves memory but not compute time.")
        elif int4_vs_int8 < 1.2:
            print("  DIAGNOSIS: INT4 dequantization overhead may be significant,")
            print("             or INT4 kernels are less optimized than INT8.")
        else:
            print("  DIAGNOSIS: Performance scales with quantization as expected.")

        # Memory bandwidth analysis
        print(f"\n  Memory bandwidth utilization:")
        print(f"    FP16: {fp16.memory_bandwidth_gbps:.1f} GB/s")
        print(f"    INT8: {int8.memory_bandwidth_gbps:.1f} GB/s")
        print(f"    INT4: {int4.memory_bandwidth_gbps:.1f} GB/s")

        # M3 Ultra theoretical bandwidth is ~800 GB/s
        # If actual bandwidth is much lower, we're compute-bound
        max_bw = max(fp16.memory_bandwidth_gbps, int8.memory_bandwidth_gbps, int4.memory_bandwidth_gbps)
        if max_bw < 300:
            print(f"  NOTE: Peak bandwidth ({max_bw:.0f} GB/s) is well below M3 Ultra's ~800 GB/s")
            print("        This suggests the operation is compute-bound.")


def main():
    parser = argparse.ArgumentParser(
        description="Profile quantization performance",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument(
        "--sizes",
        nargs="+",
        type=int,
        default=[1024, 2048, 2560, 4096],
        help="Matrix sizes to test (default: 1024 2048 2560 4096)",
    )
    parser.add_argument(
        "--batch-sizes",
        nargs="+",
        type=int,
        default=[1, 4, 16],
        help="Batch sizes to test (default: 1 4 16)",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=50,
        help="Number of iterations per test (default: 50)",
    )

    args = parser.parse_args()

    print("=" * 100)
    print("MLX QUANTIZATION PROFILER")
    print("=" * 100)
    print(f"Matrix sizes: {args.sizes}")
    print(f"Batch sizes: {args.batch_sizes}")
    print(f"Iterations: {args.iterations}")

    results = run_profiling(
        sizes=args.sizes,
        iterations=args.iterations,
        batch_sizes=args.batch_sizes,
    )

    print_summary_table(results)
    analyze_results(results)

    print("\nProfiling complete!")


if __name__ == "__main__":
    main()
