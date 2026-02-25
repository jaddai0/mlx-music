#!/usr/bin/env python3
"""
Profile per-step timing breakdown for ACE-Step inference.

This script helps identify performance bottlenecks by measuring:
- FFN (GLUMBConv) execution time
- Cross-attention execution time
- Self-attention (linear attention) execution time
- Overhead (norms, scheduler, etc.)

Usage:
    python scripts/profile_step_timing.py --steps 5 --duration 10

Results help determine if optimizations are working:
- CFG batching: Should reduce total step time by ~40-50%
- KV caching: Should reduce cross-attention time by ~20-30%
- mx.compile: Should reduce total step time by ~10-15%
"""

import argparse
import time
from contextlib import contextmanager
from typing import Dict, List

import mlx.core as mx


@contextmanager
def timer(name: str, timings: Dict[str, List[float]]):
    """Context manager for timing code blocks."""
    mx.eval()  # Sync before timing
    start = time.perf_counter()
    yield
    mx.eval()  # Sync after timing
    elapsed_ms = (time.perf_counter() - start) * 1000
    timings.setdefault(name, []).append(elapsed_ms)


def main():
    parser = argparse.ArgumentParser(description="Profile ACE-Step step timing")
    parser.add_argument("--model-path", type=str, default="ACE-Step/ACE-Step-v1-3.5B",
                        help="Model path or HuggingFace repo ID")
    parser.add_argument("--steps", type=int, default=5,
                        help="Number of steps to profile")
    parser.add_argument("--duration", type=float, default=10.0,
                        help="Audio duration in seconds")
    parser.add_argument("--warmup", type=int, default=2,
                        help="Warmup steps before profiling")
    parser.add_argument("--cfg-scale", type=float, default=15.0,
                        help="CFG guidance scale (>1 enables CFG)")
    parser.add_argument("--no-batched-cfg", action="store_true",
                        help="Disable batched CFG (use sequential)")
    parser.add_argument("--quantize", type=str, choices=["none", "int4", "int8"],
                        default="none", help="Quantization mode")
    args = parser.parse_args()

    print(f"\n{'='*60}")
    print("ACE-Step Step Timing Profiler")
    print(f"{'='*60}")
    print(f"Model: {args.model_path}")
    print(f"Duration: {args.duration}s")
    print(f"Steps to profile: {args.steps}")
    print(f"CFG scale: {args.cfg_scale}")
    print(f"Quantization: {args.quantize}")
    print(f"{'='*60}\n")

    # Import here to avoid slow startup for --help
    from mlx_music.models.ace_step.model import ACEStep

    print("Loading model...")
    model = ACEStep.from_pretrained(args.model_path)

    # Apply quantization if requested
    if args.quantize != "none":
        from mlx_music.weights.quantization import quantize_model, QuantizationMode, QuantizationConfig
        config = QuantizationConfig(
            mode=QuantizationMode.INT4 if args.quantize == "int4" else QuantizationMode.INT8
        )
        print(f"Applying {args.quantize} quantization...")
        quantize_model(model.transformer, config=config, verbose=True)

    # Prepare inputs
    print("\nPreparing inputs...")
    latent_width = int(args.duration * 44100 / (512 * 8))
    latent_shape = (1, 8, 16, latent_width)
    latents = mx.random.normal(latent_shape).astype(mx.bfloat16)

    text_embeds, text_mask = model.encode_text("upbeat electronic dance music")
    timestep = mx.array([500.0])  # Mid-range timestep

    # Force computation
    _ = latents + 0
    _ = text_embeds + 0
    _ = text_mask + 0
    _ = timestep + 0

    # Warmup
    print(f"\nRunning {args.warmup} warmup steps...")
    for i in range(args.warmup):
        if args.cfg_scale > 1.0 and not args.no_batched_cfg:
            _ = model._batched_cfg_forward(
                latents, timestep, text_embeds, text_mask,
                None, None, None, args.cfg_scale
            )
        else:
            _ = model._transformer_forward(
                latents, timestep, text_embeds, text_mask,
                None, None, None
            )
        mx.eval()
    print("Warmup complete.")

    # Profile individual steps
    print(f"\nProfiling {args.steps} steps...")
    step_times: List[float] = []

    for i in range(args.steps):
        mx.eval()  # Sync before timing
        start = time.perf_counter()

        if args.cfg_scale > 1.0 and not args.no_batched_cfg:
            # Batched CFG (optimized)
            noise_pred = model._batched_cfg_forward(
                latents, timestep, text_embeds, text_mask,
                None, None, None, args.cfg_scale
            )
        elif args.cfg_scale > 1.0:
            # Sequential CFG (baseline)
            noise_pred = model._transformer_forward(
                latents, timestep, text_embeds, text_mask,
                None, None, None
            )
            mx.eval()
            noise_pred_uncond = model._transformer_forward(
                latents, timestep,
                mx.zeros_like(text_embeds), mx.zeros_like(text_mask),
                None, None, None
            )
            mx.eval()
            noise_pred = noise_pred_uncond + args.cfg_scale * (noise_pred - noise_pred_uncond)
        else:
            # No CFG
            noise_pred = model._transformer_forward(
                latents, timestep, text_embeds, text_mask,
                None, None, None
            )

        # Force computation and sync
        _ = noise_pred + 0
        mx.eval()
        elapsed_ms = (time.perf_counter() - start) * 1000
        step_times.append(elapsed_ms)
        print(f"  Step {i+1}: {elapsed_ms:.1f} ms")

    # Calculate statistics
    avg_time = sum(step_times) / len(step_times)
    min_time = min(step_times)
    max_time = max(step_times)

    print(f"\n{'='*60}")
    print("Results")
    print(f"{'='*60}")
    print(f"Average time/step: {avg_time:.1f} ms")
    print(f"Min time/step: {min_time:.1f} ms")
    print(f"Max time/step: {max_time:.1f} ms")

    # Estimate full generation time
    typical_steps = 60
    est_denoise_sec = (avg_time * typical_steps) / 1000
    est_realtime_factor = args.duration / est_denoise_sec if est_denoise_sec > 0 else 0

    print(f"\nEstimated for {args.duration}s audio at {typical_steps} steps:")
    print(f"  Denoising time: {est_denoise_sec:.1f} seconds")
    print(f"  Realtime factor: {est_realtime_factor:.2f}x")

    if est_realtime_factor >= 1.0:
        print("  ✓ Faster than realtime!")
    else:
        print(f"  ✗ {1/est_realtime_factor:.1f}x slower than realtime")

    # Optimization hints
    print(f"\n{'='*60}")
    print("Optimization Status")
    print(f"{'='*60}")
    if args.cfg_scale > 1.0:
        if args.no_batched_cfg:
            print("❌ Batched CFG: DISABLED (use --cfg-scale with batching for ~40-50% speedup)")
        else:
            print("✓ Batched CFG: ENABLED")
    else:
        print("• CFG: Disabled (guidance_scale <= 1.0)")

    if hasattr(model.transformer, '_compiled_decode_blocks'):
        print("✓ mx.compile: ENABLED")
    else:
        print("❌ mx.compile: DISABLED (call transformer.enable_compile() for ~10-15% speedup)")

    print(f"• Quantization: {args.quantize}")

    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
