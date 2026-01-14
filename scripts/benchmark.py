#!/usr/bin/env python3
"""
Benchmark script for MLX-Music ACE-Step.

Compares generation performance across quantization levels with controlled parameters.
All benchmarks use identical:
- Random seed
- Prompt
- Number of steps
- Duration
- CFG scale

Usage:
    python scripts/benchmark.py
    python scripts/benchmark.py --iterations 3 --steps 20
"""

import argparse
import gc
import json
import sys
import time
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Add mlx_music to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import mlx.core as mx

from mlx_music.models.ace_step.transformer import ACEStepTransformer, ACEStepConfig
from mlx_music.models.ace_step.dcae import DCAE, DCAEConfig
from mlx_music.models.ace_step.vocoder import HiFiGANVocoder, VocoderConfig
from mlx_music.models.ace_step.text_encoder import get_text_encoder
from mlx_music.weights.weight_loader import (
    load_safetensors,
    convert_torch_to_mlx,
    ACE_STEP_TRANSFORMER_MAPPINGS,
    generate_transformer_block_mappings,
)
from mlx_music.weights.quantization import (
    quantize_model,
    get_model_size,
    QuantizationConfig,
)


@dataclass
class BenchmarkConfig:
    """Configuration for benchmark run."""
    prompt: str = "electronic dance music with synthesizers and drums"
    steps: int = 20
    duration: float = 5.0  # seconds of audio
    cfg_scale: float = 7.5
    seed: int = 42
    iterations: int = 3
    warmup_iterations: int = 1


@dataclass
class BenchmarkResult:
    """Results from a single benchmark run."""
    quantization: str
    model_size_mb: float
    num_parameters: int

    # Timing (seconds)
    load_time: float
    encode_time: float
    denoise_time: float
    decode_time: float
    total_time: float

    # Derived metrics
    audio_duration: float
    realtime_factor: float  # audio_duration / total_generation_time
    step_time: float  # time per denoising step

    # Memory (if available)
    peak_memory_mb: Optional[float] = None


def clear_memory():
    """Clear MLX memory cache."""
    gc.collect()
    mx.metal.clear_cache()


def load_transformer(model_path: Path, dtype: mx.Dtype, quantize: Optional[str] = None) -> Tuple[ACEStepTransformer, ACEStepConfig, float, int, float]:
    """Load transformer with optional quantization. Returns model, config, size_mb, num_params, load_time."""
    start = time.time()

    config = ACEStepConfig()
    model = ACEStepTransformer(config)

    weight_file = model_path / "ace_step_transformer" / "diffusion_pytorch_model.safetensors"
    weights = load_safetensors(weight_file, dtype=dtype)

    all_mappings = ACE_STEP_TRANSFORMER_MAPPINGS + generate_transformer_block_mappings(config.num_layers)
    weights = convert_torch_to_mlx(weights, all_mappings, strict=False)

    model.load_weights(list(weights.items()), strict=False)

    if quantize:
        if quantize == "int4":
            quant_config = QuantizationConfig.for_speed()
        elif quantize == "int8":
            quant_config = QuantizationConfig.for_quality()
        else:  # mixed
            quant_config = QuantizationConfig.for_balanced()
        quantize_model(model, quant_config)

    num_params, size_mb = get_model_size(model)
    load_time = time.time() - start

    return model, config, size_mb, num_params, load_time


def load_dcae(model_path: Path, dtype: mx.Dtype) -> DCAE:
    """Load DCAE model."""
    dcae_path = model_path / "music_dcae_f8c8"
    return DCAE.from_pretrained(dcae_path, dtype=dtype)


def load_vocoder(model_path: Path, dtype: mx.Dtype) -> Tuple[HiFiGANVocoder, int]:
    """Load vocoder and return model and sample rate."""
    vocoder_path = model_path / "music_vocoder"
    import json
    config_file = vocoder_path / "config.json"
    with open(config_file) as f:
        config_dict = json.load(f)
    sample_rate = config_dict.get("sampling_rate", 44100)
    model = HiFiGANVocoder.from_pretrained(vocoder_path, dtype=dtype)
    return model, sample_rate


def encode_prompt(text_encoder, prompt: str, dtype: mx.Dtype) -> Tuple[mx.array, mx.array, mx.array, mx.array, float]:
    """Encode prompt and return embeddings, masks, and timing."""
    start = time.time()

    text_embeds, text_mask = text_encoder.encode(prompt)
    text_embeds = text_embeds.astype(dtype)
    text_mask = text_mask.astype(mx.float32)

    null_embeds, null_mask = text_encoder.encode_null(
        batch_size=1,
        seq_length=text_embeds.shape[1],
    )
    null_embeds = null_embeds.astype(dtype)
    null_mask = null_mask.astype(mx.float32)

    mx.eval(text_embeds, text_mask, null_embeds, null_mask)
    encode_time = time.time() - start

    return text_embeds, text_mask, null_embeds, null_mask, encode_time


def run_denoising(
    transformer: ACEStepTransformer,
    latent: mx.array,
    text_embeds: mx.array,
    text_mask: mx.array,
    null_embeds: mx.array,
    null_mask: mx.array,
    speaker_embeds: mx.array,
    num_steps: int,
    cfg_scale: float,
) -> Tuple[mx.array, float]:
    """Run denoising and return result and timing."""
    start = time.time()

    batch_size = latent.shape[0]
    timesteps = mx.linspace(1.0, 0.0, num_steps + 1)[:-1]

    for t in timesteps:
        t_batch = mx.full((batch_size,), t.item())

        # CFG: batched conditional + unconditional
        latent_input = mx.concatenate([latent, latent], axis=0)
        t_input = mx.concatenate([t_batch, t_batch], axis=0)
        text_input = mx.concatenate([text_embeds, null_embeds], axis=0)
        mask_input = mx.concatenate([text_mask, null_mask], axis=0)
        speaker_input = mx.concatenate([speaker_embeds, speaker_embeds], axis=0)

        v_pred_both = transformer(
            hidden_states=latent_input,
            timestep=t_input,
            encoder_text_hidden_states=text_input,
            text_attention_mask=mask_input,
            speaker_embeds=speaker_input,
        )
        mx.eval(v_pred_both)

        v_pred_cond, v_pred_uncond = mx.split(v_pred_both, 2, axis=0)
        v_pred = v_pred_uncond + cfg_scale * (v_pred_cond - v_pred_uncond)

        dt = 1.0 / num_steps
        latent = latent - dt * v_pred

    mx.eval(latent)
    denoise_time = time.time() - start

    return latent, denoise_time


def run_decoding(latent: mx.array, dcae: DCAE, vocoder: HiFiGANVocoder) -> Tuple[mx.array, float]:
    """Decode latent to audio and return result and timing."""
    start = time.time()

    mel = dcae.decode(latent)
    mel = dcae.denormalize_mel(mel)

    audio_ch1 = vocoder.decode(mel[:, 0:1, :, :].squeeze(1))
    audio_ch2 = vocoder.decode(mel[:, 1:2, :, :].squeeze(1))
    audio = mx.concatenate([audio_ch1, audio_ch2], axis=1)

    mx.eval(audio)
    decode_time = time.time() - start

    return audio, decode_time


def run_single_benchmark(
    model_path: Path,
    config: BenchmarkConfig,
    quantization: Optional[str],
    text_encoder,
    dcae: DCAE,
    vocoder: HiFiGANVocoder,
    sample_rate: int,
) -> BenchmarkResult:
    """Run a single benchmark with given quantization level."""
    dtype = mx.bfloat16
    quant_name = quantization or "bfloat16"

    print(f"\n{'='*60}")
    print(f"Benchmarking: {quant_name.upper()}")
    print(f"{'='*60}")

    # Clear memory before loading
    clear_memory()

    # Load transformer
    print(f"  Loading transformer ({quant_name})...")
    transformer, trans_config, size_mb, num_params, load_time = load_transformer(
        model_path, dtype, quantization
    )
    print(f"    Size: {size_mb:.1f} MB, Params: {num_params:,}")
    print(f"    Load time: {load_time:.3f}s")

    # Prepare latent dimensions
    latent_channels = 8
    latent_height = 16
    latent_width = int(64 * config.duration / 0.75)

    # Run warmup iterations
    print(f"  Running {config.warmup_iterations} warmup iteration(s)...")
    for _ in range(config.warmup_iterations):
        mx.random.seed(config.seed)
        latent = mx.random.normal(shape=(1, latent_channels, latent_height, latent_width)).astype(dtype)
        speaker_embeds = mx.zeros((1, 512), dtype=dtype)

        text_embeds, text_mask, null_embeds, null_mask, _ = encode_prompt(
            text_encoder, config.prompt, dtype
        )

        _, _ = run_denoising(
            transformer, latent, text_embeds, text_mask, null_embeds, null_mask,
            speaker_embeds, config.steps, config.cfg_scale
        )
        clear_memory()

    # Run timed iterations
    print(f"  Running {config.iterations} timed iteration(s)...")
    encode_times = []
    denoise_times = []
    decode_times = []
    audio_durations = []

    for i in range(config.iterations):
        # Reset seed for reproducibility
        mx.random.seed(config.seed)

        # Create fresh latent
        latent = mx.random.normal(shape=(1, latent_channels, latent_height, latent_width)).astype(dtype)
        speaker_embeds = mx.zeros((1, 512), dtype=dtype)

        # Encode prompt
        text_embeds, text_mask, null_embeds, null_mask, encode_time = encode_prompt(
            text_encoder, config.prompt, dtype
        )
        encode_times.append(encode_time)

        # Denoise
        latent, denoise_time = run_denoising(
            transformer, latent, text_embeds, text_mask, null_embeds, null_mask,
            speaker_embeds, config.steps, config.cfg_scale
        )
        denoise_times.append(denoise_time)

        # Decode
        audio, decode_time = run_decoding(latent, dcae, vocoder)
        decode_times.append(decode_time)

        # Calculate audio duration
        audio_samples = audio.shape[-1]
        audio_duration = audio_samples / sample_rate
        audio_durations.append(audio_duration)

        print(f"    Iteration {i+1}: denoise={denoise_time:.2f}s, decode={decode_time:.2f}s, audio={audio_duration:.2f}s")

        clear_memory()

    # Calculate averages
    avg_encode = sum(encode_times) / len(encode_times)
    avg_denoise = sum(denoise_times) / len(denoise_times)
    avg_decode = sum(decode_times) / len(decode_times)
    avg_audio_duration = sum(audio_durations) / len(audio_durations)

    total_gen_time = avg_encode + avg_denoise + avg_decode
    realtime_factor = avg_audio_duration / total_gen_time
    step_time = avg_denoise / config.steps

    # Clean up transformer
    del transformer
    clear_memory()

    return BenchmarkResult(
        quantization=quant_name,
        model_size_mb=size_mb,
        num_parameters=num_params,
        load_time=load_time,
        encode_time=avg_encode,
        denoise_time=avg_denoise,
        decode_time=avg_decode,
        total_time=total_gen_time,
        audio_duration=avg_audio_duration,
        realtime_factor=realtime_factor,
        step_time=step_time,
    )


def print_results_table(results: List[BenchmarkResult], config: BenchmarkConfig):
    """Print formatted results table."""
    print("\n" + "=" * 80)
    print("BENCHMARK RESULTS")
    print("=" * 80)
    print(f"Prompt: \"{config.prompt}\"")
    print(f"Steps: {config.steps}, Duration: {config.duration}s, CFG: {config.cfg_scale}")
    print(f"Iterations: {config.iterations} (averaged)")
    print()

    # Header
    print(f"{'Quantization':<12} {'Size (MB)':<12} {'Denoise (s)':<14} {'Decode (s)':<12} {'Total (s)':<12} {'RT Factor':<12} {'Step (ms)':<12}")
    print("-" * 86)

    # Baseline for comparison
    baseline = results[0]

    for r in results:
        size_diff = f"({(1 - r.model_size_mb / baseline.model_size_mb) * 100:+.0f}%)" if r != baseline else ""
        denoise_speedup = f"({baseline.denoise_time / r.denoise_time:.2f}x)" if r != baseline else ""

        print(f"{r.quantization:<12} {r.model_size_mb:<6.1f}{size_diff:<6} {r.denoise_time:<8.2f}{denoise_speedup:<6} {r.decode_time:<12.2f} {r.total_time:<12.2f} {r.realtime_factor:<12.2f} {r.step_time * 1000:<12.1f}")

    print()
    print("RT Factor = Realtime Factor (audio duration / generation time)")
    print("           >1.0 means faster than realtime")

    # Summary
    print("\n" + "-" * 80)
    print("SUMMARY")
    print("-" * 80)

    best_speed = max(results, key=lambda r: r.realtime_factor)
    best_size = min(results, key=lambda r: r.model_size_mb)

    print(f"Fastest generation: {best_speed.quantization} ({best_speed.realtime_factor:.2f}x realtime)")
    print(f"Smallest model: {best_size.quantization} ({best_size.model_size_mb:.1f} MB)")

    if len(results) > 1:
        speedup = results[-1].realtime_factor / results[0].realtime_factor
        print(f"INT4 vs bfloat16 speedup: {speedup:.2f}x")


def save_results(results: List[BenchmarkResult], config: BenchmarkConfig, output_path: Path):
    """Save benchmark results to JSON."""
    output = {
        "timestamp": datetime.now().isoformat(),
        "config": asdict(config),
        "results": [asdict(r) for r in results],
    }

    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\nResults saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Benchmark MLX-Music generation")
    parser.add_argument("--prompt", type=str,
                       default="electronic dance music with synthesizers and drums",
                       help="Text prompt for generation")
    parser.add_argument("--steps", type=int, default=20,
                       help="Number of denoising steps")
    parser.add_argument("--duration", type=float, default=5.0,
                       help="Target audio duration in seconds")
    parser.add_argument("--cfg-scale", type=float, default=7.5,
                       help="Classifier-free guidance scale")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed for reproducibility")
    parser.add_argument("--iterations", type=int, default=3,
                       help="Number of timed iterations per quantization level")
    parser.add_argument("--warmup", type=int, default=1,
                       help="Number of warmup iterations")
    parser.add_argument("--model-path", type=str,
                       default="/Users/dustinpainter/Dev-Projects/audio-models/ACEStep-models/ACE-Step-v1-3.5B",
                       help="Path to ACE-Step model")
    parser.add_argument("--output", type=str, default="output/benchmark_results.json",
                       help="Output path for JSON results")
    args = parser.parse_args()

    model_path = Path(args.model_path)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    config = BenchmarkConfig(
        prompt=args.prompt,
        steps=args.steps,
        duration=args.duration,
        cfg_scale=args.cfg_scale,
        seed=args.seed,
        iterations=args.iterations,
        warmup_iterations=args.warmup,
    )

    print("=" * 80)
    print("MLX-MUSIC BENCHMARK")
    print("=" * 80)
    print(f"Model: ACE-Step v1 3.5B")
    print(f"Prompt: \"{config.prompt}\"")
    print(f"Steps: {config.steps}")
    print(f"Duration: {config.duration}s")
    print(f"CFG Scale: {config.cfg_scale}")
    print(f"Seed: {config.seed}")
    print(f"Iterations: {config.iterations} (+ {config.warmup_iterations} warmup)")

    dtype = mx.bfloat16

    # Load shared components (DCAE, Vocoder, Text Encoder)
    print("\nLoading shared components...")

    print("  Loading text encoder...")
    text_encoder = get_text_encoder(model_path, device="cpu", use_fp16=False)

    print("  Loading DCAE...")
    dcae = load_dcae(model_path, dtype)

    print("  Loading vocoder...")
    vocoder, sample_rate = load_vocoder(model_path, dtype)

    print(f"  Sample rate: {sample_rate} Hz")

    # Run benchmarks for each quantization level
    quantization_levels = [None, "int8", "int4"]  # None = bfloat16
    results = []

    for quant in quantization_levels:
        result = run_single_benchmark(
            model_path, config, quant,
            text_encoder, dcae, vocoder, sample_rate
        )
        results.append(result)

    # Print and save results
    print_results_table(results, config)
    save_results(results, config, output_path)


if __name__ == "__main__":
    main()
