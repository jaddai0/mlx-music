#!/usr/bin/env python3
"""Benchmark and validate MOSS-SoundEffect MLX conversion on Apple Silicon."""

from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import mlx.core as mx
from transformers import AutoTokenizer

from mlx_music.models.moss_sound_effect.config import MossSoundEffectConfig
from mlx_music.models.moss_sound_effect.generation import generate
from mlx_music.models.moss_sound_effect.model import MossSoundEffectModel
from mlx_music.models.moss_sound_effect.pipeline import SoundEffectPipeline
from mlx_music.models.moss_sound_effect.processor import build_prompt


DEFAULT_MODEL_PATH = Path("/Users/dustinpainter/Dev-Projects/audio-models/mlx-moss-sound-effect")
DEFAULT_CODEC_PATH = Path("/Users/dustinpainter/Dev-Projects/audio-models/mlx-moss-tokenizer")
DEFAULT_OUTPUT_DIR = Path("/Users/dustinpainter/Dev-Projects/audio-models/output/moss-sound-effect-proof")


@dataclass
class BenchmarkResult:
    device: str
    elapsed_sec: float
    tokens_generated: int
    tok_per_sec: float
    output_shape: tuple[int, ...]


def reset_memory():
    gc.collect()
    mx.clear_cache()
    if hasattr(mx, "reset_peak_memory"):
        mx.reset_peak_memory()
    elif hasattr(mx, "metal") and hasattr(mx.metal, "reset_peak_memory"):
        mx.metal.reset_peak_memory()


def load_tokenizer(model_path: Path):
    return AutoTokenizer.from_pretrained(
        str(model_path),
        trust_remote_code=True,
    )


def run_generation_benchmark(
    device,
    model_path: Path,
    prompt: str,
    duration: float,
    max_new_tokens: int,
    warmup_tokens: int,
) -> BenchmarkResult:
    mx.set_default_device(device)
    reset_memory()

    tokenizer = load_tokenizer(model_path)
    config = MossSoundEffectConfig.from_json(model_path / "config.json")
    model = MossSoundEffectModel.from_pretrained(model_path)
    input_ids = build_prompt(tokenizer, config, prompt, duration_seconds=duration)

    warmup = generate(model, input_ids, config, max_new_tokens=warmup_tokens)
    mx.eval(warmup[0][1])

    start = time.perf_counter()
    output = generate(model, input_ids, config, max_new_tokens=max_new_tokens)
    mx.eval(output[0][1])
    elapsed = time.perf_counter() - start

    generation_ids = output[0][1]
    result = BenchmarkResult(
        device=str(device),
        elapsed_sec=elapsed,
        tokens_generated=int(generation_ids.shape[0]),
        tok_per_sec=float(generation_ids.shape[0] / elapsed),
        output_shape=tuple(int(dim) for dim in generation_ids.shape),
    )

    del model
    gc.collect()
    mx.clear_cache()
    return result


def run_gpu_waveform_proof(
    model_path: Path,
    codec_path: Path,
    prompt: str,
    duration: float,
    max_new_tokens: int,
    output_dir: Path,
) -> dict[str, object]:
    mx.set_default_device(mx.gpu)
    reset_memory()

    pipe = SoundEffectPipeline.from_pretrained(
        model_path=model_path,
        tokenizer_path=codec_path,
    )
    start = time.perf_counter()
    waveform = pipe.generate(
        prompt,
        duration=duration,
        max_new_tokens=max_new_tokens,
    )
    if waveform is None:
        raise RuntimeError("Pipeline returned no waveform")
    mx.eval(waveform)
    elapsed = time.perf_counter() - start

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "moss_sound_effect_gpu_proof.wav"
    pipe.save_wav(waveform, output_path)

    proof = {
        "device": str(mx.default_device()),
        "waveform_shape": tuple(int(dim) for dim in waveform.shape),
        "samples": int(waveform.shape[-1]),
        "duration_sec": float(waveform.shape[-1] / pipe.config.sampling_rate),
        "elapsed_sec": elapsed,
        "sample_rate": pipe.config.sampling_rate,
        "output_path": str(output_path),
    }

    del pipe
    gc.collect()
    mx.clear_cache()
    return proof


def main():
    parser = argparse.ArgumentParser(description="Benchmark and validate MOSS-SoundEffect MLX conversion")
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--codec-path", type=Path, default=DEFAULT_CODEC_PATH)
    parser.add_argument("--prompt", type=str, default="fresh snow crunching under footsteps")
    parser.add_argument("--duration", type=float, default=0.5, help="Prompt duration hint in seconds")
    parser.add_argument("--proof-steps", type=int, default=20, help="GPU proof generation steps")
    parser.add_argument("--benchmark-steps", type=int, default=4, help="Matched GPU/CPU benchmark steps")
    parser.add_argument("--warmup-steps", type=int, default=1, help="Warmup steps before timing")
    parser.add_argument("--skip-cpu", action="store_true", help="Skip CPU comparison benchmark")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    if not args.model_path.exists():
        raise FileNotFoundError(f"Model path not found: {args.model_path}")
    if not args.codec_path.exists():
        raise FileNotFoundError(f"Codec path not found: {args.codec_path}")

    gpu_result = run_generation_benchmark(
        device=mx.gpu,
        model_path=args.model_path,
        prompt=args.prompt,
        duration=args.duration,
        max_new_tokens=args.benchmark_steps,
        warmup_tokens=args.warmup_steps,
    )
    print(f"GPU benchmark: {gpu_result}")

    cpu_result = None
    if not args.skip_cpu:
        cpu_result = run_generation_benchmark(
            device=mx.cpu,
            model_path=args.model_path,
            prompt=args.prompt,
            duration=args.duration,
            max_new_tokens=args.benchmark_steps,
            warmup_tokens=args.warmup_steps,
        )
        print(f"CPU benchmark: {cpu_result}")

    proof = run_gpu_waveform_proof(
        model_path=args.model_path,
        codec_path=args.codec_path,
        prompt=args.prompt,
        duration=args.duration,
        max_new_tokens=args.proof_steps,
        output_dir=args.output_dir,
    )
    print(f"GPU waveform proof: {proof}")

    report = {
        "timestamp": datetime.now().isoformat(),
        "prompt": args.prompt,
        "duration_hint_sec": args.duration,
        "benchmark_steps": args.benchmark_steps,
        "proof_steps": args.proof_steps,
        "gpu_generation": asdict(gpu_result),
        "cpu_generation": asdict(cpu_result) if cpu_result is not None else None,
        "speedup_vs_cpu": (
            cpu_result.elapsed_sec / gpu_result.elapsed_sec if cpu_result is not None else None
        ),
        "gpu_waveform_proof": proof,
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    report_path = args.output_dir / "benchmark_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    print(f"Report saved to {report_path}")


if __name__ == "__main__":
    main()
