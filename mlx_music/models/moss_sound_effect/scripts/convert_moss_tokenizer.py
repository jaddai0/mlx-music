#!/usr/bin/env python3
"""Convert MOSS Audio Tokenizer PyTorch weights to MLX format.

Usage:
    python convert_moss_tokenizer.py \
        --src /path/to/MOSS-Audio-Tokenizer \
        --dst /path/to/output

Or download from HuggingFace:
    python convert_moss_tokenizer.py \
        --hf-repo OpenMOSS-Team/MOSS-Audio-Tokenizer \
        --dst /path/to/output
"""

from __future__ import annotations

import argparse
import gc
import glob
import json
from pathlib import Path

import mlx.core as mx

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

from mlx_music.codecs.moss_audio_tokenizer.config import MossAudioTokenizerConfig
from mlx_music.codecs.moss_audio_tokenizer.model import MossAudioTokenizerModel


def load_weights(src_dir: Path) -> dict[str, mx.array]:
    """Load safetensors weights using MLX (supports bfloat16)."""
    weight_files = sorted(glob.glob(str(src_dir / "*.safetensors")))
    if not weight_files:
        raise FileNotFoundError(f"No .safetensors found in {src_dir}")

    print(f"Loading {len(weight_files)} weight file(s)...")
    weights = {}
    for wf in weight_files:
        print(f"  {Path(wf).name}")
        weights.update(mx.load(wf))

    return weights


def sanitize_tokenizer_weights(weights: dict[str, mx.array]) -> dict[str, mx.array]:
    """Map PyTorch checkpoint keys and shapes to the MLX tokenizer layout."""
    return MossAudioTokenizerModel.sanitize(weights)


def convert_tokenizer(
    src_dir: Path,
    dst_dir: Path,
    dtype_str: str = "float32",
):
    """Convert MOSS Audio Tokenizer weights."""
    dst_dir.mkdir(parents=True, exist_ok=True)

    # Load config
    config_path = src_dir / "config.json"
    if config_path.exists():
        with open(config_path) as f:
            config_dict = json.load(f)
        config = MossAudioTokenizerConfig.from_dict(config_dict)
    else:
        print("No config.json found, using defaults")
        config = MossAudioTokenizerConfig()

    # Load and sanitize weights
    mx_weights = load_weights(src_dir)
    print(f"Loaded {len(mx_weights)} raw weights")

    sanitized = sanitize_tokenizer_weights(mx_weights)
    print(f"Sanitized to {len(sanitized)} weights")

    # Print some key stats
    total_params = sum(v.size for v in sanitized.values())
    total_bytes = sum(v.nbytes for v in sanitized.values())
    print(f"Total parameters: {total_params:,} ({total_bytes / 1e9:.2f} GB)")

    # Cast dtype
    dtype_map = {
        "float32": mx.float32,
        "float16": mx.float16,
        "bfloat16": mx.bfloat16,
    }
    target_dtype = dtype_map.get(dtype_str, mx.float32)

    cast_weights = {}
    for k, v in sanitized.items():
        if v.dtype in (mx.float32, mx.float16, mx.bfloat16):
            cast_weights[k] = v.astype(target_dtype)
        else:
            cast_weights[k] = v

    print("Validating converted tokenizer weights...")
    model = MossAudioTokenizerModel(config)
    model.load_weights(list(cast_weights.items()))
    del model
    gc.collect()
    mx.clear_cache()
    print("  Validation passed")

    # Save weights
    output_path = dst_dir / "tokenizer_weights.safetensors"
    mx.save_safetensors(str(output_path), cast_weights)
    print(f"Saved weights to {output_path} ({output_path.stat().st_size / 1e9:.2f} GB)")

    # Save config
    config_out = config.to_dict()
    with open(dst_dir / "tokenizer_config.json", "w") as f:
        json.dump(config_out, f, indent=2)
    print(f"Saved config to {dst_dir / 'tokenizer_config.json'}")

    print("\nTokenizer conversion complete!")


def download_from_hf(repo_id: str, cache_dir: Path | None = None) -> Path:
    """Download tokenizer from HuggingFace Hub."""
    from huggingface_hub import snapshot_download
    path = snapshot_download(repo_id, cache_dir=str(cache_dir) if cache_dir else None)
    return Path(path)


def main():
    parser = argparse.ArgumentParser(
        description="Convert MOSS Audio Tokenizer weights to MLX format"
    )
    parser.add_argument(
        "--src", type=Path, default=None,
        help="Source directory with PyTorch weights"
    )
    parser.add_argument(
        "--hf-repo", type=str, default=None,
        help="HuggingFace repo ID to download from"
    )
    parser.add_argument(
        "--dst", type=Path, required=True,
        help="Output directory for MLX weights"
    )
    parser.add_argument(
        "--dtype", choices=["float32", "float16", "bfloat16"], default="float32",
        help="Target dtype (default: float32)"
    )
    parser.add_argument(
        "--cache-dir", type=Path, default=None,
        help="Cache directory for HF downloads"
    )
    args = parser.parse_args()

    if args.src is None and args.hf_repo is None:
        parser.error("Either --src or --hf-repo must be provided")

    if args.src is not None:
        src_dir = args.src
    else:
        src_dir = download_from_hf(args.hf_repo, args.cache_dir)

    convert_tokenizer(src_dir, args.dst, args.dtype)


if __name__ == "__main__":
    main()
