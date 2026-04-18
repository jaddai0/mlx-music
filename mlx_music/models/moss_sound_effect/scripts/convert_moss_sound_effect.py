#!/usr/bin/env python3
"""Convert MOSS-SoundEffect PyTorch weights to MLX format.

Usage:
    python convert_moss_sound_effect.py \
        --src /path/to/MOSS-SoundEffect \
        --dst /path/to/output \
        [--quantize 4] [--group-size 64]
"""

from __future__ import annotations

import argparse
import gc
import glob
import json
import shutil
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

from mlx_music.models.moss_sound_effect.config import MossSoundEffectConfig
from mlx_music.models.moss_sound_effect.weight_mapping import map_weights, should_quantize


def load_safetensors(src_dir: Path) -> dict[str, mx.array]:
    """Load all safetensors shards from source directory using MLX (supports bfloat16)."""
    weight_files = sorted(glob.glob(str(src_dir / "model*.safetensors")))
    if not weight_files:
        raise FileNotFoundError(f"No model*.safetensors found in {src_dir}")

    print(f"Loading {len(weight_files)} weight shard(s)...")
    weights = {}
    for wf in weight_files:
        print(f"  {Path(wf).name}")
        shard = mx.load(wf)
        weights.update(shard)

    print(f"Loaded {len(weights)} weight tensors")
    return weights


def convert_weights(
    src_dir: Path,
    dst_dir: Path,
    quantize_bits: int | None = None,
    group_size: int = 64,
):
    """Convert MOSS-SoundEffect weights from PyTorch to MLX format."""
    dst_dir.mkdir(parents=True, exist_ok=True)

    # Load HF config
    hf_config_path = src_dir / "config.json"
    with open(hf_config_path) as f:
        hf_config = json.load(f)

    config = MossSoundEffectConfig.from_hf_config(hf_config)

    # Load weights (mx.load handles bfloat16 natively)
    mx_weights = load_safetensors(src_dir)

    # Apply key mapping
    mapped = map_weights(mx_weights)
    print(f"Mapped {len(mapped)} weights")

    # Verify key counts and detect actual n_vq_weights from checkpoint
    n_emb_ext = sum(1 for k in mapped if k.startswith("emb_ext."))
    n_lm_heads = sum(1 for k in mapped if k.startswith("lm_heads."))
    n_backbone = sum(1 for k in mapped if k.startswith("language_model."))
    print(f"  Backbone: {n_backbone}, emb_ext: {n_emb_ext}, lm_heads: {n_lm_heads}")

    # Update n_vq_weights to match actual checkpoint
    config.n_vq_weights = n_emb_ext
    print(f"  Set n_vq_weights={n_emb_ext} (from checkpoint)")

    if quantize_bits is not None:
        print(f"\nQuantizing to {quantize_bits}-bit (group_size={group_size})...")
        # Load model to quantize
        model = None
        try:
            from mlx_music.models.moss_sound_effect.model import MossSoundEffectModel
            model = MossSoundEffectModel(config)
            model.load_weights(list(mapped.items()), strict=False)

            def class_predicate(path: str, module: nn.Module) -> bool:
                if not isinstance(module, nn.Linear):
                    return False
                return should_quantize(path)

            nn.quantize(model, bits=quantize_bits, group_size=group_size,
                       class_predicate=class_predicate)

            # Extract quantized weights, filtering out Qwen3's unused lm_head
            import mlx.utils
            mapped = {
                k: v for k, v in mlx.utils.tree_flatten(model.parameters())
                if "language_model.lm_head" not in k
            }

            config.quantization = {
                "bits": quantize_bits,
                "group_size": group_size,
            }
            print(f"  Quantized model weights: {len(mapped)} tensors")
        except Exception as e:
            print(f"  Warning: Quantization failed ({e}), saving full precision")

    # Save weights using MLX native safetensors (handles bfloat16, quantized types)
    print(f"\nSaving to {dst_dir}...")

    # Ensure all values are mx.array
    for k, v in mapped.items():
        if not isinstance(v, mx.array):
            mapped[k] = mx.array(v)

    total_params = sum(v.size for v in mapped.values())
    total_bytes = sum(v.nbytes for v in mapped.values())
    print(f"  Total parameters: {total_params:,} ({total_bytes / 1e9:.2f} GB)")

    max_shard_bytes = 5 * 1024 * 1024 * 1024  # 5GB

    if total_bytes <= max_shard_bytes:
        mx.save_safetensors(str(dst_dir / "weights.safetensors"), mapped)
        print(f"  Saved single file ({total_bytes / 1e9:.1f} GB)")
    else:
        # Shard
        shard_idx = 0
        current_shard = {}
        current_bytes = 0
        for k, v in mapped.items():
            if current_bytes + v.nbytes > max_shard_bytes and current_shard:
                mx.save_safetensors(str(dst_dir / f"weights-{shard_idx:05d}.safetensors"), current_shard)
                shard_idx += 1
                current_shard = {}
                current_bytes = 0
            current_shard[k] = v
            current_bytes += v.nbytes
        if current_shard:
            mx.save_safetensors(str(dst_dir / f"weights-{shard_idx:05d}.safetensors"), current_shard)
        print(f"  Saved {shard_idx + 1} shards ({total_bytes / 1e9:.1f} GB total)")

    # Save config
    config_dict = config.to_dict()
    config_dict["model_type"] = "moss_sound_effect"
    with open(dst_dir / "config.json", "w") as f:
        json.dump(config_dict, f, indent=2)

    # Copy tokenizer files
    tokenizer_files = [
        "tokenizer.json", "tokenizer_config.json",
        "special_tokens_map.json", "vocab.json", "merges.txt",
        "chat_template.jinja", "added_tokens.json",
    ]
    for tf in tokenizer_files:
        src = src_dir / tf
        if src.exists():
            shutil.copy2(src, dst_dir / tf)
            print(f"  Copied {tf}")

    print("\nValidating converted model load...")
    from mlx_music.models.moss_sound_effect.model import MossSoundEffectModel

    validated = MossSoundEffectModel.from_pretrained(dst_dir)
    sample_ids = mx.concatenate(
        [
            mx.array([[[validated.config.im_start_token_id]]], dtype=mx.int32),
            mx.full((1, 1, validated.config.n_vq), validated.config.audio_pad_code, dtype=mx.int32),
        ],
        axis=2,
    )
    logits, _ = validated(sample_ids)
    mx.eval(logits[0])
    del validated
    gc.collect()
    mx.clear_cache()
    print("  Validation passed")

    print(f"\nConversion complete! Output: {dst_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Convert MOSS-SoundEffect PyTorch weights to MLX format"
    )
    parser.add_argument(
        "--src", type=Path, required=True,
        help="Source directory with PyTorch safetensors + config.json"
    )
    parser.add_argument(
        "--dst", type=Path, required=True,
        help="Output directory for MLX weights"
    )
    parser.add_argument(
        "--quantize", "-q", type=int, choices=[4, 8], default=None,
        help="Quantize to N-bit (4 or 8)"
    )
    parser.add_argument(
        "--group-size", type=int, default=64,
        help="Quantization group size (default: 64)"
    )
    args = parser.parse_args()

    convert_weights(args.src, args.dst, args.quantize, args.group_size)


if __name__ == "__main__":
    main()
