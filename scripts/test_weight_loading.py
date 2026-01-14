#!/usr/bin/env python3
"""
Test script to verify ACE-Step transformer weight loading.

This script:
1. Creates the MLX model
2. Lists all model parameter names
3. Loads the checkpoint weights
4. Compares model params vs checkpoint keys
5. Attempts to load weights and reports any issues
"""

import sys
from pathlib import Path

# Add mlx_music to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import mlx.core as mx
import mlx.nn as nn
from safetensors import safe_open

from mlx_music.models.ace_step.transformer import ACEStepTransformer, ACEStepConfig
from mlx_music.weights.weight_loader import (
    load_safetensors,
    convert_torch_to_mlx,
    ACE_STEP_TRANSFORMER_MAPPINGS,
    generate_transformer_block_mappings,
    transpose_conv1d,
    transpose_conv2d,
)


def get_model_param_names(model: nn.Module, prefix: str = "") -> list[str]:
    """Recursively get all parameter names from a model."""
    names = []
    for name, value in model.parameters().items():
        full_name = f"{prefix}.{name}" if prefix else name
        if isinstance(value, dict):
            # Nested module
            for k, v in value.items():
                names.append(f"{full_name}.{k}")
        else:
            names.append(full_name)
    return names


def main():
    model_path = Path("/Users/dustinpainter/Dev-Projects/audio-models/ACEStep-models/ACE-Step-v1-3.5B")
    transformer_path = model_path / "ace_step_transformer" / "diffusion_pytorch_model.safetensors"

    print("=" * 60)
    print("ACE-Step Transformer Weight Loading Test")
    print("=" * 60)

    # 1. Create model
    print("\n1. Creating MLX model...")
    config = ACEStepConfig()
    model = ACEStepTransformer(config)
    print(f"   Model created with {config.num_layers} transformer blocks")

    # 2. Get model parameter names
    print("\n2. Collecting model parameter names...")
    model_params = {}
    for name, param in model.parameters().items():
        model_params[name] = param.shape if hasattr(param, 'shape') else type(param)

    # Flatten nested params
    flat_model_params = {}
    def flatten_params(d, prefix=""):
        for k, v in d.items():
            full_key = f"{prefix}.{k}" if prefix else k
            if isinstance(v, dict):
                flatten_params(v, full_key)
            elif isinstance(v, mx.array):
                flat_model_params[full_key] = v.shape
            else:
                flat_model_params[full_key] = type(v)

    flatten_params(model.parameters())
    print(f"   Found {len(flat_model_params)} model parameters")

    # Print first 20 model params
    print("\n   Sample model parameters:")
    for i, (name, shape) in enumerate(sorted(flat_model_params.items())[:20]):
        print(f"      {name}: {shape}")
    print("      ...")

    # 3. Load checkpoint keys
    print("\n3. Loading checkpoint keys...")
    with safe_open(str(transformer_path), framework="numpy") as f:
        checkpoint_keys = set(f.keys())
    print(f"   Found {len(checkpoint_keys)} checkpoint keys")

    # 4. Compare keys
    print("\n4. Comparing model params vs checkpoint keys...")

    # Keys that need transformation (Conv2d/Conv1d)
    transform_keys = set()
    all_mappings = ACE_STEP_TRANSFORMER_MAPPINGS + generate_transformer_block_mappings(config.num_layers)
    for m in all_mappings:
        if m.transform is not None:
            transform_keys.add(m.torch_key)

    # Find matches, missing, and extra
    model_param_set = set(flat_model_params.keys())

    matched = model_param_set & checkpoint_keys
    in_model_not_checkpoint = model_param_set - checkpoint_keys
    in_checkpoint_not_model = checkpoint_keys - model_param_set

    print(f"\n   Matched keys: {len(matched)}")
    print(f"   In model but NOT in checkpoint: {len(in_model_not_checkpoint)}")
    print(f"   In checkpoint but NOT in model: {len(in_checkpoint_not_model)}")

    # Show mismatches
    if in_model_not_checkpoint:
        print("\n   Keys in MODEL but not in checkpoint (may be expected for non-learnable params):")
        for k in sorted(in_model_not_checkpoint)[:30]:
            print(f"      {k}")
        if len(in_model_not_checkpoint) > 30:
            print(f"      ... and {len(in_model_not_checkpoint) - 30} more")

    if in_checkpoint_not_model:
        print("\n   Keys in CHECKPOINT but not in model (need to add to model):")
        for k in sorted(in_checkpoint_not_model)[:30]:
            print(f"      {k}")
        if len(in_checkpoint_not_model) > 30:
            print(f"      ... and {len(in_checkpoint_not_model) - 30} more")

    # 5. Try loading weights
    print("\n5. Attempting to load weights...")
    try:
        # Load raw weights
        weights = load_safetensors(transformer_path, dtype=mx.bfloat16)
        print(f"   Loaded {len(weights)} tensors from checkpoint")

        # Apply transformations
        weights = convert_torch_to_mlx(weights, all_mappings, strict=False)
        print(f"   Applied transformations to conv weights")

        # Try loading into model (strict=False to allow missing/extra keys)
        model.load_weights(list(weights.items()), strict=False)
        print("   Successfully loaded weights into model (with strict=False)!")

        # Verify a few weights
        print("\n   Verifying sample weights...")
        sample_keys = [
            "t_block.1.weight",
            "timestep_embedder.linear_1.weight",
            "transformer_blocks.0.attn.to_q.weight",
            "transformer_blocks.0.cross_attn.to_q.weight",
            "transformer_blocks.0.ff.inverted_conv.conv.weight",
            "proj_in.early_conv_layers.0.weight",
            "final_layer.scale_shift_table",
        ]
        for key in sample_keys:
            if key in weights:
                print(f"      {key}: shape={weights[key].shape}, dtype={weights[key].dtype}")

    except Exception as e:
        print(f"   Error loading weights: {e}")
        import traceback
        traceback.print_exc()

    # 6. Test forward pass
    print("\n6. Testing forward pass...")
    try:
        # Create dummy inputs
        batch_size = 1
        latent_height = 16  # n_mels / 8 = 128 / 8 = 16
        latent_width = 64   # time / 8

        hidden_states = mx.zeros((batch_size, config.in_channels, latent_height, latent_width))
        timestep = mx.array([0.5])
        text_embeds = mx.zeros((batch_size, 10, config.text_embedding_dim))
        text_mask = mx.ones((batch_size, 10))
        speaker_embeds = mx.zeros((batch_size, config.speaker_embedding_dim))

        # Forward pass
        output = model(
            hidden_states=hidden_states,
            timestep=timestep,
            encoder_text_hidden_states=text_embeds,
            text_attention_mask=text_mask,
            speaker_embeds=speaker_embeds,
        )
        mx.eval(output)
        print(f"   Output shape: {output.shape}")
        print("   Forward pass successful!")

    except Exception as e:
        print(f"   Error in forward pass: {e}")
        import traceback
        traceback.print_exc()

    print("\n" + "=" * 60)
    print("Test complete!")


if __name__ == "__main__":
    main()
