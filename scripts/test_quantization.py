#!/usr/bin/env python3
"""
Test script for MLX-native quantization of ACE-Step models.

Tests:
1. Quantize transformer model with INT4
2. Measure memory reduction
3. Verify model still produces output
"""

import sys
from pathlib import Path
import time

# Add mlx_music to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import mlx.core as mx

from mlx_music.models.ace_step.transformer import ACEStepTransformer, ACEStepConfig
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
    QuantizationMode,
)


def load_transformer(model_path: Path, dtype: mx.Dtype = mx.bfloat16):
    """Load the ACE-Step transformer with weights."""
    print("Loading transformer...")
    config = ACEStepConfig()
    model = ACEStepTransformer(config)

    # Load weights
    weight_file = model_path / "ace_step_transformer" / "diffusion_pytorch_model.safetensors"
    weights = load_safetensors(weight_file, dtype=dtype)

    # Apply transformations
    all_mappings = ACE_STEP_TRANSFORMER_MAPPINGS + generate_transformer_block_mappings(config.num_layers)
    weights = convert_torch_to_mlx(weights, all_mappings, strict=False)

    # Load into model
    model.load_weights(list(weights.items()), strict=False)

    return model, config


def test_model_forward(model, config, dtype=mx.bfloat16):
    """Run a quick forward pass to verify model works."""
    batch_size = 1
    latent_channels = 8
    latent_height = 16
    latent_width = 64

    # Create dummy inputs
    latent = mx.random.normal(shape=(batch_size, latent_channels, latent_height, latent_width))
    latent = latent.astype(dtype)

    text_embeds = mx.zeros((batch_size, 10, 768), dtype=dtype)
    text_mask = mx.ones((batch_size, 10))
    speaker_embeds = mx.zeros((batch_size, 512), dtype=dtype)
    timestep = mx.array([0.5])

    # Forward pass
    start = time.time()
    output = model(
        hidden_states=latent,
        timestep=timestep,
        encoder_text_hidden_states=text_embeds,
        text_attention_mask=text_mask,
        speaker_embeds=speaker_embeds,
    )
    mx.eval(output)
    elapsed = time.time() - start

    return output, elapsed


def main():
    model_path = Path("/Users/dustinpainter/Dev-Projects/audio-models/ACEStep-models/ACE-Step-v1-3.5B")

    print("=" * 60)
    print("ACE-Step Quantization Test")
    print("=" * 60)

    # Load model in bfloat16 first
    dtype = mx.bfloat16
    model, config = load_transformer(model_path, dtype)

    # Measure original size
    num_params, size_mb = get_model_size(model)
    print(f"\nOriginal model:")
    print(f"  Parameters: {num_params:,}")
    print(f"  Size: {size_mb:.1f} MB")

    # Test forward pass before quantization
    print("\nTesting forward pass (bfloat16)...")
    output_bf16, time_bf16 = test_model_forward(model, config, dtype)
    print(f"  Forward pass: {time_bf16:.3f}s")
    print(f"  Output mean: {mx.mean(output_bf16).item():.6f}")

    # Quantize to INT4
    print("\n" + "-" * 60)
    print("Quantizing to INT4...")
    print("-" * 60)

    quant_config = QuantizationConfig.for_speed()  # INT4
    quantize_model(model, quant_config)

    # Measure quantized size
    num_params_q, size_mb_q = get_model_size(model)
    print(f"\nQuantized model (INT4):")
    print(f"  Parameters: {num_params_q:,}")
    print(f"  Size: {size_mb_q:.1f} MB")
    print(f"  Reduction: {(1 - size_mb_q / size_mb) * 100:.1f}%")

    # Test forward pass after quantization
    print("\nTesting forward pass (INT4 quantized)...")
    output_q, time_q = test_model_forward(model, config, dtype)
    print(f"  Forward pass: {time_q:.3f}s")
    print(f"  Output mean: {mx.mean(output_q).item():.6f}")

    # Compare outputs
    diff = mx.abs(output_bf16 - output_q)
    print(f"\nOutput difference (bf16 vs INT4):")
    print(f"  Mean abs diff: {mx.mean(diff).item():.6f}")
    print(f"  Max abs diff: {mx.max(diff).item():.6f}")

    # Test INT8 quantization
    print("\n" + "-" * 60)
    print("Testing INT8 quantization...")
    print("-" * 60)

    # Reload model for INT8 test
    model_int8, _ = load_transformer(model_path, dtype)

    quant_config_int8 = QuantizationConfig.for_quality()  # INT8
    quantize_model(model_int8, quant_config_int8)

    num_params_8, size_mb_8 = get_model_size(model_int8)
    print(f"\nQuantized model (INT8):")
    print(f"  Size: {size_mb_8:.1f} MB")
    print(f"  Reduction: {(1 - size_mb_8 / size_mb) * 100:.1f}%")

    output_int8, time_int8 = test_model_forward(model_int8, config, dtype)
    print(f"  Forward pass: {time_int8:.3f}s")

    diff_int8 = mx.abs(output_bf16 - output_int8)
    print(f"  Mean abs diff from bf16: {mx.mean(diff_int8).item():.6f}")

    # Summary
    print("\n" + "=" * 60)
    print("QUANTIZATION SUMMARY")
    print("=" * 60)
    print(f"{'Mode':<12} {'Size (MB)':<12} {'Reduction':<12} {'Time (s)':<12}")
    print("-" * 48)
    print(f"{'bfloat16':<12} {size_mb:<12.1f} {'-':<12} {time_bf16:<12.3f}")
    print(f"{'INT8':<12} {size_mb_8:<12.1f} {f'{(1-size_mb_8/size_mb)*100:.0f}%':<12} {time_int8:<12.3f}")
    print(f"{'INT4':<12} {size_mb_q:<12.1f} {f'{(1-size_mb_q/size_mb)*100:.0f}%':<12} {time_q:<12.3f}")
    print("\nQuantization test PASSED!")


if __name__ == "__main__":
    main()
