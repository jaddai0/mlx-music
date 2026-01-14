#!/usr/bin/env python3
"""
ACE-Step Music Generation CLI.

Generate music from text prompts using the MLX-native ACE-Step model.

Usage:
    python scripts/generate.py --prompt "upbeat electronic dance music"
    python scripts/generate.py --prompt "calm piano melody" --steps 50 --quantize int4
    python scripts/generate.py --prompt "jazz piano" --cfg-scale 7.5
"""

import argparse
import sys
import time
from pathlib import Path

# Add mlx_music to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import mlx.core as mx
import numpy as np

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
    QuantizationMode,
)


def load_transformer(model_path: Path, dtype: mx.Dtype = mx.bfloat16, quantize: str = None):
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

    # Optionally quantize
    if quantize:
        print(f"  Quantizing to {quantize.upper()}...")
        if quantize == "int4":
            quant_config = QuantizationConfig.for_speed()
        elif quantize == "int8":
            quant_config = QuantizationConfig.for_quality()
        else:  # mixed
            quant_config = QuantizationConfig.for_balanced()
        quantize_model(model, quant_config)

    num_params, size_mb = get_model_size(model)
    print(f"  Loaded {num_params:,} parameters ({size_mb:.1f} MB)")

    return model, config


def load_dcae(model_path: Path, dtype: mx.Dtype = mx.bfloat16):
    """Load the DCAE encoder/decoder."""
    print("Loading DCAE...")
    dcae_path = model_path / "music_dcae_f8c8"

    # Load config
    import json
    config_file = dcae_path / "config.json"
    if config_file.exists():
        with open(config_file) as f:
            config_dict = json.load(f)
        config = DCAEConfig.from_dict(config_dict)
    else:
        config = DCAEConfig()

    model = DCAE(config)

    # Load weights
    weight_file = dcae_path / "diffusion_pytorch_model.safetensors"
    if weight_file.exists():
        model = DCAE.from_pretrained(dcae_path, dtype=dtype)
        print(f"  Loaded DCAE (latent_channels={config.latent_channels})")
    else:
        print(f"  Warning: DCAE weights not found")

    return model, config


def load_vocoder(model_path: Path, dtype: mx.Dtype = mx.bfloat16):
    """Load the HiFi-GAN vocoder."""
    print("Loading vocoder...")
    vocoder_path = model_path / "music_vocoder"

    # Load config
    import json
    config_file = vocoder_path / "config.json"
    if config_file.exists():
        with open(config_file) as f:
            config_dict = json.load(f)
        config = VocoderConfig.from_dict(config_dict)
    else:
        config = VocoderConfig()

    model = HiFiGANVocoder.from_pretrained(vocoder_path, dtype=dtype)
    print(f"  Loaded vocoder (sample_rate={config.sampling_rate})")

    return model, config


def denoise(
    transformer,
    latent,
    text_embeds,
    text_mask,
    speaker_embeds,
    null_text_embeds=None,
    null_text_mask=None,
    num_steps=50,
    cfg_scale=7.5,
):
    """
    Run denoising diffusion with classifier-free guidance.

    Args:
        transformer: ACE-Step transformer model
        latent: Starting noise latent
        text_embeds: Text embeddings from encoder
        text_mask: Attention mask for text
        speaker_embeds: Speaker embeddings
        null_text_embeds: Null embeddings for CFG (optional)
        null_text_mask: Null mask for CFG (optional)
        num_steps: Number of denoising steps
        cfg_scale: Classifier-free guidance scale (1.0 = no guidance)
    """
    batch_size = latent.shape[0]
    dtype = latent.dtype
    use_cfg = cfg_scale > 1.0 and null_text_embeds is not None

    # Timesteps from 1.0 to 0.0
    timesteps = mx.linspace(1.0, 0.0, num_steps + 1)[:-1]

    for i, t in enumerate(timesteps):
        t_batch = mx.full((batch_size,), t.item())

        if use_cfg:
            # Classifier-free guidance: run both conditional and unconditional
            # Concatenate inputs for batched inference
            latent_input = mx.concatenate([latent, latent], axis=0)
            t_input = mx.concatenate([t_batch, t_batch], axis=0)
            text_input = mx.concatenate([text_embeds, null_text_embeds], axis=0)
            mask_input = mx.concatenate([text_mask, null_text_mask], axis=0)
            speaker_input = mx.concatenate([speaker_embeds, speaker_embeds], axis=0)

            # Get predictions
            v_pred_both = transformer(
                hidden_states=latent_input,
                timestep=t_input,
                encoder_text_hidden_states=text_input,
                text_attention_mask=mask_input,
                speaker_embeds=speaker_input,
            )
            mx.eval(v_pred_both)

            # Split and apply CFG
            v_pred_cond, v_pred_uncond = mx.split(v_pred_both, 2, axis=0)
            v_pred = v_pred_uncond + cfg_scale * (v_pred_cond - v_pred_uncond)
        else:
            # No CFG, just conditional prediction
            v_pred = transformer(
                hidden_states=latent,
                timestep=t_batch,
                encoder_text_hidden_states=text_embeds,
                text_attention_mask=text_mask,
                speaker_embeds=speaker_embeds,
            )
            mx.eval(v_pred)

        # Euler step
        dt = 1.0 / num_steps
        latent = latent - dt * v_pred

        if (i + 1) % 10 == 0 or i == 0:
            print(f"  Step {i+1}/{num_steps}")

    return latent


def decode_to_audio(latent, dcae, vocoder):
    """Decode latent to audio using DCAE and vocoder."""
    # DCAE: latent -> mel
    mel = dcae.decode(latent)
    mel = dcae.denormalize_mel(mel)

    # Vocoder: mel -> audio (process stereo channels)
    audio_ch1 = vocoder.decode(mel[:, 0:1, :, :].squeeze(1))
    audio_ch2 = vocoder.decode(mel[:, 1:2, :, :].squeeze(1))
    audio = mx.concatenate([audio_ch1, audio_ch2], axis=1)

    return audio


def save_audio(audio, sample_rate, output_path):
    """Save audio to WAV file."""
    import wave

    # Convert to numpy
    audio_np = np.array(audio.squeeze(0))  # Remove batch dim

    # Clip and convert to int16
    audio_np = np.clip(audio_np, -1.0, 1.0)
    audio_int16 = (audio_np * 32767).astype(np.int16)

    # Handle stereo
    if audio_int16.ndim == 2:
        audio_int16 = audio_int16.T
        n_channels = 2
    else:
        n_channels = 1

    # Write WAV
    with wave.open(str(output_path), 'wb') as wav:
        wav.setnchannels(n_channels)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(audio_int16.tobytes())

    duration = len(audio_int16) / sample_rate
    print(f"Saved {duration:.2f}s audio to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Generate music with ACE-Step")
    parser.add_argument("--prompt", type=str, default="electronic dance music with heavy bass",
                       help="Text prompt for music generation")
    parser.add_argument("--steps", type=int, default=50,
                       help="Number of denoising steps (default: 50)")
    parser.add_argument("--output", type=str, default="output/generated.wav",
                       help="Output audio file path")
    parser.add_argument("--model-path", type=str,
                       default="/Users/dustinpainter/Dev-Projects/audio-models/ACEStep-models/ACE-Step-v1-3.5B",
                       help="Path to ACE-Step model directory")
    parser.add_argument("--quantize", type=str, choices=["int4", "int8", "mixed"],
                       help="Quantization mode for transformer")
    parser.add_argument("--duration", type=float, default=3.0,
                       help="Target duration in seconds (approximate)")
    parser.add_argument("--cfg-scale", type=float, default=7.5,
                       help="Classifier-free guidance scale (1.0 = no guidance)")
    parser.add_argument("--seed", type=int, default=None,
                       help="Random seed for reproducibility")
    parser.add_argument("--no-text-encoder", action="store_true",
                       help="Skip text encoder (for testing without transformers)")
    args = parser.parse_args()

    model_path = Path(args.model_path)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    dtype = mx.bfloat16

    # Set random seed if provided
    if args.seed is not None:
        mx.random.seed(args.seed)

    print("=" * 60)
    print("ACE-Step Music Generation")
    print("=" * 60)
    print(f"Prompt: {args.prompt}")
    print(f"Steps: {args.steps}")
    print(f"CFG Scale: {args.cfg_scale}")
    print(f"Quantization: {args.quantize or 'none'}")
    if args.seed is not None:
        print(f"Seed: {args.seed}")
    print()

    # Load models
    start_load = time.time()
    transformer, trans_config = load_transformer(model_path, dtype, args.quantize)
    dcae, dcae_config = load_dcae(model_path, dtype)
    vocoder, voc_config = load_vocoder(model_path, dtype)

    # Load text encoder
    text_encoder = None
    if not args.no_text_encoder:
        print("Loading text encoder...")
        text_encoder = get_text_encoder(model_path, device="cpu", use_fp16=False)
        print(f"  Text encoder ready")

    load_time = time.time() - start_load
    print(f"\nModels loaded in {load_time:.2f}s")

    # Generate latent
    print("\n" + "-" * 60)
    print("Generating...")
    print("-" * 60)

    batch_size = 1
    latent_channels = 8
    latent_height = 16
    # Approximate: 64 latent frames ≈ 0.75s audio
    latent_width = int(64 * args.duration / 0.75)

    # Start from random noise
    latent = mx.random.normal(shape=(batch_size, latent_channels, latent_height, latent_width))
    latent = latent.astype(dtype)

    # Text conditioning
    if text_encoder is not None:
        print(f"  Encoding prompt: \"{args.prompt}\"")
        text_embeds, text_mask = text_encoder.encode(args.prompt)
        text_embeds = text_embeds.astype(dtype)
        text_mask = text_mask.astype(mx.float32)

        # Null embeddings for CFG
        null_text_embeds, null_text_mask = text_encoder.encode_null(
            batch_size=batch_size,
            seq_length=text_embeds.shape[1],
        )
        null_text_embeds = null_text_embeds.astype(dtype)
        null_text_mask = null_text_mask.astype(mx.float32)
    else:
        # Placeholder embeddings (for testing)
        print("  Using placeholder embeddings (no text encoder)")
        text_embeds = mx.zeros((batch_size, 64, 768), dtype=dtype)
        text_mask = mx.ones((batch_size, 64))
        null_text_embeds = mx.zeros((batch_size, 64, 768), dtype=dtype)
        null_text_mask = mx.zeros((batch_size, 64))

    # Speaker embeddings (not used in base model)
    speaker_embeds = mx.zeros((batch_size, 512), dtype=dtype)

    # Run denoising with CFG
    start_denoise = time.time()
    latent = denoise(
        transformer,
        latent,
        text_embeds,
        text_mask,
        speaker_embeds,
        null_text_embeds=null_text_embeds,
        null_text_mask=null_text_mask,
        num_steps=args.steps,
        cfg_scale=args.cfg_scale,
    )
    denoise_time = time.time() - start_denoise
    print(f"  Denoising completed in {denoise_time:.2f}s ({denoise_time/args.steps:.3f}s/step)")

    # Decode to audio
    print("\nDecoding to audio...")
    start_decode = time.time()
    audio = decode_to_audio(latent, dcae, vocoder)
    mx.eval(audio)
    decode_time = time.time() - start_decode
    print(f"  Decoding completed in {decode_time:.2f}s")

    # Save audio
    print()
    save_audio(audio, voc_config.sampling_rate, output_path)

    # Summary
    print("\n" + "=" * 60)
    print("Generation Summary")
    print("=" * 60)
    print(f"  Model loading: {load_time:.2f}s")
    print(f"  Denoising: {denoise_time:.2f}s")
    print(f"  Decoding: {decode_time:.2f}s")
    print(f"  Total: {load_time + denoise_time + decode_time:.2f}s")


if __name__ == "__main__":
    main()
