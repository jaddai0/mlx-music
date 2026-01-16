"""Command-line interface for mlx-music."""

import argparse
import logging
import sys
from pathlib import Path
from typing import Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
)
logger = logging.getLogger("mlx_music")


def detect_model_family(model_path: str) -> str:
    """
    Auto-detect model family from model path or HuggingFace repo ID.

    Args:
        model_path: Model path or HuggingFace repo ID

    Returns:
        Model family: "ace-step", "musicgen", or "stable-audio"
    """
    model_lower = model_path.lower()

    # Check for MusicGen patterns
    if "musicgen" in model_lower or "facebook/musicgen" in model_lower:
        return "musicgen"

    # Check for Stable Audio patterns
    if "stable-audio" in model_lower or "stabilityai/stable-audio" in model_lower:
        return "stable-audio"

    # Check for ACE-Step patterns
    if "ace-step" in model_lower or "ace_step" in model_lower:
        return "ace-step"

    # Default to ace-step (original behavior)
    return "ace-step"


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="MLX Music - Generate music on Apple Silicon",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate music with ACE-Step (default)
  mlx-music generate --prompt "upbeat electronic dance music" --duration 30

  # Generate with MusicGen
  mlx-music generate --engine musicgen --model facebook/musicgen-small \\
      --prompt "jazz piano improvisation" --duration 10

  # Generate with Stable Audio
  mlx-music generate --engine stable-audio --model stabilityai/stable-audio-open-1.0 \\
      --prompt "ambient electronic music" --duration 30

  # Generate with lyrics (ACE-Step only)
  mlx-music generate --prompt "pop ballad" --lyrics "Verse 1: ..." --duration 60

  # Auto-detect model family from model path
  mlx-music generate --model facebook/musicgen-melody --prompt "orchestral theme"
        """,
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Generate command
    gen_parser = subparsers.add_parser("generate", help="Generate music")
    gen_parser.add_argument(
        "--engine",
        type=str,
        choices=["ace-step", "musicgen", "stable-audio"],
        default=None,
        help="Model family to use (auto-detected from --model if not specified)",
    )
    gen_parser.add_argument(
        "--model",
        type=str,
        default="ACE-Step/ACE-Step-v1-3.5B",
        help="Model path or HuggingFace repo ID",
    )
    gen_parser.add_argument(
        "--prompt",
        type=str,
        required=True,
        help="Text description of desired music",
    )
    gen_parser.add_argument(
        "--negative-prompt",
        type=str,
        default=None,
        help="Negative prompt for CFG (Stable Audio only)",
    )
    gen_parser.add_argument(
        "--lyrics",
        type=str,
        default=None,
        help="Optional lyrics for vocal generation (ACE-Step only)",
    )
    gen_parser.add_argument(
        "--duration",
        type=float,
        default=30.0,
        help="Duration in seconds (default: 30)",
    )
    gen_parser.add_argument(
        "--steps",
        type=int,
        default=None,
        help="Number of inference steps (default: model-specific)",
    )
    gen_parser.add_argument(
        "--guidance",
        type=float,
        default=None,
        help="Guidance scale (default: model-specific)",
    )
    gen_parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility",
    )
    gen_parser.add_argument(
        "--output",
        "-o",
        type=str,
        default="output.wav",
        help="Output file path (default: output.wav)",
    )
    gen_parser.add_argument(
        "--scheduler",
        type=str,
        choices=["euler", "heun"],
        default="euler",
        help="Scheduler type for diffusion models (default: euler)",
    )
    gen_parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose logging",
    )

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(0)

    if args.command == "generate":
        # Set logging level
        if hasattr(args, "verbose") and args.verbose:
            logging.getLogger().setLevel(logging.DEBUG)

        generate_command(args)


def generate_command(args):
    """Handle generate command."""
    from tqdm import tqdm

    from mlx_music.utils.audio_io import save_audio

    # Auto-detect model family if not specified
    engine = args.engine or detect_model_family(args.model)

    logger.info(f"Using engine: {engine}")
    logger.info(f"Loading model: {args.model}")

    if engine == "ace-step":
        _generate_ace_step(args)
    elif engine == "musicgen":
        _generate_musicgen(args)
    elif engine == "stable-audio":
        _generate_stable_audio(args)
    else:
        logger.error(f"Unknown engine: {engine}")
        sys.exit(1)


def _generate_ace_step(args):
    """Generate music using ACE-Step."""
    from tqdm import tqdm

    from mlx_music import ACEStep
    from mlx_music.utils.audio_io import save_audio

    model = ACEStep.from_pretrained(args.model)

    logger.info(f"Generating {args.duration}s of music...")
    logger.info(f"  Prompt: {args.prompt}")
    if args.lyrics:
        logger.info(f"  Lyrics: {args.lyrics[:50]}...")

    # Default values for ACE-Step
    steps = args.steps or 60
    guidance = args.guidance or 15.0

    # Progress callback
    pbar = tqdm(total=steps, desc="Generating")

    def callback(step, timestep, latents):
        pbar.update(1)

    output = model.generate(
        prompt=args.prompt,
        lyrics=args.lyrics,
        duration=args.duration,
        num_inference_steps=steps,
        guidance_scale=guidance,
        seed=args.seed,
        scheduler_type=args.scheduler,
        callback=callback,
    )

    pbar.close()

    # Save output
    save_audio(output.audio, args.output, output.sample_rate)
    logger.info(f"Generated {output.duration:.1f}s of audio at {output.sample_rate}Hz")
    logger.info(f"Saved to: {args.output}")


def _generate_musicgen(args):
    """Generate music using MusicGen."""
    from tqdm import tqdm

    from mlx_music import MusicGen
    from mlx_music.utils.audio_io import save_audio

    model = MusicGen.from_pretrained(args.model)

    logger.info(f"Generating {args.duration}s of music...")
    logger.info(f"  Prompt: {args.prompt}")

    # Default values for MusicGen
    guidance = args.guidance or 3.0

    # Check if extended generation is needed
    if args.duration > 30.0:
        logger.info("Using extended generation for duration > 30s")
        # No progress callback for extended generation currently
        output = model.generate_extended(
            prompt=args.prompt,
            duration=args.duration,
            guidance_scale=guidance,
            seed=args.seed,
        )
    else:
        # Calculate steps based on duration and frame rate
        frame_rate = model.config.frame_rate
        total_steps = int(args.duration * frame_rate)

        pbar = tqdm(total=total_steps, desc="Generating")

        def callback(step, total, codes):
            pbar.n = step
            pbar.refresh()

        output = model.generate(
            prompt=args.prompt,
            duration=args.duration,
            guidance_scale=guidance,
            seed=args.seed,
            callback=callback,
        )
        pbar.close()

    # Save output
    save_audio(output.audio, args.output, output.sample_rate)
    logger.info(f"Generated {output.duration:.1f}s of audio at {output.sample_rate}Hz")
    logger.info(f"Saved to: {args.output}")


def _generate_stable_audio(args):
    """Generate music using Stable Audio."""
    from tqdm import tqdm

    from mlx_music import StableAudio
    from mlx_music.utils.audio_io import save_audio

    model = StableAudio.from_pretrained(args.model)

    logger.info(f"Generating {args.duration}s of music...")
    logger.info(f"  Prompt: {args.prompt}")
    if args.negative_prompt:
        logger.info(f"  Negative: {args.negative_prompt[:50]}...")

    # Default values for Stable Audio
    steps = args.steps or 100
    guidance = args.guidance or 7.0

    # Progress callback
    pbar = tqdm(total=steps, desc="Generating")

    def callback(step, timestep, latents):
        pbar.update(1)

    output = model.generate(
        prompt=args.prompt,
        negative_prompt=args.negative_prompt,
        duration=args.duration,
        num_inference_steps=steps,
        guidance_scale=guidance,
        seed=args.seed,
        callback=callback,
    )

    pbar.close()

    # Save output
    save_audio(output.audio, args.output, output.sample_rate)
    logger.info(f"Generated {output.duration:.1f}s of audio at {output.sample_rate}Hz")
    logger.info(f"Saved to: {args.output}")


if __name__ == "__main__":
    main()
