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

# CLI Constants
CLI_MAX_DURATION = 600.0  # Maximum duration to prevent resource exhaustion (10 minutes)
LOG_PROMPT_MAX_LENGTH = 50  # Maximum prompt length to show in logs

# Model-specific default values
ACE_STEP_DEFAULT_STEPS = 60
ACE_STEP_DEFAULT_GUIDANCE = 15.0
MUSICGEN_DEFAULT_GUIDANCE = 3.0
STABLE_AUDIO_DEFAULT_STEPS = 100
STABLE_AUDIO_DEFAULT_GUIDANCE = 7.0


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


def validate_args(args) -> None:
    """
    Validate CLI arguments before processing.

    Args:
        args: Parsed command-line arguments

    Raises:
        SystemExit: If validation fails
    """
    # Validate duration
    if args.duration <= 0:
        logger.error(f"Duration must be positive, got {args.duration}")
        sys.exit(1)

    if args.duration > CLI_MAX_DURATION:
        logger.error(f"Duration must be <= {CLI_MAX_DURATION}s, got {args.duration}")
        sys.exit(1)

    # Validate steps if provided
    if args.steps is not None and args.steps <= 0:
        logger.error(f"Steps must be positive, got {args.steps}")
        sys.exit(1)

    # Validate guidance if provided
    if args.guidance is not None and args.guidance <= 0:
        logger.error(f"Guidance scale must be positive, got {args.guidance}")
        sys.exit(1)


def ensure_output_directory(output_path: str) -> None:
    """
    Ensure the output directory exists.

    Args:
        output_path: Path to the output file
    """
    output_dir = Path(output_path).parent
    if output_dir and not output_dir.exists():
        logger.info(f"Creating output directory: {output_dir}")
        output_dir.mkdir(parents=True, exist_ok=True)


def main() -> None:
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
        if args.verbose:
            logging.getLogger().setLevel(logging.DEBUG)

        generate_command(args)


def generate_command(args) -> None:
    """Handle generate command."""
    from mlx_music.utils.audio_io import save_audio

    # Validate arguments
    validate_args(args)

    # Auto-detect model family if not specified
    engine = args.engine or detect_model_family(args.model)

    # Warn about incompatible parameter combinations
    if args.lyrics and engine != "ace-step":
        logger.warning(f"--lyrics is only supported for ACE-Step. Ignoring for {engine}.")

    if args.negative_prompt and engine != "stable-audio":
        logger.warning(f"--negative-prompt is only supported for Stable Audio. Ignoring for {engine}.")

    # Ensure output directory exists
    ensure_output_directory(args.output)

    logger.info(f"Using engine: {engine}")
    logger.info(f"Loading model: {args.model}")

    try:
        if engine == "ace-step":
            _generate_ace_step(args)
        elif engine == "musicgen":
            _generate_musicgen(args)
        elif engine == "stable-audio":
            _generate_stable_audio(args)
        else:
            logger.error(f"Unknown engine: {engine}")
            sys.exit(1)
    except KeyboardInterrupt:
        logger.info("\nGeneration cancelled by user.")
        sys.exit(130)
    except Exception as e:
        logger.error(f"Generation failed: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


def _generate_ace_step(args) -> None:
    """Generate music using ACE-Step."""
    from tqdm import tqdm

    from mlx_music import ACEStep
    from mlx_music.utils.audio_io import save_audio

    try:
        model = ACEStep.from_pretrained(args.model)
    except Exception as e:
        raise RuntimeError(f"Failed to load ACE-Step model from '{args.model}': {e}") from e

    logger.info(f"Generating {args.duration}s of music...")
    logger.info(f"  Prompt: {args.prompt}")
    if args.lyrics:
        logger.info(f"  Lyrics: {args.lyrics[:LOG_PROMPT_MAX_LENGTH]}...")

    # Default values for ACE-Step
    steps = args.steps or ACE_STEP_DEFAULT_STEPS
    guidance = args.guidance or ACE_STEP_DEFAULT_GUIDANCE

    # Progress callback with try/finally for cleanup
    pbar = tqdm(total=steps, desc="Generating")

    def callback(step, timestep, latents):
        pbar.update(1)

    try:
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
    finally:
        pbar.close()

    # Save output
    save_audio(output.audio, args.output, output.sample_rate)
    logger.info(f"Generated {output.duration:.1f}s of audio at {output.sample_rate}Hz")
    logger.info(f"Saved to: {args.output}")


def _generate_musicgen(args) -> None:
    """Generate music using MusicGen."""
    from tqdm import tqdm

    from mlx_music import MusicGen
    from mlx_music.utils.audio_io import save_audio

    try:
        model = MusicGen.from_pretrained(args.model)
    except Exception as e:
        raise RuntimeError(f"Failed to load MusicGen model from '{args.model}': {e}") from e

    logger.info(f"Generating {args.duration}s of music...")
    logger.info(f"  Prompt: {args.prompt}")

    # Default values for MusicGen
    guidance = args.guidance or MUSICGEN_DEFAULT_GUIDANCE

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
        if frame_rate <= 0:
            raise ValueError(f"Invalid frame_rate in model config: {frame_rate}")
        total_steps = int(args.duration * frame_rate)

        pbar = tqdm(total=total_steps, desc="Generating")

        def callback(step, total, codes):
            pbar.n = step
            pbar.refresh()

        try:
            output = model.generate(
                prompt=args.prompt,
                duration=args.duration,
                guidance_scale=guidance,
                seed=args.seed,
                callback=callback,
            )
        finally:
            pbar.close()

    # Save output
    save_audio(output.audio, args.output, output.sample_rate)
    logger.info(f"Generated {output.duration:.1f}s of audio at {output.sample_rate}Hz")
    logger.info(f"Saved to: {args.output}")


def _generate_stable_audio(args) -> None:
    """Generate music using Stable Audio."""
    from tqdm import tqdm

    from mlx_music import StableAudio
    from mlx_music.utils.audio_io import save_audio

    try:
        model = StableAudio.from_pretrained(args.model)
    except Exception as e:
        raise RuntimeError(f"Failed to load Stable Audio model from '{args.model}': {e}") from e

    logger.info(f"Generating {args.duration}s of music...")
    logger.info(f"  Prompt: {args.prompt}")
    if args.negative_prompt:
        logger.info(f"  Negative: {args.negative_prompt[:LOG_PROMPT_MAX_LENGTH]}...")

    # Default values for Stable Audio
    steps = args.steps or STABLE_AUDIO_DEFAULT_STEPS
    guidance = args.guidance or STABLE_AUDIO_DEFAULT_GUIDANCE

    # Progress callback with try/finally for cleanup
    pbar = tqdm(total=steps, desc="Generating")

    def callback(step, timestep, latents):
        pbar.update(1)

    try:
        output = model.generate(
            prompt=args.prompt,
            negative_prompt=args.negative_prompt,
            duration=args.duration,
            num_inference_steps=steps,
            guidance_scale=guidance,
            seed=args.seed,
            callback=callback,
        )
    finally:
        pbar.close()

    # Save output
    save_audio(output.audio, args.output, output.sample_rate)
    logger.info(f"Generated {output.duration:.1f}s of audio at {output.sample_rate}Hz")
    logger.info(f"Saved to: {args.output}")


if __name__ == "__main__":
    main()
