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

# --- CLI Constants ---
# All configurable limits and defaults are grouped here.

# General limits
CLI_MAX_DURATION = 600.0  # Maximum duration to prevent resource exhaustion (10 minutes)
LOG_PROMPT_MAX_LENGTH = 50  # Maximum prompt length to show in logs

# ACE-Step v1 defaults
# DPM++ at 20 steps achieves similar quality to Euler at 60 steps (3x speedup)
ACE_STEP_DEFAULT_STEPS = 20  # Optimized for DPM++ scheduler
ACE_STEP_DEFAULT_SCHEDULER = "dpm++"
ACE_STEP_DEFAULT_GUIDANCE = 15.0

# ACE-Step v1.5 defaults and validation
ACE_STEP_V15_DEFAULT_STEPS = 8  # Turbo variant
ACE_STEP_V15_DEFAULT_SHIFT = 3.0
VALID_V15_VARIANTS = {"turbo", "base", "sft", "turbo-shift1", "turbo-shift3", "continuous"}
V15_SHIFT_MIN = 0.0
V15_SHIFT_MAX = 20.0

# MusicGen defaults
MUSICGEN_DEFAULT_GUIDANCE = 3.0

# Stable Audio defaults
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

    # Check for ACE-Step v1.5 patterns (check before v1 since v1.5 contains "ace-step")
    if "v15" in model_lower or "v1.5" in model_lower or "ace-step-1.5" in model_lower:
        return "ace-step-v15"

    # Check for ACE-Step v1 patterns
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

    # Validate v1.5-specific parameters
    variant = getattr(args, "variant", None)
    if variant is not None and variant not in VALID_V15_VARIANTS:
        logger.error(
            f"Invalid variant '{variant}'. "
            f"Valid variants: {', '.join(sorted(VALID_V15_VARIANTS))}"
        )
        sys.exit(1)

    shift = getattr(args, "shift", None)
    if shift is not None and not (V15_SHIFT_MIN <= shift <= V15_SHIFT_MAX):
        logger.error(
            f"Shift must be between {V15_SHIFT_MIN} and {V15_SHIFT_MAX}, got {shift}"
        )
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
        choices=["ace-step", "ace-step-v15", "musicgen", "stable-audio"],
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
        "--variant",
        type=str,
        default="turbo",
        help="Model variant for ACE-Step v1.5 (turbo, base, sft, turbo-shift1, turbo-shift3)",
    )
    gen_parser.add_argument(
        "--shift",
        type=float,
        default=None,
        help="Timestep shift for ACE-Step v1.5 (default: variant-specific)",
    )
    gen_parser.add_argument(
        "--quantization",
        type=str,
        choices=["int4", "int8", "mixed"],
        default=None,
        help="Quantization mode for ACE-Step v1.5 (int4, int8, mixed)",
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
        choices=["euler", "heun", "dpm++"],
        default=ACE_STEP_DEFAULT_SCHEDULER,
        help="Scheduler type for diffusion models (default: dpm++). "
             "DPM++ achieves similar quality to Euler with fewer steps (20 vs 60).",
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
    if args.lyrics and engine not in ("ace-step", "ace-step-v15"):
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
        elif engine == "ace-step-v15":
            _generate_ace_step_v15(args)
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
    except (ValueError, TypeError) as e:
        logger.error(f"Invalid input: {e}")
        sys.exit(1)
    except (FileNotFoundError, OSError) as e:
        logger.error(f"File error: {e}")
        sys.exit(1)
    except RuntimeError as e:
        logger.error(f"Generation failed: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)
    except ImportError as e:
        logger.error(f"Missing dependency: {e}")
        sys.exit(1)


def _generate_ace_step(args) -> None:
    """Generate music using ACE-Step."""
    from tqdm import tqdm

    from mlx_music import ACEStep
    from mlx_music.utils.audio_io import save_audio

    model_display = Path(args.model).name if "/" in args.model else args.model
    try:
        model = ACEStep.from_pretrained(args.model)
    except (FileNotFoundError, OSError) as e:
        raise RuntimeError(f"Failed to load ACE-Step model '{model_display}': model not found") from e
    except (ValueError, TypeError) as e:
        raise RuntimeError(f"Failed to load ACE-Step model '{model_display}': {e}") from e

    logger.info(f"Generating {args.duration}s of music...")
    logger.info(f"  Prompt: {args.prompt}")
    if args.lyrics:
        logger.info(f"  Lyrics: {args.lyrics[:LOG_PROMPT_MAX_LENGTH]}...")

    # Default values for ACE-Step
    # If using default scheduler (dpm++), use optimized step count
    # If user explicitly chose euler/heun, use higher step count for quality
    if args.steps is not None:
        steps = args.steps
    elif args.scheduler == "dpm++":
        steps = ACE_STEP_DEFAULT_STEPS  # 20 steps optimized for DPM++
    else:
        steps = 60  # 60 steps for Euler/Heun
    guidance = args.guidance or ACE_STEP_DEFAULT_GUIDANCE

    logger.info(f"  Scheduler: {args.scheduler} ({steps} steps)")

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


def _generate_ace_step_v15(args) -> None:
    """Generate music using ACE-Step v1.5."""
    from mlx_music import ACEStepV15
    from mlx_music.utils.audio_io import save_audio

    variant = getattr(args, "variant", "turbo")
    quantization = getattr(args, "quantization", None)

    # Sanitize model path for error messages (show basename only)
    model_display = Path(args.model).name if "/" in args.model else args.model

    try:
        model = ACEStepV15.from_pretrained(
            args.model,
            variant=variant,
            quantization=quantization,
        )
    except (FileNotFoundError, OSError) as e:
        raise RuntimeError(f"Failed to load ACE-Step v1.5 model '{model_display}': model not found") from e
    except (ValueError, TypeError) as e:
        raise RuntimeError(f"Failed to load ACE-Step v1.5 model '{model_display}': {e}") from e

    logger.info(f"Generating {args.duration}s of music (v1.5 {variant})...")
    logger.info(f"  Prompt: {args.prompt}")
    if args.lyrics:
        logger.info(f"  Lyrics: {args.lyrics[:LOG_PROMPT_MAX_LENGTH]}...")

    steps = args.steps or ACE_STEP_V15_DEFAULT_STEPS
    shift = args.shift if args.shift is not None else ACE_STEP_V15_DEFAULT_SHIFT

    logger.info(f"  Steps: {steps}, Shift: {shift}")

    result = model.generate(
        prompt=args.prompt,
        lyrics=args.lyrics or "",
        duration=args.duration,
        seed=args.seed,
        shift=shift,
        steps=steps,
    )

    # Save output
    save_audio(result["audio"], args.output, result["sample_rate"])
    logger.info(f"Generated audio at {result['sample_rate']}Hz")
    logger.info(f"Saved to: {args.output}")


def _generate_musicgen(args) -> None:
    """Generate music using MusicGen."""
    from tqdm import tqdm

    from mlx_music import MusicGen
    from mlx_music.utils.audio_io import save_audio

    model_display = Path(args.model).name if "/" in args.model else args.model
    try:
        model = MusicGen.from_pretrained(args.model)
    except (FileNotFoundError, OSError) as e:
        raise RuntimeError(f"Failed to load MusicGen model '{model_display}': model not found") from e
    except (ValueError, TypeError) as e:
        raise RuntimeError(f"Failed to load MusicGen model '{model_display}': {e}") from e

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

    model_display = Path(args.model).name if "/" in args.model else args.model
    try:
        model = StableAudio.from_pretrained(args.model)
    except (FileNotFoundError, OSError) as e:
        raise RuntimeError(f"Failed to load Stable Audio model '{model_display}': model not found") from e
    except (ValueError, TypeError) as e:
        raise RuntimeError(f"Failed to load Stable Audio model '{model_display}': {e}") from e

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
