"""
LM Planner for ACE-Step v1.5 using mlx-lm.

Uses a Qwen3 5Hz language model to generate:
1. Music metadata (BPM, key, duration, etc.) via Chain-of-Thought
2. Audio codes for the DiT model

Supports three LM sizes: 0.6B, 1.7B, 4B.
"""

import logging
import re
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import mlx.core as mx

from mlx_music.models.ace_step_v15.constrained_decoding import (
    AudioCodeProcessor,
    MetadataFSMProcessor,
    extract_audio_code_indices,
    parse_lm_output,
)
from mlx_music.models.ace_step_v15.constants import (
    DEFAULT_LM_INSTRUCTION,
    DEFAULT_LM_UNDERSTAND_INSTRUCTION,
)

logger = logging.getLogger(__name__)

# Audio code rate: 5 codes per second
CODES_PER_SECOND = 5

# Default generation parameters
DEFAULT_TEMPERATURE = 0.85
DEFAULT_MAX_METADATA_TOKENS = 500
DEFAULT_CODES_SAFETY_MARGIN = 200


class LMPlanner:
    """Qwen3 5Hz Language Model planner for music generation.

    Generates structured music metadata and audio codes using constrained
    decoding with mlx-lm.

    Usage:
        planner = LMPlanner.from_pretrained("path/to/acestep-5Hz-lm-0.6B")
        result = planner.generate(
            caption="Upbeat electronic dance music",
            lyrics="[Verse 1]\\nDancing through the night...",
            target_duration=30.0,
        )
        # result["metadata"]: {"bpm": "128", "keyscale": "C major", ...}
        # result["audio_codes"]: [42, 1337, ...]
    """

    def __init__(self, model, tokenizer, model_size: str = "0.6B"):
        self.model = model
        self.tokenizer = tokenizer
        self.model_size = model_size

        # Audio code token range
        vocab = tokenizer.get_vocab()
        self.audio_code_ids = set()
        for token_str, token_id in vocab.items():
            if token_str.startswith("<|audio_code_"):
                self.audio_code_ids.add(token_id)
        logger.info(f"Found {len(self.audio_code_ids)} audio code tokens")

    @classmethod
    def from_pretrained(
        cls,
        model_path: Union[str, Path],
        model_size: str = "0.6B",
    ) -> "LMPlanner":
        """Load LM planner from pretrained weights.

        Args:
            model_path: Path to the 5Hz LM model directory
            model_size: Model size identifier (0.6B, 1.7B, 4B)
        """
        from mlx_lm import load

        model_path = str(model_path)
        logger.info(f"Loading LM planner from {model_path}")
        model, tokenizer = load(model_path)
        logger.info(f"Loaded {model_size} LM planner")
        return cls(model, tokenizer, model_size)

    def _build_prompt(
        self,
        caption: str,
        lyrics: str = "",
        instruction: str = DEFAULT_LM_INSTRUCTION,
    ) -> str:
        """Build chat template prompt for CoT generation.

        Format:
            system: instruction
            user: caption + lyrics
            assistant: (generation starts here)
        """
        messages = [
            {"role": "system", "content": f"# Instruction\n{instruction}\n\n"},
            {
                "role": "user",
                "content": f"# Caption\n{caption}\n\n# Lyric\n{lyrics}\n",
            },
        ]
        return self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

    def _build_prompt_with_cot(
        self,
        caption: str,
        lyrics: str,
        metadata: Dict[str, str],
        instruction: str = DEFAULT_LM_INSTRUCTION,
    ) -> str:
        """Build prompt with pre-generated CoT for codes-only generation.

        The assistant message contains the metadata, and generation
        continues from there to produce audio codes.
        """
        cot_text = self._format_metadata_as_cot(metadata)

        messages = [
            {"role": "system", "content": f"# Instruction\n{instruction}\n\n"},
            {
                "role": "user",
                "content": f"# Caption\n{caption}\n\n# Lyric\n{lyrics}\n",
            },
        ]
        prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        # Append the pre-generated CoT as start of assistant response
        prompt += cot_text
        return prompt

    def _format_metadata_as_cot(self, metadata: Dict[str, str]) -> str:
        """Format metadata dict as YAML inside <think> tags."""
        lines = ["<think>"]
        field_order = [
            "bpm",
            "caption",
            "duration",
            "genres",
            "keyscale",
            "language",
            "timesignature",
        ]
        for field in field_order:
            if field in metadata:
                lines.append(f"{field}: {metadata[field]}")
        lines.append("</think>")
        return "\n".join(lines)

    def generate(
        self,
        caption: str = "",
        lyrics: str = "",
        target_duration: Optional[float] = None,
        temperature: float = DEFAULT_TEMPERATURE,
        top_p: Optional[float] = None,
        repetition_penalty: float = 1.0,
        user_metadata: Optional[Dict[str, str]] = None,
        use_constrained_decoding: bool = True,
        metadata_only: bool = False,
        seed: Optional[int] = None,
    ) -> Dict:
        """Generate music metadata and audio codes.

        Args:
            caption: Text description of desired music
            lyrics: Song lyrics (optional)
            target_duration: Target audio duration in seconds
            temperature: Sampling temperature
            top_p: Nucleus sampling threshold
            repetition_penalty: Repetition penalty
            user_metadata: Pre-set metadata values (skip generation for these)
            use_constrained_decoding: Use FSM-based constrained decoding
            metadata_only: Stop after generating metadata (no audio codes)
            seed: Random seed

        Returns:
            Dict with:
                - "metadata": Dict of metadata key-value pairs
                - "audio_codes": List of integer audio code indices
                - "raw_text": Raw LM output text
                - "time_costs": Timing information
                - "success": Whether generation completed successfully
        """
        from mlx_lm import generate as mlx_generate
        from mlx_lm import stream_generate

        time_costs = {}
        user_metadata = user_metadata or {}

        if seed is not None:
            mx.random.seed(seed)

        # Phase 1: Generate metadata (CoT)
        t0 = time.time()
        metadata, metadata_text = self._generate_metadata(
            caption=caption,
            lyrics=lyrics,
            temperature=temperature,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            user_metadata=user_metadata,
            use_constrained_decoding=use_constrained_decoding,
        )
        t1 = time.time()
        time_costs["metadata_time"] = t1 - t0
        logger.info(f"Metadata generated in {t1-t0:.2f}s: {metadata}")

        if metadata_only:
            return {
                "metadata": metadata,
                "audio_codes": [],
                "raw_text": metadata_text,
                "time_costs": time_costs,
                "success": True,
            }

        # Apply user overrides to metadata
        metadata.update(user_metadata)

        # Determine target duration from metadata or parameter
        duration = target_duration
        if duration is None and "duration" in metadata:
            try:
                duration = float(metadata["duration"])
            except (ValueError, TypeError):
                duration = 30.0

        # Phase 2: Generate audio codes
        t2 = time.time()
        audio_codes_str = self._generate_codes(
            caption=caption,
            lyrics=lyrics,
            metadata=metadata,
            target_duration=duration,
            temperature=temperature,
            repetition_penalty=repetition_penalty,
        )
        t3 = time.time()
        time_costs["codes_time"] = t3 - t2

        # Parse audio codes
        audio_codes = extract_audio_code_indices(audio_codes_str)
        time_costs["total_time"] = t3 - t0
        logger.info(
            f"Generated {len(audio_codes)} audio codes in {t3-t2:.2f}s "
            f"(total: {t3-t0:.2f}s)"
        )

        return {
            "metadata": metadata,
            "audio_codes": audio_codes,
            "raw_text": metadata_text + audio_codes_str,
            "time_costs": time_costs,
            "success": len(audio_codes) > 0,
        }

    def _generate_metadata(
        self,
        caption: str,
        lyrics: str,
        temperature: float,
        top_p: Optional[float],
        repetition_penalty: float,
        user_metadata: Dict[str, str],
        use_constrained_decoding: bool,
    ) -> Tuple[Dict[str, str], str]:
        """Phase 1: Generate metadata via CoT."""
        from mlx_lm import stream_generate
        from mlx_lm.sample_utils import make_sampler, make_logits_processors

        prompt = self._build_prompt(caption, lyrics)

        # Build sampler and logits processors using mlx-lm utilities
        sampler = make_sampler(temp=temperature, top_p=top_p or 0.0)
        logits_processors = make_logits_processors(
            repetition_penalty=repetition_penalty if repetition_penalty != 1.0 else None,
        )

        if use_constrained_decoding:
            processor = MetadataFSMProcessor(
                self.tokenizer,
                skip_genres=True,
                skip_caption=False,
                skip_language=False,
                user_metadata=user_metadata,
            )
            logits_processors.append(processor)

        # Generate with streaming to detect </think>
        text = ""
        for response in stream_generate(
            self.model,
            self.tokenizer,
            prompt,
            max_tokens=DEFAULT_MAX_METADATA_TOKENS,
            sampler=sampler,
            logits_processors=logits_processors if logits_processors else None,
        ):
            text += response.text
            # Stop when we see </think>
            if "</think>" in text:
                break

        # Parse metadata from output
        metadata, _ = parse_lm_output(text)
        return metadata, text

    def _generate_codes(
        self,
        caption: str,
        lyrics: str,
        metadata: Dict[str, str],
        target_duration: Optional[float],
        temperature: float,
        repetition_penalty: float,
    ) -> str:
        """Phase 2: Generate audio codes with pre-generated metadata."""
        from mlx_lm import stream_generate
        from mlx_lm.sample_utils import make_sampler, make_logits_processors

        prompt = self._build_prompt_with_cot(caption, lyrics, metadata)

        # Compute max tokens for target duration
        if target_duration:
            max_codes = int(target_duration * CODES_PER_SECOND)
            max_tokens = max_codes + DEFAULT_CODES_SAFETY_MARGIN
        else:
            max_tokens = 1024

        # Build sampler and logits processors
        sampler = make_sampler(temp=temperature)
        logits_processors = make_logits_processors(
            repetition_penalty=repetition_penalty if repetition_penalty != 1.0 else None,
        )

        # Audio code processor
        processor = AudioCodeProcessor(
            self.tokenizer, target_duration=target_duration
        )
        logits_processors.append(processor)

        text = ""
        for response in stream_generate(
            self.model,
            self.tokenizer,
            prompt,
            max_tokens=max_tokens,
            sampler=sampler,
            logits_processors=logits_processors,
        ):
            text += response.text

        return text

    def generate_understanding(
        self,
        audio_codes: List[int],
        caption: str = "",
    ) -> Dict[str, str]:
        """Reverse task: understand audio codes and generate metadata."""
        from mlx_lm import generate as mlx_generate

        # Build audio codes string
        codes_str = "".join(f"<|audio_code_{c}|>" for c in audio_codes)

        messages = [
            {
                "role": "system",
                "content": f"# Instruction\n{DEFAULT_LM_UNDERSTAND_INSTRUCTION}\n\n",
            },
            {
                "role": "user",
                "content": f"# Caption\n{caption}\n\n# Audio Codes\n{codes_str}\n",
            },
        ]
        prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        text = mlx_generate(
            self.model,
            self.tokenizer,
            prompt,
            max_tokens=DEFAULT_MAX_METADATA_TOKENS,
        )

        metadata, _ = parse_lm_output(text)
        return metadata


__all__ = ["LMPlanner"]
