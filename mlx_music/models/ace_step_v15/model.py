"""
High-level ACE-Step v1.5 model API for MLX.

Usage:
    model = ACEStepV15.from_pretrained("path/to/models", variant="turbo")
    result = model.generate(
        prompt="Upbeat electronic dance music",
        lyrics="[Verse 1]\\nDancing through the night...",
        duration=30, seed=42
    )
    # result["audio"]: numpy array [samples, 2], result["sample_rate"]: 48000
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import mlx.core as mx
import mlx.nn as nn

from mlx_music.models.ace_step_v15.constants import (
    DEFAULT_DIT_INSTRUCTION,
    SFT_GEN_PROMPT,
)
from mlx_music.models.ace_step_v15.config import AceStepV15Config
from mlx_music.models.ace_step_v15.dit_model import AceStepConditionGenerationModel
from mlx_music.models.ace_step_v15.vae import AutoencoderOobleck
from mlx_music.models.ace_step_v15.weight_mapping import (
    load_dit_weights,
    load_silence_latent,
    load_vae_weights,
)

logger = logging.getLogger(__name__)

# Latent rate: 25 Hz (48000 / 1920)
LATENT_HZ = 25
VAE_HOP = 1920
SAMPLE_RATE = 48000


class ACEStepV15:
    """High-level API for ACE-Step v1.5 music generation."""

    def __init__(
        self,
        model: AceStepConditionGenerationModel,
        vae: AutoencoderOobleck,
        silence_latent: mx.array,
        config: AceStepV15Config,
        text_tokenizer: Optional[Any] = None,
        text_encoder: Optional[Any] = None,
    ):
        self.model = model
        self.vae = vae
        self.silence_latent = silence_latent
        self.config = config
        self.text_tokenizer = text_tokenizer
        self.text_encoder = text_encoder
        self._missing_text_encoder_warned = False

    def quantize(self, mode: str = "mixed") -> None:
        """Quantize the DiT model for reduced memory and faster inference.

        Args:
            mode: Quantization mode - "int4", "int8", or "mixed" (INT8 attn + INT4 FFN)
        """
        from mlx_music.weights.quantization import QuantizationConfig, quantize_model

        configs = {
            "int4": QuantizationConfig.for_speed,
            "int8": QuantizationConfig.for_quality,
            "mixed": QuantizationConfig.for_balanced,
        }
        if mode not in configs:
            raise ValueError(f"Unknown quantization mode: {mode}. Use: {list(configs)}")

        config = configs[mode]()
        quantize_model(self.model, config)
        logger.info(f"Quantized DiT model with mode={mode}")

    @classmethod
    def from_pretrained(
        cls,
        models_dir: Union[str, Path],
        variant: str = "turbo",
        vae_dir: Optional[Union[str, Path]] = None,
        text_encoder_dir: Optional[Union[str, Path]] = None,
        load_text_encoder: bool = True,
        quantization: Optional[str] = None,
    ) -> "ACEStepV15":
        """Load model from pretrained weights.

        Args:
            models_dir: Root models directory containing variant subdirs and vae/
            variant: Model variant (turbo, base, sft, turbo-shift1, turbo-shift3, turbo-continuous)
            vae_dir: Override VAE directory (default: models_dir/vae)
            text_encoder_dir: Optional text encoder path (default: models_dir/Qwen3-Embedding-0.6B)
            load_text_encoder: Load Qwen3 text encoder for prompt/lyric conditioning
            quantization: Optional quantization mode ("int4", "int8", "mixed")
        """
        models_dir = Path(models_dir)
        variant_dir = models_dir / f"acestep-v15-{variant}"
        if vae_dir is None:
            vae_dir = models_dir / "vae"
        else:
            vae_dir = Path(vae_dir)
        if text_encoder_dir is None:
            text_encoder_dir = models_dir / "Qwen3-Embedding-0.6B"
        else:
            text_encoder_dir = Path(text_encoder_dir)

        # Load config
        config_path = variant_dir / "config.json"
        if not config_path.exists():
            raise FileNotFoundError(f"Config not found: {config_path}")
        config = AceStepV15Config.from_json(config_path)
        logger.info(f"Loaded config: variant={config.model_version}, layers={config.num_hidden_layers}")

        # Build and load DiT model
        model = AceStepConditionGenerationModel(config)
        dit_weights = load_dit_weights(variant_dir)
        model.load_weights(list(dit_weights.items()))
        logger.info(f"Loaded DiT model: {len(dit_weights)} parameters")

        # Build and load VAE
        vae = AutoencoderOobleck()
        vae_weights = load_vae_weights(vae_dir)
        vae.load_weights(list(vae_weights.items()))
        logger.info(f"Loaded VAE: {len(vae_weights)} parameters")

        # Load silence latent
        silence_path = variant_dir / "silence_latent.pt"
        if silence_path.exists():
            silence_latent = load_silence_latent(silence_path)
            logger.info(f"Loaded silence_latent: {silence_latent.shape}")
        else:
            # Create a zeros fallback
            silence_latent = mx.zeros((1, LATENT_HZ * 240, 64))
            logger.warning("silence_latent.pt not found, using zeros")

        text_tokenizer = None
        text_encoder = None
        if load_text_encoder:
            text_tokenizer, text_encoder = cls._try_load_text_encoder(text_encoder_dir)

        instance = cls(
            model=model,
            vae=vae,
            silence_latent=silence_latent,
            config=config,
            text_tokenizer=text_tokenizer,
            text_encoder=text_encoder,
        )

        if quantization is not None:
            instance.quantize(quantization)

        return instance

    @staticmethod
    def _try_load_text_encoder(
        text_encoder_dir: Path,
    ) -> Tuple[Optional[Any], Optional[Any]]:
        """Load Qwen3 text encoder + tokenizer if available.

        Falls back to (None, None) when dependencies or weights are unavailable.
        """
        if not text_encoder_dir.exists():
            logger.warning(
                "Text encoder directory not found: %s. Falling back to zero conditioning.",
                text_encoder_dir,
            )
            return None, None

        try:
            from transformers import AutoModel, AutoTokenizer
        except ImportError:
            logger.warning(
                "transformers is not installed. Falling back to zero text conditioning."
            )
            return None, None

        try:
            tokenizer = AutoTokenizer.from_pretrained(
                str(text_encoder_dir),
                trust_remote_code=True,
            )
            if tokenizer.pad_token_id is None:
                if tokenizer.eos_token is not None:
                    tokenizer.pad_token = tokenizer.eos_token
                elif tokenizer.unk_token is not None:
                    tokenizer.pad_token = tokenizer.unk_token

            text_encoder = AutoModel.from_pretrained(
                str(text_encoder_dir),
                trust_remote_code=True,
            )
            text_encoder.eval()
            logger.info(f"Loaded text encoder: {text_encoder_dir}")
            return tokenizer, text_encoder
        except Exception as exc:
            logger.warning(
                "Failed to load text encoder from %s: %s. Falling back to zero conditioning.",
                text_encoder_dir,
                exc,
            )
            return None, None

    @staticmethod
    def _build_timestep_schedule(steps: int, shift: float) -> List[float]:
        """Build an explicit shifted timestep schedule for arbitrary NFE.

        t_shifted = shift * t / (1 + (shift - 1) * t), with t in (0, 1].
        """
        if steps <= 0:
            raise ValueError(f"steps must be positive, got {steps}")
        if shift <= 0:
            raise ValueError(f"shift must be positive, got {shift}")

        schedule: List[float] = []
        for i in range(steps):
            t = (steps - i) / steps
            shifted_t = (shift * t) / (1.0 + (shift - 1.0) * t)
            schedule.append(float(shifted_t))
        return schedule

    @staticmethod
    def _format_lyrics(lyrics: str, vocal_language: str) -> str:
        """Format lyric conditioning text exactly like the reference pipeline."""
        language = vocal_language.strip() if vocal_language else "unknown"
        return f"# Languages\n{language}\n\n# Lyric\n{lyrics}<|endoftext|>"

    @staticmethod
    def _torch_tensor_to_mx(tensor: Any, dtype: mx.Dtype) -> mx.array:
        """Convert a torch.Tensor to an MLX array."""
        import torch

        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"Expected torch.Tensor, got {type(tensor)}")

        cpu_tensor = tensor.detach().to("cpu")
        if cpu_tensor.dtype == torch.bfloat16:
            cpu_tensor = cpu_tensor.float()
        return mx.array(cpu_tensor.numpy(), dtype=dtype)

    def _build_text_conditioning_from_strings(
        self,
        prompt: str,
        lyrics: str,
        vocal_language: str,
        instruction: str,
        dtype: mx.Dtype,
    ) -> Optional[Tuple[mx.array, mx.array, mx.array, mx.array]]:
        """Encode prompt + lyrics into DiT conditioning embeddings.

        Returns:
            (text_hidden_states, text_attention_mask, lyric_hidden_states, lyric_attention_mask)
            or None if no text encoder is available.
        """
        if self.text_tokenizer is None or self.text_encoder is None:
            return None

        try:
            import torch
        except ImportError:
            if not self._missing_text_encoder_warned:
                logger.warning(
                    "torch is not installed. Falling back to zero text conditioning."
                )
                self._missing_text_encoder_warned = True
            return None

        instruction_text = instruction.strip() if instruction else DEFAULT_DIT_INSTRUCTION
        text_prompt = SFT_GEN_PROMPT.format(instruction_text, prompt, "{}")
        lyrics_text = self._format_lyrics(lyrics, vocal_language)

        try:
            text_inputs = self.text_tokenizer(
                text_prompt,
                padding="longest",
                truncation=True,
                max_length=256,
                return_tensors="pt",
            )
            lyric_inputs = self.text_tokenizer(
                lyrics_text,
                padding="longest",
                truncation=True,
                max_length=2048,
                return_tensors="pt",
            )

            with torch.no_grad():
                text_outputs = self.text_encoder(
                    input_ids=text_inputs.input_ids,
                    attention_mask=text_inputs.attention_mask,
                )
                if hasattr(text_outputs, "last_hidden_state"):
                    text_hidden_states_pt = text_outputs.last_hidden_state
                elif isinstance(text_outputs, tuple):
                    text_hidden_states_pt = text_outputs[0]
                else:
                    text_hidden_states_pt = text_outputs

                embed_tokens = self.text_encoder.get_input_embeddings()
                if embed_tokens is None:
                    raise RuntimeError("Text encoder does not expose input embeddings")
                lyric_hidden_states_pt = embed_tokens(lyric_inputs.input_ids)

            text_hidden_states = self._torch_tensor_to_mx(text_hidden_states_pt, dtype=dtype)
            text_attention_mask = self._torch_tensor_to_mx(
                text_inputs.attention_mask, dtype=dtype
            )
            lyric_hidden_states = self._torch_tensor_to_mx(lyric_hidden_states_pt, dtype=dtype)
            lyric_attention_mask = self._torch_tensor_to_mx(
                lyric_inputs.attention_mask, dtype=dtype
            )

            return (
                text_hidden_states,
                text_attention_mask,
                lyric_hidden_states,
                lyric_attention_mask,
            )
        except Exception as exc:
            if not self._missing_text_encoder_warned:
                logger.warning(
                    "Failed to build text conditioning (%s). Falling back to zeros.", exc
                )
                self._missing_text_encoder_warned = True
            return None

    def generate(
        self,
        prompt: str = "",
        lyrics: str = "",
        vocal_language: str = "unknown",
        instruction: str = DEFAULT_DIT_INSTRUCTION,
        duration: float = 30.0,
        seed: Optional[int] = None,
        shift: float = 3.0,
        steps: Optional[int] = None,
        text_hidden_states: Optional[mx.array] = None,
        text_attention_mask: Optional[mx.array] = None,
        lyric_hidden_states: Optional[mx.array] = None,
        lyric_attention_mask: Optional[mx.array] = None,
        refer_audio_packed: Optional[mx.array] = None,
        refer_audio_order_mask: Optional[mx.array] = None,
    ) -> Dict:
        """Generate music.

        Args:
            prompt: Text description of desired music
            lyrics: Song lyrics
            vocal_language: Vocal language tag for lyric formatting (default: "unknown")
            instruction: Instruction header used by SFT prompt format
            duration: Duration in seconds (max 240)
            seed: Random seed for reproducibility
            shift: Timestep shift (1.0, 2.0, or 3.0)
            steps: Number of inference steps (overrides shift default)
            text_hidden_states: Pre-computed text embeddings (B, T, 1024)
            text_attention_mask: Text attention mask (B, T)
            lyric_hidden_states: Pre-computed lyric embeddings (B, T, 1024)
            lyric_attention_mask: Lyric attention mask (B, T)
            refer_audio_packed: Reference audio features (N, T, 64)
            refer_audio_order_mask: Batch assignment for reference audio (N,)

        Returns:
            Dict with 'audio' (numpy), 'sample_rate', 'latents', 'time_costs'
        """
        duration = min(duration, 240.0)
        latent_length = int(duration * LATENT_HZ)

        dtype = mx.bfloat16
        inferred_batch = 1

        # Build text conditioning from prompt/lyrics when embeddings are not provided.
        needs_auto_conditioning = any(
            x is None
            for x in (
                text_hidden_states,
                text_attention_mask,
                lyric_hidden_states,
                lyric_attention_mask,
            )
        )
        if needs_auto_conditioning:
            conditioned = self._build_text_conditioning_from_strings(
                prompt=prompt,
                lyrics=lyrics,
                vocal_language=vocal_language,
                instruction=instruction,
                dtype=dtype,
            )
            if conditioned is not None:
                auto_text, auto_text_mask, auto_lyric, auto_lyric_mask = conditioned
                if text_hidden_states is None:
                    text_hidden_states = auto_text
                if text_attention_mask is None:
                    text_attention_mask = auto_text_mask
                if lyric_hidden_states is None:
                    lyric_hidden_states = auto_lyric
                if lyric_attention_mask is None:
                    lyric_attention_mask = auto_lyric_mask

        # Infer batch size from provided conditioning.
        for tensor in (
            text_hidden_states,
            text_attention_mask,
            lyric_hidden_states,
            lyric_attention_mask,
        ):
            if tensor is not None and tensor.ndim >= 1:
                inferred_batch = int(tensor.shape[0])
                break

        # Fallback to zero conditioning only if no embeddings are available.
        if text_hidden_states is None:
            if not self._missing_text_encoder_warned:
                logger.warning(
                    "No text conditioning available. "
                    "Provide text embeddings or install/load the Qwen3 text encoder."
                )
                self._missing_text_encoder_warned = True
            text_hidden_states = mx.zeros(
                (inferred_batch, 1, self.config.text_hidden_dim), dtype=dtype
            )
        else:
            text_hidden_states = text_hidden_states.astype(dtype)
        if text_attention_mask is None:
            text_attention_mask = mx.ones((inferred_batch, text_hidden_states.shape[1]), dtype=dtype)
        else:
            text_attention_mask = text_attention_mask.astype(dtype)

        if lyric_hidden_states is None:
            lyric_hidden_states = mx.zeros(
                (inferred_batch, 1, self.config.text_hidden_dim), dtype=dtype
            )
        else:
            lyric_hidden_states = lyric_hidden_states.astype(dtype)
        if lyric_attention_mask is None:
            lyric_attention_mask = mx.ones((inferred_batch, lyric_hidden_states.shape[1]), dtype=dtype)
        else:
            lyric_attention_mask = lyric_attention_mask.astype(dtype)

        B = int(text_hidden_states.shape[0])

        if refer_audio_packed is None:
            refer_audio_packed = mx.zeros(
                (B, self.config.timbre_fix_frame, self.config.timbre_hidden_dim), dtype=dtype
            )
            refer_audio_order_mask = mx.arange(B, dtype=mx.int32)
        else:
            refer_audio_packed = refer_audio_packed.astype(dtype)
            if refer_audio_order_mask is None:
                refer_audio_order_mask = mx.zeros((refer_audio_packed.shape[0],), dtype=mx.int32)

        # Prepare latent inputs
        src_latents = self.silence_latent[:, :latent_length, :].astype(dtype)
        if src_latents.shape[0] != B:
            src_latents = mx.broadcast_to(src_latents, (B, latent_length, 64))

        chunk_masks = mx.ones((B, latent_length, 64), dtype=dtype)
        is_covers = mx.zeros((B,), dtype=mx.int32)
        attention_mask = mx.ones((B, latent_length), dtype=dtype)
        timesteps = None
        if steps is not None:
            timesteps = self._build_timestep_schedule(steps=steps, shift=shift)

        # Generate latents
        result = self.model.generate_audio(
            text_hidden_states=text_hidden_states,
            text_attention_mask=text_attention_mask,
            lyric_hidden_states=lyric_hidden_states,
            lyric_attention_mask=lyric_attention_mask,
            refer_audio_packed=refer_audio_packed,
            refer_audio_order_mask=refer_audio_order_mask,
            src_latents=src_latents,
            chunk_masks=chunk_masks,
            is_covers=is_covers,
            silence_latent=self.silence_latent[:, :latent_length, :].astype(dtype),
            attention_mask=attention_mask,
            seed=seed,
            fix_nfe=steps or 8,
            shift=shift,
            timesteps=timesteps,
        )

        target_latents = result["target_latents"]

        # VAE decode
        audio = self.vae.decode(target_latents.astype(mx.float32))
        mx.eval(audio)

        # Convert to numpy
        import numpy as np
        audio_np = np.array(audio[0])  # (samples, 2)

        return {
            "audio": audio_np,
            "sample_rate": SAMPLE_RATE,
            "latents": target_latents,
            "time_costs": result["time_costs"],
        }


__all__ = ["ACEStepV15"]
