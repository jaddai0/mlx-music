"""MOSS-SoundEffect 8B model (MLX).

Uses Qwen3 backbone from mlx_lm with additional audio embedding layers
and multi-head prediction (1 text head + n_vq audio heads).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import mlx.core as mx
import mlx.nn as nn
from mlx_lm.models.cache import make_prompt_cache

from .config import MossSoundEffectConfig


class MossSoundEffectModel(nn.Module):
    """MOSS-SoundEffect: Qwen3 backbone + audio VQ embeddings + multi-head output.

    Architecture:
        - Qwen3 backbone (8B, 36 layers, GQA)
        - emb_ext: n_vq Embedding layers for audio codebook channels
        - lm_heads: 1 text head (hidden->vocab) + n_vq audio heads (hidden->1025)

    Input format: (B, S, 1 + n_vq) where dim 0 is text tokens, dims 1..n_vq are audio codes.
    """

    def __init__(self, config: MossSoundEffectConfig):
        super().__init__()
        self.config = config

        # Qwen3 backbone - import from mlx_lm
        from mlx_lm.models.qwen3 import ModelArgs, Model as Qwen3Model

        backbone_args = ModelArgs(
            model_type="qwen3",
            hidden_size=config.hidden_size,
            num_hidden_layers=config.num_hidden_layers,
            num_attention_heads=config.num_attention_heads,
            num_key_value_heads=config.num_key_value_heads,
            intermediate_size=config.intermediate_size,
            vocab_size=config.vocab_size,
            rope_theta=config.rope_theta,
            max_position_embeddings=config.max_position_embeddings,
            rms_norm_eps=config.rms_norm_eps,
            head_dim=config.head_dim,
            tie_word_embeddings=False,
        )
        self.language_model = Qwen3Model(backbone_args)

        # Audio VQ embeddings: create n_vq_weights to match checkpoint,
        # but only n_vq are used at runtime
        self.emb_ext = [
            nn.Embedding(config.audio_vocab_size + 1, config.hidden_size)
            for _ in range(config.n_vq_weights)
        ]

        # Multi-head prediction:
        # Head 0: text head (hidden -> vocab_size)
        # Heads 1..n_vq_weights: audio heads (hidden -> audio_vocab_size + 1)
        # Only heads 0..n_vq are used at runtime
        self.lm_heads = [
            nn.Linear(config.hidden_size, config.vocab_size, bias=False),
        ] + [
            nn.Linear(config.hidden_size, config.audio_vocab_size + 1, bias=False)
            for _ in range(config.n_vq_weights)
        ]

    def get_input_embeddings(self, input_ids: mx.array) -> mx.array:
        """Compute combined text + audio VQ embeddings.

        Args:
            input_ids: (B, S, 1 + n_vq) - text token IDs + audio code IDs.

        Returns:
            embeddings: (B, S, hidden_size) - summed embeddings.
        """
        # Text embedding from Qwen3's embed_tokens
        text_ids = input_ids[:, :, 0]
        embeds = self.language_model.model.embed_tokens(text_ids)

        # Add audio VQ embeddings (only first n_vq channels used at runtime)
        for i in range(self.config.n_vq):
            embeds = embeds + self.emb_ext[i](input_ids[:, :, i + 1])

        return embeds

    def __call__(
        self,
        input_ids: mx.array,
        cache=None,
    ) -> tuple[list[mx.array], Any]:
        """Forward pass.

        Args:
            input_ids: (B, S, 1 + n_vq)
            cache: KV cache from previous step.

        Returns:
            logits: list of (B, S, V) arrays, one per head.
            cache: updated KV cache.
        """
        inputs_embeds = self.get_input_embeddings(input_ids)
        text_ids = input_ids[:, :, 0].astype(mx.int32)
        h = self.language_model.model(
            text_ids,
            cache=cache,
            input_embeddings=inputs_embeds,
        )

        # Project through active heads (text + n_vq audio heads)
        logits = []
        for i in range(self.config.n_vq + 1):
            head = self.lm_heads[i]
            head_logits = head(h)
            # Mask the last token (padding) for audio heads
            if i > 0:
                head_logits = mx.where(
                    mx.arange(head_logits.shape[-1]) == head_logits.shape[-1] - 1,
                    mx.array(float("-inf")),
                    head_logits,
                )
            logits.append(head_logits)

        return logits, cache

    def make_cache(self):
        """Create a KV cache compatible with the wrapped Qwen3 backbone."""
        return make_prompt_cache(self.language_model)

    @staticmethod
    def sanitize(weights: dict[str, Any]) -> dict[str, Any]:
        """Map PyTorch MOSS-SoundEffect weight keys to MLX model keys.

        Key mappings:
        - language_model.* -> language_model.* (Qwen3 naming matches mlx_lm)
        - emb_ext.N.weight -> emb_ext.N.weight (direct)
        - lm_heads.N.weight -> lm_heads.N.weight (direct)
        """
        new_weights = {}
        for key, value in weights.items():
            new_key = key

            # Qwen3 backbone: language_model.model.* stays as is
            # (mlx_lm's Qwen3 uses same naming)

            # Handle any potential naming differences
            # The HF checkpoint uses language_model.embed_tokens, language_model.layers, etc.
            # mlx_lm uses model.embed_tokens, model.layers, etc.
            # Our wrapper wraps as language_model.model.*, so we need:
            # language_model.embed_tokens -> language_model.model.embed_tokens
            if key.startswith("language_model.") and not key.startswith("language_model.model."):
                # Map language_model.X -> language_model.model.X
                suffix = key[len("language_model."):]
                new_key = f"language_model.model.{suffix}"

            new_weights[new_key] = value

        return new_weights

    @classmethod
    def from_pretrained(
        cls,
        model_path: str | Path,
        config: MossSoundEffectConfig | None = None,
        dtype: mx.Dtype = mx.bfloat16,
    ) -> "MossSoundEffectModel":
        """Load pretrained model from converted MLX weights."""
        import json
        model_path = Path(model_path)

        if config is None:
            config_path = model_path / "config.json"
            with open(config_path) as f:
                config_dict = json.load(f)
            if "language_config" in config_dict:
                config = MossSoundEffectConfig.from_hf_config(config_dict)
            else:
                config = MossSoundEffectConfig.from_dict(config_dict)

        model = cls(config)

        # Apply quantization to model structure before loading weights
        if config.quantization is not None:
            from .weight_mapping import should_quantize

            def class_predicate(path: str, module: nn.Module) -> bool:
                if not isinstance(module, nn.Linear):
                    return False
                return should_quantize(path)

            nn.quantize(
                model,
                bits=config.quantization["bits"],
                group_size=config.quantization["group_size"],
                class_predicate=class_predicate,
            )

        # Load weights (may be sharded)
        import glob
        weight_files = sorted(glob.glob(str(model_path / "*.safetensors")))

        all_weights = {}
        for wf in weight_files:
            all_weights.update(mx.load(wf))

        # Sanitize keys
        all_weights = cls.sanitize(all_weights)

        # Cast dtype (skip quantized int/uint types)
        typed = {}
        for k, v in all_weights.items():
            if v.dtype in (mx.float32, mx.float16, mx.bfloat16):
                typed[k] = v.astype(dtype)
            else:
                typed[k] = v

        model.load_weights(list(typed.items()), strict=False)
        return model
