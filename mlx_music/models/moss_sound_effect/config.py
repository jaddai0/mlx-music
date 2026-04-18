"""Configuration for MOSS-SoundEffect model."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class MossSoundEffectConfig:
    # Qwen3 backbone params
    hidden_size: int = 4096
    num_hidden_layers: int = 36
    num_attention_heads: int = 32
    num_key_value_heads: int = 8
    intermediate_size: int = 12288
    vocab_size: int = 155648
    rope_theta: float = 1_000_000.0
    max_position_embeddings: int = 40960
    rms_norm_eps: float = 1e-6
    head_dim: int = 128

    # MOSS audio params
    n_vq: int = 16  # Number of VQ channels used at runtime
    n_vq_weights: int = 32  # Number of VQ channels stored in checkpoint weights
    audio_vocab_size: int = 1024
    audio_pad_code: int = 1024
    sampling_rate: int = 24000

    # Special token IDs
    pad_token_id: int = 151643
    im_start_token_id: int = 151644
    im_end_token_id: int = 151645
    audio_start_token_id: int = 151652
    audio_end_token_id: int = 151653
    audio_user_slot_token_id: int = 151654
    audio_assistant_gen_slot_token_id: int = 151656
    audio_assistant_delay_slot_token_id: int = 151662

    # Quantization (set after conversion)
    quantization: dict | None = None

    def to_dict(self) -> dict[str, Any]:
        d = {}
        for field_name in self.__dataclass_fields__:
            d[field_name] = getattr(self, field_name)
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "MossSoundEffectConfig":
        known = {k: v for k, v in d.items() if k in cls.__dataclass_fields__}
        return cls(**known)

    @classmethod
    def from_json(cls, path: str | Path) -> "MossSoundEffectConfig":
        with open(path) as f:
            return cls.from_dict(json.load(f))

    @classmethod
    def from_hf_config(cls, hf_config: dict[str, Any]) -> "MossSoundEffectConfig":
        """Create config from HuggingFace config.json (MossTTSDelayConfig format)."""
        lang = hf_config.get("language_config", {})
        return cls(
            hidden_size=lang.get("hidden_size", 4096),
            num_hidden_layers=lang.get("num_hidden_layers", 36),
            num_attention_heads=lang.get("num_attention_heads", 32),
            num_key_value_heads=lang.get("num_key_value_heads", 8),
            intermediate_size=lang.get("intermediate_size", 12288),
            vocab_size=lang.get("vocab_size", 155648),
            rope_theta=lang.get("rope_theta", 1_000_000.0),
            max_position_embeddings=lang.get("max_position_embeddings", 40960),
            rms_norm_eps=lang.get("rms_norm_eps", 1e-6),
            head_dim=lang.get("head_dim", 128),
            n_vq=hf_config.get("n_vq", 16),
            n_vq_weights=hf_config.get("n_vq_weights", 32),
            audio_vocab_size=hf_config.get("audio_vocab_size", 1024),
            audio_pad_code=hf_config.get("audio_pad_code", 1024),
            sampling_rate=hf_config.get("sampling_rate", 24000),
            pad_token_id=hf_config.get("pad_token_id", 151643),
            im_start_token_id=hf_config.get("im_start_token_id", 151644),
            im_end_token_id=hf_config.get("im_end_token_id", 151645),
            audio_start_token_id=hf_config.get("audio_start_token_id", 151652),
            audio_end_token_id=hf_config.get("audio_end_token_id", 151653),
            audio_user_slot_token_id=hf_config.get("audio_user_slot_token_id", 151654),
            audio_assistant_gen_slot_token_id=hf_config.get("audio_assistant_gen_slot_token_id", 151656),
            audio_assistant_delay_slot_token_id=hf_config.get("audio_assistant_delay_slot_token_id", 151662),
        )
