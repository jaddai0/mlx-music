"""Configuration for MOSS Audio Tokenizer."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class MossAudioTokenizerConfig:
    sampling_rate: int = 24000
    downsample_rate: int = 1920
    causal_transformer_context_duration: float = 10.0
    quantizer_type: str = "rlfq"

    encoder_kwargs: list[dict[str, Any]] = field(default_factory=lambda: [
        {"module_type": "PatchedPretransform", "patch_size": 240},
        {
            "module_type": "Transformer",
            "input_dimension": 240, "output_dimension": 384,
            "d_model": 768, "num_heads": 12, "num_layers": 12,
            "dim_feedforward": 3072, "causal": True, "norm": "layer_norm",
            "positional_embedding": "rope", "max_period": 10000,
            "gating": "none", "layer_scale": 0.01, "conv_layout": True,
        },
        {"module_type": "PatchedPretransform", "patch_size": 2},
        {
            "module_type": "Transformer",
            "input_dimension": 768, "output_dimension": 384,
            "d_model": 768, "num_heads": 12, "num_layers": 12,
            "dim_feedforward": 3072, "causal": True, "norm": "layer_norm",
            "positional_embedding": "rope", "max_period": 10000,
            "gating": "none", "layer_scale": 0.01, "conv_layout": True,
        },
        {"module_type": "PatchedPretransform", "patch_size": 2},
        {
            "module_type": "Transformer",
            "input_dimension": 768, "output_dimension": 640,
            "d_model": 768, "num_heads": 12, "num_layers": 12,
            "dim_feedforward": 3072, "causal": True, "norm": "layer_norm",
            "positional_embedding": "rope", "max_period": 10000,
            "gating": "none", "layer_scale": 0.01, "conv_layout": True,
        },
        {"module_type": "PatchedPretransform", "patch_size": 2},
        {
            "module_type": "Transformer",
            "input_dimension": 1280, "output_dimension": 768,
            "d_model": 1280, "num_heads": 20, "num_layers": 32,
            "dim_feedforward": 5120, "causal": True, "norm": "layer_norm",
            "positional_embedding": "rope", "max_period": 10000,
            "gating": "none", "layer_scale": 0.01, "conv_layout": True,
        },
    ])

    decoder_kwargs: list[dict[str, Any]] = field(default_factory=lambda: [
        {
            "module_type": "Transformer",
            "input_dimension": 768, "output_dimension": 1280,
            "d_model": 1280, "num_heads": 20, "num_layers": 32,
            "dim_feedforward": 5120, "causal": True, "norm": "layer_norm",
            "positional_embedding": "rope", "max_period": 10000,
            "gating": "none", "layer_scale": 0.01, "conv_layout": True,
        },
        {"module_type": "PatchedPretransform", "patch_size": 2},
        {
            "module_type": "Transformer",
            "input_dimension": 640, "output_dimension": 768,
            "d_model": 768, "num_heads": 12, "num_layers": 12,
            "dim_feedforward": 3072, "causal": True, "norm": "layer_norm",
            "positional_embedding": "rope", "max_period": 10000,
            "gating": "none", "layer_scale": 0.01, "conv_layout": True,
        },
        {"module_type": "PatchedPretransform", "patch_size": 2},
        {
            "module_type": "Transformer",
            "input_dimension": 384, "output_dimension": 768,
            "d_model": 768, "num_heads": 12, "num_layers": 12,
            "dim_feedforward": 3072, "causal": True, "norm": "layer_norm",
            "positional_embedding": "rope", "max_period": 10000,
            "gating": "none", "layer_scale": 0.01, "conv_layout": True,
        },
        {"module_type": "PatchedPretransform", "patch_size": 2},
        {
            "module_type": "Transformer",
            "input_dimension": 384, "output_dimension": 768,
            "d_model": 768, "num_heads": 12, "num_layers": 12,
            "dim_feedforward": 3072, "causal": True, "norm": "layer_norm",
            "positional_embedding": "rope", "max_period": 10000,
            "gating": "none", "layer_scale": 0.01, "conv_layout": True,
        },
        {"module_type": "PatchedPretransform", "patch_size": 2},
        {
            "module_type": "Transformer",
            "input_dimension": 384, "output_dimension": 240,
            "d_model": 768, "num_heads": 12, "num_layers": 12,
            "dim_feedforward": 3072, "causal": True, "norm": "layer_norm",
            "positional_embedding": "rope", "max_period": 10000,
            "gating": "none", "layer_scale": 0.01, "conv_layout": True,
        },
        {"module_type": "PatchedPretransform", "patch_size": 240},
    ])

    quantizer_kwargs: dict[str, Any] = field(default_factory=lambda: {
        "input_dim": 768,
        "rvq_dim": 512,
        "output_dim": 768,
        "num_quantizers": 32,
        "codebook_size": 1024,
        "codebook_dim": 8,
        "quantizer_type": "rlfq",
    })

    @property
    def num_quantizers(self) -> int:
        return self.quantizer_kwargs.get("num_quantizers", 32)

    @property
    def codebook_size(self) -> int:
        return self.quantizer_kwargs.get("codebook_size", 1024)

    @property
    def frame_rate(self) -> float:
        return self.sampling_rate / self.downsample_rate

    def to_dict(self) -> dict[str, Any]:
        return {
            "sampling_rate": self.sampling_rate,
            "downsample_rate": self.downsample_rate,
            "causal_transformer_context_duration": self.causal_transformer_context_duration,
            "quantizer_type": self.quantizer_type,
            "encoder_kwargs": self.encoder_kwargs,
            "decoder_kwargs": self.decoder_kwargs,
            "quantizer_kwargs": self.quantizer_kwargs,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "MossAudioTokenizerConfig":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})

    @classmethod
    def from_json(cls, path: str | Path) -> "MossAudioTokenizerConfig":
        with open(path) as f:
            return cls.from_dict(json.load(f))
