"""MOSS Audio Tokenizer model (MLX)."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import mlx.core as mx
import mlx.nn as nn

from .config import MossAudioTokenizerConfig
from .patched_pretransform import PatchedPretransform
from .quantizer import ResidualLFQ, ResidualVQ
from .transformer import ProjectedTransformer


class MossAudioTokenizerModel(nn.Module):
    """MOSS Audio Tokenizer: encode audio waveforms to discrete codes and decode back.

    Architecture:
        - Encoder: alternating PatchedPretransform (reshape downsample) + ProjectedTransformer
        - Quantizer: Residual LFQ or RVQ (32 codebooks x 1024 tokens x 8-dim)
        - Decoder: mirror of encoder (upsample)
    """

    def __init__(self, config: MossAudioTokenizerConfig):
        super().__init__()
        self.config = config
        self.sampling_rate = config.sampling_rate
        self.downsample_rate = config.downsample_rate

        # Build encoder pipeline
        current_frame_rate = float(self.sampling_rate)
        self.encoder_modules: list[ProjectedTransformer | PatchedPretransform] = []

        for enc_kwargs in config.encoder_kwargs:
            enc_kwargs = dict(enc_kwargs)
            module_type = enc_kwargs.pop("module_type")
            if module_type == "PatchedPretransform":
                mod = PatchedPretransform(
                    patch_size=enc_kwargs["patch_size"],
                    is_downsample=True,
                )
                self.encoder_modules.append(mod)
            elif module_type == "Transformer":
                enc_kwargs.pop("conv_layout", None)
                context = int(current_frame_rate * config.causal_transformer_context_duration)
                mod = ProjectedTransformer(context=context, **enc_kwargs)
                self.encoder_modules.append(mod)
            current_frame_rate /= self.encoder_modules[-1].downsample_ratio

        # Build quantizer
        q_kwargs = dict(config.quantizer_kwargs)
        q_type = q_kwargs.pop("quantizer_type", config.quantizer_type)
        if q_type in {"rlfq", "random_prefix_rlfq"}:
            self.quantizer = ResidualLFQ(**q_kwargs)
        elif q_type in {"rvq", "spec_rvq"}:
            self.quantizer = ResidualVQ(**q_kwargs)
        else:
            raise ValueError(f"Unsupported quantizer_type: {q_type}")

        # Build decoder pipeline
        self.decoder_modules: list[ProjectedTransformer | PatchedPretransform] = []

        for dec_kwargs in config.decoder_kwargs:
            dec_kwargs = dict(dec_kwargs)
            module_type = dec_kwargs.pop("module_type")
            if module_type == "PatchedPretransform":
                mod = PatchedPretransform(
                    patch_size=dec_kwargs["patch_size"],
                    is_downsample=False,
                )
                self.decoder_modules.append(mod)
            elif module_type == "Transformer":
                dec_kwargs.pop("conv_layout", None)
                context = int(current_frame_rate * config.causal_transformer_context_duration)
                mod = ProjectedTransformer(context=context, **dec_kwargs)
                self.decoder_modules.append(mod)
            current_frame_rate *= self.decoder_modules[-1].downsample_ratio

        # Register transformer modules with nn.Module for parameter tracking
        self._encoder_transformers = [
            m for m in self.encoder_modules if isinstance(m, ProjectedTransformer)
        ]
        self._decoder_transformers = [
            m for m in self.decoder_modules if isinstance(m, ProjectedTransformer)
        ]

    def encode(self, audio: mx.array, num_quantizers: int | None = None) -> mx.array:
        """Encode audio waveform to discrete codes.

        Args:
            audio: (B, 1, T) waveform at sampling_rate Hz.
            num_quantizers: Number of codebooks to use (default: all).

        Returns:
            codes: (N_q, B, T_codes) discrete audio codes.
        """
        if audio.ndim == 2:
            audio = audio[:, None, :]  # (B, T) -> (B, 1, T)

        B, _, T = audio.shape

        # Pad to multiple of downsample_rate
        if T % self.downsample_rate != 0:
            pad_length = self.downsample_rate - (T % self.downsample_rate)
            audio = mx.pad(audio, [(0, 0), (0, 0), (0, pad_length)])

        # Run encoder
        x = audio
        for mod in self.encoder_modules:
            x = mod(x)

        # Quantize
        _, codes = self.quantizer.encode(x, n_quantizers=num_quantizers)
        return codes

    def decode(self, codes: mx.array, num_quantizers: int | None = None) -> mx.array:
        """Decode discrete codes to audio waveform.

        Args:
            codes: (N_q, B, T_codes) discrete audio codes.
            num_quantizers: Number of codebooks to use (default: all in codes).

        Returns:
            audio: (B, 1, T_samples) waveform.
        """
        if codes.ndim == 2:
            codes = codes[:, None, :]  # (N_q, T) -> (N_q, 1, T)

        if num_quantizers is not None:
            codes = codes[:num_quantizers]

        # Dequantize
        x = self.quantizer.decode_codes(codes)

        # Run decoder
        for mod in self.decoder_modules:
            x = mod(x)

        return x

    @staticmethod
    def sanitize(weights: dict[str, Any]) -> dict[str, Any]:
        """Map PyTorch checkpoint keys to MLX model keys.

        Handles:
        - Weight norm decomposition: parametrizations.weight.original0/1 -> weight_g/weight_v
        - Fused in_proj splitting: in_proj_weight / in_proj.weight -> in_proj.weight
        - Attention weight remapping: in_projs.0 / out_projs.0 -> in_proj / out_proj
        - Encoder/decoder module list indexing
        """
        new_weights = {}
        # Pattern for weight_norm parametrization
        wn_pattern = re.compile(
            r"^(.+)\.parametrizations\.weight\.original([01])$"
        )
        # Pattern for in_projs.N / out_projs.N -> in_proj / out_proj
        attn_proj_pattern = re.compile(
            r"^(.+)\.(in_projs|out_projs)\.0\.(.+)$"
        )

        for key, value in weights.items():
            # Handle weight_norm parametrization
            m = wn_pattern.match(key)
            if m:
                base = m.group(1)
                idx = m.group(2)
                if idx == "0":
                    new_key = f"{base}.weight_g"
                else:
                    new_key = f"{base}.weight_v"
                new_weights[new_key] = value
                continue

            # Handle attention projection renaming (in_projs.0 -> in_proj)
            m = attn_proj_pattern.match(key)
            if m:
                base = m.group(1)
                proj_type = m.group(2)
                suffix = m.group(3)
                # in_projs.0 -> in_proj, out_projs.0 -> out_proj
                new_proj = proj_type.rstrip("s")  # in_projs -> in_proj, out_projs -> out_proj
                new_key = f"{base}.{new_proj}.{suffix}"
                new_weights[new_key] = value
                continue

            # Handle old-style in_proj_weight -> in_proj.weight
            if key.endswith(".in_proj_weight"):
                base = key[:-len(".in_proj_weight")]
                new_weights[f"{base}.in_proj.weight"] = value
                continue
            if key.endswith(".in_proj.weight") and ".in_projs." not in key:
                new_weights[key] = value
                continue
            if key.endswith(".out_proj.weight") and ".out_projs." not in key:
                new_weights[key] = value
                continue

            # Map encoder/decoder module list keys
            new_weights[key] = value

        # Second pass: map encoder.N / decoder.N to encoder_modules / decoder_modules
        # and normalize weight-norm conv tensors to MLX's flattened layout.
        final_weights = {}
        for key, value in new_weights.items():
            if key.startswith("encoder."):
                key = key.replace("encoder.", "encoder_modules.", 1)
            if key.startswith("decoder."):
                key = key.replace("decoder.", "decoder_modules.", 1)

            # PyTorch's 1x1 Conv1d weights keep a trailing singleton dimension.
            if key.endswith(("weight_v", "weight_g")) and len(value.shape) == 3 and value.shape[-1] == 1:
                value = value.squeeze(-1)

            final_weights[key] = value

        return final_weights

    @classmethod
    def from_pretrained(
        cls,
        model_path: str | Path,
        config: MossAudioTokenizerConfig | None = None,
        dtype: mx.Dtype = mx.float32,
    ) -> "MossAudioTokenizerModel":
        """Load pretrained model from converted MLX weights."""
        model_path = Path(model_path)

        if config is None:
            config = MossAudioTokenizerConfig.from_json(model_path / "tokenizer_config.json")

        model = cls(config)

        # Load weights
        weights = mx.load(str(model_path / "tokenizer_weights.safetensors"))
        weights = cls.sanitize(weights)

        # Cast dtype
        typed_weights = {}
        for k, v in weights.items():
            if v.dtype in (mx.float32, mx.float16, mx.bfloat16):
                typed_weights[k] = v.astype(dtype)
            else:
                typed_weights[k] = v

        model.load_weights(list(typed_weights.items()))
        return model
