"""
ACE-Step v1.5 model implementation for MLX.

ACE-Step v1.5 is a hybrid music generation model combining:
- Language Model (Qwen3) as a planner for metadata and audio codes
- Diffusion Transformer (DiT) with GQA attention as a renderer
- Oobleck VAE for 48kHz stereo audio encoding/decoding
- FSQ (Finite Scalar Quantization) for discrete audio tokens

Architecture differences from v1:
- v1: Linear Transformer + DCAE + HiFi-GAN + UMT5
- v1.5: GQA DiT + Oobleck VAE + Qwen3 LM + FSQ + Qwen3-Embedding

Variants (same architecture, different training):
- turbo: 8-step distilled, no CFG
- base: 50-step, with CFG support
- sft: supervised fine-tuned base
- turbo-shift1: turbo with shift=1.0 (linear timestep schedule)
- turbo-shift3: turbo with shift=3.0 (front-loaded denoising)
- turbo-continuous: turbo trained for continuous generation
"""

from mlx_music.models.ace_step_v15.config import (
    AceStepV15Config,
    LM_VARIANTS,
    VARIANT_DEFAULTS,
)
from mlx_music.models.ace_step_v15.attention import (
    AceStepV15Attention,
    Qwen3MLP,
    RMSNorm,
    RotaryEmbedding,
    apply_rotary_pos_emb,
    create_sliding_mask,
)
from mlx_music.models.ace_step_v15.dit import (
    AceStepDiTLayer,
    AceStepEncoderLayer,
    TimestepEmbedding,
)
from mlx_music.models.ace_step_v15.dit_model import (
    AceStepConditionGenerationModel,
    AceStepDiTModel,
)
from mlx_music.models.ace_step_v15.vae import AutoencoderOobleck
from mlx_music.models.ace_step_v15.weight_mapping import (
    load_dit_weights,
    load_vae_weights,
    load_silence_latent,
)
from mlx_music.models.ace_step_v15.constrained_decoding import (
    AudioCodeProcessor,
    MetadataFSMProcessor,
    parse_lm_output,
    extract_audio_code_indices,
)
from mlx_music.models.ace_step_v15.lm_planner import LMPlanner
from mlx_music.models.ace_step_v15.model import ACEStepV15

__all__ = [
    "AceStepV15Config",
    "LM_VARIANTS",
    "VARIANT_DEFAULTS",
    "AceStepV15Attention",
    "Qwen3MLP",
    "RMSNorm",
    "RotaryEmbedding",
    "apply_rotary_pos_emb",
    "create_sliding_mask",
    "AceStepDiTLayer",
    "AceStepEncoderLayer",
    "TimestepEmbedding",
    "AceStepConditionGenerationModel",
    "AceStepDiTModel",
    "AutoencoderOobleck",
    "load_dit_weights",
    "load_vae_weights",
    "load_silence_latent",
    "AudioCodeProcessor",
    "MetadataFSMProcessor",
    "parse_lm_output",
    "extract_audio_code_indices",
    "LMPlanner",
    "ACEStepV15",
]
