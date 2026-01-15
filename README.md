# MLX Music

The first MLX-native music generation library for Apple Silicon.

Generate music from text descriptions and lyrics, optimized for M1/M2/M3/M4 Macs.

## Status: Active Development

This library is under active development with three model families supported.

## Supported Models

| Model | Status | Parameters | Description |
|-------|--------|------------|-------------|
| ACE-Step | ✅ Supported | 3.5B | Diffusion model for lyrics-to-music generation |
| MusicGen | ✅ Supported | 300M-3.3B | Autoregressive LM for text-to-music (mono/stereo) |
| Stable Audio | ✅ Supported | 1.2B | DiT-based diffusion for high-quality audio (44.1kHz stereo) |

## Installation

```bash
# From source (recommended during development)
git clone https://github.com/Dfunk55/mlx-music.git
cd mlx-music
pip install -e ".[dev]"
```

## Quick Start

### ACE-Step (Lyrics-to-Music)
```python
from mlx_music import ACEStep

model = ACEStep.from_pretrained("ACE-Step/ACE-Step-v1-3.5B")
output = model.generate(
    prompt="upbeat electronic dance music",
    lyrics="Verse 1: Dancing through the night...",
    duration=30.0,
)

import soundfile as sf
sf.write("output.wav", output.audio.T, output.sample_rate)
```

### MusicGen (Text-to-Music)
```python
from mlx_music import MusicGen

model = MusicGen.from_pretrained("facebook/musicgen-stereo-medium")
output = model.generate(
    prompt="jazz piano with drums",
    duration=10.0,
)

import soundfile as sf
sf.write("output.wav", output.audio.T, output.sample_rate)
```

### Stable Audio (High-Quality Audio)
```python
from mlx_music import StableAudio

model = StableAudio.from_pretrained("stabilityai/stable-audio-open-1.0")
output = model.generate(
    prompt="ambient electronic music with soft pads",
    duration=30.0,
    guidance_scale=7.0,
)

import soundfile as sf
sf.write("output.wav", output.audio.T, output.sample_rate)
```

## CLI Usage

```bash
# Generate music
mlx-music generate \
    --prompt "calm piano melody with soft strings" \
    --duration 30 \
    --output output.wav

# With lyrics
mlx-music generate \
    --prompt "pop ballad" \
    --lyrics "Verse 1: Under starlit skies we dance..." \
    --duration 60 \
    --output ballad.wav
```

## Architecture

MLX Music implements three model architectures:

### ACE-Step (3.5B params)
- **Linear Transformer**: 24 blocks with linear attention (O(n) complexity)
- **DCAE**: Audio encoder/decoder with 8x compression
- **HiFi-GAN Vocoder**: Mel-spectrogram to waveform
- **UMT5 Text Encoder**: Text and lyrics conditioning

### MusicGen (300M-3.3B params)
- **Autoregressive LM**: Decoder-only transformer
- **EnCodec**: Neural audio codec (mono/stereo)
- **T5 Text Encoder**: Text conditioning
- **Delay Pattern**: Efficient multi-codebook generation

### Stable Audio (1.2B params)
- **DiT Transformer**: 24-layer diffusion transformer with GQA
- **AutoencoderOobleck**: 2048x compression VAE with Snake activation
- **T5 Text Encoder**: Text conditioning
- **EDM Scheduler**: DPM-Solver++ with Karras sigmas

## Quantization

Reduce memory usage and improve performance:

```python
from mlx_music import ACEStep
from mlx_music.weights import QuantizationConfig, quantize_model

model = ACEStep.from_pretrained("ACE-Step/ACE-Step-v1-3.5B")

# Quantize for speed (INT4)
config = QuantizationConfig.for_speed()
model = quantize_model(model.transformer, config)

# Or balanced (INT8 attention + INT4 FFN)
config = QuantizationConfig.for_balanced()
```

## Requirements

- Python 3.10+
- Apple Silicon Mac (M1/M2/M3/M4)
- MLX >= 0.25.0

## Acknowledgements

- [Apple MLX Team](https://github.com/ml-explore/mlx) - MLX framework
- [ACE-Step](https://github.com/ace-step/ACE-Step) - Original model
- [MFLUX](https://github.com/filipstrand/mflux) - Architecture patterns
- [mlx-audio](https://github.com/Blaizzy/mlx-audio) - DSP utilities
- [LTX-2 MLX](https://github.com/Acelogic/LTX-2-MLX) - Reference implementation

## License

MIT License - See [LICENSE](LICENSE)

## Citation

```bibtex
@misc{mlx-music,
  author = {MLX Music Contributors},
  title = {MLX Music: Native Music Generation for Apple Silicon},
  year = {2025},
  howpublished = {\url{https://github.com/Dfunk55/mlx-music}},
}
```
