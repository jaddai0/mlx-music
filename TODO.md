# MLX-Music TODO

## Current Coverage

### MusicGen Models

| Model | Status | Notes |
|-------|--------|-------|
| `facebook/musicgen-small` | ✅ Supported | SafeTensors |
| `facebook/musicgen-medium` | ✅ Supported | SafeTensors |
| `facebook/musicgen-large` | ✅ Supported | PyTorch .bin (sharded) |
| `facebook/musicgen-melody` | ✅ Supported | SafeTensors |
| `facebook/musicgen-stereo-small` | ✅ Supported | Stereo output |
| `facebook/musicgen-stereo-medium` | ✅ Supported | Stereo output |
| `facebook/musicgen-stereo-large` | ✅ Supported | Stereo output |
| `facebook/musicgen-stereo-melody` | ✅ Supported | Stereo output |
| `facebook/musicgen-stereo-melody-large` | ✅ Supported | Stereo output |

### ACE-Step Models

| Model | Status | Notes |
|-------|--------|-------|
| `ACE-Step/ACE-Step-v1-3.5B` | ✅ Supported | Full lyric support |

### Stable Audio Models

| Model | Status | Notes |
|-------|--------|-------|
| `stabilityai/stable-audio-open-1.0` | ✅ Supported | 44.1kHz stereo, up to 47s |

### Features

| Feature | Status |
|---------|--------|
| Text-to-music generation | ✅ |
| Melody-conditioned generation | ✅ |
| Audio continuation | ✅ |
| Extended generation (>30s) | ✅ |
| Classifier-Free Guidance | ✅ |
| Top-k/Top-p sampling | ✅ |
| Parallel shard loading | ✅ |
| Lazy weight filtering | ✅ |
| Stereo audio output | ✅ |
| Lyric-conditioned generation (ACE-Step) | ✅ |
| DiT diffusion transformer (Stable Audio) | ✅ |
| High-quality 44.1kHz output (Stable Audio) | ✅ |
| EDM DPM-Solver scheduling | ✅ |

---

## Future Considerations

- Alternative text encoders (beyond T5-base)
- Multi-rate EnCodec support
- Streaming generation
- ControlNet-style fine-grained conditioning
- Additional audio models (AudioLDM, MusicLDM)
