# MLX-Music TODO

## Planned Features

### Stereo Audio Support for MusicGen

**Priority:** High
**Estimated Effort:** 3-4 days
**Status:** Not Started

#### Overview

Add support for Facebook's MusicGen stereo model variants:

| Model | Parameters | Status |
|-------|-----------|--------|
| `facebook/musicgen-stereo-small` | 300M | Not Supported |
| `facebook/musicgen-stereo-medium` | 1.5B | Not Supported |
| `facebook/musicgen-stereo-large` | 3.3B | Not Supported |
| `facebook/musicgen-stereo-melody` | 1.5B | Not Supported |
| `facebook/musicgen-stereo-melody-large` | 3.3B | Not Supported |

#### Why This is Feasible

The codebase architecture is already channel-agnostic:
- Transformer decoder operates on codebooks, not audio channels
- Generation loop is codec-agnostic
- Melody conditioning already handles multi-channel input (averages to mono)
- Audio I/O utilities already support stereo

#### Implementation Plan

**Phase 1: Configuration (~10 lines)**
- [ ] Make `audio_channels` configurable in `MusicGenDecoderConfig`
- [ ] Make `audio_channels` configurable in `MusicGenAudioEncoderConfig`
- [ ] Add validation for supported channel values (1 or 2)

**Phase 2: Codec Integration (~25 lines)**
- [ ] Update `get_encodec()` to accept `audio_channels` parameter
- [ ] Update `PlaceholderEnCodec` to respect `audio_channels` config
- [ ] Ensure `EnCodecWrapper` properly handles 2-channel audio

**Phase 3: Model Initialization (~10 lines)**
- [ ] Pass `audio_channels` from config to codec during model init
- [ ] Update `MusicGen.from_pretrained()` to detect stereo models

**Phase 4: Testing**
- [ ] Test loading stereo model weights
- [ ] Verify stereo audio generation output
- [ ] Validate backward compatibility with mono models
- [ ] Performance benchmarking (stereo vs mono)

#### Files to Modify

```
mlx_music/models/musicgen/config.py      - Add audio_channels parameter
mlx_music/codecs/__init__.py             - Update PlaceholderEnCodec
mlx_music/models/musicgen/model.py       - Pass audio_channels to codec
```

#### Technical Notes

- Stereo models use the same 4-codebook architecture as mono
- EnCodec handles the stereo-to-codebook mapping internally
- Pre-trained stereo weights should load with minimal code changes
- The transformer architecture requires NO changes

---

## Current Coverage

### MusicGen Models

| Model | Status | Notes |
|-------|--------|-------|
| `facebook/musicgen-small` | ✅ Supported | SafeTensors |
| `facebook/musicgen-medium` | ✅ Supported | SafeTensors |
| `facebook/musicgen-large` | ✅ Supported | PyTorch .bin (sharded) |
| `facebook/musicgen-melody` | ✅ Supported | SafeTensors |
| `facebook/musicgen-stereo-*` | ❌ Not Yet | See above |

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
| Stereo audio output | ❌ |

---

## Future Considerations

- Alternative text encoders (beyond T5-base)
- Multi-rate EnCodec support
- Streaming generation
- ControlNet-style fine-grained conditioning
