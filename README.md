# LTX-2.3 on Modal

Run the [Lightricks LTX-2.3](https://huggingface.co/Lightricks/LTX-2.3) video generation model (22B parameters) on Modal H200 GPUs. Uses the official Lightricks inference code — not a reimplementation.

Supports all 6 generation modes: text-to-video, image-to-video, HQ 1080p, audio-to-video, keyframe interpolation, and temporal retake.

## Setup

```bash
pip install modal
modal setup          # one-time auth
```

## Usage

```python
from generate_video import LTXVideo

# Text-to-video (5 seconds, best quality)
ltx = LTXVideo(mode="standard")
result = ltx.generate.remote(prompt="A cat sitting on a windowsill watching rain")
with open("output.mp4", "wb") as f:
    f.write(result["video_bytes"])
```

```python
# Fast mode (~4x faster, distilled model)
fast = LTXVideo(mode="fast")
result = fast.generate.remote(prompt="A dragon emerges from storm clouds", seed=77)
```

```python
# Image-to-video
ltx = LTXVideo(mode="standard")
result = ltx.generate.remote(
    prompt="She turns and smiles",
    image_bytes=open("photo.jpg", "rb").read(),
)
```

```python
# Audio-to-video
ltx = LTXVideo(mode="a2vid")
result = ltx.generate_from_audio.remote(
    prompt="A guitarist shreds a solo on stage",
    audio_bytes=open("music.wav", "rb").read(),
)
```

```python
# Keyframe interpolation
ltx = LTXVideo(mode="keyframe")
result = ltx.interpolate.remote(
    prompt="A smooth transition from day to night",
    keyframe_images=[
        (open("sunrise.jpg", "rb").read(), 0, 1.0),
        (open("sunset.jpg", "rb").read(), 120, 1.0),
    ],
    num_frames=121,
)
```

## Modes

| Mode | Steps | Resolution | Speed (5s video) | Description |
|------|-------|-----------|-------------------|-------------|
| **standard** | 30+4 | 1024x1536 | ~71s | Best quality, CFG guidance |
| **fast** | 8+4 | 1024x1536 | ~15s | Distilled model, fixed sigmas |
| **hq** | 15+4 | 1088x1920 | ~40s (2s) | 1080p, res_2s second-order sampler |
| **a2vid** | 30+4 | 1024x1536 | ~71s | Audio-conditioned generation |
| **keyframe** | 30+4 | 1024x1536 | ~85s | Interpolation between frames |
| **retake** | 40 | source | ~1490s (10s) | Edit a time region of existing video |

All two-stage modes run stage 1 at 50% resolution, then upscale 2x with a distilled LoRA refinement pass.

## Architecture

Each `(mode, precision)` combination gets its own Modal container pool. Models are loaded into GPU VRAM at container startup — no per-request loading overhead.

```
LTXVideo(mode, precision) → dedicated container pool → H200 GPU
                                                       ├── Gemma 3 12B (text encoder)
                                                       ├── Transformer (dev or distilled)
                                                       ├── Video VAE (encoder + decoder)
                                                       ├── Audio VAE + Vocoder
                                                       └── Spatial upsampler (2x)
```

Containers scale to zero after 15 minutes idle. Cold start downloads models on first run (~2 min), then they're cached on a Modal volume.

## Precision

| Precision | Quality | VRAM | Notes |
|-----------|---------|------|-------|
| **bf16** (default) | Full | ~44 GB | Recommended on H200 |
| **fp8** | Slight loss | ~22 GB | Only if VRAM-constrained |

```python
ltx = LTXVideo(mode="fast", precision="fp8")
```

## Tests

```bash
# All 8 tests in parallel (standard, fast, hq, i2v, enhance, a2vid, keyframe, retake)
modal run test_all_modes.py::run_tests

# Single test
modal run test_all_modes.py::run_tests --test 2
```

Tests require assets in the project directory: `test_image.jpeg`, `test_audio.wav`, `test_kf1.jpeg`, `test_kf2.jpeg`.

## Docs

See [docs.md](docs.md) for the full user guide — prompting tips, guidance parameters, duration/resolution tables, and all API parameters.
