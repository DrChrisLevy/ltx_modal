# LTX-2.3 Video Generation — User Guide

Generate videos from text, images, and audio using the Lightricks LTX-2.3 model (22B parameters) running on Modal H200 GPUs.

## Quick Start

```bash
# Text-to-video (5 seconds, standard mode, BF16)
uv run modal run generate_video.py --mode standard --prompt "A cat sitting on a windowsill watching rain"

# Fast mode (~4x faster)
uv run modal run generate_video.py --mode fast --prompt "..."

# HQ mode (1080p, res_2s sampler)
uv run modal run generate_video.py --mode hq --prompt "..."

# Longer video (10 seconds)
uv run modal run generate_video.py --mode fast --prompt "..." --num-frames 241

# Image-to-video (animate a photo)
uv run modal run generate_video.py --mode standard --prompt "She turns and smiles" --image photo.jpg

# FP8 precision (lower quality, uses less VRAM)
uv run modal run generate_video.py --mode standard --prompt "..." --precision fp8
```

## Architecture

Each `(mode, precision)` combination runs in its **own container pool** via Modal's parametrized functions. Containers load only the models their mode requires — no wasted VRAM.

```
LTXVideo(mode="standard")  →  Container pool A (dev transformer + dev+LoRA transformer)
LTXVideo(mode="fast")      →  Container pool B (distilled transformer only)
LTXVideo(mode="hq")        →  Container pool C (two HQ LoRA transformers)
LTXVideo(mode="a2vid")     →  Container pool D (same models as standard)
LTXVideo(mode="keyframe")  →  Container pool E (same models as standard)
LTXVideo(mode="retake")    →  Container pool F (dev transformer only, no upscaler)
```

All models are pre-loaded into GPU VRAM at container startup. No per-request model loading — diffusion starts immediately.

## Generation Modes

### standard (default)

Best quality. Uses the full dev model with classifier-free guidance.

- **Steps**: 30 (stage 1) + 4 (stage 2 refinement)
- **Resolution**: 1024x1536
- **Gen time**: ~71s for 5s video (BF16)

### fast

Fastest generation. Uses a distilled model with fixed sigma schedule.

- **Steps**: 8 (stage 1) + 4 (stage 2 refinement)
- **Resolution**: 1024x1536
- **Gen time**: ~15s for 5s video (BF16)

### hq

Highest quality. Uses res_2s second-order sampler at 1080p.

- **Steps**: 15 (stage 1) + 4 (stage 2 refinement)
- **Resolution**: 1088x1920
- **Gen time**: ~40s for 2s video (BF16)

### a2vid

Audio-conditioned video generation. Text prompt guides visuals, audio guides temporal dynamics.

- **Steps**: 30 (stage 1) + 4 (stage 2 refinement)
- **Resolution**: 1024x1536
- **Gen time**: ~71s for 5s video (BF16)
- **Requires**: Stereo WAV audio file

### keyframe

Interpolate between keyframe images with smooth transitions.

- **Steps**: 30 (stage 1) + 4 (stage 2 refinement)
- **Resolution**: 1024x1536
- **Gen time**: ~85s for 5s video (BF16)
- **Requires**: 2+ keyframe images with frame indices

### retake

Regenerate a specific time region of an existing video.

- **Steps**: 40 (single stage, no upscaling)
- **Resolution**: matches source video
- **Gen time**: ~1490s for 10s video (BF16) — slow by design
- **Requires**: Source video + time range

## CLI Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--prompt` | *(required)* | Text description of the desired video |
| `--mode` | `standard` | Mode: `standard`, `fast`, `hq` |
| `--precision` | `bf16` | Precision: `bf16` or `fp8` |
| `--seed` | `42` | Random seed for reproducibility |
| `--num-frames` | `121` | Number of frames (see duration table) |
| `--frame-rate` | `24.0` | Playback FPS |
| `--height` | auto | Output height (divisible by 64) |
| `--width` | auto | Output width (divisible by 64) |
| `--num-inference-steps` | auto | Denoising steps (auto per mode) |
| `--num-videos` | `1` | Generate multiple videos (increments seed) |
| `--image` | | Path to conditioning image |
| `--enhance-prompt` | `false` | Let Gemma 3 expand your prompt |

## Precision

| Precision | Speed | Quality | VRAM per transformer |
|-----------|-------|---------|---------------------|
| **bf16** (default) | Faster | Full precision | ~44 GB |
| **fp8** | Slower (cast overhead) | Slight loss | ~22 GB |

BF16 is recommended on H200. FP8 is only needed if VRAM-constrained.

## Duration & Frame Count

Duration = `num_frames / frame_rate`. Frames must be **8k+1** format.

| num_frames | Duration @24fps |
|-----------|-----------------|
| 49 | 2.0s |
| 73 | 3.0s |
| 97 | 4.0s |
| **121** | **5.0s (default)** |
| 145 | 6.0s |
| 193 | 8.0s |
| 241 | 10.0s |
| 481 | 20.0s |

Non-valid frame counts auto-round up to the nearest valid value.

## Resolution

Height and width must be divisible by **64**.

| Preset | Resolution | Notes |
|--------|-----------|-------|
| Default | 1024 x 1536 | standard/fast landscape |
| HQ | 1088 x 1920 | Auto for hq mode |
| Portrait | 1536 x 1024 | Swap height/width |
| Square | 1024 x 1024 | |

## Prompting Guide

Write prompts as a **single flowing paragraph** in **present tense**, 4-8 sentences. Think like a cinematographer describing a shot list. Keep within 200 words.

### Structure

1. **Shot establishment** — Camera angle and scale ("Close-up", "wide establishing shot")
2. **Scene setting** — Lighting, color palette, atmosphere
3. **Action** — Core action as a natural sequence, present tense
4. **Character definition** — Age, clothing, features. Show emotion through physical cues, not labels
5. **Camera movement** — How and when the camera moves
6. **Audio** — Ambient sound, music, dialogue

### What Works Well

- Cinematic compositions with thoughtful lighting
- Strong emotional expressions with subtle gestures
- Atmospheric elements: fog, mist, golden-hour light, rain
- Explicit camera instructions: "slow dolly in", "handheld tracking"
- Characters can talk and sing (multiple languages)

### What to Avoid

- Abstract emotions ("sad", "happy") — show it physically
- Text and logos — unreliable
- Overloaded scenes — too many characters or simultaneous actions
- Conflicting lighting descriptions

### Example

**Good:**
> A woman in a red dress walks along a rain-soaked city street at night. Neon signs in blues and pinks reflect off the wet pavement. She pauses to look up at a flickering sign, her face illuminated by its glow. The camera tracks alongside her at eye level, slowly pushing in as she turns toward the lens. Shallow depth of field blurs the background traffic into soft bokeh.

**Bad:**
> Beautiful woman walking in city, cinematic, 4K, highly detailed

## Python API

### Text/Image-to-Video

```python
from generate_video import LTXVideo

ltx = LTXVideo(mode="standard")  # or "fast", "hq"
result = ltx.generate.remote(
    prompt="A timelapse of a flower blooming in morning light",
    seed=42,
    num_frames=121,
    # Optional:
    image_bytes=open("flower.png", "rb").read(),  # image-to-video
    image_strength=1.0,  # 0.0-1.0
    enhance_prompt=False,
)
with open("output.mp4", "wb") as f:
    f.write(result["video_bytes"])
```

### Audio-to-Video

```python
ltx = LTXVideo(mode="a2vid")
result = ltx.generate_from_audio.remote(
    prompt="A guitarist shreds a solo on stage, colorful lights flash",
    audio_bytes=open("music.wav", "rb").read(),  # must be stereo WAV
    num_frames=121,
    seed=42,
    # Optional:
    audio_start_time=0.0,
    audio_max_duration=None,
    image_bytes=None,  # optional first-frame conditioning
)
```

### Keyframe Interpolation

```python
ltx = LTXVideo(mode="keyframe")
result = ltx.interpolate.remote(
    prompt="A smooth transition from day to night",
    keyframe_images=[
        (open("sunrise.jpg", "rb").read(), 0, 1.0),     # frame 0
        (open("sunset.jpg", "rb").read(), 120, 1.0),     # frame 120
    ],
    num_frames=121,
    seed=42,
)
```

### Retake (Video Editing)

```python
# Generate base video
fast = LTXVideo(mode="fast")
base = fast.generate.remote(
    prompt="A woman walks down a city street at night",
    num_frames=241,
    seed=99,
)

# Retake a section
retake = LTXVideo(mode="retake")
result = retake.retake.remote(
    video_bytes=base["video_bytes"],
    prompt="A monster crashes through buildings",
    start_time=3.0,
    end_time=8.0,
    seed=200,
    regenerate_video=True,
    regenerate_audio=True,
)
```

### Custom Precision

```python
# FP8 for lower VRAM usage
ltx = LTXVideo(mode="fast", precision="fp8")
result = ltx.generate.remote(prompt="...", seed=42)
```

## Guidance Parameters (Advanced)

Only relevant for `standard` and `hq` modes. `fast` mode ignores them.

| Parameter | Standard | HQ | Effect |
|-----------|----------|-----|--------|
| `cfg_scale` | 3.0 | 3.0 | Prompt adherence (1.0-5.0) |
| `stg_scale` | 1.0 | 0.0 | Spatio-temporal coherence (0.0-1.5) |
| `rescale_scale` | 0.7 | 0.45 | Prevents over-saturation (0.0-1.0) |

```python
ltx = LTXVideo(mode="standard")
result = ltx.generate.remote(
    prompt="...",
    cfg_scale=4.0,
    stg_scale=0.5,
    rescale_scale=0.5,
)
```

## Performance (H200, BF16, persistent models)

| Mode | 2s (49f) | 5s (121f) | 10s (241f) |
|------|----------|-----------|------------|
| **fast** | ~33s | ~15s | ~38s |
| **standard** | — | ~71s | — |
| **hq** | ~40s | — | — |
| **a2vid** | — | ~71s | — |
| **keyframe** | — | ~85s | — |
| **retake** | — | — | ~1490s |

Cost: H200 at $4.54/hr (billed per second). Typical 5s fast video: ~$0.02.

Containers scale to zero when idle (after 15 min `scaledown_window`). Cold start adds model loading time on first request.

## Output Volume

Videos auto-save to the `ltx-outputs` Modal volume with timestamped filenames and JSON metadata.

```bash
modal volume ls ltx-outputs
modal volume get ltx-outputs <filename> ./local.mp4
```

## Running Tests

```bash
# Run all 8 tests in parallel
uv run modal run test_all_modes.py::run_tests

# Run specific test
uv run modal run test_all_modes.py::run_tests --test 2

# Run with FP8 precision
uv run modal run test_all_modes.py::run_tests --precision fp8
```

Tests 4, 6, 7 require test assets in the project directory:
- `test_image.jpeg` — photo for image-to-video
- `test_audio.wav` — stereo WAV for audio-to-video
- `test_kf1.jpeg`, `test_kf2.jpeg` — keyframe images
