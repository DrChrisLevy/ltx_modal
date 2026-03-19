# LTX-2.3 Video Generation — User Guide

Generate videos from text, images, and audio using the Lightricks LTX-2.3 model (22B parameters) running on Modal H200 GPUs with FP8 quantization.

## Quick Start

```bash
# Text-to-video (5 seconds, default settings)
uv run modal run generate_video.py --prompt "A cat sitting on a windowsill watching rain"

# Fast mode (~2x faster, slightly lower quality)
uv run modal run generate_video.py --prompt "..." --mode fast

# HQ mode (1080p, res_2s sampler)
uv run modal run generate_video.py --prompt "..." --mode hq

# Longer video (10 seconds)
uv run modal run generate_video.py --prompt "..." --num-frames 241

# Image-to-video (animate a photo)
uv run modal run generate_video.py --prompt "She turns and smiles" --image photo.jpg

# Multiple videos with different seeds
uv run modal run generate_video.py --prompt "..." --num-videos 3

# Deploy as web API
uv run modal deploy generate_video.py
```

## Generation Modes

### standard (default)

Best quality. Uses the full dev model with classifier-free guidance and spatio-temporal guidance for high-fidelity output.

- **Steps**: 30 (stage 1) + 4 (stage 2 refinement)
- **Resolution**: 1024x1536 (stage 1 at 512x768, then 2x upscaled)
- **Best for**: Final output, production quality

### fast

Fastest generation. Uses a distilled model with a fixed 8-step sigma schedule — no guidance needed.

- **Steps**: 8 (stage 1) + 4 (stage 2 refinement)
- **Resolution**: 1024x1536
- **Best for**: Rapid iteration, previewing prompts, batch generation

### hq

Highest quality. Uses a second-order res_2s sampler that achieves better quality in fewer steps, at 1080p resolution.

- **Steps**: 15 (stage 1) + 4 (stage 2 refinement)
- **Resolution**: 1088x1920 (1080p)
- **Best for**: Maximum quality, final renders, showcase content

## CLI Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--prompt` | *(required)* | Text description of the desired video |
| `--mode` | `standard` | Generation mode: `standard`, `fast`, or `hq` |
| `--seed` | `42` | Random seed for reproducibility |
| `--num-frames` | `121` | Number of frames (see duration table below) |
| `--frame-rate` | `24.0` | Playback FPS |
| `--height` | auto | Output height in pixels (divisible by 64) |
| `--width` | auto | Output width in pixels (divisible by 64) |
| `--num-inference-steps` | auto | Denoising steps (auto per mode) |
| `--num-videos` | `1` | Generate multiple videos (increments seed) |
| `--image` | | Path to conditioning image for image-to-video |
| `--enhance-prompt` | `false` | Let Gemma 3 expand your prompt with cinematic detail |

### Auto Defaults by Mode

| Setting | standard | fast | hq |
|---------|----------|------|-----|
| Resolution | 1024x1536 | 1024x1536 | 1088x1920 |
| Steps | 30 | 8 (fixed) | 15 |
| CFG guidance | 3.0 | none | 3.0 |
| STG guidance | 1.0 | none | disabled |
| Rescale | 0.7 | none | 0.45 |

## Duration & Frame Count

Duration = `num_frames / frame_rate`. Frames must be **8k+1** format (the model's temporal compression requires this).

| num_frames | Duration @24fps | Notes |
|-----------|-----------------|-------|
| 49 | 2.0s | Quick test |
| 73 | 3.0s | |
| 97 | 4.0s | |
| **121** | **5.0s** | **Default** |
| 145 | 6.0s | |
| 169 | 7.0s | |
| 193 | 8.0s | |
| 241 | 10.0s | |
| 481 | 20.0s | Max tested |

If you pass a non-valid frame count, it auto-rounds up to the nearest valid value.

## Resolution

Height and width must be divisible by **64** (two-stage pipeline requires this since stage 1 runs at half resolution).

| Preset | Height x Width | Notes |
|--------|---------------|-------|
| **Default** | 1024 x 1536 | Standard/fast landscape |
| **HQ** | 1088 x 1920 | 1080p, auto for hq mode |
| Portrait | 1536 x 1024 | Swap height/width |
| Square | 1024 x 1024 | |

Higher resolution = more VRAM for VAE decode. All tested resolutions work up to 20s (481 frames) on H200.

## Writing Good Prompts

LTX-2.3 works best with detailed, **chronological** descriptions written like a shot list. Think like a cinematographer.

**Structure your prompt:**
1. Start with the main action in one sentence
2. Add specific movements and gestures
3. Describe character/object appearances precisely
4. Include background and environment details
5. Specify camera angles and movements
6. Describe lighting and colors
7. Note any changes or events

**Keep within 200 words.** Start directly with the action, keep descriptions literal and precise.

**Good prompt:**
> A woman in a red dress walks along a rain-soaked city street at night. Neon signs in blues and pinks reflect off the wet pavement. She pauses to look up at a flickering sign, her face illuminated by its glow. The camera tracks alongside her at eye level, slowly pushing in as she turns toward the lens. Shallow depth of field blurs the background traffic into soft bokeh.

**Bad prompt:**
> Beautiful woman walking in city, cinematic, 4K, highly detailed

**Enhance prompt**: Pass `--enhance-prompt` to let Gemma 3 automatically expand a short prompt into a detailed cinematic description. Useful when you have a rough idea but don't want to write the full description.

## Image-to-Video

Animate a still image. The image conditions the first frame — the model generates motion from there.

```bash
# CLI
uv run modal run generate_video.py \
    --prompt "She slowly turns her head and smiles, a breeze moves her hair" \
    --image photo.jpg \
    --mode fast

# Python API
result = ltx.generate.remote(
    prompt="She slowly turns her head and smiles",
    image_bytes=open("photo.jpg", "rb").read(),
    image_strength=1.0,  # 0.0–1.0, higher = more faithful to input image
    mode="fast",
)
```

**Tips:**
- `image_strength=1.0` (default) closely matches the input image
- Lower values (0.5-0.8) give the model more creative freedom
- The image is placed at frame 0 — the model generates forward from it
- Works with any mode (standard, fast, hq)

## Python API — Advanced Features

The CLI covers text-to-video and image-to-video. For audio-to-video, keyframe interpolation, and retake, use the Python API:

### Audio-to-Video

Generate video synchronized to an audio track. The model generates visuals that match the audio's rhythm and mood.

```python
from generate_video import LTXVideo

ltx = LTXVideo()
result = ltx.generate_from_audio.remote(
    prompt="A guitarist shreds a solo on stage, colorful lights flash and pulse",
    audio_bytes=open("music.wav", "rb").read(),
    num_frames=121,                # 5 seconds
    seed=42,
    # Optional:
    audio_start_time=0.0,          # offset into audio file (seconds)
    audio_max_duration=None,       # cap audio length
    image_bytes=None,              # optional first-frame image
)
```

**Requirements:** Audio must be **stereo WAV** (2 channels). The model's audio VAE expects stereo input.

### Keyframe Interpolation

Generate smooth video transitions between two or more keyframe images.

```python
result = ltx.interpolate.remote(
    prompt="A smooth transition from day to night, the sky shifts from warm orange to deep blue",
    keyframe_images=[
        (open("sunrise.jpg", "rb").read(), 0, 1.0),      # frame 0, full strength
        (open("sunset.jpg", "rb").read(), 120, 1.0),      # frame 120, full strength
    ],
    num_frames=121,
    seed=42,
)
```

Each keyframe is a tuple of `(image_bytes, frame_index, strength)`.

### Retake (Video Editing)

Regenerate a specific time region of an existing video while preserving the rest.

```python
# Generate base video
base = ltx.generate.remote(
    prompt="A woman walks down a city street at night",
    mode="fast",
    num_frames=241,  # 10 seconds
    seed=99,
)

# Retake a section with a different prompt
result = ltx.retake.remote(
    video_bytes=base["video_bytes"],
    prompt="A monster crashes through buildings on a city street",
    start_time=3.0,
    end_time=8.0,
    seed=200,
    # Optional:
    regenerate_video=True,     # regenerate video in the time region
    regenerate_audio=True,     # regenerate audio in the time region
    num_inference_steps=40,    # retake uses 40 steps by default
)
```

**Note:** Retake is slow (~8 min for a 10s video) because it runs the full model at 40 steps on the entire video at source resolution. The temporal mask controls which frames get regenerated.

### List Generated Videos

```python
videos = ltx.list_outputs.remote()
# Returns: [{"filename": "...", "size_mb": 1.5, "metadata": {...}}, ...]
```

Videos auto-save to the `ltx-outputs` Modal volume. Access directly:
```bash
modal volume ls ltx-outputs
modal volume get ltx-outputs <filename> ./local.mp4
```

## Web API

After deploying with `uv run modal deploy generate_video.py`:

**Generate video:**
```
GET /api_generate?prompt=A+cat+on+a+windowsill&mode=fast&num_frames=121&seed=42
→ Returns MP4 file
```

**List saved videos:**
```
GET /api_list
→ Returns JSON array
```

Swagger docs at: `<endpoint-url>/docs`

## Guidance Parameters (Advanced)

These control the diffusion process. Only relevant for `standard` and `hq` modes — `fast` mode ignores them.

| Parameter | Default (standard) | Default (hq) | Range | Effect |
|-----------|-------------------|-------------|-------|--------|
| `cfg_scale` | 3.0 | 3.0 | 1.0–5.0 | Prompt adherence. Higher = more literal, but less natural motion |
| `stg_scale` | 1.0 | 0.0 | 0.0–1.5 | Spatio-temporal coherence. Perturbs transformer blocks for better consistency |
| `rescale_scale` | 0.7 | 0.45 | 0.0–1.0 | Prevents over-saturation from strong guidance |
| `modality_scale` | 3.0 | 3.0 | 1.0–5.0 | Audio-visual sync. Set to 1.0 if you don't care about audio |
| `stg_blocks` | [28] | [] | — | Which transformer block to perturb for STG |

**Audio guidance** uses `cfg_scale=7.0` (higher than video) for stronger prompt adherence in audio generation.

Pass via Python API:
```python
result = ltx.generate.remote(
    prompt="...",
    mode="standard",
    cfg_scale=4.0,       # stronger prompt adherence
    stg_scale=0.5,       # less temporal guidance
    rescale_scale=0.5,   # less rescaling
)
```

## Performance

Tested on NVIDIA H200 (141 GB), FP8 quantization.

### Generation Speed (warm container)

| Mode | 2s (49f) | 5s (121f) | 10s (241f) | 20s (481f) |
|------|----------|-----------|------------|------------|
| **fast** | ~50s | ~50s | ~97s | ~159s |
| **standard** | ~136s | ~162s | — | — |
| **hq** | ~82s | — | — | — |

~50-55s of each call is model loading overhead (see NOTES.md for optimization roadmap).

### Cost

- H200: ~$4.85/hr (billed per second on Modal)
- Typical 5s video: ~$0.04 (warm, fast) / ~$0.12 (cold start)

## Feature Comparison with Official LTX API

| Feature | Our Modal App | Official LTX API |
|---------|--------------|------------------|
| Text-to-video | Yes (3 modes) | Yes (fast/pro) |
| Image-to-video | Yes | Yes |
| Audio-to-video | Yes | Yes (pro only) |
| Keyframe interpolation | Yes | Yes (ltx-2-3 only) |
| Retake (edit regions) | Yes | Yes (pro only) |
| Extend (lengthen video) | **No** | Yes (pro only) |
| Camera motion presets | **No** (use prompt) | Yes (dolly, jib, static) |
| 4K resolution | **No** (max 1080p) | Yes |
| Max duration | 20s tested | 20s |
| Prompt enhancement | Yes | Not documented |
| Custom guidance params | Yes (full control) | Limited |
| Seed reproducibility | Yes | Yes |
| Batch generation | Yes | No |
| Self-hosted | Yes (your GPU) | No (their cloud) |

## Running Tests

```bash
# Run all 8 tests in parallel (each on its own H200)
uv run modal run test_all_modes.py::run_tests

# Run a specific test
uv run modal run test_all_modes.py::run_tests --test 1

# Tests:
#   1. standard mode (text-to-video, 5s)
#   2. fast mode (text-to-video, 5s)
#   3. hq mode (text-to-video, 2s, 1080p)
#   4. image-to-video (real photo)
#   5. enhance_prompt
#   6. audio-to-video (real audio)
#   7. keyframe interpolation (real photos)
#   8. retake (generate base + retake region)
```

Tests 4, 6, 7 require test assets in the project directory:
- `test_image.jpeg` — photo for image-to-video
- `test_audio.wav` — stereo WAV for audio-to-video
- `test_kf1.jpeg`, `test_kf2.jpeg` — keyframe images
