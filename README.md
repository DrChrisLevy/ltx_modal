# LTX-2.3 on Modal

Run the [Lightricks LTX-2.3](https://huggingface.co/Lightricks/LTX-2.3) video generation model (22B parameters) on Modal H200 GPUs. Uses the official Lightricks inference code — not a reimplementation.

Supports all 6 generation modes: text-to-video, image-to-video, HQ 1080p, audio-to-video, keyframe interpolation, and temporal retake.

## Setup

```bash
uv add modal
uv run modal setup          # one-time auth
uv run modal deploy generate_video.py   # deploy the app
```

## Usage

```python
import modal

LTXVideo = modal.Cls.from_name("ltx-video", "LTXVideo")

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

```python
# Retake (edit a time region of existing video)
retake_handle = LTXVideo(mode="retake")
result = retake_handle.retake.remote(
    video_bytes=base_result["video_bytes"],
    prompt="A monster crashes through buildings",
    start_time=3.0,
    end_time=8.0,
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

## API Reference

### `generate()` — text/image-to-video (standard, fast, hq)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `prompt` | *(required)* | Text description |
| `negative_prompt` | `""` | What to avoid |
| `seed` | `42` | Random seed |
| `height` | auto | Output height (divisible by 64) |
| `width` | auto | Output width (divisible by 64) |
| `num_frames` | `121` | Frame count (auto-snapped to 8k+1) |
| `frame_rate` | `24.0` | Playback FPS |
| `num_inference_steps` | auto | Denoising steps |
| `cfg_scale` | `3.0` | Classifier-free guidance strength |
| `stg_scale` | auto | Spatio-temporal guidance (1.0 standard, 0.0 hq) |
| `rescale_scale` | auto | Prevents over-saturation (0.7 standard, 0.45 hq) |
| `image_bytes` | `None` | Image bytes for I2V |
| `image_strength` | `1.0` | Image conditioning strength |
| `enhance_prompt` | `False` | Let Gemma expand your prompt |

### `generate_from_audio()` — audio-to-video (a2vid)

All parameters from `generate()` plus:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `audio_bytes` | *(required)* | Stereo WAV bytes |
| `audio_start_time` | `0.0` | Where to start in the audio |
| `audio_max_duration` | `None` | Max audio duration to use |

### `interpolate()` — keyframe interpolation

Same as `generate()` but replaces `image_bytes` with:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `keyframe_images` | *(required)* | List of `(bytes, frame_idx, strength)` tuples |

### `retake()` — temporal region editing

| Parameter | Default | Description |
|-----------|---------|-------------|
| `video_bytes` | *(required)* | Source video bytes |
| `prompt` | *(required)* | Text description for the region |
| `start_time` | *(required)* | Region start (seconds) |
| `end_time` | *(required)* | Region end (seconds) |
| `seed` | `42` | Random seed |
| `negative_prompt` | `""` | What to avoid |
| `num_inference_steps` | `40` | Denoising steps |
| `cfg_scale` | `3.0` | Classifier-free guidance |
| `stg_scale` | `1.0` | Spatio-temporal guidance |
| `rescale_scale` | `0.7` | Prevents over-saturation |
| `regenerate_video` | `True` | Regenerate video in region |
| `regenerate_audio` | `True` | Regenerate audio in region |
| `enhance_prompt` | `False` | Let Gemma expand your prompt |

### Return value

All methods return a dict:

```python
{
    "video_bytes": b"...",       # raw MP4 bytes
    "filename": "20260323_...",  # filename on output volume
    "duration": 5.04,            # seconds
    "size_mb": 12.3,
    "mode": "standard",
    "gen_time_s": 71.2,
}
```

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

Containers scale to zero after 15 minutes idle. Cold start downloads models on first run (~2 min), then they're cached on the `ltx-models` Modal volume.

All generated videos are saved to the `ltx-outputs` Modal volume with timestamped filenames and JSON metadata:

```bash
modal volume ls ltx-outputs
modal volume get ltx-outputs <filename> ./local.mp4
```

### Persistent models vs. default Lightricks behavior

The default Lightricks pipelines are designed for consumer GPUs where VRAM is tight. Between stages, they `del transformer` → `gc.collect()` → `torch.cuda.empty_cache()` → reload the transformer from disk for stage 2. The `ModelLedger` uses a `DummyRegistry` by default — no caching, every `transformer()` call reads weights from safetensors on disk.

This is safe but slow. On an H200 with 141 GB of VRAM, there's no reason to unload anything.

This project patches the `ModelLedger` at container startup so that `transformer()`, `video_encoder()`, etc. return pre-built GPU-resident models instead of rebuilding from disk. Both stage 1 and stage 2 transformers (with different LoRA configurations) stay in VRAM simultaneously. The pipeline's `del transformer; cleanup_memory()` calls still run, but they only drop a local reference — the patched lambda keeps the model alive.

Result: zero disk I/O between stages, no GC pauses, no model reconstruction. Diffusion starts immediately on every request.

## Duration & Frame Count

Duration = `num_frames / frame_rate`. Frames must be **8k+1** format (auto-rounded if not).

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

## Resolution

Height and width must be divisible by **64**.

| Preset | Resolution | Notes |
|--------|-----------|-------|
| Default | 1024 x 1536 | standard/fast landscape |
| HQ | 1088 x 1920 | Auto for hq mode |
| Portrait | 1536 x 1024 | Swap height/width |
| Square | 1024 x 1024 | |

## Precision

| Precision | Quality | VRAM | Notes |
|-----------|---------|------|-------|
| **bf16** (default) | Full | ~44 GB | Recommended on H200 |
| **fp8** | Slight loss | ~22 GB | Only if VRAM-constrained |

```python
ltx = LTXVideo(mode="fast", precision="fp8")
result = ltx.generate.remote(prompt="...", seed=42)
```

## Guidance Parameters

Only relevant for `standard` and `hq` modes. `fast` mode ignores them.

| Parameter | Standard | HQ | Effect |
|-----------|----------|-----|--------|
| `cfg_scale` | 3.0 | 3.0 | Prompt adherence (1.0-5.0) |
| `stg_scale` | 1.0 | 0.0 | Spatio-temporal coherence (0.0-1.5) |
| `rescale_scale` | 0.7 | 0.45 | Prevents over-saturation (0.0-1.0) |

## Prompting Guide

Write prompts as a **single flowing paragraph** in **present tense**, 4-8 sentences. Think like a cinematographer describing a shot list. Keep within 200 words.

**Structure:** Shot establishment → scene setting → action → character definition → camera movement → audio.

**Good:**
> A woman in a red dress walks along a rain-soaked city street at night. Neon signs in blues and pinks reflect off the wet pavement. She pauses to look up at a flickering sign, her face illuminated by its glow. The camera tracks alongside her at eye level, slowly pushing in as she turns toward the lens. Shallow depth of field blurs the background traffic into soft bokeh.

**Bad:**
> Beautiful woman walking in city, cinematic, 4K, highly detailed

**Tips:**
- Cinematic compositions with thoughtful lighting
- Show emotion through physical cues, not labels ("sad", "happy")
- Explicit camera instructions: "slow dolly in", "handheld tracking"
- Atmospheric elements: fog, mist, golden-hour light, rain
- Characters can talk and sing (multiple languages)
- Avoid: text/logos, overloaded scenes, conflicting lighting

## Performance (H200, BF16)

| Mode | 2s (49f) | 5s (121f) | 10s (241f) |
|------|----------|-----------|------------|
| **fast** | ~33s | ~15s | ~38s |
| **standard** | — | ~71s | — |
| **hq** | ~40s | — | — |
| **a2vid** | — | ~71s | — |
| **keyframe** | — | ~85s | — |
| **retake** | — | — | ~1490s |

Cost: H200 at $4.54/hr (billed per second). Typical 5s fast video: ~$0.02.

## Tests

```bash
# All 8 tests in parallel (standard, fast, hq, i2v, enhance, a2vid, keyframe, retake)
uv run modal run test_all_modes.py::run_tests

# Single test
uv run modal run test_all_modes.py::run_tests --test 2
```

Tests require assets in the project directory: `test_image.jpeg`, `test_audio.wav`, `test_kf1.jpeg`, `test_kf2.jpeg`.
