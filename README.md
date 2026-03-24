# LTX-2.3 on Modal

Run [Lightricks LTX-2.3](https://huggingface.co/Lightricks/LTX-2.3) (22B parameters) on Modal H200 GPUs. All 6 generation modes — text-to-video, image-to-video, HQ, audio-to-video, keyframe interpolation, and temporal retake.

Uses the official Lightricks inference code directly — not a reimplementation. The default code in the ltx repo loads and unloads models from disk between stages (designed for consumer GPUs). Here, all models — text encoder, transformers, VAEs, vocoder, upsampler — are loaded into H200 VRAM once at container startup and patched into the pipeline's `ModelLedger` so they stay resident. Two-stage modes keep both transformers (~88 GB) in memory simultaneously. Zero disk I/O between stages, no GC pauses — diffusion starts immediately on every request.

## Setup

```bash
uv add modal
uv run modal setup                       # one-time auth
```

Create a `.env` file with your Hugging Face token (needed for the [Gemma 3](https://huggingface.co/google/gemma-3-12b-it-qat-q4_0-unquantized) text encoder, which requires accepting Google's license):

```
HF_TOKEN=hf_your_token_here
```

Deploy:

```bash
uv run modal deploy generate_video.py
```

Models download automatically on first request and are cached on a Modal volume.

## Generate a video

```python
import modal

LTXVideo = modal.Cls.from_name("ltx-video", "LTXVideo")

prompt = """A sleek black sports car races out of a tunnel into blinding golden hour light, The camera view is from the inside of the car, looking out the windshield."""

ltx = LTXVideo(mode="standard")
result = ltx.generate.remote(prompt=prompt)

with open("car_race.mp4", "wb") as f:
    f.write(result["video_bytes"])
```

Videos are also saved to the `ltx-outputs` Modal volume with JSON metadata.

## Modes

| Mode | Description |
|------|-------------|
| **standard** | Best quality text/image-to-video |
| **fast** | ~4x faster, distilled model |
| **hq** | 1080p with second-order sampler |
| **a2vid** | Audio-conditioned video generation |
| **keyframe** | Interpolation between keyframe images |
| **retake** | Edit a time region of existing video |

## Examples

### Image-to-video

```python
ltx = LTXVideo(mode="standard")
result = ltx.generate.remote(
    prompt="INT. SPACECRAFT COCKPIT – anime cel-shaded style. A teenage male pilot in an orange mission flight suit grips a joystick, cockpit HUD screens glowing green, red warning lights pulsing on the walls. The camera holds nearly static on the existing framing, with a very subtle slow push-in toward the pilot's face. His eyes shift slightly, jaw tightens, and his grip hand micro-adjusts on the joystick. The red emergency lights continue their slow rhythmic pulse. A low cockpit alarm hums faintly in the background.",
    image_bytes=open("test_image.jpeg", "rb").read(),
    seed=42,
)
with open("animated_image.mp4", "wb") as f:
    f.write(result["video_bytes"])
```

### Audio-to-video

```python
ltx = LTXVideo(mode="a2vid")
result = ltx.generate_from_audio.remote(
prompt="A guitarist shreds a solo on stage",
    audio_bytes=open("test_audio.wav", "rb").read(),
)
with open("audio_video.mp4", "wb") as f:
    f.write(result["video_bytes"])
```

### Keyframe interpolation

```python
ltx = LTXVideo(mode="keyframe")
result = ltx.interpolate.remote(
    prompt="An astronaut in a white spacesuit walks steadily forward across a rocky alien ridge at dawn. The camera is static, wide cinematic shot from behind. He steps off the jagged rock edge and strides forward into the vast orange desert, growing slightly smaller in frame. His arms swing naturally mid-walk. The dawn sky gradually brightens — the orange horizon glow expands upward into lavender. Two moons hang motionless in the indigo sky. Dust drifts faintly around his boots. Smooth, slow, cinematic movement. Photorealistic.",
    keyframe_images=[
        (open("img1.jpeg", "rb").read(), 0, 1.0),
        (open("img2.jpeg", "rb").read(), 120, 1.0),
    ],
    num_frames=121,
)
with open("interpolated_video.mp4", "wb") as f:
    f.write(result["video_bytes"])
```

### Retake (edit a time region)

```python
# Generate a base video first
ltx = LTXVideo(mode="standard")
base = ltx.generate.remote(prompt="A calm city street at night, neon signs, wet pavement")

# Replace seconds 2-4 with new content
retake = LTXVideo(mode="retake")
result = retake.retake.remote(
    video_bytes=base["video_bytes"],
    prompt="A sudden explosion lights up the street, debris flies everywhere",
    start_time=2.0,
    end_time=4.0,
)
with open("retake_video.mp4", "wb") as f:
    f.write(result["video_bytes"])
```

### Comparing Standard vs Fast VS HQ VS FP8

```python
prompt="INT. SUNLIT APARTMENT – LATE AFTERNOON fluffy tabby cat sits curled on a windowsill, warm golden light washing across its fur, soft shadows pooling beneath it. The camera holds a gentle static medium shot, barely breathing with the faintest slow push-in toward the cat's face. The cat blinks slowly, shifts its weight, and turns its head slightly toward the glass. A curtain beside it sways once in a lazy draft, then stills. Outside, leaves flutter and birds chirp softly in the distance."
ltx = LTXVideo(mode="standard")
result = ltx.generate.remote(prompt=prompt)
with open("cat_standard.mp4", "wb") as f:
    f.write(result["video_bytes"])

ltx = LTXVideo(mode="hq")
result = ltx.generate.remote(prompt=prompt)
with open("cat_hq.mp4", "wb") as f:
    f.write(result["video_bytes"])

ltx = LTXVideo(mode="fast")
result = ltx.generate.remote(prompt=prompt)
with open("cat_fast.mp4", "wb") as f:
    f.write(result["video_bytes"])

ltx = LTXVideo(mode="standard", precision="fp8")
result = ltx.generate.remote(prompt=prompt)
with open("cat_standard_fp8.mp4", "wb") as f:
    f.write(result["video_bytes"])
```

---

## Reference

### Duration

`num_frames / frame_rate`. Frames auto-snap to 8k+1 format.

| Frames | Duration @24fps |
|--------|-----------------|
| 49 | 2s |
| 97 | 4s |
| **121** | **5s (default)** |
| 241 | 10s |
| 481 | 20s |

### Resolution

Must be divisible by 64. Defaults: 1024x1536 (standard/fast), 1088x1920 (hq).

### Precision

| | Quality | VRAM | Notes |
|-|---------|------|-------|
| **bf16** | Full | ~44 GB | Default, recommended on H200 |
| **fp8** | Slight loss | ~22 GB | Only if VRAM-constrained |

### Guidance (standard/hq only, ignored by fast)

| Parameter | Standard | HQ | Range |
|-----------|----------|-----|-------|
| `cfg_scale` | 3.0 | 3.0 | 1.0-5.0 |
| `stg_scale` | 1.0 | 0.0 | 0.0-1.5 |
| `rescale_scale` | 0.7 | 0.45 | 0.0-1.0 |

### `generate()` parameters

| Parameter | Default | |
|-----------|---------|---|
| `prompt` | *required* | Text description |
| `negative_prompt` | `""` | What to avoid |
| `seed` | `42` | Random seed |
| `height` / `width` | auto | Divisible by 64 |
| `num_frames` | `121` | Auto-snapped to 8k+1 |
| `frame_rate` | `24.0` | Playback FPS |
| `num_inference_steps` | auto | Denoising steps |
| `cfg_scale` | `3.0` | Classifier-free guidance |
| `stg_scale` | auto | Spatio-temporal guidance |
| `rescale_scale` | auto | Prevents over-saturation |
| `image_bytes` | `None` | Image bytes for I2V |
| `image_strength` | `1.0` | Conditioning strength |
| `enhance_prompt` | `False` | Gemma prompt expansion |

### `generate_from_audio()` adds

| Parameter | Default | |
|-----------|---------|---|
| `audio_bytes` | *required* | Stereo WAV bytes |
| `audio_start_time` | `0.0` | Offset into audio |
| `audio_max_duration` | `None` | Max duration to use |

### `interpolate()` replaces `image_bytes` with

| Parameter | Default | |
|-----------|---------|---|
| `keyframe_images` | *required* | List of `(bytes, frame_idx, strength)` |

### `retake()` parameters

| Parameter | Default | |
|-----------|---------|---|
| `video_bytes` | *required* | Source video |
| `prompt` | *required* | Description for edited region |
| `start_time` / `end_time` | *required* | Region bounds (seconds) |
| `regenerate_video` | `True` | Regenerate video track |
| `regenerate_audio` | `True` | Regenerate audio track |

Plus `seed`, `negative_prompt`, `num_inference_steps` (40), `cfg_scale`, `stg_scale`, `rescale_scale`, `enhance_prompt`.

---

## Volumes

| Volume | Path | Purpose |
|--------|------|---------|
| `ltx-models` | `/models` | Cached model weights (auto-downloaded) |
| `ltx-outputs` | `/outputs` | Generated videos + JSON metadata |

```bash
modal volume ls ltx-outputs
modal volume get ltx-outputs 20260323_sunset_s42.mp4 ./local.mp4
```

## Architecture

Each `(mode, precision)` pair gets its own container pool. Models are loaded into VRAM once at startup — diffusion starts immediately on every request.

```
LTXVideo(mode, precision) → container pool → H200 GPU
                                             ├── Gemma 3 12B text encoder
                                             ├── Transformer (dev or distilled)
                                             ├── Video VAE encoder + decoder
                                             ├── Audio VAE + Vocoder
                                             └── Spatial upsampler (2x)
```

Containers scale to zero after 15 min idle. Cost: ~$0.02 per 5s fast video (H200 at $4.54/hr, billed per second).

### Why persistent models?

The default Lightricks pipelines delete and reload the transformer from disk between stages — designed for consumer GPUs where VRAM is tight. On an H200 with 141 GB, there's no reason to unload.

This project patches the `ModelLedger` at startup so all models stay GPU-resident. The pipeline's `del transformer; cleanup_memory()` calls still run but only drop a local reference — the patched lambda keeps the model alive. Zero disk I/O between stages.

## Prompting

Write a **single paragraph in present tense**, 4-8 sentences. Think like a cinematographer.

> A woman in a red dress walks along a rain-soaked city street at night. Neon signs in blues and pinks reflect off the wet pavement. She pauses to look up at a flickering sign, her face illuminated by its glow. The camera tracks alongside her at eye level, slowly pushing in as she turns toward the lens. Shallow depth of field blurs the background traffic into soft bokeh.

Not this: *"Beautiful woman walking in city, cinematic, 4K, highly detailed"*

Tips: explicit camera instructions ("slow dolly in"), atmospheric elements (fog, rain, golden hour), physical emotion cues instead of labels. Characters can talk and sing.

## Tests

```bash
uv run modal run test_all_modes.py::run_tests        # all 8 in parallel
uv run modal run test_all_modes.py::run_tests --test 2  # single test
```

Requires: `test_image.jpeg`, `test_audio.wav`, `test_kf1.jpeg`, `test_kf2.jpeg` in the project directory.
