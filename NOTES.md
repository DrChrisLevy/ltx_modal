# LTX-2.3 on Modal — Notes

## What We Built

A Modal app (`generate_video.py`) that runs **Lightricks LTX-2.3 (22B parameter)** video generation on an **NVIDIA H200 (141 GB)** with FP8 quantization. Supports all LTX-2.3 generation modes from a single endpoint.

```bash
# Text-to-video (standard, 30 steps)
uv run modal run generate_video.py --prompt "A cat sitting on a windowsill"

# Fast mode (distilled, 8+4 steps, ~2x faster)
uv run modal run generate_video.py --prompt "..." --mode fast

# HQ mode (res_2s sampler, 15 steps, 1080p)
uv run modal run generate_video.py --prompt "..." --mode hq

# Longer video (10s)
uv run modal run generate_video.py --prompt "..." --mode fast --num-frames 241

# Image-to-video
uv run modal run generate_video.py --prompt "..." --image input.png

# Deploy web API
uv run modal deploy generate_video.py
```

## Architecture

```
Local machine                    Modal Cloud (H200 141GB)
─────────────                    ────────────────────────
generate_video.py  ──────────►  LTXVideo.setup()           [@modal.enter]
  (uv run modal run)                │ downloads models if needed (~120GB)
                                    │ creates 6 pipeline objects (lightweight)
                                    │ stays warm 15 min between calls
                                    ▼
                               LTXVideo.generate()          [reuses loaded model]
                                    │ runs diffusion + 2x upscale
                                    │ VAE decode (tiled) + MP4 encode
                                    │ saves to output volume
                                    ▼
output.mp4  ◄──────────────────  returns video bytes + metadata

Web API (after deploy):
  GET /api_generate?prompt=...&mode=fast  →  returns MP4
  GET /api_list                           →  returns JSON list
```

## Available Generation Modes

| Mode | Pipeline | Steps | Default Resolution | Use Case |
|------|----------|-------|--------------------|----------|
| `standard` | TI2VidTwoStagesPipeline | 30 | 1024x1536 | Best quality |
| `fast` | DistilledPipeline | 8+4 | 1024x1536 | Fastest, good quality |
| `hq` | TI2VidTwoStagesHQPipeline | 15 | 1088x1920 | Res_2s sampler, 1080p |

All three modes support image-to-video conditioning via `--image` / `image_bytes` parameter.

### Specialized Methods (Python API only)

| Method | Pipeline | Description |
|--------|----------|-------------|
| `generate_from_audio()` | A2VidPipelineTwoStage | Audio-driven video generation |
| `interpolate()` | KeyframeInterpolationPipeline | Smooth interpolation between keyframe images |
| `retake()` | RetakePipeline | Regenerate a specific time region of existing video |

## Model Architecture — What Gets Loaded

### Two Inference Modes

There are really only **two inference modes** for the core diffusion:

1. **Full (dev)** — 30 steps with classifier-free guidance (positive vs negative prompt each step). Best quality.
2. **Distilled** — 8 steps, no guidance needed. A student model trained to mimic the dev model's output in fewer steps. Same architecture (22B DiT), different weights.

Everything else is just variations on how these combine in the two-stage pipeline:

| Pipeline | Stage 1 | Stage 2 | What's different |
|----------|---------|---------|-----------------|
| **standard** | dev, 30 steps | dev + distilled LoRA, 4 steps | Euler sampler |
| **hq** | dev + distilled LoRA, 15 steps | dev + distilled LoRA, 4 steps | Res_2s sampler, different guidance |
| **fast** | distilled, 8 steps | distilled, 4 steps | No guidance at all |

The feature-specific pipelines (a2vid, keyframe, retake, ic_lora) use these same models but handle different inputs.

### What's Inside Each Checkpoint

Both checkpoint files are **monolithic** — they contain ALL components in one file, keyed by prefix:

```
ltx-2.3-22b-dev.safetensors (46GB)          ltx-2.3-22b-distilled.safetensors (46GB)
├── diffusion_model.*  (transformer 22B)     ├── diffusion_model.*  (transformer 22B)
├── vae.encoder.*      (video VAE enc)       ├── vae.encoder.*
├── vae.decoder.*      (video VAE dec)       ├── vae.decoder.*
├── audio_vae.encoder.*                      ├── audio_vae.encoder.*
├── audio_vae.decoder.*                      ├── audio_vae.decoder.*
├── vocoder.*                                ├── vocoder.*
└── embeddings_processor.*                   └── embeddings_processor.*
```

Distillation only retrains the **transformer**. The VAEs, vocoder, and embeddings processor are almost certainly identical between the two files. But Lightricks ships them as monolithic checkpoints, so we download ~46GB of duplicate VAE/vocoder/embeddings weights.

### Actual Component Sizes

| Component | Size (approx) | Shared between dev/distilled? |
|-----------|--------------|-------------------------------|
| Transformer | ~40GB | **No** — different weights per mode |
| Video VAE encoder + decoder | ~2GB | Yes (same in both checkpoints) |
| Audio VAE encoder + decoder | ~1GB | Yes |
| Vocoder | ~0.5GB | Yes |
| Embeddings processor | ~2GB | Yes |
| Gemma 3 12B text encoder | ~24GB | Yes (separate file, always the same) |
| Spatial upscaler | ~1GB | Yes (separate file) |

### The Distilled LoRA Trick

The **distilled LoRA** (7.6GB) captures the difference between dev and distilled transformer weights. In the `standard` two-stage pipeline:

- **Stage 1**: Run the dev transformer for 30 steps (best quality diffusion)
- **Stage 2**: Apply the distilled LoRA to the dev transformer → it behaves like the distilled model → 4 quick refinement steps

This avoids loading a second 46GB checkpoint for stage 2. The distilled LoRA is only needed when using the dev model for stage 1 and wanting fast stage 2 refinement.

### Files on Volume (~120GB)

| File | Size | Used By |
|------|------|---------|
| `ltx-2.3-22b-dev.safetensors` | 46.1 GB | standard, hq, a2vid, keyframe, retake |
| `ltx-2.3-22b-distilled.safetensors` | 46.1 GB | fast (contains ~46GB duplicate VAE/vocoder weights) |
| `ltx-2.3-22b-distilled-lora-384.safetensors` | 7.6 GB | standard, hq stage 2 |
| `ltx-2.3-spatial-upscaler-x2-1.0.safetensors` | 1 GB | all two-stage modes |
| Gemma 3 12B text encoder | ~24 GB | all modes |

Models download automatically on first container start via `@modal.enter()`. No separate CPU download step needed.

## Pipeline Architecture

### Two-Stage Flow (standard, hq, a2vid, keyframe)

1. **Text encoding**: Gemma 3 12B encodes prompt → freed from VRAM
2. **Stage 1**: Diffusion at half resolution (512x768 latent) → 30 steps (standard) or 15 steps (hq)
3. **Stage 2**: Spatial upscaling to 2x (1024x1536) via upscaler + distilled LoRA refinement (3-4 steps)
4. **VAE decode**: Tiled decode of upscaled latents to pixel space
5. **Encode**: H.264 MP4 via ffmpeg + audio from vocoder

### Distilled Flow (fast)

1. **Text encoding**: Gemma 3 12B → freed
2. **Stage 1**: 8 fixed-sigma distilled steps at half resolution (no CFG guidance needed)
3. **Stage 2**: Upsample + 3 refinement steps
4. **VAE decode + encode**: Same as above

### ModelLedger Pattern

Pipelines use a `ModelLedger` for lazy model loading. The `ModelLedger` is a factory that builds model instances from a checkpoint — it does NOT hold loaded models in memory.

- Pipeline objects are lightweight config holders — creating 6 of them uses negligible VRAM
- Each call to `transformer()`, `video_decoder()`, etc. **builds a new model instance** from the checkpoint file
- Between stages, pipelines call `del transformer; cleanup_memory()` to free VRAM
- Peak VRAM = one transformer + one VAE component at a time
- The `DummyRegistry` (default) re-reads from disk every time. `StateDictRegistry` caches raw weights in CPU RAM for faster rebuilds.

### Model Loading Overhead

Every `generate()` call re-does the full load cycle — read safetensors from disk, instantiate model, apply LoRA, quantize to FP8, move to GPU. This takes ~50-55 seconds per call even on a warm container:

| Run | Actual Compute | Total Time | **Loading Overhead** |
|-----|---------------|------------|---------------------|
| fast, 49f | ~7s | 58s | **~51s** |
| fast, 241f | ~42s | 97s | **~55s** |
| fast, 481f (20s video) | ~100s | 159s | **~59s** |

Options to reduce this:
1. **`StateDictRegistry`** — cache weights in CPU RAM after first load. Subsequent calls copy from RAM → GPU instead of re-reading 46GB from Modal volume. Expected: ~10-15s overhead instead of ~55s.
2. **Keep transformer on GPU** — write custom inference loop that holds the model between requests. Requires not using the pipeline classes directly.

## Performance (H200 141GB, FP8)

### Fast Mode (DistilledPipeline, 8+4 steps)

| Frames | Duration | Resolution | Gen Time | Status |
|--------|----------|------------|----------|--------|
| 49 | 2.0s | 1024x1536 | ~58s | OK |
| 121 | 5.0s | 1024x1536 | ~101s | OK |
| 193 | 8.0s | 1024x1536 | ~74s | OK |
| 241 | 10.0s | 1024x1536 | ~97s | OK |
| 481 | 20.0s | 1024x1536 | ~159s | OK |

### Standard Mode (TI2VidTwoStagesPipeline, 30 steps)

| Frames | Duration | Resolution | Gen Time | Status |
|--------|----------|------------|----------|--------|
| 49 | 2.0s | 1024x1536 | ~136s | OK |
| 121 | 5.0s | 1024x1536 | ~136s | OK |

### Cold Start

- First call on new container: ~90-120s model load + generation time
- Warm container (within 15 min `scaledown_window`): generation time only
- Model download (first ever): ~15-30 min for ~120GB

### Cost

- **H200**: ~$4.85/hr (billed per second)
- **Typical cost per 5s video**: ~$0.04 (warm, fast) / ~$0.12 (cold)

### Battle Test Results (all modes, warm container)

All 8 modes tested with real inputs where applicable (`test_all_modes.py`):

| Test | Mode | Input | Duration | Gen Time |
|------|------|-------|----------|----------|
| standard | text-to-video | text only | 5.0s | 162s |
| fast | text-to-video | text only | 5.0s | 50s |
| hq (1080p) | text-to-video | text only | 2.0s | 82s |
| image-to-video | fast + real photo | 5.0s | 40s |
| enhance_prompt | fast + short text | 2.0s | 55s |
| audio-to-video | standard + real WAV | 5.0s | 108s |
| keyframe interp | standard + 2 real photos | 5.0s | 119s |
| retake | full model, 40 steps | 5.0s | 486s |

Warm container times (after first call absorbs cold start). Retake is slow by design — 40 full-model steps at source resolution, no two-stage.

### Community Benchmarks (from HuggingFace discussion #16)

- RTX 5090 (32GB): 481 frames at 1280x720, FP8 distilled → 82s
- RTX 5090 (32GB): 481 frames at 1920x1080, FP8 distilled → 547s
- RTX 5090 + RTX Pro 6000: 241 frames at 1920x1088, Dev two-stage → 222s

## Critical Bug Found & Fixed: Lazy VAE Decode + torch.no_grad()

### The Problem

The pipelines return `(Iterator[torch.Tensor], Audio)` — the video is a **lazy generator**. The actual VAE 3D convolutions execute when `encode_video()` consumes the iterator, NOT when the pipeline `__call__` returns.

Our initial code wrapped only the pipeline call in `torch.no_grad()`:

```python
# WRONG — VAE decode happens OUTSIDE no_grad!
with torch.no_grad():
    video, audio = pipeline(...)   # returns lazy iterator, no decode yet
# Back in grad mode — autograd tracks all conv3d ops during decode
result = encode_result(video, ...)  # THIS is where VAE decode runs
```

Without `no_grad`, PyTorch stores all intermediate tensors for the autograd graph. The 22B transformer + VAE decoder with 3D convolutions at full resolution consumed 138+ GB just in autograd overhead — OOMing on a 141GB H200 at resolutions that work on a 32GB RTX 5090.

### The Fix

Wrap `encode_video` (which consumes the lazy iterator) in `torch.no_grad()` too:

```python
# In _encode_result():
with torch.no_grad():
    encode_video(video=video, ...)  # VAE decode happens here, grad-free
```

The upstream CLI avoids this by wrapping the entire `main()` in `@torch.inference_mode()`.

### Impact

| Config | Before Fix | After Fix |
|--------|-----------|-----------|
| fast, 1024x1536, 49f | OK | OK |
| fast, 1024x1536, 121f | **OOM** | OK (101s) |
| fast, 1024x1536, 193f | **OOM** | OK (74s) |
| fast, 1024x1536, 241f | **OOM** | OK (97s) |

## Previous Bug Fixed: Resolution Semantics

The two-stage pipeline's `height`/`width` parameters are the **final output** resolution. Stage 1 runs at `height//2, width//2`.

The v1 code passed `height=512, width=768`, meaning:
- Stage 1 ran at 256x384 (tiny!)
- Final output was 512x768

The correct defaults (matching upstream CLI):
- `height=1024, width=1536` → stage 1 at 512x768, output at 1024x1536
- Must be divisible by 64 (not 32) for two-stage

## All Parameters

### CLI Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--prompt` | (sunset scene) | Text description |
| `--mode` | standard | standard, fast, hq |
| `--seed` | 42 | Random seed |
| `--num-frames` | 121 | Frame count (must be 8k+1) |
| `--frame-rate` | 24.0 | Playback fps |
| `--height` | 0 (auto) | Output height (divisible by 64) |
| `--width` | 0 (auto) | Output width (divisible by 64) |
| `--num-inference-steps` | 0 (auto) | Denoising steps |
| `--num-videos` | 1 | Batch count (increments seed) |
| `--image` | "" | Path to conditioning image |
| `--enhance-prompt` | false | Use Gemma to enhance the prompt |

### Duration Control

Duration = `num_frames / frame_rate`. Frames must follow `8k + 1` format.

| num_frames | @24fps | Tested (fast) |
|-----------|--------|---------------|
| 49 | 2.0s | OK |
| 73 | 3.0s | — |
| 97 | 4.0s | — |
| 121 | 5.0s | OK |
| 145 | 6.0s | — |
| 169 | 7.0s | — |
| 193 | 8.0s | OK |
| 217 | 9.0s | — |
| 241 | 10.0s | OK |

### Guidance Parameters (standard/hq modes)

| Parameter | Standard | HQ | Effect |
|-----------|----------|-----|--------|
| `cfg_scale` | 3.0 | 3.0 | Prompt adherence |
| `stg_scale` | 1.0 | 0.0 | Spatio-temporal guidance |
| `rescale_scale` | 0.7 | 0.45 | Variance rescaling |
| `stg_blocks` | [28] | [] | Transformer blocks for STG |
| Audio `cfg_scale` | 7.0 | 7.0 | Audio prompt adherence |

Fast mode uses distilled inference — no guidance parameters needed.

### Mode-Specific Defaults

| Setting | standard | fast | hq |
|---------|----------|------|-----|
| Resolution | 1024x1536 | 1024x1536 | 1088x1920 |
| Steps | 30 | 8+4 (fixed) | 15 |
| Negative prompt | yes | no | yes |
| CFG guidance | yes | no | yes |
| STG guidance | yes | no | no |

## Image-to-Video

Pass `--image input.png` on CLI (or `image_bytes=...` in Python). The image conditions the first frame:

```python
result = ltx.generate.remote(
    prompt="A timelapse of a flower blooming",
    image_bytes=open("flower.png", "rb").read(),
    image_strength=1.0,  # 0.0–1.0
)
```

## Output Volume

Videos auto-save to `ltx-outputs` with timestamped filenames and JSON metadata.

```bash
modal volume ls ltx-outputs
modal volume get ltx-outputs <filename> ./local.mp4
```

## Web API

After `uv run modal deploy generate_video.py`:

- **`GET /api_generate?prompt=...&mode=fast&num_frames=121`** → MP4
- **`GET /api_list`** → JSON list of saved videos

Swagger docs at endpoint URL + `/docs`.

## LTX-2 Codebase Structure

The [LTX-2 repo](https://github.com/Lightricks/LTX-2) is a monorepo:

```
packages/
  ltx-core/       — Core model (22B DiT transformer, Video/Audio VAEs, Gemma text encoder)
  ltx-pipelines/  — 8 pipeline implementations (the inference entry points)
  ltx-trainer/    — Training/fine-tuning (LoRA, IC-LoRA, full)
```

### Core Model Architecture

- **Transformer**: 48 blocks, asymmetric dual-stream (14B video + 5B audio + 3B cross-modal)
- **Video VAE**: 32x spatial, 8x temporal compression → 128-channel latents
- **Audio VAE**: Mel spectrogram → latent → HiFi-GAN vocoder (24kHz stereo)
- **Text encoder**: Gemma 3 12B with multi-layer feature extraction

### Available Pipelines in Repo

| Pipeline | File | Description |
|----------|------|-------------|
| TI2VidTwoStagesPipeline | `ti2vid_two_stages.py` | Production text/image-to-video (recommended) |
| TI2VidTwoStagesHQPipeline | `ti2vid_two_stages_hq.py` | Res_2s sampler, fewer steps |
| TI2VidOneStagePipeline | `ti2vid_one_stage.py` | Single-stage, no upsampling |
| DistilledPipeline | `distilled.py` | Fastest, 8 fixed sigmas |
| ICLoraPipeline | `ic_lora.py` | Video-to-video control (pose, depth, etc.) |
| KeyframeInterpolationPipeline | `keyframe_interpolation.py` | Interpolate between images |
| A2VidPipelineTwoStage | `a2vid_two_stage.py` | Audio-driven video |
| RetakePipeline | `retake.py` | Edit time regions of existing video |

### Not Yet Implemented in Our App

- **ICLoraPipeline** — Needs separate IC-LoRA model files + control signal videos. Could add later.
- **TI2VidOneStagePipeline** — Lower quality than two-stage, no upsampling.
- **Temporal upscaler** — `ltx-2.3-temporal-upscaler-x2-1.0.safetensors` is available on HF. Doubles frame count post-generation (121f@24fps → 241f@48fps for smoother playback).

## Issues Encountered & Fixes

### 1. Lazy VAE decode outside torch.no_grad() (CRITICAL)

Pipeline returns lazy iterator. Actual VAE conv3d runs when encode_video consumes it. Must wrap encode_video in no_grad too, otherwise autograd stores all intermediates → OOM at 138GB.

### 2. Resolution semantics wrong

Two-stage pipeline height/width = final output, not stage 1. Was passing 512x768 (stage 1 at 256x384). Fixed to 1024x1536 (stage 1 at 512x768).

### 3. `transformers` version incompatibility

`transformers>=5.0` breaks `Gemma3TextConfig`. Pin to `transformers>=4.52,<5`.

### 4. CUDA Error 802 — "system not yet initialized"

GPU driver not ready at container start. Fixed with `CUDA_MODULE_LOADING=LAZY` + retry loop (10 attempts, 2s sleep).

### 5. Build system (`uv_build`) — bypassed

`ltx-core` and `ltx-pipelines` use `uv_build` which isn't pip-installable. Set `PYTHONPATH` to source directories instead.

### 6. `@modal.web_endpoint` deprecated

Renamed to `@modal.fastapi_endpoint` in Modal >= 1.3.5. Requires `fastapi[standard]` explicitly.

## Next Step: Persistent Models (Eliminate ~55s Reload Overhead)

### The Problem

Currently, every `generate()` call re-reads 46GB from disk, rebuilds the model, applies LoRA, quantizes to FP8, and moves to GPU. This takes ~55s — often more than the actual diffusion. The pipeline classes are designed for one-shot CLI usage and don't support keeping models warm.

### The Plan

Keep both transformers + Gemma + upscaler resident on GPU in `@modal.enter()`. Write a custom inference loop that reuses them instead of going through the pipeline classes.

**VRAM budget (H200 141GB):**

| Component | VRAM (FP8) | Notes |
|-----------|-----------|-------|
| Dev transformer | ~22GB | For standard/hq/a2vid/keyframe/retake |
| Distilled transformer | ~22GB | For fast mode |
| Gemma 3 12B | ~24GB | Text encoder, shared |
| Spatial upscaler | ~1GB | Shared |
| **Always resident** | **~69GB** | 49% of H200 |
| **Remaining for inference** | **~72GB** | Diffusion activations + VAE decode |

72GB free is plenty — our 20s video at 1024x1536 already works in the current setup.

**Expected result:** First call loads everything (~90s cold start). Every subsequent call on a warm container: ~0s model loading → just diffusion + decode time.

| Run | Current (reload each call) | After fix (persistent) |
|-----|---------------------------|----------------------|
| fast, 49f | 58s (7s compute + 51s load) | ~7s |
| fast, 241f | 97s (42s compute + 55s load) | ~42s |
| fast, 481f | 159s (100s compute + 59s load) | ~100s |

### Implementation Approach

The pipeline source code in `ltx-core` and `ltx-pipelines` shows exactly what each pipeline does internally. The core loop is:

1. Encode text with Gemma → get video/audio context embeddings
2. Create initial noise + conditioning
3. Run denoising loop: `for sigma in sigmas: video_state = stepper(transformer(video_state))`
4. Upsample latent with spatial upscaler
5. Run stage 2 denoising loop (same transformer, different LoRA/sigmas)
6. Decode with video VAE + audio VAE + vocoder

We replicate this logic but with persistent model references instead of load-use-delete cycles. The building blocks are all in `ltx-core`: schedulers, guiders, noisers, patchifiers, denoising loops.

### Other Future Work

- **Memory snapshots** — `enable_memory_snapshot=True` on Modal to cut cold start from ~90s to ~10-30s
- **IC-LoRA** — Video-to-video control (pose, depth, style). Needs IC-LoRA model files + distilled checkpoint.
- **Temporal upscaler** — `ltx-2.3-temporal-upscaler-x2-1.0.safetensors` doubles frame count post-generation (smoother playback)
- **1.5x spatial upscaler** — `ltx-2.3-spatial-upscaler-x1.5-1.0.safetensors` for lower VRAM than 2x

## References

- LTX-2.3 HuggingFace: https://huggingface.co/Lightricks/LTX-2.3
- LTX-2.3 render times discussion: https://huggingface.co/Lightricks/LTX-2.3/discussions/16
- LTX-2 GitHub: https://github.com/Lightricks/LTX-2
- LTX-2 pipelines README: https://github.com/Lightricks/LTX-2/blob/main/packages/ltx-pipelines/README.md
- Modal GPU guide: https://modal.com/docs/guide/gpu
- Modal model weights: https://modal.com/docs/guide/model-weights
- Modal memory snapshots: https://modal.com/docs/guide/memory-snapshots
