# LTX-2.3 on Modal — Notes

## What We Built

A Modal app (`generate_video.py`) that runs the **Lightricks LTX-2.3 (22B parameter)** text-to-video model on an **NVIDIA H100 80GB** GPU with FP8 quantization. It generates ~5 second videos (121 frames, 768x512, 25fps) from text prompts.

```bash
uv run modal run generate_video.py --prompt "A cat sitting on a windowsill"
```

## Architecture

```
Local machine                    Modal Cloud
─────────────                    ───────────
generate_video.py  ──────────►  download_models()     [no GPU, 2h timeout]
  (uv run modal run)                │ downloads ~80GB to Volume
                                    ▼
                               generate()             [H100 80GB, 30min timeout]
                                    │ loads pipeline (FP8)
                                    │ runs 40-step diffusion
                                    │ encodes to MP4
                                    ▼
output.mp4  ◄──────────────────  returns video bytes
```

### Model Components (~80GB total on Volume)

| File | Size | Source |
|------|------|--------|
| `ltx-2.3-22b-dev.safetensors` | 46.1 GB | `Lightricks/LTX-2.3` |
| `ltx-2.3-22b-distilled-lora-384.safetensors` | 7.6 GB | `Lightricks/LTX-2.3` |
| `ltx-2.3-spatial-upscaler-x2-1.0.safetensors` | 1 GB | `Lightricks/LTX-2.3` |
| Gemma 3 12B text encoder | ~24 GB | `google/gemma-3-12b-it-qat-q4_0-unquantized` |

### Pipeline: Two-Stage (`TI2VidTwoStagesPipeline`)

1. **Stage 1**: Generates low-res latent video using the dev checkpoint + distilled LoRA (40 steps)
2. **Stage 2**: Spatial upscaling (3 steps) via the upscaler model
3. **Encode**: VAE decodes latents to pixels, then encodes to H.264 MP4

## Issues Encountered & Fixes

### 1. `output_path` not a valid kwarg

The pipeline's `__call__` returns `(video_iterator, audio)` — it does NOT accept `output_path`. You must call `encode_video()` separately:

```python
from ltx_pipelines.utils.media_io import encode_video
from ltx_core.model.video_vae import TilingConfig, get_video_chunks_number

video, audio = pipeline(prompt=..., ...)

encode_video(
    video=video,
    fps=frame_rate,
    audio=audio,
    output_path="/tmp/output.mp4",
    video_chunks_number=get_video_chunks_number(num_frames, TilingConfig.default()),
)
```

### 2. `transformers` version incompatibility

`transformers>=5.0` breaks `Gemma3TextConfig` — the `rope_local_base_freq` attribute was renamed/removed. **Pin to `transformers>=4.52,<5`** (resolved to 4.57.6).

### 3. CUDA Error 802 — "system not yet initialized"

Intermittent issue where the GPU driver isn't ready when PyTorch first tries to initialize CUDA. Two fixes applied:

- **`CUDA_MODULE_LOADING=LAZY`** env var (defers CUDA module loading)
- **Retry loop** at the start of `generate()`:

```python
for attempt in range(10):
    try:
        torch.cuda.init()
        break
    except RuntimeError:
        if attempt < 9:
            time.sleep(2)
        else:
            raise
```

### 4. `torch.inference_mode()` vs `torch.no_grad()`

The VAE decoder uses operations that conflict with `inference_mode()` (conv3d backward tracking). Using **`torch.no_grad()`** instead works fine. The LTX-2 repo's own `main()` uses `@torch.inference_mode()` but that works because their entry point wraps the entire flow differently.

### 5. Build system (`uv_build`) — avoided entirely

The `ltx-core` and `ltx-pipelines` packages use `uv_build` as their build backend, which isn't pip-installable. Instead of fighting the build system, we just set **`PYTHONPATH`** to point at the source directories:

```python
.env({"PYTHONPATH": "/ltx2/packages/ltx-core/src:/ltx2/packages/ltx-pipelines/src"})
```

## Key Parameters

### Pipeline `__call__` signature

```python
pipeline(
    prompt: str,
    negative_prompt: str,        # "" for none
    seed: int,
    height: int,                 # must be divisible by 32
    width: int,                  # must be divisible by 32
    num_frames: int,             # must be 8k+1 (e.g. 121 = 8*15+1)
    frame_rate: float,
    num_inference_steps: int,    # 40 recommended for two-stage
    video_guider_params: MultiModalGuiderParams,
    audio_guider_params: MultiModalGuiderParams,
    images: list,                # [] for text-to-video
    tiling_config: TilingConfig | None,
) -> tuple[Iterator[torch.Tensor], Audio]
```

### Guidance Parameters

```python
MultiModalGuiderParams(
    cfg_scale=3.0,       # text adherence (1.0 = disabled)
    stg_scale=1.0,       # spatio-temporal coherence
    rescale_scale=0.7,   # variance matching / saturation control
    modality_scale=3.0,  # audio-visual sync
    skip_step=0,
    stg_blocks=[29],     # which transformer blocks for STG
)
```

### Frame Count Constraint

Frames must follow `8k + 1` format: 9, 17, 25, 33, 41, 49, 57, 65, 73, 81, 89, 97, 105, 113, 121, 129, ...

## Performance

- **Diffusion (40 steps)**: ~33s on H100 at 1.2 it/s
- **Upscaling (3 steps)**: ~1.7s
- **VAE decode + encode**: ~3s
- **Total generation**: ~40s (excluding model load)
- **Model load (cold start)**: ~60-120s from Volume
- **Cost**: ~$3.95/hr for H100 (billed per second)

## What To Do Next

### 1. Refactor to `@app.cls` pattern (recommended)

Convert from bare functions to Modal's class pattern. This loads the model once per container and reuses it across requests:

```python
@app.cls(
    image=image,
    gpu="H100",
    volumes={MODEL_DIR: volume},
    timeout=30 * 60,
    scaledown_window=15 * 60,  # keep container warm 15 min
)
class LTXVideo:
    @modal.enter()
    def load_pipeline(self):
        self.pipeline = TI2VidTwoStagesPipeline(...)

    @modal.method()
    def generate(self, prompt: str, seed: int = 42) -> bytes:
        video, audio = self.pipeline(...)
        encode_video(...)
        return open("/tmp/output.mp4", "rb").read()
```

### 2. Add memory snapshots for fast cold starts

```python
@app.cls(
    gpu="H100",
    enable_memory_snapshot=True,
)
class LTXVideo:
    @modal.enter(snap=True)
    def load_to_cpu(self):
        # Load weights to CPU — gets snapshotted
        self.pipeline = TI2VidTwoStagesPipeline(...)

    @modal.enter()
    def move_to_gpu(self):
        # After snapshot restore, just move to GPU (~seconds vs ~minutes)
        self.pipeline.to("cuda")
```

This cuts cold start from ~60-120s to ~10-30s. Requires `modal deploy` (snapshots don't work with `modal run`).

### 3. Try GPU memory snapshots (alpha)

```python
@app.cls(
    gpu="H100",
    enable_memory_snapshot=True,
    experimental_options={"enable_gpu_snapshot": True},
)
```

Could bring cold start to 3-10s. Set `TORCHINDUCTOR_COMPILE_THREADS=1` and `XFORMERS_ENABLE_TRITON=1` env vars.

### 4. Add a web endpoint

```python
@modal.fastapi_endpoint(docs=True)
def web(self, prompt: str, seed: int = 42):
    video_bytes = self.generate.local(prompt, seed)
    return Response(content=video_bytes, media_type="video/mp4")
```

### 5. Image-to-video

The pipeline supports image conditioning:

```python
from ltx_pipelines.utils.args import ImageConditioningInput

pipeline(
    prompt="...",
    images=[ImageConditioningInput("input.jpg", 0, 1.0, 33)],
    ...
)
```

### 6. Higher resolution / longer videos

- **Resolution**: Try 1024x576 or 768x1024 (must be divisible by 32). May need to drop FP8 or use H200 for VRAM.
- **More frames**: 193 (8*24+1) = ~7.7s at 25fps. 257 (8*32+1) = ~10.3s.
- **Temporal upscaler**: `ltx-2.3-temporal-upscaler-x2-1.0.safetensors` (262 MB) can double frame count for smoother motion.

### 7. Distilled pipeline (fastest)

For 8-step generation (~4x faster, lower quality):

```python
from ltx_pipelines.distilled import DistilledPipeline

pipeline = DistilledPipeline(
    checkpoint_path="ltx-2.3-22b-distilled.safetensors",  # 46.1 GB standalone
    gemma_root=gemma_dir,
)
```

### 8. Deploy as always-on service

```bash
uv run modal deploy generate_video.py
```

Add `min_containers=1` to keep a warm container running at all times (~$2,844/mo for H100).

## References

- LTX-2.3 HuggingFace: https://huggingface.co/Lightricks/LTX-2.3
- LTX-2 GitHub: https://github.com/Lightricks/LTX-2
- LTX-2 pipelines README: https://github.com/Lightricks/LTX-2/blob/main/packages/ltx-pipelines/README.md
- Modal GPU guide: https://modal.com/docs/guide/gpu
- Modal model weights: https://modal.com/docs/guide/model-weights
- Modal cold start: https://modal.com/docs/guide/cold-start
- Modal memory snapshots: https://modal.com/docs/guide/memory-snapshots
- Modal LTX example (older model): https://modal.com/docs/examples/ltx
