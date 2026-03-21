"""
LTX-2.3 Video Generation on Modal (H200, FP8).

Modes:
  standard — 30-step diffusion + 2x upscale, best quality (1024x1536)
  fast     — distilled 8+4 steps, ~2x faster (1024x1536)
  hq       — res_2s sampler, 15 steps, 1080p (1088x1920)

Features:
  text-to-video | image-to-video | audio-to-video
  keyframe interpolation | video retake (edit time regions)

Usage:
    uv run modal run generate_video.py --prompt "A cat sitting on a windowsill"
    uv run modal run generate_video.py --prompt "..." --mode fast
    uv run modal run generate_video.py --prompt "..." --mode hq
    uv run modal run generate_video.py --prompt "..." --num-frames 241      # 10s
    uv run modal run generate_video.py --prompt "..." --image photo.jpg     # image-to-video
    uv run modal run generate_video.py --prompt "..." --enhance-prompt      # auto-enhance
    uv run modal deploy generate_video.py                                   # web API

See docs.md for full parameter reference, prompting guide, and Python API examples.
"""

import modal

app = modal.App("ltx-video")

model_volume = modal.Volume.from_name("ltx-models", create_if_missing=True)
output_volume = modal.Volume.from_name("ltx-outputs", create_if_missing=True)
MODEL_DIR = "/models"
OUTPUT_DIR = "/outputs"

LTX_DIR = f"{MODEL_DIR}/ltx"
GEMMA_DIR = f"{MODEL_DIR}/gemma"

LTX_FILES = [
    "ltx-2.3-22b-dev.safetensors",
    "ltx-2.3-22b-distilled.safetensors",
    "ltx-2.3-22b-distilled-lora-384.safetensors",
    "ltx-2.3-spatial-upscaler-x2-1.0.safetensors",
]

hf_secret = modal.Secret.from_name("huggingface-secret")

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git", "ffmpeg")
    .run_commands(
        "git clone https://github.com/Lightricks/LTX-2.git /ltx2",
        "pip install uv-build",
        "pip install -e /ltx2/packages/ltx-core -e /ltx2/packages/ltx-pipelines 'transformers>=4.52,<5'",
    )
    .uv_pip_install(
        "xformers>=0.0.30",
        "huggingface_hub",
        "sentencepiece",
        "protobuf",
        "fastapi[standard]",
    )
    .env(
        {
            "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
            "CUDA_MODULE_LOADING": "LAZY",
        }
    )
)


def _snap_frames(n: int) -> int:
    """Round up to nearest valid frame count (8k+1)."""
    if (n - 1) % 8 != 0:
        n = ((n - 1) // 8 + 1) * 8 + 1
    return n


# ---------------------------------------------------------------------------
# Main class — all pipelines, one container
# ---------------------------------------------------------------------------


@app.cls(
    image=image,
    gpu="H200",
    volumes={MODEL_DIR: model_volume, OUTPUT_DIR: output_volume},
    timeout=30 * 60,
    secrets=[hf_secret],
    scaledown_window=15 * 60,
)
class LTXVideo:
    @modal.enter()
    def setup(self):
        import time

        import torch

        for attempt in range(10):
            try:
                torch.cuda.init()
                break
            except RuntimeError:
                if attempt < 9:
                    time.sleep(2)
                else:
                    raise
        torch.cuda.set_device(0)
        torch.set_float32_matmul_precision("high")
        print(f"GPU: {torch.cuda.get_device_name(0)}")

        self._ensure_models()
        self._create_pipelines()
        print("All pipelines ready.")

    def _ensure_models(self):
        import os
        from pathlib import Path

        from huggingface_hub import hf_hub_download, snapshot_download

        os.makedirs(LTX_DIR, exist_ok=True)
        for f in LTX_FILES:
            if not (Path(LTX_DIR) / f).exists():
                print(f"Downloading {f}...")
                hf_hub_download("Lightricks/LTX-2.3", f, local_dir=LTX_DIR)

        if not Path(GEMMA_DIR).exists() or not any(
            Path(GEMMA_DIR).glob("*.safetensors")
        ):
            print("Downloading Gemma 3 12B text encoder...")
            snapshot_download(
                "google/gemma-3-12b-it-qat-q4_0-unquantized", local_dir=GEMMA_DIR
            )

        model_volume.commit()

    def _create_pipelines(self):
        """Create pipeline objects — these are lightweight config holders.
        Actual model weights load on-demand during generation."""
        from ltx_core.loader import (
            LTXV_LORA_COMFY_RENAMING_MAP,
            LoraPathStrengthAndSDOps,
        )
        from ltx_core.quantization import QuantizationPolicy

        quant = QuantizationPolicy.fp8_cast()
        dev_ckpt = f"{LTX_DIR}/ltx-2.3-22b-dev.safetensors"
        dist_ckpt = f"{LTX_DIR}/ltx-2.3-22b-distilled.safetensors"
        upscaler = f"{LTX_DIR}/ltx-2.3-spatial-upscaler-x2-1.0.safetensors"
        dist_lora = [
            LoraPathStrengthAndSDOps(
                f"{LTX_DIR}/ltx-2.3-22b-distilled-lora-384.safetensors",
                0.6,
                LTXV_LORA_COMFY_RENAMING_MAP,
            )
        ]

        shared = dict(gemma_root=GEMMA_DIR, quantization=quant)
        two_stage = dict(
            checkpoint_path=dev_ckpt,
            distilled_lora=dist_lora,
            spatial_upsampler_path=upscaler,
            loras=[],
            **shared,
        )

        from ltx_pipelines.a2vid_two_stage import A2VidPipelineTwoStage
        from ltx_pipelines.distilled import DistilledPipeline
        from ltx_pipelines.keyframe_interpolation import KeyframeInterpolationPipeline
        from ltx_pipelines.retake import RetakePipeline
        from ltx_pipelines.ti2vid_two_stages import TI2VidTwoStagesPipeline
        from ltx_pipelines.ti2vid_two_stages_hq import TI2VidTwoStagesHQPipeline

        self._pipelines = {
            "standard": TI2VidTwoStagesPipeline(**two_stage),
            "hq": TI2VidTwoStagesHQPipeline(
                checkpoint_path=dev_ckpt,
                distilled_lora=dist_lora,
                distilled_lora_strength_stage_1=0.25,
                distilled_lora_strength_stage_2=0.5,
                spatial_upsampler_path=upscaler,
                loras=(),
                **shared,
            ),
            "fast": DistilledPipeline(
                distilled_checkpoint_path=dist_ckpt,
                spatial_upsampler_path=upscaler,
                loras=[],
                **shared,
            ),
            "a2vid": A2VidPipelineTwoStage(**two_stage),
            "keyframe": KeyframeInterpolationPipeline(**two_stage),
            "retake": RetakePipeline(
                checkpoint_path=dev_ckpt, loras=[], **shared
            ),
        }

    # -------------------------------------------------------------------
    # Helpers
    # -------------------------------------------------------------------

    def _prep_images(self, image_bytes: bytes | None, strength: float = 1.0):
        """Convert raw image bytes to ImageConditioningInput list."""
        if image_bytes is None:
            return []
        import tempfile

        from ltx_pipelines.utils.args import ImageConditioningInput

        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            f.write(image_bytes)
            path = f.name
        return [ImageConditioningInput(path, 0, strength, 33)]

    def _prep_images_multi(self, images_data: list[tuple[bytes, int, float]]):
        """Convert list of (bytes, frame_idx, strength) to ImageConditioningInput list."""
        import tempfile

        from ltx_pipelines.utils.args import ImageConditioningInput

        result = []
        for img_bytes, frame_idx, strength in images_data:
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
                f.write(img_bytes)
                path = f.name
            result.append(ImageConditioningInput(path, frame_idx, strength, 33))
        return result

    def _write_temp(self, data: bytes, suffix: str) -> str:
        import tempfile

        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as f:
            f.write(data)
            return f.name

    def _encode_result(
        self, video, audio, num_frames, frame_rate, prompt, seed, save_name=None
    ):
        """Encode video+audio to MP4, save to volume, return result dict.

        IMPORTANT: video is a lazy iterator — the VAE decode happens here when
        encode_video consumes it. We must stay inside no_grad/inference_mode.
        """
        import torch
        from ltx_core.model.video_vae import TilingConfig, get_video_chunks_number
        from ltx_pipelines.utils.media_io import encode_video

        tiling = TilingConfig.default()
        chunks = get_video_chunks_number(num_frames, tiling)

        path = "/tmp/output.mp4"
        with torch.no_grad():
            encode_video(
                video=video,
                fps=frame_rate,
                audio=audio,
                output_path=path,
                video_chunks_number=chunks,
            )

        with open(path, "rb") as f:
            video_bytes = f.read()

        filename = self._save(
            video_bytes, prompt, seed, num_frames / frame_rate, save_name
        )
        return {
            "video_bytes": video_bytes,
            "filename": filename,
            "duration": round(num_frames / frame_rate, 2),
            "size_mb": round(len(video_bytes) / 1024 / 1024, 2),
        }

    def _save(self, video_bytes, prompt, seed, duration, name=None):
        import json
        import os
        from datetime import datetime, timezone

        os.makedirs(OUTPUT_DIR, exist_ok=True)

        if name:
            fn = name if name.endswith(".mp4") else f"{name}.mp4"
        else:
            ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
            safe = (
                "".join(c if c.isalnum() or c in " -_" else "" for c in prompt)[:50]
                .strip()
                .replace(" ", "_")
            )
            fn = f"{ts}_{safe}_s{seed}.mp4"

        with open(f"{OUTPUT_DIR}/{fn}", "wb") as f:
            f.write(video_bytes)

        with open(f"{OUTPUT_DIR}/{fn}.json", "w") as f:
            json.dump(
                {
                    "prompt": prompt,
                    "seed": seed,
                    "duration": round(duration, 2),
                    "size_mb": round(len(video_bytes) / 1024 / 1024, 2),
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                },
                f,
                indent=2,
            )

        output_volume.commit()
        return fn

    def _video_guider(self, cfg_scale, stg_scale, rescale_scale):
        from ltx_core.components.guiders import MultiModalGuiderParams

        return MultiModalGuiderParams(
            cfg_scale=cfg_scale,
            stg_scale=stg_scale,
            rescale_scale=rescale_scale,
            modality_scale=3.0,
            stg_blocks=[28] if stg_scale > 0 else [],
        )

    def _audio_guider(self, stg_scale, rescale_scale):
        from ltx_core.components.guiders import MultiModalGuiderParams

        return MultiModalGuiderParams(
            cfg_scale=7.0,
            stg_scale=stg_scale,
            rescale_scale=rescale_scale,
            modality_scale=3.0,
            stg_blocks=[28] if stg_scale > 0 else [],
        )

    # -------------------------------------------------------------------
    # Generation methods
    # -------------------------------------------------------------------

    @modal.method()
    def generate(
        self,
        prompt: str,
        mode: str = "standard",
        negative_prompt: str = "",
        seed: int = 42,
        height: int | None = None,
        width: int | None = None,
        num_frames: int = 121,
        frame_rate: float = 24.0,
        num_inference_steps: int | None = None,
        cfg_scale: float = 3.0,
        stg_scale: float | None = None,
        rescale_scale: float | None = None,
        image_bytes: bytes | None = None,
        image_strength: float = 1.0,
        enhance_prompt: bool = False,
    ) -> dict:
        """Text/image-to-video generation.

        mode: standard (30 steps), fast (8+4 steps), hq (15 steps, res_2s sampler)
        height/width: final output resolution (divisible by 64).
            Defaults — standard/fast: 1024x1536, hq: 1088x1920
        """
        import time

        import torch
        from ltx_core.model.video_vae import TilingConfig

        num_frames = _snap_frames(num_frames)
        images = self._prep_images(image_bytes, image_strength)
        tiling = TilingConfig.default()

        # Mode-specific defaults
        if mode == "hq":
            height = height or 1088
            width = width or 1920
            num_inference_steps = num_inference_steps or 15
            stg_scale = stg_scale if stg_scale is not None else 0.0
            rescale_scale = rescale_scale if rescale_scale is not None else 0.45
        else:
            height = height or 1024
            width = width or 1536
            num_inference_steps = num_inference_steps or 30
            stg_scale = stg_scale if stg_scale is not None else 1.0
            rescale_scale = rescale_scale if rescale_scale is not None else 0.7

        assert height % 64 == 0 and width % 64 == 0, (
            f"height/width must be divisible by 64 for two-stage, got {height}x{width}"
        )

        dur = num_frames / frame_rate
        print(
            f"generate [{mode}] {dur:.1f}s ({num_frames}f) "
            f"{width}x{height} steps={num_inference_steps} seed={seed}"
        )
        print(f"  prompt: {prompt[:120]}")

        t0 = time.time()

        with torch.no_grad():
            if mode == "fast":
                video, audio = self._pipelines["fast"](
                    prompt=prompt,
                    seed=seed,
                    height=height,
                    width=width,
                    num_frames=num_frames,
                    frame_rate=frame_rate,
                    images=images,
                    tiling_config=tiling,
                    enhance_prompt=enhance_prompt,
                )
            else:
                pipeline = self._pipelines[mode]
                video, audio = pipeline(
                    prompt=prompt,
                    negative_prompt=negative_prompt,
                    seed=seed,
                    height=height,
                    width=width,
                    num_frames=num_frames,
                    frame_rate=frame_rate,
                    num_inference_steps=num_inference_steps,
                    video_guider_params=self._video_guider(
                        cfg_scale, stg_scale, rescale_scale
                    ),
                    audio_guider_params=self._audio_guider(
                        stg_scale, rescale_scale
                    ),
                    images=images,
                    tiling_config=tiling,
                    enhance_prompt=enhance_prompt,
                )

        gen_time = time.time() - t0
        result = self._encode_result(
            video, audio, num_frames, frame_rate, prompt, seed
        )
        result["mode"] = mode
        result["gen_time_s"] = round(gen_time, 1)
        print(
            f"  done in {gen_time:.0f}s | {result['size_mb']} MB | {result['filename']}"
        )
        return result

    @modal.method()
    def generate_from_audio(
        self,
        prompt: str,
        audio_bytes: bytes,
        negative_prompt: str = "",
        seed: int = 42,
        height: int = 1024,
        width: int = 1536,
        num_frames: int = 121,
        frame_rate: float = 24.0,
        num_inference_steps: int = 30,
        cfg_scale: float = 3.0,
        stg_scale: float = 1.0,
        rescale_scale: float = 0.7,
        image_bytes: bytes | None = None,
        image_strength: float = 1.0,
        audio_start_time: float = 0.0,
        audio_max_duration: float | None = None,
        enhance_prompt: bool = False,
    ) -> dict:
        """Audio-driven video generation. Pass audio bytes (wav/mp3) to condition video."""
        import time

        import torch
        from ltx_core.model.video_vae import TilingConfig

        num_frames = _snap_frames(num_frames)
        assert height % 64 == 0 and width % 64 == 0

        audio_path = self._write_temp(audio_bytes, ".wav")

        # A2Vid uses list[tuple[str, int, float]] for images, not ImageConditioningInput
        images = []
        if image_bytes is not None:
            img_path = self._write_temp(image_bytes, ".png")
            images = [(img_path, 0, image_strength)]

        print(
            f"audio-to-video {num_frames / frame_rate:.1f}s "
            f"{width}x{height} seed={seed}"
        )
        t0 = time.time()

        with torch.no_grad():
            video, audio = self._pipelines["a2vid"](
                prompt=prompt,
                negative_prompt=negative_prompt,
                seed=seed,
                height=height,
                width=width,
                num_frames=num_frames,
                frame_rate=frame_rate,
                num_inference_steps=num_inference_steps,
                video_guider_params=self._video_guider(
                    cfg_scale, stg_scale, rescale_scale
                ),
                images=images,
                audio_path=audio_path,
                audio_start_time=audio_start_time,
                audio_max_duration=audio_max_duration,
                tiling_config=TilingConfig.default(),
                enhance_prompt=enhance_prompt,
            )

        gen_time = time.time() - t0
        result = self._encode_result(
            video, audio, num_frames, frame_rate, prompt, seed
        )
        result["mode"] = "a2vid"
        result["gen_time_s"] = round(gen_time, 1)
        print(f"  done in {gen_time:.0f}s | {result['filename']}")
        return result

    @modal.method()
    def interpolate(
        self,
        prompt: str,
        keyframe_images: list[tuple[bytes, int, float]],
        negative_prompt: str = "",
        seed: int = 42,
        height: int = 1024,
        width: int = 1536,
        num_frames: int = 121,
        frame_rate: float = 24.0,
        num_inference_steps: int = 30,
        cfg_scale: float = 3.0,
        stg_scale: float = 1.0,
        rescale_scale: float = 0.7,
        enhance_prompt: bool = False,
    ) -> dict:
        """Interpolate between keyframe images.

        keyframe_images: list of (image_bytes, frame_idx, strength) tuples.
        Example: [(start_img, 0, 1.0), (end_img, 120, 1.0)]
        """
        import time

        import torch
        from ltx_core.model.video_vae import TilingConfig

        num_frames = _snap_frames(num_frames)
        assert height % 64 == 0 and width % 64 == 0
        images = self._prep_images_multi(keyframe_images)

        print(f"keyframe interpolation {num_frames / frame_rate:.1f}s seed={seed}")
        t0 = time.time()

        with torch.no_grad():
            video, audio = self._pipelines["keyframe"](
                prompt=prompt,
                negative_prompt=negative_prompt,
                seed=seed,
                height=height,
                width=width,
                num_frames=num_frames,
                frame_rate=frame_rate,
                num_inference_steps=num_inference_steps,
                video_guider_params=self._video_guider(
                    cfg_scale, stg_scale, rescale_scale
                ),
                audio_guider_params=self._audio_guider(stg_scale, rescale_scale),
                images=images,
                tiling_config=TilingConfig.default(),
                enhance_prompt=enhance_prompt,
            )

        gen_time = time.time() - t0
        result = self._encode_result(
            video, audio, num_frames, frame_rate, prompt, seed
        )
        result["mode"] = "keyframe"
        result["gen_time_s"] = round(gen_time, 1)
        print(f"  done in {gen_time:.0f}s | {result['filename']}")
        return result

    @modal.method()
    def retake(
        self,
        video_bytes: bytes,
        prompt: str,
        start_time: float,
        end_time: float,
        seed: int = 42,
        negative_prompt: str = "",
        num_inference_steps: int = 40,
        cfg_scale: float = 3.0,
        stg_scale: float = 1.0,
        rescale_scale: float = 0.7,
        regenerate_video: bool = True,
        regenerate_audio: bool = True,
        enhance_prompt: bool = False,
    ) -> dict:
        """Regenerate a time region [start_time, end_time] of an existing video."""
        import time

        import torch
        from ltx_core.model.video_vae import TilingConfig

        video_path = self._write_temp(video_bytes, ".mp4")

        print(
            f"retake [{start_time:.1f}s - {end_time:.1f}s] "
            f"video={regenerate_video} audio={regenerate_audio} seed={seed}"
        )
        t0 = time.time()

        with torch.no_grad():
            video, audio_tensor = self._pipelines["retake"](
                video_path=video_path,
                prompt=prompt,
                start_time=start_time,
                end_time=end_time,
                seed=seed,
                negative_prompt=negative_prompt,
                num_inference_steps=num_inference_steps,
                video_guider_params=self._video_guider(
                    cfg_scale, stg_scale, rescale_scale
                ),
                audio_guider_params=self._audio_guider(stg_scale, rescale_scale),
                regenerate_video=regenerate_video,
                regenerate_audio=regenerate_audio,
                enhance_prompt=enhance_prompt,
                tiling_config=TilingConfig.default(),
            )

        # Retake uses source video's frame count/rate
        from ltx_pipelines.utils.media_io import get_videostream_metadata

        fps, num_frames, _w, _h = get_videostream_metadata(video_path)
        frame_rate = fps

        gen_time = time.time() - t0
        result = self._encode_result(
            video, audio_tensor, num_frames, frame_rate, prompt, seed
        )
        result["mode"] = "retake"
        result["gen_time_s"] = round(gen_time, 1)
        print(f"  done in {gen_time:.0f}s | {result['filename']}")
        return result

    @modal.method()
    def list_outputs(self) -> list[dict]:
        """List all videos on the output volume."""
        import json
        from pathlib import Path

        out = Path(OUTPUT_DIR)
        if not out.exists():
            return []

        videos = []
        for mp4 in sorted(out.glob("*.mp4")):
            entry = {
                "filename": mp4.name,
                "size_mb": round(mp4.stat().st_size / 1024 / 1024, 2),
            }
            meta_path = Path(f"{mp4}.json")
            if meta_path.exists():
                with open(meta_path) as f:
                    entry["metadata"] = json.load(f)
            videos.append(entry)
        return videos

    # -------------------------------------------------------------------
    # Web API
    # -------------------------------------------------------------------

    @modal.fastapi_endpoint(docs=True)
    def api_generate(
        self,
        prompt: str,
        mode: str = "standard",
        seed: int = 42,
        height: int | None = None,
        width: int | None = None,
        num_frames: int = 121,
        frame_rate: float = 24.0,
        num_inference_steps: int | None = None,
        enhance_prompt: bool = False,
    ):
        """GET endpoint — returns MP4 video."""
        from fastapi.responses import Response

        result = self.generate.local(
            prompt=prompt,
            mode=mode,
            seed=seed,
            height=height,
            width=width,
            num_frames=num_frames,
            frame_rate=frame_rate,
            num_inference_steps=num_inference_steps,
            enhance_prompt=enhance_prompt,
        )
        return Response(
            content=result["video_bytes"],
            media_type="video/mp4",
            headers={
                "Content-Disposition": f'attachment; filename="{result["filename"]}"'
            },
        )

    @modal.fastapi_endpoint(docs=True)
    def api_list(self):
        """GET endpoint — list generated videos."""
        return self.list_outputs.local()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


@app.local_entrypoint()
def main(
    prompt: str = "A cinematic shot of a golden sunset over a calm ocean, with gentle waves reflecting warm light and seabirds gliding overhead",
    mode: str = "standard",
    seed: int = 42,
    num_frames: int = 121,
    frame_rate: float = 24.0,
    height: int = 0,
    width: int = 0,
    num_inference_steps: int = 0,
    num_videos: int = 1,
    image: str = "",
    enhance_prompt: bool = False,
):
    """Generate video(s) from text using LTX-2.3 on H200."""
    num_frames = _snap_frames(num_frames)
    print(f"Mode: {mode} | {num_frames / frame_rate:.1f}s | seed={seed}")

    image_bytes = None
    if image:
        with open(image, "rb") as f:
            image_bytes = f.read()
        print(f"Image conditioning: {image}")

    ltx = LTXVideo()

    kwargs = dict(
        prompt=prompt,
        mode=mode,
        seed=seed,
        num_frames=num_frames,
        frame_rate=frame_rate,
        image_bytes=image_bytes,
        enhance_prompt=enhance_prompt,
        height=height or None,
        width=width or None,
        num_inference_steps=num_inference_steps or None,
    )

    for i in range(num_videos):
        kwargs["seed"] = seed + i
        result = ltx.generate.remote(**kwargs)
        fname = f"output_{i}.mp4" if num_videos > 1 else "output.mp4"
        with open(fname, "wb") as f:
            f.write(result["video_bytes"])
        print(
            f"Saved: {fname} ({result['duration']}s, {result['size_mb']} MB, "
            f"{result['gen_time_s']}s gen)"
        )
    print("Done.")
