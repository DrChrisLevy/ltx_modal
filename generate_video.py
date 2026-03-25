"""
LTX-2.3 Video Generation on Modal — Parametrized by Mode.

Each (mode, precision) combination runs in its own container pool,
loading only the models that mode requires.

Modes:
  standard — 30-step diffusion + 2x upscale, best quality (1024x1536)
  fast     — distilled 8+4 steps, ~2x faster (1024x1536)
  hq       — res_2s sampler, 15 steps, 1080p (1088x1920)
  a2vid    — audio-conditioned video generation (1024x1536)
  keyframe — interpolation between keyframe images (1024x1536)
  retake   — regenerate a time region of existing video
"""

import modal

app = modal.App("ltx-video")

model_volume = modal.Volume.from_name("ltx-models", create_if_missing=True)
output_volume = modal.Volume.from_name("ltx-outputs", create_if_missing=True)
MODEL_DIR = "/models"
OUTPUT_DIR = "/outputs"

LTX_DIR = f"{MODEL_DIR}/ltx"
GEMMA_DIR = f"{MODEL_DIR}/gemma"


hf_secret = modal.Secret.from_dotenv()

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
# Parametrized class — each (mode, precision) gets its own container pool
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
    mode: str = modal.parameter()
    precision: str = modal.parameter(default="bf16")

    @modal.enter()
    def setup(self):
        import torch

        torch.set_float32_matmul_precision("high")
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"Mode: {self.mode} | Precision: {self.precision}")

        self._ensure_models()
        self._create_pipeline()
        self._load_persistent_models()

        vram = torch.cuda.memory_allocated() / 1024**3
        print(f"Ready. VRAM used by persistent models: {vram:.1f} GB")

    def _ensure_models(self):
        import os
        from pathlib import Path

        from huggingface_hub import hf_hub_download, snapshot_download

        os.makedirs(LTX_DIR, exist_ok=True)
        need_dev = self.mode in ("standard", "hq", "a2vid", "keyframe", "retake")
        need_dist = self.mode == "fast"
        need_lora = self.mode in ("standard", "hq", "a2vid", "keyframe")
        need_upscaler = self.mode != "retake"

        files = []
        if need_dev:
            files.append("ltx-2.3-22b-dev.safetensors")
        if need_dist:
            files.append("ltx-2.3-22b-distilled.safetensors")
        if need_lora:
            files.append("ltx-2.3-22b-distilled-lora-384.safetensors")
        if need_upscaler:
            files.append("ltx-2.3-spatial-upscaler-x2-1.0.safetensors")

        for f in files:
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

    def _create_pipeline(self):
        """Create only the pipeline for self.mode."""
        from ltx_core.loader import (
            LTXV_LORA_COMFY_RENAMING_MAP,
            LoraPathStrengthAndSDOps,
        )
        from ltx_core.quantization import QuantizationPolicy

        quant = QuantizationPolicy.fp8_cast() if self.precision == "fp8" else None
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

        if self.mode == "standard":
            from ltx_pipelines.ti2vid_two_stages import TI2VidTwoStagesPipeline

            self._pipeline = TI2VidTwoStagesPipeline(**two_stage)
        elif self.mode == "fast":
            from ltx_pipelines.distilled import DistilledPipeline

            self._pipeline = DistilledPipeline(
                distilled_checkpoint_path=dist_ckpt,
                spatial_upsampler_path=upscaler,
                loras=[],
                **shared,
            )
        elif self.mode == "hq":
            from ltx_pipelines.ti2vid_two_stages_hq import TI2VidTwoStagesHQPipeline

            self._pipeline = TI2VidTwoStagesHQPipeline(
                checkpoint_path=dev_ckpt,
                distilled_lora=dist_lora,
                distilled_lora_strength_stage_1=0.25,
                distilled_lora_strength_stage_2=0.5,
                spatial_upsampler_path=upscaler,
                loras=(),
                **shared,
            )
        elif self.mode == "a2vid":
            from ltx_pipelines.a2vid_two_stage import A2VidPipelineTwoStage

            self._pipeline = A2VidPipelineTwoStage(**two_stage)
        elif self.mode == "keyframe":
            from ltx_pipelines.keyframe_interpolation import (
                KeyframeInterpolationPipeline,
            )

            self._pipeline = KeyframeInterpolationPipeline(**two_stage)
        elif self.mode == "retake":
            from ltx_pipelines.retake import RetakePipeline

            self._pipeline = RetakePipeline(
                checkpoint_path=dev_ckpt, loras=[], **shared
            )
        else:
            raise ValueError(f"Unknown mode: {self.mode}")

    def _load_persistent_models(self):
        """Build models once and patch the pipeline's ModelLedger(s) to
        return pre-built GPU-resident models instead of rebuilding from disk.

        Only loads what this mode needs — no wasted VRAM.
        """

        def patch(
            ledger, *, text_enc, emb, venc, vdec, aenc, adec, voc, ups=None, xfmr=None
        ):
            """Replace a ModelLedger's factory methods with cached returns."""
            ledger.text_encoder = lambda te=text_enc: te
            ledger.gemma_embeddings_processor = lambda e=emb: e
            ledger.video_encoder = lambda v=venc: v
            ledger.video_decoder = lambda v=vdec: v
            ledger.audio_encoder = lambda a=aenc: a
            ledger.audio_decoder = lambda a=adec: a
            ledger.vocoder = lambda v=voc: v
            if ups is not None:
                ledger.spatial_upsampler = lambda u=ups: u
            if xfmr is not None:
                ledger.transformer = lambda x=xfmr: x

        if self.mode == "fast":
            ledger = self._pipeline.model_ledger
            print("Loading text encoder (Gemma 3 12B)...")
            text_enc = ledger.text_encoder()
            print("Loading spatial upsampler...")
            ups = ledger.spatial_upsampler()
            print("Loading distilled components...")
            emb = ledger.gemma_embeddings_processor()
            venc = ledger.video_encoder()
            vdec = ledger.video_decoder()
            aenc = ledger.audio_encoder()
            adec = ledger.audio_decoder()
            voc = ledger.vocoder()
            print("Loading distilled transformer...")
            xfmr = ledger.transformer()
            patch(
                ledger,
                text_enc=text_enc,
                emb=emb,
                venc=venc,
                vdec=vdec,
                aenc=aenc,
                adec=adec,
                voc=voc,
                ups=ups,
                xfmr=xfmr,
            )

        elif self.mode == "retake":
            ledger = self._pipeline.model_ledger
            print("Loading text encoder (Gemma 3 12B)...")
            text_enc = ledger.text_encoder()
            print("Loading dev components...")
            emb = ledger.gemma_embeddings_processor()
            venc = ledger.video_encoder()
            vdec = ledger.video_decoder()
            aenc = ledger.audio_encoder()
            adec = ledger.audio_decoder()
            voc = ledger.vocoder()
            print("Loading dev transformer...")
            xfmr = ledger.transformer()
            patch(
                ledger,
                text_enc=text_enc,
                emb=emb,
                venc=venc,
                vdec=vdec,
                aenc=aenc,
                adec=adec,
                voc=voc,
                xfmr=xfmr,
            )

        else:
            # Two-stage pipelines: standard, hq, a2vid, keyframe
            s1_ledger = self._pipeline.stage_1_model_ledger
            s2_ledger = self._pipeline.stage_2_model_ledger

            print("Loading text encoder (Gemma 3 12B)...")
            text_enc = s1_ledger.text_encoder()
            print("Loading spatial upsampler...")
            ups = s1_ledger.spatial_upsampler()
            print("Loading dev components...")
            emb = s1_ledger.gemma_embeddings_processor()
            venc = s1_ledger.video_encoder()
            vdec = s1_ledger.video_decoder()
            aenc = s1_ledger.audio_encoder()
            adec = s1_ledger.audio_decoder()
            voc = s1_ledger.vocoder()

            print("Loading stage 1 transformer...")
            s1_xfmr = s1_ledger.transformer()
            print("Loading stage 2 transformer...")
            s2_xfmr = s2_ledger.transformer()

            kw = dict(
                text_enc=text_enc,
                emb=emb,
                venc=venc,
                vdec=vdec,
                aenc=aenc,
                adec=adec,
                voc=voc,
                ups=ups,
            )
            patch(s1_ledger, **kw, xfmr=s1_xfmr)
            patch(s2_ledger, **kw, xfmr=s2_xfmr)

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
        """Encode video+audio to MP4, save to volume, return result dict."""
        import tempfile

        import torch
        from ltx_core.model.video_vae import TilingConfig, get_video_chunks_number
        from ltx_pipelines.utils.media_io import encode_video

        tiling = TilingConfig.default()
        chunks = get_video_chunks_number(num_frames, tiling)

        path = tempfile.mktemp(suffix=".mp4", dir="/tmp")
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
                    "mode": self.mode,
                    "precision": self.precision,
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
        """Text/image-to-video generation (standard, fast, hq modes)."""
        if self.mode not in ("standard", "fast", "hq"):
            raise ValueError(f"generate() not supported in {self.mode} mode")
        import time

        import torch
        from ltx_core.model.video_vae import TilingConfig

        num_frames = _snap_frames(num_frames)
        images = self._prep_images(image_bytes, image_strength)
        tiling = TilingConfig.default()

        if self.mode == "hq":
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
            f"height/width must be divisible by 64, got {height}x{width}"
        )

        dur = num_frames / frame_rate
        print(
            f"generate [{self.mode}] {dur:.1f}s ({num_frames}f) "
            f"{width}x{height} steps={num_inference_steps} seed={seed}"
        )
        print(f"  prompt: {prompt[:120]}")

        t0 = time.time()

        with torch.no_grad():
            if self.mode == "fast":
                video, audio = self._pipeline(
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
                video, audio = self._pipeline(
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
                    tiling_config=tiling,
                    enhance_prompt=enhance_prompt,
                )

        gen_time = time.time() - t0
        result = self._encode_result(video, audio, num_frames, frame_rate, prompt, seed)
        result["mode"] = self.mode
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
        """Audio-driven video generation (a2vid mode)."""
        if self.mode != "a2vid":
            raise ValueError(f"generate_from_audio() not supported in {self.mode} mode")
        import time

        import torch
        from ltx_core.model.video_vae import TilingConfig

        num_frames = _snap_frames(num_frames)
        assert height % 64 == 0 and width % 64 == 0

        audio_path = self._write_temp(audio_bytes, ".wav")

        images = []
        if image_bytes is not None:
            from ltx_pipelines.utils.args import ImageConditioningInput

            img_path = self._write_temp(image_bytes, ".png")
            images = [ImageConditioningInput(img_path, 0, image_strength, 33)]

        print(
            f"audio-to-video {num_frames / frame_rate:.1f}s "
            f"{width}x{height} seed={seed}"
        )
        t0 = time.time()

        with torch.no_grad():
            video, audio = self._pipeline(
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
        result = self._encode_result(video, audio, num_frames, frame_rate, prompt, seed)
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
        """Interpolate between keyframe images (keyframe mode).

        keyframe_images: list of (image_bytes, frame_idx, strength) tuples.
        Example: [(start_img, 0, 1.0), (end_img, 120, 1.0)]
        """
        if self.mode != "keyframe":
            raise ValueError(f"interpolate() not supported in {self.mode} mode")
        import time

        import torch
        from ltx_core.model.video_vae import TilingConfig

        num_frames = _snap_frames(num_frames)
        assert height % 64 == 0 and width % 64 == 0
        images = self._prep_images_multi(keyframe_images)

        print(f"keyframe interpolation {num_frames / frame_rate:.1f}s seed={seed}")
        t0 = time.time()

        with torch.no_grad():
            video, audio = self._pipeline(
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
        result = self._encode_result(video, audio, num_frames, frame_rate, prompt, seed)
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
        """Regenerate a time region [start_time, end_time] of an existing video (retake mode)."""
        if self.mode != "retake":
            raise ValueError(f"retake() not supported in {self.mode} mode")
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
            video, audio_tensor = self._pipeline(
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
