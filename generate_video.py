"""
LTX-2.3 Video Generation on Modal

Runs the Lightricks LTX-2.3 (22B) text-to-video model on an H100 GPU.
Uses the two-stage pipeline with distilled LoRA + spatial upscaling, and
FP8 quantization to fit in VRAM.

Usage:
    uv run modal run generate_video.py --prompt "A cat sitting on a windowsill"
"""

import modal

app = modal.App("ltx-video")

# Persistent volume for caching ~80GB of model weights across runs
volume = modal.Volume.from_name("ltx-models", create_if_missing=True)
MODEL_DIR = "/models"

# HuggingFace secret (for gated Gemma model download)
hf_secret = modal.Secret.from_name("huggingface-secret")

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git", "ffmpeg")
    .pip_install(
        "torch~=2.7",
        "torchaudio",
        "einops",
        "numpy",
        "transformers>=4.52,<5",
        "safetensors",
        "accelerate",
        "scipy>=1.14",
        "av",
        "tqdm",
        "pillow",
        "huggingface_hub",
        "sentencepiece",
        "protobuf",
    )
    .run_commands("git clone https://github.com/Lightricks/LTX-2.git /ltx2")
    .env(
        {
            "PYTHONPATH": "/ltx2/packages/ltx-core/src:/ltx2/packages/ltx-pipelines/src",
            "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
            "CUDA_MODULE_LOADING": "LAZY",
        }
    )
)


@app.function(
    image=image,
    volumes={MODEL_DIR: volume},
    timeout=7200,
    secrets=[hf_secret],
)
def download_models():
    """Download all model weights to the persistent volume."""
    import os
    from pathlib import Path

    from huggingface_hub import hf_hub_download, snapshot_download

    ltx_dir = f"{MODEL_DIR}/ltx"
    gemma_dir = f"{MODEL_DIR}/gemma"

    # Skip if already downloaded
    if (
        (Path(ltx_dir) / "ltx-2.3-22b-dev.safetensors").exists()
        and Path(gemma_dir).exists()
        and any(Path(gemma_dir).glob("*.safetensors"))
    ):
        print("Models already downloaded, skipping.")
        return

    os.makedirs(ltx_dir, exist_ok=True)

    # LTX-2.3 model files (~55GB total)
    ltx_files = [
        "ltx-2.3-22b-dev.safetensors",  # 46.1 GB - main checkpoint
        "ltx-2.3-22b-distilled-lora-384.safetensors",  # 7.6 GB - distilled LoRA
        "ltx-2.3-spatial-upscaler-x2-1.0.safetensors",  # 1 GB - spatial upscaler
    ]
    for filename in ltx_files:
        if not (Path(ltx_dir) / filename).exists():
            print(f"Downloading {filename}...")
            hf_hub_download("Lightricks/LTX-2.3", filename, local_dir=ltx_dir)
        else:
            print(f"{filename} already exists, skipping.")

    # Gemma 3 text encoder (~24GB)
    if not Path(gemma_dir).exists() or not any(Path(gemma_dir).glob("*.safetensors")):
        print("Downloading Gemma 3 12B text encoder...")
        snapshot_download(
            "google/gemma-3-12b-it-qat-q4_0-unquantized", local_dir=gemma_dir
        )
    else:
        print("Gemma model already exists, skipping.")

    volume.commit()
    print("All models downloaded and committed to volume!")


@app.function(
    image=image,
    gpu="H100",
    volumes={MODEL_DIR: volume},
    timeout=1800,
)
def generate(prompt: str, seed: int = 42):
    """Generate a video from a text prompt using LTX-2.3."""
    import time

    import torch

    # Retry CUDA init — GPU may not be ready immediately in containers
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
    print(f"CUDA device: {torch.cuda.get_device_name(0)}")

    from ltx_core.components.guiders import MultiModalGuiderParams
    from ltx_core.loader import LTXV_LORA_COMFY_RENAMING_MAP, LoraPathStrengthAndSDOps
    from ltx_core.model.video_vae import TilingConfig, get_video_chunks_number
    from ltx_core.quantization import QuantizationPolicy
    from ltx_pipelines.ti2vid_two_stages import TI2VidTwoStagesPipeline
    from ltx_pipelines.utils.media_io import encode_video

    ltx_dir = f"{MODEL_DIR}/ltx"
    gemma_dir = f"{MODEL_DIR}/gemma"

    num_frames = 121
    frame_rate = 25.0

    distilled_lora = [
        LoraPathStrengthAndSDOps(
            f"{ltx_dir}/ltx-2.3-22b-distilled-lora-384.safetensors",
            0.6,
            LTXV_LORA_COMFY_RENAMING_MAP,
        ),
    ]

    print("Loading pipeline (FP8 quantized)...")
    pipeline = TI2VidTwoStagesPipeline(
        checkpoint_path=f"{ltx_dir}/ltx-2.3-22b-dev.safetensors",
        distilled_lora=distilled_lora,
        spatial_upsampler_path=f"{ltx_dir}/ltx-2.3-spatial-upscaler-x2-1.0.safetensors",
        gemma_root=gemma_dir,
        loras=[],
        quantization=QuantizationPolicy.fp8_cast(),
    )

    video_guider_params = MultiModalGuiderParams(
        cfg_scale=3.0,
        stg_scale=1.0,
        rescale_scale=0.7,
        modality_scale=3.0,
        skip_step=0,
        stg_blocks=[29],
    )

    audio_guider_params = MultiModalGuiderParams(
        cfg_scale=7.0,
        stg_scale=1.0,
        rescale_scale=0.7,
        modality_scale=3.0,
        skip_step=0,
        stg_blocks=[29],
    )

    tiling_config = TilingConfig.default()
    video_chunks_number = get_video_chunks_number(num_frames, tiling_config)

    print(f"Generating video for: {prompt}")
    with torch.no_grad():
        video, audio = pipeline(
            prompt=prompt,
            negative_prompt="",
            seed=seed,
            height=512,
            width=768,
            num_frames=num_frames,
            frame_rate=frame_rate,
            num_inference_steps=40,
            video_guider_params=video_guider_params,
            audio_guider_params=audio_guider_params,
            images=[],
            tiling_config=tiling_config,
        )

    output_path = "/tmp/output.mp4"
    print("Encoding video...")
    encode_video(
        video=video,
        fps=frame_rate,
        audio=audio,
        output_path=output_path,
        video_chunks_number=video_chunks_number,
    )

    print("Video generated!")
    with open(output_path, "rb") as f:
        return f.read()


@app.local_entrypoint()
def main(
    prompt: str = "A cinematic shot of a golden sunset over a calm ocean, with gentle waves reflecting warm light and seabirds gliding overhead",
    seed: int = 42,
):
    print("Step 1/2: Ensuring models are downloaded...")
    download_models.remote()

    print("Step 2/2: Generating video...")
    video_data = generate.remote(prompt=prompt, seed=seed)

    with open("output.mp4", "wb") as f:
        f.write(video_data)
    print(f"Done! Video saved to output.mp4")
