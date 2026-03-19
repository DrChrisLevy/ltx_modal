"""
Battle test all LTX-2.3 generation modes.

Usage:
    uv run modal run test_all_modes.py::run_tests           # run all tests
    uv run modal run test_all_modes.py::run_tests --test 1   # run specific test

Tests:
    1. standard mode (text-to-video, 121 frames)
    2. fast mode (text-to-video, 121 frames)
    3. hq mode (text-to-video, 49 frames, 1088x1920)
    4. image-to-video (real photo, fast, 121 frames)
    5. enhance_prompt (fast, 49 frames)
    6. audio-to-video (real audio, 121 frames)
    7. keyframe interpolation (real photos, 121 frames)
    8. retake (generate base then retake 1-3s region)
"""

from generate_video import app, LTXVideo


@app.local_entrypoint()
def run_tests(test: int = 0):
    ltx = LTXVideo()

    tests = [
        test_standard_121,
        test_fast_121,
        test_hq_49,
        test_image_to_video,
        test_enhance_prompt,
        test_audio_to_video,
        test_keyframe_interpolation,
        test_retake,
    ]

    if test > 0:
        tests[test - 1](ltx, test)
    else:
        for i, fn in enumerate(tests, 1):
            fn(ltx, i)


def _save(result, name):
    fname = f"test_{name}.mp4"
    with open(fname, "wb") as f:
        f.write(result["video_bytes"])
    print(
        f"  -> {fname} | {result['duration']}s | {result['size_mb']} MB | "
        f"{result.get('gen_time_s', '?')}s gen | mode={result.get('mode', '?')}"
    )
    print()


# --- Text-to-video modes ---


def test_standard_121(ltx, num):
    print(f"=== Test {num}: standard mode, 121 frames (5s) ===")
    result = ltx.generate.remote(
        prompt="A cinematic shot of a golden sunset over a calm ocean, gentle waves reflecting warm light, seabirds gliding overhead, the camera slowly pans right",
        mode="standard",
        num_frames=121,
        seed=42,
    )
    _save(result, f"{num}_standard_121f")


def test_fast_121(ltx, num):
    print(f"=== Test {num}: fast mode, 121 frames (5s) ===")
    result = ltx.generate.remote(
        prompt="A massive dragon emerges from storm clouds, lightning crackling around its wings as it descends toward a medieval castle",
        mode="fast",
        num_frames=121,
        seed=77,
    )
    _save(result, f"{num}_fast_121f")


def test_hq_49(ltx, num):
    print(f"=== Test {num}: hq mode, 49 frames (2s), 1088x1920 ===")
    result = ltx.generate.remote(
        prompt="Macro shot of a dewdrop on a rose petal at sunrise, the droplet acts as a lens revealing an inverted garden, rack focus from the drop to the blurred background",
        mode="hq",
        num_frames=49,
        seed=11,
    )
    _save(result, f"{num}_hq_49f")


# --- Image/audio conditioned modes (real inputs) ---


def test_image_to_video(ltx, num):
    print(f"=== Test {num}: image-to-video (real photo, fast, 121 frames) ===")

    with open("test_image.jpeg", "rb") as f:
        image_bytes = f.read()
    print(f"  Image: test_image.jpeg ({len(image_bytes) / 1024:.0f} KB)")

    result = ltx.generate.remote(
        prompt="A woman slowly turns her head toward the camera and smiles, soft natural lighting, her hair moves gently in a breeze, shallow depth of field",
        mode="fast",
        num_frames=121,
        seed=42,
        image_bytes=image_bytes,
        image_strength=1.0,
    )
    _save(result, f"{num}_i2v_fast")


def test_enhance_prompt(ltx, num):
    print(f"=== Test {num}: enhance_prompt (fast, 49 frames) ===")
    result = ltx.generate.remote(
        prompt="a dog playing in snow",
        mode="fast",
        num_frames=49,
        seed=22,
        enhance_prompt=True,
    )
    _save(result, f"{num}_enhanced_49f")


def test_audio_to_video(ltx, num):
    print(f"=== Test {num}: audio-to-video (real audio, 121 frames) ===")

    with open("test_audio.wav", "rb") as f:
        audio_bytes = f.read()
    print(f"  Audio: test_audio.wav ({len(audio_bytes) / 1024:.0f} KB)")

    result = ltx.generate_from_audio.remote(
        prompt="A guitarist shreds an electric guitar solo on stage, colorful stage lights flash and pulse, the crowd cheers, smoke fills the air, cinematic concert footage",
        audio_bytes=audio_bytes,
        num_frames=121,
        seed=42,
    )
    _save(result, f"{num}_a2v")


def test_keyframe_interpolation(ltx, num):
    print(f"=== Test {num}: keyframe interpolation (real photos, 121 frames) ===")

    with open("test_kf1.jpeg", "rb") as f:
        kf1_bytes = f.read()
    with open("test_kf2.jpeg", "rb") as f:
        kf2_bytes = f.read()
    print(
        f"  KF1: test_kf1.jpeg ({len(kf1_bytes) / 1024:.0f} KB), "
        f"KF2: test_kf2.jpeg ({len(kf2_bytes) / 1024:.0f} KB)"
    )

    keyframes = [
        (kf1_bytes, 0, 1.0),
        (kf2_bytes, 120, 1.0),
    ]

    result = ltx.interpolate.remote(
        prompt="A smooth cinematic transition between two scenes, the camera glides through space, soft lighting shifts gradually",
        keyframe_images=keyframes,
        num_frames=121,
        seed=42,
    )
    _save(result, f"{num}_keyframe")


def test_retake(ltx, num):
    print(f"=== Test {num}: retake (regenerate middle of video, same prompt, different seed) ===")

    print("  Generating base video...")
    base = ltx.generate.remote(
        prompt="A woman walks down a city street at night, neon signs reflecting on wet pavement",
        mode="fast",
        num_frames=121,
        seed=99,
    )
    _save(base, f"{num}_retake_base")

    # Retake the middle 2 seconds with a dramatically different prompt.
    print("  Retaking 1.5-3.5s region with monster prompt...")
    result = ltx.retake.remote(
        video_bytes=base["video_bytes"],
        prompt="A giant monster crashes through buildings on a city street at night, debris and dust flying everywhere, cars flipping, pure chaos and destruction",
        start_time=1.5,
        end_time=3.5,
        seed=200,
    )
    _save(result, f"{num}_retake_result")
