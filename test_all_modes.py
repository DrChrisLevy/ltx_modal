"""
Battle test all LTX-2.3 generation modes.

Each test runs on its own container in parallel (up to 8 H200s).

Usage:
    uv run modal run test_all_modes.py::run_tests           # run all in parallel
    uv run modal run test_all_modes.py::run_tests --test 1   # run specific test

Tests:
    1. standard mode (text-to-video, 121 frames)
    2. fast mode (text-to-video, 121 frames)
    3. hq mode (text-to-video, 49 frames, 1088x1920)
    4. image-to-video (real photo, fast, 121 frames)
    5. enhance_prompt (fast, 49 frames)
    6. audio-to-video (real audio, 121 frames)
    7. keyframe interpolation (real photos, 121 frames)
    8. retake (generate 10s base then retake 3-8s with monster)
"""

from generate_video import app, LTXVideo


def _save(result, name):
    fname = f"test_{name}.mp4"
    with open(fname, "wb") as f:
        f.write(result["video_bytes"])
    print(
        f"  -> {fname} | {result['duration']}s | {result['size_mb']} MB | "
        f"{result.get('gen_time_s', '?')}s gen | mode={result.get('mode', '?')}"
    )


def _load_file(path):
    with open(path, "rb") as f:
        return f.read()


@app.local_entrypoint()
def run_tests(test: int = 0):
    ltx = LTXVideo()

    if test > 0:
        _run_single(ltx, test)
        return

    # Fire all tests in parallel — each .spawn() gets its own container
    print("Launching tests 1-7 in parallel...")

    calls = {}

    calls["1_standard_121f"] = ltx.generate.spawn(
        prompt="A cinematic shot of a golden sunset over a calm ocean, gentle waves reflecting warm light, seabirds gliding overhead, the camera slowly pans right",
        mode="standard", num_frames=121, seed=42,
    )
    calls["2_fast_121f"] = ltx.generate.spawn(
        prompt="A massive dragon emerges from storm clouds, lightning crackling around its wings as it descends toward a medieval castle",
        mode="fast", num_frames=121, seed=77,
    )
    calls["3_hq_49f"] = ltx.generate.spawn(
        prompt="Macro shot of a dewdrop on a rose petal at sunrise, the droplet acts as a lens revealing an inverted garden, rack focus from the drop to the blurred background",
        mode="hq", num_frames=49, seed=11,
    )
    calls["4_i2v_fast"] = ltx.generate.spawn(
        prompt="A woman slowly turns her head toward the camera and smiles, soft natural lighting, her hair moves gently in a breeze, shallow depth of field",
        mode="fast", num_frames=121, seed=42,
        image_bytes=_load_file("test_image.jpeg"), image_strength=1.0,
    )
    calls["5_enhanced_49f"] = ltx.generate.spawn(
        prompt="a dog playing in snow",
        mode="fast", num_frames=49, seed=22, enhance_prompt=True,
    )
    calls["6_a2v"] = ltx.generate_from_audio.spawn(
        prompt="A guitarist shreds an electric guitar solo on stage, colorful stage lights flash and pulse, the crowd cheers, smoke fills the air, cinematic concert footage",
        audio_bytes=_load_file("test_audio.wav"), num_frames=121, seed=42,
    )

    kf1, kf2 = _load_file("test_kf1.jpeg"), _load_file("test_kf2.jpeg")
    calls["7_keyframe"] = ltx.interpolate.spawn(
        prompt="A smooth cinematic transition between two scenes, the camera glides through space, soft lighting shifts gradually",
        keyframe_images=[(kf1, 0, 1.0), (kf2, 120, 1.0)],
        num_frames=121, seed=42,
    )

    # Test 8: retake needs base video first (sequential), then retake in parallel
    print("Test 8: generating 10s base video for retake...")
    base = ltx.generate.remote(
        prompt="A woman walks down a quiet city street at night, neon signs reflecting on wet pavement, calm atmosphere",
        mode="fast", num_frames=241, seed=99,
    )
    _save(base, "8_retake_base")

    print("Test 8: retaking 3-8s region with monster...")
    calls["8_retake_result"] = ltx.retake.spawn(
        video_bytes=base["video_bytes"],
        prompt="A giant monster crashes through buildings on a city street at night, debris and dust flying everywhere, cars flipping, explosions, pure chaos and destruction",
        start_time=3.0, end_time=8.0, seed=200,
    )

    # Collect all results
    print(f"\nWaiting for {len(calls)} results...")
    for name, call in calls.items():
        print(f"  Collecting {name}...")
        result = call.get()
        _save(result, name)

    print(f"\nAll {len(calls)} tests complete.")


def _run_single(ltx, test):
    """Run a single test by number."""
    if test == 1:
        print("=== Test 1: standard mode, 121 frames ===")
        r = ltx.generate.remote(
            prompt="A cinematic shot of a golden sunset over a calm ocean, gentle waves reflecting warm light, seabirds gliding overhead, the camera slowly pans right",
            mode="standard", num_frames=121, seed=42,
        )
        _save(r, "1_standard_121f")

    elif test == 2:
        print("=== Test 2: fast mode, 121 frames ===")
        r = ltx.generate.remote(
            prompt="A massive dragon emerges from storm clouds, lightning crackling around its wings as it descends toward a medieval castle",
            mode="fast", num_frames=121, seed=77,
        )
        _save(r, "2_fast_121f")

    elif test == 3:
        print("=== Test 3: hq mode, 49 frames, 1088x1920 ===")
        r = ltx.generate.remote(
            prompt="Macro shot of a dewdrop on a rose petal at sunrise, the droplet acts as a lens revealing an inverted garden, rack focus from the drop to the blurred background",
            mode="hq", num_frames=49, seed=11,
        )
        _save(r, "3_hq_49f")

    elif test == 4:
        print("=== Test 4: image-to-video (real photo, fast, 121 frames) ===")
        r = ltx.generate.remote(
            prompt="A woman slowly turns her head toward the camera and smiles, soft natural lighting, her hair moves gently in a breeze, shallow depth of field",
            mode="fast", num_frames=121, seed=42,
            image_bytes=_load_file("test_image.jpeg"), image_strength=1.0,
        )
        _save(r, "4_i2v_fast")

    elif test == 5:
        print("=== Test 5: enhance_prompt (fast, 49 frames) ===")
        r = ltx.generate.remote(
            prompt="a dog playing in snow",
            mode="fast", num_frames=49, seed=22, enhance_prompt=True,
        )
        _save(r, "5_enhanced_49f")

    elif test == 6:
        print("=== Test 6: audio-to-video (real audio, 121 frames) ===")
        r = ltx.generate_from_audio.remote(
            prompt="A guitarist shreds an electric guitar solo on stage, colorful stage lights flash and pulse, the crowd cheers, smoke fills the air, cinematic concert footage",
            audio_bytes=_load_file("test_audio.wav"), num_frames=121, seed=42,
        )
        _save(r, "6_a2v")

    elif test == 7:
        print("=== Test 7: keyframe interpolation (real photos, 121 frames) ===")
        kf1, kf2 = _load_file("test_kf1.jpeg"), _load_file("test_kf2.jpeg")
        r = ltx.interpolate.remote(
            prompt="A smooth cinematic transition between two scenes, the camera glides through space, soft lighting shifts gradually",
            keyframe_images=[(kf1, 0, 1.0), (kf2, 120, 1.0)],
            num_frames=121, seed=42,
        )
        _save(r, "7_keyframe")

    elif test == 8:
        print("=== Test 8: retake (10s base, monster at 3-8s) ===")
        print("  Generating 10s base video...")
        base = ltx.generate.remote(
            prompt="A woman walks down a quiet city street at night, neon signs reflecting on wet pavement, calm atmosphere",
            mode="fast", num_frames=241, seed=99,
        )
        _save(base, "8_retake_base")
        print("  Retaking 3-8s region...")
        r = ltx.retake.remote(
            video_bytes=base["video_bytes"],
            prompt="A giant monster crashes through buildings on a city street at night, debris and dust flying everywhere, cars flipping, explosions, pure chaos and destruction",
            start_time=3.0, end_time=8.0, seed=200,
        )
        _save(r, "8_retake_result")
