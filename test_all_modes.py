"""
Battle test all LTX-2.3 generation modes.

Each test gets its own container pool via modal.parameter(mode=...).

Usage:
    uv run modal run test_all_modes.py::run_tests           # run all in parallel
    uv run modal run test_all_modes.py::run_tests --test 1   # run specific test

Tests:
    1. standard mode (text-to-video, 121 frames)
    2. fast mode (text-to-video, 121 frames)
    3. hq mode (text-to-video, 49 frames, 1088x1920)
    4. image-to-video (fast, 121 frames)
    5. enhance_prompt (fast, 49 frames)
    6. audio-to-video (a2vid, 121 frames)
    7. keyframe interpolation (121 frames)
    8. retake (generate 10s base via fast, then retake 3-8s)
"""

from generate_video import app, LTXVideo


def _save(result, name, precision="fp8"):
    fname = f"test_{name}_{precision}.mp4"
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
def run_tests(test: int = 0, precision: str = "bf16"):
    # Create one handle per mode — each routes to its own container pool
    standard = LTXVideo(mode="standard", precision=precision)
    fast = LTXVideo(mode="fast", precision=precision)
    hq = LTXVideo(mode="hq", precision=precision)
    a2vid = LTXVideo(mode="a2vid", precision=precision)
    keyframe = LTXVideo(mode="keyframe", precision=precision)
    retake = LTXVideo(mode="retake", precision=precision)

    if test > 0:
        _run_single(test, standard=standard, fast=fast, hq=hq,
                    a2vid=a2vid, keyframe=keyframe, retake=retake, precision=precision)
        return

    # Fire all tests in parallel — each goes to its mode's container pool
    print("Launching tests 1-7 in parallel...")

    calls = {}

    calls["1_standard_121f"] = standard.generate.spawn(
        prompt="A cinematic shot of a golden sunset over a calm ocean, gentle waves reflecting warm light, seabirds gliding overhead, the camera slowly pans right",
        num_frames=121, seed=42,
    )
    calls["2_fast_121f"] = fast.generate.spawn(
        prompt="A massive dragon emerges from storm clouds, lightning crackling around its wings as it descends toward a medieval castle",
        num_frames=121, seed=77,
    )
    calls["3_hq_49f"] = hq.generate.spawn(
        prompt="Macro shot of a dewdrop on a rose petal at sunrise, the droplet acts as a lens revealing an inverted garden, rack focus from the drop to the blurred background",
        num_frames=49, seed=11,
    )
    calls["4_i2v_fast"] = fast.generate.spawn(
        prompt="A woman slowly turns her head toward the camera and smiles, soft natural lighting, her hair moves gently in a breeze, shallow depth of field",
        num_frames=121, seed=42,
        image_bytes=_load_file("test_image.jpeg"), image_strength=1.0,
    )
    calls["5_enhanced_49f"] = fast.generate.spawn(
        prompt="a dog playing in snow",
        num_frames=49, seed=22, enhance_prompt=True,
    )
    calls["6_a2v"] = a2vid.generate_from_audio.spawn(
        prompt="A guitarist shreds an electric guitar solo on stage, colorful stage lights flash and pulse, the crowd cheers, smoke fills the air, cinematic concert footage",
        audio_bytes=_load_file("test_audio.wav"), num_frames=121, seed=42,
    )

    kf1, kf2 = _load_file("test_kf1.jpeg"), _load_file("test_kf2.jpeg")
    calls["7_keyframe"] = keyframe.interpolate.spawn(
        prompt="A smooth cinematic transition between two scenes, the camera glides through space, soft lighting shifts gradually",
        keyframe_images=[(kf1, 0, 1.0), (kf2, 120, 1.0)],
        num_frames=121, seed=42,
    )

    # Test 8: retake needs base video first (sequential)
    print("Test 8: generating 10s base video for retake...")
    base = fast.generate.remote(
        prompt="A woman walks down a quiet city street at night, neon signs reflecting on wet pavement, calm atmosphere",
        num_frames=241, seed=99,
    )
    _save(base, "8_retake_base", precision)

    print("Test 8: retaking 3-8s region with monster...")
    calls["8_retake_result"] = retake.retake.spawn(
        video_bytes=base["video_bytes"],
        prompt="A giant monster crashes through buildings on a city street at night, debris and dust flying everywhere, cars flipping, explosions, pure chaos and destruction",
        start_time=3.0, end_time=8.0, seed=200,
    )

    # Collect all results
    print(f"\nWaiting for {len(calls)} results...")
    for name, call in calls.items():
        print(f"  Collecting {name}...")
        result = call.get()
        _save(result, name, precision)

    print(f"\nAll {len(calls)} tests complete.")


def _run_single(test, *, standard, fast, hq, a2vid, keyframe, retake, precision="fp8"):
    """Run a single test by number."""
    if test == 1:
        print("=== Test 1: standard mode, 121 frames ===")
        r = standard.generate.remote(
            prompt="A cinematic shot of a golden sunset over a calm ocean, gentle waves reflecting warm light, seabirds gliding overhead, the camera slowly pans right",
            num_frames=121, seed=42,
        )
        _save(r, "1_standard_121f", precision)

    elif test == 2:
        print("=== Test 2: fast mode, 121 frames ===")
        r = fast.generate.remote(
            prompt="A massive dragon emerges from storm clouds, lightning crackling around its wings as it descends toward a medieval castle",
            num_frames=121, seed=77,
        )
        _save(r, "2_fast_121f", precision)

    elif test == 3:
        print("=== Test 3: hq mode, 49 frames, 1088x1920 ===")
        r = hq.generate.remote(
            prompt="Macro shot of a dewdrop on a rose petal at sunrise, the droplet acts as a lens revealing an inverted garden, rack focus from the drop to the blurred background",
            num_frames=49, seed=11,
        )
        _save(r, "3_hq_49f", precision)

    elif test == 4:
        print("=== Test 4: image-to-video (fast, 121 frames) ===")
        r = fast.generate.remote(
            prompt="A woman slowly turns her head toward the camera and smiles, soft natural lighting, her hair moves gently in a breeze, shallow depth of field",
            num_frames=121, seed=42,
            image_bytes=_load_file("test_image.jpeg"), image_strength=1.0,
        )
        _save(r, "4_i2v_fast", precision)

    elif test == 5:
        print("=== Test 5: enhance_prompt (fast, 49 frames) ===")
        r = fast.generate.remote(
            prompt="a dog playing in snow",
            num_frames=49, seed=22, enhance_prompt=True,
        )
        _save(r, "5_enhanced_49f", precision)

    elif test == 6:
        print("=== Test 6: audio-to-video (a2vid, 121 frames) ===")
        r = a2vid.generate_from_audio.remote(
            prompt="A guitarist shreds an electric guitar solo on stage, colorful stage lights flash and pulse, the crowd cheers, smoke fills the air, cinematic concert footage",
            audio_bytes=_load_file("test_audio.wav"), num_frames=121, seed=42,
        )
        _save(r, "6_a2v", precision)

    elif test == 7:
        print("=== Test 7: keyframe interpolation (121 frames) ===")
        kf1, kf2 = _load_file("test_kf1.jpeg"), _load_file("test_kf2.jpeg")
        r = keyframe.interpolate.remote(
            prompt="A smooth cinematic transition between two scenes, the camera glides through space, soft lighting shifts gradually",
            keyframe_images=[(kf1, 0, 1.0), (kf2, 120, 1.0)],
            num_frames=121, seed=42,
        )
        _save(r, "7_keyframe", precision)

    elif test == 8:
        print("=== Test 8: retake (10s base, monster at 3-8s) ===")
        print("  Generating 10s base video...")
        base = fast.generate.remote(
            prompt="A woman walks down a quiet city street at night, neon signs reflecting on wet pavement, calm atmosphere",
            num_frames=241, seed=99,
        )
        _save(base, "8_retake_base", precision)
        print("  Retaking 3-8s region...")
        r = retake.retake.remote(
            video_bytes=base["video_bytes"],
            prompt="A giant monster crashes through buildings on a city street at night, debris and dust flying everywhere, cars flipping, explosions, pure chaos and destruction",
            start_time=3.0, end_time=8.0, seed=200,
        )
        _save(r, "8_retake_result", precision)
