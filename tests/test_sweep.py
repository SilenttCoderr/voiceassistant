import asyncio
import json
import wave
from pathlib import Path

import pytest

import sweep

RATE = 16000
FRAME_MS = 20.0


def states(pattern: str) -> dict:
    """`pattern` is one char per 20 ms frame: `-` quiet, `S` speaking."""
    return {
        "frame_ms": FRAME_MS,
        "state": ["SPEAKING" if char == "S" else "QUIET" for char in pattern],
    }


def write_wav(pcm: bytes, path):
    with wave.open(str(path), "wb") as out:
        out.setnchannels(1)
        out.setsampwidth(2)
        out.setframerate(RATE)
        out.writeframes(pcm)


def test_expand_grid_is_a_cartesian_product_over_the_base_config():
    configs = sweep.expand_grid({"backend": ["ten", "silero"], "confidence": [0.4, 0.8]})

    assert len(configs) == 4
    assert {c["confidence"] for c in configs} == {0.4, 0.8}
    # untouched base keys survive, so a winning row is a complete config
    assert all(c["min_volume"] == sweep.BASE_CONFIG["min_volume"] for c in configs)


def test_expand_grid_drops_thresholds_that_do_not_apply_to_the_backend():
    configs = sweep.expand_grid(
        {"backend": ["ten", "silero"], "ten_threshold": [0.5, 0.7]}
    )

    # ten keeps both thresholds; silero collapses to one config instead of two
    assert len(configs) == 3
    assert [c for c in configs if c["backend"] == "silero"] == [
        {**sweep.BASE_CONFIG, "backend": "silero"}
    ]


def test_speech_mask_marks_only_frames_inside_a_region():
    mask = sweep._speech_mask([(0.1, 0.2)], frames=20, frame_ms=FRAME_MS)

    assert list(mask).count(True) == 5
    assert not mask[4] and mask[5] and mask[9] and not mask[10]


def test_frame_stats_counts_misses_inside_regions():
    # region covers frames 0-9; the VAD only fires for the second half
    stats = sweep.frame_stats(states("-----SSSSS"), [(0.0, 0.2)])

    assert stats["speech_frames"] == 10
    assert stats["missed_frames"] == 5
    assert stats["missed_regions"] == 0
    assert stats["lags"] == [100.0]
    assert stats["false_triggers"] == 0


def test_frame_stats_counts_a_trigger_outside_every_region_as_false():
    stats = sweep.frame_stats(states("SS--------"), [])

    assert stats["false_triggers"] == 1
    assert stats["false_frames"] == 2
    assert stats["speech_frames"] == 0


def test_frame_stats_forgives_an_onset_just_inside_the_grace_window():
    # fires 40 ms before the region starts at 0.2 s — early, not a false trigger
    stats = sweep.frame_stats(states("--------SSSSS"), [(0.2, 0.26)])

    assert stats["false_triggers"] == 0
    assert stats["lags"] == [0.0]


def test_frame_stats_reports_the_tail_held_after_a_region_ends():
    stats = sweep.frame_stats(states("SSSSSSS---"), [(0.0, 0.1)])

    assert stats["tails"] == [40.0]  # two frames of hangover past 0.1 s


def test_frame_stats_flags_a_region_the_vad_never_noticed():
    stats = sweep.frame_stats(states("----------"), [(0.0, 0.2)])

    assert stats["missed_regions"] == 1
    assert stats["lags"] == []


def test_combine_pools_counters_across_clips():
    metrics = sweep.combine(
        [
            sweep.frame_stats(states("-----SSSSS"), [(0.0, 0.2)]),
            sweep.frame_stats(states("SS--------"), []),
        ]
    )

    assert metrics["miss"] == 0.5
    assert metrics["false_triggers"] == 1
    assert metrics["onset_ms"] == 100.0


def test_total_score_ranks_the_cleaner_config_first():
    clean = sweep.combine([sweep.frame_stats(states("SSSSSSSSSS"), [(0.0, 0.2)])])
    leaky = sweep.combine([sweep.frame_stats(states("-----SSSSS"), [(0.0, 0.2)])])

    assert sweep.total_score(clean) < sweep.total_score(leaky)


def test_propose_labels_finds_the_loud_stretch():
    import numpy as np

    t = np.arange(RATE) / RATE
    tone = (0.4 * np.sin(2 * np.pi * 180 * t) * 32767).astype("<i2").tobytes()
    quiet = b"\x00\x00" * (RATE // 2)

    regions = sweep.propose_labels(quiet + tone + quiet)

    assert len(regions) == 1
    assert regions[0][0] == pytest.approx(0.5, abs=0.05)
    assert regions[0][1] == pytest.approx(1.5, abs=0.05)


def test_load_clips_pairs_wavs_with_labels_and_reports_the_rest(tmp_path):
    write_wav(b"\x00\x00" * RATE, tmp_path / "good.wav")
    (tmp_path / "good.json").write_text(json.dumps({"speech": [[0.1, 0.4]]}))
    write_wav(b"\x00\x00" * RATE, tmp_path / "bare.wav")

    clips, unlabelled = sweep.load_clips(tmp_path)

    assert unlabelled == ["bare.wav"]
    assert [clip["name"] for clip in clips] == ["good"]
    assert clips[0]["speech"] == [(0.1, 0.4)]


def test_run_sweep_denoises_each_clip_once_per_chain(monkeypatch):
    calls = []

    async def fake_chain(pcm, config):
        calls.append(config["noise_filter"])
        return pcm

    async def fake_analyze(pcm, config):
        return states("SSSSS")

    monkeypatch.setattr(sweep.lab, "apply_noise_chain", fake_chain)
    monkeypatch.setattr(sweep.lab, "analyze", fake_analyze)

    clips = [{"name": "a", "pcm": b"\x00\x00" * RATE, "speech": [(0.0, 0.1)]}]
    configs = sweep.expand_grid(
        {"confidence": [0.4, 0.6, 0.8], "noise_filter": ["none", "highpass"]}
    )

    rows = asyncio.run(sweep.run_sweep(clips, configs))

    assert len(rows) == 6
    assert calls == ["none", "highpass"]  # not once per config
    assert all("score" in row for row in rows)


def test_run_sweep_records_a_failing_config_instead_of_stopping(monkeypatch):
    async def boom(pcm, config):
        if config["backend"] == "silero":
            raise RuntimeError("nope")
        return states("SSSSS")

    monkeypatch.setattr(sweep.lab, "apply_noise_chain", lambda pcm, config: _same(pcm))
    monkeypatch.setattr(sweep.lab, "analyze", boom)

    clips = [{"name": "a", "pcm": b"\x00\x00" * RATE, "speech": [(0.0, 0.1)]}]
    rows = asyncio.run(sweep.run_sweep(clips, sweep.expand_grid({"backend": ["ten", "silero"]})))

    assert [("error" in row) for row in rows] == [False, True]
    assert "nope" in rows[1]["error"]


async def _same(pcm):
    return pcm


def test_denoise_cache_pins_the_audio_even_when_the_filter_drifts(tmp_path, monkeypatch):
    """The cache is what makes two sweeps comparable.

    RNNoise fed by soxr at HQ does not reproduce byte for byte, so the fake filter
    here returns something different every call — exactly the behaviour the cache
    exists to absorb.
    """
    calls = []

    async def drifting(pcm, config):
        calls.append(config["noise_filter"])
        return pcm[: len(pcm) - 2 * len(calls)]

    monkeypatch.setattr(sweep.lab, "apply_noise_chain", drifting)
    config = {**sweep.BASE_CONFIG, "noise_filter": "rnnoise"}
    audio = b"\x01\x02" * RATE

    first = asyncio.run(sweep.denoise(audio, config, tmp_path))
    second = asyncio.run(sweep.denoise(audio, config, tmp_path))

    assert first == second
    assert calls == ["rnnoise"]  # second call served from disk


def test_denoise_cache_separates_different_chains(tmp_path, monkeypatch):
    async def by_chain(pcm, config):
        return pcm if config["noise_filter"] == "rnnoise" else pcm[:100]

    monkeypatch.setattr(sweep.lab, "apply_noise_chain", by_chain)
    audio = b"\x01\x02" * RATE

    rnnoise = asyncio.run(
        sweep.denoise(audio, {**sweep.BASE_CONFIG, "noise_filter": "rnnoise"}, tmp_path)
    )
    highpass = asyncio.run(
        sweep.denoise(audio, {**sweep.BASE_CONFIG, "noise_filter": "highpass"}, tmp_path)
    )

    assert rnnoise != highpass
    assert len(list(tmp_path.glob("*.wav"))) == 2


def test_denoise_without_a_cache_dir_never_writes(tmp_path, monkeypatch):
    async def passthrough(pcm, config):
        return pcm

    monkeypatch.setattr(sweep.lab, "apply_noise_chain", passthrough)

    asyncio.run(sweep.denoise(b"\x01\x02" * RATE, sweep.BASE_CONFIG, None))

    assert list(tmp_path.iterdir()) == []


def test_load_clips_accepts_a_recording_ffmpeg_has_to_decode(tmp_path, monkeypatch):
    """Voice Recorder writes m4a, so a clip folder will not be all WAV."""
    decoded = []

    def fake_decode(path):
        decoded.append(Path(path).name)
        return b"\x01\x02" * RATE

    monkeypatch.setattr(sweep.lab, "decode_audio", fake_decode)
    (tmp_path / "Recording (2).m4a").write_bytes(b"aac")
    (tmp_path / "Recording (2).json").write_text(json.dumps({"speech": [[0.1, 0.4]]}))

    clips, unlabelled = sweep.load_clips(tmp_path)

    assert unlabelled == []
    assert [clip["name"] for clip in clips] == ["Recording (2)"]
    assert decoded == ["Recording (2).m4a"]


def test_load_clips_takes_one_clip_per_stem(tmp_path, monkeypatch):
    # both formats of the same recording share the one labels file
    monkeypatch.setattr(sweep.lab, "decode_audio", lambda path: b"\x01\x02" * RATE)
    write_wav(b"\x00\x00" * RATE, tmp_path / "me.wav")
    (tmp_path / "me.m4a").write_bytes(b"aac")
    (tmp_path / "me.json").write_text(json.dumps({"speech": []}))

    clips, _ = sweep.load_clips(tmp_path)

    assert len(clips) == 1


def test_load_clips_reports_an_unlabelled_recording_of_any_format(tmp_path, monkeypatch):
    monkeypatch.setattr(sweep.lab, "decode_audio", lambda path: b"\x01\x02" * RATE)
    (tmp_path / "bare.m4a").write_bytes(b"aac")

    clips, unlabelled = sweep.load_clips(tmp_path)

    assert clips == []
    assert unlabelled == ["bare.m4a"]


def test_run_sweep_reports_progress_for_every_config_and_clip(monkeypatch):
    steps = []

    async def fake_analyze(pcm, config):
        return states("SSSSS")

    monkeypatch.setattr(sweep.lab, "apply_noise_chain", _passthrough)
    monkeypatch.setattr(sweep.lab, "analyze", fake_analyze)

    clips = [
        {"name": "a", "pcm": b"\x00\x00" * RATE, "speech": [(0.0, 0.1)]},
        {"name": "b", "pcm": b"\x00\x00" * RATE, "speech": []},
    ]
    configs = sweep.expand_grid({"confidence": [0.4, 0.6, 0.8]})

    asyncio.run(sweep.run_sweep(clips, configs, on_progress=steps.append))

    assert steps[0] == {"done": 0, "total": 6, "clip": None}
    assert steps[-1]["done"] == 6 and steps[-1]["total"] == 6
    assert [s["done"] for s in steps] == sorted(s["done"] for s in steps)
    assert {s["clip"] for s in steps if s["clip"]} == {"a", "b"}


def test_progress_still_reaches_the_end_when_a_config_fails(monkeypatch):
    """A stalled bar is indistinguishable from a hung sweep, so it must not stall."""
    steps = []

    async def boom(pcm, config):
        if config["backend"] == "silero":
            raise RuntimeError("nope")
        return states("SSSSS")

    monkeypatch.setattr(sweep.lab, "apply_noise_chain", _passthrough)
    monkeypatch.setattr(sweep.lab, "analyze", boom)

    clips = [{"name": "a", "pcm": b"\x00\x00" * RATE, "speech": [(0.0, 0.1)]}] * 2
    configs = sweep.expand_grid({"backend": ["ten", "silero"]})

    asyncio.run(sweep.run_sweep(clips, configs, on_progress=steps.append))

    assert steps[-1]["done"] == steps[-1]["total"] == 4


async def _passthrough(pcm, config):
    return pcm
