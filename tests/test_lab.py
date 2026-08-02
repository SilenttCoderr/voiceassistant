import asyncio
import base64
import io
import json
import os
import wave
from pathlib import Path

import pytest

import lab

RATE = 16000


def write_wav(samples: bytes, rate: int = RATE, channels: int = 1) -> bytes:
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as out:
        out.setnchannels(channels)
        out.setsampwidth(2)
        out.setframerate(rate)
        out.writeframes(samples)
    return buffer.getvalue()


def tone(seconds: float, rate: int = RATE) -> bytes:
    import numpy as np

    t = np.arange(int(rate * seconds)) / rate
    signal = 0.4 * np.sin(2 * np.pi * 180 * t)
    return (signal * 32767).astype("<i2").tobytes()


def test_decode_wav_passes_through_16k_mono():
    pcm = tone(0.5)

    assert lab.decode_wav(write_wav(pcm)) == pcm


def test_decode_wav_downmixes_stereo():
    import numpy as np

    mono = np.frombuffer(tone(0.2), dtype="<i2")
    stereo = np.repeat(mono, 2).astype("<i2").tobytes()

    decoded = lab.decode_wav(write_wav(stereo, channels=2))

    assert len(decoded) == len(mono) * 2


def test_decode_wav_resamples_to_16k():
    decoded = lab.decode_wav(write_wav(tone(0.5, rate=48000), rate=48000))

    assert abs(len(decoded) / 2 - RATE * 0.5) < RATE * 0.02


def test_decode_wav_rejects_non_pcm16():
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as out:
        out.setnchannels(1)
        out.setsampwidth(1)
        out.setframerate(RATE)
        out.writeframes(b"\x00" * 100)

    with pytest.raises(ValueError, match="16-bit PCM"):
        lab.decode_wav(buffer.getvalue())


def test_environment_overrides_are_restored(monkeypatch):
    monkeypatch.setenv("VAD_CONFIDENCE", "0.9")
    monkeypatch.delenv("TEN_VAD_THRESHOLD", raising=False)

    with lab._environment({"VAD_CONFIDENCE": "0.1", "TEN_VAD_THRESHOLD": "0.2"}):
        assert os.environ["VAD_CONFIDENCE"] == "0.1"
        assert os.environ["TEN_VAD_THRESHOLD"] == "0.2"

    assert os.environ["VAD_CONFIDENCE"] == "0.9"
    assert "TEN_VAD_THRESHOLD" not in os.environ


def test_env_overrides_skip_unset_backend_thresholds():
    overrides = lab._env_overrides(
        {
            "backend": "silero",
            "confidence": 0.5,
            "start_secs": 0.2,
            "stop_secs": 0.45,
            "min_volume": 0.35,
        }
    )

    assert overrides["VAD_BACKEND"] == "silero"
    assert "TEN_VAD_THRESHOLD" not in overrides
    assert "FIREREDVAD_SPEECH_THRESHOLD" not in overrides


def test_gemini_backend_cannot_be_replayed():
    config = {
        "backend": "gemini",
        "confidence": 0.5,
        "start_secs": 0.2,
        "stop_secs": 0.45,
        "min_volume": 0.35,
    }

    with pytest.raises(ValueError, match="cannot be replayed"):
        asyncio.run(lab.analyze(tone(0.2), config))


def test_noise_filter_none_is_a_passthrough():
    pcm = tone(0.2)

    for name in (None, "", "none", "browser"):
        assert asyncio.run(lab.apply_noise_filter(pcm, name)) is pcm


def test_rnnoise_filter_changes_the_audio():
    pytest.importorskip("pyrnnoise")

    import numpy as np

    rng = np.random.default_rng(0)
    noisy = (rng.normal(0, 0.2, RATE).clip(-1, 1) * 32767).astype("<i2").tobytes()

    cleaned = asyncio.run(lab.apply_noise_filter(noisy, "rnnoise"))

    assert cleaned != noisy
    # Filters buffer internally, so a few milliseconds of tail can be missing.
    # HQ resampling has a longer filter than QQ, so it buffers more before emitting.
    assert abs(len(cleaned) - len(noisy)) < RATE * 2 * 0.10
    assert np.frombuffer(cleaned, "<i2").std() < np.frombuffer(noisy, "<i2").std()


def _band_energy(pcm: bytes, low: float, high: float) -> float:
    import numpy as np

    samples = np.frombuffer(pcm, "<i2").astype(float)
    spectrum = np.abs(np.fft.rfft(samples))
    freqs = np.fft.rfftfreq(samples.size, 1 / RATE)
    return float(spectrum[(freqs >= low) & (freqs < high)].mean())


def test_highpass_removes_rumble_and_keeps_speech():
    import numpy as np

    t = np.arange(RATE) / RATE
    speech = 0.35 * sum(np.sin(2 * np.pi * f * t) / (i + 1) for i, f in enumerate([160, 320, 480])) / 3
    rumble = 0.25 * np.sin(2 * np.pi * 45 * t)
    pcm = (np.clip(speech + rumble, -1, 1) * 32767).astype("<i2").tobytes()

    filtered = asyncio.run(lab.apply_noise_filter(pcm, "highpass"))

    assert _band_energy(filtered, 20, 100) < _band_energy(pcm, 20, 100) * 0.5
    assert _band_energy(filtered, 150, 600) > _band_energy(pcm, 150, 600) * 0.9


def test_filter_chain_runs_every_stage():
    pytest.importorskip("pyrnnoise")
    pcm = tone(0.3)

    chained = asyncio.run(lab.apply_noise_filter(pcm, "highpass+rnnoise"))
    single = asyncio.run(lab.apply_noise_filter(pcm, "highpass"))

    assert chained != single
    assert chained != pcm


def test_deepfilternet_runs_in_its_own_python():
    command = lab.DEEPFILTER_COMMAND

    assert command[:4] == ["uv", "run", "--python", "3.11"]
    assert "--no-project" in command
    # torchaudio 2.1 dropped torchaudio.backend, which deepfilternet 0.5.6 imports.
    assert "torchaudio==2.0.2" in command
    assert Path(command[-1]).name == "deepfilter_worker.py"
    assert Path(command[-1]).exists()


@pytest.mark.skipif(
    os.getenv("LAB_TEST_DEEPFILTER") != "1",
    reason="spawns a separate 3.11 environment; set LAB_TEST_DEEPFILTER=1 to run",
)
def test_deepfilternet_denoises_end_to_end():
    import numpy as np

    rng = np.random.default_rng(0)
    noisy = (rng.normal(0, 0.2, RATE).clip(-1, 1) * 32767).astype("<i2").tobytes()

    cleaned = asyncio.run(lab.apply_noise_filter(noisy, "deepfilternet"))

    assert len(cleaned) == len(noisy)
    assert np.frombuffer(cleaned, "<i2").std() < np.frombuffer(noisy, "<i2").std()


def test_unknown_noise_filter_is_rejected():
    with pytest.raises(ValueError, match="NOISE_FILTER must be one of"):
        asyncio.run(lab.apply_noise_filter(tone(0.1), "nope"))


def test_mel_spectrogram_shape_and_range():
    import base64

    import numpy as np

    mel = lab.mel_spectrogram(tone(1.0))
    raw = np.frombuffer(base64.b64decode(mel["data"]), dtype="uint8")

    assert mel["mels"] == 64
    assert mel["hop_ms"] == 10.0
    assert raw.size == mel["frames"] * mel["mels"]
    assert raw.max() == 255


def test_mel_spectrogram_separates_silence_from_tone():
    import base64

    import numpy as np

    silence = b"\x00\x00" * (RATE // 2)
    mel = lab.mel_spectrogram(silence + tone(0.5))
    frames = np.frombuffer(base64.b64decode(mel["data"]), dtype="uint8").reshape(
        mel["frames"], mel["mels"]
    )

    quiet_energy = frames[: RATE // 2 // 160 - 5].mean()
    loud_energy = frames[RATE // 2 // 160 + 5 :].mean()
    assert loud_energy > quiet_energy


def test_mel_spectrogram_handles_clip_shorter_than_one_window():
    mel = lab.mel_spectrogram(tone(0.005))

    assert mel["frames"] >= 1


def test_analyze_all_isolates_a_failing_backend():
    shared = {
        "confidence": 0.5,
        "start_secs": 0.2,
        "stop_secs": 0.45,
        "min_volume": 0.35,
        "smart_turn": False,
    }
    configs = [dict(shared, backend="silero"), dict(shared, backend="gemini")]

    results = asyncio.run(lab.analyze_all(tone(0.3), configs))

    assert "result" in results[0]
    assert "error" not in results[0]
    assert "error" in results[1]
    assert results[1]["config"]["backend"] == "gemini"


def test_analyze_returns_aligned_traces():
    config = {
        "backend": "silero",
        "confidence": 0.5,
        "start_secs": 0.2,
        "stop_secs": 0.45,
        "min_volume": 0.35,
        "smart_turn": False,
    }

    result = asyncio.run(lab.analyze(tone(1.0), config))

    frames = len(result["state"])
    assert frames > 0
    assert len(result["confidence"]) == frames
    assert len(result["volume"]) == frames
    assert len(result["peak"]) == frames
    assert result["duration_secs"] == pytest.approx(1.0, abs=0.05)
    assert set(result["state"]) <= {"QUIET", "STARTING", "SPEAKING", "STOPPING"}
    assert all(isinstance(value, float) for value in result["confidence"])


def test_steady_resamplers_ignores_a_gap_between_chunks():
    """A pause mid-clip must not change the audio, or every score moves with it.

    Pipecat clears the resampler history after `clear_after_secs` of wall clock, so
    without the patch a GC pause or a model load between two chunks silently rewrites
    the rest of the clip. Faking an old timestamp reproduces that pause exactly.
    """
    import numpy as np
    from pipecat.audio.resamplers.soxr_stream_resampler import SOXRStreamAudioResampler

    rng = np.random.default_rng(0)
    audio = (rng.normal(0, 0.2, RATE).clip(-1, 1) * 32767).astype("<i2").tobytes()
    chunks = [audio[at : at + 640] for at in range(0, len(audio), 640)]

    def resample(stale: bool) -> bytes:
        # QQ, because HQ rounds two different ways run to run and would mask the gap
        resampler = SOXRStreamAudioResampler(quality="QQ")
        out = []
        for index, chunk in enumerate(chunks):
            if stale and index == len(chunks) // 2:
                resampler._last_resample_time = 0  # as if the machine stalled here
            out.append(asyncio.run(resampler.resample(chunk, RATE, 48000)))
        return b"".join(out)

    assert resample(stale=True) != resample(stale=False)

    with lab._steady_resamplers():
        assert resample(stale=True) == resample(stale=False)


def test_steady_resamplers_restores_pipecat_afterwards():
    from pipecat.audio.resamplers.soxr_stream_resampler import SOXRStreamAudioResampler

    original = SOXRStreamAudioResampler._maybe_clear_internal_state

    with lab._steady_resamplers():
        assert SOXRStreamAudioResampler._maybe_clear_internal_state is not original

    assert SOXRStreamAudioResampler._maybe_clear_internal_state is original


def test_decode_audio_reads_wav_without_ffmpeg(tmp_path, monkeypatch):
    import subprocess

    def refuse(*args, **kwargs):
        raise AssertionError("WAV must not need ffmpeg")

    monkeypatch.setattr(subprocess, "run", refuse)
    source = tmp_path / "clip.wav"
    source.write_bytes(lab._wav_bytes(tone(0.2)))

    assert lab.decode_audio(source) == tone(0.2)


def test_decode_audio_sends_m4a_through_ffmpeg_as_16k_mono(tmp_path, monkeypatch):
    import subprocess

    seen = {}

    def fake_run(command, **kwargs):
        seen["command"] = command
        # ffmpeg writes to the path it was handed as the last argument
        with wave.open(command[-1], "wb") as out:
            out.setnchannels(1)
            out.setsampwidth(2)
            out.setframerate(RATE)
            out.writeframes(tone(0.1))
        return subprocess.CompletedProcess(command, 0, b"", b"")

    monkeypatch.setattr(subprocess, "run", fake_run)
    source = tmp_path / "Recording (2).m4a"
    source.write_bytes(b"not really aac")

    assert lab.decode_audio(source) == tone(0.1)

    command = seen["command"]
    assert command[0] == "ffmpeg"
    assert str(source) in command
    # mono at the lab's own rate, so nothing has to be resampled a second time
    assert command[command.index("-ac") + 1] == "1"
    assert command[command.index("-ar") + 1] == str(RATE)


def test_decode_audio_says_how_to_install_a_missing_ffmpeg(tmp_path, monkeypatch):
    import subprocess

    def missing(*args, **kwargs):
        raise FileNotFoundError(2, "No such file or directory: 'ffmpeg'")

    monkeypatch.setattr(subprocess, "run", missing)
    source = tmp_path / "clip.m4a"
    source.write_bytes(b"whatever")

    with pytest.raises(RuntimeError, match="needs ffmpeg"):
        lab.decode_audio(source)


def test_decode_audio_reports_what_ffmpeg_complained_about(tmp_path, monkeypatch):
    import subprocess

    def fake_run(command, **kwargs):
        return subprocess.CompletedProcess(
            command, 1, b"", b"Invalid data found when processing input\n"
        )

    monkeypatch.setattr(subprocess, "run", fake_run)
    source = tmp_path / "broken.m4a"
    source.write_bytes(b"truncated")

    with pytest.raises(RuntimeError, match="Invalid data found"):
        lab.decode_audio(source)


def test_safe_clip_file_strips_any_path_it_is_given():
    """Directories are dropped rather than rejected, so nothing can leave `clips/`."""
    for name in ("../../secrets.wav", "/etc/passwd.wav", "sub/dir/me.wav"):
        cleaned = lab.safe_clip_file(name)
        assert "/" not in cleaned and "\\" not in cleaned
        assert not cleaned.startswith("..")


def test_safe_clip_file_refuses_names_that_are_not_a_recording():
    for name in ("", "   ", "../.env", ".hidden.wav"):
        with pytest.raises(ValueError):
            lab.safe_clip_file(name)


def test_safe_clip_file_keeps_a_plain_recording():
    assert lab.safe_clip_file("Recording (2).m4a") == "Recording (2).m4a"
    assert lab.safe_clip_file("me-1.wav") == "me-1.wav"


def test_safe_clip_file_rejects_a_non_audio_suffix():
    with pytest.raises(ValueError, match="not one of"):
        lab.safe_clip_file("notes.txt")


def test_save_clip_writes_the_audio_and_the_labels_together(tmp_path, monkeypatch):
    monkeypatch.setattr(lab, "CLIPS_DIR", tmp_path)
    wav = base64.b64encode(lab._wav_bytes(tone(0.5))).decode()

    result = lab.save_clip({"name": "me-1", "wav": wav, "speech": [[0.1, 0.4]]})

    assert result == {"name": "me-1", "regions": 1}
    assert lab.decode_wav((tmp_path / "me-1.wav").read_bytes()) == tone(0.5)
    assert json.loads((tmp_path / "me-1.json").read_text())["speech"] == [[0.1, 0.4]]


def test_save_clip_can_update_labels_without_resending_the_audio(tmp_path, monkeypatch):
    monkeypatch.setattr(lab, "CLIPS_DIR", tmp_path)
    (tmp_path / "me-1.wav").write_bytes(lab._wav_bytes(tone(0.2)))
    before = (tmp_path / "me-1.wav").read_bytes()

    lab.save_clip({"name": "me-1", "wav": None, "speech": [[0.0, 0.1]]})

    assert (tmp_path / "me-1.wav").read_bytes() == before
    assert json.loads((tmp_path / "me-1.json").read_text())["speech"] == [[0.0, 0.1]]


def test_save_clip_accepts_a_clip_with_no_speech_at_all(tmp_path, monkeypatch):
    """A pure-noise clip is the whole point of the false trigger metric."""
    monkeypatch.setattr(lab, "CLIPS_DIR", tmp_path)
    wav = base64.b64encode(lab._wav_bytes(tone(0.2))).decode()

    lab.save_clip({"name": "train", "wav": wav, "speech": []})

    assert json.loads((tmp_path / "train.json").read_text())["speech"] == []


def test_save_clip_rejects_a_backwards_region(tmp_path, monkeypatch):
    monkeypatch.setattr(lab, "CLIPS_DIR", tmp_path)

    with pytest.raises(ValueError, match="ends before it starts"):
        lab.save_clip({"name": "me-1", "wav": None, "speech": [[0.4, 0.1]]})

    assert not (tmp_path / "me-1.json").exists()


def test_save_clip_cannot_be_talked_into_writing_outside_the_folder(tmp_path, monkeypatch):
    monkeypatch.setattr(lab, "CLIPS_DIR", tmp_path)
    wav = base64.b64encode(lab._wav_bytes(tone(0.1))).decode()

    lab.save_clip({"name": "../../escaped", "wav": wav, "speech": []})

    # the traversal is stripped, not honoured: both files stay in the clips folder
    assert sorted(p.name for p in tmp_path.iterdir()) == ["escaped.json", "escaped.wav"]
    assert not (tmp_path.parent / "escaped.wav").exists()


def test_list_clips_reports_labels_where_they_exist(tmp_path, monkeypatch):
    monkeypatch.setattr(lab, "CLIPS_DIR", tmp_path)
    (tmp_path / "labelled.wav").write_bytes(lab._wav_bytes(tone(0.1)))
    (tmp_path / "labelled.json").write_text(json.dumps({"speech": [[0.0, 0.1]]}))
    (tmp_path / "bare.m4a").write_bytes(b"aac")

    listed = {item["name"]: item for item in lab.list_clips()}

    assert listed["labelled"]["speech"] == [[0.0, 0.1]]
    assert listed["labelled"]["file"] == "labelled.wav"
    assert listed["bare"]["speech"] is None
    assert listed["bare"]["file"] == "bare.m4a"


def test_run_sweep_request_says_what_is_missing_when_nothing_is_labelled(tmp_path, monkeypatch):
    monkeypatch.setattr(lab, "CLIPS_DIR", tmp_path)
    (tmp_path / "bare.wav").write_bytes(lab._wav_bytes(tone(0.1)))

    with pytest.raises(ValueError, match="no labelled clips"):
        lab.run_sweep_request({"grid": {"backend": ["silero"]}})
