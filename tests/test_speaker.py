import asyncio
import io
import sys
import wave

import numpy as np
import pytest

import speaker
import sweep

RATE = 16000
FRAME_MS = 20.0


class FakeStream:
    def __init__(self):
        self.samples = None

    def accept_waveform(self, sample_rate, waveform):
        assert sample_rate == RATE
        self.samples = waveform

    def input_finished(self):
        pass


class FakeExtractor:
    """Returns a vector that depends only on how loud the audio is.

    Enough to drive the gate deterministically: a loud segment and a quiet one land
    on opposite sides of any sensible threshold.
    """

    def __init__(self, ready=True):
        self._ready = ready

    def create_stream(self):
        return FakeStream()

    def is_ready(self, stream):
        return self._ready

    def compute(self, stream):
        loud = float(np.abs(stream.samples).mean()) > 0.1
        return [1.0, 0.0] if loud else [0.0, 1.0]


def write_wav(path, seconds=1.0, amplitude=0.4):
    t = np.arange(int(RATE * seconds)) / RATE
    samples = (amplitude * np.sin(2 * np.pi * 180 * t) * 32767).astype("<i2")
    with wave.open(str(path), "wb") as out:
        out.setnchannels(1)
        out.setsampwidth(2)
        out.setframerate(RATE)
        out.writeframes(samples.tobytes())


def pcm(seconds: float, amplitude: float = 0.4) -> bytes:
    t = np.arange(int(RATE * seconds)) / RATE
    return (amplitude * np.sin(2 * np.pi * 180 * t) * 32767).astype("<i2").tobytes()


def test_similarity_is_cosine_and_ignores_magnitude():
    assert speaker.similarity([1.0, 0.0], [2.0, 0.0]) == pytest.approx(1.0)
    assert speaker.similarity([1.0, 0.0], [0.0, 3.0]) == pytest.approx(0.0)
    assert speaker.similarity([1.0, 0.0], [-1.0, 0.0]) == pytest.approx(-1.0)


def test_similarity_of_a_zero_vector_is_zero_not_a_crash():
    assert speaker.similarity([0.0, 0.0], [1.0, 0.0]) == 0.0


def test_embed_returns_a_unit_vector():
    vector = speaker.embed(FakeExtractor(), pcm(1.0))

    assert float(np.linalg.norm(vector)) == pytest.approx(1.0)


def test_embed_rejects_a_clip_the_model_cannot_use():
    with pytest.raises(ValueError, match="too short"):
        speaker.embed(FakeExtractor(ready=False), pcm(0.01))


def test_enroll_averages_every_wav_in_a_directory(tmp_path):
    write_wav(tmp_path / "one.wav")
    write_wav(tmp_path / "two.wav")

    vector = speaker.enroll(FakeExtractor(), tmp_path)

    assert float(np.linalg.norm(vector)) == pytest.approx(1.0)


def test_enroll_rejects_an_empty_directory(tmp_path):
    with pytest.raises(RuntimeError, match="no enrollment audio"):
        speaker.enroll(FakeExtractor(), tmp_path)


def test_segments_finds_each_run_of_speech():
    states = ["QUIET", "STARTING", "SPEAKING", "QUIET", "SPEAKING"]

    assert speaker.segments(states, FRAME_MS) == [(1, 3), (4, 5)]


def test_segments_closes_a_run_that_reaches_the_end():
    assert speaker.segments(["SPEAKING", "SPEAKING"], FRAME_MS) == [(0, 2)]


def _result(states):
    return {"frame_ms": FRAME_MS, "state": list(states)}


def test_gate_silences_a_segment_that_is_not_the_enrolled_speaker():
    # 50 frames = 1 s of quiet audio, which the fake extractor scores as "not me"
    result = _result(["SPEAKING"] * 50)
    quiet = pcm(1.0, amplitude=0.0)

    gated = speaker.gate(result, quiet, [1.0, 0.0], FakeExtractor(), threshold=0.5)

    assert set(gated["state"]) == {"QUIET"}
    assert gated["speaker"][0]["kept"] is False
    assert gated["speaker"][0]["similarity"] == pytest.approx(0.0)


def test_gate_keeps_the_enrolled_speaker():
    result = _result(["SPEAKING"] * 50)

    gated = speaker.gate(result, pcm(1.0), [1.0, 0.0], FakeExtractor(), threshold=0.5)

    assert set(gated["state"]) == {"SPEAKING"}
    assert gated["speaker"][0]["kept"] is True


def test_gate_abstains_on_a_segment_too_short_to_judge():
    # 10 frames = 200 ms, below MIN_GATE_SECS — no embedding is even attempted
    result = _result(["SPEAKING"] * 10)
    quiet = pcm(0.2, amplitude=0.0)

    gated = speaker.gate(result, quiet, [1.0, 0.0], FakeExtractor(), threshold=0.5)

    assert set(gated["state"]) == {"SPEAKING"}
    assert gated["speaker"][0]["similarity"] is None
    assert gated["speaker"][0]["kept"] is True


def test_gate_leaves_the_original_result_untouched():
    result = _result(["SPEAKING"] * 50)

    speaker.gate(result, pcm(1.0, amplitude=0.0), [1.0, 0.0], FakeExtractor(), 0.5)

    assert set(result["state"]) == {"SPEAKING"}


def test_resolve_model_explains_where_to_put_a_missing_model():
    with pytest.raises(RuntimeError, match="speaker model not found"):
        speaker.resolve_model("no-such-model.onnx")


def test_create_extractor_names_the_extra_when_sherpa_is_missing(monkeypatch):
    monkeypatch.setitem(sys.modules, "sherpa_onnx", None)

    with pytest.raises(RuntimeError, match="uv sync --extra speaker"):
        speaker.create_extractor()


def test_sweep_gates_only_when_an_enrollment_and_threshold_are_both_set(monkeypatch):
    gated = []

    async def fake_analyze(audio, config):
        return _result(["SPEAKING"] * 50)

    def fake_gate(result, audio, enrolled, extractor, threshold):
        gated.append(threshold)
        return result

    monkeypatch.setattr(sweep.lab, "apply_noise_chain", _passthrough)
    monkeypatch.setattr(sweep.lab, "analyze", fake_analyze)
    monkeypatch.setattr(sweep.speaker, "gate", fake_gate)
    monkeypatch.setattr(sweep, "_speaker_gate", lambda cache, config: ("x", "y"))

    clips = [{"name": "a", "pcm": pcm(1.0), "speech": [(0.0, 1.0)]}]
    configs = sweep.expand_grid(
        {"speaker_enroll": [None, "enroll"], "speaker_threshold": [None, 0.5]}
    )

    asyncio.run(sweep.run_sweep(clips, configs))

    assert gated == [0.5]  # only the config with both set


async def _passthrough(audio, config):
    return audio
