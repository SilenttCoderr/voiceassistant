"""Local VAD and turn-detection lab.

Record noisy audio once, then replay it through the real `create_vad()` backends as
many times as you like while sweeping thresholds. Everything here drives the same
code path the agent uses, so a setting that looks good in the lab is a setting you
can copy straight into `.env`.
"""

import asyncio
import base64
import io
import json
import os
import wave
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

from dotenv import load_dotenv

from pipecat.audio.turn.base_turn_analyzer import EndOfTurnState
from pipecat.audio.vad.vad_analyzer import VADState

from vad import create_vad

load_dotenv(override=False)

SAMPLE_RATE = 16000
FILTER_CHUNK_BYTES = 640  # 20 ms of 16 kHz int16 mono
LAB_HTML = Path(__file__).with_name("lab.html")


@contextmanager
def _environment(overrides: dict[str, str]):
    previous = {name: os.environ.get(name) for name in overrides}
    os.environ.update(overrides)
    try:
        yield
    finally:
        for name, value in previous.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def decode_wav(raw: bytes) -> bytes:
    """Return 16 kHz mono int16 PCM from a WAV file."""
    with wave.open(io.BytesIO(raw)) as source:
        if source.getsampwidth() != 2:
            raise ValueError("WAV must be 16-bit PCM")
        channels = source.getnchannels()
        rate = source.getframerate()
        pcm = source.readframes(source.getnframes())

    if channels > 1:
        import numpy as np

        frames = np.frombuffer(pcm, dtype="<i2").reshape(-1, channels)
        pcm = frames.mean(axis=1).astype("<i2").tobytes()

    if rate != SAMPLE_RATE:
        import numpy as np
        import soxr

        samples = np.frombuffer(pcm, dtype="<i2").astype("float32")
        pcm = soxr.resample(samples, rate, SAMPLE_RATE).astype("<i2").tobytes()

    return pcm


def _env_overrides(config: dict) -> dict[str, str]:
    overrides = {
        "VAD_BACKEND": str(config["backend"]),
        "VAD_CONFIDENCE": str(config["confidence"]),
        "VAD_START_SECS": str(config["start_secs"]),
        "VAD_STOP_SECS": str(config["stop_secs"]),
        "VAD_MIN_VOLUME": str(config["min_volume"]),
    }
    if config.get("ten_threshold") is not None:
        overrides["TEN_VAD_THRESHOLD"] = str(config["ten_threshold"])
    if config.get("firered_threshold") is not None:
        overrides["FIREREDVAD_SPEECH_THRESHOLD"] = str(config["firered_threshold"])
    return overrides


def _filter_env_overrides(config: dict) -> dict[str, str]:
    overrides = {}
    if config.get("noise_mix") is not None:
        overrides["NOISE_MIX"] = str(config["noise_mix"])
    if config.get("rnnoise_quality"):
        overrides["RNNOISE_QUALITY"] = str(config["rnnoise_quality"])
    if config.get("highpass_hz") is not None:
        overrides["HIGHPASS_HZ"] = str(config["highpass_hz"])
    return overrides


def _create_turn_analyzer(config: dict):
    from pipecat.audio.turn.smart_turn.base_smart_turn import SmartTurnParams
    from pipecat.audio.turn.smart_turn.local_smart_turn_v3 import LocalSmartTurnAnalyzerV3

    return LocalSmartTurnAnalyzerV3(
        sample_rate=SAMPLE_RATE,
        params=SmartTurnParams(
            stop_secs=float(config.get("smart_turn_stop_secs", 3.0)),
            max_duration_secs=float(config.get("smart_turn_max_duration_secs", 8.0)),
        ),
    )


DEEPFILTER_COMMAND = [
    "uv", "run", "--python", "3.11", "--no-project",
    "--index-strategy", "unsafe-best-match",
    "--with", "deepfilternet==0.5.6",
    "--with", "torch==2.0.1",
    "--with", "torchaudio==2.0.2",
    "python", str(Path(__file__).with_name("deepfilter_worker.py")),
]


def deepfilternet_denoise(pcm: bytes) -> bytes:
    """Denoise via DeepFilterNet in a separate Python 3.11 environment.

    It cannot share this venv: no cp312 wheel for `deepfilterlib`, a `numpy<2` pin, and it
    needs `torchaudio<2.1`. Running it out of process keeps the project on 3.12.
    """
    import subprocess
    import tempfile

    environment = dict(os.environ)
    environment.setdefault("UV_EXTRA_INDEX_URL", "https://download.pytorch.org/whl/cpu")

    with tempfile.TemporaryDirectory() as directory:
        source = Path(directory) / "in.wav"
        target = Path(directory) / "out.wav"
        source.write_bytes(_wav_bytes(pcm))

        completed = subprocess.run(
            [*DEEPFILTER_COMMAND, str(source), str(target)],
            capture_output=True,
            env=environment,
        )
        if completed.returncode != 0 or not target.exists():
            tail = completed.stderr.decode("utf-8", "replace").strip().splitlines()[-3:]
            raise RuntimeError("DeepFilterNet failed: " + " | ".join(tail))
        return decode_wav(target.read_bytes())


def _wav_bytes(pcm: bytes) -> bytes:
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as out:
        out.setnchannels(1)
        out.setsampwidth(2)
        out.setframerate(SAMPLE_RATE)
        out.writeframes(pcm)
    return buffer.getvalue()


async def apply_noise_filter(pcm: bytes, name: str | None) -> bytes:
    """Clean the clip with a Pipecat audio filter, the way the transport would.

    Fed in 20 ms chunks rather than one buffer so the filter sees the same frame sizes it
    gets in the pipeline. Filters buffer internally, so the result is typically a few
    milliseconds shorter than the input.
    """
    if not name or name in ("none", "browser"):
        return pcm

    # Hand consecutive Pipecat stages to the agent's own chain builder so NOISE_MIX is
    # applied once around them, exactly as it is in the pipeline. DeepFilterNet is not a
    # Pipecat filter, so it splits the chain.
    pending: list[str] = []
    for stage in (part.strip() for part in name.split("+")):
        if stage == "deepfilternet":
            if pending:
                pcm = await _run_pipecat_chain(pcm, "+".join(pending))
                pending = []
            pcm = deepfilternet_denoise(pcm)
        else:
            pending.append(stage)
    if pending:
        pcm = await _run_pipecat_chain(pcm, "+".join(pending))
    return pcm


async def apply_noise_chain(pcm: bytes, config: dict) -> bytes:
    """Apply a slot's denoise chain under that slot's own filter settings."""
    with _environment(_filter_env_overrides(config)):
        return await apply_noise_filter(pcm, config.get("noise_filter"))


async def _run_pipecat_chain(pcm: bytes, chain: str) -> bytes:
    from bot import create_audio_filter

    audio_filter = create_audio_filter(chain)
    if audio_filter is None:
        return pcm

    await audio_filter.start(SAMPLE_RATE)
    try:
        chunks = [
            await audio_filter.filter(pcm[at : at + FILTER_CHUNK_BYTES])
            for at in range(0, len(pcm), FILTER_CHUNK_BYTES)
        ]
    finally:
        await audio_filter.stop()
    return b"".join(chunks)


async def analyze(pcm: bytes, config: dict) -> dict:
    """Replay PCM through one VAD configuration and return per-frame traces."""
    pcm = await apply_noise_chain(pcm, config)

    with _environment(_env_overrides(config)):
        analyzer = create_vad(config["backend"])

    if analyzer is None:
        raise ValueError(
            "Gemini VAD runs server-side and cannot be replayed here; pick a local backend"
        )

    # The pipeline normally does this at transport start; offline we do it ourselves.
    analyzer.set_sample_rate(SAMPLE_RATE)

    turn = _create_turn_analyzer(config) if config.get("smart_turn") else None
    if turn is not None:
        turn.set_sample_rate(SAMPLE_RATE)

    frame_bytes = analyzer.num_frames_required() * 2
    confidences: list[float] = []
    original_voice_confidence = analyzer.voice_confidence

    def recording_voice_confidence(buffer: bytes) -> float:
        value = original_voice_confidence(buffer)
        confidences.append(_as_float(value))
        return value

    analyzer.voice_confidence = recording_voice_confidence

    states: list[str] = []
    volumes: list[float] = []
    peaks: list[float] = []
    turns: list[dict] = []

    try:
        for start in range(0, len(pcm) - frame_bytes + 1, frame_bytes):
            frame = pcm[start : start + frame_bytes]
            state = await analyzer.analyze_audio(frame)
            states.append(state.name)
            volumes.append(float(getattr(analyzer, "_prev_volume", 0.0)))
            peaks.append(_frame_peak(frame))

            if turn is not None:
                speech = state in (VADState.STARTING, VADState.SPEAKING)
                if turn.append_audio(frame, speech) == EndOfTurnState.COMPLETE:
                    result, metrics = await turn.analyze_end_of_turn()
                    turns.append(
                        {
                            "frame": len(states) - 1,
                            "complete": result == EndOfTurnState.COMPLETE,
                            "probability": (
                                None if metrics is None else _as_float(metrics.probability)
                            ),
                        }
                    )
    finally:
        await analyzer.cleanup()
        if turn is not None:
            await turn.cleanup()

    # `confidence` is recorded inside the analyzer, so it can lag the state list by a
    # frame if a backend buffers internally. Trim to the shorter of the two.
    length = min(len(states), len(confidences))
    return {
        "frame_ms": frame_bytes / 2 / SAMPLE_RATE * 1000,
        "duration_secs": len(pcm) / 2 / SAMPLE_RATE,
        "confidence": confidences[:length],
        "volume": volumes[:length],
        "peak": peaks[:length],
        "state": states[:length],
        "turns": turns,
    }


def _mel_filterbank(n_fft: int, n_mels: int, rate: int, fmin: float, fmax: float):
    import numpy as np

    to_mel = lambda hz: 2595.0 * np.log10(1.0 + hz / 700.0)  # noqa: E731
    to_hz = lambda mel: 700.0 * (10.0 ** (mel / 2595.0) - 1.0)  # noqa: E731

    points = to_hz(np.linspace(to_mel(fmin), to_mel(fmax), n_mels + 2))
    bins = np.floor((n_fft + 1) * points / rate).astype(int)
    bins = np.clip(bins, 0, n_fft // 2)

    filterbank = np.zeros((n_mels, n_fft // 2 + 1), dtype="float32")
    for m in range(n_mels):
        left, centre, right = bins[m], max(bins[m + 1], bins[m] + 1), max(bins[m + 2], bins[m + 1] + 2)
        right = min(right, n_fft // 2)
        if centre >= right:
            continue
        for k in range(left, centre):
            filterbank[m, k] = (k - left) / max(centre - left, 1)
        for k in range(centre, right):
            filterbank[m, k] = (right - k) / max(right - centre, 1)
    return filterbank


def mel_spectrogram(pcm: bytes, n_fft: int = 512, hop: int = 160, n_mels: int = 64) -> dict:
    """Log-mel spectrogram as base64 uint8, laid out frame-major for the canvas."""
    import numpy as np

    samples = np.frombuffer(pcm, dtype="<i2").astype("float32") / 32768.0
    if samples.size < n_fft:
        samples = np.pad(samples, (0, n_fft - samples.size))

    count = 1 + (samples.size - n_fft) // hop
    indices = np.arange(n_fft)[None, :] + hop * np.arange(count)[:, None]
    frames = samples[indices] * np.hanning(n_fft).astype("float32")

    power = np.abs(np.fft.rfft(frames, axis=1)) ** 2
    mel = power @ _mel_filterbank(n_fft, n_mels, SAMPLE_RATE, 50.0, SAMPLE_RATE / 2).T
    log_mel = np.log10(mel + 1e-10)

    floor = log_mel.max() - 6.0 if log_mel.size else 0.0
    scaled = np.clip((log_mel - floor) / 6.0, 0.0, 1.0)

    return {
        "mels": n_mels,
        "frames": int(count),
        "hop_ms": hop / SAMPLE_RATE * 1000,
        "data": base64.b64encode((scaled * 255).astype("uint8").tobytes()).decode(),
    }


async def render_audio(pcm: bytes, config: dict, mode: str) -> bytes:
    """Produce the audio a configuration actually yields, so it can be listened to.

    ``filtered`` is the clip after the noise chain — what the agent forwards upstream.
    ``gated`` keeps only the frames the VAD called speech, concatenated. If words are
    losing their first syllable, ``gated`` is where you hear it.
    """
    filtered = await apply_noise_chain(pcm, config)
    if mode != "gated":
        return filtered

    # Already filtered above; don't run the chain twice.
    result = await analyze(filtered, {**config, "noise_filter": "none"})
    frame_bytes = int(round(result["frame_ms"] * SAMPLE_RATE / 1000)) * 2

    kept = bytearray()
    for index, state in enumerate(result["state"]):
        if state in ("STARTING", "SPEAKING"):
            kept += filtered[index * frame_bytes : (index + 1) * frame_bytes]
    return bytes(kept)


async def analyze_all(pcm: bytes, configs: list[dict]) -> list[dict]:
    """Replay the same audio through several configurations for side-by-side comparison.

    A backend that fails to load (missing extra, missing key, missing model) is reported
    in its own entry so the remaining backends still produce a comparison.
    """
    results = []
    for config in configs:
        try:
            results.append({"config": config, "result": await analyze(pcm, config)})
        except Exception as exc:
            results.append({"config": config, "error": f"{type(exc).__name__}: {exc}"})
    return results


def _as_float(value) -> float:
    """Backends return floats, numpy scalars, or single-element arrays."""
    try:
        return float(value)
    except TypeError:
        import numpy as np

        flat = np.asarray(value).reshape(-1)
        return float(flat[0]) if flat.size else 0.0


def _frame_peak(frame: bytes) -> float:
    import numpy as np

    samples = np.frombuffer(frame, dtype="<i2")
    if samples.size == 0:
        return 0.0
    return float(np.abs(samples).max() / 32768.0)


class LabHandler(BaseHTTPRequestHandler):
    def _send(self, status: int, content_type: str, body: bytes) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        if self.path in ("/", "/index.html"):
            self._send(200, "text/html; charset=utf-8", LAB_HTML.read_bytes())
            return
        self._send(404, "text/plain; charset=utf-8", b"not found")

    def do_POST(self) -> None:
        if self.path == "/render":
            self._render()
            return
        if self.path != "/analyze":
            self._send(404, "text/plain; charset=utf-8", b"not found")
            return

        length = int(self.headers.get("Content-Length", 0))
        try:
            request = json.loads(self.rfile.read(length))
            pcm = decode_wav(base64.b64decode(request["wav"]))
            configs = request.get("configs") or [request["config"]]
            results = asyncio.run(analyze_all(pcm, configs))
            payload = {
                "results": results,
                "mel": mel_spectrogram(pcm),
                "duration_secs": len(pcm) / 2 / SAMPLE_RATE,
            }
        except Exception as exc:
            body = json.dumps({"error": f"{type(exc).__name__}: {exc}"}).encode()
            self._send(400, "application/json", body)
            return

        self._send(200, "application/json", json.dumps(payload).encode())

    def _render(self) -> None:
        length = int(self.headers.get("Content-Length", 0))
        try:
            request = json.loads(self.rfile.read(length))
            pcm = decode_wav(base64.b64decode(request["wav"]))
            rendered = asyncio.run(
                render_audio(pcm, request["config"], request.get("mode", "filtered"))
            )
        except Exception as exc:
            body = json.dumps({"error": f"{type(exc).__name__}: {exc}"}).encode()
            self._send(400, "application/json", body)
            return

        self._send(200, "audio/wav", _wav_bytes(rendered))

    def log_message(self, *args) -> None:
        return


def main() -> None:
    port = int(os.getenv("LAB_PORT", "7861"))
    print(f"VAD lab running at http://127.0.0.1:{port}")
    HTTPServer(("127.0.0.1", port), LabHandler).serve_forever()


if __name__ == "__main__":
    main()
