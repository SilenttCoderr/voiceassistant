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
# What a clip may be recorded as. Anything but WAV goes through ffmpeg.
AUDIO_SUFFIXES = (".wav", ".m4a", ".mp3", ".mp4", ".aac", ".ogg", ".opus", ".flac", ".webm")
LAB_HTML = Path(__file__).with_name("lab.html")
SWEEP_HTML = Path(__file__).with_name("sweep.html")
CLIPS_DIR = Path(__file__).with_name("clips")


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


def decode_audio(path: str | Path) -> bytes:
    """Return 16 kHz mono int16 PCM from any file ffmpeg can read.

    WAV is decoded in process; everything else costs one ffmpeg call. Windows Voice
    Recorder writes m4a, and nothing in this venv decodes AAC — `soundfile` covers
    MP3, FLAC and OGG but not that.
    """
    path = Path(path)
    if path.suffix.lower() == ".wav":
        return decode_wav(path.read_bytes())
    return decode_wav(_ffmpeg_to_wav(path))


def _ffmpeg_to_wav(path: Path) -> bytes:
    """Transcode to 16 kHz mono WAV through a temp file.

    A temp file rather than a pipe: piped WAV carries an unknown length in its header,
    which stdlib `wave` reads as a frame count of nonsense.
    """
    import subprocess
    import tempfile

    with tempfile.TemporaryDirectory() as directory:
        target = Path(directory) / "decoded.wav"
        try:
            completed = subprocess.run(
                # fmt: off
                [
                    "ffmpeg", "-nostdin", "-loglevel", "error", "-y",
                    "-i", str(path),
                    "-ac", "1", "-ar", str(SAMPLE_RATE), str(target),
                ],
                # fmt: on
                capture_output=True,
            )
        except FileNotFoundError as exc:
            raise RuntimeError(
                f"{path.suffix} needs ffmpeg to decode, and it is not on PATH. "
                "Install it with 'winget install Gyan.FFmpeg', or record WAV instead."
            ) from exc

        if completed.returncode != 0 or not target.exists():
            tail = completed.stderr.decode("utf-8", "replace").strip().splitlines()[-3:]
            raise RuntimeError(f"ffmpeg could not decode {path.name}: " + " | ".join(tail))
        return target.read_bytes()


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


@contextmanager
def _steady_resamplers():
    """Stop Pipecat's stream resampler from clearing its history on wall-clock gaps.

    `SOXRStreamAudioResampler` wipes its filter history whenever more than
    `clear_after_secs` (0.2 by default) passes between two chunks. In a live call
    that is right — stale history after a silence causes artefacts. Replaying a file
    it is wrong and invisible: a GC pause or a model load between two chunks clears
    the history mid-clip, so the same WAV denoises differently on every run and every
    score moves with it. The class supports this via `clear_after_secs=None`, but
    `RNNoiseFilter` builds its own resamplers and passes only the quality through.

    Lab only. The live pipeline in `bot.py` keeps the timing-based clearing.
    """
    from pipecat.audio.resamplers.soxr_stream_resampler import SOXRStreamAudioResampler

    original = SOXRStreamAudioResampler._maybe_clear_internal_state
    SOXRStreamAudioResampler._maybe_clear_internal_state = lambda self: None
    try:
        yield
    finally:
        SOXRStreamAudioResampler._maybe_clear_internal_state = original


async def _run_pipecat_chain(pcm: bytes, chain: str) -> bytes:
    from bot import create_audio_filter

    audio_filter = create_audio_filter(chain)
    if audio_filter is None:
        return pcm

    with _steady_resamplers():
        return await _drive_filter(audio_filter, pcm)


async def _drive_filter(audio_filter, pcm: bytes) -> bytes:
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


def safe_clip_file(name: str) -> str:
    """A clip file name that cannot escape `clips/`.

    The body of a POST is untrusted even on a local tool: `../../.env` would otherwise
    be a valid clip name.
    """
    cleaned = Path(str(name)).name.strip()
    if not cleaned or cleaned.startswith("."):
        raise ValueError(f"unusable clip name: {name!r}")
    if Path(cleaned).suffix.lower() not in AUDIO_SUFFIXES:
        raise ValueError(f"{cleaned} is not one of {', '.join(AUDIO_SUFFIXES)}")
    return cleaned


def list_clips() -> list[dict]:
    """Every recording in `clips/`, with its labels if it has any."""
    found: dict[str, dict] = {}
    for suffix in AUDIO_SUFFIXES:
        for recording in sorted(CLIPS_DIR.glob(f"*{suffix}")):
            if recording.stem in found:
                continue
            labels = recording.with_suffix(".json")
            speech = None
            if labels.exists():
                speech = json.loads(labels.read_text(encoding="utf-8"))["speech"]
            found[recording.stem] = {
                "name": recording.stem,
                "file": recording.name,
                "speech": speech,
            }
    return list(found.values())


def save_clip(request: dict) -> dict:
    """Write a clip and its labels together, so the pair can never drift apart.

    The browser has already decoded and downmixed whatever was dropped in, so what
    arrives is always 16 kHz mono WAV — an m4a from Voice Recorder included.
    """
    # Validated even when no audio is attached, so the labels file is bound by the same
    # rule as the recording and the two always land side by side.
    name = Path(safe_clip_file(Path(str(request["name"])).stem + ".wav")).stem
    speech = [[float(start), float(end)] for start, end in request.get("speech", [])]
    for start, end in speech:
        if end <= start:
            raise ValueError(f"region {start:g}-{end:g} ends before it starts")

    CLIPS_DIR.mkdir(parents=True, exist_ok=True)
    if request.get("wav"):
        # Decode first: a malformed upload must not leave a half-written clip behind.
        pcm = decode_wav(base64.b64decode(request["wav"]))
        (CLIPS_DIR / (name + ".wav")).write_bytes(_wav_bytes(pcm))

    (CLIPS_DIR / (name + ".json")).write_text(
        json.dumps({"speech": speech}, indent=2), encoding="utf-8"
    )
    return {"name": name, "regions": len(speech)}


def run_sweep_request(request: dict, on_progress=None) -> dict:
    """Run a sweep for the browser. Imported here because `sweep` imports this module."""
    import sweep

    clips, unlabelled = sweep.load_clips(CLIPS_DIR)
    if not clips:
        raise ValueError("no labelled clips yet — label one and save it first")

    configs = sweep.expand_grid(request.get("grid") or sweep.DEFAULT_GRID)
    weights = request.get("weights") or sweep.DEFAULT_WEIGHTS
    rows = asyncio.run(
        sweep.run_sweep(clips, configs, weights, CLIPS_DIR / ".cache", on_progress)
    )
    return {
        "rows": rows,
        "unlabelled": unlabelled,
        "clips": [clip["name"] for clip in clips],
        "seconds": sum(len(clip["pcm"]) for clip in clips) / 2 / SAMPLE_RATE,
    }


class LabHandler(BaseHTTPRequestHandler):
    def handle(self) -> None:
        try:
            super().handle()
        except (ConnectionAbortedError, ConnectionResetError, BrokenPipeError):
            # The browser reloaded, closed the tab, or dropped one of the speculative
            # connections it opens ahead of time. There is nobody left to answer, and
            # the default handler prints a traceback that reads like a server fault.
            pass

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
        if self.path == "/sweep":
            self._send(200, "text/html; charset=utf-8", SWEEP_HTML.read_bytes())
            return
        if self.path == "/clips":
            self._send(200, "application/json", json.dumps(list_clips()).encode())
            return
        if self.path.startswith("/clips/"):
            self._clip_audio(self.path[len("/clips/") :])
            return
        self._send(404, "text/plain; charset=utf-8", b"not found")

    def _clip_audio(self, name: str) -> None:
        from urllib.parse import unquote

        try:
            path = CLIPS_DIR / safe_clip_file(unquote(name))
            body = path.read_bytes()
        except (ValueError, OSError):
            self._send(404, "text/plain; charset=utf-8", b"not found")
            return
        # The browser decodes it, so the exact container does not matter here.
        self._send(200, "application/octet-stream", body)

    def do_POST(self) -> None:
        if self.path == "/render":
            self._render()
            return
        if self.path == "/clips":
            self._json_command(save_clip)
            return
        if self.path == "/sweep":
            self._sweep_stream()
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

    def _json_command(self, handler) -> None:
        """Read a JSON body, hand it to `handler`, send back whatever it returns."""
        length = int(self.headers.get("Content-Length", 0))
        try:
            payload = handler(json.loads(self.rfile.read(length)))
        except Exception as exc:
            body = json.dumps({"error": f"{type(exc).__name__}: {exc}"}).encode()
            self._send(400, "application/json", body)
            return
        self._send(200, "application/json", json.dumps(payload, default=str).encode())

    def _sweep_stream(self) -> None:
        """Stream the sweep as newline-delimited JSON: progress lines, then the result.

        A sweep runs for minutes and this server is single threaded, so it cannot answer
        a separate progress endpoint while it works. Streaming the one response instead
        needs no threads and no shared state — the progress *is* the reply, arriving as
        it happens. No Content-Length: the body ends when the connection closes.
        """
        length = int(self.headers.get("Content-Length", 0))
        request = json.loads(self.rfile.read(length))

        self.send_response(200)
        self.send_header("Content-Type", "application/x-ndjson")
        self.end_headers()

        def emit(payload: dict) -> None:
            self.wfile.write((json.dumps(payload, default=str) + "\n").encode())
            self.wfile.flush()

        try:
            # Headers are already out, so a failure here has to travel as a line of the
            # body rather than as a status code.
            emit({"result": run_sweep_request(request, emit)})
        except Exception as exc:
            emit({"error": f"{type(exc).__name__}: {exc}"})

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
