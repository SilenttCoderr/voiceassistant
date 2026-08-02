"""Speaker gate: is this the enrolled voice, or someone else in the room?

VAD answers "is anyone talking". In a train or a boardroom that is the wrong question
— the agent should only wake for one person. A speaker embedding turns a stretch of
speech into a vector and compares it against an enrolled recording, so a stranger two
seats away can be dropped before the agent ever answers them.

The model is a plain ONNX file from the sherpa-onnx zoo and the "backend" is just
which file you point at, so TitaNet and CAM++ swap without a code change:

    https://github.com/k2-fsa/sherpa-onnx/releases/tag/speaker-recongition-models
    pretrained_models/speaker/3dspeaker_speech_campplus_sv_zh_en_16k-common_advanced.onnx
    pretrained_models/speaker/nemo_en_titanet_large.onnx

Embeddings are text- and language-independent — they model how a voice sounds, not
what it says — which is why an English-trained model is worth measuring on Hindi.
Measure it with `sweep.py`; there is no Hinglish-trained model to fall back on.
"""

from pathlib import Path

SAMPLE_RATE = 16000
MODEL_DIR = Path(__file__).with_name("pretrained_models") / "speaker"
DEFAULT_MODEL = "3dspeaker_speech_campplus_sv_zh_en_16k-common_advanced.onnx"

SPEECH_STATES = ("STARTING", "SPEAKING")

# An embedding from a fragment this short is closer to noise than to a voiceprint, so
# the gate abstains instead of guessing. Those segments stay in and stay counted.
MIN_GATE_SECS = 0.4


def create_extractor(model: str | Path | None = None, num_threads: int = 1):
    """Load an ONNX speaker embedding model through sherpa-onnx."""
    try:
        import sherpa_onnx
    except ModuleNotFoundError as exc:
        if exc.name != "sherpa_onnx":
            raise
        raise RuntimeError(
            "Speaker gate selected; install it with 'uv sync --extra speaker'"
        ) from exc

    path = resolve_model(model)
    config = sherpa_onnx.SpeakerEmbeddingExtractorConfig(
        model=str(path), num_threads=num_threads
    )
    if not config.validate():
        raise RuntimeError(f"sherpa-onnx rejected the speaker model at {path}")
    return sherpa_onnx.SpeakerEmbeddingExtractor(config)


def resolve_model(model: str | Path | None) -> Path:
    """Accept a full path, or a bare file name inside `pretrained_models/speaker`."""
    path = Path(model or DEFAULT_MODEL)
    if not path.exists():
        path = MODEL_DIR / path.name
    if not path.exists():
        raise RuntimeError(
            f"speaker model not found: {model or DEFAULT_MODEL}. Download one into "
            f"{MODEL_DIR} from the sherpa-onnx speaker-recongition-models release."
        )
    return path


def embed(extractor, pcm: bytes):
    """Embed 16 kHz mono int16 PCM as one L2-normalised float32 vector.

    Normalising here rather than at the comparison keeps every stored enrollment on
    the same scale, so averaging several recordings stays meaningful.
    """
    import numpy as np

    samples = np.frombuffer(pcm, dtype="<i2").astype("float32") / 32768.0
    stream = extractor.create_stream()
    stream.accept_waveform(sample_rate=SAMPLE_RATE, waveform=samples)
    stream.input_finished()
    if not extractor.is_ready(stream):
        raise ValueError("clip is too short for the speaker model")

    vector = np.array(extractor.compute(stream), dtype="float32")
    norm = float(np.linalg.norm(vector))
    return vector / norm if norm else vector


def similarity(one, other) -> float:
    """Cosine similarity. Roughly -1 to 1; same speaker sits high, strangers low."""
    import numpy as np

    scale = float(np.linalg.norm(one)) * float(np.linalg.norm(other))
    return float(np.dot(one, other) / scale) if scale else 0.0


def enroll(extractor, source: str | Path):
    """Average the embeddings of a WAV file, or of every WAV in a directory.

    More recordings help: one clip pins the model to that day's mic, room and mood.
    """
    import lab
    import numpy as np

    path = Path(source)
    if path.is_dir():
        files = sorted(
            file for suffix in lab.AUDIO_SUFFIXES for file in path.glob(f"*{suffix}")
        )
    else:
        files = [path]
    if not files:
        raise RuntimeError(f"no enrollment audio in {path}")

    vectors = [embed(extractor, lab.decode_audio(file)) for file in files]
    mean = np.mean(vectors, axis=0)
    norm = float(np.linalg.norm(mean))
    return mean / norm if norm else mean


def segments(states: list[str], frame_ms: float) -> list[tuple[int, int]]:
    """Contiguous runs of speech as [start, end) frame indices."""
    found: list[tuple[int, int]] = []
    start = None
    for index, state in enumerate(states):
        if state in SPEECH_STATES and start is None:
            start = index
        elif state not in SPEECH_STATES and start is not None:
            found.append((start, index))
            start = None
    if start is not None:
        found.append((start, len(states)))
    return found


def gate(result: dict, pcm: bytes, enrolled, extractor, threshold: float) -> dict:
    """Silence the speech segments that do not sound like the enrolled speaker.

    Offline shortcut: this judges each segment as a whole. Live, the decision is due
    after the first few hundred milliseconds, so read the result as the ceiling of
    what the gate can do rather than what it will do inside the pipeline.
    """
    frame_ms = result["frame_ms"]
    frame_bytes = int(frame_ms / 1000 * SAMPLE_RATE) * 2
    states = list(result["state"])
    judged: list[dict] = []

    for start, end in segments(states, frame_ms):
        audio = pcm[start * frame_bytes : end * frame_bytes]
        if len(audio) / 2 / SAMPLE_RATE < MIN_GATE_SECS:
            judged.append({"start": start, "end": end, "similarity": None, "kept": True})
            continue

        score = similarity(embed(extractor, audio), enrolled)
        keep = score >= threshold
        if not keep:
            states[start:end] = ["QUIET"] * (end - start)
        judged.append({"start": start, "end": end, "similarity": score, "kept": keep})

    return {**result, "state": states, "speaker": judged}
