"""Rank VAD + denoise settings against labelled clips, instead of tuning by ear.

`lab.html` compares a handful of slots on one clip by eye. This is the other half of
the same lab: replay a folder of labelled clips through a whole parameter grid and
score every combination, so a setting gets picked by a number that can be re-checked
after any later change to the chain.

    clips/train.wav
    clips/train.json    {"speech": [[1.2, 3.4], [6.0, 7.1]]}

Region times are seconds into the clip. `"speech": []` marks a clip that must never
trigger at all — pure background, or other people talking near the mic.

    uv run sweep.py label clips/train.wav    # propose a labels file, correct it by ear
    uv run sweep.py run --top 15             # rank the grid

Scores are only as good as the labels: three honest clips beat thirty guessed ones.
"""

import argparse
import asyncio
import hashlib
import json
import statistics
from itertools import product
from pathlib import Path

import lab
import speaker

CLIPS_DIR = Path(__file__).with_name("clips")
SPEECH_STATES = ("STARTING", "SPEAKING")

# Firing this soon before a labelled region is an early onset, not a false trigger.
GRACE_SECS = 0.2

# Every key `lab.analyze` needs. The grid overrides some of these; the rest stay fixed
# so a sweep result is a complete config that can be copied into `.env`.
BASE_CONFIG = {
    "backend": "ten",
    "confidence": 0.5,
    "start_secs": 0.2,
    "stop_secs": 0.45,
    "min_volume": 0.35,
    "noise_filter": "none",
    "noise_mix": 1.0,
    "smart_turn": False,
    # The speaker gate is off until both of these are set: an enrollment (a WAV or a
    # directory of them) and a similarity threshold to compare against.
    "speaker_enroll": None,
    "speaker_threshold": None,
    "speaker_model": speaker.DEFAULT_MODEL,
}

# Starting point, not a recommendation — copy it into a JSON file and edit.
DEFAULT_GRID = {
    "backend": ["ten", "silero", "firered"],
    "confidence": [0.4, 0.6, 0.8],
    "min_volume": [0.2, 0.35, 0.6],
    "noise_filter": ["none", "highpass", "highpass+rnnoise"],
}

# miss is a 0-1 ratio, the rest are per-minute counts or milliseconds. Weights say
# how many points one unit of each costs; edit them to match what actually annoys you.
DEFAULT_WEIGHTS = {
    "miss": 100.0,
    "false_per_min": 5.0,
    "onset_ms": 0.02,
    "tail_ms": 0.005,
}

METRICS = ("score", "miss", "false_per_min", "false_secs", "onset_ms", "tail_ms")

# Only meaningful for their own backend; dropped elsewhere so the grid doesn't expand
# into duplicate configs that take the same time to run and produce the same row.
BACKEND_KEYS = {"ten_threshold": "ten", "firered_threshold": "firered"}


def load_clips(folder: Path | str = CLIPS_DIR) -> tuple[list[dict], list[str]]:
    """Read `<name>.<audio>` + `<name>.json` pairs. Returns (clips, unlabelled names).

    Any format ffmpeg reads works, so a phone or Voice Recorder m4a can be dropped in
    as it is. One clip per stem: if both `me.wav` and `me.m4a` exist the first by
    suffix order wins, since they share the one labels file.
    """
    folder = Path(folder)
    recordings = sorted(
        (file for suffix in lab.AUDIO_SUFFIXES for file in folder.glob(f"*{suffix}")),
        key=lambda file: (file.stem, lab.AUDIO_SUFFIXES.index(file.suffix.lower())),
    )

    clips: list[dict] = []
    unlabelled: list[str] = []
    seen: set[str] = set()
    for recording in recordings:
        if recording.stem in seen:
            continue
        labels = recording.with_suffix(".json")
        if not labels.exists():
            unlabelled.append(recording.name)
            continue
        seen.add(recording.stem)
        regions = json.loads(labels.read_text(encoding="utf-8"))["speech"]
        clips.append(
            {
                "name": recording.stem,
                "pcm": lab.decode_audio(recording),
                "speech": [(float(start), float(end)) for start, end in regions],
            }
        )
    return clips, unlabelled


def fingerprint(config: dict) -> str:
    """A config's identity, so the same settings are never scored twice."""
    return json.dumps(config, sort_keys=True, default=str)


def expand_grid(grid: dict[str, list], base: dict | None = None) -> list[dict]:
    """Cartesian product of the grid, merged onto `base`, de-duplicated."""
    keys = list(grid)
    configs: list[dict] = []
    seen: set[str] = set()
    for values in product(*(grid[key] for key in keys)):
        config = {**(base or BASE_CONFIG), **dict(zip(keys, values))}
        for key, backend in BACKEND_KEYS.items():
            if config.get("backend") != backend:
                config.pop(key, None)
        if fingerprint(config) not in seen:
            seen.add(fingerprint(config))
            configs.append(config)
    return configs


def _speech_mask(regions, frames: int, frame_ms: float):
    """Per-frame boolean: does this frame's midpoint fall inside a labelled region?"""
    import numpy as np

    centres = (np.arange(frames) + 0.5) * frame_ms / 1000
    mask = np.zeros(frames, dtype=bool)
    for start, end in regions:
        mask |= (centres >= start) & (centres < end)
    return mask


def frame_stats(result: dict, regions) -> dict:
    """Raw counters for one clip under one config, ready to pool across clips.

    Denoise filters here truncate the tail rather than shifting the signal (see
    `MixedAudioFilter`), so labels taken from the raw clip still line up.
    """
    import numpy as np

    frame_ms = result["frame_ms"]
    detected = np.array([state in SPEECH_STATES for state in result["state"]], dtype=bool)
    frames = int(detected.size)
    if frames == 0:
        raise ValueError("analysis produced no frames — clip shorter than one VAD frame")

    mask = _speech_mask(regions, frames, frame_ms)
    # Onsets are judged against a grace-widened mask so that catching a word 100 ms
    # early counts as a good onset, while the miss/false-seconds totals stay strict.
    grace = _speech_mask(
        [(max(0.0, start - GRACE_SECS), end) for start, end in regions], frames, frame_ms
    )

    onsets = np.flatnonzero(detected & ~np.concatenate(([False], detected[:-1])))

    lags: list[float] = []
    tails: list[float] = []
    missed_regions = 0
    for start, end in regions:
        first = min(frames, max(0, int(start * 1000 / frame_ms)))
        last = min(frames, max(first, int(end * 1000 / frame_ms)))
        inside = np.flatnonzero(detected[first:last])
        if inside.size == 0:
            missed_regions += 1
            continue
        lags.append(float(inside[0]) * frame_ms)
        # How long the VAD keeps holding the turn open after the words stop — this is
        # what `stop_secs` buys and what the user feels as reply latency.
        after = np.flatnonzero(~detected[last:])
        tails.append(float(after[0] if after.size else frames - last) * frame_ms)

    return {
        "speech_frames": int(mask.sum()),
        "missed_frames": int((mask & ~detected).sum()),
        "noise_frames": int((~grace).sum()),
        "false_frames": int((detected & ~grace).sum()),
        "false_triggers": int(sum(1 for index in onsets if not grace[index])),
        "regions": len(regions),
        "missed_regions": missed_regions,
        "lags": lags,
        "tails": tails,
        "frame_ms": frame_ms,
    }


def combine(stats: list[dict]) -> dict:
    """Pool per-clip counters into one set of metrics."""
    speech = sum(item["speech_frames"] for item in stats)
    noise_secs = sum(item["noise_frames"] * item["frame_ms"] for item in stats) / 1000
    false_secs = sum(item["false_frames"] * item["frame_ms"] for item in stats) / 1000
    triggers = sum(item["false_triggers"] for item in stats)
    lags = [lag for item in stats for lag in item["lags"]]
    tails = [tail for item in stats for tail in item["tails"]]

    return {
        "miss": sum(item["missed_frames"] for item in stats) / speech if speech else 0.0,
        "false_per_min": triggers / (noise_secs / 60) if noise_secs else 0.0,
        "false_secs": false_secs,
        "false_triggers": triggers,
        "onset_ms": statistics.median(lags) if lags else 0.0,
        "tail_ms": statistics.median(tails) if tails else 0.0,
        "missed_regions": sum(item["missed_regions"] for item in stats),
        "regions": sum(item["regions"] for item in stats),
    }


def total_score(metrics: dict, weights: dict | None = None) -> float:
    """Weighted sum, lower is better. Sort by a raw column instead if you disagree."""
    weights = weights or DEFAULT_WEIGHTS
    return sum(weight * metrics[key] for key, weight in weights.items())


CHAIN_KEYS = ("noise_filter", "noise_mix", "rnnoise_quality", "highpass_hz")


def _chain_key(config: dict) -> str:
    return json.dumps({key: config.get(key) for key in CHAIN_KEYS}, sort_keys=True, default=str)


async def denoise(pcm: bytes, config: dict, cache_dir: Path | None) -> bytes:
    """Denoise a clip, remembering the result on disk.

    RNNoise is a recurrent net fed by soxr's stream resampler, and at HQ that
    resampler rounds two different ways run to run — only a couple of LSBs, but the
    net amplifies them until VAD decisions move and close scores swap places. Keeping
    the denoised audio means a clip is filtered once, ever, and every later sweep
    replays the same bytes. It also skips the slowest stage on repeat runs.

    Cached files are plain WAVs: listen to one to hear exactly what got scored. The
    key covers the audio and the filter settings but not the filter code, so delete
    the folder after changing `noise.py` or upgrading Pipecat.
    """
    if cache_dir is None:
        return await lab.apply_noise_chain(pcm, config)

    digest = hashlib.md5(pcm).hexdigest()[:16]
    chain = hashlib.md5(_chain_key(config).encode()).hexdigest()[:8]
    path = Path(cache_dir) / f"{digest}-{chain}.wav"
    if path.exists():
        return lab.decode_wav(path.read_bytes())

    filtered = await lab.apply_noise_chain(pcm, config)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(lab._wav_bytes(filtered))
    return filtered


def _speaker_gate(cache: dict, config: dict):
    """Load the model and enrollment once per (model, enrollment) pair."""
    key = (config["speaker_model"], config["speaker_enroll"])
    if key not in cache:
        extractor = speaker.create_extractor(config["speaker_model"])
        cache[key] = (extractor, speaker.enroll(extractor, config["speaker_enroll"]))
    return cache[key]


async def run_sweep(
    clips: list[dict],
    configs: list[dict],
    weights: dict | None = None,
    cache_dir: Path | None = None,
    on_progress=None,
) -> list[dict]:
    """Score every config against every clip. Denoise output is cached per chain.

    Denoising is the slow part (DeepFilterNet spawns a subprocess), and it does not
    depend on the VAD settings, so it runs once per clip+chain and every VAD variant
    replays the same filtered audio.
    """
    filtered: dict[tuple[str, str], bytes] = {}
    gates: dict[tuple, tuple] = {}
    rows: list[dict] = []

    # One step per config-and-clip, so a long sweep moves several times per config
    # instead of sitting still. The first step of a chain is much slower than the rest:
    # it is the one that actually denoises, the others read the cache.
    total = len(configs) * len(clips)
    done = 0

    def report(clip_name: str | None = None) -> None:
        if on_progress:
            on_progress({"done": done, "total": total, "clip": clip_name})

    report()
    for index, config in enumerate(configs):
        chain = _chain_key(config)
        stats: list[dict] = []
        try:
            for clip in clips:
                cached = filtered.get((clip["name"], chain))
                if cached is None:
                    cached = await denoise(clip["pcm"], config, cache_dir)
                    filtered[(clip["name"], chain)] = cached
                result = await lab.analyze(cached, {**config, "noise_filter": "none"})
                if config.get("speaker_enroll") and config.get("speaker_threshold") is not None:
                    extractor, enrolled = _speaker_gate(gates, config)
                    result = speaker.gate(
                        result, cached, enrolled, extractor, config["speaker_threshold"]
                    )
                stats.append(frame_stats(result, clip["speech"]))
                done += 1
                report(clip["name"])
        except Exception as exc:  # one bad backend must not abandon the sweep
            rows.append({"config": config, "error": f"{type(exc).__name__}: {exc}"})
            # Skip the clips this config never reached, or the bar stalls short of the end
            done = (index + 1) * len(clips)
            report()
            continue

        metrics = combine(stats)
        rows.append({"config": config, **metrics, "score": total_score(metrics, weights)})

    return rows


def propose_labels(pcm: bytes, sensitivity: float = 0.15) -> list[list[float]]:
    """Energy-gate a clip into speech regions — a first draft to correct, not truth.

    It keys off the clip's own loudest frame, so it only works on a clip where the
    speech is clearly louder than the background. On a real train recording expect to
    fix the numbers by ear.
    """
    import numpy as np

    frame = int(lab.SAMPLE_RATE * 0.02)
    samples = np.frombuffer(pcm, dtype="<i2").astype("float32") / 32768.0
    usable = samples[: samples.size // frame * frame].reshape(-1, frame)
    if usable.size == 0:
        return []

    rms = np.sqrt((usable**2).mean(axis=1))
    loud = rms > max(0.01, sensitivity * rms.max())

    regions: list[list[float]] = []
    for index, is_loud in enumerate(loud):
        start = index * 0.02
        if is_loud and regions and start - regions[-1][1] <= 0.3:
            regions[-1][1] = start + 0.02  # bridge the pause inside one utterance
        elif is_loud:
            regions.append([start, start + 0.02])
    return [region for region in regions if region[1] - region[0] >= 0.2]


def _varying_keys(configs: list[dict]) -> list[str]:
    keys = sorted({key for config in configs for key in config})
    return [key for key in keys if len({str(c.get(key)) for c in configs}) > 1]


def _print_table(rows: list[dict], configs: list[dict], top: int, sort: str) -> None:
    ranked = sorted(
        (row for row in rows if "error" not in row), key=lambda row: row[sort]
    )[:top]
    keys = _varying_keys(configs) or ["backend"]

    header = f"{'score':>7} {'miss%':>6} {'fals/min':>8} {'fals_s':>7} {'onset':>6} {'tail':>6}  "
    print(header + "  ".join(f"{key:>10}" for key in keys))
    for row in ranked:
        print(
            f"{row['score']:>7.1f} {row['miss'] * 100:>6.1f} {row['false_per_min']:>8.2f} "
            f"{row['false_secs']:>7.1f} {row['onset_ms']:>6.0f} {row['tail_ms']:>6.0f}  "
            + "  ".join(f"{str(row['config'].get(key)):>10}" for key in keys)
        )

    for row in rows:
        if "error" in row:
            print(f"  skipped {row['config'].get('backend')}: {row['error']}")


def _command_run(args: argparse.Namespace) -> None:
    grid, weights = DEFAULT_GRID, DEFAULT_WEIGHTS
    if args.grid:
        data = json.loads(Path(args.grid).read_text(encoding="utf-8"))
        grid = data.get("grid", data)
        weights = data.get("weights", DEFAULT_WEIGHTS)

    clips, unlabelled = load_clips(args.clips)
    for name in unlabelled:
        print(f"  no labels for {name} — skipped (write {Path(name).stem}.json)")
    if not clips:
        raise SystemExit(f"no labelled clips in {args.clips}")

    configs = expand_grid(grid)
    seconds = sum(len(clip["pcm"]) for clip in clips) / 2 / lab.SAMPLE_RATE
    print(f"{len(configs)} configs x {len(clips)} clips ({seconds:.0f}s of audio)")

    cache_dir = None if args.no_cache else Path(args.clips) / ".cache"
    rows = asyncio.run(run_sweep(clips, configs, weights, cache_dir))
    _print_table(rows, configs, args.top, args.sort)

    if args.json:
        Path(args.json).write_text(json.dumps(rows, indent=2, default=str), encoding="utf-8")
        print(f"full results -> {args.json}")


def _command_label(args: argparse.Namespace) -> None:
    wav = Path(args.wav)
    target = wav.with_suffix(".json")
    if target.exists() and not args.force:
        raise SystemExit(f"{target} exists — pass --force to overwrite")

    regions = propose_labels(lab.decode_audio(wav), args.sensitivity)
    target.write_text(json.dumps({"speech": regions}, indent=2), encoding="utf-8")
    print(f"{len(regions)} regions -> {target}  (check them before trusting a sweep)")


def _command_similarity(args: argparse.Namespace) -> None:
    """Print how close each clip sounds to the enrollment, to place a threshold."""
    extractor = speaker.create_extractor(args.model)
    enrolled = speaker.enroll(extractor, args.enroll)

    for name in args.wav:
        pcm = lab.decode_audio(name)
        score = speaker.similarity(speaker.embed(extractor, pcm), enrolled)
        print(f"{score:>6.3f}  {name}")

    print(
        "\nput the threshold in the gap between your clips and the strangers'. "
        "A narrow gap means the gate will cost you real turns."
    )


def main() -> None:
    from loguru import logger

    # Pipecat logs a model load per analyzer, which is one line per config per clip.
    logger.remove()

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    commands = parser.add_subparsers(dest="command", required=True)

    run = commands.add_parser("run", help="score the grid against every labelled clip")
    run.add_argument("--clips", default=str(CLIPS_DIR))
    run.add_argument("--grid", help="JSON file: {\"grid\": {...}, \"weights\": {...}}")
    run.add_argument("--top", type=int, default=20)
    run.add_argument("--sort", default="score", choices=METRICS)
    run.add_argument("--json", help="write every row, including losers, to this file")
    run.add_argument(
        "--no-cache",
        action="store_true",
        help="re-denoise instead of reusing clips/.cache (scores will drift a little)",
    )
    run.set_defaults(func=_command_run)

    label = commands.add_parser("label", help="propose a labels file for one clip")
    label.add_argument("wav")
    label.add_argument("--sensitivity", type=float, default=0.15)
    label.add_argument("--force", action="store_true")
    label.set_defaults(func=_command_label)

    close = commands.add_parser(
        "similarity", help="score clips against an enrollment, to pick a threshold"
    )
    close.add_argument("wav", nargs="+")
    close.add_argument("--enroll", required=True, help="a WAV, or a directory of them")
    close.add_argument("--model", default=speaker.DEFAULT_MODEL)
    close.set_defaults(func=_command_similarity)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
