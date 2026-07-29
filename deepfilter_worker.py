"""DeepFilterNet denoiser, run in its own Python 3.11 environment.

DeepFilterNet cannot live in the project venv: `deepfilterlib` has no cp312 wheel, the
package pins `numpy<2`, and it only works with `torchaudio<2.1` (newer releases dropped
`torchaudio.backend`, which it imports). Rather than downgrade the whole project, this
script is executed by `uv run --python 3.11 --with ...`, reading a WAV on stdin and
writing the denoised WAV to stdout.

Not suitable for the live agent: `enhance()` is an utterance-level API, so this is for
offline comparison in the lab only.
"""

import io
import sys
import wave

import numpy as np
import torch
from df.enhance import enhance, init_df

TARGET_RATE = 48000


def read_wav(raw: bytes) -> tuple[np.ndarray, int]:
    with wave.open(io.BytesIO(raw)) as source:
        rate = source.getframerate()
        pcm = source.readframes(source.getnframes())
    return np.frombuffer(pcm, dtype="<i2").astype("float32") / 32768.0, rate


def write_wav(samples: np.ndarray, rate: int) -> bytes:
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as out:
        out.setnchannels(1)
        out.setsampwidth(2)
        out.setframerate(rate)
        out.writeframes((np.clip(samples, -1.0, 1.0) * 32767).astype("<i2").tobytes())
    return buffer.getvalue()


def resample(samples: np.ndarray, source_rate: int, target_rate: int) -> np.ndarray:
    if source_rate == target_rate:
        return samples
    count = int(round(samples.size * target_rate / source_rate))
    source_positions = np.linspace(0, samples.size - 1, samples.size)
    target_positions = np.linspace(0, samples.size - 1, count)
    return np.interp(target_positions, source_positions, samples).astype("float32")


def main() -> None:
    # File in, file out: the model logs to stdout, so a pipe would be corrupted.
    source_path, target_path = sys.argv[1], sys.argv[2]
    with open(source_path, "rb") as handle:
        samples, rate = read_wav(handle.read())

    model, state, _ = init_df()
    upsampled = resample(samples, rate, TARGET_RATE)
    enhanced = enhance(model, state, torch.from_numpy(upsampled).unsqueeze(0))
    restored = resample(enhanced.squeeze(0).numpy(), TARGET_RATE, rate)

    # Length can drift by a sample or two through the two resamples.
    if restored.size < samples.size:
        restored = np.pad(restored, (0, samples.size - restored.size))
    with open(target_path, "wb") as handle:
        handle.write(write_wav(restored[: samples.size], rate))


if __name__ == "__main__":
    main()
