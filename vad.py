import os
import sys
from array import array
from math import isfinite

from pipecat.audio.vad.vad_analyzer import VADAnalyzer, VADParams

SUPPORTED_BACKENDS = ("gemini", "silero", "ten", "firered", "cobra")


def _float_env(
    name: str,
    default: float,
    *,
    minimum: float | None = None,
    maximum: float | None = None,
) -> float:
    try:
        value = float(os.getenv(name, str(default)))
    except ValueError as exc:
        raise RuntimeError(f"{name} must be a number") from exc
    if not isfinite(value):
        raise RuntimeError(f"{name} must be a finite number")
    if minimum is not None and value < minimum or maximum is not None and value > maximum:
        if maximum is not None:
            raise RuntimeError(f"{name} must be between {minimum:g} and {maximum:g}")
        raise RuntimeError(f"{name} must be at least {minimum:g}")
    return value


def common_vad_params() -> VADParams:
    return VADParams(
        confidence=_float_env("VAD_CONFIDENCE", 0.5, minimum=0, maximum=1),
        start_secs=_float_env("VAD_START_SECS", 0.2, minimum=0),
        stop_secs=_float_env("VAD_STOP_SECS", 0.45, minimum=0),
        min_volume=_float_env("VAD_MIN_VOLUME", 0.35, minimum=0, maximum=1),
    )


class CobraVADAnalyzer(VADAnalyzer):
    def __init__(self, access_key: str, params: VADParams | None = None):
        try:
            import pvcobra
        except ModuleNotFoundError as exc:
            if exc.name != "pvcobra":
                raise
            raise RuntimeError(
                "Cobra selected; install the Cobra optional dependencies with "
                "'uv sync --extra cobra'"
            ) from exc

        self._cobra = pvcobra.create(access_key=access_key)
        cobra_sample_rate = self._cobra.sample_rate
        if cobra_sample_rate != 16000:
            self._cobra.delete()
            self._cobra = None
            raise RuntimeError(
                f"Cobra requires {cobra_sample_rate} Hz, expected 16000 Hz"
            )
        super().__init__(sample_rate=16000, params=params)

    def num_frames_required(self) -> int:
        return self._cobra.frame_length

    def voice_confidence(self, buffer: bytes) -> float:
        samples = array("h")
        samples.frombytes(buffer)
        if sys.byteorder != "little":
            samples.byteswap()
        return self._cobra.process(samples)

    async def cleanup(self):
        cobra, self._cobra = self._cobra, None
        try:
            if cobra is not None:
                cobra.delete()
        finally:
            await super().cleanup()


class TenVADAnalyzer(VADAnalyzer):
    def __init__(self, threshold: float, params: VADParams | None = None):
        try:
            from ten_vad import TenVad
        except ModuleNotFoundError as exc:
            if exc.name != "ten_vad":
                raise
            raise RuntimeError(
                "TEN selected; install it with 'uv sync --extra ten'"
            ) from exc

        self._ten = TenVad(hop_size=256, threshold=threshold)
        super().__init__(sample_rate=16000, params=params)

    def num_frames_required(self) -> int:
        return 256

    def voice_confidence(self, buffer: bytes) -> float:
        import numpy as np

        probability, flag = self._ten.process(np.frombuffer(buffer, dtype="<i2"))
        return probability if flag else 0.0

    async def cleanup(self):
        self._ten = None
        await super().cleanup()


def create_vad(backend: str | None = None) -> VADAnalyzer | None:
    selected = (backend or os.getenv("VAD_BACKEND", "gemini")).strip().lower()
    if selected not in SUPPORTED_BACKENDS:
        raise ValueError(
            "VAD_BACKEND must be one of: " + ", ".join(SUPPORTED_BACKENDS)
        )
    if selected == "gemini":
        return None

    params = common_vad_params()
    if selected == "silero":
        from pipecat.audio.vad.silero import SileroVADAnalyzer

        return SileroVADAnalyzer(sample_rate=16000, params=params)
    if selected == "ten":
        threshold = _float_env("TEN_VAD_THRESHOLD", 0.6, minimum=0, maximum=1)
        return TenVADAnalyzer(threshold=threshold, params=params)
    if selected == "firered":
        model_dir = os.getenv("FIREREDVAD_MODEL_DIR")
        if not model_dir:
            raise RuntimeError("FIREREDVAD_MODEL_DIR is required when VAD_BACKEND=firered")
        speech_threshold = _float_env(
            "FIREREDVAD_SPEECH_THRESHOLD", 0.6, minimum=0, maximum=1
        )
        use_gpu_value = os.getenv("FIREREDVAD_USE_GPU", "0")
        if use_gpu_value not in ("0", "1"):
            raise RuntimeError("FIREREDVAD_USE_GPU must be exactly 0 or 1")
        try:
            from pipecat_firered_vad import FireVadAnalyzer
        except ModuleNotFoundError as exc:
            if exc.name != "pipecat_firered_vad":
                raise
            raise RuntimeError(
                "FireRed selected; install it with 'uv sync --extra firered'"
            ) from exc
        from fireredvad.core.constants import FRAME_LENGTH_SAMPLE

        class CompatibleFireVadAnalyzer(FireVadAnalyzer):
            def num_frames_required(self) -> int:
                return FRAME_LENGTH_SAMPLE

            def voice_confidence(self, buffer: bytes) -> float:
                import numpy as np

                frame = np.frombuffer(buffer, dtype="<i2")
                if len(frame) != FRAME_LENGTH_SAMPLE:
                    return 0.0
                result = self._vad.detect_frame(frame)
                probability = getattr(result, "raw_prob", 0.0)
                return float(np.clip(probability, 0.0, 1.0))

        return CompatibleFireVadAnalyzer(
            model_dir=model_dir,
            sample_rate=16000,
            params=params,
            speech_threshold=speech_threshold,
            use_gpu=use_gpu_value == "1",
        )

    access_key = os.getenv("PICOVOICE_ACCESS_KEY")
    if not access_key:
        raise RuntimeError("PICOVOICE_ACCESS_KEY is required when VAD_BACKEND=cobra")
    return CobraVADAnalyzer(access_key, params=params)
