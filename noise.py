"""Local audio filters that Pipecat does not ship."""

from pipecat.audio.filters.base_audio_filter import BaseAudioFilter
from pipecat.frames.frames import FilterControlFrame, FilterEnableFrame


class MixedAudioFilter(BaseAudioFilter):
    """Blend a filter's output back with the untouched input.

    Denoisers that remove everything tend to sound robotic: suppressing a band to silence
    leaves "musical noise", and the ear reads the total absence of room tone as artificial.
    Letting a little of the original through masks those artifacts and keeps speech sounding
    natural, at the cost of some residual noise. ``wet=1.0`` is the fully denoised signal,
    ``wet=0.0`` the original.

    The inner filter buffers, so its output lags the input. Holding the dry signal in a FIFO
    and consuming exactly as many samples as the filter emits keeps the two aligned.
    """

    def __init__(self, inner: BaseAudioFilter, wet: float = 0.85) -> None:
        if not 0.0 <= wet <= 1.0:
            raise RuntimeError("NOISE_MIX must be between 0 and 1")
        self._inner = inner
        self._wet = wet
        self._dry = bytearray()

    async def start(self, sample_rate: int) -> None:
        self._dry.clear()
        await self._inner.start(sample_rate)

    async def stop(self) -> None:
        self._dry.clear()
        await self._inner.stop()

    async def process_frame(self, frame: FilterControlFrame) -> None:
        await self._inner.process_frame(frame)

    async def filter(self, audio: bytes) -> bytes:
        import numpy as np

        self._dry.extend(audio)
        wet = await self._inner.filter(audio)
        if not wet:
            return b""

        dry = bytes(self._dry[: len(wet)])
        del self._dry[: len(wet)]
        if len(dry) < len(wet):
            dry += b"\x00" * (len(wet) - len(dry))

        blended = (
            np.frombuffer(wet, dtype="<i2").astype("float32") * self._wet
            + np.frombuffer(dry, dtype="<i2").astype("float32") * (1.0 - self._wet)
        )
        return np.clip(blended, -32768, 32767).astype("<i2").tobytes()


class ChainedAudioFilter(BaseAudioFilter):
    """Run several audio filters in sequence, left to right.

    Lets cheap fixed filtering run before an expensive model, e.g. dropping rumble with a
    high-pass so the denoiser only has to deal with what is left in the speech band.
    """

    def __init__(self, filters: list[BaseAudioFilter]) -> None:
        self._filters = filters

    async def start(self, sample_rate: int) -> None:
        for audio_filter in self._filters:
            await audio_filter.start(sample_rate)

    async def stop(self) -> None:
        for audio_filter in self._filters:
            await audio_filter.stop()

    async def process_frame(self, frame: FilterControlFrame) -> None:
        for audio_filter in self._filters:
            await audio_filter.process_frame(frame)

    async def filter(self, audio: bytes) -> bytes:
        for audio_filter in self._filters:
            audio = await audio_filter.filter(audio)
            # Filters that buffer internally return b"" until they have a full frame;
            # passing that on would make later stages resample empty audio.
            if not audio:
                return b""
        return audio


class HighPassFilter(BaseAudioFilter):
    """Butterworth high-pass, for rumble that sits below the speech band.

    Train and traffic noise is concentrated under ~100 Hz, where speech carries almost
    nothing. Removing it before the VAD raises the contrast between speech and the room
    without touching the speech band itself. Unlike a denoiser this is a fixed filter: it
    cannot remove competing voices, only low-frequency energy.
    """

    def __init__(self, cutoff_hz: float = 100.0, order: int = 2) -> None:
        self._cutoff_hz = cutoff_hz
        self._order = order
        self._filtering = True
        self._coefficients = None
        self._state = None

    async def start(self, sample_rate: int) -> None:
        from scipy.signal import butter, lfilter_zi

        nyquist = sample_rate / 2
        if not 0 < self._cutoff_hz < nyquist:
            raise RuntimeError(
                f"HIGHPASS_HZ must be between 0 and {nyquist:g} for {sample_rate} Hz audio"
            )
        b, a = butter(self._order, self._cutoff_hz / nyquist, btype="highpass")
        self._coefficients = (b, a)
        # Carry filter state across chunks, otherwise every chunk boundary rings.
        self._state = lfilter_zi(b, a)

    async def stop(self) -> None:
        self._coefficients = None
        self._state = None

    async def process_frame(self, frame: FilterControlFrame) -> None:
        if isinstance(frame, FilterEnableFrame):
            self._filtering = frame.enable

    async def filter(self, audio: bytes) -> bytes:
        import numpy as np
        from scipy.signal import lfilter

        if not self._filtering or self._coefficients is None or not audio:
            return audio

        b, a = self._coefficients
        samples = np.frombuffer(audio, dtype="<i2").astype("float32")
        filtered, self._state = lfilter(b, a, samples, zi=self._state)
        return np.clip(filtered, -32768, 32767).astype("<i2").tobytes()
