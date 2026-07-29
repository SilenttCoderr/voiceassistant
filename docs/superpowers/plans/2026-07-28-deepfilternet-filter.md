# DeepFilterNet Filter Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: use `subagent-driven-development` (recommended) or `executing-plans` to implement this plan task-by-task.

**Goal:** Add DeepFilterNet as an optional in-process Windows noise filter shared by the live Pipecat agent and offline VAD lab.

**Architecture:** A project-owned `DeepFilterNetFilter` implements Pipecat's `BaseAudioFilter`, retains DeepFilterNet model/STFT state across calls, and owns 16 kHz to 48 kHz streaming resampling. `bot.create_audio_filter()` remains the single filter factory used by both production and `lab.py`.

**Tech Stack:** Python 3.11, Pipecat 1.x, DeepFilterNet 0.5.6, PyTorch, NumPy 1.x, SOXR, pytest.

## Global Constraints

- Use `requires-python = ">=3.11,<3.12"`.
- Pin `deepfilternet==0.5.6` and `numpy>=1.22,<2`.
- Keep transport, VAD, and pipeline audio at 16 kHz.
- Keep DeepFilterNet optional and imported only when selected.
- Use the default pretrained DeepFilterNet3 model.
- Do not call clip-oriented `enhance()` once per transport chunk.
- Do not silently bypass filtering after model startup failure.
- Do not change the default `NOISE_FILTER=browser`.
- Do not edit `.env`.
- Commit commands are checkpoints only; execute them only after explicit user approval.

---

### Task 1: Python 3.11 and Optional Dependencies

**Files:**
- Create: `.python-version`
- Modify: `pyproject.toml:5-16`
- Modify: `uv.lock`

**Interfaces:**
- Produces: `uv sync --extra deepfilter` on Windows Python 3.11.

- [ ] **Step 1: Pin the project interpreter**

Create `.python-version`:

```text
3.11
```

Change `pyproject.toml`:

```toml
requires-python = ">=3.11,<3.12"
```

- [ ] **Step 2: Add the optional extra**

Add under `[project.optional-dependencies]`:

```toml
deepfilter = [
    "deepfilternet==0.5.6",
    "numpy>=1.22,<2",
    "torch>=2,<3",
]
```

PyTorch is explicit because DeepFilterNet imports it but does not declare it in its package metadata.

- [ ] **Step 3: Resolve and install with Python 3.11**

Run:

```powershell
rtk uv lock --python 3.11
rtk uv sync --python 3.11 --extra deepfilter
```

Expected: resolution succeeds using a CPython 3.11 Windows `deepfilterlib` wheel; no local Rust build is attempted.

- [ ] **Step 4: Verify runtime versions and imports**

Run:

```powershell
rtk uv run python -c "import sys, numpy, torch, df, libdf; print(sys.version_info[:2], numpy.__version__, torch.__version__)"
```

Expected: Python reports `(3, 11)`, NumPy reports a version below `2`, and imports succeed.

- [ ] **Step 5: Verify the existing suite on the migrated runtime**

Run:

```powershell
rtk uv run pytest -v
```

Expected: all non-optional tests pass before DeepFilterNet code is added. The RNNoise integration test may skip because `uv sync --extra deepfilter` removes extras not named in that command. Stop and diagnose any failure.

- [ ] **Step 6: Checkpoint commit after approval**

```powershell
rtk git add .python-version pyproject.toml uv.lock
rtk git commit -m "build: use Python 3.11 for DeepFilterNet"
```

---

### Task 2: Stateful DeepFilterNet Audio Filter

**Files:**
- Create: `deepfilter_filter.py`
- Create: `tests/test_deepfilter_filter.py`

**Interfaces:**
- Produces: `DeepFilterNetFilter(BaseAudioFilter)` with `start()`, `filter()`, `process_frame()`, and `stop()`.
- Depends on: Pipecat `SOXRStreamAudioResampler`, DeepFilterNet's `init_df()` and feature/inference helpers.

- [ ] **Step 1: Write failing lifecycle and buffering tests**

Create `tests/test_deepfilter_filter.py`:

```python
import asyncio

import pytest
from pipecat.frames.frames import FilterEnableFrame

import deepfilter_filter


class PassThroughResampler:
    def __init__(self, **kwargs):
        pass

    async def resample(self, audio, source_rate, target_rate):
        return audio


def prepared_filter(monkeypatch):
    monkeypatch.setattr(
        deepfilter_filter, "SOXRStreamAudioResampler", PassThroughResampler
    )
    audio_filter = deepfilter_filter.DeepFilterNetFilter()
    enhanced = []

    def load_model():
        audio_filter._hop_samples = 4

    def enhance(audio):
        enhanced.append(audio)
        return audio

    monkeypatch.setattr(audio_filter, "_load_model", load_model)
    monkeypatch.setattr(audio_filter, "_enhance_48k", enhance)
    asyncio.run(audio_filter.start(16000))
    return audio_filter, enhanced


def test_filter_buffers_complete_model_hops_and_reuses_state(monkeypatch):
    audio_filter, enhanced = prepared_filter(monkeypatch)
    half_hop = b"\x01\x00" * 2

    assert asyncio.run(audio_filter.filter(half_hop)) == b""
    assert asyncio.run(audio_filter.filter(half_hop)) == half_hop * 2
    assert asyncio.run(audio_filter.filter(half_hop * 2)) == half_hop * 2
    assert enhanced == [half_hop * 2, half_hop * 2]


def test_filter_enable_frame_bypasses_processing(monkeypatch):
    audio_filter, enhanced = prepared_filter(monkeypatch)
    audio = b"\x01\x00" * 4

    asyncio.run(audio_filter.process_frame(FilterEnableFrame(enable=False)))

    assert asyncio.run(audio_filter.filter(audio)) == audio
    assert enhanced == []


def test_filter_requires_start():
    with pytest.raises(RuntimeError, match="DeepFilterNet filter has not started"):
        asyncio.run(deepfilter_filter.DeepFilterNetFilter().filter(b"\x00\x00"))


def test_filter_rejects_non_project_sample_rate(monkeypatch):
    monkeypatch.setattr(
        deepfilter_filter, "SOXRStreamAudioResampler", PassThroughResampler
    )

    with pytest.raises(RuntimeError, match="DeepFilterNet filter requires 16000 Hz input"):
        asyncio.run(deepfilter_filter.DeepFilterNetFilter().start(48000))


def test_stop_releases_runtime_state(monkeypatch):
    audio_filter, _ = prepared_filter(monkeypatch)

    asyncio.run(audio_filter.stop())

    assert audio_filter._ready is False
    assert audio_filter._model is None
    assert audio_filter._df_state is None
    assert audio_filter._buffer == bytearray()
```

- [ ] **Step 2: Run tests and verify RED**

Run:

```powershell
rtk uv run pytest tests/test_deepfilter_filter.py -v
```

Expected: collection fails because `deepfilter_filter.py` does not exist.

- [ ] **Step 3: Implement the filter lifecycle and streaming inference**

Create `deepfilter_filter.py`:

```python
import numpy as np

from pipecat.audio.filters.base_audio_filter import BaseAudioFilter
from pipecat.audio.resamplers.soxr_stream_resampler import SOXRStreamAudioResampler
from pipecat.frames.frames import FilterControlFrame, FilterEnableFrame


class DeepFilterNetFilter(BaseAudioFilter):
    def __init__(self) -> None:
        self._filtering = True
        self._ready = False
        self._sample_rate = 0
        self._buffer = bytearray()
        self._resampler_in = None
        self._resampler_out = None
        self._model = None
        self._df_state = None
        self._hop_samples = 0
        self._nb_df = 0
        self._torch = None
        self._df_features = None
        self._as_complex = None
        self._device = None

    def _load_model(self) -> None:
        import torch
        from df.enhance import df_features, init_df
        from df.model import ModelParams
        from df.modules import get_device
        from df.utils import as_complex

        model, df_state, _, _ = init_df(log_file=None)
        model.eval()
        device = get_device()
        if hasattr(model, "reset_h0"):
            model.reset_h0(batch_size=1, device=device)

        self._model = model
        self._df_state = df_state
        self._hop_samples = df_state.hop_size()
        self._nb_df = getattr(
            model, "nb_df", getattr(model, "df_bins", ModelParams().nb_df)
        )
        self._torch = torch
        self._df_features = df_features
        self._as_complex = as_complex
        self._device = device

    async def start(self, sample_rate: int):
        if sample_rate != 16000:
            raise RuntimeError("DeepFilterNet filter requires 16000 Hz input")
        self._sample_rate = sample_rate
        self._buffer.clear()
        self._resampler_in = SOXRStreamAudioResampler(quality="QQ")
        self._resampler_out = SOXRStreamAudioResampler(quality="QQ")
        self._load_model()
        self._ready = True

    async def stop(self):
        self._ready = False
        self._buffer.clear()
        self._resampler_in = None
        self._resampler_out = None
        self._model = None
        self._df_state = None
        self._torch = None
        self._df_features = None
        self._as_complex = None
        self._device = None

    async def process_frame(self, frame: FilterControlFrame):
        if isinstance(frame, FilterEnableFrame):
            self._filtering = frame.enable

    def _enhance_48k(self, audio: bytes) -> bytes:
        samples = np.frombuffer(audio, dtype="<i2").astype("float32") / 32768.0
        tensor = self._torch.from_numpy(samples).unsqueeze(0)
        spec, erb_feat, spec_feat = self._df_features(
            tensor, self._df_state, self._nb_df, device=self._device
        )
        enhanced = self._model(spec.clone(), erb_feat, spec_feat)[0].cpu()
        enhanced = self._as_complex(enhanced.squeeze(1))
        output = np.asarray(self._df_state.synthesis(enhanced.numpy())).reshape(-1)
        return (np.clip(output, -1.0, 1.0) * 32767).astype("<i2").tobytes()

    async def filter(self, audio: bytes) -> bytes:
        if not self._ready:
            raise RuntimeError("DeepFilterNet filter has not started")
        if not self._filtering:
            return audio

        resampled = await self._resampler_in.resample(audio, self._sample_rate, 48000)
        self._buffer.extend(resampled)
        hop_bytes = self._hop_samples * 2
        complete_bytes = len(self._buffer) // hop_bytes * hop_bytes
        if complete_bytes == 0:
            return b""

        chunk = bytes(self._buffer[:complete_bytes])
        del self._buffer[:complete_bytes]
        enhanced = self._enhance_48k(chunk)
        return await self._resampler_out.resample(enhanced, 48000, self._sample_rate)
```

- [ ] **Step 4: Run filter tests and verify GREEN**

Run:

```powershell
rtk uv run pytest tests/test_deepfilter_filter.py -v
```

Expected: all five tests pass.

- [ ] **Step 5: Checkpoint commit after approval**

```powershell
rtk git add deepfilter_filter.py tests/test_deepfilter_filter.py
rtk git commit -m "feat: add stateful DeepFilterNet audio filter"
```

---

### Task 3: Filter Factory Integration

**Files:**
- Modify: `tests/test_bot.py:135-230`
- Modify: `bot.py:82-111`

**Interfaces:**
- Consumes: `NOISE_FILTER=deepfilter`.
- Produces: a lazily constructed `DeepFilterNetFilter`.

- [ ] **Step 1: Extend failing factory tests**

Update the invalid filter expectation:

```python
match="NOISE_FILTER must be one of: browser, rnnoise, koala, deepfilter"
```

Add this case to `test_optional_noise_filters_are_constructed_lazily`:

```python
(
    "deepfilter",
    "df",
    "deepfilter_filter",
    "DeepFilterNetFilter",
    {},
),
```

Add this case to `test_optional_filter_dependency_errors_are_specific`:

```python
(
    "deepfilter",
    "df",
    "install it with 'uv sync --extra deepfilter'",
),
```

Add `("deepfilter", "df")` to the missing-transitive-dependency parameter list.

- [ ] **Step 2: Run focused tests and verify RED**

Run:

```powershell
rtk uv run pytest tests/test_bot.py -k "noise_filter or optional_filter" -v
```

Expected: DeepFilterNet cases fail because the factory does not support the name.

- [ ] **Step 3: Add the lazy factory branch**

Insert before the final `ValueError` in `create_audio_filter()`:

```python
    if selected == "deepfilter":
        try:
            import df  # noqa: F401
            from deepfilter_filter import DeepFilterNetFilter
        except ModuleNotFoundError as exc:
            if exc.name != "df":
                raise
            raise RuntimeError(
                "DeepFilterNet selected; install it with 'uv sync --extra deepfilter'"
            ) from exc
        return DeepFilterNetFilter()
```

Update the final message:

```python
raise ValueError(
    "NOISE_FILTER must be one of: browser, rnnoise, koala, deepfilter"
)
```

- [ ] **Step 4: Run factory tests and verify GREEN**

Run:

```powershell
rtk uv run pytest tests/test_bot.py -k "noise_filter or optional_filter" -v
```

Expected: all selected tests pass.

- [ ] **Step 5: Checkpoint commit after approval**

```powershell
rtk git add bot.py tests/test_bot.py
rtk git commit -m "feat: select DeepFilterNet noise filtering"
```

---

### Task 4: Lab and Documentation Integration

**Files:**
- Modify: `lab.html:95-124`
- Modify: `tests/test_lab.py:107-133`
- Modify: `README.md:93-168,210-250`

**Interfaces:**
- Produces: `deepfilter` in every lab slot's denoise selector and documented setup.

- [ ] **Step 1: Add a failing lab factory-delegation test**

Add:

```python
def test_deepfilter_lab_uses_the_production_filter_factory(monkeypatch):
    calls = []

    class FakeFilter:
        async def start(self, sample_rate):
            calls.append(("start", sample_rate))

        async def filter(self, audio):
            calls.append(("filter", audio))
            return audio

        async def stop(self):
            calls.append(("stop",))

    import bot

    monkeypatch.setattr(
        bot, "create_audio_filter", lambda name: FakeFilter() if name == "deepfilter" else None
    )
    pcm = tone(0.02)

    assert asyncio.run(lab.apply_noise_filter(pcm, "deepfilter")) == pcm
    assert calls == [("start", 16000), ("filter", pcm), ("stop",)]
```

- [ ] **Step 2: Run the test and verify GREEN against existing delegation**

Run:

```powershell
rtk uv run pytest tests/test_lab.py::test_deepfilter_lab_uses_the_production_filter_factory -v
```

Expected: PASS because `lab.py` already delegates all non-browser names to `create_audio_filter()`.

- [ ] **Step 3: Add DeepFilterNet to the lab selector**

Change:

```javascript
const NOISE_FILTERS = ["none", "rnnoise", "koala", "deepfilter"];
```

- [ ] **Step 4: Document installation and current limitations**

Replace the README's blocked DeepFilterNet row with:

```markdown
| `deepfilter` | free, local | `uv sync --extra deepfilter`; Python 3.11, model downloads on first use. |
```

Add a `### DeepFilterNet` setup section containing the command `uv sync --extra deepfilter` in a PowerShell code block, followed by:

```markdown
Set `NOISE_FILTER=deepfilter`. The project adapter resamples the 16 kHz transport
audio to DeepFilterNet's 48 kHz model rate and back while preserving inference state
between transport frames. The default DeepFilterNet3 model downloads on first use.
```

Update lab comparison commands to include `--extra deepfilter` when that option is tested. Keep `NOISE_FILTER=browser` in `.env.example`; add no DeepFilterNet-specific variable.

- [ ] **Step 5: Run documentation assertions**

Run:

```powershell
rtk uv run pytest tests/test_bot.py -k readme -v
rtk uv run pytest tests/test_lab.py -v
```

Expected: README assertions and all lab tests pass.

- [ ] **Step 6: Checkpoint commit after approval**

```powershell
rtk git add lab.html tests/test_lab.py README.md
rtk git commit -m "docs: add DeepFilterNet to the VAD lab"
```

---

### Task 5: Real Model Smoke and Complete Verification

**Files:**
- Verify only; do not edit `.env`.

**Interfaces:**
- Verifies: real Windows wheel, model download, stateful adapter output, and real-time viability.

- [ ] **Step 1: Run a cached-model adapter smoke**

Run this after the first model download completes:

```powershell
rtk uv run --extra deepfilter python -c "exec('import asyncio, time\nimport numpy as np\nfrom lab import apply_noise_filter\nrate=16000\nt=np.arange(rate*10)/rate\npcm=(0.2*np.sin(2*np.pi*180*t)*32767).astype(\"<i2\").tobytes()\nstart=time.perf_counter()\nout=asyncio.run(apply_noise_filter(pcm, \"deepfilter\"))\nelapsed=time.perf_counter()-start\nprint(len(out), elapsed)\nassert out and elapsed < 10')"
```

Expected: non-empty output and cached execution under the 10-second clip duration. If it fails, profile before live use rather than weakening the assertion.

- [ ] **Step 2: Compare one real noisy WAV in the lab**

Run:

```powershell
rtk uv run --extra deepfilter lab.py
```

Open `http://127.0.0.1:7861`, load the same WAV into two otherwise identical slots, select `none` in one and `deepfilter` in the other, and confirm both produce aligned traces without slot errors.

- [ ] **Step 3: Run complete automated verification**

Run:

```powershell
rtk uv run pytest -v
rtk uv run python -m py_compile bot.py vad.py lab.py deepfilter_filter.py
rtk git diff --check
```

Expected: all tests pass, compile emits no output, and diff check emits no output.

- [ ] **Step 4: Inspect only intended changes**

Run:

```powershell
rtk git status --short
rtk git diff -- .python-version pyproject.toml uv.lock deepfilter_filter.py bot.py lab.html README.md tests/test_deepfilter_filter.py tests/test_bot.py tests/test_lab.py
```

Expected: no `.env` change and no unrelated concurrent work included.

- [ ] **Step 5: Final checkpoint commit after approval**

```powershell
rtk git add .python-version pyproject.toml uv.lock deepfilter_filter.py bot.py lab.html README.md tests/test_deepfilter_filter.py tests/test_bot.py tests/test_lab.py
rtk git commit -m "feat: add DeepFilterNet noise filtering"
```
