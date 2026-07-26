# Pipecat Voice Agent Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: superpowers:subagent-driven-development (recommended) or superpowers:executing-plans. Implement this plan task-by-task.

**Goal:** Build a minimal Hindi-first browser voice agent with Gemini Live and SmallWebRTC, then add independently selectable VAD and noise-filter options without adding a custom frontend or telephony implementation.

**Architecture:** `bot(runner_args)` creates only the selected Pipecat transport and passes it to `run_agent(transport, runner_args)`. `run_agent` owns Gemini Live, context, events, metrics, and lifecycle; `vad.py` owns local VAD selection and the Cobra adapter. Gemini server VAD is the default, browser WebRTC processing is the default noise baseline, and Exotel remains documentation-only readiness for a later plan.

**Tech Stack:** Python 3.11/3.12, uv, `pipecat-ai[google,runner,webrtc]>=1.4.0,<2`, Gemini Live, SmallWebRTC, pytest, optional Silero/TEN/FireRed/Cobra VAD, optional RNNoise/Koala filtering.

## Global Constraints

- Support Python `>=3.11,<3.13` only.
- Use `uv`; do not create `requirements.txt` or another package-manager workflow.
- Keep application code in `bot.py` and `vad.py`; keep tests in `tests/test_bot.py` and `tests/test_vad.py`.
- Use Pipecat's built-in development runner client; do not create React, HTML, JavaScript, or another frontend.
- Keep `run_agent(transport, runner_args)` transport-independent.
- Keep internal input audio at 16 kHz mono PCM.
- Default to Gemini server-side VAD; local VAD disables Gemini VAD and uses `realtime_service_mode=True`.
- Keep TEN, FireRed, Cobra, RNNoise, and Koala imports lazy so the base installation works without them.
- Keep noise filtering independent from VAD selection.
- Do not implement Exotel webhooks, WebSockets, API calls, serializers, or call control in this plan.
- Never log API keys or raw environment values.
- Use PowerShell-compatible commands and prefix applicable commands with `rtk`.
- Perform only the explicitly requested first-iteration GitHub commit and push; do not add commit steps after later tasks.

---

### Task 1: Create the Minimal Python Project and Configuration Contract

**Files:**
- Create: `pyproject.toml`
- Create: `.env.example`
- Create: `.gitignore`
- Create: `tests/test_bot.py`
- Create: `bot.py`

**Interfaces:**
- Produces: `get_google_api_key() -> str`
- Produces: `SYSTEM_INSTRUCTION: str`
- Produces: base environment variables consumed by later tasks.

- [ ] **Step 1: Write the failing startup/configuration test**

Create `tests/test_bot.py`:

```python
import pytest

import bot


def test_google_api_key_is_required(monkeypatch):
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)

    with pytest.raises(RuntimeError, match="GOOGLE_API_KEY is required"):
        bot.get_google_api_key()


def test_google_api_key_is_returned_without_logging_or_transforming(monkeypatch):
    monkeypatch.setenv("GOOGLE_API_KEY", "test-key")

    assert bot.get_google_api_key() == "test-key"


def test_prompt_is_hindi_first_and_speech_friendly():
    prompt = bot.SYSTEM_INSTRUCTION.lower()

    assert "hindi" in prompt
    assert "english" in prompt
    assert "markdown" in prompt
    assert "brief" in prompt
```

- [ ] **Step 2: Create the package metadata and install the base environment**

Create `pyproject.toml`:

```toml
[project]
name = "voiceassistant"
version = "0.1.0"
description = "Hindi-first Pipecat browser voice agent"
requires-python = ">=3.11,<3.13"
dependencies = [
    "pipecat-ai[google,runner,webrtc]>=1.4.0,<2",
    "python-dotenv>=1,<2",
]

[project.optional-dependencies]
cobra = ["pvcobra>=3,<4"]
ten = ["pipecat-ten-vad"]
firered = ["pipecat-firered-vad"]
rnnoise = ["pipecat-ai[rnnoise]>=1.4.0,<2"]
koala = ["pipecat-ai[koala]>=1.4.0,<2"]

[dependency-groups]
dev = ["pytest>=8,<9"]

[tool.pytest.ini_options]
testpaths = ["tests"]
```

Run:

```powershell
rtk uv sync
```

Expected: uv creates `.venv` and `uv.lock`, installs Pipecat and pytest, and exits with code 0.

- [ ] **Step 3: Run the test and verify it fails because `bot.py` does not exist**

Run:

```powershell
rtk uv run pytest tests/test_bot.py -v
```

Expected: collection fails with `ModuleNotFoundError: No module named 'bot'`.

- [ ] **Step 4: Add the minimal configuration implementation**

Create `bot.py`:

```python
import os

from dotenv import load_dotenv

load_dotenv(override=True)

SYSTEM_INSTRUCTION = (
    "You are a friendly Hindi-first voice assistant. Speak primarily in Hindi, "
    "allow natural Hindi-English code-switching, and follow the user's language. "
    "Keep every response brief, conversational, and suitable for speech. "
    "Do not use Markdown, bullets, emojis, or formatting that cannot be spoken."
)


def get_google_api_key() -> str:
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError("GOOGLE_API_KEY is required; copy .env.example to .env and set it")
    return api_key
```

Create `.env.example`:

```dotenv
GOOGLE_API_KEY=
GEMINI_MODEL=models/gemini-2.5-flash-native-audio-preview-12-2025
GEMINI_VOICE=Charon
VAD_BACKEND=gemini
VAD_CONFIDENCE=0.7
VAD_START_SECS=0.2
VAD_STOP_SECS=0.3
VAD_MIN_VOLUME=0.6
TEN_VAD_THRESHOLD=0.6
FIREREDVAD_MODEL_DIR=
FIREREDVAD_USE_GPU=0
PICOVOICE_ACCESS_KEY=
NOISE_FILTER=browser
KOALA_ACCESS_KEY=
```

Create `.gitignore`:

```gitignore
.env
.venv/
__pycache__/
.pytest_cache/
*.py[cod]
.serena/
pretrained_models/
FireRedVAD/
```

- [ ] **Step 5: Run the startup/configuration tests**

Run:

```powershell
rtk uv run pytest tests/test_bot.py -v
```

Expected: `3 passed`.

### Task 2: Build Iteration 1, the Smallest Runnable Browser Agent

**Files:**
- Modify: `bot.py`
- Modify: `tests/test_bot.py`
- Create: `README.md`

**Interfaces:**
- Consumes: `get_google_api_key() -> str`, `SYSTEM_INSTRUCTION`.
- Produces: `run_agent(transport: BaseTransport, runner_args: RunnerArguments) -> None`.
- Produces: `bot(runner_args: RunnerArguments) -> None`.
- Produces: SmallWebRTC transport configuration with 16 kHz input.

- [ ] **Step 1: Add failing tests for the browser transport and Gemini defaults**

Append to `tests/test_bot.py`:

```python
def test_gemini_defaults(monkeypatch):
    monkeypatch.delenv("GEMINI_MODEL", raising=False)
    monkeypatch.delenv("GEMINI_VOICE", raising=False)

    assert bot.get_gemini_model() == "models/gemini-2.5-flash-native-audio-preview-12-2025"
    assert bot.get_gemini_voice() == "Charon"


def test_webrtc_transport_uses_16khz_input():
    params = bot.webrtc_transport_params()

    assert params.audio_in_enabled is True
    assert params.audio_out_enabled is True
    assert params.audio_in_sample_rate == 16000
```

- [ ] **Step 2: Run the new tests and verify they fail**

Run:

```powershell
rtk uv run pytest tests/test_bot.py -v
```

Expected: three original tests pass; failures report missing `get_gemini_model`, `get_gemini_voice`, and `webrtc_transport_params`.

- [ ] **Step 3: Replace `bot.py` with the complete iteration-1 agent**

Replace `bot.py` with:

```python
import logging
import os

from dotenv import load_dotenv

from pipecat.frames.frames import LLMRunFrame
from pipecat.observers.user_bot_latency_observer import UserBotLatencyObserver
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.worker import PipelineParams, PipelineWorker
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import (
    AssistantTurnStoppedMessage,
    LLMContextAggregatorPair,
    UserTurnMessageAddedMessage,
)
from pipecat.runner.types import RunnerArguments
from pipecat.runner.utils import create_transport
from pipecat.services.google.gemini_live.llm import (
    GeminiLiveLLMService,
    GeminiModalities,
)
from pipecat.transports.base_transport import BaseTransport, TransportParams
from pipecat.workers.runner import WorkerRunner

load_dotenv(override=True)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("voiceassistant")

SYSTEM_INSTRUCTION = (
    "You are a friendly Hindi-first voice assistant. Speak primarily in Hindi, "
    "allow natural Hindi-English code-switching, and follow the user's language. "
    "Keep every response brief, conversational, and suitable for speech. "
    "Do not use Markdown, bullets, emojis, or formatting that cannot be spoken."
)
DEFAULT_GEMINI_MODEL = "models/gemini-2.5-flash-native-audio-preview-12-2025"


def get_google_api_key() -> str:
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError("GOOGLE_API_KEY is required; copy .env.example to .env and set it")
    return api_key


def get_gemini_model() -> str:
    return os.getenv("GEMINI_MODEL", DEFAULT_GEMINI_MODEL)


def get_gemini_voice() -> str:
    return os.getenv("GEMINI_VOICE", "Charon")


def webrtc_transport_params() -> TransportParams:
    return TransportParams(
        audio_in_enabled=True,
        audio_out_enabled=True,
        audio_in_sample_rate=16000,
    )


async def run_agent(transport: BaseTransport, runner_args: RunnerArguments) -> None:
    llm = GeminiLiveLLMService(
        api_key=get_google_api_key(),
        settings=GeminiLiveLLMService.Settings(
            model=get_gemini_model(),
            voice=get_gemini_voice(),
            modalities=GeminiModalities.AUDIO,
            system_instruction=SYSTEM_INSTRUCTION,
        ),
    )

    context = LLMContext()
    user_aggregator, assistant_aggregator = LLMContextAggregatorPair(
        context,
        realtime_service_mode=True,
    )
    pipeline = Pipeline(
        [
            transport.input(),
            user_aggregator,
            llm,
            transport.output(),
            assistant_aggregator,
        ]
    )

    latency_observer = UserBotLatencyObserver()
    worker = PipelineWorker(
        pipeline,
        params=PipelineParams(
            audio_in_sample_rate=16000,
            enable_metrics=True,
            enable_usage_metrics=True,
        ),
        observers=[latency_observer],
        idle_timeout_secs=runner_args.pipeline_idle_timeout_secs,
    )

    @latency_observer.event_handler("on_latency_measured")
    async def on_latency_measured(observer, latency):
        logger.info("user_to_bot_latency_seconds=%.3f", latency)

    @latency_observer.event_handler("on_first_bot_speech_latency")
    async def on_first_bot_speech_latency(observer, latency):
        logger.info("first_bot_speech_latency_seconds=%.3f", latency)

    @user_aggregator.event_handler("on_user_turn_started")
    async def on_user_turn_started(aggregator, strategy):
        logger.info("user_speaking_started strategy=%s", type(strategy).__name__)

    @user_aggregator.event_handler("on_user_turn_message_added")
    async def on_user_turn_message_added(
        aggregator, message: UserTurnMessageAddedMessage
    ):
        logger.info("user_transcript=%s", message.content)

    @assistant_aggregator.event_handler("on_assistant_turn_started")
    async def on_assistant_turn_started(aggregator):
        logger.info("assistant_speaking_started")

    @assistant_aggregator.event_handler("on_assistant_turn_stopped")
    async def on_assistant_turn_stopped(
        aggregator, message: AssistantTurnStoppedMessage
    ):
        logger.info(
            "assistant_transcript=%s interrupted=%s",
            message.content,
            message.interrupted,
        )

    @worker.event_handler("on_pipeline_error")
    async def on_pipeline_error(worker, frame):
        logger.error("pipeline_error=%s fatal=%s", frame.error, frame.fatal)

    @worker.turn_tracking_observer.event_handler("on_turn_ended")
    async def on_turn_ended(observer, turn_number, duration, was_interrupted):
        logger.info(
            "turn_ended number=%d duration_seconds=%.3f interrupted=%s",
            turn_number,
            duration,
            was_interrupted,
        )

    @transport.event_handler("on_client_connected")
    async def on_client_connected(transport, client):
        logger.info("client_connected")

    @worker.rtvi.event_handler("on_client_ready")
    async def on_client_ready(rtvi):
        logger.info("client_ready")
        context.add_message(
            {
                "role": "developer",
                "content": "Greet the user briefly in Hindi and ask how you can help.",
            }
        )
        await worker.queue_frame(LLMRunFrame())

    @transport.event_handler("on_client_disconnected")
    async def on_client_disconnected(transport, client):
        logger.info("client_disconnected")
        await worker.cancel()

    runner = WorkerRunner(handle_sigint=runner_args.handle_sigint)
    await runner.add_workers(worker)
    await runner.run()


async def bot(runner_args: RunnerArguments) -> None:
    transport = await create_transport(
        runner_args,
        {"webrtc": webrtc_transport_params},
    )
    await run_agent(transport, runner_args)


if __name__ == "__main__":
    from pipecat.runner.run import main

    main()
```

- [ ] **Step 4: Run all automated iteration-1 tests**

Run:

```powershell
rtk uv run pytest -v
```

Expected: `5 passed`.

- [ ] **Step 5: Add the iteration-1 README**

Create `README.md`:

```markdown
# voiceassistant

Minimal Hindi-first Pipecat voice agent using Gemini Live and the built-in SmallWebRTC browser client.

## Requirements

- Python 3.11 or 3.12
- uv
- Google AI Studio API key with Gemini Live model access

## Run

```powershell
Copy-Item .env.example .env
# Set GOOGLE_API_KEY in .env
rtk uv sync
rtk uv run bot.py -t webrtc
```

Open `http://localhost:7860/client`, allow microphone access, and connect.

The browser's WebRTC echo cancellation, automatic gain control, and noise suppression are the iteration-1 noise baseline. No custom frontend or server-side filter is required.

## Test

```powershell
rtk uv run pytest -v
```

## Scope

Iteration 1 is browser-only. Exotel calling, public deployment, persistence, tools, and authentication are excluded.
```

### Task 3: Verify Iteration 1 and Publish the Requested Public GitHub Checkpoint

**Files:**
- Verify only: `bot.py`, `tests/test_bot.py`, `pyproject.toml`, `.env.example`, `.gitignore`, `README.md`, `uv.lock`, design and plan documents.

**Interfaces:**
- Verifies the browser agent end to end before any local VAD or server-side noise dependency is added.
- Produces the only commit and push required by this plan.

- [ ] **Step 1: Run clean automated verification**

Run:

```powershell
rtk uv sync
rtk uv run pytest -v
```

Expected: dependency sync exits 0 and pytest reports `5 passed`.

- [ ] **Step 2: Start the WebRTC runner**

Run:

```powershell
rtk uv run bot.py -t webrtc
```

Expected: the runner reports a server at `http://localhost:7860/client` with no missing-key or import error.

- [ ] **Step 3: Complete the iteration-1 browser checklist**

- Open `http://localhost:7860/client` and connect.
- Confirm a short Hindi greeting plays.
- Hold at least three Hindi turns.
- Switch naturally between Hindi and English.
- Interrupt the assistant and confirm the current response stops.
- Pause mid-sentence and note whether Gemini ends the turn too early.
- Stay silent for 10 seconds and confirm no false response is generated.
- Disconnect and reconnect; confirm a fresh greeting and no prior-session context.
- Confirm logs show connection, client-ready, assistant transcript, interruption state, turn timing, and latency where Gemini emits the required frames.
- Stop the runner with Ctrl+C.

Expected: all checks pass. If authentication, quota, or model access fails, the log names the service failure without printing `GOOGLE_API_KEY`.

- [ ] **Step 4: Initialize git and inspect the complete intended change set**

Run each command separately:

```powershell
rtk git init
rtk git branch -M main
rtk git status --short
rtk git diff -- .gitignore .env.example README.md bot.py pyproject.toml tests/test_bot.py docs/superpowers/specs/2026-07-26-pipecat-voice-agent-design.md docs/superpowers/plans/2026-07-26-pipecat-voice-agent.md
rtk git log --oneline -10
```

Expected: status lists only intended project files plus `uv.lock`; `git diff` is empty because the files are still untracked. Confirm from status that `.env`, secrets, `.venv`, `.serena`, model weights, and unrelated files are absent. The log command reports that the new repository has no commits yet.

- [ ] **Step 5: Stage only the intended first-iteration files and inspect the staged diff**

Run:

```powershell
rtk git add .gitignore .env.example README.md bot.py pyproject.toml uv.lock tests/test_bot.py docs/superpowers/specs/2026-07-26-pipecat-voice-agent-design.md docs/superpowers/plans/2026-07-26-pipecat-voice-agent.md
rtk git status --short
rtk git diff --cached
```

Expected: exactly the listed files are staged; no secret or generated environment is present.

- [ ] **Step 6: Create the first-iteration commit**

Run:

```powershell
rtk git commit -m "feat: add minimal Gemini Live browser agent"
```

Expected: one root commit is created successfully.

- [ ] **Step 7: Authenticate GitHub CLI and create the public repository**

Run:

```powershell
rtk gh auth status
rtk gh repo create voiceassistant --public --source . --remote origin
```

Expected: `gh auth status` confirms an authenticated account; GitHub creates a public repository named `voiceassistant` and configures `origin`. If that repository name already exists for the authenticated owner, stop and report the conflict instead of deleting or overwriting it.

- [ ] **Step 8: Push and report the repository URL**

Run:

```powershell
rtk git push -u origin main
rtk git status --short
rtk gh repo view --json url --jq .url
```

Expected: `main` is tracking `origin/main`, status is clean, and the final command prints the public repository URL to report to the user.

### Task 4: Add the Swappable VAD Factory and Cobra Adapter

**Files:**
- Create: `vad.py`
- Create: `tests/test_vad.py`

**Interfaces:**
- Produces: `create_vad(backend: str | None = None) -> VADAnalyzer | None`.
- Produces: `CobraVADAnalyzer(access_key: str, params: VADParams | None = None)` implementing `num_frames_required()`, `voice_confidence(buffer)`, and `cleanup()`.
- Returns `None` for `gemini`; returns a local analyzer for `silero`, `ten`, `firered`, or `cobra`.

- [ ] **Step 1: Write factory tests that do not load optional models**

Create `tests/test_vad.py`:

```python
import builtins
import sys
from types import SimpleNamespace

import pytest

import vad


def test_default_backend_is_gemini(monkeypatch):
    monkeypatch.delenv("VAD_BACKEND", raising=False)

    assert vad.create_vad() is None


def test_backend_name_is_case_insensitive():
    assert vad.create_vad(" GEMINI ") is None


def test_invalid_backend_lists_supported_values():
    with pytest.raises(
        ValueError,
        match="VAD_BACKEND must be one of: gemini, silero, ten, firered, cobra",
    ):
        vad.create_vad("unknown")


@pytest.mark.parametrize(
    ("backend", "missing_module", "message"),
    [
        ("ten", "pipecat_ten_vad", "install the TEN VAD optional dependencies"),
        ("firered", "pipecat_firered_vad", "install the FireRed VAD optional dependencies"),
        ("cobra", "pvcobra", "install the Cobra optional dependencies"),
    ],
)
def test_optional_dependency_errors_are_backend_specific(
    monkeypatch, backend, missing_module, message
):
    real_import = builtins.__import__

    def blocked_import(name, *args, **kwargs):
        if name == missing_module:
            raise ModuleNotFoundError(name)
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", blocked_import)
    if backend == "cobra":
        monkeypatch.setenv("PICOVOICE_ACCESS_KEY", "test-key")
    if backend == "firered":
        monkeypatch.setenv("FIREREDVAD_MODEL_DIR", "model-dir")

    with pytest.raises(RuntimeError, match=message):
        vad.create_vad(backend)


def test_cobra_adapter_converts_little_endian_pcm_and_cleans_up(monkeypatch):
    class FakeCobra:
        sample_rate = 16000
        frame_length = 4

        def __init__(self):
            self.samples = None
            self.deleted = False

        def process(self, samples):
            self.samples = list(samples)
            return 0.75

        def delete(self):
            self.deleted = True

    engine = FakeCobra()
    monkeypatch.setitem(
        sys.modules,
        "pvcobra",
        SimpleNamespace(create=lambda access_key: engine),
    )

    analyzer = vad.CobraVADAnalyzer("test-key")
    analyzer.set_sample_rate(16000)

    assert analyzer.num_frames_required() == 4
    assert analyzer.voice_confidence(b"\x01\x00\xfe\xff\x03\x00\xfc\xff") == 0.75
    assert engine.samples == [1, -2, 3, -4]

    import asyncio

    asyncio.run(analyzer.cleanup())
    assert engine.deleted is True
```

- [ ] **Step 2: Run the VAD tests and verify they fail because `vad.py` does not exist**

Run:

```powershell
rtk uv run pytest tests/test_vad.py -v
```

Expected: collection fails with `ModuleNotFoundError: No module named 'vad'`.

- [ ] **Step 3: Implement the minimal VAD module**

Create `vad.py`:

```python
import os
import sys
from array import array

from pipecat.audio.vad.vad_analyzer import VADAnalyzer, VADParams

SUPPORTED_BACKENDS = ("gemini", "silero", "ten", "firered", "cobra")


def _float_env(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except ValueError as exc:
        raise RuntimeError(f"{name} must be a number") from exc


def common_vad_params() -> VADParams:
    return VADParams(
        confidence=_float_env("VAD_CONFIDENCE", 0.7),
        start_secs=_float_env("VAD_START_SECS", 0.2),
        stop_secs=_float_env("VAD_STOP_SECS", 0.3),
        min_volume=_float_env("VAD_MIN_VOLUME", 0.6),
    )


class CobraVADAnalyzer(VADAnalyzer):
    def __init__(self, access_key: str, params: VADParams | None = None):
        try:
            import pvcobra
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "Cobra selected; install the Cobra optional dependencies with "
                "'rtk uv sync --extra cobra'"
            ) from exc

        self._cobra = pvcobra.create(access_key=access_key)
        cobra_sample_rate = self._cobra.sample_rate
        if cobra_sample_rate != 16000:
            self._cobra.delete()
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
        self._cobra.delete()
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
        try:
            from pipecat_ten_vad import TenVadAnalyzer
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "TEN selected; install the TEN VAD optional dependencies described in README.md"
            ) from exc
        return TenVadAnalyzer(
            sample_rate=16000,
            threshold=_float_env("TEN_VAD_THRESHOLD", 0.6),
            params=params,
        )

    if selected == "firered":
        model_dir = os.getenv("FIREREDVAD_MODEL_DIR")
        if not model_dir:
            raise RuntimeError("FIREREDVAD_MODEL_DIR is required when VAD_BACKEND=firered")
        try:
            from pipecat_firered_vad import FireVadAnalyzer
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "FireRed selected; install the FireRed VAD optional dependencies described in README.md"
            ) from exc
        return FireVadAnalyzer(
            model_dir=model_dir,
            sample_rate=16000,
            params=params,
            use_gpu=os.getenv("FIREREDVAD_USE_GPU", "0") == "1",
        )

    access_key = os.getenv("PICOVOICE_ACCESS_KEY")
    if not access_key:
        raise RuntimeError("PICOVOICE_ACCESS_KEY is required when VAD_BACKEND=cobra")
    return CobraVADAnalyzer(access_key, params=params)
```

- [ ] **Step 4: Run the VAD factory and adapter tests**

Run:

```powershell
rtk uv run pytest tests/test_vad.py -v
```

Expected: `7 passed`.

### Task 5: Wire Local VAD into Gemini Live Without Changing the Pipeline Shape

**Files:**
- Modify: `bot.py`
- Modify: `tests/test_bot.py`

**Interfaces:**
- Consumes: `create_vad() -> VADAnalyzer | None`.
- Gemini mode: `GeminiVADParams(disabled=False)` and no local analyzer.
- Local mode: `GeminiVADParams(disabled=True)` and `LLMUserAggregatorParams(vad_analyzer=analyzer)`.

- [ ] **Step 1: Add a failing test for VAD-mode configuration**

Append to `tests/test_bot.py`:

```python
def test_vad_mode_switches_gemini_server_vad():
    server_vad, user_params = bot.vad_configuration(None)
    assert server_vad.disabled is False
    assert user_params is None

    analyzer = object()
    server_vad, user_params = bot.vad_configuration(analyzer)
    assert server_vad.disabled is True
    assert user_params.vad_analyzer is analyzer
```

- [ ] **Step 2: Run the test and verify the missing helper failure**

Run:

```powershell
rtk uv run pytest tests/test_bot.py::test_vad_mode_switches_gemini_server_vad -v
```

Expected: FAIL with `AttributeError: module 'bot' has no attribute 'vad_configuration'`.

- [ ] **Step 3: Add the imports and helper to `bot.py`**

Add these imports with the existing Pipecat imports:

```python
from pipecat.processors.aggregators.llm_response_universal import LLMUserAggregatorParams
from pipecat.services.google.gemini_live.llm import GeminiVADParams

from vad import create_vad
```

Add this function before `run_agent`:

```python
def vad_configuration(analyzer):
    if analyzer is None:
        return GeminiVADParams(disabled=False), None
    return (
        GeminiVADParams(disabled=True),
        LLMUserAggregatorParams(vad_analyzer=analyzer),
    )
```

- [ ] **Step 4: Update `run_agent` to create and apply the selected analyzer**

At the start of `run_agent`, before constructing `GeminiLiveLLMService`, add:

```python
    analyzer = create_vad()
    gemini_vad, user_params = vad_configuration(analyzer)
    logger.info("vad_backend=%s", os.getenv("VAD_BACKEND", "gemini").strip().lower())
```

Add `vad=gemini_vad` to the existing `GeminiLiveLLMService.Settings` call:

```python
        settings=GeminiLiveLLMService.Settings(
            model=get_gemini_model(),
            voice=get_gemini_voice(),
            modalities=GeminiModalities.AUDIO,
            system_instruction=SYSTEM_INSTRUCTION,
            vad=gemini_vad,
        ),
```

Pass `user_params=user_params` to the aggregator pair:

```python
    user_aggregator, assistant_aggregator = LLMContextAggregatorPair(
        context,
        realtime_service_mode=True,
        user_params=user_params,
    )
```

- [ ] **Step 5: Run all tests**

Run:

```powershell
rtk uv run pytest -v
```

Expected: `13 passed`.

- [ ] **Step 6: Smoke-test Gemini and Silero modes**

Run Gemini mode:

```powershell
$env:VAD_BACKEND="gemini"
rtk uv run bot.py -t webrtc
```

Expected: log includes `vad_backend=gemini`; browser conversation works with Gemini server VAD.

Stop the runner, then run Silero mode:

```powershell
$env:VAD_BACKEND="silero"
rtk uv run bot.py -t webrtc
```

Expected: log includes `vad_backend=silero`; first startup may download/load the Silero ONNX model; browser conversation, interruption, and user speaking events work with 16 kHz local VAD.

### Task 6: Add Optional Noise Filters Independently of VAD

**Files:**
- Modify: `bot.py`
- Modify: `tests/test_bot.py`

**Interfaces:**
- Produces: `create_audio_filter(name: str | None = None) -> BaseAudioFilter | None`.
- `browser` returns `None`; `rnnoise` and `koala` import only inside their branches.
- Transport configuration accepts `audio_filter` without reading or changing `VAD_BACKEND`.

- [ ] **Step 1: Add failing tests for the browser baseline and invalid filter**

Append to `tests/test_bot.py`:

```python
def test_browser_noise_filter_is_the_default(monkeypatch):
    monkeypatch.delenv("NOISE_FILTER", raising=False)

    assert bot.create_audio_filter() is None


def test_invalid_noise_filter_is_rejected():
    with pytest.raises(
        ValueError,
        match="NOISE_FILTER must be one of: browser, rnnoise, koala",
    ):
        bot.create_audio_filter("unknown")


def test_koala_requires_an_access_key(monkeypatch):
    monkeypatch.delenv("KOALA_ACCESS_KEY", raising=False)

    with pytest.raises(RuntimeError, match="KOALA_ACCESS_KEY is required"):
        bot.create_audio_filter("koala")
```

- [ ] **Step 2: Run the filter tests and verify they fail**

Run:

```powershell
rtk uv run pytest tests/test_bot.py -k "noise_filter or koala" -v
```

Expected: failures report missing `create_audio_filter`.

- [ ] **Step 3: Implement the lazy noise-filter factory**

Add to `bot.py` before `webrtc_transport_params`:

```python
def create_audio_filter(name: str | None = None):
    selected = (name or os.getenv("NOISE_FILTER", "browser")).strip().lower()
    if selected == "browser":
        return None
    if selected == "rnnoise":
        try:
            from pipecat.audio.filters.rnnoise_filter import RNNoiseFilter
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "RNNoise selected; install it with 'rtk uv sync --extra rnnoise'"
            ) from exc
        return RNNoiseFilter()
    if selected == "koala":
        access_key = os.getenv("KOALA_ACCESS_KEY")
        if not access_key:
            raise RuntimeError("KOALA_ACCESS_KEY is required when NOISE_FILTER=koala")
        try:
            from pipecat.audio.filters.koala_filter import KoalaFilter
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "Koala selected; install it with 'rtk uv sync --extra koala'"
            ) from exc
        return KoalaFilter(access_key=access_key)
    raise ValueError("NOISE_FILTER must be one of: browser, rnnoise, koala")
```

Change the transport helper signature and body:

```python
def webrtc_transport_params(audio_filter=None) -> TransportParams:
    return TransportParams(
        audio_in_enabled=True,
        audio_out_enabled=True,
        audio_in_sample_rate=16000,
        audio_in_filter=audio_filter,
    )
```

Change `bot` to create the filter independently and pass it into the WebRTC params factory:

```python
async def bot(runner_args: RunnerArguments) -> None:
    audio_filter = create_audio_filter()
    logger.info("noise_filter=%s", os.getenv("NOISE_FILTER", "browser").strip().lower())
    transport = await create_transport(
        runner_args,
        {"webrtc": lambda: webrtc_transport_params(audio_filter)},
    )
    await run_agent(transport, runner_args)
```

- [ ] **Step 4: Run all tests**

Run:

```powershell
rtk uv run pytest -v
```

Expected: `16 passed`.

- [ ] **Step 5: Verify the base remains dependency-free and optionally smoke-test RNNoise**

Run the browser baseline:

```powershell
$env:NOISE_FILTER="browser"
$env:VAD_BACKEND="gemini"
rtk uv run bot.py -t webrtc
```

Expected: log includes `noise_filter=browser`; no RNNoise or Koala import error occurs.

After stopping the runner, optionally install and run RNNoise:

```powershell
rtk uv sync --extra rnnoise
$env:NOISE_FILTER="rnnoise"
rtk uv run bot.py -t webrtc
```

Expected: log includes `noise_filter=rnnoise`; browser conversation works. Do not compare filters until the browser baseline has already passed.

### Task 7: Document Optional VAD Setup, Comparison Procedure, and Exotel Readiness

**Files:**
- Modify: `README.md`

**Interfaces:**
- Documents exact optional installation commands and environment values.
- Documents a repeatable VAD comparison procedure.
- Documents Exotel boundaries only; produces no Exotel code.

- [ ] **Step 1: Replace `README.md` with the complete operational documentation**

Replace `README.md` with:

```markdown
# voiceassistant

Minimal Hindi-first Pipecat voice agent using Gemini Live and Pipecat's built-in SmallWebRTC browser client.

## Requirements

- Python 3.11 or 3.12
- uv
- Google AI Studio API key with Gemini Live model access

## Base Setup

```powershell
Copy-Item .env.example .env
# Set GOOGLE_API_KEY in .env
rtk uv sync
rtk uv run pytest -v
rtk uv run bot.py -t webrtc
```

Open `http://localhost:7860/client`, allow microphone access, and connect.

## Runtime Configuration

- `GEMINI_MODEL`: defaults to `models/gemini-2.5-flash-native-audio-preview-12-2025`.
- `GEMINI_VOICE`: defaults to `Charon`.
- `VAD_BACKEND`: `gemini` (default), `silero`, `ten`, `firered`, or `cobra`.
- `NOISE_FILTER`: `browser` (default), `rnnoise`, or `koala`.

The internal input rate is always 16 kHz. Optional imports are lazy, so base Gemini mode does not require local VAD or server-side noise packages.

## VAD Backends

### Gemini

No extra installation. Set `VAD_BACKEND=gemini`. Gemini Live performs server-side turn detection.

### Silero

No extra installation beyond the base Pipecat package. Set `VAD_BACKEND=silero`.

### TEN VAD

TEN is community-maintained and tested separately from Pipecat releases.

```powershell
rtk uv sync --extra ten
rtk uv pip install "git+https://github.com/TEN-framework/ten-vad.git"
```

Set `VAD_BACKEND=ten`. TEN requires Windows x64 and 16 kHz audio. Tune `TEN_VAD_THRESHOLD` separately from Pipecat's `VAD_CONFIDENCE`.

### FireRedVAD

FireRed is community-maintained. Its upstream Python package is not published on PyPI.

```powershell
rtk uv sync --extra firered
rtk git clone https://github.com/FireRedTeam/FireRedVAD.git
rtk uv pip install -r FireRedVAD/requirements.txt
rtk uvx --from "huggingface_hub[cli]" huggingface-cli download FireRedTeam/FireRedVAD --local-dir pretrained_models/FireRedVAD
$env:PYTHONPATH="$PWD\FireRedVAD;$env:PYTHONPATH"
```

Set `VAD_BACKEND=firered` and `FIREREDVAD_MODEL_DIR=pretrained_models/FireRedVAD/Stream-VAD`. Set `FIREREDVAD_USE_GPU=1` only when a working CUDA environment exists.

### Picovoice Cobra

```powershell
rtk uv sync --extra cobra
```

Set `VAD_BACKEND=cobra` and `PICOVOICE_ACCESS_KEY`. The local adapter converts 16-bit little-endian mono PCM to Cobra frames, uses Cobra's required frame length, and releases the engine at pipeline cleanup.

## VAD Tuning

- `VAD_CONFIDENCE`: Pipecat confidence threshold, default `0.7`.
- `VAD_START_SECS`: confirmed speech required before speech start, default `0.2`.
- `VAD_STOP_SECS`: silence required before speech end, default `0.3`.
- `VAD_MIN_VOLUME`: minimum normalized volume, default `0.6`.

Change one setting at a time. Backend-specific controls remain separate because they are not equivalent across models.

## Noise Filtering

`NOISE_FILTER=browser` uses browser WebRTC echo cancellation, automatic gain control, and noise suppression and requires no server dependency.

RNNoise is optional and should only be evaluated after the browser baseline works:

```powershell
rtk uv sync --extra rnnoise
```

Set `NOISE_FILTER=rnnoise`.

Koala is optional and independent of Cobra VAD:

```powershell
rtk uv sync --extra koala
```

Set `NOISE_FILTER=koala` and `KOALA_ACCESS_KEY`. `PICOVOICE_ACCESS_KEY` and `KOALA_ACCESS_KEY` are intentionally separate settings.

## Manual Voice Checklist

- Hear a short Hindi greeting.
- Hold a multi-turn Hindi conversation.
- Switch naturally between Hindi and English.
- Interrupt the assistant and confirm response cancellation.
- Pause mid-sentence and check for premature turn end.
- Stay silent and check for false turns.
- Disconnect and reconnect as a fresh session.
- Confirm configuration, authentication, quota, and model errors are understandable and contain no secrets.

## VAD Comparison Procedure

For every installed backend, keep the microphone, room, browser, speaker distance, Gemini model, prompt, and noise filter unchanged. Run one quiet-room pass and one repeatable noisy-room pass with the same Hindi and Hindi-English phrases.

Record:

| Backend | Condition | Missed starts | False starts | Clipped first syllables | End delay | Premature ends | Barge-in | CPU/GPU | Install notes |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| gemini | quiet |  |  |  |  |  |  |  |  |
| gemini | noisy |  |  |  |  |  |  |  |  |
| silero | quiet |  |  |  |  |  |  |  |  |
| silero | noisy |  |  |  |  |  |  |  |  |

Add rows only for optional backends that are actually installed. This is a learning comparison, not a statistically rigorous benchmark.

## Events and Metrics

Logs expose client connection/readiness/disconnection, available user and assistant turn events, transcripts, interruption state, turn duration, first-speech latency, user-to-bot latency where frames permit it, Pipecat service metrics, and pipeline errors. Gemini server VAD may expose fewer local user-speaking frames than local VAD modes.

## Exotel Readiness, Not Implementation

Exotel work is excluded from this implementation. A later plan will add:

- Runner-selected WebSocket transport and Pipecat `ExotelFrameSerializer`.
- Public media WebSocket endpoint and Exotel flow/webhook configuration.
- Inbound and outbound call control.
- Explicit hang-up handling.
- Telephone-specific 8 kHz testing and presets.

The current boundary is ready for that later work: `bot(runner_args)` owns transport creation, `run_agent(transport, runner_args)` owns reusable conversation behavior, and the internal input rate stays 16 kHz. Exotel's 8 kHz PCM must be resampled at the serializer boundary; no Exotel dependency or conditional belongs in `run_agent`.
```

- [ ] **Step 2: Run the complete automated suite after documentation changes**

Run:

```powershell
rtk uv sync
rtk uv run pytest -v
```

Expected: sync exits 0 and pytest reports `16 passed`.

- [ ] **Step 3: Verify lazy optional imports in a base-only environment**

Run:

```powershell
$env:VAD_BACKEND="gemini"
$env:NOISE_FILTER="browser"
rtk uv run python -c "import bot, vad; assert vad.create_vad() is None; assert bot.create_audio_filter() is None; print('base imports ok')"
```

Expected: prints `base imports ok` without importing TEN, FireRed, Cobra, RNNoise, or Koala.

- [ ] **Step 4: Run final browser verification with the base defaults**

Run:

```powershell
$env:VAD_BACKEND="gemini"
$env:NOISE_FILTER="browser"
rtk uv run bot.py -t webrtc
```

Expected: the browser agent connects, greets in Hindi, supports code-switching and interruption, logs transcripts/metrics/errors where available, and disconnects cleanly. Confirm no Exotel server, route, serializer, credential, or call-control code exists.

- [ ] **Step 5: Inspect the final uncommitted post-checkpoint changes without committing them**

Run:

```powershell
rtk git status --short
rtk git diff
```

Expected: only the intended post-checkpoint VAD, noise-filter, test, dependency, lockfile, and README changes are shown. Do not commit or push these changes unless the user separately requests it.
