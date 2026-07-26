# voiceassistant

Minimal Hindi-first Pipecat voice agent using Gemini Live and Pipecat's built-in SmallWebRTC browser client.

## Requirements

- Python 3.11 or 3.12
- [uv](https://docs.astral.sh/uv/)
- Google AI Studio API key with Gemini Live model access

## Base Setup

```powershell
# Preserve an existing .env; create it only when absent
if (-not (Test-Path -LiteralPath '.env')) { Copy-Item '.env.example' '.env' }
# Set GOOGLE_API_KEY in .env
uv sync
uv run pytest -v
uv run bot.py -t webrtc
```

Open `http://localhost:7860/client`, allow microphone access, and connect. SmallWebRTC browser operation has been verified.

## Runtime Configuration

- `GEMINI_MODEL`: defaults to `models/gemini-2.5-flash-native-audio-preview-12-2025`, as shown in `.env.example`.
- `GEMINI_VOICE`: defaults to `Charon`.
- `VAD_BACKEND`: `gemini` (default), `silero`, `ten`, `firered`, or `cobra`.
- `NOISE_FILTER`: `browser` (default), `rnnoise`, or `koala`.

The model remains configurable through `GEMINI_MODEL`; model choice can dominate end-to-end latency. In manual use, latency improved after selecting a faster Gemini Live model through `GEMINI_MODEL`. The model identifier is intentionally not recorded here because availability and naming can vary by account and release.

The internal input rate is always 16 kHz. Optional imports are lazy, so base Gemini mode does not require local VAD or server-side noise packages. Shell environment values override `.env`; this repository does not overwrite an existing ignored `.env`.

## VAD Backends

### Gemini

No extra installation is required. Set `VAD_BACKEND=gemini`. Gemini Live performs server-side turn detection. This mode may expose fewer local user-speaking and turn events than local VAD modes.

### Silero

No extra installation beyond the base Pipecat package is required. Set `VAD_BACKEND=silero`. Silero local VAD has passed manual browser testing; its first startup may download or load the ONNX model.

### TEN VAD

TEN is community-maintained and tested separately from Pipecat releases. This project provides a minimal Pipecat adapter around the pinned upstream native package, so treat it as an optional evaluation path.

Exact Windows PowerShell setup:

```powershell
uv sync --extra ten
```

Set `VAD_BACKEND=ten`. TEN requires Windows x64 and 16 kHz audio. `TEN_VAD_THRESHOLD` is the native speech gate; frames it rejects return zero confidence. Pipecat then applies `VAD_CONFIDENCE` as a second gate to accepted probabilities.

### FireRedVAD

FireRed is community-maintained. The `fireredvad[cpu]==0.0.2` upstream package and `pipecat-firered-vad==0.1.0` adapter are installed from PyPI by the optional extra; model files and Windows compatibility can still vary.

Exact Windows PowerShell setup from the project root:

```powershell
uv sync --extra firered
uvx --from "huggingface_hub[cli]" huggingface-cli download FireRedTeam/FireRedVAD --local-dir pretrained_models/FireRedVAD
```

Set `VAD_BACKEND=firered` and `FIREREDVAD_MODEL_DIR=pretrained_models/FireRedVAD/Stream-VAD`. `FIREREDVAD_SPEECH_THRESHOLD` defaults to `0.6`; lower values accept softer speech and higher values reject more noise. Set `FIREREDVAD_USE_GPU=1` only when a working CUDA environment exists.

### Picovoice Cobra

```powershell
uv sync --extra cobra
```

Set `VAD_BACKEND=cobra` and `PICOVOICE_ACCESS_KEY`. The local adapter converts 16-bit little-endian mono PCM to Cobra frames, uses Cobra's required frame length, and releases the engine during pipeline cleanup.

## VAD Tuning

- `VAD_CONFIDENCE`: Pipecat confidence threshold, default `0.7`.
- `VAD_START_SECS`: confirmed speech required before speech start, default `0.2`.
- `VAD_STOP_SECS`: silence required before speech end, default `0.3`.
- `VAD_MIN_VOLUME`: minimum normalized volume, default `0.6`.
- `TEN_VAD_THRESHOLD`: native TEN speech gate, default `0.6`; `VAD_CONFIDENCE` remains Pipecat's second gate.
- `FIREREDVAD_SPEECH_THRESHOLD`: FireRed's model-level speech gate, default `0.6`.

Silero uses the shared Pipecat controls (`VAD_CONFIDENCE`, `VAD_START_SECS`, `VAD_STOP_SECS`, and `VAD_MIN_VOLUME`). It does not expose FireRed's internal smoothing-window or minimum-frame settings.

Change one setting at a time. Backend-specific controls remain separate because they are not equivalent across models.

## Noise Filtering

Noise filtering is selected independently from VAD. Any supported `NOISE_FILTER` can be paired with any installed `VAD_BACKEND`.

### Browser

`NOISE_FILTER=browser` uses browser WebRTC echo cancellation, automatic gain control, and noise suppression and requires no server dependency. Use this baseline first.

### RNNoise

```powershell
uv sync --extra rnnoise
```

Set `NOISE_FILTER=rnnoise`.

### Picovoice Koala

```powershell
uv sync --extra koala
```

Set `NOISE_FILTER=koala` and `KOALA_ACCESS_KEY`. Koala filtering is independent of Cobra VAD. `PICOVOICE_ACCESS_KEY` and `KOALA_ACCESS_KEY` are intentionally separate settings.

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

For every installed backend, keep the microphone, room, browser, speaker distance, `GEMINI_MODEL`, prompt, and noise filter unchanged. Run one quiet-room pass and one repeatable noisy-room pass with the same Hindi and Hindi-English phrases. Change only `VAD_BACKEND`, then repeat the checklist.

Record:

| Backend | Condition | Missed starts | False starts | Clipped first syllables | End delay | Premature ends | Barge-in | CPU/GPU | Install notes |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| gemini | quiet |  |  |  |  |  |  |  |  |
| gemini | noisy |  |  |  |  |  |  |  |  |
| silero | quiet |  |  |  |  |  |  |  |  |
| silero | noisy |  |  |  |  |  |  |  |  |

Add rows only for optional backends that are actually installed. This is a learning comparison, not a statistically rigorous benchmark. Compare latency only while holding `GEMINI_MODEL` constant; otherwise model speed can obscure VAD differences.

## Events and Metrics

Pipeline and service metric frames are enabled. Current logs expose first-speech latency; user-to-bot latency and turn events are exposed only where local VAD frames exist. Logs also expose connection events, transcripts, interruption state, and pipeline errors when their frames are available.

Gemini server VAD does not provide the local VAD frames required by Pipecat turn tracking and frame-driven user-to-bot latency measurement.

## Exotel Readiness, Not Implementation

Exotel code is intentionally excluded. A later implementation will need:

- Runner-selected WebSocket transport and Pipecat `ExotelFrameSerializer`.
- A public media WebSocket endpoint and Exotel flow/webhook configuration.
- Inbound and outbound call control.
- Explicit hang-up handling.
- Telephone-specific 8 kHz testing and presets.

The current boundary is ready for that later work: `bot(runner_args)` owns transport creation, `run_agent(transport, runner_args)` owns reusable conversation behavior, and the internal input rate stays 16 kHz. Exotel's 8 kHz PCM must be resampled at the serializer boundary; no Exotel dependency or conditional belongs in `run_agent`.
