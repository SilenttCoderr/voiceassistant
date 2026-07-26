# Pipecat Voice Agent Design

## Goal

Build a local, browser-based Pipecat voice agent for learning real-time voice AI end to end. The first agent will converse primarily in Hindi, support Hindi-English code-switching, allow interruptions, expose useful runtime events, and make voice activity detection (VAD) implementations swappable.

The design must preserve a clean path to future Exotel inbound and outbound calling without restructuring the core agent.

## Scope

### Included

- Python 3.11+ Pipecat server.
- Local browser connection through Pipecat's `SmallWebRTCTransport` and development runner.
- Gemini Live native speech-to-speech service using a Google AI Studio API key.
- Hindi-first conversational prompt with concise, spoken responses.
- Conversation context, greeting, interruption handling, clean disconnect, and reconnect as a new session.
- Connection, speaking, transcription, interruption, latency, and service error visibility where Pipecat and Gemini expose those events.
- Selectable VAD backends: Gemini server VAD, Silero, TEN VAD, FireRedVAD, and Picovoice Cobra.
- Browser WebRTC audio processing as the default noise-suppression baseline.
- Documentation and a repeatable manual comparison procedure for VAD behavior.
- Transport-independent agent construction for later Exotel support.

### Excluded From The First Implementation

- Public deployment.
- Exotel account setup, phone-number configuration, public webhooks, and call initiation.
- Authentication, database storage, long-term memory, business tools, and knowledge retrieval.
- React or a custom production frontend.
- Guaranteed support for Indic languages beyond Hindi.
- Installing every optional VAD backend by default.
- Production observability and automated conversational evaluation.

## Architecture

The application has two boundaries:

1. `bot(runner_args)` creates the requested transport using Pipecat's runner utilities.
2. `run_agent(transport)` creates and runs the transport-independent conversational pipeline.

The initial transport is SmallWebRTC. Future Exotel support supplies a WebSocket transport configured with Pipecat's `ExotelFrameSerializer` while reusing `run_agent(transport)` unchanged.

### Initial Audio Flow

Browser microphone -> SmallWebRTC input -> Pipecat pipeline -> Gemini Live -> SmallWebRTC output -> browser speaker.

Gemini Live performs speech understanding, response generation, and voice synthesis in one real-time service. This keeps the first build small while still exposing Pipecat transports, frames, context, lifecycle, VAD configuration, interruptions, and metrics.

### Future Exotel Audio Flow

Phone call -> Exotel media WebSocket -> `ExotelFrameSerializer` -> Pipecat pipeline -> Gemini Live -> serializer -> Exotel call.

Exotel uses 8 kHz raw linear PCM. The serializer will resample telephone audio to the agent's 16 kHz internal input rate. Telephone audio quality and VAD thresholds will be evaluated separately from browser audio.

## Components

### Bot Entry Point

The entry point accepts Pipecat `RunnerArguments`, defines transport parameters, delegates transport creation to Pipecat's `create_transport`, and passes the result to `run_agent`.

The initial configuration enables WebRTC. Exotel will later be added as another transport parameter entry rather than as a separate agent implementation.

### Agent Pipeline

`run_agent(transport)` owns:

- Gemini Live service configuration.
- Hindi-first system instruction.
- LLM context and realtime context aggregators.
- The selected VAD analyzer.
- Pipeline construction and worker lifecycle.
- Greeting, client-ready, disconnect, and error handlers.
- Metrics and diagnostic logging.

No browser-specific or Exotel-specific behavior belongs in this function.

### Browser Client

Milestone one uses Pipecat's development runner client rather than a custom frontend. It provides microphone access, connect/disconnect behavior, and local WebRTC signaling with no paid transport account.

A custom UI is deferred until a concrete requirement is not met by the runner client.

## Gemini Live Configuration

- API key: `GOOGLE_API_KEY` from `.env`.
- Model: configurable through `GEMINI_MODEL`, defaulting to `models/gemini-2.5-flash-native-audio-preview-12-2025`, Pipecat's documented Gemini Live default at design time.
- Response modality: audio.
- Language behavior: Hindi-first, permit natural Hindi-English code-switching, and respond in the user's language unless asked otherwise.
- Response style: brief, conversational, no Markdown-oriented formatting, and suitable for speech playback.
- Context: in-memory and limited to the current session.

Quota, authentication, model-access, and connection failures must be logged clearly without printing API keys.

## Swappable VAD Design

`VAD_BACKEND` selects one backend:

| Value | Backend | Setup |
| --- | --- | --- |
| `gemini` | Gemini Live server-side VAD | Default; no local model |
| `silero` | Pipecat `SileroVADAnalyzer` | Included with Pipecat; CPU ONNX |
| `ten` | Community `TenVadAnalyzer` | Optional package; 16 kHz; Windows x64 supported |
| `firered` | Community `FireVadAnalyzer` | Optional package, upstream repository, and downloaded weights; 16 kHz |
| `cobra` | Picovoice Cobra through a small Pipecat adapter | Optional `pvcobra` package and `PICOVOICE_ACCESS_KEY` |

A single factory creates the selected analyzer and validates backend-specific configuration. Optional imports remain inside their backend branches so the default Gemini/Silero installation does not require every VAD dependency.

When `VAD_BACKEND=gemini`, Gemini server VAD stays enabled. Any local backend disables Gemini server VAD and supplies the analyzer to Pipecat's realtime context aggregator with `realtime_service_mode=True`.

Common configuration will cover the closest supported equivalents of:

- Speech confidence or backend threshold.
- Required speech duration before speech start.
- Silence duration before end of turn.

Backend-specific controls remain optional environment settings rather than being forced into a misleading universal abstraction.

FireRed exposes `FIREREDVAD_SPEECH_THRESHOLD`, defaulting to `0.6` and validated within `0.0` to `1.0`. This model-level gate is separate from Pipecat's shared `VAD_CONFIDENCE`, `VAD_START_SECS`, `VAD_STOP_SECS`, and `VAD_MIN_VOLUME` controls. Silero uses those shared Pipecat controls and does not expose FireRed's internal smoothing or minimum-frame settings.

### Cobra Adapter

Pipecat does not currently document a built-in Cobra analyzer. The project will therefore contain one minimal adapter implementing Pipecat's VAD analyzer contract around `pvcobra`.

The adapter will:

- Initialize Cobra with `PICOVOICE_ACCESS_KEY`.
- Enforce Cobra's required sample rate and frame length.
- Convert incoming 16-bit mono PCM frames to the input expected by Cobra.
- Return voice probability to Pipecat.
- Release Cobra resources during cleanup.

## Noise Suppression

Noise suppression and VAD remain separate concerns.

The first build relies on browser WebRTC echo cancellation, automatic gain control, and noise suppression where the browser enables them. The documentation will identify this as the baseline so VAD comparisons are interpreted correctly.

Optional server-side filters such as RNNoise or Picovoice Koala may be added after the base agent works. A filter will be selected independently from `VAD_BACKEND`; adding one must not change the agent or VAD interfaces.

## Events And Observability

Development logs should make these states visible when available:

- Client connected, ready, disconnected, and reconnected.
- User speaking started/stopped.
- Bot speaking started/stopped.
- User and assistant transcript events.
- Interruption/barge-in events.
- User-to-bot latency and Pipecat service metrics.
- Missing configuration, invalid API key, quota exhaustion, model access failure, and transport failure.

Logs must not include secrets or raw environment values.

## Error Handling

- Validate required environment variables before starting a session.
- Fail with a backend-specific setup message when an optional VAD is selected but unavailable.
- Cancel the worker and release VAD/service resources when the client disconnects.
- Treat a browser reconnect as a new in-memory conversation; cross-session history is out of scope.
- Let Pipecat/Gemini reconnection behavior handle transient service disconnects, while logging the final failure if retries are exhausted.
- Keep the browser session usable after an agent process fails by returning a clear startup or connection error.

## Verification

### Automated Checks

- A small VAD factory test verifies supported names, the default, invalid-name errors, and optional dependency errors without loading all optional models.
- A startup/configuration smoke check verifies that the base modules import and missing required keys produce a useful message.

### Manual Voice Checklist

- Start the local runner and connect from the browser.
- Hear a short Hindi greeting.
- Hold a multi-turn Hindi conversation.
- Switch naturally between Hindi and English.
- Interrupt the bot while it is speaking and confirm prompt cancellation.
- Pause mid-sentence and verify the VAD does not end the turn too aggressively.
- Remain silent and verify the bot does not generate false turns.
- Disconnect and reconnect cleanly.
- Confirm invalid-key and exhausted-quota failures are understandable.

### VAD Comparison

Each available local VAD will be tested with the same Hindi speech samples and the same quiet/noisy conditions. Record:

- Missed speech starts.
- False speech starts.
- Clipped initial syllables.
- End-of-turn delay.
- Premature end-of-turn events.
- Barge-in responsiveness.
- CPU/GPU usage and installation difficulty.

The comparison is a learning benchmark, not a claim of statistically rigorous model ranking.

## Exotel Readiness

The first implementation remains browser-only, but these decisions prevent a future rewrite:

- The agent accepts a Pipecat base transport instead of constructing WebRTC internally.
- Transport creation stays in the runner entry point.
- The internal speech path uses 16 kHz audio, with Exotel resampling handled by `ExotelFrameSerializer`.
- Conversation behavior, Gemini configuration, VAD selection, metrics, and errors stay transport-independent.
- Telephony-specific metadata and DTMF frames will be handled at the transport/session edge.

Future inbound Exotel work adds Exotel flow/webhook configuration and a public WebSocket endpoint. Future outbound work additionally invokes Exotel's call API. Both directions require explicit call termination because Pipecat's Exotel serializer does not provide automatic hang-up.

Telephony will need separate presets and validation because 8 kHz phone audio is narrower than browser audio. This is configuration and testing work, not an agent architecture change.

## Implementation Milestones

1. Create the minimal Python project and Gemini Live WebRTC conversation.
2. Add session events, transcripts where available, metrics, and clear errors.
3. Add the VAD factory and Silero local VAD mode.
4. Add optional TEN, FireRedVAD, and Cobra backends with isolated dependencies.
5. Document and run the VAD comparison procedure.
6. Evaluate optional RNNoise or Koala filtering without blocking the base project.
7. In a later project phase, add the Exotel transport and inbound/outbound call control.

## Success Criteria

The first phase is successful when the user can run the project locally, talk to a Hindi-first Gemini voice agent in the browser, interrupt it naturally, inspect relevant events and latency, and switch among installed VAD backends without changing pipeline code.

The architecture is successful when adding Exotel requires transport and call-control code but no rewrite of the core agent, Gemini service, prompt, context, VAD selection, or diagnostic components.
