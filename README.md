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
uv sync
uv run bot.py -t webrtc
```

Open `http://localhost:7860/client`, allow microphone access, and connect.

The browser's WebRTC echo cancellation, automatic gain control, and noise suppression are the iteration-1 noise baseline. No custom frontend or server-side filter is required.

Gemini server VAD does not emit Pipecat's local user-speaking frames. User-speaking, turn-duration, and user-to-bot latency events become available when a local VAD backend is selected in a later iteration. Transcripts, service metrics, pipeline errors, and first-bot-speech latency remain available in iteration 1.

## Test

```powershell
uv run pytest -v
```

## Scope

Iteration 1 is browser-only. Exotel calling, public deployment, persistence, tools, and authentication are excluded.
