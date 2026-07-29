# CLAUDE.md

Hindi-first browser voice agent: Pipecat + Gemini Live (native speech-to-speech) over
SmallWebRTC. The point of the repo is learning real-time voice AI, with **swappable VAD
backends** as the main axis of experimentation.

## Commands

```powershell
uv sync                     # base deps only; optional backends are extras
uv run pytest -v            # full suite (fast, no network, no API key needed)
uv run pytest tests/test_vad.py -k firered -v
uv run bot.py -t webrtc     # then open http://localhost:7860/client
```

Prefix shell commands with `rtk` (token-saving proxy). Use `rtk proxy <cmd>` when you need
byte-exact output — `rtk` and the headroom proxy both compress tool output lossily, which
mangles source files read through `Read`/`Grep`. Reading `.py` files in <60-line chunks, or
`rtk proxy sed -n 'A,Bp' file`, gives faithful text.

Optional extras, install only when testing that path:
`uv sync --extra ten|firered|cobra|rnnoise|koala`

## Layout

- `bot.py` — two boundaries, deliberately: `bot(runner_args)` builds the transport,
  `run_agent(transport, runner_args)` builds the transport-independent pipeline. Keep them
  separate; Exotel telephony is a planned future transport that must reuse `run_agent`
  unchanged.
- `vad.py` — `create_vad(backend)` factory + local `VADAnalyzer` adapters
  (`CobraVADAnalyzer`, `TenVADAnalyzer`, `CompatibleFireVadAnalyzer`).
- `lab.py` + `lab.html` — offline VAD/turn tuning lab (stdlib `http.server`, vanilla JS,
  no build step). Replays a recorded WAV through the real `create_vad()` while sweeping
  thresholds. Drives the analyzers directly, so it must call `set_sample_rate(16000)` on
  both the VAD and turn analyzer — the pipeline normally does that at transport start.
- `tests/` — pure unit tests; every optional dependency is faked via `sys.modules`.
- `docs/superpowers/{specs,plans}/` — dated design spec and task-by-task plans. Read the
  spec before changing architecture; it defines what is intentionally excluded.

Config is entirely environment variables — see `.env.example` for the full list.

## Invariants (each one is enforced by a test)

- **UTF-8 before pipecat.** `_configure_utf8_streams()` runs at the top of `bot.py`, before
  any `pipecat` import. `test_direct_run_configures_utf8_before_pipecat_imports` parses the
  AST to enforce it — do not tidy the import order.
- **README is test-asserted.** `test_readme_uses_plain_uv_commands` (no `rtk ` in README)
  and `test_readme_metrics_match_current_logs` pin exact README sentences. Docs and code
  change together.
- **`.env.example` is test-asserted** (`test_env_example_uses_balanced_firered_defaults`).
- **Never touch `.env`** — gitignored, holds real keys and the user's live tuning.
- **16 kHz everywhere** internally, in transport params and pipeline params alike.
- **Optional imports stay lazy and inside the branch that needs them.** Catch
  `ModuleNotFoundError`, re-`raise` when `exc.name` is not the expected package, otherwise
  raise a `RuntimeError` naming the exact `uv sync --extra ...` command. A missing
  *transitive* dependency must not be reported as a missing optional backend.
- **Validate env vars before constructing anything**, via `_float_env` in `vad.py`. Errors
  must name the variable and never echo secrets.
- **Cleanup on failure:** `run_agent` awaits `analyzer.cleanup()` if pipeline construction
  raises.

## VAD wiring

`VAD_BACKEND=gemini` (default) returns `None` from `create_vad`, which means Gemini
server-side VAD (`GeminiVADParams(disabled=False)`), no turn tracking, and no frame-driven
user-to-bot latency. Any other backend returns a local analyzer, disables Gemini VAD, and
enables turn tracking. `observability_configuration()` keeps the logged notice honest about
which events actually exist — keep it that way rather than over-promising.

Adding a backend: add to `SUPPORTED_BACKENDS`, add a branch in `create_vad`, add the extra
to `pyproject.toml`, add faked-module tests, then document it in README and `.env.example`.

## Known trap

`bot.py` calls `load_dotenv(override=False)` at import time, so importing `bot` in tests
leaks the developer's real `.env` into `os.environ`. Tests that read a VAD env var must
`monkeypatch.setenv`/`delenv` it explicitly, or they pass or fail depending on the local
`.env`.
