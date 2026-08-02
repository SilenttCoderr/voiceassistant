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

Set `VAD_BACKEND=firered` and `FIREREDVAD_MODEL_DIR=pretrained_models/FireRedVAD/Stream-VAD`. `FIREREDVAD_SPEECH_THRESHOLD` defaults to `0.6`; lower values accept softer speech and higher values reject more noise. FireRed's thresholded decision is authoritative, while Pipecat controls turn timing. Set `FIREREDVAD_USE_GPU=1` only when a working CUDA environment exists.

### Picovoice Cobra

```powershell
uv sync --extra cobra
```

Set `VAD_BACKEND=cobra` and `PICOVOICE_ACCESS_KEY`. The local adapter converts 16-bit little-endian mono PCM to Cobra frames, uses Cobra's required frame length, and releases the engine during pipeline cleanup.

## VAD Tuning

- `VAD_CONFIDENCE`: Pipecat confidence threshold, default `0.5`. FireRed emits binary confidence, so this remains a simple acceptance gate.
- `VAD_START_SECS`: confirmed speech required before speech start, default `0.20`.
- `VAD_STOP_SECS`: silence required before speech end, default `0.45`.
- `VAD_MIN_VOLUME`: minimum normalized volume, default `0.35`.
- `TEN_VAD_THRESHOLD`: native TEN speech gate, default `0.6`; `VAD_CONFIDENCE` remains Pipecat's second gate.
- `FIREREDVAD_SPEECH_THRESHOLD`: FireRed's model-level speech gate, default `0.6`.

The balanced FireRed preset targets mixed boardroom, street, metro, and train conditions: threshold `0.6`, FireRed smoothing window `5`, start delay `0.20s`, stop delay `0.45s`, and minimum volume `0.35`. FireRed's internal minimum speech/silence frame counters do not define Pipecat turn boundaries in this integration.

Silero uses the shared Pipecat controls (`VAD_CONFIDENCE`, `VAD_START_SECS`, `VAD_STOP_SECS`, and `VAD_MIN_VOLUME`) directly. It does not expose FireRed's internal smoothing-window or minimum-frame settings.

Change one setting at a time. Backend-specific controls remain separate because they are not equivalent across models.

## VAD Lab

```powershell
# Start the lab, then open http://127.0.0.1:7861
uv run lab.py

# Serve on a different port
$env:LAB_PORT = '8000'; uv run lab.py

# Comparing backends needs them installed together in one command, because each
# `uv sync` makes the environment match exactly the extras named in that call and
# removes the rest. Name every backend you want to compare:
uv sync --extra ten --extra firered --extra cobra

# Which backends are actually installed right now
uv run python -c "import importlib.util as u; print({n: u.find_spec(m) is not None for n, m in {'ten':'ten_vad','firered':'pipecat_firered_vad','cobra':'pvcobra'}.items()})"
```

An offline lab for tuning VAD and turn detection against a fixed recording, instead of
guessing at parameters during a live call.

Record a clip in the environment that actually breaks the agent — a train, a boardroom —
or load an existing `.wav`. Then sweep thresholds against that same audio as many times as
you like. Analysis runs through the project's own `create_vad()` and the same Smart Turn
model the agent uses, so a setting that looks right in the lab is a setting you can paste
into `.env`.

### Slots

A slot is one backend plus its own settings. `+ add slot` creates another, and the
duplicate button (⧉) copies a slot so the *same* model can be run twice at different
thresholds. Every slot is analysed against the same clip in a single request, so
`silero @ confidence 0.3` and `silero @ confidence 0.7` can be compared directly rather
than across two recordings.

### Reading the view

- **mel** — 64-band log-mel spectrogram, 10 ms hop, shared by every slot. Speech shows as
  stacked horizontal harmonics; steady machinery shows as flat bands; a train shows as
  broadband smear.
- **wave** — per-frame peak amplitude.
- Then one row per slot, colour-matched to the slot card:
  - **confidence** — that backend's raw per-frame score, with the slot's `VAD_CONFIDENCE`
    drawn as a dashed line. Anything above the line counts as speech.
  - **state** — the resulting `QUIET` / `STARTING` / `SPEAKING` / `STOPPING` band, which is
    what actually drives turn taking.

Backends run at different frame sizes (Silero 32 ms, FireRed 25 ms, TEN 16 ms), so rows are
positioned by timestamp rather than by frame index, and stay aligned with each other and
with the spectrogram.

### Denoising

Each slot has its own `denoise` setting, applied to the clip before the VAD sees it —
exactly where `NOISE_FILTER` sits in the pipeline. Because it is per slot, the honest test
is two slots that differ only in that field: same backend, same thresholds, one raw and one
cleaned. Filters buffer internally, so a cleaned slot's audio is a few milliseconds shorter
than the raw one.

The `browser cleanup` checkbox in the audio panel controls echo cancellation, noise
suppression and gain control *at record time*. Leave it off to capture the untouched room;
tick it to record what the agent hears today, since `NOISE_FILTER=browser` is the shipped
default.

What is actually available:

| Option | Cost | Notes |
|---|---|---|
| `browser` | free | WebRTC AEC/NS/AGC, no server dependency. The shipped default. |
| `highpass` | free, local | Butterworth high-pass, `HIGHPASS_HZ` (default `100`). Also a `NOISE_FILTER` option. |
| `rnnoise` | free, local | `uv sync --extra rnnoise`. Speech-selective, runs on CPU. |
| `highpass+rnnoise` | free, local | Any filters joined with `+` chain left to right, in the agent as well as the lab. |
| `deepfilternet` | free, lab only | Runs out of process; see below. |
| `koala` | Picovoice key | `uv sync --extra koala`, needs `KOALA_ACCESS_KEY`. |
| ai-coustics | commercial licence | Pipecat ships `aic_filter`, needs a `license_key`. Not wired up here. |
| Krisp Viva | commercial licence | Pipecat ships `krisp_viva_filter`, needs an `api_key`. Not wired up here. |

Filters chain with `+`, so `NOISE_FILTER=highpass+rnnoise` drops rumble before the denoiser
runs, leaving it only the speech band to work on. Stages run left to right. `browser` adds
no server-side filter and simply drops out of a chain.

`highpass` is the targeted answer to train rumble: speech carries almost nothing below
100 Hz, so cutting that band raises the contrast between voice and a rumbling room. On a
synthetic test it removed 74% of the energy below 100 Hz while leaving the 150–600 Hz
speech band within 0.4%. It is a fixed filter, not a model — cheap, predictable, and unable
to remove anything that overlaps the speech band.

#### DeepFilterNet

Available in the lab only, and it does **not** live in this venv. It cannot: `deepfilterlib`
has no cp312 wheel, the package pins `numpy<2`, and it imports `torchaudio.backend`, which
torchaudio removed in 2.1. Instead `lab.py` shells out to

```
uv run --python 3.11 --no-project --index-strategy unsafe-best-match \
  --with deepfilternet==0.5.6 --with torch==2.0.1 --with torchaudio==2.0.2 \
  python deepfilter_worker.py <in.wav> <out.wav>
```

so the project stays on Python 3.12 with numpy 2.x. uv builds that environment once and
caches it; the first run downloads torch and the DeepFilterNet3 checkpoint. Expect a few
seconds per clip — `enhance()` is an utterance-level API, which is also why this is a lab
comparison tool and not a `NOISE_FILTER` the live agent can use.

Its integration test is opt-in, since it spawns that second environment:

```powershell
$env:LAB_TEST_DEEPFILTER = '1'; uv run pytest tests/test_lab.py -k deepfilternet -v
```

RNNoise and `highpass` are the only free, local, in-process options. Everything stronger
either needs a paid key or, like DeepFilterNet, has to run out of process.

Worth keeping in mind while reading results: every one of these is trained to keep speech
and remove everything else, so they help against a train's mechanical noise and do nothing
about a boardroom full of other people talking.

### Listening

Click anywhere on the view to play from that moment; drag to play just that span. A white
playhead tracks the audio across every lane. This is how you settle whether a stretch the
VAD called speech was actually a voice — look at the spectrogram, then listen to the same
span and decide for yourself.

Dashed vertical markers show Smart Turn predictions with their probability. The summary
table reports speech percentage, segment count and turn ends per slot. Segment count is the
one to watch: a slot reporting many more segments than the others is chopping one sentence
into several turns, which in the agent reads as the user stopping and starting repeatedly.

A slot that cannot load — missing extra, missing key, missing model — is reported in its own
row of the table, and the remaining slots still produce a comparison.

Recording deliberately disables browser echo cancellation, noise suppression and gain
control, so what you capture is the raw room rather than an already-cleaned signal.

Two limits worth knowing. `VAD_BACKEND=gemini` cannot be replayed, because that VAD runs
inside Google's service; the lab rejects it. And Smart Turn's `max_duration_secs` trimming
keys off wall-clock arrival times, so offline replay does not reproduce it for segments
longer than that setting.

## Sweep

The lab compares a few slots on one clip by eye. `sweep.py` is the other half: it
replays a folder of labelled clips through a whole parameter grid and scores every
combination, so a setting gets chosen by a number rather than a hunch — and the same
number can be recomputed after any later change to the chain.

### The UI

`uv run lab.py`, then open <http://localhost:7861/sweep>. This is the whole loop in one
page: draw where the speech is, save, pick what to vary, run.

- Drag across empty canvas to add a region, drag an edge to move it, drag the middle to
  slide the whole thing. Click a region and press Delete to remove it. A click without a
  drag seeks the player instead of leaving a sliver behind.
- **save labels** writes `clips/<name>.wav` and `clips/<name>.json` together, so the pair
  can never drift apart. The browser decodes and downmixes first, which means an m4a from
  Voice Recorder can be dropped straight in and the server only ever stores 16 kHz mono
  WAV — no ffmpeg involved on this path.
- The sweep panel takes comma-separated numbers per axis and checkboxes for the rest. It
  shows the config count as you type, because 4 backends × 5 confidences × 4 filters is
  80 runs and worth noticing before you start one.
- Results land in a sortable table: click any metric header to rank by it instead of by
  the weighted score. Only settings that actually differ get a column.

Filling in the speaker-gate enroll path adds `speaker_threshold: null` automatically, so
the ungated baseline is always in the table next to the gated rows.

The page waits for the whole grid — the server is single threaded, so a long sweep blocks
it. That is fine for one person on localhost; watch the terminal if you want progress.

Everything below also works from the command line, on the same files.

A clip is a WAV plus a labels file of the same name:

```
clips/train-window-seat.wav
clips/train-window-seat.json    {"speech": [[1.2, 3.4], [6.0, 7.1]]}
```

Region times are seconds into the clip. `"speech": []` marks a clip that must never
trigger at all — pure background, or other people talking near the mic. Those negative
clips are what make a false-trigger rate mean anything, so record a few.

Any format ffmpeg can read works from the command line — `clips/train.m4a` is fine, it
gets transcoded once per run. Without ffmpeg on PATH the error says so and names the
install command. The UI never needs it, because the browser decodes first.

```powershell
uv run sweep.py label clips/train-window-seat.wav   # propose regions, then fix by ear
uv run sweep.py run --top 15                        # rank the grid
uv run sweep.py run --grid mygrid.json --sort false_per_min --json out.json
```

`label` is an energy gate keyed off the clip's own loudest frame. It gives a usable
draft when speech is clearly above the background and a poor one when it is not —
always check its output before trusting a score built on it.

The grid file is `{"grid": {"backend": ["ten"], "confidence": [0.4, 0.6]}, "weights": {...}}`;
the defaults live in `DEFAULT_GRID` and `DEFAULT_WEIGHTS` in `sweep.py`. Any key
`lab.py` understands is sweepable, including `noise_filter` chains such as
`highpass+rnnoise`. Keys that belong to another backend are dropped before the grid
expands, so `ten_threshold` does not multiply the silero rows.

Columns, all lower-is-better:

- `miss%` — labelled speech the VAD never flagged. High means clipped words.
- `fals/min` — separate speech onsets outside every labelled region, per minute of
  non-speech. This is the number that predicts the agent interrupting itself.
- `fals_s` — total seconds spent falsely in speech.
- `onset` — median milliseconds from a region's start to the VAD noticing.
- `tail` — median milliseconds the VAD holds the turn open after the words stop. This
  is what `stop_secs` buys and what the user feels as reply latency.

`score` is a weighted sum of the four, and the weights are a starting guess, not a
finding. If the ranking disagrees with your ears, sort by the single column you care
about instead of arguing with the weights.

Denoising is the slow part and does not depend on the VAD settings, so each clip is
filtered once per chain and every VAD variant replays the same filtered audio. A
config that cannot load — missing extra, missing key — is reported under the table and
the rest of the sweep still finishes.

### Repeatability

A score is worthless if the same command gives a different answer twice, and this
took two fixes to get right.

Pipecat's `SOXRStreamAudioResampler` clears its filter history whenever more than
`clear_after_secs` of **wall clock** passes between chunks. That is correct for a
live call and wrong for an offline replay: a GC pause or a model load between two
chunks wiped the history mid-clip and rewrote the rest of the audio. `lab.py` now
suppresses that during replay only — the live pipeline keeps it.

Underneath that, soxr's stream resampler at `HQ`/`VHQ` with int16 rounds two
different ways run to run. It is only ±2 LSB against a signal peak of ~25000, but
RNNoise is a recurrent net and amplifies it until VAD decisions move a frame and
close scores swap places. `QQ` and the float32 path are unaffected.

So denoised audio is cached under `clips/.cache/` as plain WAVs, keyed by the audio
and the filter settings. A clip is filtered once, ever, and every later sweep replays
the same bytes — three consecutive sweeps now produce byte-identical JSON, and repeat
runs are about 30% faster. The cache key does not cover the filter *code*, so delete
the folder after changing `noise.py` or upgrading Pipecat. `--no-cache` re-denoises
every time, and the scores will drift a little when you do.

You can also just listen to a cached file to hear exactly what got scored.

Scores are only as good as the labels. Three honest clips of the room that actually
breaks the agent beat thirty guessed ones.

## Speaker Gate

VAD answers "is anyone talking". On a train that is the wrong question — the agent
should wake for one person and ignore the man two seats away. A speaker embedding
turns a stretch of speech into a vector and compares it against an enrolled
recording, so strangers get dropped before the agent answers them.

```powershell
uv sync --extra speaker
```

Models are plain ONNX files from the [sherpa-onnx zoo](https://github.com/k2-fsa/sherpa-onnx/releases/tag/speaker-recongition-models),
downloaded into `pretrained_models/speaker/`. The backend is only which file you point
at, so these swap without a code change:

| File | Size | Trained on |
| --- | --- | --- |
| `3dspeaker_speech_campplus_sv_zh_en_16k-common_advanced.onnx` | 27 MB | 200k speakers, code-switched Mandarin+English |
| `nemo_en_titanet_large.onnx` | 97 MB | English (VoxCeleb etc.) |

There is no Hinglish-trained speaker model — not on Hugging Face, not from AI4Bharat,
not from NVIDIA. It matters less than it sounds: embeddings are text- and
language-independent, they model how a voice sounds rather than what it says. Treat
that as a claim to test on your own clips, not a guarantee.

First pick a threshold. Enroll from one WAV or a directory of them, then score clips
of yourself and clips of other people:

```powershell
uv run sweep.py similarity --enroll clips/enroll/ clips/me-1.wav clips/stranger-1.wav
```

On the sherpa-onnx sample speakers, CAM++ gives 0.854 for the enrolled speaker
against 0.191 and 0.022 for two strangers; TitaNet gives 0.825 against 0.290 and
0.169. Put the threshold in the gap. A narrow gap means the gate will cost you real
turns.

Then sweep it like any other parameter — `speaker_enroll` plus `speaker_threshold` in
the grid turns the gate on, and `speaker_threshold: null` leaves it off for a
baseline row:

```json
{"grid": {"speaker_enroll": ["clips/enroll"], "speaker_threshold": [null, 0.4, 0.5, 0.7]}}
```

On a 17 s clip of one enrolled speaker followed by a stranger, the gate is the whole
difference between answering the stranger and not:

```
  score  miss% fals/min  fals_s   speaker_threshold
   44.9   40.1     0.00     0.0                 0.4
   44.9   40.1     0.00     0.0                 0.5
  114.0   59.6     0.00     0.0                 0.7   <- now eating real speech
  180.8   40.1    27.17     5.3                None   <- stranger wakes the agent
```

Two honest caveats. The gate judges each segment whole, which is a luxury the live
pipeline will not have — it has to decide a few hundred milliseconds in — so read
these numbers as the ceiling, not the forecast. And segments shorter than 400 ms are
left alone, because an embedding from a fragment that short is closer to noise than
to a voiceprint; they stay in the audio and stay counted against you.

Nothing in `bot.py` calls any of this yet. Measure first, wire it into the pipeline
once a sweep says it earns its latency.

## Turn Detection

VAD decides whether someone is speaking. Turn detection decides whether *you have finished
speaking*. It runs on top of local VAD frames, so it is active only when `VAD_BACKEND` is not
`gemini`.

- `TURN_DETECTION`: `smart` (default) or `vad`.
- `SMART_TURN_STOP_SECS`: maximum silence before the turn is forced closed, default `3.0`.
- `SMART_TURN_MAX_DURATION_SECS`: longest analysed speech segment, default `8.0`.

`smart` uses the Smart Turn v3.2 ONNX model bundled inside the `pipecat-ai` package, which
predicts whether an utterance sounds complete rather than ending the turn on silence alone.
Nothing is downloaded and no extra is required. This is also Pipecat's own default stop
strategy; the setting exists so the choice is visible in logs and can be switched off.

`vad` falls back to silence-timeout turn taking, ending the turn purely on `VAD_STOP_SECS`.
Use it as the A/B baseline when measuring whether the turn model actually helps in noise.

Every prediction is logged as `turn_prediction complete=<bool> probability=<float>
e2e_ms=<float>`. That line is the fastest way to tell whether a premature turn end came from
the turn model or from VAD timing.

The turn model's language coverage for Hindi is not verified here. Compare `smart` against
`vad` on your own audio before trusting it for Hindi-first conversations.

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
