# Pipecat Deep Dive — Architecture, and What It Means for This Product

Written against **Pipecat 1.6.0**, read from the local install at
`.venv/Lib/site-packages/pipecat`, not from the GitHub README. Every claim below
has a `file:line` you can check. Where the installed code and the public docs
disagree, the installed code wins — that's the code you actually run.

---

## 1. The mental model

Pipecat is not a "voice agent framework" in the sense of giving you a bot. It is
a **typed frame bus with an interruption model**. You assemble a list of
processors; frames flow through them in two directions; the framework's real
value is that it knows what to throw away when the user interrupts.

Three ideas carry almost everything:

1. **Frames** — every event is a dataclass. Audio chunks, transcripts, LLM
   tokens, errors, "user started speaking", "end the pipeline". 127 frame
   classes in `frames/frames.py`.
2. **Direction** — frames travel `DOWNSTREAM` (transport in → LLM → transport
   out) or `UPSTREAM` (a service telling earlier stages something). Upstream is
   not an afterthought; it's how transcripts get back to the turn logic.
3. **Interruption priority** — the frame's *category* decides its fate when the
   user barges in.

### Frame categories (`frames/frames.py:101-152`)

| Category | Ordering | On interruption |
|---|---|---|
| `SystemFrame` | Jumps the queue | **Survives** — unaffected by interruptions |
| `DataFrame` | In order | **Cancelled** |
| `ControlFrame` | In order | **Cancelled** |
| `UninterruptibleFrame` (mixin) | In order | **Survives** — stays queued, task not cancelled |

This table is the single most important thing to internalise. When you add a
processor and it "randomly stops working during barge-in", it's because you put
your payload in a `DataFrame`. When a cleanup step must always run, it needs the
`UninterruptibleFrame` mixin — that's why `EndFrame`, `StopFrame` and
`FunctionCallResultFrame` all carry it (`frames.py:765, 1745, 1769`). A function
call that's already in flight must return its result even if the user interrupts,
or the LLM context is left with a dangling tool call.

`InputAudioRawFrame` is a `SystemFrame` (`frames.py:1295`) — input audio is never
dropped by an interruption, which is what you want: the user's new speech is
exactly what caused the interruption.

### Pipeline (`pipeline/pipeline.py`)

A `Pipeline` is a list of `FrameProcessor`s wrapped in a `PipelineSource` and
`PipelineSink`. Source forwards downstream frames onward and hands upstream
frames to a callback; sink does the mirror. That's the whole abstraction. A
`PipelineWorker` (`pipeline/worker.py`, 56K) wraps it with lifecycle, metrics,
heartbeats, idle timeouts and observers.

---

## 2. The audio input path — where your noise work actually lives

This is the part most relevant to this repo, and the ordering is not obvious
from the docs.

Inside `transports/base_input.py`, the audio task handler does:

```
_audio_task_handler()                      # base_input.py:267
  └─ frame.audio = await audio_in_filter.filter(frame.audio)   # :282-283
```

**The `audio_in_filter` mutates the raw bytes before anything else sees them.**
VAD, turn analysis, and the LLM all receive filtered audio. So the noise stack in
`noise.py` — wired via `TransportParams(audio_in_filter=...)` in `bot.py:158` —
sits at architecturally the right place. Nothing downstream can recover the
original signal, which cuts both ways: a filter that mangles speech mangles it
for the VAD too, which is exactly the robotic-artifact problem hit with
`highpass+rnnoise`.

Filter lifecycle is managed for you: `start(sample_rate)` at `:131-132`, `stop()`
on end/cancel/cleanup at `:143, :167, :174`.

**Underused capability:** `base_input.py:245-246` handles
`FilterUpdateSettingsFrame` and forwards it to the live filter. There is also
`FilterEnableFrame` (`frames.py:2175`). You can retune or bypass noise
suppression *mid-call* by pushing a frame — no reconnect. For a product where
call conditions change (user walks out of the market into a room), that's a real
feature sitting unused.

### VAD (`audio/vad/vad_controller.py`)

`VADController` wraps a `VADAnalyzer` and turns audio into a `VADState` machine:
`QUIET → STARTING → SPEAKING → STOPPING`. Note `:144` — it calls
`set_sample_rate(frame.audio_in_sample_rate)` from the `StartFrame`. That's the
production path that `lab.py` has to imitate manually, which is why the lab must
call `set_sample_rate(16000)` on both the VAD and the turn analyzer by hand.

Available analyzers: `silero.py`, `aic_vad.py`, `aic_quail_vad.py`,
`krisp_viva_vad.py` — plus the local adapters in this repo's `vad.py`.

---

## 3. Turn-taking — the most important subsystem, and the one this repo barely uses

Pipecat 1.6 decomposes "whose turn is it" into three independent strategy
families under `turns/`. This is a substantial redesign and it is where the
answers to the noise-defense problem actually live.

### Start strategies (`turns/user_start/`) — *should the user's speech count as a turn?*

| Strategy | What it does |
|---|---|
| `VADUserTurnStartStrategy` | Turn starts on VAD speech. Pure acoustics. |
| `TranscriptionUserTurnStartStrategy` | Turn starts when a transcript arrives. |
| `MinWordsUserTurnStartStrategy` | **Turn starts only after N words.** |
| `WakePhraseUserTurnStartStrategy` | Turn starts only after a wake phrase. |
| `KrispVivaIPUserTurnStartStrategy` | ML model predicts real interruption vs backchannel. |
| `ExternalUserTurnStartStrategy` | Something outside the pipeline decides. |

**Default** (`user_turn_strategies.py:27-40`):
`[VADUserTurnStartStrategy(), TranscriptionUserTurnStartStrategy()]`.

`MinWordsUserTurnStartStrategy` deserves attention. Its core line:

```python
min_words = self._min_words if self._bot_speaking else 1
```

It demands N words to interrupt a *speaking* bot, but only 1 word when the bot is
silent. That is adaptive interruption handling and transcript-based garbage
rejection, in one built-in class, with an asymmetry that's exactly right: be
strict about barge-in, permissive about normal turns. A background "यार" won't
clear a 3-word bar while the bot is talking.

`KrispVivaIPUserTurnStartStrategy` is the commercial version — its docstring
states it "distinguishes genuine user interruptions from backchannels (e.g.
'uh-huh', 'yeah')" by running Krisp's Interruption Prediction model on audio
collected after VAD fires, gating `trigger_user_turn_started()` on a probability
threshold. Needs the `krisp_audio` SDK.

### Stop strategies (`turns/user_stop/`) — *has the user finished?*

- `SpeechTimeoutUserTurnStopStrategy` — silence timeout. Simple, dumb, fast.
- `TurnAnalyzerUserTurnStopStrategy` — semantic end-of-turn via Smart Turn v3.
  **This repo's default** (`bot.py:169-177`).
- `LLMTurnCompletionUserTurnStopStrategy` — the LLM judges completeness.
- `DeferredUserTurnStopStrategy` / `deferred()` — delay finalisation.
- `ExternalUserTurnStopStrategy`.

`FilterIncompleteUserTurnStrategies` is a preset worth knowing: the LLM must
prefix every response with `✓` (complete), `○` (incomplete short) or `◐`
(incomplete long). Only `✓` finalises the turn; the others keep it open so the
user can continue. For Hindi speakers who pause mid-sentence — a common failure
mode for silence-timeout turn detection — this is a strong option.

### Mute strategies (`turns/user_mute/`) — *ignore the user entirely, for now*

`always`, `first_speech`, `function_call`, `mute_until_first_bot_complete`.
The function-call one is genuinely useful: don't let noise interrupt the bot while
a tool call is resolving.

---

## 4. Services, transports, serializers

**Services** (`services/`) — 55+ vendors, uniform base classes in
`services/stt_service.py`, `tts_service.py`, `llm_service.py`. Swapping vendors is
a constructor change.

Notable for this product: **`services/sarvam/`** — Sarvam AI, Indic-specialised.
`stt.py` maps `Language.HI_IN → "hi-IN"` and offers `saarika:v2.5`, `saaras:v2.5`,
`saaras:v3`, with per-model capability flags including `supports_vad_params`.
For a Hindi-first product this is a materially better STT bet than a
general-purpose Western model. Sarvam also ships TTS (`tts.py`, 47.9K).

**Transports** (`transports/`) — `smallwebrtc` (this repo), `daily`, `livekit`,
`websocket`, `whatsapp`, plus avatar transports. `bot.py` already splits
`bot()` from `run_agent()` so the pipeline is transport-agnostic; that split is
correct and worth defending.

**Serializers** (`serializers/`) — `exotel`, `twilio`, `plivo`, `telnyx`,
`genesys`, `vonage`. `ExotelFrameSerializer` defaults to
`exotel_sample_rate: int = 8000` and resamples both directions against the
pipeline rate. Telephony is a serializer + websocket transport, not a rewrite —
which is what the existing Exotel plan already concluded.

**Flows** (`flows/`) — now shipped *inside* `pipecat-ai`; the standalone
`pipecat-ai-flows` package is deprecated and `__init__.py` actively warns if both
are installed. `manager.py` (36K) + `actions.py` give you node-based conversation
state machines. Relevant when this stops being a demo and needs to book
appointments or collect structured data reliably.

**Observers** (`observers/`) — `turn_tracking_observer`,
`user_bot_latency_observer`, `startup_timing_observer`. Already wired in `bot.py`.

---

## 5. What this means for the product

### 5.1 Correction to my earlier advice on the STT-confidence idea

I previously proposed writing a custom ~50-line post-hoc transcript gate. That
was wrong in a way that matters: **`MinWordsUserTurnStartStrategy` already does
it, natively, and better** — pre-turn rather than post-hoc, with the
bot-speaking/bot-silent asymmetry built in. Use the framework, not a bespoke
processor.

Second correction, in the other direction: **no Pipecat STT service exposes
`confidence` as a first-class field.** Grepping `services/deepgram/stt.py`,
`soniox/stt.py`, `gladia/stt.py` for "confidence" returns nothing. `TranscriptionFrame`
has a `result: Any | None` field holding the raw provider payload
(`frames.py:446-465`), and Deepgram populates it with `result=message`
(`deepgram/stt.py:720`), so `frame.result.channel.alternatives[0].confidence` is
reachable — but you'd be reading through an untyped escape hatch, not an API.
Plan for that if you go cascading.

### 5.2 The honest constraint on native audio

With `GeminiModalities.AUDIO`, Gemini's input transcript is pushed `UPSTREAM`
(`gemini_live/llm.py:1890-1909`) and buffered until sentence punctuation or a
`asyncio.sleep(0.5)` timeout (`:1911-1929`). Because your pipeline is
`transport.input() → user_aggregator → llm`, those upstream frames *do* reach the
turn strategies in the aggregator. So `MinWordsUserTurnStartStrategy` is wired
correctly here.

But: Gemini Live emits no `InterimTranscriptionFrame`, so `use_interim=True`
buys nothing, and the 0.5s buffer means the gate decides late. It will suppress
turns, but not instantly.

**This is the real fork in the road.** Native speech-to-speech gives you the best
Hindi prosody and code-switching, and the worst control. Cascading gives you a
transcript before the LLM ever runs, and every gate becomes exact — at the cost
of latency and that native-audio quality. Every noise-defense layer you want is
easier in a cascade. That tension won't resolve itself; pick deliberately.

### 5.3 Concrete, ordered recommendations

**Do now — cheap, native, no new dependencies:**

1. Add `start=[VADUserTurnStartStrategy(), MinWordsUserTurnStartStrategy(min_words=3)]`
   to `create_turn_strategies()`. `bot.py` currently passes only `stop=[...]`, so
   you silently inherit the permissive defaults. Make `min_words` an env var and
   sweep it in the lab like you do VAD thresholds.
2. Add a mute strategy for function calls once you have tools, so market noise
   can't interrupt a tool call mid-flight.
3. Expose `FilterUpdateSettingsFrame` / `FilterEnableFrame` so noise settings are
   tunable mid-call rather than fixed at connect.

**Evaluate next:**

4. `FilterIncompleteUserTurnStrategies` against real Hindi recordings with
   mid-sentence pauses. This may beat Smart Turn v3 for your speakers — or not.
   It's an empirical question and you already own the lab to answer it.
5. A Sarvam STT arm in the lab. Not to switch architectures — to measure how much
   an Indic-tuned model buys you on your actual noisy recordings.

**Decide deliberately:**

6. Native vs cascading (§5.2). Don't drift into it; run the lab comparison.
7. Krisp VIVA (filter + VAD + turn analyzer + IP strategy) is the only
   off-the-shelf answer to background *voices* rather than background *noise*.
   It's commercial, needs `.kef` model files and an API key. If babble in Indian
   environments is the product-defining problem, this is the serious option and
   RNNoise/DeepFilterNet are not substitutes — they suppress noise, not speech.

**Don't do:**

8. Don't write bespoke processors for anything in the strategy tables above.
   That's the mistake I nearly walked you into.

---

## 6. Quick reference — where to look

| Question | File |
|---|---|
| What frames exist? | `frames/frames.py` (127 classes) |
| What survives interruption? | `frames/frames.py:101-152` |
| Where does my noise filter run? | `transports/base_input.py:267-283` |
| How does VAD state work? | `audio/vad/vad_controller.py` |
| Should this speech start a turn? | `turns/user_start/` |
| Has the user finished? | `turns/user_stop/` |
| Turn strategy defaults | `turns/user_turn_strategies.py:27-55` |
| Telephony wire format | `serializers/exotel.py` |
| Hindi/Indic STT + TTS | `services/sarvam/` |
| Conversation state machines | `flows/manager.py` |
