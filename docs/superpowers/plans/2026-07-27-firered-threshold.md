# FireRed Speech Threshold Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: superpowers:executing-plans. Implement this plan task-by-task.

**Goal:** Expose FireRed's model-level speech threshold through a validated environment setting defaulting to `0.6`.

**Architecture:** Reuse the existing `_float_env` validation and pass the value only to `CompatibleFireVadAnalyzer`. Silero and other VAD backends continue using the shared Pipecat VAD settings.

**Tech Stack:** Python 3.12, pytest, Pipecat 1.6, FireRedVAD.

## Global Constraints

- Preserve unrelated user changes in `.env.example`.
- Do not modify the ignored `.env`.
- Validate the threshold in the inclusive range `0.0` to `1.0`.
- Keep Silero configuration unchanged.

### Task 1: Add the FireRed Threshold Setting

**Files:**
- Modify: `tests/test_vad.py`
- Modify: `vad.py`
- Modify: `.env.example`
- Modify: `README.md`

**Interfaces:**
- Consumes: `FIREREDVAD_SPEECH_THRESHOLD`.
- Produces: `CompatibleFireVadAnalyzer(..., speech_threshold=<validated value>)`.

- [ ] **Step 1: Write failing tests**

Add tests proving the default `0.6` is forwarded and values outside `0.0..1.0` fail before FireRed construction.

- [ ] **Step 2: Run focused tests and verify RED**

Run: `rtk uv run pytest tests/test_vad.py -k firered_speech_threshold -v`

Expected: failures because the setting is not read or forwarded.

- [ ] **Step 3: Implement the minimal setting**

In the FireRed branch of `create_vad`, add:

```python
speech_threshold = _float_env(
    "FIREREDVAD_SPEECH_THRESHOLD", 0.6, minimum=0, maximum=1
)
```

Pass `speech_threshold=speech_threshold` to `CompatibleFireVadAnalyzer`.

- [ ] **Step 4: Document the setting**

Add `FIREREDVAD_SPEECH_THRESHOLD=0.6` to `.env.example` without changing other values. Document that Silero uses `VAD_CONFIDENCE`, `VAD_START_SECS`, `VAD_STOP_SECS`, and `VAD_MIN_VOLUME` instead.

- [ ] **Step 5: Verify GREEN**

Run:

```powershell
rtk uv run pytest tests/test_vad.py -v
rtk uv run pytest -q
rtk uv run python -m compileall -q vad.py tests
rtk git diff --check
```

Expected: all tests pass and checks exit successfully.
