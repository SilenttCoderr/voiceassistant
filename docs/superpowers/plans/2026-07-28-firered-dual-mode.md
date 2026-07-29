# FireRed Dual-Mode Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: use `subagent-driven-development` (recommended) or `executing-plans` to implement this plan task-by-task.

**Goal:** Preserve FireRed's custom threshold `0.6` default while adding opt-in preset modes `0` through `3` to the live agent and VAD lab.

**Architecture:** `create_vad()` validates one mutually exclusive FireRed configuration and passes either `speech_threshold` or `mode` to the community analyzer. The lab always overrides `FIREREDVAD_MODE`, including an empty value for custom mode, so `.env` cannot leak into a slot.

**Tech Stack:** Python 3.11, Pipecat, FireRedVAD, pytest, vanilla JavaScript.

## Global Constraints

- Keep custom threshold `0.6` as the default.
- Accept only blank or integer `0` through `3` for `FIREREDVAD_MODE`.
- Never pass `mode` and `speech_threshold` together.
- Keep optional imports lazy and preserve missing transitive dependency errors.
- Do not edit `.env`.
- Commit commands are checkpoints only; execute them only after explicit user approval.

---

### Task 1: FireRed Factory Configuration

**Files:**
- Modify: `tests/test_vad.py:108-213`
- Modify: `vad.py:126-166`

**Interfaces:**
- Consumes: `FIREREDVAD_MODE`, `FIREREDVAD_SPEECH_THRESHOLD`.
- Produces: `create_vad("firered")` passing exactly one of `mode: int` or `speech_threshold: float`.

- [ ] **Step 1: Add failing preset and validation tests**

Add a local helper and tests near the existing FireRed tests:

```python
def install_fake_firered(monkeypatch, calls):
    class FakeFireVadAnalyzer:
        def __init__(self, **kwargs):
            calls.append(kwargs)

    constants = ModuleType("fireredvad.core.constants")
    constants.FRAME_LENGTH_SAMPLE = 400
    monkeypatch.setitem(
        sys.modules,
        "pipecat_firered_vad",
        SimpleNamespace(FireVadAnalyzer=FakeFireVadAnalyzer),
    )
    monkeypatch.setitem(sys.modules, "fireredvad", ModuleType("fireredvad"))
    monkeypatch.setitem(sys.modules, "fireredvad.core", ModuleType("fireredvad.core"))
    monkeypatch.setitem(sys.modules, "fireredvad.core.constants", constants)
    monkeypatch.setenv("FIREREDVAD_MODEL_DIR", "model-dir")


@pytest.mark.parametrize("mode", [0, 1, 2, 3])
def test_firered_mode_is_passed_without_custom_threshold(monkeypatch, mode):
    calls = []
    install_fake_firered(monkeypatch, calls)
    monkeypatch.setenv("FIREREDVAD_MODE", str(mode))
    monkeypatch.setenv("FIREREDVAD_SPEECH_THRESHOLD", "not-used")

    vad.create_vad("firered")

    assert calls[0]["mode"] == mode
    assert "speech_threshold" not in calls[0]


def test_blank_firered_mode_uses_custom_threshold(monkeypatch):
    calls = []
    install_fake_firered(monkeypatch, calls)
    monkeypatch.setenv("FIREREDVAD_MODE", "")
    monkeypatch.setenv("FIREREDVAD_SPEECH_THRESHOLD", "0.7")

    vad.create_vad("firered")

    assert calls[0]["speech_threshold"] == 0.7
    assert "mode" not in calls[0]


@pytest.mark.parametrize("mode", ["-1", "4", "1.5", "loud"])
def test_invalid_firered_mode_is_rejected_before_construction(monkeypatch, mode):
    calls = []
    install_fake_firered(monkeypatch, calls)
    monkeypatch.setenv("FIREREDVAD_MODE", mode)

    with pytest.raises(RuntimeError, match="FIREREDVAD_MODE must be blank or an integer from 0 to 3"):
        vad.create_vad("firered")

    assert calls == []
```

Update `test_firered_speech_threshold_is_validated_before_construction` to explicitly blank the mode:

```python
monkeypatch.setenv("FIREREDVAD_MODE", "")
```

Also add the same line to `test_firered_adapter_uses_upstream_frame_length_and_confidence`, which relies on custom-threshold behavior. This prevents the developer's `.env` from selecting a preset during either test.

- [ ] **Step 2: Run focused tests and verify RED**

Run:

```powershell
rtk uv run pytest tests/test_vad.py -k "firered_mode or blank_firered or speech_threshold" -v
```

Expected: preset tests fail because `mode` is ignored and the invalid threshold is still parsed.

- [ ] **Step 3: Implement mutually exclusive FireRed arguments**

Replace the unconditional threshold parsing and constructor arguments in `create_vad()` with:

```python
        mode_value = os.getenv("FIREREDVAD_MODE", "").strip()
        firered_args = {}
        if mode_value:
            try:
                mode = int(mode_value)
            except ValueError as exc:
                raise RuntimeError(
                    "FIREREDVAD_MODE must be blank or an integer from 0 to 3"
                ) from exc
            if mode not in range(4):
                raise RuntimeError(
                    "FIREREDVAD_MODE must be blank or an integer from 0 to 3"
                )
            firered_args["mode"] = mode
        else:
            firered_args["speech_threshold"] = _float_env(
                "FIREREDVAD_SPEECH_THRESHOLD", 0.6, minimum=0, maximum=1
            )
```

Construct the analyzer with:

```python
        return CompatibleFireVadAnalyzer(
            model_dir=model_dir,
            sample_rate=16000,
            params=params,
            use_gpu=use_gpu_value == "1",
            **firered_args,
        )
```

- [ ] **Step 4: Run FireRed tests and verify GREEN**

Run:

```powershell
rtk uv run pytest tests/test_vad.py -k firered -v
```

Expected: all FireRed tests pass.

- [ ] **Step 5: Checkpoint commit after approval**

```powershell
rtk git add vad.py tests/test_vad.py
rtk git commit -m "feat: support FireRed preset modes"
```

---

### Task 2: Lab FireRed Mode Selection

**Files:**
- Modify: `tests/test_lab.py:78-92`
- Modify: `lab.py:72-84`
- Modify: `lab.html:101-169`

**Interfaces:**
- Consumes: lab slot field `firered_mode`, values `custom`, `0`, `1`, `2`, `3`.
- Produces: explicit `FIREREDVAD_MODE` override and threshold only for custom mode.

- [ ] **Step 1: Add failing lab override tests**

Add:

```python
def test_firered_lab_custom_mode_clears_environment_mode():
    overrides = lab._env_overrides(
        {
            "backend": "firered",
            "confidence": 0.5,
            "start_secs": 0.2,
            "stop_secs": 0.45,
            "min_volume": 0.35,
            "firered_mode": "custom",
            "firered_threshold": 0.7,
        }
    )

    assert overrides["FIREREDVAD_MODE"] == ""
    assert overrides["FIREREDVAD_SPEECH_THRESHOLD"] == "0.7"


def test_firered_lab_preset_omits_custom_threshold():
    overrides = lab._env_overrides(
        {
            "backend": "firered",
            "confidence": 0.5,
            "start_secs": 0.2,
            "stop_secs": 0.45,
            "min_volume": 0.35,
            "firered_mode": "3",
            "firered_threshold": 0.7,
        }
    )

    assert overrides["FIREREDVAD_MODE"] == "3"
    assert "FIREREDVAD_SPEECH_THRESHOLD" not in overrides
```

- [ ] **Step 2: Run lab tests and verify RED**

Run:

```powershell
rtk uv run pytest tests/test_lab.py -k firered_lab -v
```

Expected: both tests fail because `_env_overrides()` does not emit `FIREREDVAD_MODE`.

- [ ] **Step 3: Implement explicit lab overrides**

Replace the FireRed threshold branch in `_env_overrides()` with:

```python
    if config.get("backend") == "firered":
        mode = str(config.get("firered_mode", "custom"))
        overrides["FIREREDVAD_MODE"] = "" if mode == "custom" else mode
        if mode == "custom" and config.get("firered_threshold") is not None:
            overrides["FIREREDVAD_SPEECH_THRESHOLD"] = str(
                config["firered_threshold"]
            )
```

- [ ] **Step 4: Add the lab selector**

Add `firered_mode` to each slot:

```javascript
firered_mode: from ? from.firered_mode : "custom",
```

In `renderSlots()`, replace the single FireRed threshold selection with:

```javascript
const fireRed = slot.backend === "firered";
const threshold = slot.backend === "ten" ? "ten_threshold"
                : fireRed && slot.firered_mode === "custom" ? "firered_threshold" : null;
```

Add this control before the numeric grid:

```javascript
${fireRed ? `<label style="margin-top:6px"><span style="width:auto;flex:1;font-size:11px">FireRed mode</span>
  <select data-id="${slot.id}" data-key="firered_mode">
    ${["custom", "0", "1", "2", "3"].map((m) =>
      `<option${m === slot.firered_mode ? " selected" : ""}>${m}</option>`).join("")}
  </select></label>` : ""}
```

Update the rerender condition:

```javascript
if (key === "backend" || key === "smart_turn" || key === "firered_mode") renderSlots();
```

Update `rowLabel()` so FireRed reports either `mode=N` or `thr=N`:

```javascript
const extra = c.backend === "ten" ? " thr=" + c.ten_threshold
            : c.backend === "firered" && c.firered_mode !== "custom"
              ? " mode=" + c.firered_mode
            : c.backend === "firered" ? " thr=" + c.firered_threshold : "";
```

- [ ] **Step 5: Run lab tests and verify GREEN**

Run:

```powershell
rtk uv run pytest tests/test_lab.py -v
```

Expected: all lab tests pass.

- [ ] **Step 6: Checkpoint commit after approval**

```powershell
rtk git add lab.py lab.html tests/test_lab.py
rtk git commit -m "feat: compare FireRed modes in VAD lab"
```

---

### Task 3: Defaults, Documentation, and Verification

**Files:**
- Modify: `.env.example:13-15`
- Modify: `README.md:57-91`
- Modify: `tests/test_vad.py:206-213`

**Interfaces:**
- Produces: documented blank mode default and copyable mode examples.

- [ ] **Step 1: Extend the asserted example defaults**

Add to `test_env_example_uses_balanced_firered_defaults()`:

```python
assert values["FIREREDVAD_MODE"] == ""
```

- [ ] **Step 2: Run the default test and verify RED**

Run:

```powershell
rtk uv run pytest tests/test_vad.py::test_env_example_uses_balanced_firered_defaults -v
```

Expected: FAIL because `FIREREDVAD_MODE` is absent.

- [ ] **Step 3: Add the blank default**

Insert before `FIREREDVAD_SPEECH_THRESHOLD`:

```dotenv
FIREREDVAD_MODE=
```

- [ ] **Step 4: Document custom and preset behavior**

Update the FireRed section to state:

```markdown
Leave `FIREREDVAD_MODE` blank to use `FIREREDVAD_SPEECH_THRESHOLD=0.6`.
Set `FIREREDVAD_MODE` to `0`, `1`, `2`, or `3` to use FireRed's complete preset;
mode `3` is the most aggressive. A preset ignores the custom threshold.
```

Update the VAD Lab section to explain that each FireRed slot chooses `custom` or one preset.

- [ ] **Step 5: Run complete verification**

Run:

```powershell
rtk uv run pytest -v
rtk uv run python -m py_compile bot.py vad.py lab.py
rtk git diff --check
```

Expected: 75 or more tests pass, compile emits no output, and diff check emits no output.

- [ ] **Step 6: Checkpoint commit after approval**

```powershell
rtk git add .env.example README.md tests/test_vad.py
rtk git commit -m "docs: describe FireRed custom and preset modes"
```
