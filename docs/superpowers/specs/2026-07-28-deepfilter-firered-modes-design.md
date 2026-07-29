# DeepFilterNet and FireRed Modes Design

**Status:** Approved

## Goal

Add DeepFilterNet as an optional noise filter for the Windows live agent and offline VAD lab. Support FireRed's built-in modes without changing the current custom-threshold default.

## Constraints

- Keep the agent's internal and transport audio rate at 16 kHz.
- DeepFilterNet processes full-band audio at 48 kHz, so its adapter owns resampling.
- DeepFilterNet 0.5.6 has Windows wheels through Python 3.11 and requires NumPy below 2.
- Optional imports remain lazy and errors name the exact `uv sync --extra ...` command.
- The live pipeline and lab must use the same filter implementation.
- Existing browser, RNNoise, and Koala behavior remains unchanged.

## Runtime Compatibility

The project development runtime moves from Python 3.12 to Python 3.11. `pyproject.toml` will use `requires-python = ">=3.11,<3.12"` and define a `deepfilter` optional extra containing DeepFilterNet 0.5.6 and its required compatible runtime dependencies.

DeepFilterNet is not installed by the base `uv sync`. Users select it with:

```powershell
uv sync --extra deepfilter
```

The default pretrained DeepFilterNet3 model is used. Custom model selection, GPU configuration, and attenuation controls are outside this change.

## DeepFilterNet Filter

Add a project-owned `DeepFilterNetFilter` implementing Pipecat's `BaseAudioFilter` lifecycle:

- `start(sample_rate)` validates 16 kHz input, loads the model once, creates streaming analysis/synthesis state, and creates 16-to-48 kHz and 48-to-16 kHz stream resamplers.
- `filter(audio)` buffers PCM16 input, processes complete DeepFilterNet frames while preserving model and STFT state across calls, and returns PCM16 at 16 kHz.
- `stop()` releases model, state, resamplers, and buffered audio.
- `process_frame()` supports Pipecat's standard filter enable/disable control.

The adapter must not call the public clip-oriented `enhance()` function for each transport chunk because that resets model state and can create discontinuities. It will use the same inference stages with state retained across calls.

DeepFilterNet's initial algorithmic delay and final incomplete frame are acceptable. The lab already permits a short filtered tail difference, and the live filter must keep buffering bounded to the model's frame requirements.

## Filter Selection

`create_audio_filter()` accepts `deepfilter` in addition to `browser`, `rnnoise`, and `koala`. Imports occur only inside the DeepFilterNet branch.

Missing DeepFilterNet dependencies raise:

```text
DeepFilterNet selected; install it with 'uv sync --extra deepfilter'
```

Model download, model load, or unsupported-runtime failures are raised instead of silently returning unfiltered audio.

The lab adds `deepfilter` to each slot's denoise selector and continues calling `create_audio_filter()`, ensuring recorded comparisons exercise the production adapter.

## FireRed Configuration

FireRed supports both preset mode and custom threshold configuration:

- `FIREREDVAD_MODE` defaults to blank.
- Blank mode uses `FIREREDVAD_SPEECH_THRESHOLD`, default `0.6`.
- Non-blank mode must be an integer from `0` through `3` and is passed to FireRed as `mode`.
- A preset mode does not also pass `speech_threshold`; FireRed owns all values in that preset.
- Pipecat continues to own `VAD_CONFIDENCE`, `VAD_START_SECS`, `VAD_STOP_SECS`, and `VAD_MIN_VOLUME`.

This preserves current behavior while making mode 3 available as an explicit aggressive option.

The lab adds a FireRed mode selector with `custom`, `0`, `1`, `2`, and `3`. The threshold input is visible only for `custom`. Lab environment overrides always set `FIREREDVAD_MODE`, including an empty value for custom mode, so a value from `.env` cannot leak into a comparison slot.

## Error Handling

- Invalid `FIREREDVAD_MODE` values fail before constructing FireRed and name the variable.
- The custom threshold is validated only when mode is blank.
- Optional dependency handling distinguishes the expected missing package from missing transitive dependencies.
- A failing lab filter or backend remains isolated to its slot.
- DeepFilterNet startup failure aborts live startup rather than reporting a filter that is not active.

## Tests

The network-free unit suite will cover:

- Lazy DeepFilterNet construction and its exact install error.
- Preservation of unexpected transitive import errors.
- DeepFilterNet lifecycle, buffering, state reuse, output format, and filter enable/disable behavior with faked model modules.
- `create_audio_filter("deepfilter")` and lab slot selection using the same adapter.
- FireRed custom threshold, modes 0 through 3, invalid modes, and no simultaneous mode/threshold arguments.
- Lab override restoration and prevention of `.env` mode leakage.
- `.env.example` and README defaults.

A manual Windows smoke check will install the `deepfilter` extra, allow the default model to load, process a fixed WAV through the lab, and confirm processing is faster than the clip duration before live use.

## Non-Goals

- A separate DeepFilterNet worker or sidecar.
- Python 3.12 support through locally compiled Rust bindings.
- Linux LADSPA or PipeWire integration.
- GPU selection, custom model paths, attenuation limits, or a DeepFilterNet-specific lab tuning panel.
- Changing the default noise filter from `browser`.
