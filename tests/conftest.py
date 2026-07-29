import pytest

# Importing `bot` runs `load_dotenv(override=False)` at module scope, which leaks the
# developer's real `.env` into `os.environ` for the whole session. Any test that reads one
# of these settings would then pass or fail depending on local tuning. Clear them once, so
# tests see documented defaults and opt in explicitly with `monkeypatch.setenv`.
CONFIGURED_BY_DOTENV = (
    "GEMINI_MODEL",
    "GEMINI_VOICE",
    "VAD_BACKEND",
    "VAD_CONFIDENCE",
    "VAD_START_SECS",
    "VAD_STOP_SECS",
    "VAD_MIN_VOLUME",
    "TURN_DETECTION",
    "SMART_TURN_STOP_SECS",
    "SMART_TURN_MAX_DURATION_SECS",
    "TEN_VAD_THRESHOLD",
    "FIREREDVAD_MODEL_DIR",
    "FIREREDVAD_SPEECH_THRESHOLD",
    "FIREREDVAD_USE_GPU",
    "PICOVOICE_ACCESS_KEY",
    "NOISE_FILTER",
    "HIGHPASS_HZ",
    "NOISE_MIX",
    "RNNOISE_QUALITY",
    "KOALA_ACCESS_KEY",
)


@pytest.fixture(autouse=True)
def isolate_dotenv_settings(monkeypatch):
    for name in CONFIGURED_BY_DOTENV:
        monkeypatch.delenv(name, raising=False)
