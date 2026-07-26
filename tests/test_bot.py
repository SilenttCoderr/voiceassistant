from pathlib import Path
import sys

import pytest

import bot


def test_google_api_key_is_required(monkeypatch):
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)

    with pytest.raises(RuntimeError, match="GOOGLE_API_KEY is required"):
        bot.get_google_api_key()


def test_google_api_key_is_returned_without_logging_or_transforming(monkeypatch):
    monkeypatch.setenv("GOOGLE_API_KEY", "test-key")

    assert bot.get_google_api_key() == "test-key"


def test_prompt_is_hindi_first_and_speech_friendly():
    prompt = bot.SYSTEM_INSTRUCTION.lower()

    assert "hindi" in prompt
    assert "english" in prompt
    assert "markdown" in prompt
    assert "brief" in prompt


def test_gemini_defaults(monkeypatch):
    monkeypatch.delenv("GEMINI_MODEL", raising=False)
    monkeypatch.delenv("GEMINI_VOICE", raising=False)

    assert bot.get_gemini_model() == "models/gemini-2.5-flash-native-audio-preview-12-2025"
    assert bot.get_gemini_voice() == "Charon"


def test_webrtc_transport_uses_16khz_input():
    params = bot.webrtc_transport_params()

    assert params.audio_in_enabled is True
    assert params.audio_out_enabled is True
    assert params.audio_in_sample_rate == 16000


def test_default_observability_notice_is_honest():
    notice = bot.LOCAL_VAD_OBSERVABILITY_NOTICE.lower()

    assert "local user-speaking" in notice
    assert "user-to-bot latency" in notice
    assert "local vad" in notice


def test_readme_uses_plain_uv_commands():
    readme = Path("README.md").read_text(encoding="utf-8")

    assert "rtk " not in readme
    assert "uv sync" in readme
    assert "uv run bot.py -t webrtc" in readme
    assert "uv run pytest -v" in readme


def test_configure_stdout_utf8_reconfigures_windows_console(monkeypatch):
    class FakeStdout:
        encoding = "cp1252"

        def reconfigure(self, *, encoding):
            self.encoding = encoding

    stdout = FakeStdout()
    monkeypatch.setattr(sys, "stdout", stdout)

    bot._configure_stdout_utf8()

    assert stdout.encoding == "utf-8"
