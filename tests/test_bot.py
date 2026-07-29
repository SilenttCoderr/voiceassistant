import asyncio
import ast
import builtins
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from types import SimpleNamespace

import pytest

import bot


class FakeAudioFilter(bot.BaseAudioFilter):
    async def start(self, sample_rate):
        pass

    async def stop(self):
        pass

    async def process_frame(self, frame):
        pass

    async def filter(self, audio):
        return audio


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


def test_shell_environment_overrides_dotenv():
    project_root = Path.cwd()
    with tempfile.TemporaryDirectory(dir=project_root) as directory:
        isolated_root = Path(directory)
        shutil.copyfile(project_root / "bot.py", isolated_root / "bot.py")
        (isolated_root / ".env").write_text("VAD_BACKEND=gemini\n", encoding="utf-8")

        env = os.environ.copy()
        env["PYTHONPATH"] = os.pathsep.join((directory, str(project_root)))
        env["VAD_BACKEND"] = "silero"

        result = subprocess.run(
            [sys.executable, "-c", "import os, bot; print(os.getenv('VAD_BACKEND'))"],
            cwd=isolated_root,
            env=env,
            capture_output=True,
            text=True,
            encoding="utf-8",
            check=True,
        )

    assert result.stdout.strip().splitlines()[-1] == "silero"


def test_missing_gemini_configuration_does_not_construct_vad(monkeypatch):
    constructed = False

    def fake_create_vad():
        nonlocal constructed
        constructed = True

    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    monkeypatch.setattr(bot, "create_vad", fake_create_vad)

    with pytest.raises(RuntimeError, match="GOOGLE_API_KEY is required"):
        asyncio.run(bot.run_agent(None, None))

    assert constructed is False


def test_startup_failure_cleans_local_analyzer_once(monkeypatch):
    class FakeAnalyzer:
        cleanup_calls = 0

        async def cleanup(self):
            self.cleanup_calls += 1

    class FailingService:
        class Settings:
            def __init__(self, **kwargs):
                pass

        def __init__(self, **kwargs):
            raise RuntimeError("service construction failed")

    analyzer = FakeAnalyzer()
    monkeypatch.setenv("GOOGLE_API_KEY", "test-key")
    monkeypatch.setattr(bot, "create_vad", lambda: analyzer)
    monkeypatch.setattr(bot, "GeminiLiveLLMService", FailingService)

    with pytest.raises(RuntimeError, match="service construction failed"):
        asyncio.run(bot.run_agent(None, None))

    assert analyzer.cleanup_calls == 1


def test_webrtc_transport_uses_16khz_input():
    params = bot.webrtc_transport_params()

    assert params.audio_in_enabled is True
    assert params.audio_out_enabled is True
    assert params.audio_in_sample_rate == 16000


def test_browser_noise_filter_is_the_default(monkeypatch):
    monkeypatch.delenv("NOISE_FILTER", raising=False)

    assert bot.create_audio_filter() is None


def test_invalid_noise_filter_is_rejected():
    with pytest.raises(
        ValueError,
        match="NOISE_FILTER must be one of: browser, highpass, rnnoise, koala",
    ):
        bot.create_audio_filter("unknown")


def test_koala_requires_an_access_key(monkeypatch):
    monkeypatch.delenv("KOALA_ACCESS_KEY", raising=False)

    with pytest.raises(RuntimeError, match="KOALA_ACCESS_KEY is required"):
        bot.create_audio_filter("koala")


@pytest.mark.parametrize(
    ("filter_name", "dependency", "filter_module", "class_name", "expected_kwargs"),
    [
        (
            "rnnoise",
            "pyrnnoise",
            "pipecat.audio.filters.rnnoise_filter",
            "RNNoiseFilter",
            {"resampler_quality": "HQ"},
        ),
        (
            "koala",
            "pvkoala",
            "pipecat.audio.filters.koala_filter",
            "KoalaFilter",
            {"access_key": "test-key"},
        ),
    ],
)
def test_optional_noise_filters_are_constructed_lazily(
    monkeypatch, filter_name, dependency, filter_module, class_name, expected_kwargs
):
    calls = []

    class FakeFilter:
        def __init__(self, **kwargs):
            calls.append(kwargs)

    monkeypatch.setitem(sys.modules, dependency, SimpleNamespace())
    monkeypatch.setitem(sys.modules, filter_module, SimpleNamespace(**{class_name: FakeFilter}))
    monkeypatch.setenv("KOALA_ACCESS_KEY", "test-key")

    assert isinstance(bot.create_audio_filter(filter_name), FakeFilter)
    assert calls == [expected_kwargs]


@pytest.mark.parametrize(
    ("filter_name", "missing_module", "message"),
    [
        ("rnnoise", "pyrnnoise", "install it with 'uv sync --extra rnnoise'"),
        ("koala", "pvkoala", "install it with 'uv sync --extra koala'"),
    ],
)
def test_optional_filter_dependency_errors_are_specific(
    monkeypatch, filter_name, missing_module, message
):
    real_import = builtins.__import__

    def blocked_import(name, *args, **kwargs):
        if name == missing_module:
            raise ModuleNotFoundError(f"No module named '{name}'", name=name)
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", blocked_import)
    monkeypatch.setenv("KOALA_ACCESS_KEY", "test-key")

    with pytest.raises(RuntimeError, match=message):
        bot.create_audio_filter(filter_name)


@pytest.mark.parametrize(
    ("filter_name", "selected_module"),
    [("rnnoise", "pyrnnoise"), ("koala", "pvkoala")],
)
def test_optional_filter_imports_preserve_transitive_module_errors(
    monkeypatch, filter_name, selected_module
):
    real_import = builtins.__import__

    def blocked_import(name, *args, **kwargs):
        if name == selected_module:
            raise ModuleNotFoundError(
                "No module named 'transitive_dependency'",
                name="transitive_dependency",
            )
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", blocked_import)
    monkeypatch.setenv("KOALA_ACCESS_KEY", "test-key")

    with pytest.raises(ModuleNotFoundError) as exc_info:
        bot.create_audio_filter(filter_name)
    assert exc_info.value.name == "transitive_dependency"


def test_webrtc_transport_accepts_audio_filter():
    audio_filter = FakeAudioFilter()

    assert bot.webrtc_transport_params(audio_filter).audio_in_filter is audio_filter


def test_bot_passes_noise_filter_to_webrtc_without_changing_vad(monkeypatch):
    audio_filter = FakeAudioFilter()
    captured = {}

    async def fake_create_transport(runner_args, factories):
        captured["params"] = factories["webrtc"]()
        return object()

    async def fake_run_agent(transport, runner_args):
        captured["transport"] = transport

    monkeypatch.setenv("VAD_BACKEND", "ten")
    monkeypatch.setattr(bot, "create_audio_filter", lambda: audio_filter)
    monkeypatch.setattr(bot, "create_transport", fake_create_transport)
    monkeypatch.setattr(bot, "run_agent", fake_run_agent)

    asyncio.run(bot.bot(SimpleNamespace()))

    assert captured["params"].audio_in_filter is audio_filter
    assert os.environ["VAD_BACKEND"] == "ten"


def test_default_observability_notice_is_honest():
    turn_tracking, notice = bot.observability_configuration(None)
    notice = notice.lower()

    assert turn_tracking is False
    assert "gemini server vad" in notice
    assert "user-speaking" in notice
    assert "user-to-bot latency" in notice
    assert "unavailable" in notice


def test_local_vad_enables_frame_based_observability():
    turn_tracking, notice = bot.observability_configuration(object())
    notice = notice.lower()

    assert turn_tracking is True
    assert "local vad" in notice
    assert "user-speaking" in notice
    assert "turn tracking" in notice
    assert "interruption" in notice
    assert "user-to-bot latency" in notice


def test_readme_uses_plain_uv_commands():
    readme = Path("README.md").read_text(encoding="utf-8")

    assert "rtk " not in readme
    assert "uv sync" in readme
    assert "uv run bot.py -t webrtc" in readme
    assert "uv run pytest -v" in readme


def test_configure_utf8_streams_reconfigures_windows_console(monkeypatch):
    class FakeStream:
        encoding = "cp1252"

        def reconfigure(self, *, encoding):
            self.encoding = encoding

    stdout = FakeStream()
    stderr = FakeStream()
    monkeypatch.setattr(sys, "stdout", stdout)
    monkeypatch.setattr(sys, "stderr", stderr)

    bot._configure_utf8_streams()

    assert stdout.encoding == "utf-8"
    assert stderr.encoding == "utf-8"


def test_direct_run_configures_utf8_before_pipecat_imports():
    source = Path("bot.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    pipecat_import_line = min(
        node.lineno
        for node in ast.walk(tree)
        if (
            isinstance(node, ast.ImportFrom)
            and node.module
            and node.module.startswith("pipecat")
        )
        or (
            isinstance(node, ast.Import)
            and any(alias.name.startswith("pipecat") for alias in node.names)
        )
    )
    setup_call_line = min(
        node.lineno
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "_configure_utf8_streams"
    )

    assert setup_call_line < pipecat_import_line


def test_readme_metrics_match_current_logs():
    readme = Path("README.md").read_text(encoding="utf-8")

    assert "Pipeline and service metric frames are enabled" in readme
    assert "Current logs expose first-speech latency" in readme
    assert "user-to-bot latency and turn events are exposed only where local VAD frames exist" in readme
    assert "Pipecat service metrics" not in readme


def test_vad_mode_switches_gemini_server_vad():
    server_vad, user_params = bot.vad_configuration(None)
    assert server_vad.disabled is False
    assert user_params is None

    analyzer = object()
    server_vad, user_params = bot.vad_configuration(analyzer)
    assert server_vad.disabled is True
    assert user_params.vad_analyzer is analyzer


class BaseAudioFilterStub:
    """Minimal stand-in for a Pipecat audio filter."""

    async def start(self, sample_rate):
        return None

    async def stop(self):
        return None

    async def process_frame(self, frame):
        return None

    async def filter(self, audio):
        return audio


def test_noise_mix_wraps_the_whole_chain_once(monkeypatch):
    pytest.importorskip("pyrnnoise")
    monkeypatch.setenv("NOISE_MIX", "0.8")

    mixed = bot.create_audio_filter("highpass+rnnoise")

    assert type(mixed).__name__ == "MixedAudioFilter"
    assert mixed._wet == 0.8
    # Wrapped once, around the chain, not once per stage.
    assert type(mixed._inner).__name__ == "ChainedAudioFilter"
    assert all(type(f).__name__ != "MixedAudioFilter" for f in mixed._inner._filters)


def test_noise_mix_of_one_adds_no_wrapper(monkeypatch):
    monkeypatch.setenv("NOISE_MIX", "1.0")

    assert type(bot.create_audio_filter("highpass")).__name__ == "HighPassFilter"


def test_noise_mix_is_validated(monkeypatch):
    monkeypatch.setenv("NOISE_MIX", "1.5")

    with pytest.raises(RuntimeError, match="NOISE_MIX must be between 0 and 1"):
        bot.create_audio_filter("highpass")


def test_rnnoise_quality_is_validated(monkeypatch):
    pytest.importorskip("pyrnnoise")
    monkeypatch.setenv("RNNOISE_QUALITY", "SUPERB")

    with pytest.raises(RuntimeError, match="RNNOISE_QUALITY must be one of"):
        bot.create_audio_filter("rnnoise")


def test_rnnoise_defaults_above_pipecats_lowest_quality(monkeypatch):
    pytest.importorskip("pyrnnoise")
    monkeypatch.delenv("RNNOISE_QUALITY", raising=False)

    assert bot.create_audio_filter("rnnoise")._resampler_quality == "HQ"


def test_mixed_filter_blends_towards_the_original():
    import asyncio

    import numpy as np

    from noise import MixedAudioFilter

    class Silencer(BaseAudioFilterStub):
        async def filter(self, audio):
            return b"\x00\x00" * (len(audio) // 2)

    original = (np.full(320, 1000, dtype="<i2")).tobytes()

    async def run(wet):
        mixed = MixedAudioFilter(Silencer(), wet=wet)
        await mixed.start(16000)
        out = await mixed.filter(original)
        await mixed.stop()
        return np.frombuffer(out, "<i2")[0]

    # Inner filter outputs silence, so the result is purely the dry share.
    assert asyncio.run(run(1.0)) == 0
    assert asyncio.run(run(0.75)) == pytest.approx(250, abs=2)
    assert asyncio.run(run(0.0)) == pytest.approx(1000, abs=2)


def test_noise_filters_can_be_chained():
    pytest.importorskip("pyrnnoise")

    chained = bot.create_audio_filter("highpass+rnnoise")

    assert type(chained).__name__ == "ChainedAudioFilter"
    assert [type(f).__name__ for f in chained._filters] == ["HighPassFilter", "RNNoiseFilter"]


def test_browser_drops_out_of_a_chain():
    single = bot.create_audio_filter("browser+highpass")

    assert type(single).__name__ == "HighPassFilter"
    assert bot.create_audio_filter("browser+browser") is None


def test_chained_filter_runs_every_stage():
    import asyncio

    import numpy as np

    chained = bot.create_audio_filter("highpass+highpass")

    async def run():
        await chained.start(16000)
        t = np.arange(16000) / 16000
        signal = 0.35 * np.sin(2 * np.pi * 200 * t) + 0.25 * np.sin(2 * np.pi * 45 * t)
        pcm = (np.clip(signal, -1, 1) * 32767).astype("<i2").tobytes()
        out = b"".join([await chained.filter(pcm[i : i + 640]) for i in range(0, len(pcm), 640)])
        await chained.stop()
        return pcm, out

    pcm, out = asyncio.run(run())

    def low_band(raw):
        samples = np.frombuffer(raw, "<i2").astype(float)
        spectrum = np.abs(np.fft.rfft(samples))
        freqs = np.fft.rfftfreq(samples.size, 1 / 16000)
        return spectrum[(freqs >= 20) & (freqs < 100)].mean()

    # Two high-passes in series attenuate more than one would.
    assert low_band(out) < low_band(pcm) * 0.2
    assert len(out) == len(pcm)


def test_turn_detection_defaults_to_smart_turn(monkeypatch):
    monkeypatch.delenv("TURN_DETECTION", raising=False)

    strategies = bot.create_turn_strategies()

    assert [type(s).__name__ for s in strategies.stop] == ["TurnAnalyzerUserTurnStopStrategy"]


def test_plain_vad_turn_detection_skips_the_turn_model():
    strategies = bot.create_turn_strategies("vad")

    assert [type(s).__name__ for s in strategies.stop] == ["SpeechTimeoutUserTurnStopStrategy"]


def test_invalid_turn_detection_is_rejected():
    with pytest.raises(ValueError, match="TURN_DETECTION must be one of: smart, vad"):
        bot.create_turn_strategies("unknown")


def test_smart_turn_params_come_from_the_environment(monkeypatch):
    monkeypatch.setenv("SMART_TURN_STOP_SECS", "1.5")
    monkeypatch.setenv("SMART_TURN_MAX_DURATION_SECS", "12")

    analyzer = bot.create_turn_strategies("smart").stop[0]._turn_analyzer

    assert analyzer._params.stop_secs == 1.5
    assert analyzer._params.max_duration_secs == 12


def test_smart_turn_settings_are_validated_before_construction(monkeypatch):
    monkeypatch.setenv("SMART_TURN_STOP_SECS", "-1")

    with pytest.raises(RuntimeError, match="SMART_TURN_STOP_SECS must be at least 0"):
        bot.create_turn_strategies("smart")
