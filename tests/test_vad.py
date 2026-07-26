import builtins
import asyncio
import sys
from types import ModuleType, SimpleNamespace

import pytest

import vad


def test_default_backend_is_gemini(monkeypatch):
    monkeypatch.delenv("VAD_BACKEND", raising=False)

    assert vad.create_vad() is None


def test_backend_name_is_case_insensitive():
    assert vad.create_vad(" GEMINI ") is None


def test_invalid_backend_lists_supported_values():
    with pytest.raises(
        ValueError,
        match="VAD_BACKEND must be one of: gemini, silero, ten, firered, cobra",
    ):
        vad.create_vad("unknown")


@pytest.mark.parametrize(
    ("backend", "missing_module", "message"),
    [
        ("ten", "ten_vad", "install it with 'uv sync --extra ten'"),
        (
            "firered",
            "pipecat_firered_vad",
            "install it with 'uv sync --extra firered'",
        ),
        ("cobra", "pvcobra", "install the Cobra optional dependencies"),
    ],
)
def test_optional_dependency_errors_are_backend_specific(
    monkeypatch, backend, missing_module, message
):
    real_import = builtins.__import__

    def blocked_import(name, *args, **kwargs):
        if name == missing_module:
            raise ModuleNotFoundError(f"No module named '{name}'", name=name)
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", blocked_import)
    if backend == "cobra":
        monkeypatch.setenv("PICOVOICE_ACCESS_KEY", "test-key")
    if backend == "firered":
        monkeypatch.setenv("FIREREDVAD_MODEL_DIR", "model-dir")

    with pytest.raises(RuntimeError, match=message):
        vad.create_vad(backend)


def test_common_vad_params_reject_invalid_numbers(monkeypatch):
    monkeypatch.setenv("VAD_CONFIDENCE", "not-a-number")

    with pytest.raises(RuntimeError, match="VAD_CONFIDENCE must be a number"):
        vad.common_vad_params()


@pytest.mark.parametrize(
    ("name", "value", "message"),
    [
        ("VAD_CONFIDENCE", "nan", "VAD_CONFIDENCE must be a finite number"),
        ("VAD_CONFIDENCE", "-0.1", "VAD_CONFIDENCE must be between 0 and 1"),
        ("VAD_MIN_VOLUME", "1.1", "VAD_MIN_VOLUME must be between 0 and 1"),
        ("VAD_START_SECS", "-0.1", "VAD_START_SECS must be at least 0"),
        ("VAD_STOP_SECS", "inf", "VAD_STOP_SECS must be a finite number"),
    ],
)
def test_common_vad_params_reject_non_finite_and_out_of_range_values(
    monkeypatch, name, value, message
):
    monkeypatch.setenv(name, value)

    with pytest.raises(RuntimeError, match=message):
        vad.common_vad_params()


@pytest.mark.parametrize("value", ["nan", "-0.1", "1.1"])
def test_ten_threshold_is_validated_before_construction(monkeypatch, value):
    constructed = False

    def fake_constructor(**kwargs):
        nonlocal constructed
        constructed = True

    monkeypatch.setitem(
        sys.modules,
        "ten_vad",
        SimpleNamespace(TenVad=fake_constructor),
    )
    monkeypatch.setenv("TEN_VAD_THRESHOLD", value)

    with pytest.raises(RuntimeError, match="TEN_VAD_THRESHOLD"):
        vad.create_vad("ten")
    assert constructed is False


def test_firered_gpu_flag_is_validated_before_construction(monkeypatch):
    constructed = False

    def fake_constructor(**kwargs):
        nonlocal constructed
        constructed = True

    monkeypatch.setitem(
        sys.modules,
        "pipecat_firered_vad",
        SimpleNamespace(FireVadAnalyzer=fake_constructor),
    )
    monkeypatch.setenv("FIREREDVAD_MODEL_DIR", "model-dir")
    monkeypatch.setenv("FIREREDVAD_USE_GPU", "true")

    with pytest.raises(
        RuntimeError, match="FIREREDVAD_USE_GPU must be exactly 0 or 1"
    ):
        vad.create_vad("firered")
    assert constructed is False


def test_firered_adapter_uses_upstream_frame_length_and_confidence(monkeypatch):
    class FakeEngine:
        def __init__(self):
            self.frames = []
            self.results = iter(
                (
                    SimpleNamespace(smoothed_prob=1.25, raw_prob=0.5),
                    SimpleNamespace(raw_prob=-0.25),
                )
            )

        def detect_frame(self, frame):
            self.frames.append(frame)
            return next(self.results)

    class CommunityFireVadAnalyzer:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self._vad = FakeEngine()
            self.reset_calls = 0

        def num_frames_required(self):
            return 160

        def reset(self):
            self.reset_calls += 1

    community = ModuleType("pipecat_firered_vad")
    community.FireVadAnalyzer = CommunityFireVadAnalyzer
    constants = ModuleType("fireredvad.core.constants")
    constants.FRAME_LENGTH_SAMPLE = 400
    monkeypatch.setitem(sys.modules, "pipecat_firered_vad", community)
    monkeypatch.setitem(sys.modules, "fireredvad", ModuleType("fireredvad"))
    monkeypatch.setitem(sys.modules, "fireredvad.core", ModuleType("fireredvad.core"))
    monkeypatch.setitem(sys.modules, "fireredvad.core.constants", constants)
    monkeypatch.setenv("FIREREDVAD_MODEL_DIR", "model-dir")

    analyzer = vad.create_vad("firered")
    frame = bytes(400 * 2)

    assert isinstance(analyzer, CommunityFireVadAnalyzer)
    assert analyzer.kwargs["model_dir"] == "model-dir"
    assert analyzer.kwargs["speech_threshold"] == 0.6
    assert analyzer.num_frames_required() == 400
    assert analyzer.voice_confidence(frame) == 1.0
    assert analyzer.voice_confidence(frame) == 0.0
    assert all(len(sent) == 400 and sent.dtype.name == "int16" for sent in analyzer._vad.frames)
    analyzer.reset()
    assert analyzer.reset_calls == 1


@pytest.mark.parametrize("value", ["nan", "-0.1", "1.1"])
def test_firered_speech_threshold_is_validated_before_construction(monkeypatch, value):
    constructed = False

    class FakeFireVadAnalyzer:
        def __init__(self, **kwargs):
            nonlocal constructed
            constructed = True

    monkeypatch.setitem(
        sys.modules,
        "pipecat_firered_vad",
        SimpleNamespace(FireVadAnalyzer=FakeFireVadAnalyzer),
    )
    monkeypatch.setenv("FIREREDVAD_MODEL_DIR", "model-dir")
    monkeypatch.setenv("FIREREDVAD_SPEECH_THRESHOLD", value)

    with pytest.raises(RuntimeError, match="FIREREDVAD_SPEECH_THRESHOLD"):
        vad.create_vad("firered")
    assert constructed is False


@pytest.mark.parametrize(
    ("backend", "selected_module"),
    [
        ("ten", "ten_vad"),
        ("firered", "pipecat_firered_vad"),
        ("cobra", "pvcobra"),
    ],
)
def test_optional_imports_preserve_missing_transitive_dependencies(
    monkeypatch, backend, selected_module
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
    if backend == "cobra":
        monkeypatch.setenv("PICOVOICE_ACCESS_KEY", "test-key")
    if backend == "firered":
        monkeypatch.setenv("FIREREDVAD_MODEL_DIR", "model-dir")

    with pytest.raises(ModuleNotFoundError) as exc_info:
        vad.create_vad(backend)
    assert exc_info.value.name == "transitive_dependency"


def test_cobra_adapter_converts_little_endian_pcm_and_cleans_up(monkeypatch):
    class FakeCobra:
        sample_rate = 16000
        frame_length = 4

        def __init__(self):
            self.samples = None
            self.delete_calls = 0

        def process(self, samples):
            self.samples = list(samples)
            return 0.75

        def delete(self):
            self.delete_calls += 1

    engine = FakeCobra()
    monkeypatch.setitem(
        sys.modules,
        "pvcobra",
        SimpleNamespace(create=lambda access_key: engine),
    )

    analyzer = vad.CobraVADAnalyzer("test-key")
    analyzer.set_sample_rate(16000)

    assert analyzer.num_frames_required() == 4
    assert analyzer.voice_confidence(b"\x01\x00\xfe\xff\x03\x00\xfc\xff") == 0.75
    assert engine.samples == [1, -2, 3, -4]

    asyncio.run(analyzer.cleanup())
    asyncio.run(analyzer.cleanup())
    assert engine.delete_calls == 1


def test_ten_adapter_converts_pcm_and_returns_probability(monkeypatch):
    class FakeTenVad:
        def __init__(self, hop_size, threshold):
            self.hop_size = hop_size
            self.threshold = threshold
            self.samples = None

        def process(self, samples):
            self.samples = samples.tolist()
            return 0.875, 1

    monkeypatch.setitem(sys.modules, "ten_vad", SimpleNamespace(TenVad=FakeTenVad))

    analyzer = vad.TenVADAnalyzer(threshold=0.6)
    analyzer.set_sample_rate(16000)

    assert analyzer.num_frames_required() == 256
    assert analyzer.voice_confidence(b"\x01\x00\xfe\xff" * 128) == 0.875
    assert analyzer._ten.samples[:4] == [1, -2, 1, -2]
    assert analyzer._ten.hop_size == 256
    assert analyzer._ten.threshold == 0.6

    asyncio.run(analyzer.cleanup())
    asyncio.run(analyzer.cleanup())
    assert analyzer._ten is None


def test_ten_native_threshold_flag_gates_probability(monkeypatch):
    class FakeTenVad:
        def __init__(self, hop_size, threshold):
            self.flags = iter((0, 1))

        def process(self, samples):
            return 0.875, next(self.flags)

    monkeypatch.setitem(sys.modules, "ten_vad", SimpleNamespace(TenVad=FakeTenVad))
    analyzer = vad.TenVADAnalyzer(threshold=0.6)

    buffer = bytes(512)
    assert analyzer.voice_confidence(buffer) == 0.0
    assert analyzer.voice_confidence(buffer) == 0.875


def test_ten_factory_uses_project_adapter(monkeypatch):
    class FakeTenVad:
        def __init__(self, hop_size, threshold):
            self.hop_size = hop_size
            self.threshold = threshold

    monkeypatch.setitem(sys.modules, "ten_vad", SimpleNamespace(TenVad=FakeTenVad))
    monkeypatch.setenv("TEN_VAD_THRESHOLD", "0.65")

    analyzer = vad.create_vad("ten")

    assert isinstance(analyzer, vad.TenVADAnalyzer)
    assert analyzer._ten.hop_size == 256
    assert analyzer._ten.threshold == 0.65


def test_optional_dependencies_use_current_upstreams():
    pyproject = open("pyproject.toml", encoding="utf-8").read()

    assert "pipecat-ten-vad" not in pyproject
    assert (
        'ten = ["ten-vad @ git+https://github.com/TEN-framework/ten-vad.git@22a3bcd4509d0faaa8eef4881e8af5f39c178950"]'
        in pyproject
    )
    assert (
        'firered = ["fireredvad[cpu]==0.0.2", "pipecat-firered-vad==0.1.0"]'
        in pyproject
    )
