import logging
import os
import sys


def _configure_utf8_streams() -> None:
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure:
            reconfigure(encoding="utf-8")


if __name__ == "__main__":
    _configure_utf8_streams()


from dotenv import load_dotenv

from pipecat.audio.filters.base_audio_filter import BaseAudioFilter
from pipecat.frames.frames import LLMRunFrame, MetricsFrame
from pipecat.metrics.metrics import TurnMetricsData
from pipecat.observers.user_bot_latency_observer import UserBotLatencyObserver
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.worker import PipelineParams, PipelineWorker
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import (
    AssistantTurnStoppedMessage,
    LLMContextAggregatorPair,
    LLMUserAggregatorParams,
    UserTurnMessageAddedMessage,
)
from pipecat.runner.types import RunnerArguments
from pipecat.runner.utils import create_transport
from pipecat.services.google.gemini_live.llm import (
    GeminiLiveLLMService,
    GeminiModalities,
    GeminiVADParams,
)
from pipecat.transports.base_transport import BaseTransport, TransportParams
from pipecat.turns.user_turn_strategies import UserTurnStrategies
from pipecat.workers.runner import WorkerRunner

from vad import _float_env, create_vad

load_dotenv(override=False)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("voiceassistant")

SYSTEM_INSTRUCTION = (
    "You are a friendly Hindi-first voice assistant. Speak primarily in Hindi, "
    "allow natural Hindi-English code-switching, and follow the user's language. "
    "Keep every response brief, conversational, and suitable for speech. "
    "Do not use Markdown, bullets, emojis, or formatting that cannot be spoken."
)
DEFAULT_GEMINI_MODEL = "models/gemini-2.5-flash-native-audio-preview-12-2025"
GEMINI_VAD_OBSERVABILITY_NOTICE = (
    "Gemini server VAD enabled; Pipecat user-speaking frames, turn tracking, "
    "frame-driven interruption, and user-to-bot latency are unavailable. "
    "Gemini handles VAD and interruption server-side."
)
LOCAL_VAD_OBSERVABILITY_NOTICE = (
    "Local VAD observability enabled: user-speaking frames, turn tracking, "
    "interruption, and user-to-bot latency measurements."
)


def get_google_api_key() -> str:
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError("GOOGLE_API_KEY is required; copy .env.example to .env and set it")
    return api_key


def get_gemini_model() -> str:
    return os.getenv("GEMINI_MODEL", DEFAULT_GEMINI_MODEL)


def get_gemini_voice() -> str:
    return os.getenv("GEMINI_VOICE", "Charon")


def create_audio_filter(name: str | None = None) -> BaseAudioFilter | None:
    selected = (name or os.getenv("NOISE_FILTER", "browser")).strip().lower()
    built = _build_audio_filter(selected)
    if built is None:
        return None

    # Blending a little of the original back in hides the artifacts aggressive denoising
    # leaves behind. Applied once, around the whole chain.
    wet = _float_env("NOISE_MIX", 1.0, minimum=0, maximum=1)
    if wet >= 1.0:
        return built
    from noise import MixedAudioFilter

    return MixedAudioFilter(built, wet=wet)


def _build_audio_filter(selected: str) -> BaseAudioFilter | None:
    if "+" in selected:
        # "highpass+rnnoise" runs the stages in order. "browser" contributes no server-side
        # filter, so it drops out of the chain.
        stages = [_build_audio_filter(part.strip()) for part in selected.split("+")]
        stages = [stage for stage in stages if stage is not None]
        if not stages:
            return None
        if len(stages) == 1:
            return stages[0]
        from noise import ChainedAudioFilter

        return ChainedAudioFilter(stages)
    if selected == "browser":
        return None
    if selected == "rnnoise":
        try:
            import pyrnnoise  # noqa: F401
            from pipecat.audio.filters.rnnoise_filter import RNNoiseFilter
        except ModuleNotFoundError as exc:
            if exc.name != "pyrnnoise":
                raise
            raise RuntimeError(
                "RNNoise selected; install it with 'uv sync --extra rnnoise'"
            ) from exc
        # Pipecat defaults to "QQ", the lowest soxr quality, and RNNoise round-trips
        # 16k -> 48k -> 16k. That double resample is audible; "HQ" costs little.
        quality = os.getenv("RNNOISE_QUALITY", "HQ").strip().upper()
        if quality not in ("QQ", "LQ", "MQ", "HQ", "VHQ"):
            raise RuntimeError("RNNOISE_QUALITY must be one of: QQ, LQ, MQ, HQ, VHQ")
        return RNNoiseFilter(resampler_quality=quality)
    if selected == "highpass":
        from noise import HighPassFilter

        return HighPassFilter(cutoff_hz=_float_env("HIGHPASS_HZ", 100.0, minimum=1))
    if selected == "koala":
        access_key = os.getenv("KOALA_ACCESS_KEY")
        if not access_key:
            raise RuntimeError("KOALA_ACCESS_KEY is required when NOISE_FILTER=koala")
        try:
            import pvkoala  # noqa: F401
            from pipecat.audio.filters.koala_filter import KoalaFilter
        except ModuleNotFoundError as exc:
            if exc.name != "pvkoala":
                raise
            raise RuntimeError(
                "Koala selected; install it with 'uv sync --extra koala'"
            ) from exc
        return KoalaFilter(access_key=access_key)
    raise ValueError(
        "NOISE_FILTER must be one of: browser, highpass, rnnoise, koala "
        "(or several joined with '+', such as highpass+rnnoise)"
    )


def webrtc_transport_params(audio_filter: BaseAudioFilter | None = None) -> TransportParams:
    return TransportParams(
        audio_in_enabled=True,
        audio_out_enabled=True,
        audio_in_sample_rate=16000,
        audio_in_filter=audio_filter,
    )


def create_turn_strategies(selected: str | None = None) -> UserTurnStrategies:
    selected = (selected or os.getenv("TURN_DETECTION", "smart")).strip().lower()
    if selected == "smart":
        from pipecat.audio.turn.smart_turn.base_smart_turn import SmartTurnParams
        from pipecat.audio.turn.smart_turn.local_smart_turn_v3 import LocalSmartTurnAnalyzerV3
        from pipecat.turns.user_stop import TurnAnalyzerUserTurnStopStrategy

        analyzer = LocalSmartTurnAnalyzerV3(
            params=SmartTurnParams(
                stop_secs=_float_env("SMART_TURN_STOP_SECS", 3.0, minimum=0),
                max_duration_secs=_float_env("SMART_TURN_MAX_DURATION_SECS", 8.0, minimum=1),
            )
        )
        return UserTurnStrategies(
            stop=[TurnAnalyzerUserTurnStopStrategy(turn_analyzer=analyzer)]
        )
    if selected == "vad":
        from pipecat.turns.user_stop import SpeechTimeoutUserTurnStopStrategy

        return UserTurnStrategies(stop=[SpeechTimeoutUserTurnStopStrategy()])
    raise ValueError("TURN_DETECTION must be one of: smart, vad")


def vad_configuration(analyzer):
    if analyzer is None:
        return GeminiVADParams(disabled=False), None
    return (
        GeminiVADParams(disabled=True),
        LLMUserAggregatorParams(
            vad_analyzer=analyzer,
            user_turn_strategies=create_turn_strategies(),
        ),
    )


def observability_configuration(analyzer) -> tuple[bool, str]:
    if analyzer is None:
        return False, GEMINI_VAD_OBSERVABILITY_NOTICE
    return True, LOCAL_VAD_OBSERVABILITY_NOTICE


async def run_agent(transport: BaseTransport, runner_args: RunnerArguments) -> None:
    api_key = get_google_api_key()
    model = get_gemini_model()
    voice = get_gemini_voice()
    analyzer = create_vad()
    try:
        gemini_vad, user_params = vad_configuration(analyzer)
        enable_turn_tracking, observability_notice = observability_configuration(analyzer)
        logger.info("vad_backend=%s", os.getenv("VAD_BACKEND", "gemini").strip().lower())
        if analyzer is not None:
            logger.info(
                "turn_detection=%s", os.getenv("TURN_DETECTION", "smart").strip().lower()
            )
        logger.info(observability_notice)
        llm = GeminiLiveLLMService(
            api_key=api_key,
            settings=GeminiLiveLLMService.Settings(
                model=model,
                voice=voice,
                modalities=GeminiModalities.AUDIO,
                system_instruction=SYSTEM_INSTRUCTION,
                vad=gemini_vad,
            ),
        )

        context = LLMContext()
        user_aggregator, assistant_aggregator = LLMContextAggregatorPair(
            context,
            realtime_service_mode=True,
            user_params=user_params,
        )
        pipeline = Pipeline(
            [
                transport.input(),
                user_aggregator,
                llm,
                transport.output(),
                assistant_aggregator,
            ]
        )

        latency_observer = UserBotLatencyObserver()
        worker = PipelineWorker(
            pipeline,
            params=PipelineParams(
                audio_in_sample_rate=16000,
                enable_metrics=True,
                enable_usage_metrics=True,
            ),
            observers=[latency_observer],
            idle_timeout_secs=runner_args.pipeline_idle_timeout_secs,
            enable_turn_tracking=enable_turn_tracking,
        )
    except Exception:
        if analyzer is not None:
            await analyzer.cleanup()
        raise

    @latency_observer.event_handler("on_first_bot_speech_latency")
    async def on_first_bot_speech_latency(observer, latency):
        logger.info("first_bot_speech_latency_seconds=%.3f", latency)

    @latency_observer.event_handler("on_latency_measured")
    async def on_latency_measured(observer, latency):
        logger.info("user_to_bot_latency_seconds=%.3f", latency)

    worker.set_reached_downstream_filter((MetricsFrame,))

    @worker.event_handler("on_frame_reached_downstream")
    async def on_frame_reached_downstream(worker, frame):
        for data in frame.data:
            if isinstance(data, TurnMetricsData):
                logger.info(
                    "turn_prediction complete=%s probability=%.3f e2e_ms=%.1f",
                    data.is_complete,
                    data.probability,
                    data.e2e_processing_time_ms,
                )

    @user_aggregator.event_handler("on_user_turn_started")
    async def on_user_turn_started(aggregator, strategy):
        logger.info("user_speaking_started")

    @user_aggregator.event_handler("on_user_turn_stopped")
    async def on_user_turn_stopped(aggregator, strategy, message):
        logger.info("user_speaking_stopped")

    if worker.turn_tracking_observer:

        @worker.turn_tracking_observer.event_handler("on_turn_started")
        async def on_turn_started(observer, turn_number):
            logger.info("turn_started=%d", turn_number)

        @worker.turn_tracking_observer.event_handler("on_turn_ended")
        async def on_turn_ended(observer, turn_number, duration, was_interrupted):
            logger.info(
                "turn_ended=%d duration_seconds=%.3f interrupted=%s",
                turn_number,
                duration,
                was_interrupted,
            )

    @user_aggregator.event_handler("on_user_turn_message_added")
    async def on_user_turn_message_added(
        aggregator, message: UserTurnMessageAddedMessage
    ):
        logger.info("user_transcript=%s", message.content)

    @assistant_aggregator.event_handler("on_assistant_turn_started")
    async def on_assistant_turn_started(aggregator):
        logger.info("assistant_speaking_started")

    @assistant_aggregator.event_handler("on_assistant_turn_stopped")
    async def on_assistant_turn_stopped(
        aggregator, message: AssistantTurnStoppedMessage
    ):
        logger.info(
            "assistant_transcript=%s interrupted=%s",
            message.content,
            message.interrupted,
        )

    @worker.event_handler("on_pipeline_error")
    async def on_pipeline_error(worker, frame):
        logger.error("pipeline_error=%s fatal=%s", frame.error, frame.fatal)

    @transport.event_handler("on_client_connected")
    async def on_client_connected(transport, client):
        logger.info("client_connected")

    @worker.rtvi.event_handler("on_client_ready")
    async def on_client_ready(rtvi):
        logger.info("client_ready")
        context.add_message(
            {
                "role": "developer",
                "content": "Greet the user briefly in Hindi and ask how you can help.",
            }
        )
        await worker.queue_frame(LLMRunFrame())

    @transport.event_handler("on_client_disconnected")
    async def on_client_disconnected(transport, client):
        logger.info("client_disconnected")
        await worker.cancel()

    runner = WorkerRunner(handle_sigint=runner_args.handle_sigint)
    await runner.add_workers(worker)
    await runner.run()


async def bot(runner_args: RunnerArguments) -> None:
    audio_filter = create_audio_filter()
    logger.info("noise_filter=%s", os.getenv("NOISE_FILTER", "browser").strip().lower())
    transport = await create_transport(
        runner_args,
        {"webrtc": lambda: webrtc_transport_params(audio_filter)},
    )
    await run_agent(transport, runner_args)


if __name__ == "__main__":
    from pipecat.runner.run import main

    main()
