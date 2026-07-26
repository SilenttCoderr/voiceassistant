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
from pipecat.frames.frames import LLMRunFrame
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
from pipecat.workers.runner import WorkerRunner

from vad import create_vad

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
        return RNNoiseFilter()
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
    raise ValueError("NOISE_FILTER must be one of: browser, rnnoise, koala")


def webrtc_transport_params(audio_filter: BaseAudioFilter | None = None) -> TransportParams:
    return TransportParams(
        audio_in_enabled=True,
        audio_out_enabled=True,
        audio_in_sample_rate=16000,
        audio_in_filter=audio_filter,
    )


def vad_configuration(analyzer):
    if analyzer is None:
        return GeminiVADParams(disabled=False), None
    return (
        GeminiVADParams(disabled=True),
        LLMUserAggregatorParams(vad_analyzer=analyzer),
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
