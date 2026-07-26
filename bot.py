import logging
import os
import sys

from dotenv import load_dotenv

from pipecat.frames.frames import LLMRunFrame
from pipecat.observers.user_bot_latency_observer import UserBotLatencyObserver
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.worker import PipelineParams, PipelineWorker
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import (
    AssistantTurnStoppedMessage,
    LLMContextAggregatorPair,
    UserTurnMessageAddedMessage,
)
from pipecat.runner.types import RunnerArguments
from pipecat.runner.utils import create_transport
from pipecat.services.google.gemini_live.llm import (
    GeminiLiveLLMService,
    GeminiModalities,
)
from pipecat.transports.base_transport import BaseTransport, TransportParams
from pipecat.workers.runner import WorkerRunner

load_dotenv(override=True)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("voiceassistant")

SYSTEM_INSTRUCTION = (
    "You are a friendly Hindi-first voice assistant. Speak primarily in Hindi, "
    "allow natural Hindi-English code-switching, and follow the user's language. "
    "Keep every response brief, conversational, and suitable for speech. "
    "Do not use Markdown, bullets, emojis, or formatting that cannot be spoken."
)
DEFAULT_GEMINI_MODEL = "models/gemini-2.5-flash-native-audio-preview-12-2025"
LOCAL_VAD_OBSERVABILITY_NOTICE = (
    "Local user-speaking and user-to-bot latency events are disabled; "
    "select a local VAD backend in a later task to enable them."
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


def webrtc_transport_params() -> TransportParams:
    return TransportParams(
        audio_in_enabled=True,
        audio_out_enabled=True,
        audio_in_sample_rate=16000,
    )


def _configure_stdout_utf8() -> None:
    reconfigure = getattr(sys.stdout, "reconfigure", None)
    if reconfigure:
        reconfigure(encoding="utf-8")


async def run_agent(transport: BaseTransport, runner_args: RunnerArguments) -> None:
    logger.info(LOCAL_VAD_OBSERVABILITY_NOTICE)
    llm = GeminiLiveLLMService(
        api_key=get_google_api_key(),
        settings=GeminiLiveLLMService.Settings(
            model=get_gemini_model(),
            voice=get_gemini_voice(),
            modalities=GeminiModalities.AUDIO,
            system_instruction=SYSTEM_INSTRUCTION,
        ),
    )

    context = LLMContext()
    user_aggregator, assistant_aggregator = LLMContextAggregatorPair(
        context,
        realtime_service_mode=True,
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
        enable_turn_tracking=False,
    )

    @latency_observer.event_handler("on_first_bot_speech_latency")
    async def on_first_bot_speech_latency(observer, latency):
        logger.info("first_bot_speech_latency_seconds=%.3f", latency)

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
    transport = await create_transport(
        runner_args,
        {"webrtc": webrtc_transport_params},
    )
    await run_agent(transport, runner_args)


if __name__ == "__main__":
    _configure_stdout_utf8()
    from pipecat.runner.run import main

    main()
