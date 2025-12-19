#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import os

import aiohttp
from dotenv import load_dotenv
from loguru import logger

from pipecat.frames.frames import LLMRunFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response import (
    LLMUserAggregatorParams,
)
from pipecat.processors.aggregators.llm_response_universal import (
    LLMContextAggregatorPair,
)
from pipecat.services.elevenlabs.tts import ElevenLabsTTSService
from pipecat.processors.frameworks.rtvi import RTVIConfig, RTVIObserver, RTVIProcessor
from pipecat.runner.types import RunnerArguments
from pipecat.runner.utils import create_transport
from pipecat.services.openai.base_llm import BaseOpenAILLMService
from pipecat.services.openai.llm import OpenAILLMService
from pipecat.services.speechmatics.stt import SpeechmaticsSTTService
from pipecat.transcriptions.language import Language
from pipecat.transports.base_transport import BaseTransport, TransportParams

load_dotenv(override=True)

# We store functions so objects (e.g. SileroVADAnalyzer) don't get
# instantiated. The function will be called when the desired transport gets
# selected.
transport_params = {
    "webrtc": lambda: TransportParams(
        audio_in_enabled=True,
        audio_out_enabled=True,
    ),
}


async def run_bot(transport: BaseTransport, runner_args: RunnerArguments):
    """Run example using Speechmatics STT.

    This example demonstrates a complete Speechmatics integration with Speech-to-Text
    service:

    STT Features:
    - Diarization to identify and distinguish between different speakers
    - Words spoken by each speaker are wrapped with XML tags for LLM processing
    - System context instructions help the LLM understand multi-party conversations
    - ENHANCED operating point by default for optimal accuracy

    For more information:
    - STT: https://docs.speechmatics.com/rt-api-ref
    """

    logger.info("Starting bot")

    async with aiohttp.ClientSession() as session:
        stt = SpeechmaticsSTTService(
            api_key=os.getenv("SPEECHMATICS_API_KEY"),
            params=SpeechmaticsSTTService.InputParams(
                language=Language.EN,
                turn_detection_mode=SpeechmaticsSTTService.TurnDetectionMode.SMART_TURN,
                speaker_active_format="<{speaker_id}>{text}</{speaker_id}>",
            ),
        )

        tts = ElevenLabsTTSService(
            api_key=os.getenv("ELEVENLABS_API_KEY"),
            voice_id="97U3B7htAA7UsCIDST8b",
            model="eleven_turbo_v2_5",
            aiohttp_session=session,
        )

        llm = OpenAILLMService(
            api_key=os.getenv("OPENAI_API_KEY"),
            params=BaseOpenAILLMService.InputParams(temperature=0.75),
        )

        messages = [
            {
                "role": "system",
                "content": (
                    "You are a helpful and witty British assistant called Humphrey. "
                    "Your goal is to demonstrate your capabilities in a succinct way. "
                    "Your output will be spoken aloud, so avoid special characters that can't easily be spoken, such as emojis or bullet points. "
                    "Always include punctuation in your responses. "
                    "Give very short replies - do not give longer replies unless strictly necessary. "
                    "Respond to what the user said in a concise, funny, creative and helpful way. "
                    "Use `<Sn/>` tags to identify different speakers - do not use tags in your replies."
                ),
            },
        ]

        context = LLMContext(messages)
        context_aggregator = LLMContextAggregatorPair(
            context,
            user_params=LLMUserAggregatorParams(aggregation_timeout=0.005),
        )

        rtvi = RTVIProcessor(config=RTVIConfig(config=[]))

        pipeline = Pipeline(
            [
                transport.input(),  # Transport user input
                stt,  # STT
                context_aggregator.user(),  # User responses
                llm,  # LLM
                rtvi,  # UI updates
                tts,  # TTS
                transport.output(),  # Transport bot output
                context_aggregator.assistant(),  # Assistant spoken responses
            ]
        )

        task = PipelineTask(
            pipeline,
            params=PipelineParams(
                enable_metrics=True,
                enable_usage_metrics=True,
            ),
            observers=[RTVIObserver(rtvi)],
            idle_timeout_secs=runner_args.pipeline_idle_timeout_secs,
        )

        @rtvi.event_handler("on_client_ready")
        async def on_client_ready(rtvi):
            logger.info("Pipecat client ready.")
            await rtvi.set_bot_ready()

        @transport.event_handler("on_client_connected")
        async def on_client_connected(transport, client):
            logger.info("Client connected")
            messages.append(
                {"role": "system", "content": "Say a short hello to the user."}
            )
            await task.queue_frames([LLMRunFrame()])

        @transport.event_handler("on_client_disconnected")
        async def on_client_disconnected(transport, client):
            logger.info("Client disconnected")
            await task.cancel()

        runner = PipelineRunner(handle_sigint=runner_args.handle_sigint)

        await runner.run(task)


async def bot(runner_args: RunnerArguments):
    """Main bot entry point compatible with Pipecat Cloud."""
    transport = await create_transport(runner_args, transport_params)
    await run_bot(transport, runner_args)


if __name__ == "__main__":
    from pipecat.runner.run import main

    main()
