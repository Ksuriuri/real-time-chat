"""Command-line entry point: load env, build clients, run orchestrator."""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import signal
import sys
from pathlib import Path

from dotenv import load_dotenv

from aec_processor import AecProcessor
from audio_io import MicSource, SpeakerSink
from conversation import AsyncArkStream, ConversationHistory
from orchestrator import (
    ChatOrchestrator,
    OrchestratorConfig,
    build_asr_factory,
    build_executor,
)
from tts_client import SeedTtsClient
from vad_client import VadClient

# Allow importing the sibling ``llm`` module without packaging.
_LLM_DIR = Path(__file__).resolve().parent.parent / "llm"
if str(_LLM_DIR) not in sys.path:
    sys.path.insert(0, str(_LLM_DIR))

from ark_language_model import ArkLanguageModel  # noqa: E402


LOGGER = logging.getLogger("chat")

DEFAULT_VAD_URL = "ws://127.0.0.1:8010/v1/audio/vad/stream"
DEFAULT_SYSTEM_PROMPT = "你是一个语音助手，回答简短自然，控制在两句话以内。"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Realtime voice chat orchestrator (CLI)."
    )
    parser.add_argument(
        "--env-file",
        type=Path,
        default=None,
        help="Path to .env file (default: <repo-root>/.env).",
    )
    parser.add_argument("--vad-url", default=None)
    parser.add_argument("--vad-min-silence-ms", type=int, default=None)
    parser.add_argument("--vad-min-speech-ms", type=int, default=None)
    parser.add_argument("--vad-threshold", type=float, default=None)
    parser.add_argument(
        "--vad-min-gap-ms",
        type=int,
        default=None,
        help=(
            "Anti-jitter merge gap. Hold speech_end this long; if a new "
            "speech_start arrives within the window, treat it as continuous "
            "speech (default from VAD_MIN_GAP_MS or 300)."
        ),
    )
    parser.add_argument("--asr-chunk-ms", type=int, default=200)
    parser.add_argument("--asr-finalize-timeout-s", type=float, default=2.0)
    parser.add_argument("--llm-max-output-tokens", type=int, default=512)
    parser.add_argument("--llm-temperature", type=float, default=None)
    parser.add_argument("--system-prompt", default=None)
    parser.add_argument(
        "--log-level",
        default=None,
        help="Logging level, e.g. DEBUG / INFO / WARNING (default: INFO).",
    )
    return parser.parse_args()


def configure_logging(level: str) -> None:
    logging.basicConfig(
        level=level.upper(),
        format="%(asctime)s %(levelname)-7s %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )


def env(key: str, default: str | None = None) -> str | None:
    value = os.getenv(key)
    if value is None:
        return default
    value = value.strip().strip('"').strip("'").strip()
    if value == "":
        return default
    return value


def env_int(key: str, default: int | None) -> int | None:
    value = env(key)
    if value is None:
        return default
    return int(value)


def env_float(key: str, default: float | None) -> float | None:
    value = env(key)
    if value is None:
        return default
    return float(value)


def env_bool(key: str, default: bool) -> bool:
    value = env(key)
    if value is None:
        return default
    return value.strip().lower() not in {"0", "false", "no", "off"}


def _build_aec(enable: bool, *, ns_level: int, stream_delay_ms: int) -> AecProcessor | None:
    if not enable:
        return None
    if not AecProcessor.is_available():
        LOGGER.warning(
            "ENABLE_AEC=1 but pywebrtc-audio is not installed; running without AEC."
        )
        return None
    try:
        aec = AecProcessor(ns_level=ns_level, stream_delay_ms=stream_delay_ms)
    except Exception as exc:
        LOGGER.warning("AEC initialization failed (%s); running without AEC.", exc)
        return None
    LOGGER.info(
        "AEC enabled (WebRTC AEC3 + NS level=%d, stream_delay_ms=%d)",
        ns_level,
        stream_delay_ms,
    )
    return aec


_PLACEHOLDER_PREFIXES = ("your-", "your_")


def require(value: str | None, name: str) -> str:
    if not value:
        raise SystemExit(f"{name} is required (set it in .env or environment).")
    if value.lower().startswith(_PLACEHOLDER_PREFIXES):
        raise SystemExit(
            f"{name} still holds the placeholder value {value!r}. "
            "Edit your .env file with the real key."
        )
    return value


def _mask(value: str) -> str:
    """Render a credential as ``len=NN tail=…ABCD`` for safe logging."""
    if not value:
        return "<empty>"
    tail = value[-4:] if len(value) >= 4 else value
    return f"len={len(value)} tail=…{tail}"


def _load_env_file(override: Path | None) -> Path | None:
    """Load the single source-of-truth .env file.

    Resolution order:
      1. ``--env-file`` argument if provided.
      2. ``<repo-root>/.env``.
    """
    repo_root = Path(__file__).resolve().parent.parent
    candidate = override if override is not None else repo_root / ".env"
    if candidate.exists():
        load_dotenv(candidate, override=True)
        return candidate
    return None


async def amain(args: argparse.Namespace) -> None:
    loaded = _load_env_file(args.env_file)

    log_level = args.log_level or env("LOG_LEVEL", "INFO") or "INFO"
    configure_logging(log_level)

    if loaded is not None:
        LOGGER.info("Loaded env file: %s", loaded)
    else:
        LOGGER.warning("No .env file found; relying on shell environment only.")

    ark_api_key = require(env("ARK_API_KEY"), "ARK_API_KEY")
    asr_api_key = require(env("VOLCENGINE_ASR_API_KEY"), "VOLCENGINE_ASR_API_KEY")
    tts_api_key = require(env("VOLCENGINE_TTS_API_KEY"), "VOLCENGINE_TTS_API_KEY")
    LOGGER.info(
        "Credentials: ARK %s | ASR %s | TTS %s",
        _mask(ark_api_key),
        _mask(asr_api_key),
        _mask(tts_api_key),
    )

    vad_url = args.vad_url or env("VAD_WS_URL", DEFAULT_VAD_URL)
    vad_api_key = env("VAD_API_KEY")
    vad_min_silence_ms = (
        args.vad_min_silence_ms
        if args.vad_min_silence_ms is not None
        else env_int("VAD_MIN_SILENCE_MS", 800)
    )
    vad_min_speech_ms = (
        args.vad_min_speech_ms
        if args.vad_min_speech_ms is not None
        else env_int("VAD_MIN_SPEECH_MS", 250)
    )
    vad_threshold = (
        args.vad_threshold
        if args.vad_threshold is not None
        else env_float("VAD_THRESHOLD", 0.5)
    )
    vad_min_gap_ms = (
        args.vad_min_gap_ms
        if args.vad_min_gap_ms is not None
        else env_int("VAD_MIN_GAP_MS", 300)
    )

    system_prompt = (
        args.system_prompt
        or env("SYSTEM_PROMPT")
        or DEFAULT_SYSTEM_PROMPT
    )

    llm = ArkLanguageModel(
        api_key=ark_api_key,
        base_url=env("ARK_BASE_URL"),
        model=env("ARK_MODEL"),
    )
    history = ConversationHistory(system_prompt=system_prompt)
    executor = build_executor()
    ark_stream = AsyncArkStream(llm, executor=executor)

    vad_client = VadClient(
        url=vad_url,
        api_key=vad_api_key,
        threshold=vad_threshold,
        min_speech_ms=vad_min_speech_ms,
        min_silence_ms=vad_min_silence_ms,
        min_gap_ms=vad_min_gap_ms,
    )

    asr_factory = build_asr_factory(
        api_key=asr_api_key,
        resource_id=env("VOLCENGINE_ASR_RESOURCE_ID", "volc.seedasr.sauc.duration") or "volc.seedasr.sauc.duration",
        chunk_ms=args.asr_chunk_ms,
    )

    tts_client = SeedTtsClient(
        api_key=tts_api_key,
        resource_id=env("VOLCENGINE_TTS_RESOURCE_ID", "seed-tts-2.0") or "seed-tts-2.0",
        speaker=env("TTS_SPEAKER", "zh_female_vv_uranus_bigtts") or "zh_female_vv_uranus_bigtts",
        model=env("TTS_MODEL", "seed-tts-2.0-standard") or "seed-tts-2.0-standard",
    )

    aec = _build_aec(
        env_bool("ENABLE_AEC", True),
        ns_level=env_int("AEC_NS_LEVEL", 2) or 2,
        stream_delay_ms=env_int("AEC_STREAM_DELAY_MS", 0) or 0,
    )

    speaker = SpeakerSink(aec=aec)
    orchestrator: ChatOrchestrator | None = None

    loop = asyncio.get_running_loop()

    async def on_pcm(pcm: bytes) -> None:
        if orchestrator is not None:
            await orchestrator.on_mic_pcm(pcm)

    mic = MicSource(loop=loop, on_pcm=on_pcm, aec=aec)

    orchestrator = ChatOrchestrator(
        vad_client=vad_client,
        mic=mic,
        speaker=speaker,
        asr_factory=asr_factory,
        tts_client=tts_client,
        ark_stream=ark_stream,
        history=history,
        config=OrchestratorConfig(
            asr_chunk_ms=args.asr_chunk_ms,
            asr_finalize_timeout_s=args.asr_finalize_timeout_s,
            llm_max_output_tokens=args.llm_max_output_tokens,
            llm_temperature=args.llm_temperature,
        ),
    )

    stop = asyncio.Event()

    def _trigger_stop() -> None:
        LOGGER.info("Stop signal received")
        stop.set()
        asyncio.create_task(orchestrator.stop())

    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, _trigger_stop)
        except NotImplementedError:
            pass

    LOGGER.info("Starting realtime chat. Speak into the microphone; Ctrl+C to quit.")
    try:
        await orchestrator.run()
    finally:
        executor.shutdown(wait=False, cancel_futures=True)


def main() -> None:
    args = parse_args()
    try:
        asyncio.run(amain(args))
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
