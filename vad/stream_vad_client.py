from __future__ import annotations

import argparse
import asyncio
import json
import time
from pathlib import Path
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

import librosa
import numpy as np
import websockets


DEFAULT_URL = "ws://127.0.0.1:8010/v1/audio/vad/stream"
DEFAULT_SAMPLE_RATE = 16000
DEFAULT_CHUNK_MS = 32


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Send an audio file to the local VAD WebSocket as realtime PCM16 chunks."
    )
    parser.add_argument("audio", type=Path, help="Audio file to stream")
    parser.add_argument("--url", default=DEFAULT_URL, help="VAD WebSocket URL")
    parser.add_argument("--api-key", default=None, help="Bearer token for VAD_API_KEY")
    parser.add_argument("--threshold", type=float, default=None, help="Speech threshold")
    parser.add_argument(
        "--min-silence-ms",
        type=int,
        default=None,
        help="Silence duration before emitting speech_end",
    )
    parser.add_argument(
        "--min-speech-ms",
        type=int,
        default=None,
        help="Minimum speech duration before emitting speech_start",
    )
    parser.add_argument(
        "--min-gap-ms",
        type=int,
        default=None,
        help=(
            "Anti-jitter merge gap. Hold speech_end this long; if a new "
            "speech_start arrives within the window, the events are merged "
            "into one continuous segment (server default 300)."
        ),
    )
    parser.add_argument(
        "--chunk-ms",
        type=int,
        default=DEFAULT_CHUNK_MS,
        help="PCM chunk duration to send per WebSocket frame",
    )
    parser.add_argument(
        "--no-realtime",
        action="store_true",
        help="Send chunks as fast as possible instead of sleeping between chunks",
    )
    return parser.parse_args()


def with_query(url: str, params: dict[str, int | float | None]) -> str:
    parts = urlsplit(url)
    query = dict(parse_qsl(parts.query, keep_blank_values=True))
    query.update({key: str(value) for key, value in params.items() if value is not None})
    return urlunsplit((parts.scheme, parts.netloc, parts.path, urlencode(query), parts.fragment))


def load_pcm16(audio_path: Path, sample_rate: int) -> np.ndarray:
    audio, _ = librosa.load(audio_path, sr=sample_rate, mono=True)
    audio = np.clip(audio, -1.0, 1.0)
    return (audio * 32767.0).astype("<i2")


async def receive_events(websocket: websockets.ClientConnection) -> None:
    async for message in websocket:
        event = json.loads(message)
        print(json.dumps(event, ensure_ascii=False))
        if event.get("event") == "done":
            return


async def stream_audio(args: argparse.Namespace) -> None:
    if not args.audio.exists():
        raise FileNotFoundError(args.audio)
    if args.chunk_ms <= 0:
        raise ValueError("--chunk-ms must be greater than 0")

    url = with_query(
        args.url,
        {
            "threshold": args.threshold,
            "min_speech_ms": args.min_speech_ms,
            "min_silence_ms": args.min_silence_ms,
            "min_gap_ms": args.min_gap_ms,
        },
    )
    headers = {"Authorization": f"Bearer {args.api_key}"} if args.api_key else {}
    pcm = load_pcm16(args.audio, DEFAULT_SAMPLE_RATE)
    chunk_samples = max(1, DEFAULT_SAMPLE_RATE * args.chunk_ms // 1000)
    start_time = time.perf_counter()

    async with websockets.connect(url, additional_headers=headers) as websocket:
        receiver = asyncio.create_task(receive_events(websocket))

        for offset in range(0, len(pcm), chunk_samples):
            chunk = pcm[offset : offset + chunk_samples]
            await websocket.send(chunk.tobytes())
            if not args.no_realtime:
                await asyncio.sleep(len(chunk) / DEFAULT_SAMPLE_RATE)

        await websocket.send(json.dumps({"event": "eof"}))
        await receiver

    elapsed_s = time.perf_counter() - start_time
    audio_duration_s = len(pcm) / DEFAULT_SAMPLE_RATE
    print(
        json.dumps(
            {
                "event": "client_summary",
                "audio_duration_s": round(audio_duration_s, 3),
                "elapsed_s": round(elapsed_s, 3),
                "realtime": not args.no_realtime,
            },
            ensure_ascii=False,
        )
    )


def main() -> None:
    asyncio.run(stream_audio(parse_args()))


if __name__ == "__main__":
    main()
