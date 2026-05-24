from __future__ import annotations

import argparse
import asyncio
import gzip
import json
import math
import os
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import librosa
import numpy as np
import websockets


WS_URL = "wss://openspeech.bytedance.com/api/v3/sauc/bigmodel"
DEFAULT_RESOURCE_ID = "volc.seedasr.sauc.duration"
TARGET_RATE = 16000

MSG_FULL_CLIENT = 0x1
MSG_AUDIO_ONLY = 0x2
MSG_FULL_SERVER = 0x9
MSG_ERROR = 0xF

SER_NONE = 0x0
SER_JSON = 0x1
COMP_GZIP = 0x1

FLAG_NO_SEQ = 0x0
FLAG_POS_SEQ = 0x1
FLAG_LAST_NO_SEQ = 0x2
FLAG_NEG_SEQ = 0x3


@dataclass
class LatencyStats:
    audio_duration_s: float
    send_done_s: float | None = None
    first_text_s: float | None = None
    first_definite_s: float | None = None
    final_s: float | None = None

    @property
    def tail_latency_s(self) -> float | None:
        if self.final_s is None or self.send_done_s is None:
            return None
        return self.final_s - self.send_done_s


def make_header(
    message_type: int,
    flags: int,
    serialization: int,
    compression: int,
) -> bytes:
    return bytes(
        [
            0x11,
            (message_type << 4) | flags,
            (serialization << 4) | compression,
            0x00,
        ]
    )


def pack_full_client_request(payload: dict[str, Any]) -> bytes:
    body = gzip.compress(json.dumps(payload, ensure_ascii=False).encode("utf-8"))
    return (
        make_header(MSG_FULL_CLIENT, FLAG_NO_SEQ, SER_JSON, COMP_GZIP)
        + len(body).to_bytes(4, "big")
        + body
    )


def pack_audio_request(chunk: bytes, sequence: int, *, last: bool) -> bytes:
    body = gzip.compress(chunk)
    flags = FLAG_NEG_SEQ if last else FLAG_POS_SEQ
    wire_sequence = -sequence if last else sequence
    return (
        make_header(MSG_AUDIO_ONLY, flags, SER_NONE, COMP_GZIP)
        + int(wire_sequence).to_bytes(4, "big", signed=True)
        + len(body).to_bytes(4, "big")
        + body
    )


def decode_frame(data: bytes) -> dict[str, Any]:
    header_len = (data[0] & 0x0F) * 4
    message_type = data[1] >> 4
    flags = data[1] & 0x0F
    serialization = data[2] >> 4
    compression = data[2] & 0x0F
    offset = header_len
    sequence: int | None = None

    if message_type == MSG_FULL_SERVER and flags in {FLAG_POS_SEQ, FLAG_NEG_SEQ}:
        sequence = int.from_bytes(data[offset : offset + 4], "big", signed=True)
        offset += 4

    if message_type == MSG_ERROR:
        code = int.from_bytes(data[offset : offset + 4], "big", signed=False)
        offset += 4
        size = int.from_bytes(data[offset : offset + 4], "big", signed=False)
        offset += 4
        payload = data[offset : offset + size].decode("utf-8", errors="replace")
        return {"kind": "error", "code": code, "payload": payload}

    size = int.from_bytes(data[offset : offset + 4], "big", signed=False)
    offset += 4
    payload: bytes | dict[str, Any] | None = data[offset : offset + size]

    if compression == COMP_GZIP and payload:
        payload = gzip.decompress(payload)
    if serialization == SER_JSON:
        payload = json.loads(payload.decode("utf-8")) if payload else None

    return {
        "kind": "server",
        "message_type": message_type,
        "flags": flags,
        "sequence": sequence,
        "payload": payload,
    }


def load_pcm16_mono_16k(audio_path: Path) -> bytes:
    audio, _ = librosa.load(audio_path, sr=TARGET_RATE, mono=True)
    samples = np.clip(audio, -1.0, 1.0)
    return (samples * 32767.0).astype("<i2").tobytes()


def build_request(*, enable_nonstream: bool) -> dict[str, Any]:
    request: dict[str, Any] = {
        "model_name": "bigmodel",
        "enable_itn": True,
        "enable_punc": True,
        "show_utterances": True,
        "result_type": "full",
    }
    if enable_nonstream:
        request.update({"enable_nonstream": True, "ssd_version": "200"})

    return {
        "user": {"uid": "real-time-chat-local"},
        "audio": {
            "format": "pcm",
            "codec": "raw",
            "rate": TARGET_RATE,
            "bits": 16,
            "channel": 1,
        },
        "request": request,
    }


async def connect_websocket(headers: dict[str, str]) -> Any:
    try:
        return await websockets.connect(
            WS_URL,
            additional_headers=headers,
            max_size=1_000_000_000,
            ping_interval=None,
        )
    except TypeError:
        return await websockets.connect(
            WS_URL,
            extra_headers=headers,
            max_size=1_000_000_000,
            ping_interval=None,
        )


async def receive_results(
    websocket: Any,
    stats: LatencyStats,
    start_time: float,
    *,
    print_stream: bool,
) -> str:
    final_text = ""
    last_text = ""

    while True:
        frame = decode_frame(await websocket.recv())
        now_s = time.perf_counter() - start_time

        if frame.get("kind") == "error":
            raise RuntimeError(
                f"Volcengine ASR error {frame.get('code')}: {frame.get('payload')}"
            )

        payload = frame.get("payload")
        result = payload.get("result") if isinstance(payload, dict) else None
        if isinstance(result, dict):
            text = result.get("text") or ""
            if text:
                final_text = text
                if stats.first_text_s is None:
                    stats.first_text_s = now_s
                if print_stream and text != last_text:
                    print(f"STREAM_TEXT: {text}")
                    last_text = text

            for utterance in result.get("utterances") or []:
                if utterance.get("definite"):
                    if stats.first_definite_s is None:
                        stats.first_definite_s = now_s
                    if print_stream:
                        print(
                            "DEFINITE: "
                            f"{utterance.get('start_time')}..{utterance.get('end_time')} "
                            f"{utterance.get('text')}"
                        )

        if frame.get("flags") in {FLAG_LAST_NO_SEQ, FLAG_NEG_SEQ}:
            stats.final_s = now_s
            return final_text


async def transcribe_streaming(args: argparse.Namespace) -> dict[str, Any]:
    api_key = args.api_key or os.getenv("VOLCENGINE_ASR_API_KEY")
    if not api_key:
        raise RuntimeError("Set VOLCENGINE_ASR_API_KEY or pass --api-key.")

    audio_path = Path(args.audio_path).expanduser()
    pcm = load_pcm16_mono_16k(audio_path)
    stats = LatencyStats(audio_duration_s=len(pcm) / 2 / TARGET_RATE)

    request_id = str(uuid.uuid4())
    headers = {
        "X-Api-Key": api_key,
        "X-Api-Resource-Id": args.resource_id,
        "X-Api-Request-Id": request_id,
        "X-Api-Connect-Id": request_id,
        "X-Api-Sequence": "-1",
    }

    websocket = await connect_websocket(headers)
    async with websocket:
        start_time = time.perf_counter()
        receiver = asyncio.create_task(
            receive_results(
                websocket,
                stats,
                start_time,
                print_stream=args.print_stream,
            )
        )

        await websocket.send(
            pack_full_client_request(
                build_request(enable_nonstream=args.enable_nonstream)
            )
        )

        chunk_size = TARGET_RATE * 2 * args.chunk_ms // 1000
        chunk_count = math.ceil(len(pcm) / chunk_size)
        for index, offset in enumerate(range(0, len(pcm), chunk_size), 1):
            sequence = index + 1
            chunk = pcm[offset : offset + chunk_size]
            await websocket.send(
                pack_audio_request(chunk, sequence, last=index == chunk_count)
            )
            if args.realtime and index != chunk_count:
                await asyncio.sleep(args.chunk_ms / 1000)

        stats.send_done_s = time.perf_counter() - start_time
        final_text = await asyncio.wait_for(receiver, timeout=args.final_timeout_s)

    return {
        "text": final_text,
        "request_id": request_id,
        "audio_duration_s": round(stats.audio_duration_s, 3),
        "send_done_s": round(stats.send_done_s, 3) if stats.send_done_s else None,
        "first_text_s": round(stats.first_text_s, 3) if stats.first_text_s else None,
        "first_definite_s": (
            round(stats.first_definite_s, 3) if stats.first_definite_s else None
        ),
        "final_s": round(stats.final_s, 3) if stats.final_s else None,
        "tail_latency_s": (
            round(stats.tail_latency_s, 3) if stats.tail_latency_s is not None else None
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Transcribe audio with Volcengine ASR 2.0 bidirectional streaming."
    )
    parser.add_argument("audio_path", help="Path to the input audio file.")
    parser.add_argument("--api-key", help="Volcengine ASR API key.")
    parser.add_argument("--chunk-ms", type=int, default=200, help="Audio chunk size.")
    parser.add_argument(
        "--resource-id",
        default=DEFAULT_RESOURCE_ID,
        help="Volcengine ASR resource id.",
    )
    parser.add_argument(
        "--enable-nonstream",
        action="store_true",
        help="Enable ASR 2.0 second-pass nonstream recognition.",
    )
    parser.add_argument(
        "--no-realtime",
        dest="realtime",
        action="store_false",
        help="Send chunks as fast as possible instead of real-time pacing.",
    )
    parser.add_argument(
        "--print-stream",
        action="store_true",
        help="Print partial and definite streaming results.",
    )
    parser.add_argument(
        "--final-timeout-s",
        type=float,
        default=15.0,
        help="Seconds to wait for the final response after sending audio.",
    )
    parser.set_defaults(realtime=True)
    return parser.parse_args()


def main() -> None:
    result = asyncio.run(transcribe_streaming(parse_args()))
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
