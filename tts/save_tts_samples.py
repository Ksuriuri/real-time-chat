"""Save TTS sample audio to disk for latency / quality comparison.

Official doc links: ../docs/official-links.md
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import json
import os
import struct
import uuid
from pathlib import Path

import httpx
import ormsgpack
import websockets

DEFAULT_TEXT = "你好，这是一次语音合成延迟测试。"
DEFAULT_FISH_VOICE = "7f92f8afb8ec43bf81429cc1c9199cb1"
DEFAULT_SEED_VOICE = "zh_female_vv_uranus_bigtts"
OUTPUT_DIR = Path(__file__).resolve().parent / "output"


def save_fish_stream(api_key: str, text: str, output: Path) -> int:
    body = {
        "text": text,
        "reference_id": DEFAULT_FISH_VOICE,
        "format": "mp3",
        "latency": "normal",
    }
    headers = {
        "authorization": f"Bearer {api_key}",
        "content-type": "application/msgpack",
        "model": "s2-pro",
    }
    total = 0
    with output.open("wb") as handle, httpx.Client(timeout=60.0) as client:
        with client.stream(
            "POST",
            "https://api.fish.audio/v1/tts",
            content=ormsgpack.packb(body),
            headers=headers,
        ) as response:
            response.raise_for_status()
            for chunk in response.iter_bytes():
                if chunk:
                    handle.write(chunk)
                    total += len(chunk)
    return total


def save_seed_http_chunked(api_key: str, text: str, output: Path) -> int:
    headers = {
        "X-Api-Key": api_key,
        "X-Api-Resource-Id": "seed-tts-2.0",
        "X-Api-Request-Id": str(uuid.uuid4()),
        "Content-Type": "application/json",
    }
    body = {
        "user": {"uid": "tts-sample"},
        "req_params": {
            "text": text,
            "speaker": DEFAULT_SEED_VOICE,
            "model": "seed-tts-2.0-standard",
            "audio_params": {"format": "mp3", "sample_rate": 24000},
        },
    }
    total = 0
    buf = ""
    with output.open("wb") as handle, httpx.Client(timeout=60.0) as client:
        with client.stream(
            "POST",
            "https://openspeech.bytedance.com/api/v3/tts/unidirectional",
            headers=headers,
            json=body,
        ) as response:
            response.raise_for_status()
            for chunk in response.iter_bytes():
                if not chunk:
                    continue
                buf += chunk.decode("utf-8", "replace")
                while True:
                    newline = buf.find("\n")
                    if newline == -1:
                        break
                    line = buf[:newline].strip()
                    buf = buf[newline + 1 :]
                    if not line:
                        continue
                    event = json.loads(line)
                    data = event.get("data")
                    if isinstance(data, str) and data:
                        audio = base64.b64decode(data)
                        handle.write(audio)
                        total += len(audio)
    return total


def _parse_ws_frame(data: bytes) -> tuple[int | None, bytes]:
    header_size = (data[0] & 0x0F) * 4
    flags = data[1] & 0x0F
    pos = header_size
    event = None
    if flags & 0x04:
        event = struct.unpack(">I", data[pos : pos + 4])[0]
        pos += 4
        sid_len = struct.unpack(">I", data[pos : pos + 4])[0]
        pos += 4 + sid_len
    payload_size = struct.unpack(">I", data[pos : pos + 4])[0]
    pos += 4
    return event, data[pos : pos + payload_size]


def _build_ws_send_text(payload: dict) -> bytes:
    body = json.dumps(payload, ensure_ascii=False).encode()
    return bytes([0x11, 0x10, 0x10, 0x00]) + struct.pack(">I", len(body)) + body


def _build_ws_finish_connection() -> bytes:
    return bytes([0x11, 0x14, 0x10, 0x00]) + struct.pack(">I", 2)


async def _save_seed_ws_standard(api_key: str, text: str, output: Path) -> int:
    headers = {
        "X-Api-Key": api_key,
        "X-Api-Resource-Id": "seed-tts-2.0",
        "X-Api-Connect-Id": str(uuid.uuid4()),
    }
    body = {
        "user": {"uid": "tts-sample"},
        "req_params": {
            "text": text,
            "speaker": DEFAULT_SEED_VOICE,
            "model": "seed-tts-2.0-standard",
            "audio_params": {"format": "mp3", "sample_rate": 24000},
        },
    }
    total = 0
    with output.open("wb") as handle:
        async with websockets.connect(
            "wss://openspeech.bytedance.com/api/v3/tts/unidirectional/stream",
            additional_headers=headers,
            open_timeout=30,
        ) as ws:
            await ws.send(_build_ws_send_text(body))
            while True:
                message = await asyncio.wait_for(ws.recv(), timeout=30)
                event, payload = _parse_ws_frame(message)
                if event == 352 and payload:
                    handle.write(payload)
                    total += len(payload)
                elif event in (152, 351):
                    break
            try:
                await ws.send(_build_ws_finish_connection())
            except Exception:
                pass
    return total


def save_seed_ws_standard(api_key: str, text: str, output: Path) -> int:
    return asyncio.run(_save_seed_ws_standard(api_key, text, output))


def main() -> None:
    parser = argparse.ArgumentParser(description="Save TTS sample audio to disk.")
    parser.add_argument("--text", default=DEFAULT_TEXT)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--fish-api-key", default=os.getenv("FISH_API_KEY"))
    parser.add_argument(
        "--volcengine-tts-api-key",
        default=os.getenv("VOLCENGINE_TTS_API_KEY"),
    )
    args = parser.parse_args()

    if not args.fish_api_key:
        raise RuntimeError("Set FISH_API_KEY or pass --fish-api-key.")
    if not args.volcengine_tts_api_key:
        raise RuntimeError("Set VOLCENGINE_TTS_API_KEY or pass --volcengine-tts-api-key.")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    jobs = [
        ("fish_stream.mp3", lambda path: save_fish_stream(args.fish_api_key, args.text, path)),
        (
            "seed_http_chunked_standard.mp3",
            lambda path: save_seed_http_chunked(args.volcengine_tts_api_key, args.text, path),
        ),
        (
            "seed_ws_standard.mp3",
            lambda path: save_seed_ws_standard(args.volcengine_tts_api_key, args.text, path),
        ),
    ]

    failures: list[str] = []
    for filename, runner in jobs:
        output = args.output_dir / filename
        try:
            size = runner(output)
            print(f"{output} ({size} bytes)")
        except Exception as exc:
            failures.append(f"{filename}: {exc}")
            if output.exists() and output.stat().st_size == 0:
                output.unlink()
            print(f"FAILED {filename}: {exc}")

    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
