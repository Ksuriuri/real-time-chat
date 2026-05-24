"""Volcengine ASR 2.0 bidirectional streaming session.

Wire protocol mirrors ``asr/volcengine_streaming_asr.py``: a 4-byte header
followed by an optional sequence and a gzipped JSON or PCM payload. We expose
an async session that can be reused per utterance.
"""

from __future__ import annotations

import asyncio
import gzip
import json
import logging
import time
import uuid
from dataclasses import dataclass
from typing import Any

import websockets


LOGGER = logging.getLogger(__name__)

WS_URL = "wss://openspeech.bytedance.com/api/v3/sauc/bigmodel"
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


def _make_header(
    message_type: int, flags: int, serialization: int, compression: int
) -> bytes:
    return bytes(
        [
            0x11,
            (message_type << 4) | flags,
            (serialization << 4) | compression,
            0x00,
        ]
    )


def _pack_full_client_request(payload: dict[str, Any]) -> bytes:
    body = gzip.compress(json.dumps(payload, ensure_ascii=False).encode("utf-8"))
    return (
        _make_header(MSG_FULL_CLIENT, FLAG_NO_SEQ, SER_JSON, COMP_GZIP)
        + len(body).to_bytes(4, "big")
        + body
    )


def _pack_audio_request(chunk: bytes, sequence: int, *, last: bool) -> bytes:
    body = gzip.compress(chunk)
    flags = FLAG_NEG_SEQ if last else FLAG_POS_SEQ
    wire_sequence = -sequence if last else sequence
    return (
        _make_header(MSG_AUDIO_ONLY, flags, SER_NONE, COMP_GZIP)
        + int(wire_sequence).to_bytes(4, "big", signed=True)
        + len(body).to_bytes(4, "big")
        + body
    )


def _decode_frame(data: bytes) -> dict[str, Any]:
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


@dataclass
class AsrResult:
    text: str
    request_id: str
    duration_s: float
    first_text_s: float | None
    final_s: float | None


class StreamingAsrSession:
    """One short-lived Volcengine ASR streaming session per utterance."""

    def __init__(
        self,
        *,
        api_key: str,
        resource_id: str,
        chunk_ms: int = 200,
    ) -> None:
        self._api_key = api_key
        self._resource_id = resource_id
        self._chunk_bytes = TARGET_RATE * 2 * chunk_ms // 1000
        self._ws: Any | None = None
        # Server auto-assigns sequence=1 to the initial full-client JSON
        # request, so the first audio frame must start at 2.
        self._sequence = 2
        self._buffer = bytearray()
        self._send_lock = asyncio.Lock()
        self._receiver: asyncio.Task[str] | None = None
        self._latest_text = ""
        self._first_text_s: float | None = None
        # Set the first time the server returns a non-empty partial text. The
        # orchestrator uses this as a "real speech confirmed" signal to gate
        # barge-in against acoustic-echo false triggers.
        self.partial_text_event = asyncio.Event()
        self._final_s: float | None = None
        self._start_time: float | None = None
        self._send_done_s: float | None = None
        self.request_id = str(uuid.uuid4())

    async def start(self) -> None:
        api_key = (self._api_key or "").strip()
        resource_id = (self._resource_id or "").strip()
        if not api_key or not resource_id:
            raise RuntimeError(
                "ASR session missing api_key or resource_id; check VOLCENGINE_ASR_*."
            )

        headers = {
            "X-Api-Key": api_key,
            "X-Api-Resource-Id": resource_id,
            "X-Api-Request-Id": self.request_id,
            "X-Api-Connect-Id": self.request_id,
            "X-Api-Sequence": "-1",
        }
        try:
            try:
                self._ws = await websockets.connect(
                    WS_URL,
                    additional_headers=headers,
                    max_size=1_000_000_000,
                    ping_interval=None,
                )
            except TypeError:
                self._ws = await websockets.connect(
                    WS_URL,
                    extra_headers=headers,
                    max_size=1_000_000_000,
                    ping_interval=None,
                )
        except websockets.InvalidStatus as exc:
            tail = api_key[-4:] if len(api_key) >= 4 else api_key
            raise RuntimeError(
                f"Volcengine ASR rejected handshake: {exc}. "
                f"Sent X-Api-Key (len={len(api_key)} tail=…{tail}) "
                f"and X-Api-Resource-Id={resource_id!r}. "
                "Verify VOLCENGINE_ASR_API_KEY (no whitespace, no quotes, real value)."
            ) from exc

        self._start_time = time.perf_counter()
        await self._ws.send(_pack_full_client_request(self._build_request()))
        self._receiver = asyncio.create_task(
            self._recv_loop(), name=f"asr-recv-{self.request_id[:8]}"
        )
        LOGGER.debug("ASR session %s started", self.request_id)

    async def send_pcm(self, pcm: bytes) -> None:
        if not pcm or self._ws is None:
            return
        self._buffer.extend(pcm)
        while len(self._buffer) >= self._chunk_bytes:
            chunk = bytes(self._buffer[: self._chunk_bytes])
            del self._buffer[: self._chunk_bytes]
            await self._send_chunk(chunk, last=False)

    async def finalize(self, *, timeout_s: float = 2.0) -> AsrResult:
        if self._ws is None or self._receiver is None or self._start_time is None:
            return AsrResult(
                text="",
                request_id=self.request_id,
                duration_s=0.0,
                first_text_s=None,
                final_s=None,
            )

        tail = bytes(self._buffer)
        self._buffer.clear()
        await self._send_chunk(tail, last=True)
        self._send_done_s = time.perf_counter() - self._start_time

        try:
            text = await asyncio.wait_for(self._receiver, timeout=timeout_s)
        except asyncio.TimeoutError:
            LOGGER.warning(
                "ASR finalize timed out after %.2fs, falling back to last partial",
                timeout_s,
            )
            text = self._latest_text
            self._receiver.cancel()

        return AsrResult(
            text=text,
            request_id=self.request_id,
            duration_s=round(self._send_done_s or 0.0, 3),
            first_text_s=(
                round(self._first_text_s, 3) if self._first_text_s is not None else None
            ),
            final_s=(round(self._final_s, 3) if self._final_s is not None else None),
        )

    async def close(self) -> None:
        if self._receiver is not None and not self._receiver.done():
            self._receiver.cancel()
            try:
                await self._receiver
            except (asyncio.CancelledError, Exception):
                pass
        self._receiver = None
        if self._ws is not None:
            try:
                await self._ws.close()
            except Exception:
                pass
            self._ws = None

    async def _send_chunk(self, chunk: bytes, *, last: bool) -> None:
        assert self._ws is not None
        sequence = self._sequence
        self._sequence += 1
        frame = _pack_audio_request(chunk, sequence, last=last)
        async with self._send_lock:
            await self._ws.send(frame)

    async def _recv_loop(self) -> str:
        assert self._ws is not None and self._start_time is not None
        try:
            while True:
                raw = await self._ws.recv()
                frame = _decode_frame(raw)
                now_s = time.perf_counter() - self._start_time

                if frame.get("kind") == "error":
                    raise RuntimeError(
                        f"Volcengine ASR error {frame.get('code')}: {frame.get('payload')}"
                    )

                payload = frame.get("payload")
                result = payload.get("result") if isinstance(payload, dict) else None
                if isinstance(result, dict):
                    text = result.get("text") or ""
                    if text:
                        self._latest_text = text
                        if self._first_text_s is None:
                            self._first_text_s = now_s
                        self.partial_text_event.set()

                if frame.get("flags") in {FLAG_LAST_NO_SEQ, FLAG_NEG_SEQ}:
                    self._final_s = now_s
                    return self._latest_text
        except websockets.ConnectionClosed:
            return self._latest_text

    def _build_request(self) -> dict[str, Any]:
        return {
            "user": {"uid": "real-time-chat-orchestrator"},
            "audio": {
                "format": "pcm",
                "codec": "raw",
                "rate": TARGET_RATE,
                "bits": 16,
                "channel": 1,
            },
            "request": {
                "model_name": "bigmodel",
                "enable_itn": True,
                "enable_punc": True,
                "show_utterances": True,
                "result_type": "full",
            },
        }
