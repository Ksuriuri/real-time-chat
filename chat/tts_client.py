"""seed-tts-2.0 WebSocket client streaming raw PCM 24 kHz audio.

Frame layout matches ``tts/save_tts_samples.py``: one optional event id +
session id length, followed by a 4-byte payload length and the payload itself.
We request ``format=pcm`` so payloads can be fed directly to the speaker sink
without an MP3 decoder.
"""

from __future__ import annotations

import asyncio
import json
import logging
import struct
import uuid
from collections.abc import AsyncIterator
from typing import Any

import websockets


LOGGER = logging.getLogger(__name__)

WS_URL = "wss://openspeech.bytedance.com/api/v3/tts/unidirectional/stream"

EVENT_AUDIO_CHUNK = 352
EVENT_SESSION_FINISHED = 152
EVENT_TTS_FINISHED = 351

DEFAULT_SAMPLE_RATE = 24000


def _build_send_text_frame(payload: dict[str, Any]) -> bytes:
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    return bytes([0x11, 0x10, 0x10, 0x00]) + struct.pack(">I", len(body)) + body


def _build_finish_connection_frame() -> bytes:
    return bytes([0x11, 0x14, 0x10, 0x00]) + struct.pack(">I", 2)


def _parse_frame(data: bytes) -> tuple[int | None, bytes]:
    header_size = (data[0] & 0x0F) * 4
    flags = data[1] & 0x0F
    pos = header_size
    event: int | None = None
    if flags & 0x04:
        event = struct.unpack(">I", data[pos : pos + 4])[0]
        pos += 4
        sid_len = struct.unpack(">I", data[pos : pos + 4])[0]
        pos += 4 + sid_len
    payload_size = struct.unpack(">I", data[pos : pos + 4])[0]
    pos += 4
    return event, data[pos : pos + payload_size]


class SeedTtsClient:
    """Stream PCM audio for one text segment per call."""

    def __init__(
        self,
        *,
        api_key: str,
        resource_id: str = "seed-tts-2.0",
        speaker: str,
        model: str = "seed-tts-2.0-standard",
        sample_rate: int = DEFAULT_SAMPLE_RATE,
        recv_timeout_s: float = 10.0,
    ) -> None:
        self._api_key = api_key
        self._resource_id = resource_id
        self._speaker = speaker
        self._model = model
        self._sample_rate = sample_rate
        self._recv_timeout_s = recv_timeout_s

    @property
    def sample_rate(self) -> int:
        return self._sample_rate

    async def synthesize(
        self,
        text: str,
        *,
        cancel_event: asyncio.Event | None = None,
    ) -> AsyncIterator[bytes]:
        """Yield raw PCM int16 chunks for the given text segment."""
        if not text.strip():
            return
        async for chunk in self._run(text, cancel_event):
            yield chunk

    async def _run(
        self, text: str, cancel_event: asyncio.Event | None
    ) -> AsyncIterator[bytes]:
        api_key = (self._api_key or "").strip()
        resource_id = (self._resource_id or "").strip()
        if not api_key or not resource_id:
            raise RuntimeError(
                "TTS client missing api_key or resource_id; check VOLCENGINE_TTS_*."
            )

        headers = {
            "X-Api-Key": api_key,
            "X-Api-Resource-Id": resource_id,
            "X-Api-Connect-Id": str(uuid.uuid4()),
        }
        body = {
            "user": {"uid": "real-time-chat-orchestrator"},
            "req_params": {
                "text": text,
                "speaker": self._speaker,
                "model": self._model,
                "audio_params": {"format": "pcm", "sample_rate": self._sample_rate},
            },
        }

        try:
            try:
                ws = await websockets.connect(
                    WS_URL, additional_headers=headers, open_timeout=15
                )
            except TypeError:
                ws = await websockets.connect(
                    WS_URL, extra_headers=headers, open_timeout=15
                )
        except websockets.InvalidStatus as exc:
            tail = api_key[-4:] if len(api_key) >= 4 else api_key
            raise RuntimeError(
                f"Volcengine TTS rejected handshake: {exc}. "
                f"Sent X-Api-Key (len={len(api_key)} tail=…{tail}) "
                f"and X-Api-Resource-Id={resource_id!r}. "
                "Verify VOLCENGINE_TTS_API_KEY."
            ) from exc

        async with ws:
            await ws.send(_build_send_text_frame(body))
            while True:
                if cancel_event is not None and cancel_event.is_set():
                    LOGGER.debug("TTS cancelled before completion")
                    break
                try:
                    raw = await asyncio.wait_for(
                        ws.recv(), timeout=self._recv_timeout_s
                    )
                except asyncio.TimeoutError:
                    LOGGER.warning("TTS recv timed out, closing session")
                    break
                except websockets.ConnectionClosed as exc:
                    LOGGER.info("TTS WS closed: %s", exc)
                    break

                event, payload = _parse_frame(raw)
                if event == EVENT_AUDIO_CHUNK and payload:
                    yield payload
                elif event in (EVENT_SESSION_FINISHED, EVENT_TTS_FINISHED):
                    break

            try:
                await ws.send(_build_finish_connection_frame())
            except Exception:
                pass
