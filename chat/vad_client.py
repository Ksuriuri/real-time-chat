"""Long-lived WebSocket client to the local Silero VAD service."""

from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass
from typing import Any
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

import websockets


LOGGER = logging.getLogger(__name__)


@dataclass
class VadEvent:
    kind: str  # "speech_start" | "speech_end" | "ready" | "error" | "closed"
    sample: int | None = None
    seconds: float | None = None
    detail: str | None = None


def _with_query(url: str, params: dict[str, Any]) -> str:
    parts = urlsplit(url)
    query = dict(parse_qsl(parts.query, keep_blank_values=True))
    query.update({k: str(v) for k, v in params.items() if v is not None})
    return urlunsplit(
        (parts.scheme, parts.netloc, parts.path, urlencode(query), parts.fragment)
    )


class VadClient:
    """Stream PCM to the local VAD WS and surface speech_start/end events."""

    def __init__(
        self,
        *,
        url: str,
        api_key: str | None = None,
        threshold: float | None = None,
        min_speech_ms: int | None = None,
        min_silence_ms: int | None = None,
        speech_pad_ms: int | None = None,
        min_gap_ms: int | None = None,
    ) -> None:
        self._url = _with_query(
            url,
            {
                "threshold": threshold,
                "min_speech_ms": min_speech_ms,
                "min_silence_ms": min_silence_ms,
                "speech_pad_ms": speech_pad_ms,
                "min_gap_ms": min_gap_ms,
            },
        )
        self._headers = {"Authorization": f"Bearer {api_key}"} if api_key else None
        self._ws: Any | None = None
        self._recv_task: asyncio.Task[None] | None = None
        self.events: asyncio.Queue[VadEvent] = asyncio.Queue()

    async def connect(self) -> None:
        try:
            self._ws = await websockets.connect(
                self._url,
                additional_headers=self._headers,
                ping_interval=20,
            )
        except TypeError:
            self._ws = await websockets.connect(
                self._url,
                extra_headers=self._headers,
                ping_interval=20,
            )
        self._recv_task = asyncio.create_task(
            self._recv_loop(), name="vad-recv-loop"
        )
        LOGGER.info("Connected to VAD: %s", self._url)

    async def send_pcm(self, pcm: bytes) -> None:
        if self._ws is None:
            return
        try:
            await self._ws.send(pcm)
        except websockets.ConnectionClosed:
            await self.events.put(VadEvent(kind="closed"))

    async def reset(self) -> None:
        if self._ws is None:
            return
        try:
            await self._ws.send(json.dumps({"event": "reset"}))
        except websockets.ConnectionClosed:
            await self.events.put(VadEvent(kind="closed"))

    async def close(self) -> None:
        if self._recv_task is not None:
            self._recv_task.cancel()
            try:
                await self._recv_task
            except (asyncio.CancelledError, Exception):
                pass
            self._recv_task = None
        if self._ws is not None:
            try:
                await self._ws.close()
            finally:
                self._ws = None

    async def _recv_loop(self) -> None:
        assert self._ws is not None
        try:
            async for raw in self._ws:
                try:
                    payload = json.loads(raw)
                except (TypeError, ValueError):
                    continue
                event = payload.get("event")
                if event == "speech_start":
                    await self.events.put(
                        VadEvent(
                            kind="speech_start",
                            sample=int(payload.get("start", 0)),
                            seconds=payload.get("start_s"),
                        )
                    )
                elif event == "speech_end":
                    await self.events.put(
                        VadEvent(
                            kind="speech_end",
                            sample=int(payload.get("end", 0)),
                            seconds=payload.get("end_s"),
                        )
                    )
                elif event == "error":
                    await self.events.put(
                        VadEvent(kind="error", detail=str(payload.get("detail")))
                    )
                elif event == "ready":
                    await self.events.put(VadEvent(kind="ready"))
        except websockets.ConnectionClosed as exc:
            LOGGER.info("VAD WS closed: %s", exc)
        finally:
            await self.events.put(VadEvent(kind="closed"))
