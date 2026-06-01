"""Fish Audio TTS client streaming raw PCM 24 kHz audio.

Drop-in alternative to :class:`tts_client.SeedTtsClient`. It exposes the same
``synthesize(text, *, cancel_event) -> AsyncIterator[bytes]`` contract so the
orchestrator can switch providers without any other change.

We request ``format="pcm"`` with ``sample_rate=24000`` so payloads are mono
int16 PCM at the speaker's native rate and can be fed straight to
``SpeakerSink`` with no MP3 decode or resample step.

Voice cloning uses Fish's zero-shot ``references`` field: a local audio sample
(WAV/MP3/FLAC) plus its transcript. Accuracy of the transcript matters for
clone quality. ``reference_id`` (a hosted voice model) is supported as a
fallback when no local sample is configured.
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

import httpx
import ormsgpack


LOGGER = logging.getLogger(__name__)

TTS_URL = "https://api.fish.audio/v1/tts"

DEFAULT_SAMPLE_RATE = 24000


class FishTtsClient:
    """Stream PCM audio for one text segment per call via Fish Audio."""

    def __init__(
        self,
        *,
        api_key: str,
        model: str = "s1",
        reference_audio_path: str | Path | None = None,
        reference_text: str = "",
        reference_id: str | None = None,
        sample_rate: int = DEFAULT_SAMPLE_RATE,
        latency: str = "balanced",
        temperature: float = 0.7,
        top_p: float = 0.7,
        recv_timeout_s: float = 30.0,
    ) -> None:
        self._api_key = (api_key or "").strip()
        self._model = (model or "s1").strip()
        self._reference_id = (reference_id or "").strip() or None
        self._sample_rate = sample_rate
        self._latency = latency
        self._temperature = temperature
        self._top_p = top_p
        self._recv_timeout_s = recv_timeout_s

        # Load the local reference sample once. Zero-shot cloning sends the raw
        # bytes on every request, so we keep them resident instead of re-reading
        # the file per synthesis call.
        self._references: list[dict[str, Any]] = []
        if reference_audio_path:
            path = Path(reference_audio_path).expanduser()
            if not path.exists():
                raise RuntimeError(
                    f"FISH_REFERENCE_AUDIO points to a missing file: {path}"
                )
            audio_bytes = path.read_bytes()
            if not audio_bytes:
                raise RuntimeError(f"Reference audio is empty: {path}")
            self._references = [
                {"audio": audio_bytes, "text": reference_text or ""}
            ]
            LOGGER.info(
                "Fish TTS using local reference audio %s (%d bytes, transcript %s)",
                path,
                len(audio_bytes),
                "set" if reference_text else "empty",
            )
        elif self._reference_id:
            LOGGER.info("Fish TTS using hosted reference_id=%s", self._reference_id)
        else:
            LOGGER.warning(
                "Fish TTS has no reference audio or reference_id; using the "
                "model's default voice."
            )

    @property
    def sample_rate(self) -> int:
        return self._sample_rate

    async def synthesize(
        self,
        text: str,
        *,
        cancel_event: "Any | None" = None,
    ) -> AsyncIterator[bytes]:
        """Yield raw PCM int16 chunks for the given text segment."""
        if not text.strip():
            return
        async for chunk in self._run(text, cancel_event):
            yield chunk

    def _build_body(self, text: str) -> dict[str, Any]:
        body: dict[str, Any] = {
            "text": text,
            "format": "pcm",
            "sample_rate": self._sample_rate,
            "latency": self._latency,
            "temperature": self._temperature,
            "top_p": self._top_p,
            "normalize": True,
        }
        if self._references:
            body["references"] = self._references
        elif self._reference_id:
            body["reference_id"] = self._reference_id
        return body

    async def _run(
        self, text: str, cancel_event: "Any | None"
    ) -> AsyncIterator[bytes]:
        if not self._api_key:
            raise RuntimeError("Fish TTS client missing api_key; check FISH_API_KEY.")

        headers = {
            "authorization": f"Bearer {self._api_key}",
            "content-type": "application/msgpack",
            "model": self._model,
        }
        content = ormsgpack.packb(self._build_body(text))

        timeout = httpx.Timeout(self._recv_timeout_s, connect=15.0)
        # PCM frames are int16 little-endian; HTTP chunk boundaries can split a
        # sample, so carry a trailing odd byte over to the next yield to keep
        # everything 2-byte aligned for the speaker and AEC.
        carry = b""
        async with httpx.AsyncClient(timeout=timeout) as client:
            async with client.stream(
                "POST", TTS_URL, content=content, headers=headers
            ) as response:
                if response.status_code != 200:
                    detail = (await response.aread()).decode("utf-8", "replace")
                    tail = (
                        self._api_key[-4:]
                        if len(self._api_key) >= 4
                        else self._api_key
                    )
                    raise RuntimeError(
                        f"Fish TTS HTTP {response.status_code}: {detail[:300]}. "
                        f"Sent Authorization Bearer (len={len(self._api_key)} "
                        f"tail=…{tail}) model={self._model!r}. Verify FISH_API_KEY "
                        "and that the model/voice is enabled for your account."
                    )

                async for chunk in response.aiter_bytes():
                    if cancel_event is not None and cancel_event.is_set():
                        LOGGER.debug("Fish TTS cancelled before completion")
                        break
                    if not chunk:
                        continue
                    data = carry + chunk
                    aligned = len(data) - (len(data) % 2)
                    carry = data[aligned:]
                    if aligned:
                        yield data[:aligned]
