"""sounddevice-backed microphone input and speaker output.

The orchestrator runs in asyncio while sounddevice callbacks fire on a separate
audio thread. We bridge them with ``loop.call_soon_threadsafe`` so PCM frames
arrive as awaitables on the asyncio side, and we keep playback drained by a
dedicated background writer thread.
"""

from __future__ import annotations

import asyncio
import logging
import threading
from collections import deque
from collections.abc import Awaitable, Callable
from typing import Any

import numpy as np
import sounddevice as sd


LOGGER = logging.getLogger(__name__)

MIC_SAMPLE_RATE = 16000
MIC_BLOCK_SIZE = 512  # 32 ms at 16 kHz, matches Silero VAD chunk size.
SPEAKER_SAMPLE_RATE = 24000
SPEAKER_BLOCK_SIZE = 480  # 20 ms at 24 kHz.

PcmHandler = Callable[[bytes], Awaitable[None]]


class RingBuffer:
    """Fixed-size byte ring buffer for keeping recent mic audio."""

    def __init__(self, max_bytes: int) -> None:
        self._max_bytes = max_bytes
        self._buf = bytearray()
        self._lock = threading.Lock()

    def append(self, chunk: bytes) -> None:
        with self._lock:
            self._buf.extend(chunk)
            overflow = len(self._buf) - self._max_bytes
            if overflow > 0:
                del self._buf[:overflow]

    def snapshot(self) -> bytes:
        with self._lock:
            return bytes(self._buf)

    def clear(self) -> None:
        with self._lock:
            self._buf.clear()


class MicSource:
    """Capture mono int16 PCM at 16 kHz and forward to async handlers."""

    def __init__(
        self,
        *,
        loop: asyncio.AbstractEventLoop,
        on_pcm: PcmHandler,
        preroll_ms: int = 500,
    ) -> None:
        self._loop = loop
        self._on_pcm = on_pcm
        self._preroll = RingBuffer(
            max_bytes=int(MIC_SAMPLE_RATE * preroll_ms / 1000) * 2
        )
        self._stream: sd.RawInputStream | None = None

    @property
    def preroll(self) -> RingBuffer:
        return self._preroll

    def start(self) -> None:
        if self._stream is not None:
            return

        def _callback(
            indata: Any,
            _frames: int,
            _time_info: Any,
            status: sd.CallbackFlags,
        ) -> None:
            if status:
                LOGGER.warning("mic callback status: %s", status)
            chunk = bytes(indata)
            self._preroll.append(chunk)
            asyncio.run_coroutine_threadsafe(self._on_pcm(chunk), self._loop)

        self._stream = sd.RawInputStream(
            samplerate=MIC_SAMPLE_RATE,
            channels=1,
            dtype="int16",
            blocksize=MIC_BLOCK_SIZE,
            callback=_callback,
        )
        self._stream.start()
        LOGGER.info(
            "Microphone started: %d Hz mono int16, blocksize=%d",
            MIC_SAMPLE_RATE,
            MIC_BLOCK_SIZE,
        )

    def stop(self) -> None:
        if self._stream is None:
            return
        try:
            self._stream.stop()
            self._stream.close()
        finally:
            self._stream = None


class SpeakerSink:
    """Play queued int16 PCM at 24 kHz with low-latency barge-in support."""

    def __init__(self) -> None:
        self._stream: sd.RawOutputStream | None = None
        self._queue: deque[bytes] = deque()
        self._cv = threading.Condition()
        self._writer: threading.Thread | None = None
        self._stop = False
        self._idle = threading.Event()
        self._idle.set()

    def start(self) -> None:
        if self._stream is not None:
            return
        self._stop = False
        self._idle.set()
        self._stream = sd.RawOutputStream(
            samplerate=SPEAKER_SAMPLE_RATE,
            channels=1,
            dtype="int16",
            blocksize=SPEAKER_BLOCK_SIZE,
        )
        self._stream.start()
        self._writer = threading.Thread(
            target=self._run_writer, name="SpeakerWriter", daemon=True
        )
        self._writer.start()
        LOGGER.info(
            "Speaker started: %d Hz mono int16, blocksize=%d",
            SPEAKER_SAMPLE_RATE,
            SPEAKER_BLOCK_SIZE,
        )

    def enqueue(self, pcm: bytes) -> None:
        if not pcm:
            return
        with self._cv:
            self._queue.append(pcm)
            self._idle.clear()
            self._cv.notify()

    def is_active(self) -> bool:
        return not self._idle.is_set()

    async def wait_drained(self) -> None:
        await asyncio.get_running_loop().run_in_executor(None, self._idle.wait)

    def flush_and_restart(self) -> None:
        """Discard queued audio, abort the current stream, and start a fresh one."""
        self._teardown_writer()
        if self._stream is not None:
            try:
                self._stream.abort()
                self._stream.close()
            except Exception as exc:
                LOGGER.warning("speaker abort failed: %s", exc)
            finally:
                self._stream = None
        self.start()

    def stop(self) -> None:
        self._teardown_writer()
        if self._stream is not None:
            try:
                self._stream.stop()
                self._stream.close()
            except Exception as exc:
                LOGGER.warning("speaker stop failed: %s", exc)
            finally:
                self._stream = None

    def _teardown_writer(self) -> None:
        with self._cv:
            self._stop = True
            self._queue.clear()
            self._cv.notify_all()
        if self._writer is not None:
            self._writer.join(timeout=1.0)
            self._writer = None
        self._idle.set()

    def _run_writer(self) -> None:
        while True:
            with self._cv:
                while not self._queue and not self._stop:
                    self._idle.set()
                    self._cv.wait()
                if self._stop:
                    return
                chunk = self._queue.popleft()
                self._idle.clear()
            stream = self._stream
            if stream is None:
                continue
            try:
                stream.write(chunk)
            except Exception as exc:
                LOGGER.warning("speaker write failed: %s", exc)
                return


def pcm_bytes_to_seconds(pcm: bytes, sample_rate: int = SPEAKER_SAMPLE_RATE) -> float:
    return len(pcm) / 2 / sample_rate


def silence_pcm(duration_ms: int, sample_rate: int = MIC_SAMPLE_RATE) -> bytes:
    samples = int(sample_rate * duration_ms / 1000)
    return np.zeros(samples, dtype="<i2").tobytes()
