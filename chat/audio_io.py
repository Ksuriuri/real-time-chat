"""sounddevice-backed microphone input and speaker output.

The orchestrator runs in asyncio while sounddevice callbacks fire on a separate
audio thread. We bridge them with ``loop.call_soon_threadsafe`` so PCM frames
arrive as awaitables on the asyncio side. Playback uses a callback-driven
output stream so the stream stays open for the lifetime of the process and
barge-in just clears a shared byte buffer.
"""

from __future__ import annotations

import asyncio
import logging
import threading
from collections.abc import Awaitable, Callable
from typing import Any

import numpy as np
import sounddevice as sd

from aec_processor import AecProcessor


LOGGER = logging.getLogger(__name__)

MIC_SAMPLE_RATE = 16000
MIC_BLOCK_SIZE = 512  # 32 ms at 16 kHz, matches Silero VAD chunk size.
SPEAKER_SAMPLE_RATE = 24000
SPEAKER_BLOCK_SIZE = 480  # 20 ms at 24 kHz.

PcmHandler = Callable[[bytes], Awaitable[None]]


def _resample_to_aec(pcm_24k: np.ndarray, target_rate: int) -> np.ndarray:
    """Downsample 24 kHz speaker PCM to the AEC sample rate (typically 16k).

    Imported lazily so the module is still usable when soxr isn't available
    (the AEC integration path is the only place the resampler is needed).
    """
    import soxr

    return soxr.resample(pcm_24k, SPEAKER_SAMPLE_RATE, target_rate)


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
        aec: AecProcessor | None = None,
    ) -> None:
        self._loop = loop
        self._on_pcm = on_pcm
        self._preroll = RingBuffer(
            max_bytes=int(MIC_SAMPLE_RATE * preroll_ms / 1000) * 2
        )
        self._stream: sd.RawInputStream | None = None
        # When set, AEC runs synchronously inside the mic callback so VAD,
        # ASR and the preroll buffer all see echo-suppressed audio.
        self._aec = aec

    @property
    def preroll(self) -> RingBuffer:
        return self._preroll

    def start(self) -> None:
        if self._stream is not None:
            return

        aec = self._aec

        def _callback(
            indata: Any,
            _frames: int,
            _time_info: Any,
            status: sd.CallbackFlags,
        ) -> None:
            if status:
                LOGGER.warning("mic callback status: %s", status)
            if aec is not None:
                # Run AEC inline: convert to int16 numpy view, process, then
                # serialize back to bytes for downstream consumers. The audio
                # callback runs on a real-time priority thread, so we keep the
                # work O(frames) and let pywebrtc-audio release the GIL.
                near = np.frombuffer(indata, dtype="<i2")
                clean = aec.process(near)
                chunk = clean.tobytes()
            else:
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
            "Microphone started: %d Hz mono int16, blocksize=%d, aec=%s",
            MIC_SAMPLE_RATE,
            MIC_BLOCK_SIZE,
            "on" if aec is not None else "off",
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
    """Play queued int16 PCM at 24 kHz with low-latency barge-in support.

    The output stream is callback-driven and stays open for the lifetime of
    the process. Barge-in just clears the pending byte buffer and the AEC far
    reference; the callback immediately starts emitting silence until the
    next chunk lands. This avoids the close+reopen pattern that occasionally
    trips macOS CoreAudio with ``PaErrorCode -9986`` ("Invalid Property
    Value") when two barge-ins land in quick succession.
    """

    def __init__(self, *, aec: AecProcessor | None = None) -> None:
        self._stream: sd.RawOutputStream | None = None
        # Single lock guards both the pending playback buffer AND the AEC
        # far-reference ordering: we need flush()'s clear_far() to never be
        # immediately followed by a stale push_far() from an in-flight
        # callback, otherwise echo from the *previous* utterance leaks past
        # the canceller right when we start listening to the user.
        self._lock = threading.Lock()
        self._buf = bytearray()
        self._idle = threading.Event()
        self._idle.set()
        # When set, every block the callback hands to the driver is mirrored
        # (resampled to the AEC rate) into the AEC far-side buffer so the
        # canceller has a reference signal of what is appearing as echo on
        # the mic.
        self._aec = aec

    def start(self) -> None:
        if self._stream is not None:
            return
        self._idle.set()
        self._stream = sd.RawOutputStream(
            samplerate=SPEAKER_SAMPLE_RATE,
            channels=1,
            dtype="int16",
            blocksize=SPEAKER_BLOCK_SIZE,
            callback=self._callback,
        )
        self._stream.start()
        LOGGER.info(
            "Speaker started: %d Hz mono int16, blocksize=%d",
            SPEAKER_SAMPLE_RATE,
            SPEAKER_BLOCK_SIZE,
        )

    def enqueue(self, pcm: bytes) -> None:
        if not pcm:
            return
        with self._lock:
            self._buf.extend(pcm)
            self._idle.clear()

    def is_active(self) -> bool:
        return not self._idle.is_set()

    async def wait_drained(self) -> None:
        await asyncio.get_running_loop().run_in_executor(None, self._idle.wait)

    def flush(self) -> None:
        """Discard pending playback audio and the matching AEC far reference.

        Called on barge-in. The CoreAudio stream is *not* torn down — the
        callback simply starts producing silence once ``_buf`` is empty. We
        deliberately keep AEC3's adapted filter (only ``clear_far`` here, not
        ``reset``) so the canceller doesn't need to re-converge (~1s) every
        time the user interrupts.
        """
        with self._lock:
            self._buf.clear()
            self._idle.set()
            if self._aec is not None:
                self._aec.clear_far()

    def stop(self) -> None:
        if self._stream is not None:
            try:
                self._stream.stop()
                self._stream.close()
            except Exception as exc:
                LOGGER.warning("speaker stop failed: %s", exc)
            finally:
                self._stream = None
        with self._lock:
            self._buf.clear()
            self._idle.set()

    def _callback(
        self,
        outdata: Any,
        frames: int,
        _time_info: Any,
        status: sd.CallbackFlags,
    ) -> None:
        """Fill ``outdata`` from ``_buf``, pad with silence, mirror to AEC.

        Runs on PortAudio's real-time audio thread, so we keep the work
        O(frames): pop bytes from the front of ``_buf``, zero-pad the tail
        when we don't have enough, and push the just-produced samples into
        the AEC far buffer *before* the driver actually plays them. That
        keeps far/near aligned to within a single 20 ms block instead of an
        entire TTS payload, which is what AEC3 needs to maintain its ~30 dB
        suppression on macOS where CoreAudio's own buffer is 50–100 ms.
        """
        if status:
            LOGGER.warning("speaker callback status: %s", status)
        needed = frames * 2  # int16 mono
        with self._lock:
            avail = min(len(self._buf), needed)
            if avail > 0:
                played = bytes(self._buf[:avail])
                del self._buf[:avail]
                outdata[:avail] = played
            if avail < needed:
                outdata[avail:needed] = b"\x00" * (needed - avail)
            if self._aec is not None and avail > 0:
                try:
                    pcm_24k = np.frombuffer(played, dtype="<i2")
                    far_aec = _resample_to_aec(pcm_24k, self._aec.sample_rate)
                    self._aec.push_far(far_aec)
                except Exception as exc:
                    LOGGER.warning("AEC far push failed: %s", exc)
            if not self._buf:
                self._idle.set()


def pcm_bytes_to_seconds(pcm: bytes, sample_rate: int = SPEAKER_SAMPLE_RATE) -> float:
    return len(pcm) / 2 / sample_rate


def silence_pcm(duration_ms: int, sample_rate: int = MIC_SAMPLE_RATE) -> bytes:
    samples = int(sample_rate * duration_ms / 1000)
    return np.zeros(samples, dtype="<i2").tobytes()
