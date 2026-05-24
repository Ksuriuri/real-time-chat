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
    """Play queued int16 PCM at 24 kHz with low-latency barge-in support."""

    def __init__(self, *, aec: AecProcessor | None = None) -> None:
        self._stream: sd.RawOutputStream | None = None
        self._queue: deque[bytes] = deque()
        self._cv = threading.Condition()
        self._writer: threading.Thread | None = None
        self._stop = False
        self._idle = threading.Event()
        self._idle.set()
        # When set, every chunk written to the speaker is mirrored (resampled
        # to the AEC rate) into the AEC far-side buffer so the canceller has
        # a reference signal of what should be appearing as echo in the mic.
        self._aec = aec

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
        # Drop only the unplayed far-side samples; keep AEC3's adapted filter
        # so the next utterance benefits from the convergence we already paid
        # for. A full reset() here would force ~1s of re-convergence and lets
        # echo bleed through into the ASR session that just opened.
        if self._aec is not None:
            self._aec.clear_far()
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
                self._write_with_far(stream, chunk)
            except Exception as exc:
                LOGGER.warning("speaker write failed: %s", exc)
                return

    def _write_with_far(self, stream: sd.RawOutputStream, chunk: bytes) -> None:
        """Write ``chunk`` to the speaker and mirror it to the AEC far side.

        Volcengine TTS streams ~400 ms PCM payloads, but macOS CoreAudio's
        output buffer is closer to 50–100 ms, so a single ``stream.write`` of
        the whole payload blocks for hundreds of ms while the OS plays out
        what was already queued. If we pushed the far reference only after
        that ``write`` returned, AEC's reference for the *first* part of the
        payload would arrive long after the mic has already captured the
        corresponding echo — and AEC3 can't cancel echo whose reference shows
        up late (drops from ~30 dB suppression to ~1 dB in our tests). The
        consequence is the assistant's TTS bleeds into ASR and gets
        re-transcribed as user input.

        To keep the reference in step with playback, walk the payload in
        20 ms slices and push each slice's far reference *before* handing it
        to the audio driver. ``stream.write`` paces us naturally once the OS
        buffer is full, so far/near stay aligned to within a single driver
        block instead of an entire TTS payload.
        """
        if self._aec is None:
            stream.write(chunk)
            return

        try:
            pcm_24k = np.frombuffer(chunk, dtype="<i2")
            far_aec = _resample_to_aec(pcm_24k, self._aec.sample_rate)
        except Exception as exc:
            LOGGER.warning(
                "AEC far-resample failed (%s); writing without reference",
                exc,
            )
            stream.write(chunk)
            return

        slice_bytes = SPEAKER_BLOCK_SIZE * 2  # 20 ms at 24 kHz, int16
        slice_aec = self._aec.sample_rate * SPEAKER_BLOCK_SIZE // SPEAKER_SAMPLE_RATE
        total_bytes = len(chunk)
        far_total = len(far_aec)
        pos_b = 0
        pos_aec = 0
        while pos_b < total_bytes:
            end_b = min(pos_b + slice_bytes, total_bytes)
            end_aec = min(pos_aec + slice_aec, far_total)
            if end_aec > pos_aec:
                self._aec.push_far(far_aec[pos_aec:end_aec])
                pos_aec = end_aec
            stream.write(chunk[pos_b:end_b])
            pos_b = end_b
        # Resampling can yield a few stray samples beyond what we already
        # accounted for; flush them so the far buffer doesn't drift behind.
        if pos_aec < far_total:
            self._aec.push_far(far_aec[pos_aec:])


def pcm_bytes_to_seconds(pcm: bytes, sample_rate: int = SPEAKER_SAMPLE_RATE) -> float:
    return len(pcm) / 2 / sample_rate


def silence_pcm(duration_ms: int, sample_rate: int = MIC_SAMPLE_RATE) -> bytes:
    samples = int(sample_rate * duration_ms / 1000)
    return np.zeros(samples, dtype="<i2").tobytes()
