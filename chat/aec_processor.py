"""WebRTC AEC3 wrapper used to suppress speaker→mic acoustic echo.

The module owns a single `pywebrtc_audio.AudioProcessor` instance plus a
thread-safe ring buffer for the far-side reference signal. Two threads touch
it: the speaker writer thread pushes the PCM it just sent to the audio driver
(resampled to the AEC sample rate), and the mic input callback pops a matching
chunk and runs `process()` to produce echo-suppressed audio.

`stream_delay_ms=0` puts AEC3 in DELAY_AGNOSTIC mode: the algorithm tracks the
real round-trip delay between far and near via cross-correlation, so we don't
need to measure macOS's CoreAudio latency manually. It tolerates ±200ms of
drift (e.g. AirPods vs. built-in speakers) without retuning.
"""

from __future__ import annotations

import logging
import threading

import numpy as np

LOGGER = logging.getLogger(__name__)

try:
    from pywebrtc_audio import AudioProcessor
except ImportError:  # pragma: no cover - exercised when the wheel is missing
    AudioProcessor = None  # type: ignore[assignment]


class AecProcessor:
    """Echo cancellation + noise suppression for the mic capture stream."""

    @staticmethod
    def is_available() -> bool:
        return AudioProcessor is not None

    def __init__(
        self,
        *,
        sample_rate: int = 16000,
        max_buffer_ms: int = 1000,
        ns_level: int = 2,
        stream_delay_ms: int = 0,
    ) -> None:
        if AudioProcessor is None:
            raise RuntimeError(
                "pywebrtc-audio is not installed; install it or unset ENABLE_AEC."
            )
        self._ap = AudioProcessor(
            sample_rate=sample_rate,
            num_channels=1,
            echo_cancellation=True,
            noise_suppression=True,
            high_pass_filter=True,
            auto_gain_control=False,
            ns_level=ns_level,
            stream_delay_ms=stream_delay_ms,
        )
        self._sample_rate = sample_rate
        self._max_far_samples = sample_rate * max_buffer_ms // 1000
        self._far_buf = np.zeros(0, dtype=np.int16)
        self._lock = threading.Lock()

    @property
    def sample_rate(self) -> int:
        return self._sample_rate

    def push_far(self, pcm_int16: np.ndarray) -> None:
        """Append speaker-side reference PCM (already at AEC sample rate)."""
        if pcm_int16.dtype != np.int16:
            pcm_int16 = pcm_int16.astype(np.int16, copy=False)
        with self._lock:
            self._far_buf = np.concatenate((self._far_buf, pcm_int16))
            overflow = len(self._far_buf) - self._max_far_samples
            if overflow > 0:
                # Trim the oldest samples; keeping only ~1s of history is plenty
                # for AEC3 to track the playback path without unbounded growth
                # if the mic stream stalls for any reason.
                self._far_buf = self._far_buf[overflow:]

    def process(self, near_int16: np.ndarray) -> np.ndarray:
        """Return echo-cancelled mic PCM matching the input length."""
        if near_int16.dtype != np.int16:
            near_int16 = near_int16.astype(np.int16, copy=False)
        n = len(near_int16)
        with self._lock:
            available = len(self._far_buf)
            if available >= n:
                far = self._far_buf[:n].copy()
                self._far_buf = self._far_buf[n:]
            else:
                far = np.zeros(n, dtype=np.int16)
                if available:
                    far[:available] = self._far_buf
                    self._far_buf = np.zeros(0, dtype=np.int16)
            return self._ap.process(near_int16, far)

    def clear_far(self) -> None:
        """Drop pending far-side reference samples without resetting AEC state.

        Called after `SpeakerSink.flush_and_restart` aborts queued playback.
        We deliberately keep AEC3's adapted filter coefficients so the canceller
        doesn't have to re-converge (~1s) on every barge-in; only the unplayed
        reference samples that no longer correspond to anything coming out of
        the speaker are discarded.
        """
        with self._lock:
            self._far_buf = np.zeros(0, dtype=np.int16)

    def reset(self) -> None:
        """Drop far-side state AND flush AEC3 filter coefficients.

        Heavy-handed: forces full re-convergence on the next playback. Only
        appropriate when the acoustic path itself changes (e.g. switching
        between built-in speakers and a Bluetooth headset). Routine barge-ins
        should call `clear_far()` instead.
        """
        with self._lock:
            self._far_buf = np.zeros(0, dtype=np.int16)
        try:
            self._ap.reset()
        except Exception as exc:
            LOGGER.warning("AEC reset failed: %s", exc)
