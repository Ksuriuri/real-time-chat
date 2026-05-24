from __future__ import annotations

import json
import os
import tempfile
import time
from pathlib import Path
from threading import Lock
from typing import Any

import librosa
import numpy as np
from fastapi import (
    Depends,
    FastAPI,
    File,
    Form,
    Header,
    HTTPException,
    UploadFile,
    WebSocket,
    WebSocketDisconnect,
)
from fastapi.concurrency import run_in_threadpool
from silero_vad import VADIterator, get_speech_timestamps, load_silero_vad


SAMPLE_RATE = int(os.getenv("VAD_SAMPLE_RATE", "16000"))
if SAMPLE_RATE not in {8000, 16000}:
    raise ValueError("VAD_SAMPLE_RATE must be either 8000 or 16000")

CHUNK_SIZE = 512 if SAMPLE_RATE == 16000 else 256
MODEL_RUNTIME = "silero-vad-onnx"
API_KEY = os.getenv("VAD_API_KEY")
PRELOAD_MODEL = os.getenv("VAD_PRELOAD", "true").lower() not in {"0", "false", "no"}
DEFAULT_THRESHOLD = float(os.getenv("VAD_THRESHOLD", "0.5"))
DEFAULT_MIN_SPEECH_MS = int(os.getenv("VAD_MIN_SPEECH_MS", "250"))
DEFAULT_MIN_SILENCE_MS = int(os.getenv("VAD_MIN_SILENCE_MS", "100"))
DEFAULT_SPEECH_PAD_MS = int(os.getenv("VAD_SPEECH_PAD_MS", "30"))
# Anti-jitter: hold speech_end for this long; if a new speech_start arrives
# within the window, both events are merged into one continuous segment. This
# is independent from `min_silence_ms` (which is the silence threshold inside
# Silero's VADIterator) and acts as a post-processing debounce on top of it.
DEFAULT_MIN_GAP_MS = int(os.getenv("VAD_MIN_GAP_MS", "300"))

app = FastAPI(
    title="Real-Time Chat VAD",
    version="0.1.0",
    description="Local Silero VAD service backed by ONNX Runtime.",
)

_model: Any | None = None
_model_lock = Lock()
_file_inference_lock = Lock()


class AudioDecodeError(ValueError):
    pass


def require_api_key(authorization: str | None = Header(default=None)) -> None:
    if not API_KEY:
        return

    expected = f"Bearer {API_KEY}"
    if authorization != expected:
        raise HTTPException(status_code=401, detail="Invalid authorization token")


def create_model() -> Any:
    return load_silero_vad(onnx=True, opset_version=16)


def get_model() -> Any:
    global _model

    if _model is None:
        with _model_lock:
            if _model is None:
                _model = create_model()

    return _model


@app.on_event("startup")
async def preload_model() -> None:
    if PRELOAD_MODEL:
        await run_in_threadpool(get_model)


def load_audio(audio_path: Path) -> np.ndarray:
    try:
        audio, _ = librosa.load(audio_path, sr=SAMPLE_RATE, mono=True)
    except Exception as exc:
        raise AudioDecodeError("Uploaded file is not a supported audio file") from exc

    return np.asarray(audio, dtype=np.float32)


def validate_threshold(threshold: float) -> float:
    if not 0.0 < threshold < 1.0:
        raise HTTPException(status_code=400, detail="threshold must be between 0 and 1")
    return threshold


def speech_duration(segments: list[dict[str, float]]) -> float:
    return sum(max(0.0, segment["end"] - segment["start"]) for segment in segments)


def detect_speech_file(
    audio_path: Path,
    *,
    threshold: float,
    min_speech_ms: int,
    min_silence_ms: int,
    speech_pad_ms: int,
) -> dict[str, Any]:
    start_time = time.perf_counter()
    audio = load_audio(audio_path)

    with _file_inference_lock:
        segments = get_speech_timestamps(
            audio,
            get_model(),
            threshold=threshold,
            sampling_rate=SAMPLE_RATE,
            min_speech_duration_ms=min_speech_ms,
            min_silence_duration_ms=min_silence_ms,
            speech_pad_ms=speech_pad_ms,
            return_seconds=True,
            time_resolution=3,
        )

    elapsed_s = time.perf_counter() - start_time
    duration_s = len(audio) / SAMPLE_RATE
    speech_s = speech_duration(segments)

    return {
        "segments": segments,
        "has_speech": bool(segments),
        "model": MODEL_RUNTIME,
        "runtime": "onnxruntime",
        "sample_rate": SAMPLE_RATE,
        "duration_s": round(duration_s, 3),
        "speech_duration_s": round(speech_s, 3),
        "speech_ratio": round(speech_s / duration_s, 4) if duration_s else 0.0,
        "elapsed_s": round(elapsed_s, 3),
    }


def pcm16le_to_float32(data: bytes) -> np.ndarray:
    if len(data) % 2:
        raise ValueError("PCM16 payload length must be divisible by 2")
    return np.frombuffer(data, dtype="<i2").astype(np.float32) / 32768.0


def make_stream_event(mark: dict[str, int]) -> dict[str, Any]:
    if "start" in mark:
        sample = int(mark["start"])
        return {
            "event": "speech_start",
            "start": sample,
            "start_s": round(sample / SAMPLE_RATE, 3),
        }

    sample = int(mark["end"])
    return {
        "event": "speech_end",
        "end": sample,
        "end_s": round(sample / SAMPLE_RATE, 3),
    }


def force_stream_end(iterator: VADIterator) -> dict[str, int] | None:
    if not iterator.triggered:
        return None

    end_sample = int(iterator.current_sample)
    iterator.triggered = False
    iterator.temp_end = 0
    return {"end": end_sample}


def websocket_authorized(websocket: WebSocket) -> bool:
    if not API_KEY:
        return True
    return websocket.headers.get("authorization") == f"Bearer {API_KEY}"


@app.get("/health")
def health() -> dict[str, Any]:
    return {
        "ok": True,
        "model": MODEL_RUNTIME,
        "model_loaded": _model is not None,
        "preload": PRELOAD_MODEL,
        "runtime": "onnxruntime",
        "sample_rate": SAMPLE_RATE,
        "chunk_size": CHUNK_SIZE,
    }


@app.post("/v1/audio/speech-timestamps")
async def create_speech_timestamps(
    _: None = Depends(require_api_key),
    file: UploadFile = File(...),
    threshold: float = Form(DEFAULT_THRESHOLD),
    min_speech_ms: int = Form(DEFAULT_MIN_SPEECH_MS),
    min_silence_ms: int = Form(DEFAULT_MIN_SILENCE_MS),
    speech_pad_ms: int = Form(DEFAULT_SPEECH_PAD_MS),
) -> dict[str, Any]:
    threshold = validate_threshold(threshold)
    suffix = Path(file.filename or "audio.wav").suffix or ".wav"

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp_path = Path(tmp.name)
        tmp.write(await file.read())

    try:
        return await run_in_threadpool(
            detect_speech_file,
            tmp_path,
            threshold=threshold,
            min_speech_ms=min_speech_ms,
            min_silence_ms=min_silence_ms,
            speech_pad_ms=speech_pad_ms,
        )
    except AudioDecodeError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    finally:
        tmp_path.unlink(missing_ok=True)


@app.websocket("/v1/audio/vad/stream")
async def stream_vad(websocket: WebSocket) -> None:
    if not websocket_authorized(websocket):
        await websocket.close(code=1008)
        return

    await websocket.accept()
    try:
        threshold = validate_threshold(
            float(websocket.query_params.get("threshold", DEFAULT_THRESHOLD))
        )
    except (HTTPException, ValueError) as exc:
        detail = exc.detail if isinstance(exc, HTTPException) else "Invalid threshold"
        await websocket.send_json({"event": "error", "detail": detail})
        await websocket.close(code=1008)
        return
    min_silence_ms = int(
        websocket.query_params.get("min_silence_ms", DEFAULT_MIN_SILENCE_MS)
    )
    min_speech_ms = int(websocket.query_params.get("min_speech_ms", DEFAULT_MIN_SPEECH_MS))
    speech_pad_ms = int(websocket.query_params.get("speech_pad_ms", DEFAULT_SPEECH_PAD_MS))
    min_gap_ms = max(
        0, int(websocket.query_params.get("min_gap_ms", DEFAULT_MIN_GAP_MS))
    )

    # A dedicated model keeps ONNX recurrent state isolated per live stream.
    model = await run_in_threadpool(create_model)
    iterator = VADIterator(
        model,
        threshold=threshold,
        sampling_rate=SAMPLE_RATE,
        min_silence_duration_ms=min_silence_ms,
        speech_pad_ms=speech_pad_ms,
    )
    pending = np.empty(0, dtype=np.float32)
    min_speech_samples = int(SAMPLE_RATE * min_speech_ms / 1000)
    min_gap_samples = int(SAMPLE_RATE * min_gap_ms / 1000)
    active_start_sample: int | None = None
    # `start_sent` tracks whether the client currently sees an open speech
    # segment (speech_start emitted, matching speech_end not yet emitted).
    # During a merge window it stays True even though `active_start_sample`
    # has been cleared, because the public state is still "speaking".
    start_sent = False
    pending_end_sample: int | None = None

    async def maybe_emit_delayed_start() -> None:
        nonlocal start_sent

        if active_start_sample is None or start_sent:
            return
        if iterator.current_sample - active_start_sample < min_speech_samples:
            return

        await websocket.send_json(make_stream_event({"start": active_start_sample}))
        start_sent = True

    async def flush_pending_end() -> None:
        nonlocal pending_end_sample, start_sent

        if pending_end_sample is None:
            return
        await websocket.send_json(make_stream_event({"end": pending_end_sample}))
        pending_end_sample = None
        start_sent = False

    async def emit_pending_end_if_due() -> None:
        if pending_end_sample is None:
            return
        if iterator.current_sample - pending_end_sample < min_gap_samples:
            return
        await flush_pending_end()

    async def handle_stream_mark(mark: dict[str, int]) -> None:
        nonlocal active_start_sample, start_sent, pending_end_sample

        if "start" in mark:
            new_start = int(mark["start"])
            # Merge with a deferred end if the silence between them is shorter
            # than min_gap_ms. Both events are dropped and the previous segment
            # is treated as still ongoing from the client's perspective.
            if (
                pending_end_sample is not None
                and new_start - pending_end_sample < min_gap_samples
            ):
                pending_end_sample = None
                active_start_sample = new_start
                # `start_sent` stays True: the client is still in "speaking".
                return

            # Real new segment: flush any deferred end first so the client
            # sees a clean end -> start transition.
            if pending_end_sample is not None:
                await flush_pending_end()
            active_start_sample = new_start
            start_sent = False
            await maybe_emit_delayed_start()
            return

        end_sample = int(mark["end"])
        if active_start_sample is not None:
            long_enough = end_sample - active_start_sample >= min_speech_samples
            if start_sent or long_enough:
                if not start_sent:
                    await websocket.send_json(
                        make_stream_event({"start": active_start_sample})
                    )
                    start_sent = True
                # Defer the end emission for min_gap_ms. It will be flushed by
                # emit_pending_end_if_due, by a real new speech_start, or on
                # eof/disconnect.
                pending_end_sample = end_sample
            # else: too-short blip with no public start -> drop both.

        active_start_sample = None

    await websocket.send_json(
        {
            "event": "ready",
            "sample_rate": SAMPLE_RATE,
            "sample_format": "pcm_s16le",
            "channels": 1,
            "chunk_size": CHUNK_SIZE,
            "runtime": "onnxruntime",
        }
    )

    try:
        while True:
            message = await websocket.receive()
            if message.get("type") == "websocket.disconnect":
                return

            if data := message.get("bytes"):
                try:
                    samples = pcm16le_to_float32(data)
                except ValueError as exc:
                    await websocket.send_json({"event": "error", "detail": str(exc)})
                    continue

                pending = np.concatenate((pending, samples))
                while len(pending) >= CHUNK_SIZE:
                    chunk = pending[:CHUNK_SIZE]
                    pending = pending[CHUNK_SIZE:]
                    if mark := iterator(chunk):
                        await handle_stream_mark(mark)
                    await maybe_emit_delayed_start()
                    await emit_pending_end_if_due()
                continue

            if text := message.get("text"):
                try:
                    payload = json.loads(text)
                except json.JSONDecodeError:
                    await websocket.send_json(
                        {"event": "error", "detail": "Control messages must be JSON"}
                    )
                    continue
                event = payload.get("event")
                if event == "reset":
                    pending = np.empty(0, dtype=np.float32)
                    iterator.reset_states()
                    active_start_sample = None
                    start_sent = False
                    pending_end_sample = None
                    await websocket.send_json({"event": "reset"})
                elif event == "eof":
                    if len(pending):
                        padded = np.pad(pending, (0, CHUNK_SIZE - len(pending)))
                        if mark := iterator(padded):
                            await handle_stream_mark(mark)
                        pending = np.empty(0, dtype=np.float32)
                    if mark := force_stream_end(iterator):
                        await handle_stream_mark(mark)
                    await flush_pending_end()
                    await websocket.send_json({"event": "done"})
                    break
                else:
                    await websocket.send_json(
                        {"event": "error", "detail": "Unsupported control event"}
                    )
    except WebSocketDisconnect:
        return
