from __future__ import annotations

import os
import tempfile
import time
from pathlib import Path
from threading import Lock
from typing import Any

import librosa
import numpy as np
from fastapi import Depends, FastAPI, File, Form, Header, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool
from huggingface_hub import snapshot_download
from mlx_speech.generation import CohereAsrModel


MODEL_REPO_ID = os.getenv(
    "ASR_MODEL_REPO_ID", "mlx-community/cohere-transcribe-03-2026-mlx-8bit"
)
MODEL_SUBDIR = os.getenv("ASR_MODEL_SUBDIR", "mlx-int8")
DEFAULT_MODEL_CACHE_DIR = (
    Path(__file__).resolve().parents[1]
    / "checkpoints"
    / "asr"
    / MODEL_REPO_ID.replace("/", "__")
)
MODEL_CACHE_DIR = Path(os.getenv("ASR_MODEL_CACHE_DIR", str(DEFAULT_MODEL_CACHE_DIR)))
API_KEY = os.getenv("ASR_API_KEY")
PRELOAD_MODEL = os.getenv("ASR_PRELOAD", "true").lower() not in {"0", "false", "no"}

app = FastAPI(
    title="Real-Time Chat ASR",
    version="0.1.0",
    description="Local ASR service backed by Cohere Transcribe 03-2026 on MLX.",
)

_model: CohereAsrModel | None = None
_model_lock = Lock()


class AudioDecodeError(ValueError):
    pass


def require_api_key(authorization: str | None = Header(default=None)) -> None:
    if not API_KEY:
        return

    expected = f"Bearer {API_KEY}"
    if authorization != expected:
        raise HTTPException(status_code=401, detail="Invalid authorization token")


def model_dir() -> Path:
    local_root = snapshot_download(
        repo_id=MODEL_REPO_ID,
        local_dir=MODEL_CACHE_DIR,
        allow_patterns=f"{MODEL_SUBDIR}/*",
    )
    return Path(local_root) / MODEL_SUBDIR


def get_model() -> CohereAsrModel:
    global _model

    if _model is None:
        with _model_lock:
            if _model is None:
                _model = CohereAsrModel.from_path(str(model_dir()))

    return _model


@app.on_event("startup")
async def preload_model() -> None:
    if PRELOAD_MODEL:
        await run_in_threadpool(get_model)


def load_audio_16k(audio_path: Path) -> np.ndarray:
    try:
        audio, _ = librosa.load(audio_path, sr=16000, mono=True)
    except Exception as exc:
        raise AudioDecodeError("Uploaded file is not a supported audio file") from exc

    return np.asarray(audio, dtype=np.float32)


def cohere_text(result: Any) -> str:
    text = getattr(result, "text", result)
    if isinstance(text, list):
        return " ".join(str(item) for item in text)
    return str(text)


def transcribe_file(
    audio_path: Path,
    *,
    language: str,
    response_format: str,
) -> dict[str, Any] | str:
    start_time = time.perf_counter()
    audio = load_audio_16k(audio_path)
    result = get_model().transcribe(audio, sample_rate=16000, language=language)
    elapsed_s = time.perf_counter() - start_time
    text = cohere_text(result)

    if response_format == "text":
        return text

    return {
        "text": text,
        "model": MODEL_REPO_ID,
        "language": language,
        "duration_s": round(len(audio) / 16000, 3),
        "elapsed_s": round(elapsed_s, 3),
    }


@app.get("/health")
def health() -> dict[str, Any]:
    return {
        "ok": True,
        "model": MODEL_REPO_ID,
        "model_loaded": _model is not None,
        "preload": PRELOAD_MODEL,
        "runtime": "mlx",
    }


@app.post("/v1/audio/transcriptions")
async def create_transcription(
    _: None = Depends(require_api_key),
    file: UploadFile = File(...),
    language: str = Form("en"),
    response_format: str = Form("json"),
) -> Any:
    if response_format not in {"json", "text"}:
        raise HTTPException(
            status_code=400,
            detail="response_format must be either 'json' or 'text'",
        )

    suffix = Path(file.filename or "audio.wav").suffix or ".wav"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp_path = Path(tmp.name)
        tmp.write(await file.read())

    try:
        result = await run_in_threadpool(
            transcribe_file,
            tmp_path,
            language=language,
            response_format=response_format,
        )
        return result
    except AudioDecodeError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    finally:
        tmp_path.unlink(missing_ok=True)
