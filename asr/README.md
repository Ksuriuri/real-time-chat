# ASR Service

Local automatic speech recognition service for the real-time chat agent.

The default runtime is Apple Silicon MLX with the int8 quantized Cohere Transcribe
03-2026 model. On this Mac, this is a better fit than vLLM because vLLM is mainly
optimized for CUDA/server GPUs, while MLX runs directly on Apple GPU/Unified
Memory.

## Model

- Runtime: `mlx-speech`
- Default model: `mlx-community/cohere-transcribe-03-2026-mlx-8bit`
- Local cache: `../checkpoints/asr/mlx-community__cohere-transcribe-03-2026-mlx-8bit`
- API shape: OpenAI-compatible `POST /v1/audio/transcriptions`

The model files are about 4.13 GB and download on install or first startup.

## Install

Use the install script from the repository root:

```bash
cd asr
./install.sh
```

The script uses `uv`, creates `.venv` with Python 3.13, installs
`requirements.txt`, checks imports, and downloads/preloads the model.

To install dependencies without downloading the model:

```bash
cd asr
ASR_DOWNLOAD_MODEL=false ./install.sh
```

## Start

```bash
cd asr
uv run uvicorn app:app --host 127.0.0.1 --port 8011
```

Health check:

```bash
curl http://127.0.0.1:8011/health
```

Transcribe audio:

```bash
curl -X POST http://127.0.0.1:8011/v1/audio/transcriptions \
  -F "file=@/path/to/audio.wav" \
  -F "language=en"
```

## Optional API Key

Set `ASR_API_KEY` to require bearer-token auth:

```bash
cd asr
ASR_API_KEY=local-dev-token uv run uvicorn app:app --host 127.0.0.1 --port 8011
```

Then call:

```bash
curl -X POST http://127.0.0.1:8011/v1/audio/transcriptions \
  -H "Authorization: Bearer local-dev-token" \
  -F "file=@/path/to/audio.wav" \
  -F "language=en"
```

## Configuration

```bash
ASR_MODEL_REPO_ID=mlx-community/cohere-transcribe-03-2026-mlx-8bit
ASR_MODEL_SUBDIR=mlx-int8
ASR_MODEL_CACHE_DIR=../checkpoints/asr/mlx-community__cohere-transcribe-03-2026-mlx-8bit
ASR_PRELOAD=true
ASR_API_KEY=
```

`ASR_PRELOAD=true` loads the model during service startup so the first real
transcription request does not pay the model loading cost.

## Volcengine Streaming ASR

`volcengine_streaming_asr.py` is a local test client for Volcengine Doubao
streaming ASR 2.0 bidirectional mode. It reads the API key from
`VOLCENGINE_ASR_API_KEY` and defaults to the hourly ASR 2.0 resource id
`volc.seedasr.sauc.duration`.

Official documentation:
[Volcengine Doubao streaming ASR API](https://www.volcengine.com/docs/6561/1354869?lang=zh).
Use it when debugging WebSocket auth headers, resource ids, binary protocol
frames, sequence numbers, chunk sizes, and ASR error codes.

```bash
cd asr
VOLCENGINE_ASR_API_KEY=your-volcengine-asr-api-key \
uv run python volcengine_streaming_asr.py "/path/to/audio.wav" --print-stream
```

The script converts input audio to 16 kHz mono PCM, sends 200 ms chunks in
real time, and prints JSON with the final text plus latency measurements:

```json
{
  "text": "transcribed text",
  "audio_duration_s": 11.447,
  "send_done_s": 11.557,
  "first_text_s": 1.002,
  "first_definite_s": 7.512,
  "final_s": 11.644,
  "tail_latency_s": 0.087
}
```

Use `--no-realtime` to send chunks as fast as possible, or
`--enable-nonstream` to enable ASR 2.0 second-pass nonstream recognition.

## Response

```json
{
  "text": "transcribed text",
  "model": "mlx-community/cohere-transcribe-03-2026-mlx-8bit",
  "language": "en",
  "duration_s": 3.2,
  "elapsed_s": 0.8
}
```

Use `response_format=text` for a plain text response.
