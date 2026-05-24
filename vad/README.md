# VAD Service

Local voice activity detection service for the real-time chat agent.

The default runtime is Silero VAD with ONNX Runtime on CPU. This is the best
local fit for low-latency real-time gating: the model is small, runs with
single-thread CPU inference, and supports 16 kHz chunks of 512 samples
(about 32 ms).

## Model

- Runtime: `onnxruntime`
- Model package: `silero-vad[onnx-cpu]`
- Default sample rate: `16000`
- Realtime frame: PCM16 mono, 512 samples per inference frame
- HTTP API: `POST /v1/audio/speech-timestamps`
- Streaming API: `WS /v1/audio/vad/stream`

Silero VAD supports 8 kHz and 16 kHz audio. Keep the default 16 kHz path unless
the upstream audio is already telephony-grade 8 kHz.

## Install

Use the install script from the `vad` directory:

```bash
cd vad
./install.sh
```

The script uses `uv`, creates `.venv` with Python 3.13, installs
`requirements.txt`, checks imports, and preloads the ONNX model.

To install dependencies without preloading the model:

```bash
cd vad
VAD_PRELOAD_MODEL=false ./install.sh
```

## Start

```bash
cd vad
uv run uvicorn app:app --host 127.0.0.1 --port 8010
```

Health check:

```bash
curl http://127.0.0.1:8010/health
```

Detect speech segments in an audio file:

```bash
curl -X POST http://127.0.0.1:8010/v1/audio/speech-timestamps \
  -F "file=@/path/to/audio.wav"
```

## Realtime WebSocket

Connect to:

```text
ws://127.0.0.1:8010/v1/audio/vad/stream
```

Send binary frames as raw PCM16 little-endian mono audio at 16 kHz. The service
buffers input and runs inference every 512 samples. It emits JSON events:

```json
{"event": "ready", "sample_rate": 16000, "sample_format": "pcm_s16le", "channels": 1}
{"event": "speech_start", "start": 7680, "start_s": 0.48}
{"event": "speech_end", "end": 25120, "end_s": 1.57}
```

Send a JSON control message to flush the final segment:

```json
{"event": "eof"}
```

Reset stream state:

```json
{"event": "reset"}
```

Stream a local audio file through the same WebSocket interface:

```bash
cd vad
uv run python stream_vad_client.py "/Users/haoyu/Downloads/珊瑚1.wav"
```

By default the client sends PCM chunks in real time. Use `--no-realtime` to
send as fast as possible for local benchmarking:

```bash
uv run python stream_vad_client.py "/Users/haoyu/Downloads/珊瑚1.wav" --no-realtime
```

Tune VAD parameters per stream:

```bash
uv run python stream_vad_client.py "/path/to/audio.wav" \
  --threshold 0.5 \
  --min-speech-ms 250 \
  --min-silence-ms 100 \
  --min-gap-ms 300
```

## Optional API Key

Set `VAD_API_KEY` to require bearer-token auth:

```bash
cd vad
VAD_API_KEY=local-dev-token uv run uvicorn app:app --host 127.0.0.1 --port 8010
```

HTTP calls:

```bash
curl -X POST http://127.0.0.1:8010/v1/audio/speech-timestamps \
  -H "Authorization: Bearer local-dev-token" \
  -F "file=@/path/to/audio.wav"
```

WebSocket clients should send the same `Authorization: Bearer local-dev-token`
header during connection.

## Configuration

```bash
VAD_API_KEY=
VAD_PRELOAD=true
VAD_SAMPLE_RATE=16000
VAD_THRESHOLD=0.5
VAD_MIN_SPEECH_MS=250
VAD_MIN_SILENCE_MS=100
VAD_SPEECH_PAD_MS=30
VAD_MIN_GAP_MS=300
```

`VAD_MIN_GAP_MS` is an anti-jitter merge gap layered on top of
`VAD_MIN_SILENCE_MS`. The server holds each `speech_end` for this many
milliseconds; if a new `speech_start` would fire within the window the events
are dropped and the segment is treated as continuous. Set it to `0` to
disable. This prevents rapid speaker/non-speaker flapping when the user takes
a brief mid-sentence breath that just barely exceeds `min_silence_ms`.

Use query parameters to tune a realtime stream without restarting:

```text
ws://127.0.0.1:8010/v1/audio/vad/stream?threshold=0.5&min_silence_ms=100&min_gap_ms=300
```

Use form fields to tune a file request:

```bash
curl -X POST http://127.0.0.1:8010/v1/audio/speech-timestamps \
  -F "file=@/path/to/audio.wav" \
  -F "threshold=0.5" \
  -F "min_speech_ms=250" \
  -F "min_silence_ms=100" \
  -F "speech_pad_ms=30"
```

## Response

```json
{
  "segments": [{"start": 0.48, "end": 1.57}],
  "has_speech": true,
  "model": "silero-vad-onnx",
  "runtime": "onnxruntime",
  "sample_rate": 16000,
  "duration_s": 3.2,
  "speech_duration_s": 1.09,
  "speech_ratio": 0.3406,
  "elapsed_s": 0.01
}
```
