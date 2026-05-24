# Official Documentation Links

Central index of vendor docs referenced by this repo. Use these when debugging
auth headers, resource ids, binary/WebSocket frames, latency modes, and error
codes.

Last updated: 2026-05-23

## Volcengine 豆包语音 — TTS (`seed-tts-2.0`)

| Topic | Link |
| --- | --- |
| TTS API overview (V3 WebSocket family) | https://www.volcengine.com/docs/6561/1329505?lang=zh |
| HTTP Chunked / SSE unidirectional V3 | https://www.volcengine.com/docs/6561/1598757?lang=zh |
| Voice list (1.0 / 2.0 speakers) | https://www.volcengine.com/docs/6561/1257544?lang=zh |
| TTS 2.0 capability overview | https://www.volcengine.com/docs/6561/1871062?lang=zh |
| API Key management | https://www.volcengine.com/docs/6561/2119699?lang=zh |
| Console FAQ (app id / access key) | https://www.volcengine.com/docs/6561/196768?lang=zh |

### Endpoints used in this repo

| Mode | URL | Code |
| --- | --- | --- |
| HTTP Chunked | `https://openspeech.bytedance.com/api/v3/tts/unidirectional` | `tts/save_tts_samples.py` |
| HTTP SSE | `https://openspeech.bytedance.com/api/v3/tts/unidirectional/sse` | latency tests |
| WS unidirectional | `wss://openspeech.bytedance.com/api/v3/tts/unidirectional/stream` | `chat/tts_client.py`, `tts/save_tts_samples.py` |
| WS bidirectional | `wss://openspeech.bytedance.com/api/v3/tts/bidirection` | LLM token streaming (not wired yet) |

### Request headers (V3)

- `X-Api-Key` — API key from console
- `X-Api-Resource-Id` — `seed-tts-2.0` (see `.env.example`)
- `X-Api-Request-Id` — optional UUID (HTTP)
- `X-Api-Connect-Id` — optional UUID (WebSocket)

### Model / latency knobs

- `seed-tts-2.0-standard` — lower latency, default for realtime chat
- `seed-tts-2.0-expressive` — richer expression, higher variance

## Fish Audio — TTS

| Topic | Link |
| --- | --- |
| Text to Speech guide | https://docs.fish.audio/developer-guide/core-features/text-to-speech |
| Full doc index (`llms.txt`) | https://docs.fish.audio/llms.txt |
| Stream with timestamps (SSE) | https://docs.fish.audio/api-reference/endpoint/openapi-v1/text-to-speech-stream-with-timestamps |
| API reference intro | https://docs.fish.audio/api-reference/introduction |

### Endpoints used in this repo

| Mode | URL | Code |
| --- | --- | --- |
| HTTP / stream | `https://api.fish.audio/v1/tts` | `tts/save_tts_samples.py` |
| SSE + timestamps | `https://api.fish.audio/v1/tts/stream/with-timestamp` | latency tests |

### Request notes

- Body: MessagePack (`Content-Type: application/msgpack`)
- Header: `Authorization: Bearer <key>`, `model: s2-pro`
- Latency: `normal` (quality) vs `balanced` (~300 ms, slightly less stable)

## Volcengine 豆包语音 — ASR 2.0

| Topic | Link |
| --- | --- |
| Streaming ASR bidirectional API | https://www.volcengine.com/docs/6561/1354869?lang=zh |

### Endpoint used in this repo

| Mode | URL | Code |
| --- | --- | --- |
| WS bidirectional | `wss://openspeech.bytedance.com/api/v3/sauc/bigmodel` | `asr/volcengine_streaming_asr.py`, `chat/asr_client.py` |

Default resource id: `volc.seedasr.sauc.duration`

## Volcengine Ark — LLM

| Topic | Link |
| --- | --- |
| Ark product docs | https://www.volcengine.com/docs/82379/1099455?lang=zh |
| API key setup | https://www.volcengine.com/docs/82379/1399008 |

### Endpoint used in this repo

| Mode | URL | Code |
| --- | --- | --- |
| Responses API | `https://ark.cn-beijing.volces.com/api/v3` | `llm/ark_language_model.py`, `chat/orchestrator.py` |

Default model: `doubao-seed-2-0-lite-260428`

## Local services (no vendor API)

| Service | Doc / upstream | Code |
| --- | --- | --- |
| Silero VAD | https://github.com/snakers4/silero-vad | `vad/` |
| MLX Cohere Transcribe ASR | https://huggingface.co/mlx-community/cohere-transcribe-03-2026-mlx-8bit | `asr/` |

## Repo mapping

| Env var | Vendor doc section |
| --- | --- |
| `VOLCENGINE_TTS_API_KEY` | Volcengine TTS → API Key management |
| `VOLCENGINE_TTS_RESOURCE_ID` | Volcengine TTS → HTTP Chunked / SSE V3 |
| `TTS_SPEAKER`, `TTS_MODEL` | Volcengine TTS → 2 2.0 voice list |
| `VOLCENGINE_ASR_API_KEY` | Volcengine ASR 2.0 |
| `ARK_API_KEY`, `ARK_MODEL` | Volcengine Ark |
| `FISH_API_KEY` | Fish Audio TTS (benchmark script only) |

## Sample artifacts

Generated TTS samples and this link index:

- Audio: `tts/output/` (gitignored)
- Links: `docs/official-links.md`
