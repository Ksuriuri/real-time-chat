# Realtime Chat Orchestrator

CLI orchestrator that wires the existing modules into a closed-loop voice
agent:

- **VAD** — local Silero VAD WebSocket service (`../vad`).
- **ASR** — Volcengine Doubao ASR 2.0 bidirectional streaming.
- **LLM** — Ark Responses API with `doubao-seed-2-0-lite-260428` (multi-turn).
- **TTS** — `seed-tts-2.0` unidirectional streaming WebSocket, PCM 24 kHz.
- **Audio** — `sounddevice` mic input (16 kHz int16) and speaker output
  (24 kHz int16).

## Pipeline

```
mic 16k -> VAD WS                -> speech_start / speech_end
       \-> ring buffer (500ms preroll)
                  |
                  v   on speech_start: open ASR session, replay preroll
                ASR WS (Volcengine, 200ms chunks)
                  |   on speech_end: finalize -> final text
                  v
                LLM (multi-turn history) --tokens-->
                                    chunk by sentence/punctuation/length
                                            |
                                            v
                                          TTS WS (seed-tts-2.0, PCM 24k)
                                            |
                                            v
                                        speaker 24k

speech_start while assistant talking -> barge-in:
   cancel LLM, abort current TTS, flush speaker, save partial text in history.
```

## Install

```bash
cd chat
./install.sh
```

The script creates `.venv` (Python 3.13), installs `requirements.txt`, and
performs a quick import smoke check. macOS may prompt for microphone access
on first run.

## Configure

Single source of truth: `<repo-root>/.env`.

```bash
cd ..
cp .env.example .env
$EDITOR .env
```

Required keys:

- `ARK_API_KEY` — Volcengine Ark for the Doubao LLM.
- `VOLCENGINE_ASR_API_KEY` — streaming ASR 2.0 (`volc.seedasr.sauc.duration`).
- `VOLCENGINE_TTS_API_KEY` — `seed-tts-2.0`.

`start.sh` sources this file so every subprocess (VAD uvicorn, chat CLI)
inherits the same environment. If you run a module's standalone test script
directly and want to reuse the same keys, do:

```bash
set -a; source ../.env; set +a
uv run python volcengine_streaming_asr.py /path/to/audio.wav
```

### Choosing a TTS provider

`TTS_PROVIDER` selects the voice backend:

- `volcengine` (default) — `seed-tts-2.0`, uses `VOLCENGINE_TTS_API_KEY`.
- `fish` — Fish Audio, uses `FISH_API_KEY`.

To use Fish Audio with **your own reference voice** (zero-shot cloning), set:

```bash
TTS_PROVIDER=fish
FISH_API_KEY=your-fish-audio-api-key
FISH_REFERENCE_AUDIO=/abs/path/to/your_sample.wav   # 10-30s clear speech
FISH_REFERENCE_TEXT=这段参考音频里说的原文          # exact transcript
```

The orchestrator requests `format=pcm, sample_rate=24000`, so Fish audio feeds
the speaker directly with no decode/resample step. `FISH_MODEL` (`s1` /
`s2-pro`) and `FISH_LATENCY` (`normal` / `balanced` / `low`) are optional. If
you skip `FISH_REFERENCE_AUDIO` you can instead point `FISH_REFERENCE_ID` at a
hosted voice model from fish.audio.

Optional knobs:

- `VAD_MIN_SILENCE_MS=500` — silence (ms) before VAD declares the user done
  (this is what triggers ASR finalize and the LLM call). Lower = snappier
  end-of-turn but more cuts on mid-sentence pauses.
- `VAD_MIN_SPEECH_MS=250` — minimum sustained voice before VAD fires
  `speech_start` (also the barge-in threshold).
- `TTS_SPEAKER`, `TTS_MODEL` — `seed-tts-2.0` voice and model variant.
- `SYSTEM_PROMPT` — system message prepended to every LLM call.

## Run

### One-click launcher (recommended)

From the repo root:

```bash
./start.sh
```

This auto-installs missing venvs, starts the local VAD service in the
background (logs to `.cache/vad.log`, pid in `.cache/vad.pid`), waits for it
to report healthy, then runs the chat orchestrator in the foreground. Ctrl+C
shuts everything down cleanly.

### Manual two-terminal flow

Terminal 1 — start the VAD service:

```bash
cd vad
uv run uvicorn app:app --host 127.0.0.1 --port 8010
```

Terminal 2 — start the orchestrator:

```bash
cd chat
uv run python cli.py
```

Speak into the mic. After ~0.8 s of silence the assistant replies through the
speaker; speaking again while it talks interrupts it immediately and starts a
new user turn. Ctrl+C exits cleanly.

## Tuning

| Setting | Where | Effect |
| --- | --- | --- |
| `--vad-min-silence-ms` | CLI / env | Endpoint-of-utterance threshold. Lower = snappier but more cuts; higher = more patient. |
| `--vad-min-speech-ms` | CLI / env | Filters short noises. Below this no `speech_start` fires (so no barge-in either). |
| `--asr-chunk-ms` | CLI | ASR audio frame size. 200 ms balances latency and overhead. |
| `--asr-finalize-timeout-s` | CLI | Max wait after `speech_end` before falling back to the last partial transcript. |
| `--llm-max-output-tokens` | CLI | Cap on assistant length. |
| `MID_SENTENCE_MIN_LEN` / `HARD_FLUSH_LEN` / `CJK_MAX_BUFFER_LEN` | `orchestrator.py` | TTS chunking thresholds. `HARD_FLUSH_LEN` only applies to the first chunk (first-audio latency); later chunks wait for a punctuation break up to `CJK_MAX_BUFFER_LEN` so words aren't split. Tweak if first audio is too late or too choppy. |

## Troubleshooting

- **`sounddevice` errors on macOS**: grant microphone permission to your
  terminal. If using PortAudio for the first time, `uv pip install --reinstall
  sounddevice` rebuilds bindings.
- **"VAD closed; stopping orchestrator"**: the local VAD service stopped or
  was unreachable. Check it is running on the configured `VAD_WS_URL`.
- **No audio output / clipped speech**: `seed-tts-2.0` returns 24 kHz mono
  int16. If you change `audio_params.sample_rate`, update
  `audio_io.SPEAKER_SAMPLE_RATE` to match.
- **Barge-in feels twitchy**: raise `VAD_MIN_SPEECH_MS` (e.g. 350 ms) so
  short coughs / "嗯" no longer interrupt the assistant.
- **Assistant text gets truncated with `[interrupted]`**: that is by design —
  it preserves whatever the model produced before the user spoke over it.

## Out of scope (next stage)

- Browser frontend over FastAPI WebSocket (the orchestrator is structured so
  the same state machine can be reused; only `audio_io.py` would be replaced
  by a websocket-bridged source/sink).
