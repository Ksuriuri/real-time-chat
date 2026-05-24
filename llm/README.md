# LLM Service

Volcengine Ark language-model client for the real-time chat agent.

This module is separate from `asr/`: ASR handles local speech-to-text, while this
module calls Ark language models such as Doubao Seed 2.0.

## Models

- Default: `doubao-seed-2-0-lite-260428`
- Alternative model: `doubao-seed-2-0-mini-260428`
- Base URL: `https://ark.cn-beijing.volces.com/api/v3`

## Official Docs

- Ark product docs: https://www.volcengine.com/docs/82379/1099455?lang=zh
- API key environment setup: https://www.volcengine.com/docs/82379/1399008

Thinking mode is disabled by default for realtime latency:

```python
thinking={"type": "disabled"}
```

## Install

Use the install script from this directory:

```bash
cd llm
./install.sh
```

The script uses `uv`, creates `.venv` with Python 3.13, installs
`requirements.txt`, and checks imports.

## Configuration

Set the API key in your shell or a local `.env` loader:

```bash
export ARK_API_KEY=your-ark-api-key
export ARK_MODEL=doubao-seed-2-0-lite-260428
export ARK_BASE_URL=https://ark.cn-beijing.volces.com/api/v3
```

Do not commit real API keys. Use `.env.example` as the template.

## Usage

Stream a text response:

```python
from ark_language_model import ArkLanguageModel

llm = ArkLanguageModel()
for token in llm.stream_text(
    "请用一句话解释：为什么低延迟对实时语音对话很重要？",
    max_output_tokens=15,
):
    print(token.text, end="", flush=True)
```

Measure short-output TTFT:

```bash
cd llm
ARK_API_KEY=your-ark-api-key uv run python measure_ttft.py \
  --model doubao-seed-2-0-lite-260428 \
  --max-output-tokens 15
```

Switch to mini:

```bash
ARK_MODEL=doubao-seed-2-0-mini-260428 uv run python measure_ttft.py
```
