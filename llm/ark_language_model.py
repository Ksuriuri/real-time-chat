from __future__ import annotations

import os
import time
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from typing import Any

from volcenginesdkarkruntime import Ark


DEFAULT_BASE_URL = "https://ark.cn-beijing.volces.com/api/v3"
DEFAULT_MODEL = "doubao-seed-2-0-lite-260428"


@dataclass(frozen=True)
class StreamToken:
    text: str
    event_type: str
    elapsed_s: float


class ArkLanguageModel:
    """Volcengine Ark Responses API wrapper for language-model calls."""

    def __init__(
        self,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        model: str | None = None,
    ) -> None:
        self.api_key = api_key or os.getenv("ARK_API_KEY")
        if not self.api_key:
            raise ValueError("ARK_API_KEY is required")

        self.base_url = base_url or os.getenv("ARK_BASE_URL", DEFAULT_BASE_URL)
        self.model = model or os.getenv("ARK_MODEL", DEFAULT_MODEL)
        self.client = Ark(base_url=self.base_url, api_key=self.api_key)

    def create_response(
        self,
        prompt: str,
        *,
        model: str | None = None,
        max_output_tokens: int | None = None,
        thinking_disabled: bool = True,
        temperature: float | None = None,
    ) -> Any:
        return self.client.responses.create(
            model=model or self.model,
            input=self._text_input(prompt),
            max_output_tokens=max_output_tokens,
            thinking=self._thinking(thinking_disabled),
            temperature=temperature,
        )

    def stream_text(
        self,
        prompt: str,
        *,
        model: str | None = None,
        max_output_tokens: int | None = None,
        thinking_disabled: bool = True,
        temperature: float | None = None,
    ) -> Iterator[StreamToken]:
        yield from self._stream(
            self._text_input(prompt),
            model=model,
            max_output_tokens=max_output_tokens,
            thinking_disabled=thinking_disabled,
            temperature=temperature,
        )

    def stream_messages(
        self,
        messages: Iterable[dict[str, Any]],
        *,
        model: str | None = None,
        max_output_tokens: int | None = None,
        thinking_disabled: bool = True,
        temperature: float | None = None,
    ) -> Iterator[StreamToken]:
        """Stream tokens from a multi-turn conversation history.

        ``messages`` is an iterable of ``{"role": "system"|"user"|"assistant",
        "content": str}`` dicts. The Ark Responses API distinguishes between
        ``input_text`` (user/system) and ``output_text`` (assistant) content
        types, so we translate roles accordingly.
        """
        yield from self._stream(
            self._messages_input(messages),
            model=model,
            max_output_tokens=max_output_tokens,
            thinking_disabled=thinking_disabled,
            temperature=temperature,
        )

    def _stream(
        self,
        ark_input: list[dict[str, Any]],
        *,
        model: str | None,
        max_output_tokens: int | None,
        thinking_disabled: bool,
        temperature: float | None,
    ) -> Iterator[StreamToken]:
        start_time = time.perf_counter()
        stream = self.client.responses.create(
            model=model or self.model,
            input=ark_input,
            stream=True,
            max_output_tokens=max_output_tokens,
            thinking=self._thinking(thinking_disabled),
            temperature=temperature,
        )

        for event in stream:
            event_type = getattr(event, "type", type(event).__name__)
            text = self._delta_text(event)
            if text:
                yield StreamToken(
                    text=text,
                    event_type=event_type,
                    elapsed_s=time.perf_counter() - start_time,
                )

    @staticmethod
    def _text_input(prompt: str) -> list[dict[str, Any]]:
        return [
            {
                "role": "user",
                "content": [{"type": "input_text", "text": prompt}],
            }
        ]

    @staticmethod
    def _messages_input(messages: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
        """Translate role-tagged messages into Ark Responses API ``input`` items.

        The Ark API mirrors OpenAI Responses API: when replaying assistant
        messages back as input, each item must be a ``type=message`` entry
        carrying ``status="completed"`` and ``output_text`` content. User /
        system messages use ``input_text`` content with no status.
        """
        ark_input: list[dict[str, Any]] = []
        for message in messages:
            role = message["role"]
            content = message["content"]
            text = content if isinstance(content, str) else str(content)
            if role == "assistant":
                ark_input.append(
                    {
                        "type": "message",
                        "role": "assistant",
                        "status": "completed",
                        "content": [{"type": "output_text", "text": text}],
                    }
                )
            else:
                ark_input.append(
                    {
                        "role": role,
                        "content": [{"type": "input_text", "text": text}],
                    }
                )
        return ark_input

    @staticmethod
    def _thinking(disabled: bool) -> dict[str, str] | None:
        if disabled:
            return {"type": "disabled"}
        return None

    @staticmethod
    def _delta_text(event: Any) -> str:
        if getattr(event, "type", None) != "response.output_text.delta":
            return ""

        delta = getattr(event, "delta", None)
        if isinstance(delta, str):
            return delta
        return ""
