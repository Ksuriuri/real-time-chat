"""Conversation history + async wrapper around the synchronous Ark stream.

The Ark Responses API ships a synchronous generator. We bridge it to asyncio
by running the iterator in a worker thread and feeding tokens through an
``asyncio.Queue``. Each pull checks ``cancel_event`` so a barge-in can
short-circuit the LLM mid-response.
"""

from __future__ import annotations

import asyncio
import logging
import sys
import threading
from collections.abc import AsyncIterator
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# Allow importing the sibling ``llm`` module without packaging.
_LLM_DIR = Path(__file__).resolve().parent.parent / "llm"
if str(_LLM_DIR) not in sys.path:
    sys.path.insert(0, str(_LLM_DIR))

from ark_language_model import ArkLanguageModel, StreamToken  # noqa: E402


LOGGER = logging.getLogger(__name__)


@dataclass
class ConversationHistory:
    system_prompt: str | None = None
    messages: list[dict[str, str]] = field(default_factory=list)
    max_turns: int = 12

    def add_user(self, text: str) -> None:
        self.messages.append({"role": "user", "content": text})
        self._trim()

    def add_assistant(self, text: str) -> None:
        self.messages.append({"role": "assistant", "content": text})
        self._trim()

    def to_ark_messages(self) -> list[dict[str, str]]:
        result: list[dict[str, str]] = []
        if self.system_prompt:
            result.append({"role": "system", "content": self.system_prompt})
        result.extend(self.messages)
        return result

    def _trim(self) -> None:
        if self.max_turns <= 0:
            return
        max_messages = self.max_turns * 2
        if len(self.messages) > max_messages:
            del self.messages[: len(self.messages) - max_messages]


_SENTINEL: Any = object()


class AsyncArkStream:
    """Run the synchronous Ark generator on a worker thread."""

    def __init__(
        self,
        llm: ArkLanguageModel,
        *,
        executor: ThreadPoolExecutor | None = None,
    ) -> None:
        self._llm = llm
        self._executor = executor

    async def stream(
        self,
        history: ConversationHistory,
        *,
        cancel_event: asyncio.Event,
        max_output_tokens: int | None = None,
        thinking_disabled: bool = True,
        temperature: float | None = None,
    ) -> AsyncIterator[StreamToken]:
        loop = asyncio.get_running_loop()
        queue: asyncio.Queue[Any] = asyncio.Queue(maxsize=64)
        messages = history.to_ark_messages()
        # Worker-side stop signal. Distinct from the user-facing barge-in
        # `cancel_event` so a clean stream finish (or a generator that the
        # caller stopped iterating) doesn't masquerade as a barge-in to the
        # orchestrator's "interrupted" bookkeeping.
        worker_stop = threading.Event()

        def _worker() -> None:
            try:
                for token in self._llm.stream_messages(
                    messages,
                    max_output_tokens=max_output_tokens,
                    thinking_disabled=thinking_disabled,
                    temperature=temperature,
                ):
                    if cancel_event.is_set() or worker_stop.is_set():
                        break
                    asyncio.run_coroutine_threadsafe(queue.put(token), loop)
            except Exception as exc:
                asyncio.run_coroutine_threadsafe(queue.put(exc), loop)
            finally:
                asyncio.run_coroutine_threadsafe(queue.put(_SENTINEL), loop)

        future = loop.run_in_executor(self._executor, _worker)
        try:
            while True:
                if cancel_event.is_set():
                    break
                try:
                    item = await asyncio.wait_for(queue.get(), timeout=0.05)
                except asyncio.TimeoutError:
                    continue
                if item is _SENTINEL:
                    break
                if isinstance(item, Exception):
                    raise item
                yield item
        finally:
            worker_stop.set()
            await future
