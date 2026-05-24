"""Realtime voice chat state machine.

Wires Silero VAD (local WS), Volcengine ASR (remote WS), Ark LLM and
seed-tts-2.0 (remote WS) into a single asyncio loop. Mic audio is mirrored to
the VAD client at all times so the user can barge-in even while the assistant
is speaking; the ASR session is only opened during user speech windows.
"""

from __future__ import annotations

import asyncio
import logging
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

from asr_client import StreamingAsrSession
from audio_io import MicSource, SpeakerSink
from conversation import AsyncArkStream, ConversationHistory
from tts_client import SeedTtsClient
from vad_client import VadClient, VadEvent


LOGGER = logging.getLogger(__name__)

SENTENCE_TERMINATORS = "。？！；…!?\n"
MID_SENTENCE_BREAKS = "，、：:,;"
MID_SENTENCE_MIN_LEN = 10
HARD_FLUSH_LEN = 30


def pop_speakable_chunk(pending: str) -> str:
    """Return the longest leading slice of ``pending`` ready to be spoken.

    Returns an empty string when more characters should be buffered.
    """
    if not pending:
        return ""

    for index, char in enumerate(pending):
        if char in SENTENCE_TERMINATORS:
            return pending[: index + 1]

    if len(pending) >= MID_SENTENCE_MIN_LEN:
        for index, char in enumerate(pending):
            if char in MID_SENTENCE_BREAKS:
                return pending[: index + 1]

    if len(pending) >= HARD_FLUSH_LEN:
        return pending

    return ""


@dataclass
class OrchestratorConfig:
    asr_chunk_ms: int = 200
    asr_finalize_timeout_s: float = 2.0
    llm_max_output_tokens: int | None = 512
    llm_temperature: float | None = None
    log_partial_text: bool = False


class ChatOrchestrator:
    def __init__(
        self,
        *,
        vad_client: VadClient,
        mic: MicSource,
        speaker: SpeakerSink,
        asr_factory,
        tts_client: SeedTtsClient,
        ark_stream: AsyncArkStream,
        history: ConversationHistory,
        config: OrchestratorConfig | None = None,
    ) -> None:
        self._vad = vad_client
        self._mic = mic
        self._speaker = speaker
        self._asr_factory = asr_factory
        self._tts = tts_client
        self._ark = ark_stream
        self._history = history
        self._config = config or OrchestratorConfig()

        self._asr_session: StreamingAsrSession | None = None
        self._asr_lock = asyncio.Lock()
        # Side-channel buffer that captures mic PCM arriving between a
        # speech_start event and the moment the ASR session is ready to receive
        # audio. Without it, the head of the utterance is lost during barge-in
        # / WebSocket setup because `_asr_session` is still None.
        self._pending_pcm: list[bytes] | None = None
        self._assistant_task: asyncio.Task[None] | None = None
        self._assistant_cancel: asyncio.Event | None = None
        # Watcher that waits for the ASR partial-text confirmation before
        # cancelling an assistant turn. Lets us defer barge-in when the
        # speaker is producing audio so acoustic echo can't falsely trigger
        # cancellation.
        self._barge_in_watcher: asyncio.Task[None] | None = None
        self._stop_event = asyncio.Event()

    async def run(self) -> None:
        await self._vad.connect()
        self._speaker.start()
        loop = asyncio.get_running_loop()
        self._mic_loop = loop
        self._mic.start()

        try:
            await self._event_loop()
        finally:
            await self._shutdown()

    async def stop(self) -> None:
        self._stop_event.set()

    async def on_mic_pcm(self, pcm: bytes) -> None:
        await self._vad.send_pcm(pcm)
        pending = self._pending_pcm
        if pending is not None:
            # Session not yet ready; stash the frame so it can be drained once
            # `_handle_speech_start` finishes wiring up the ASR connection.
            pending.append(pcm)
            return
        session = self._asr_session
        if session is None:
            return
        try:
            await session.send_pcm(pcm)
        except Exception as exc:
            LOGGER.debug("ASR send failed (dropping session): %s", exc)
            self._asr_session = None
            try:
                await session.close()
            except Exception:
                pass

    async def _event_loop(self) -> None:
        events = self._vad.events
        stop_task = asyncio.create_task(self._stop_event.wait(), name="stop-wait")
        try:
            while not self._stop_event.is_set():
                event_task = asyncio.create_task(events.get(), name="vad-event")
                done, _ = await asyncio.wait(
                    {event_task, stop_task}, return_when=asyncio.FIRST_COMPLETED
                )
                if stop_task in done:
                    event_task.cancel()
                    return
                event = event_task.result()
                await self._handle_vad_event(event)
        finally:
            stop_task.cancel()

    async def _handle_vad_event(self, event: VadEvent) -> None:
        try:
            if event.kind == "speech_start":
                LOGGER.info("VAD speech_start @ %.2fs", event.seconds or 0.0)
                await self._handle_speech_start()
            elif event.kind == "speech_end":
                LOGGER.info("VAD speech_end @ %.2fs", event.seconds or 0.0)
                await self._handle_speech_end()
            elif event.kind == "error":
                LOGGER.error("VAD error: %s", event.detail)
            elif event.kind == "closed":
                LOGGER.warning("VAD closed; stopping orchestrator")
                self._stop_event.set()
        except Exception as exc:
            LOGGER.warning(
                "Recovering from %s handler error: %s", event.kind, exc
            )
            await self._discard_asr_session()

    async def _handle_speech_start(self) -> None:
        # Capture the preroll *before* any barge-in / network waits so the
        # leading edge of the utterance survives long setup latencies. The
        # `_pending_pcm` buffer captures everything that arrives after this
        # point until the ASR session takes over.
        preroll = self._mic.preroll.snapshot()
        pending: list[bytes] = []
        self._pending_pcm = pending

        # Defer barge-in when the speaker is currently producing audio so a
        # mic-feedback / acoustic-echo blip can't kill the assistant. The
        # cancel decision is gated on real ASR confirmation below (either the
        # partial-text watcher or the non-empty finalize in `speech_end`).
        speaker_was_active = self._speaker.is_active()

        try:
            if not speaker_was_active:
                await self._barge_in_if_speaking()

            async with self._asr_lock:
                if self._asr_session is not None:
                    await self._asr_session.close()
                    self._asr_session = None
                self._cancel_barge_in_watcher()
                session = self._asr_factory()
                try:
                    await session.start()
                    if preroll:
                        await session.send_pcm(preroll)
                    # Drain the side-channel buffer. New frames may arrive at
                    # each await; popping from the front lets us finish even
                    # if `on_mic_pcm` keeps appending.
                    while pending:
                        chunk = pending.pop(0)
                        await session.send_pcm(chunk)
                except Exception:
                    await session.close()
                    raise
                # Atomic handoff: switching `_asr_session` and clearing the
                # pending buffer in the same sync block keeps `on_mic_pcm`
                # from ever seeing a state where neither path accepts audio.
                self._asr_session = session
                self._pending_pcm = None

            if speaker_was_active:
                self._barge_in_watcher = asyncio.create_task(
                    self._watch_for_barge_in(session),
                    name="barge-in-watcher",
                )
        except Exception:
            self._pending_pcm = None
            raise

    async def _watch_for_barge_in(self, session: StreamingAsrSession) -> None:
        """Cancel the running assistant turn once ASR confirms real speech."""
        try:
            await session.partial_text_event.wait()
        except asyncio.CancelledError:
            return
        # Only barge in if this watcher is still the canonical one. A newer
        # speech_start replaces it via `_cancel_barge_in_watcher`, so a stale
        # watcher must not race against the next turn.
        if self._barge_in_watcher is not asyncio.current_task():
            return
        LOGGER.info("Barge-in: ASR confirmed user speech")
        await self._barge_in_if_speaking()

    def _cancel_barge_in_watcher(self) -> None:
        watcher = self._barge_in_watcher
        if watcher is None:
            return
        self._barge_in_watcher = None
        if not watcher.done():
            watcher.cancel()

    async def _handle_speech_end(self) -> None:
        async with self._asr_lock:
            session = self._asr_session
            self._asr_session = None

        if session is None:
            return

        try:
            try:
                result = await session.finalize(
                    timeout_s=self._config.asr_finalize_timeout_s
                )
            except Exception as exc:
                LOGGER.warning("ASR finalize failed: %s", exc)
                return
        finally:
            await session.close()

        # Stop the partial-text watcher: from here on, the speech_end path
        # owns the cancel decision and will barge in itself if the result is
        # real speech.
        self._cancel_barge_in_watcher()

        text = (result.text or "").strip()
        if not text:
            LOGGER.info("ASR returned empty text; ignoring utterance")
            return

        LOGGER.info("USER: %s", text)
        # If barge-in was deferred (speaker was active and the watcher hadn't
        # fired yet), cancel the running assistant turn now that we have a
        # real transcription before kicking off the next one.
        await self._barge_in_if_speaking()
        self._history.add_user(text)
        cancel = asyncio.Event()
        self._assistant_cancel = cancel
        self._assistant_task = asyncio.create_task(
            self._run_assistant_turn(cancel), name="assistant-turn"
        )

    async def _discard_asr_session(self) -> None:
        async with self._asr_lock:
            session = self._asr_session
            self._asr_session = None
            self._pending_pcm = None
            self._cancel_barge_in_watcher()
        if session is not None:
            try:
                await session.close()
            except Exception:
                pass

    async def _barge_in_if_speaking(self) -> None:
        task = self._assistant_task
        cancel = self._assistant_cancel
        if task is None or task.done():
            return

        LOGGER.info("Barge-in: cancelling assistant turn")
        if cancel is not None:
            cancel.set()
        self._speaker.flush_and_restart()
        try:
            await asyncio.wait_for(task, timeout=2.0)
        except asyncio.TimeoutError:
            LOGGER.warning("Assistant task did not finish within barge-in timeout")
        except Exception as exc:
            LOGGER.warning("Assistant task error during barge-in: %s", exc)
        finally:
            self._assistant_task = None
            self._assistant_cancel = None

    async def _run_assistant_turn(self, cancel: asyncio.Event) -> None:
        pending = ""
        spoken_segments: list[str] = []

        try:
            async for token in self._ark.stream(
                self._history,
                cancel_event=cancel,
                max_output_tokens=self._config.llm_max_output_tokens,
                temperature=self._config.llm_temperature,
            ):
                if cancel.is_set():
                    break
                pending += token.text
                if self._config.log_partial_text:
                    LOGGER.debug("LLM token: %r", token.text)

                while True:
                    chunk = pop_speakable_chunk(pending)
                    if not chunk:
                        break
                    pending = pending[len(chunk):]
                    await self._speak_chunk(chunk, cancel)
                    spoken_segments.append(chunk)
                    if cancel.is_set():
                        break

            if pending and not cancel.is_set():
                await self._speak_chunk(pending, cancel)
                spoken_segments.append(pending)
                pending = ""
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            LOGGER.exception("assistant turn failed: %s", exc)
        finally:
            spoken_text = "".join(spoken_segments)
            generated_text = (spoken_text + pending).strip()
            interrupted = cancel.is_set()
            # Only persist the assistant turn when the model actually produced
            # text. An empty "[interrupted]" item would just confuse the next
            # turn and is rejected by the Ark Responses API anyway.
            if generated_text:
                stored = generated_text + (" [interrupted]" if interrupted else "")
                LOGGER.info(
                    "ASSISTANT%s: %s",
                    " (interrupted)" if interrupted else "",
                    generated_text,
                )
                self._history.add_assistant(stored)
            elif interrupted:
                LOGGER.info("ASSISTANT (interrupted before producing tokens)")

            if not interrupted:
                await self._speaker.wait_drained()

    async def _speak_chunk(self, text: str, cancel: asyncio.Event) -> None:
        if not text.strip():
            return
        LOGGER.info("TTS chunk: %s", text)
        try:
            async for pcm in self._tts.synthesize(text, cancel_event=cancel):
                if cancel.is_set():
                    break
                self._speaker.enqueue(pcm)
        except Exception as exc:
            LOGGER.warning("TTS failed for chunk %r: %s", text, exc)

    async def _shutdown(self) -> None:
        self._cancel_barge_in_watcher()
        if self._assistant_cancel is not None:
            self._assistant_cancel.set()
        if self._assistant_task is not None and not self._assistant_task.done():
            try:
                await asyncio.wait_for(self._assistant_task, timeout=1.0)
            except (asyncio.TimeoutError, Exception):
                self._assistant_task.cancel()
        if self._asr_session is not None:
            await self._asr_session.close()
            self._asr_session = None

        self._mic.stop()
        self._speaker.stop()
        await self._vad.close()


def build_asr_factory(
    *, api_key: str, resource_id: str, chunk_ms: int = 200
):
    def factory() -> StreamingAsrSession:
        return StreamingAsrSession(
            api_key=api_key, resource_id=resource_id, chunk_ms=chunk_ms
        )

    return factory


def build_executor() -> ThreadPoolExecutor:
    return ThreadPoolExecutor(max_workers=2, thread_name_prefix="ark-stream")
