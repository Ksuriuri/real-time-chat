"""Realtime voice chat state machine.

Wires Silero VAD (local WS), Volcengine ASR (remote WS), Ark LLM and
seed-tts-2.0 (remote WS) into a single asyncio loop. Mic audio is mirrored to
the VAD client at all times so the user can barge-in even while the assistant
is speaking; the ASR session is only opened during user speech windows.
"""

from __future__ import annotations

import asyncio
import difflib
import logging
import re
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

from aec_processor import AecProcessor
from asr_client import StreamingAsrSession
from audio_io import MicSource, SpeakerSink
from conversation import AsyncArkStream, ConversationHistory
from tts_client import SeedTtsClient
from vad_client import VadClient, VadEvent


LOGGER = logging.getLogger(__name__)

# Emoji / pictograph ranges to strip before TTS so the voice doesn't try to
# vocalize them (the LLM sprinkles things like 🥺 into replies).
_EMOJI_PATTERN = re.compile(
    "["
    "\U0001f000-\U0001faff"  # emoticons, pictographs, transport, supplemental, ext-A
    "\U00002600-\U000026ff"  # miscellaneous symbols
    "\U00002700-\U000027bf"  # dingbats
    "\U00002b00-\U00002bff"  # misc symbols and arrows (e.g. ⭐)
    "\U00002300-\U000023ff"  # misc technical (e.g. ⏰⌚)
    "\U0000fe00-\U0000fe0f"  # variation selectors
    "\U0000200d"             # zero-width joiner (emoji sequences)
    "\U000020e3"             # combining enclosing keycap
    "]+",
    flags=re.UNICODE,
)

# Marks Fish (and other TTS) tend to vocalize literally or glitch on. Collapse
# ellipsis / em-dash / "..." runs into a single pause comma instead of letting
# the model read them out loud.
_TTS_PAUSE_PATTERN = re.compile(r"[…⋯]+|—{2,}|—|\.{3,}|·{2,}")
# Stylistic / markdown noise that carries no speech value.
_TTS_DROP_PATTERN = re.compile(r"[~～*#`_<>^|]+")
# At least one CJK / kana / hangul / latin / digit means there is something
# real to speak; otherwise the chunk is punctuation-only and we skip it.
_SPEAKABLE_PATTERN = re.compile(
    r"[0-9A-Za-z\u3040-\u30ff\u3400-\u4dbf\u4e00-\u9fff\uac00-\ud7af]"
)
_MULTI_COMMA_PATTERN = re.compile(r"[，,]{2,}")


def strip_emoji(text: str) -> str:
    """Remove emoji / pictographs while keeping normal punctuation intact."""
    return _EMOJI_PATTERN.sub("", text)


def normalize_for_tts(text: str) -> str:
    """Clean a chunk before synthesis.

    Strips emoji, turns ellipsis/dash runs into a single pause comma, drops
    markdown / tone noise, and returns ``""`` when nothing speakable remains so
    the caller can skip punctuation-only chunks (e.g. a lone "…") that make
    Fish read garbage.
    """
    text = strip_emoji(text)
    text = _TTS_PAUSE_PATTERN.sub("，", text)
    text = _TTS_DROP_PATTERN.sub("", text)
    text = _MULTI_COMMA_PATTERN.sub("，", text)
    if not _SPEAKABLE_PATTERN.search(text):
        return ""
    # Drop pause commas / whitespace that ended up leading or trailing.
    return text.strip().strip("，,").strip()


SENTENCE_TERMINATORS = "。？！；…!?\n"
MID_SENTENCE_BREAKS = "，、：:,;～~"
MID_SENTENCE_MIN_LEN = 10
HARD_FLUSH_LEN = 30
# Safety net for CJK buffers once the first chunk has already been spoken.
# Much larger than HARD_FLUSH_LEN so we keep waiting for a natural punctuation
# break (terminator / comma) instead of slicing a word like "小进展" in half.
# Only fires for a runaway burst with no punctuation; by then playback is
# already underway so the extra buffering is inaudible.
CJK_MAX_BUFFER_LEN = 50
# Safety net for pure-English buffers that haven't produced a sentence
# terminator yet. Generous so we don't slice a sentence mid-clause, but
# still bounded so a runaway unpunctuated response eventually flushes.
ENGLISH_HARD_FLUSH_LEN = 120

# Hiragana / katakana / CJK ideographs / half-width kana. If any of these
# appear in the pending buffer we treat the content as CJK-style (no spaces,
# dense per-char) and allow the mid-sentence comma break + 30-char flush.
# Pure-English buffers skip those and wait for a real sentence terminator
# so we don't chop a sentence at the first comma.
_CJK_RE = re.compile(r"[\u3040-\u30ff\u3400-\u4dbf\u4e00-\u9fff\uff66-\uff9f]")

# Strip punctuation and whitespace before comparing ASR output against the
# assistant's recently-spoken text. Keeps both Chinese full-width punctuation
# and ASCII punctuation out of the way without touching the actual characters.
_NORMALIZE_RE = re.compile(
    r"[\s\u3000\.,?!;:。，？！；：、…\"'“”‘’()\[\]{}<>～~`\-_/\\]+"
)


def _normalize_for_match(text: str) -> str:
    return _NORMALIZE_RE.sub("", text).lower()


def _has_cjk(text: str) -> bool:
    return bool(_CJK_RE.search(text))


def _terminator_end(pending: str) -> int:
    """Return the index *after* the first sentence terminator in ``pending``.

    Returns -1 when no terminator is present. ASCII period uses a one-char
    lookahead so we don't chop ``3.14`` / ``e.g.`` apart: ``.`` only counts
    when it's followed by whitespace or sits at the end of the buffer (the
    end-of-buffer case is safe because the stream-end flush will still emit
    whatever follows once it arrives).
    """
    length = len(pending)
    for index, char in enumerate(pending):
        if char in SENTENCE_TERMINATORS:
            return index + 1
        if char == ".":
            next_index = index + 1
            if next_index >= length or pending[next_index].isspace():
                return next_index
    return -1


def pop_speakable_chunk(pending: str, *, latency_critical: bool = False) -> str:
    """Return the longest leading slice of ``pending`` ready to be spoken.

    Returns an empty string when more characters should be buffered.

    ``latency_critical`` should be ``True`` only while we still owe the user
    first audio. In that window we accept an aggressive mid-word hard-flush to
    keep first-audio latency bounded; once playback has started we keep
    buffering until a natural punctuation boundary so a word like "小进展" is
    never split across two chunks.
    """
    if not pending:
        return ""

    end = _terminator_end(pending)
    if end != -1:
        return pending[:end]

    if _has_cjk(pending):
        # CJK / mixed content: dense per-char and missing word boundaries,
        # so cut on a comma once the chunk is long enough.
        if len(pending) >= MID_SENTENCE_MIN_LEN:
            for index, char in enumerate(pending):
                if char in MID_SENTENCE_BREAKS and index + 1 >= MID_SENTENCE_MIN_LEN:
                    return pending[: index + 1]
        # No punctuation break yet. Hard-flush mid-word only for the very
        # first chunk (first-audio latency); afterwards wait much longer so we
        # don't slice a word, falling back to the safety net only on a runaway
        # unpunctuated burst.
        flush_len = HARD_FLUSH_LEN if latency_critical else CJK_MAX_BUFFER_LEN
        if len(pending) >= flush_len:
            space = pending.rfind(" ")
            if space >= MID_SENTENCE_MIN_LEN:
                return pending[:space]
            return pending
        return ""

    # Pure English / whitespace-delimited buffer: wait for a sentence
    # terminator so we never speak half a clause. The high-water mark
    # below only fires as a safety net when the model produces an
    # unusually long unpunctuated burst, and it always lands on a word
    # boundary so the TTS chunk reads naturally.
    if len(pending) >= ENGLISH_HARD_FLUSH_LEN:
        space = pending.rfind(" ")
        if space >= MID_SENTENCE_MIN_LEN:
            return pending[:space]
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
        aec: AecProcessor | None = None,
    ) -> None:
        self._vad = vad_client
        self._mic = mic
        self._speaker = speaker
        self._asr_factory = asr_factory
        self._tts = tts_client
        self._ark = ark_stream
        self._history = history
        self._config = config or OrchestratorConfig()
        # AEC exposes a double-talk signal (near-end speech detected while the
        # speaker is playing). When present it lets a barge-in through even if
        # the transcription overlaps recent assistant speech, which is the
        # case the text-only self-loop filter wrongly suppresses at high
        # speaker volume (loud residual echo dominates the partial).
        self._aec = aec

        self._asr_session: StreamingAsrSession | None = None
        self._asr_lock = asyncio.Lock()
        # Serializes barge-in attempts so the watcher and the speech_end
        # handler can't both be awaiting the same assistant task concurrently.
        # Without this, cancelling the watcher mid-barge-in cascades through
        # `wait_for` into the assistant task itself, and the duplicate
        # `wait_for` in the speech_end path then surfaces a CancelledError
        # that takes down the whole orchestrator.
        self._barge_in_lock = asyncio.Lock()
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
        # Recently-synthesized assistant text used to filter ASR self-loops:
        # if AEC leaks a fragment through and ASR transcribes it, the text
        # will overlap with what we just told TTS to speak. Each entry is
        # `(monotonic_timestamp_s, normalized_text)`.
        self._spoken_window: deque[tuple[float, str]] = deque()
        self._spoken_window_ttl_s = 8.0
        # Below this (normalized) length we don't trust the self-loop check
        # because short phrases like "好的" appear in many real utterances.
        self._self_loop_min_chars = 4
        # Minimum ASR partial length (normalized chars) before the watcher
        # triggers a barge-in. Kept smaller than `_self_loop_min_chars` so we
        # don't wait for the echo-filter threshold: shorter is faster but more
        # exposed to AEC leakage being transcribed as a few characters and
        # falsely cancelling the assistant. 2 is a reasonable middle ground;
        # set to 1 for maximum responsiveness, raise back to 4 if you see
        # spurious cancellations while wearing no headphones.
        self._barge_in_min_chars = 2
        # Monotonic time of the latest speech_start, used to ask "did AEC see
        # double-talk during *this* utterance?". The grace window reaches back
        # before the event because VAD only emits speech_start after
        # min_speech_ms (+ padding), so the user's voice — and the double-talk
        # votes it triggers — precede the event by a few hundred ms.
        self._speech_start_t = 0.0
        self._dt_grace_s = 1.0
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
        self._speech_start_t = time.monotonic()
        pending: list[bytes] = []
        self._pending_pcm = pending

        # Defer barge-in when an assistant turn is in flight so a mic-feedback
        # / acoustic-echo blip (or a spurious VAD trigger) can't kill it. We
        # can't gate on `speaker.is_active()` alone: there's a ~1s window after
        # the LLM responds where we've dispatched TTS chunks but the first PCM
        # frame hasn't been synthesized back yet, so the speaker reads as idle
        # even though a real turn is underway. A speech_start landing in that
        # window used to take the unconditional-cancel path and kill the turn
        # before any audio played. Treat "task exists and not done" as
        # in-flight too, so both the generating and the speaking phases get the
        # ASR-confirmation guard below.
        speaker_was_active = self._speaker.is_active()
        assistant_in_flight = (
            self._assistant_task is not None and not self._assistant_task.done()
        )
        defer_barge_in = speaker_was_active or assistant_in_flight

        try:
            if not defer_barge_in:
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

            if defer_barge_in:
                self._barge_in_watcher = asyncio.create_task(
                    self._watch_for_barge_in(session),
                    name="barge-in-watcher",
                )
        except Exception:
            self._pending_pcm = None
            raise

    async def _watch_for_barge_in(self, session: StreamingAsrSession) -> None:
        """Cancel the running assistant turn once ASR confirms real speech.

        Loops over partial-text updates so we can re-evaluate each one. If the
        text looks like an echo of what the assistant just said, we silently
        clear the event and keep waiting; only a transcription that doesn't
        match the recently-spoken window triggers the actual barge-in.
        """
        while True:
            try:
                await session.text_updated.wait()
            except asyncio.CancelledError:
                return
            # A newer speech_start replaces this watcher via
            # `_cancel_barge_in_watcher`, so a stale watcher must not race
            # against the next turn.
            if self._barge_in_watcher is not asyncio.current_task():
                return
            text = session.latest_text
            session.text_updated.clear()
            if not text:
                continue
            if self._is_self_loop(text):
                if self._double_talk_active():
                    LOGGER.info(
                        "Double-talk detected; overriding echo filter for "
                        "barge-in (%r)",
                        text[:40],
                    )
                else:
                    LOGGER.info(
                        "Suppressing echo-likely partial: %r",
                        text[:40],
                    )
                    continue
            # Short partials happen at the start of every utterance; defer
            # the barge-in until the transcription has enough content for
            # us to confidently distinguish real speech from a fragment of
            # echo. The full text is still committed at speech_end.
            if len(_normalize_for_match(text)) < self._barge_in_min_chars:
                continue
            LOGGER.info(
                "Barge-in: ASR confirmed user speech (%r)", text[:40]
            )
            await self._barge_in_if_speaking()
            return

    def _double_talk_active(self) -> bool:
        """True when AEC detected near-end speech during the current utterance.

        Used to override the text-based self-loop filter: if the user is
        genuinely talking over the assistant (double-talk), the transcription
        may overlap recent assistant speech because loud residual echo bleeds
        into the partial, yet the barge-in is real and must go through.
        """
        if self._aec is None:
            return False
        last_t = self._aec.last_double_talk_t()
        if last_t <= 0.0:
            return False
        return last_t >= self._speech_start_t - self._dt_grace_s

    def _cancel_barge_in_watcher(self) -> None:
        watcher = self._barge_in_watcher
        if watcher is None:
            return
        self._barge_in_watcher = None
        if not watcher.done():
            watcher.cancel()

    def _record_spoken(self, text: str) -> None:
        """Push assistant TTS text into the rolling self-loop window."""
        norm = _normalize_for_match(text)
        if not norm:
            return
        now = time.monotonic()
        self._spoken_window.append((now, norm))
        cutoff = now - self._spoken_window_ttl_s
        while self._spoken_window and self._spoken_window[0][0] < cutoff:
            self._spoken_window.popleft()

    def _is_self_loop(self, text: str) -> bool:
        """True only when ASR text confidently overlaps a recent utterance.

        Short text (e.g. "继续讲", "换个话题", "行") returns False because
        the heuristic is too noisy below ~4 characters: incidental overlap
        of one or two common characters yields high ratios. Genuine TTS
        echo almost always produces multi-character fragments because TTS
        chunks aren't single characters. The watcher loop has its own
        "wait for more text" rule that handles ASR's early short partials
        independently.
        """
        norm = _normalize_for_match(text)
        if len(norm) < self._self_loop_min_chars:
            return False
        cutoff = time.monotonic() - self._spoken_window_ttl_s
        while self._spoken_window and self._spoken_window[0][0] < cutoff:
            self._spoken_window.popleft()
        # ASR routinely inserts/drops one or two filler characters when
        # transcribing TTS echo (e.g. assistant said "讲个软乎乎的" -> ASR
        # "讲一个软乎乎的"), so a strict substring match misses those. Sum
        # the lengths of all matching blocks via Ratcliff-Obershelp and
        # treat the transcription as echo when 70% or more of its content
        # aligns with one of the recent utterances AND the matched portion
        # is at least 5 characters in total. The absolute floor stops short
        # paraphrases like "换个话题" (3 chars matching "换点话题") from
        # being mistaken for echo of unrelated assistant proposals.
        for _, recent in self._spoken_window:
            if not recent:
                continue
            if norm in recent or recent in norm:
                return True
            matcher = difflib.SequenceMatcher(
                None, norm, recent, autojunk=False
            )
            total_match = sum(block.size for block in matcher.get_matching_blocks())
            if total_match >= 5 and total_match / len(norm) >= 0.7:
                return True
        return False

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
        if self._is_self_loop(text):
            if self._double_talk_active():
                LOGGER.info(
                    "Double-talk detected; keeping transcription despite echo "
                    "match (%r)",
                    text[:60],
                )
            else:
                LOGGER.info(
                    "Discarding self-loop transcription (%r); assistant continues.",
                    text[:60],
                )
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
        async with self._barge_in_lock:
            task = self._assistant_task
            cancel = self._assistant_cancel
            if task is None or task.done():
                return

            LOGGER.info("Barge-in: cancelling assistant turn")
            if cancel is not None:
                cancel.set()
            self._speaker.flush()
            try:
                # `shield` keeps a cancellation of *this* coroutine from
                # cascading into the assistant task. The cancel event above
                # is already telling it to wind down cleanly; we just want
                # to wait for that to finish.
                await asyncio.wait_for(asyncio.shield(task), timeout=2.0)
            except asyncio.TimeoutError:
                LOGGER.warning("Assistant task did not finish within barge-in timeout")
            except asyncio.CancelledError:
                # Only re-raise if our caller is actually cancelling us.
                # Otherwise this CancelledError came from the awaited task
                # being cancelled by another path (e.g. a sibling watcher
                # whose `wait_for` was torn down), and swallowing it keeps
                # the orchestrator alive.
                current = asyncio.current_task()
                if current is not None and current.cancelling():
                    raise
            except Exception as exc:
                LOGGER.warning("Assistant task error during barge-in: %s", exc)
            finally:
                if self._assistant_task is task:
                    self._assistant_task = None
                    self._assistant_cancel = None

    async def _run_assistant_turn(self, cancel: asyncio.Event) -> None:
        pending = ""
        spoken_segments: list[str] = []

        # TTS pipeline: each ready chunk fans out to its own short-lived
        # WebSocket synthesis task, while a single consumer drains them in
        # order and forwards PCM to the speaker. The bounded queue means
        # chunk N+1's WS handshake + first audio frame happen concurrently
        # with chunk N's playback instead of stacking serially after it,
        # which is what made gaps between TTS chunks visible in the logs.
        # `seed-tts-2.0/unidirectional/stream` is one-synthesis-per-WS by
        # protocol, so we can't truly reuse a single connection here; the
        # parallel handshakes in this pipeline are the closest equivalent.
        PipelineItem = tuple[str, "asyncio.Queue[bytes | None]"]
        pipeline: asyncio.Queue[PipelineItem | None] = asyncio.Queue(maxsize=2)
        synth_tasks: list[asyncio.Task[None]] = []

        async def synth_worker(text: str, pcm_q: "asyncio.Queue[bytes | None]") -> None:
            try:
                async for pcm in self._tts.synthesize(text, cancel_event=cancel):
                    if cancel.is_set():
                        break
                    await pcm_q.put(pcm)
            except Exception as exc:
                LOGGER.warning("TTS failed for chunk %r: %s", text, exc)
            finally:
                # None marks "this chunk's PCM stream is done" so the
                # consumer can advance to the next pipeline entry.
                await pcm_q.put(None)

        async def speaker_consumer() -> None:
            while True:
                item = await pipeline.get()
                if item is None:
                    return
                _, pcm_q = item
                while True:
                    pcm = await pcm_q.get()
                    if pcm is None:
                        break
                    if cancel.is_set():
                        # Keep draining so synth tasks can finish and the
                        # producer's `pipeline.put` doesn't block forever,
                        # but stop pushing audio to the speaker.
                        continue
                    self._speaker.enqueue(pcm)

        consumer_task = asyncio.create_task(speaker_consumer(), name="tts-consumer")

        async def submit(text: str) -> None:
            text = normalize_for_tts(text)
            if not text:
                return
            LOGGER.info("TTS chunk: %s", text)
            # Record what we're about to speak before any TTS / playback
            # latency so a quick echo round-trip (~1s) is already in the
            # self-loop window when ASR transcribes it.
            self._record_spoken(text)
            spoken_segments.append(text)
            pcm_q: asyncio.Queue[bytes | None] = asyncio.Queue()
            # Reserve the slot before spawning so the bounded queue actually
            # caps concurrent WS handshakes; otherwise a fast LLM stream
            # could spawn an unbounded number of synth tasks faster than
            # the consumer drains them.
            await pipeline.put((text, pcm_q))
            synth_tasks.append(
                asyncio.create_task(synth_worker(text, pcm_q), name="tts-synth")
            )

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
                    # Latency only matters until the user has heard something;
                    # before that we tolerate an aggressive mid-word flush, after
                    # that we hold out for a natural punctuation break.
                    chunk = pop_speakable_chunk(
                        pending, latency_critical=not spoken_segments
                    )
                    if not chunk:
                        break
                    pending = pending[len(chunk):]
                    await submit(chunk)
                    if cancel.is_set():
                        break

            if pending and not cancel.is_set():
                await submit(pending)
                pending = ""
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            LOGGER.exception("assistant turn failed: %s", exc)
        finally:
            # Signal end-of-stream and let the consumer drain whatever's
            # already queued. The consumer keeps pulling even under cancel
            # (just dropping frames), so this put never deadlocks on a
            # full queue.
            try:
                await pipeline.put(None)
            except Exception:
                pass

            # Synth workers exit promptly when `cancel` is set (the TTS
            # client checks it between recvs), so we await them rather
            # than cancelling — we want any tail audio that has already
            # arrived to make it to the speaker on a clean finish.
            for t in synth_tasks:
                try:
                    await t
                except (asyncio.CancelledError, Exception):
                    pass
            try:
                await consumer_task
            except (asyncio.CancelledError, Exception):
                pass

            spoken_text = "".join(spoken_segments)
            generated_text = strip_emoji(spoken_text + pending).strip()
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
