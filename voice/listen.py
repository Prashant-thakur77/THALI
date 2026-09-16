"""Speechmatics realtime listener (plan Phase 6.1-6.4, 6.6).

Features used, each visible in the emitted events:
- partials (``enable_partials``)          -> barge-in matching while a skill runs
- end-of-turn (``ConversationConfig.end_of_utterance_silence_trigger=0.6``, ``EndOfUtterance``) -> dispatch
- speaker diarization (``diarization="speaker"``)  -> speaker focus: the first speaker who says the wake
  phrase ("Thali, listen") becomes the operator; other speakers' commands are logged and ignored
- custom dictionary (``additional_vocab``)  -> arm A / arm B / mug / fork / spoon / plate / drawer / Thali
- language "hi" for the Hindi/Hinglish session (``--language hi``)
- latency: speech-end (EndOfUtterance end_time) -> command-ready wall time is measured per utterance

Audio comes from a WAV/PCM file (tests, demo recordings) or a microphone (``--mic``, needs sounddevice).
The listener is transport-only: it yields ``Utterance`` objects; the runtime decides what to do with them.
"""

from __future__ import annotations

import asyncio
import os
import re
import time
import wave
from dataclasses import dataclass, field
from pathlib import Path
from typing import AsyncIterator, Callable

from dotenv import load_dotenv
from speechmatics.rt import (AsyncClient, AudioEncoding, AudioFormat, ConversationConfig, ServerMessageType,
                             TranscriptionConfig)

from voice.parser import detect_barge_in, parse

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")

VOCAB = ["arm A", "arm B", "Thali", "mug", "fork", "spoon", "plate", "drawer", "bottle", "pour", "handoff", "gently"]
WAKE = re.compile(r"\b(thali|sous ?chef|tally|tali)\b[,!]?\s*(listen|sun|suno)?", re.I)


@dataclass
class Utterance:
    text: str
    speaker: str | None
    is_operator: bool
    t_speech_end_audio: float          # seconds into the audio stream (from EndOfUtterance/AddTranscript metadata)
    t_ready_wall: float                # wall clock when the utterance was assembled
    t_first_partial_wall: float | None
    partials: int
    parse: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {k: v for k, v in self.__dict__.items()}


class Listener:
    def __init__(self, language: str = "en", api_key: str | None = None, focus: str = "wake",
                 end_of_utterance_s: float = 0.6, on_partial: Callable[[str, str | None], None] | None = None):
        """focus: "wake" (operator = first speaker to say the wake phrase), "first" (operator = first speaker heard),
        "none" (everyone is obeyed)."""
        self.language = language
        self.api_key = api_key or os.environ.get("SPEECHMATICS_API_KEY")
        if not self.api_key:
            raise RuntimeError("SPEECHMATICS_API_KEY missing (put it in .env)")
        self.focus = focus
        self.operator: str | None = None
        self.ignored: list[dict] = []
        self.on_partial = on_partial
        self.cfg = TranscriptionConfig(
            language=language, enable_partials=True, diarization="speaker",
            additional_vocab=[{"content": w} for w in VOCAB],
            conversation_config=ConversationConfig(end_of_utterance_silence_trigger=end_of_utterance_s),
        )
        self.latencies: list[dict] = []

    async def utterances(self, audio: AsyncIterator[bytes] | Path, sample_rate: int = 16000) -> AsyncIterator[Utterance]:
        queue: asyncio.Queue[Utterance | None] = asyncio.Queue()
        buf: list[tuple[str, str | None]] = []      # (text, speaker) finals since the last utterance
        state = {"first_partial": None, "partials": 0, "t_stream_start": time.time()}

        def on_partial(msg: dict) -> None:
            text = msg["metadata"]["transcript"].strip()
            if not text:
                return
            state["partials"] += 1
            if state["first_partial"] is None:
                state["first_partial"] = time.time()
            spk = _speaker(msg)
            if self.on_partial:
                self.on_partial(text, spk)

        def on_final(msg: dict) -> None:
            text = msg["metadata"]["transcript"].strip()
            if text:
                buf.append((text, _speaker(msg)))

        def on_eou(msg: dict) -> None:
            if not buf:
                return
            text = " ".join(t for t, _ in buf).strip()
            speakers = [s for _, s in buf if s]
            spk = max(set(speakers), key=speakers.count) if speakers else None
            u = self._make_utterance(text, spk, float(msg["metadata"]["end_time"]), state)
            buf.clear()
            state["first_partial"], state["partials"] = None, 0
            queue.put_nowait(u)

        async with AsyncClient(api_key=self.api_key) as client:
            client.on(ServerMessageType.ADD_PARTIAL_TRANSCRIPT, on_partial)
            client.on(ServerMessageType.ADD_TRANSCRIPT, on_final)
            client.on(ServerMessageType.END_OF_UTTERANCE, on_eou)
            fmt = AudioFormat(encoding=AudioEncoding.PCM_S16LE, sample_rate=sample_rate)

            async def run() -> None:
                try:
                    if isinstance(audio, Path):
                        with open(audio, "rb") as f:
                            await client.transcribe(f, transcription_config=self.cfg, audio_format=fmt)
                    else:
                        await client.start_session(transcription_config=self.cfg, audio_format=fmt)
                        async for chunk in audio:
                            await client.send_audio(chunk)
                        await client.stop_session()
                    if buf:  # trailing speech without a final EndOfUtterance
                        on_eou({"metadata": {"end_time": -1.0}})
                finally:
                    queue.put_nowait(None)

            task = asyncio.create_task(run())
            while True:
                u = await queue.get()
                if u is None:
                    break
                yield u
            await task

    def _make_utterance(self, text: str, speaker: str | None, t_end_audio: float, state: dict) -> Utterance:
        is_wake = bool(WAKE.search(text))
        if self.operator is None and speaker and (is_wake or self.focus == "first"):
            self.operator = speaker
        is_op = self.focus == "none" or (self.operator is not None and speaker == self.operator)
        clean = WAKE.sub("", text).strip(" ,.")
        u = Utterance(text=clean or text, speaker=speaker, is_operator=is_op, t_speech_end_audio=t_end_audio,
                      t_ready_wall=time.time(), t_first_partial_wall=state["first_partial"], partials=state["partials"])
        u.parse = parse(u.text).as_dict()
        if not is_op:
            self.ignored.append({"speaker": speaker, "text": text})
        return u


def _speaker(msg: dict) -> str | None:
    for r in msg.get("results", []):
        alt = r.get("alternatives") or []
        if alt and alt[0].get("speaker"):
            return alt[0]["speaker"]
    return None


def wav_pcm_path(path: Path) -> tuple[Path, int]:
    """Speechmatics accepts the WAV container directly; return the path and its sample rate."""
    with wave.open(str(path)) as w:
        return path, w.getframerate()


async def mic_chunks(sample_rate: int = 16000, chunk_ms: int = 100) -> AsyncIterator[bytes]:  # pragma: no cover
    import sounddevice as sd
    q: asyncio.Queue[bytes] = asyncio.Queue()
    loop = asyncio.get_event_loop()
    stream = sd.RawInputStream(samplerate=sample_rate, channels=1, dtype="int16", blocksize=int(sample_rate * chunk_ms / 1000),
                               callback=lambda indata, frames, t, status: loop.call_soon_threadsafe(q.put_nowait, bytes(indata)))
    with stream:
        while True:
            yield await q.get()


async def transcribe_file(path: Path, language: str = "en", focus: str = "first", retries: int = 4) -> tuple[list[Utterance], Listener]:
    """One file -> utterances. Retries with backoff on the service's quota/concurrency error (code 4005)."""
    p, sr = wav_pcm_path(path)
    last: Exception | None = None
    for k in range(retries):
        lst = Listener(language=language, focus=focus)
        try:
            return [u async for u in lst.utterances(p, sample_rate=sr)], lst
        except Exception as e:  # TransportError / quota_exceeded
            last = e
            if "quota" not in str(e).lower() and "4005" not in str(e):
                raise
            await asyncio.sleep(15 * (k + 1))
    raise RuntimeError(f"Speechmatics quota still exceeded after {retries} tries: {last}")


if __name__ == "__main__":  # pragma: no cover
    import argparse, json
    ap = argparse.ArgumentParser()
    ap.add_argument("wav", type=Path)
    ap.add_argument("--language", default="en")
    ap.add_argument("--focus", default="first", choices=["wake", "first", "none"])
    a = ap.parse_args()
    utts, lst = asyncio.run(transcribe_file(a.wav, a.language, a.focus))
    for u in utts:
        print(json.dumps(u.as_dict(), indent=1))
    print("operator:", lst.operator, "ignored:", lst.ignored)
