"""Speechmatics TTS confirmations (plan Phase 6.5): "pouring now, say stop anytime".

Text in, 16 kHz PCM/WAV bytes out, cached per phrase under data/tts/ so the demo never waits twice.
Playback (``play``) is optional and needs sounddevice; the runtime only needs the bytes and the timing.
"""

from __future__ import annotations

import asyncio
import hashlib
import os
import time
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")
CACHE = ROOT / "data" / "tts"

PHRASES = {
    "start": "Okay. {plan}",
    "step": "{skill} with arm {arm}.",
    "pour": "Pouring now. Say stop anytime.",
    "stop": "Stopped.",
    "resume": "Continuing.",
    "other_arm": "Switching to the other arm.",
    "done": "The table is set. Enjoy your meal.",
    "blocked": "I can't do that: {reason}.",
    "ignored": "I only take commands from the person who said Thali, listen.",
    "gentle": "Gentle mode on. I will move slowly.",
}


async def synthesize(text: str, api_key: str | None = None) -> bytes:
    from speechmatics.tts import AsyncClient, OutputFormat
    key = api_key or os.environ.get("SPEECHMATICS_API_KEY")
    if not key:
        raise RuntimeError("SPEECHMATICS_API_KEY missing")
    CACHE.mkdir(parents=True, exist_ok=True)
    f = CACHE / (hashlib.sha1(text.encode()).hexdigest()[:16] + ".wav")
    if f.exists():
        return f.read_bytes()
    async with AsyncClient(api_key=key) as client:
        res = await client.generate(text=text, output_format=OutputFormat.WAV_16000)
        audio = await res.read()
    f.write_bytes(audio)
    return audio


def say(kind_or_text: str, **kw) -> dict:
    """Synchronous helper for the runtime: returns {"text", "bytes", "latency_s", "cached"}."""
    text = PHRASES.get(kind_or_text, kind_or_text).format(**kw) if kw or kind_or_text in PHRASES else kind_or_text
    f = CACHE / (hashlib.sha1(text.encode()).hexdigest()[:16] + ".wav")
    cached = f.exists()
    t0 = time.perf_counter()
    audio = asyncio.run(synthesize(text))
    return {"text": text, "bytes": len(audio), "latency_s": time.perf_counter() - t0, "cached": cached}


def play(audio: bytes, sample_rate: int = 16000) -> None:  # pragma: no cover
    import io, wave
    import numpy as np
    import sounddevice as sd
    try:
        with wave.open(io.BytesIO(audio)) as w:
            data = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16)
            sample_rate = w.getframerate()
    except wave.Error:
        data = np.frombuffer(audio, dtype=np.int16)
    sd.play(data, sample_rate)
    sd.wait()


if __name__ == "__main__":  # pragma: no cover
    import sys
    print(say(" ".join(sys.argv[1:]) or "pour"))
