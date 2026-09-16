"""Barge-in (plan Phase 6.2): partial transcripts matched against stop | wait | other arm | no while a skill runs.

The runtime threads this between the listener's ``on_partial`` callback and the executing skill: the skill's
``stop`` predicate polls ``BargeIn.pending`` every control step, so "stop" halts the arm mid-motion (within one
20 ms step of the partial arriving) and "use the other arm" queues a replan.  Interrupt/resume follows
jawad-glitch's arm_control.py: the interrupted skill is remembered and resumed on "continue".
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field

from voice.parser import detect_barge_in


@dataclass
class BargeIn:
    operator: str | None = None          # only the operator's partials count (speaker focus)
    pending: str | None = None           # stop | other_arm | resume
    t_partial: float | None = None
    events: list[dict] = field(default_factory=list)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)
    _last_text: str = ""

    def on_partial(self, text: str, speaker: str | None) -> None:
        if self.operator is not None and speaker is not None and speaker != self.operator:
            self.events.append({"t": time.time(), "speaker": speaker, "text": text, "ignored": True})
            return
        if text == self._last_text:
            return
        self._last_text = text
        kind = detect_barge_in(text)
        if kind is None:
            return
        with self._lock:
            if self.pending != kind:
                self.pending = kind
                self.t_partial = time.time()
                self.events.append({"t": self.t_partial, "speaker": speaker, "text": text, "kind": kind, "ignored": False})

    def take(self) -> str | None:
        with self._lock:
            k, self.pending = self.pending, None
            return k

    def stop_requested(self) -> bool:
        with self._lock:
            return self.pending in ("stop", "other_arm")
