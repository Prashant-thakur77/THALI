"""Run one command end to end (plan Phase 7): ``python -m runtime.demo --seed 3 --voice voice/test_samples/normal.wav``.

Voice input from a WAV (Speechmatics realtime, speaker focus on), or ``--command "..."`` text, or ``--mic``.
``--barge-in "stop@4"`` injects an operator partial "stop" 4 s after the arms start moving and ``--resume-after 3``
resumes 3 s later; ``--barge-in "other arm@4"`` swaps arms for the interrupted step.  Writes
results/demo_seed<N>.json and appends to results/audit.jsonl.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import threading
import time
from pathlib import Path

import gymnasium as gym

import souschef_env  # noqa: F401
from planner.plan import Planner
from runtime.state_machine import Runtime
from verifier.rules import Verifier
from voice.parser import intents_to_command, parse

ROOT = Path(__file__).resolve().parent.parent


class VideoRecorder:
    """Side-by-side front + overhead frames with a text HUD (state, last TTS line), piped to ffmpeg."""

    def __init__(self, env, path: Path, every: int = 4, hud=None, fps: float = 50.0):
        import subprocess
        from PIL import ImageDraw, ImageFont  # noqa: F401
        self.env, self.path, self.every, self.hud = env, path, every, hud
        self.n = 0
        self.frames = 0
        path.parent.mkdir(parents=True, exist_ok=True)
        self.w, self.h = 2 * 640, 480 + 40
        self.proc = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{self.w}x{self.h}",
                                      "-r", f"{fps / every:.3f}", "-i", "-", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "23", str(path)], stdin=subprocess.PIPE)

    def tick(self) -> None:
        self.n += 1
        if self.n % self.every:
            return
        import numpy as np
        from PIL import Image, ImageDraw
        was = self.env.render_enabled
        self.env.render_enabled = True
        front = self.env.render_camera("front", 640, 480)
        over = self.env.render_camera("overhead", 640, 480)
        self.env.render_enabled = was
        canvas = Image.new("RGB", (self.w, self.h), (20, 20, 24))
        canvas.paste(Image.fromarray(front), (0, 40))
        canvas.paste(Image.fromarray(over), (640, 40))
        d = ImageDraw.Draw(canvas)
        state, said = self.hud() if self.hud else ("", "")
        d.text((10, 10), f"Thali  |  {state}  |  {said}"[:140], fill=(240, 240, 240))
        self.proc.stdin.write(np.asarray(canvas, dtype=np.uint8).tobytes())
        self.frames += 1

    def close(self) -> None:
        self.proc.stdin.close()
        self.proc.wait()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=3)
    ap.add_argument("--split", default="test")
    ap.add_argument("--command", default=None)
    ap.add_argument("--voice", type=Path, default=None, help="WAV file to transcribe with Speechmatics")
    ap.add_argument("--language", default="en")
    ap.add_argument("--mic", action="store_true")
    ap.add_argument("--planner", default="auto", choices=["auto", "vlm", "rules"])
    ap.add_argument("--device", default="CPU")
    ap.add_argument("--barge-in", default=None, help='e.g. "stop@4" or "other arm@4" (seconds after arms start)')
    ap.add_argument("--resume-after", type=float, default=3.0)
    ap.add_argument("--tts", action="store_true", help="synthesize confirmations with Speechmatics TTS")
    ap.add_argument("--no-camera-check", action="store_true")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--video", type=Path, default=None, help="record front+overhead frames to this mp4 (ffmpeg)")
    ap.add_argument("--video-every", type=int, default=4, help="record one frame every N control steps (4 -> 12.5 fps)")
    args = ap.parse_args()

    env = gym.make("souschef_env/Thali-v0", disable_env_checker=True).unwrapped
    planner = Planner(backend=args.planner, device=args.device)
    say_log: list[dict] = []

    def say(text: str) -> None:
        rec = {"t": time.time(), "text": text}
        if args.tts:
            from voice.speak import say as tts_say
            rec.update(tts_say(text))
        say_log.append(rec)
        print(f"  🗣 {text}")

    rt = Runtime(env, planner, Verifier(), say=say, camera_check=not args.no_camera_check,
                 audit_path=ROOT / "results" / "audit.jsonl")
    recorder = VideoRecorder(env, args.video, every=args.video_every, hud=lambda: (rt.state, say_log[-1]["text"] if say_log else "")) if args.video else None
    if recorder:
        rt.on_step = recorder.tick

    # ---- get the command
    t_speech_end = None
    voice_rec = None
    if args.voice or args.mic:
        from voice.listen import Listener, transcribe_file
        if args.voice:
            utts, lst = asyncio.run(transcribe_file(args.voice, args.language, focus="first"))
            ops = [u for u in utts if u.is_operator]
            text = " ".join(u.text for u in ops)
            t_speech_end = ops[-1].t_ready_wall if ops else None
            voice_rec = {"file": str(args.voice), "operator": lst.operator, "utterances": [u.as_dict() for u in utts], "ignored": lst.ignored}
            rt.barge.operator = lst.operator
        else:  # pragma: no cover
            raise SystemExit("--mic: use voice/listen.py's mic_chunks with a Listener; not wired in the demo")
        pr = parse(text)
        command = intents_to_command(pr.intents) or text
        print(f"heard: {text!r}\nparsed: {command!r}")
    else:
        command = args.command or "Open the top drawer, pick up the plate with arm A, place it on the table, pick up the mug with arm B, pour water into the mug with arm A"
        t_speech_end = time.time()

    # ---- optional injected barge-in (a partial transcript from the operator)
    if args.barge_in:
        kind, at = args.barge_in.split("@")
        at = float(at)

        def inject() -> None:
            while rt.state not in ("EXECUTING",):
                time.sleep(0.05)
            time.sleep(at)
            print(f"  🎤 (partial) {kind!r}")
            rt.barge.on_partial(kind, rt.barge.operator)
            if kind.strip() == "stop":
                time.sleep(args.resume_after)
                print("  🎤 (partial) 'continue'")
                rt.barge.on_partial("continue", rt.barge.operator)

        threading.Thread(target=inject, daemon=True).start()

    log = rt.run_command(command, seed=args.seed, split=args.split, t_speech_end=t_speech_end)
    if recorder:
        recorder.close()
        print(f"video: {args.video} ({recorder.frames} frames)")
    out = log.as_dict() | {"voice": voice_rec, "tts": say_log, "planner_backend": planner.backend, "device": args.device}
    path = args.out or ROOT / "results" / f"demo_seed{args.seed}.json"
    path.write_text(json.dumps(out, indent=2, default=str))
    print(f"\n{log.state} success={log.success} subgoals={log.subgoals}\nlatency={log.latency} replans={log.replans} barge_ins={len(log.barge_ins)} -> {path}")


if __name__ == "__main__":
    main()
