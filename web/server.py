"""Thali Live — the whole closed loop in a browser.

One simulator, one runtime, one worker thread.  The page shows the front and overhead cameras as an MJPEG stream while
the arms execute, and every audit record the runtime writes (state transitions, the plan and its verdict, each skill's
oracle / camera / table checks, replans, the final result) is pushed to the page as a server-sent event, so what the
page shows *is* the hash-chained audit log.  Commands come typed, from a preset, or from one of the bundled voice
samples through Speechmatics (when SPEECHMATICS_API_KEY is set); "stop" / "continue" buttons inject barge-in partials
exactly as the microphone path does.

    MUJOCO_GL=egl uvicorn web.server:app --host 0.0.0.0 --port 7860
"""

from __future__ import annotations

import asyncio
import io
import json
import os
import queue
import threading
import time
from pathlib import Path
from typing import Any

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image, ImageDraw
from pydantic import BaseModel

ROOT = Path(__file__).resolve().parent.parent
STATIC = ROOT / "web" / "static"
RESULTS = ROOT / "results"
try:  # local runs read the API keys from .env; a Space passes them as secrets
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env")
except Exception:  # pragma: no cover
    pass

app = FastAPI(title="Thali Live")


# ----------------------------------------------------------------------------- the one simulator
class Sim:
    """Owns the env + runtime; runs one command at a time in a worker thread; keeps the latest camera frame."""

    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.events: list[dict] = []          # everything the runtime logged, in order, for SSE replay
        self.frame_jpeg: bytes | None = None
        self.frame_id = 0
        self.busy = False
        self.current: dict[str, Any] = {}
        self.last_log = None
        self.said: str = ""
        self.env = None
        self.rt = None
        self.jobs: queue.Queue = queue.Queue()
        self.stream_every = int(os.environ.get("THALI_STREAM_EVERY", "5"))   # control steps between frames (50 Hz / 5 = 10 fps)
        self.vlm_available = (ROOT / "planner" / "qwen2vl_int4").exists()
        self.anomaly_available = any((ROOT / "anomaly" / d).exists() for d in ("model_diffreal", "model_diff", "model"))
        threading.Thread(target=self._worker, daemon=True).start()

    # -- lazy construction (import cost + scene compile happen on the worker, not at import time)
    def _build(self) -> None:
        import gymnasium as gym
        import souschef_env  # noqa: F401
        from planner.plan import Planner
        from runtime.state_machine import Runtime
        from verifier.rules import Verifier

        self.env = gym.make("souschef_env/Thali-v0", disable_env_checker=True).unwrapped
        self.planner = Planner(backend="auto" if self.vlm_available else "rules", device="CPU")
        self.rt = Runtime(self.env, self.planner, Verifier(), say=self._say, camera_check=True,
                          audit_path=RESULTS / "audit_web.jsonl")
        self.rt.on_step = self._tick
        self.rt.on_event = self._on_event
        self.env.reset(seed=0, options={"split": "test"})
        self._capture()

    def _say(self, text: str) -> None:
        self.said = text
        self._push({"kind": "say", "payload": {"text": text}})

    def _on_event(self, kind: str, payload: dict, rec: dict) -> None:
        self._push({"kind": kind, "payload": _jsonable(payload), "seq": rec["seq"], "sha256": rec["sha256"][:16]})

    def _push(self, ev: dict) -> None:
        ev["id"] = len(self.events)
        ev["t"] = round(time.time(), 3)
        with self.lock:
            self.events.append(ev)

    def _tick(self) -> None:
        self.frame_id += 1
        if self.frame_id % self.stream_every == 0:
            self._capture()

    def _capture(self) -> None:
        was = self.env.render_enabled
        self.env.render_enabled = True
        front = self.env.render_camera("front", 480, 360)
        over = self.env.render_camera("overhead", 480, 360)
        self.env.render_enabled = was
        canvas = Image.new("RGB", (960, 392), (11, 14, 20))
        canvas.paste(Image.fromarray(front), (0, 32))
        canvas.paste(Image.fromarray(over), (480, 32))
        d = ImageDraw.Draw(canvas)
        state = self.rt.state if self.rt else "IDLE"
        d.text((10, 9), f"Thali  |  {state}  |  {self.said}"[:120], fill=(235, 235, 240))
        d.text((10, 372), "front", fill=(150, 160, 175))
        d.text((490, 372), "overhead", fill=(150, 160, 175))
        buf = io.BytesIO()
        canvas.save(buf, format="JPEG", quality=80)
        self.frame_jpeg = buf.getvalue()

    # -- jobs
    def submit(self, job: dict) -> None:
        if self.busy:
            raise HTTPException(409, "a command is already running")
        self.jobs.put(job)

    def enqueue(self, jobs: list[dict]) -> None:
        """Queue several commands back to back (a guided tour); follow-ups keep the scene."""
        for j in jobs:
            self.jobs.put(j)

    def _worker(self) -> None:
        try:
            self._build()
            self._push({"kind": "ready", "payload": {"vlm": self.vlm_available, "anomaly": self.anomaly_available,
                                                     "speechmatics": bool(os.environ.get("SPEECHMATICS_API_KEY"))}})
        except Exception as e:  # pragma: no cover
            self._push({"kind": "error", "payload": {"text": f"simulator failed to start: {type(e).__name__}: {e}"}})
            return
        while True:
            job = self.jobs.get()
            self.busy = True
            self.current = job
            try:
                self._run(job)
            except Exception as e:
                self._push({"kind": "error", "payload": {"text": f"{type(e).__name__}: {e}"}})
            finally:
                self.busy = False
                self.current = {}
                self._capture()

    def _run(self, job: dict) -> None:
        from voice.parser import intents_to_command, parse

        command, t_speech_end = job.get("command"), None
        if job.get("voice") or job.get("voice_path"):
            from voice.listen import transcribe_file
            if job.get("voice_path"):                      # a recording uploaded from the visitor's microphone
                path = Path(job["voice_path"])
                lang = job.get("language", "en")
                job["voice"] = path.name
            else:
                path = ROOT / "voice" / "test_samples" / job["voice"]
                lang = "hi" if job["voice"].startswith("hindi") else "en"
            self._push({"kind": "listening", "payload": {"file": job["voice"], "language": lang}})
            utts, lst = asyncio.run(transcribe_file(path, lang, focus="first"))
            ops = [u for u in utts if u.is_operator]
            heard = " ".join(u.text for u in ops)
            t_speech_end = ops[-1].t_ready_wall if ops else None
            self.rt.barge.operator = lst.operator
            self._push({"kind": "heard", "payload": {"text": heard, "operator": lst.operator,
                                                     "ignored": [u.get("text") for u in lst.ignored][:5], "utterances": [u.as_dict() for u in utts]}})
            pr = parse(heard)
            command = intents_to_command(pr.intents) or heard
        else:
            pr = parse(command)
            canonical = intents_to_command(pr.intents)
            self._push({"kind": "parsed", "payload": {"intents": [i.as_dict() for i in pr.intents], "command": canonical or command}})
            command = canonical or command
            t_speech_end = time.time()
        self.rt.concurrent = bool(job.get("concurrent", False))
        if job.get("anomaly") and self.anomaly_available and self.rt.anomaly_check is None:
            from anomaly.check import TableAnomalyCheck
            self.rt.anomaly_check = TableAnomalyCheck("CPU")
        elif not job.get("anomaly"):
            self.rt.anomaly_check = None
        # the Planner falls back to the rule planner whenever its VLM handle is None: switch per job
        if not hasattr(self, "_vlm"):
            self._vlm = self.planner.vlm
        self.planner.vlm = self._vlm if (job.get("planner") == "vlm" and self._vlm is not None) else None
        log = self.rt.run_command(command, seed=int(job.get("seed", 0)), split=job.get("split", "test"),
                                  t_speech_end=t_speech_end, reset=not job.get("keep_scene", False))
        self.last_log = log
        self._push({"kind": "done", "payload": _jsonable({"success": log.success, "subgoals": log.subgoals, "latency": log.latency,
                                                          "replans": log.replans, "sim_steps": log.sim_steps, "wall_s": log.wall_s,
                                                          "audit_records": self.rt.audit.seq, "audit_head": self.rt.audit.prev_hash[:16]})})


def _jsonable(v: Any) -> Any:
    if isinstance(v, dict):
        return {k: _jsonable(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_jsonable(x) for x in v]
    if isinstance(v, (np.floating, np.integer)):
        return v.item()
    if isinstance(v, np.bool_):
        return bool(v)
    if isinstance(v, np.ndarray):
        return v.tolist()
    return v


SIM = Sim()


# ----------------------------------------------------------------------------- API
class RunRequest(BaseModel):
    command: str | None = None
    voice: str | None = None          # one of voice/test_samples/*.wav
    planner: str = "rules"            # rules | vlm
    seed: int = 0
    split: str = "test"
    concurrent: bool = False
    anomaly: bool = False
    keep_scene: bool = False          # follow-up command on the current table (e.g. "clear the table" after "set the table")


@app.post("/api/run")
def run(req: RunRequest):
    if not req.command and not req.voice:
        raise HTTPException(400, "command or voice sample required")
    SIM.submit(req.model_dump())
    return {"ok": True}


class BargeRequest(BaseModel):
    kind: str   # stop | continue | other_arm


@app.post("/api/bargein")
def bargein(req: BargeRequest):
    if SIM.rt is None:
        raise HTTPException(503, "simulator not ready")
    text = {"stop": "stop", "continue": "continue", "other_arm": "use the other arm"}.get(req.kind, req.kind)
    SIM.rt.barge.on_partial(text, SIM.rt.barge.operator)
    SIM._push({"kind": "bargein_button", "payload": {"text": text}})
    return {"ok": True}


@app.get("/api/status")
def status():
    return {"busy": SIM.busy, "state": SIM.rt.state if SIM.rt else "STARTING", "current": SIM.current, "events": len(SIM.events),
            "vlm": SIM.vlm_available, "anomaly": SIM.anomaly_available, "speechmatics": bool(os.environ.get("SPEECHMATICS_API_KEY"))}


@app.get("/api/samples")
def samples():
    tx = {}
    p = ROOT / "voice" / "test_samples" / "transcripts.txt"
    if p.exists():
        for line in p.read_text().splitlines():
            if "\t" in line:
                k, v = line.split("\t", 1)
                tx[k.strip()] = v.strip()
    return [{"file": f.name, "transcript": tx.get(f.stem, tx.get(f.name, ""))} for f in sorted((ROOT / "voice" / "test_samples").glob("*.wav"))]


@app.get("/api/results")
def results():
    """The evidence index the README is rendered from, as JSON (name -> the headline numbers)."""
    out = {}
    for f in sorted(RESULTS.glob("*.json")):
        try:
            d = json.loads(f.read_text())
        except Exception:
            continue
        if not isinstance(d, dict):   # a few result files are plain lists (event streams)
            continue
        head = {k: v for k, v in d.items() if isinstance(v, (int, float, str, bool)) and k not in ("caption",)}
        out[f.name] = head
    return out


@app.get("/api/events")
async def events(after: int = -1):
    """Server-sent events: replay everything after ``after`` then stream new records."""

    async def gen():
        last = after
        while True:
            with SIM.lock:
                new = [e for e in SIM.events if e["id"] > last]
            for e in new:
                last = e["id"]
                yield f"id: {e['id']}\ndata: {json.dumps(e)}\n\n"
            if not new:
                yield ": keepalive\n\n"
                await asyncio.sleep(0.25)

    return StreamingResponse(gen(), media_type="text/event-stream", headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@app.get("/stream")
async def stream():
    async def gen():
        last = None
        while True:
            f = SIM.frame_jpeg
            if f is not None and f is not last:
                last = f
                yield b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: " + str(len(f)).encode() + b"\r\n\r\n" + f + b"\r\n"
            await asyncio.sleep(0.08)

    return StreamingResponse(gen(), media_type="multipart/x-mixed-replace; boundary=frame")


@app.get("/frame.jpg")
def frame():
    if SIM.frame_jpeg is None:
        raise HTTPException(503, "no frame yet")
    return StreamingResponse(io.BytesIO(SIM.frame_jpeg), media_type="image/jpeg")


@app.get("/")
def index():
    return FileResponse(STATIC / "index.html")


app.mount("/static", StaticFiles(directory=STATIC), name="static")
