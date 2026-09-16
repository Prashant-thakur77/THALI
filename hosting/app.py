"""Thali hosted demo (plan Phase 10.2): Gradio app with pre-recorded runs + the live planner/verifier.

    pip install -r hosting/requirements.txt && python hosting/app.py

Tabs: (1) Replay -- the recorded runs in results/ (steps, oracle vs camera, latency, barge-ins) and the seed
videos if present; (2) Plan -- type a command, see the parser's intents, the plan (VLM if the exported model is
present, else the rule planner) and the verifier's verdict with reasons, on the real seed-0 scene;
(3) Verify -- paste any JSON plan and get ALLOW / REORDER / BLOCK; (4) Results -- the rendered evidence tables.
"""

from __future__ import annotations

import json
from pathlib import Path

import gradio as gr

ROOT = Path(__file__).resolve().parent.parent
R = ROOT / "results"

import sys
sys.path.insert(0, str(ROOT))
from planner.plan import Planner  # noqa: E402
from verifier.rules import Verifier, World  # noqa: E402
from voice.parser import intents_to_command, parse  # noqa: E402

_state: dict = {}


def _scene():
    if "world" not in _state:
        import gymnasium as gym
        import souschef_env  # noqa: F401
        from souschef_env.scene_description import describe, scene_state
        env = gym.make("souschef_env/Thali-v0", disable_env_checker=True).unwrapped
        obs, _ = env.reset(seed=0)
        st = scene_state(env.model, env.data)
        _state.update({"img": obs["pixels"]["overhead"], "st": st, "txt": describe(env.model, env.data), "world": World.from_scene_state(st),
                       "planner": Planner(backend="auto", device="CPU"), "verifier": Verifier()})
    return _state


def plan_command(text: str):
    s = _scene()
    pr = parse(text)
    cmd = intents_to_command(pr.intents) or text
    r = s["planner"].plan(cmd, s["st"], s["txt"], s["img"], verifier=s["verifier"], world=s["world"])
    return (json.dumps(pr.as_dict(), indent=1), cmd, f"{r.source} ({r.latency_s:.1f} s)  verdict: {r.verdict}",
            json.dumps(r.plan, indent=1), json.dumps([{k: v for k, v in a.items() if k != "raw"} for a in r.attempts], indent=1))


def verify_plan(js: str):
    s = _scene()
    try:
        plan = json.loads(js)
    except json.JSONDecodeError as e:
        return f"invalid JSON: {e}", ""
    v = s["verifier"].verify(plan, s["world"])
    return v.verdict, "\n".join(f"[{i.code}] step {i.step}: {i.message}" for i in v.issues) + ("\n\nfixed plan:\n" + json.dumps(v.plan, indent=1) if v.verdict == "REORDER" else "")


def replay(name: str):
    d = json.loads((R / name).read_text())
    rows = [[s["idx"], s["step"]["skill"], s["step"]["arm"], "✓" if s["oracle_ok"] else "✗", {True: "✓", False: "✗", None: "—"}[s["camera_ok"]],
             s.get("camera_backend"), "yes" if s["interrupted"] else "", f"{s['t_end'] - s['t_start']:.1f}s"] for s in d["steps"]]
    meta = {k: d[k] for k in ("command", "plan_source", "verdicts", "success", "subgoals", "latency", "replans", "barge_ins", "state") if k in d}
    video = R / name.replace(".json", ".mp4")
    return rows, json.dumps(meta, indent=1), (str(video) if video.exists() else None)


with gr.Blocks(title="Thali") as demo:
    gr.Markdown("# Thali — voice-controlled two-arm table setting\nTwo SO-101 arms in MuJoCo · Speechmatics realtime · local Qwen2-VL planner on OpenVINO · verifier + hash-chained audit. "
                "Measured on an i7-13650HX + UHD iGPU (no NPU).")
    with gr.Tab("Replay"):
        runs = sorted(p.name for p in R.glob("demo_*.json"))
        pick = gr.Dropdown(runs, value=runs[0] if runs else None, label="recorded run")
        table = gr.Dataframe(headers=["step", "skill", "arm", "oracle", "camera", "camera backend", "interrupted", "wall"], interactive=False)
        meta = gr.Code(language="json", label="run")
        vid = gr.Video(label="video (if recorded)")
        pick.change(replay, pick, [table, meta, vid])
        if runs:
            demo.load(replay, pick, [table, meta, vid])
    with gr.Tab("Plan"):
        cmd = gr.Textbox(value="open the top drawer, put a fork by the plate and pour me a little water, gently", label="command (English / Hinglish)")
        btn = gr.Button("plan + verify")
        intents = gr.Code(language="json", label="parsed intents")
        canon = gr.Textbox(label="canonical command")
        src = gr.Textbox(label="planner source / verdict")
        plan = gr.Code(language="json", label="plan")
        attempts = gr.Code(language="json", label="attempts (VLM raw omitted)")
        btn.click(plan_command, cmd, [intents, canon, src, plan, attempts])
    with gr.Tab("Verify"):
        js = gr.Code(language="json", value=json.dumps({"steps": [{"skill": "pour", "arm": "a"}]}, indent=1), label="plan JSON")
        vbtn = gr.Button("verify")
        verdict = gr.Textbox(label="verdict")
        reasons = gr.Textbox(label="reasons", lines=8)
        vbtn.click(verify_plan, js, [verdict, reasons])
    with gr.Tab("Results"):
        ev = ROOT / "docs" / "EVIDENCE.md"
        gr.Markdown(ev.read_text() if ev.exists() else "run `python -m docs.render_readme`")

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
