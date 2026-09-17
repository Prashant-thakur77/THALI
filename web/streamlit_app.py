"""Thali Live for Streamlit Community Cloud (free, ~1 GB): the same simulator + runtime as web/server.py, in a Streamlit page.

Deploy: share.streamlit.io -> New app -> repo Prashant-thakur77/THALI, branch main, main file web/streamlit_app.py
(requirements-streamlit.txt and packages.txt are picked up automatically; add SPEECHMATICS_API_KEY under Secrets).
Runs locally too:  MUJOCO_GL=glfw streamlit run web/streamlit_app.py

What this host cannot do: the OpenVINO Qwen2-VL planner and the PatchCore table check (no exported models, no openvino
here) -- the page says so and uses the rule planner; everything else (physics, expert, verifier, queues, audit chain,
Speechmatics on the bundled samples, barge-in) is the real code path.
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.environ.setdefault("MUJOCO_GL", "osmesa" if not os.environ.get("DISPLAY") and "MUJOCO_GL" not in os.environ else os.environ.get("MUJOCO_GL", "glfw"))
os.environ.setdefault("THALI_STREAM_EVERY", "8")
try:  # Streamlit Cloud secrets -> environment (no secrets file locally: .env is read by web.server)
    if "SPEECHMATICS_API_KEY" in st.secrets:
        os.environ["SPEECHMATICS_API_KEY"] = st.secrets["SPEECHMATICS_API_KEY"]
except Exception:
    pass

st.set_page_config(page_title="Thali Live", page_icon="🍽️", layout="wide")


@st.cache_resource(show_spinner="starting the simulator…")
def sim():
    from web.server import SIM   # one simulator + worker thread per process, shared by every session
    for _ in range(120):
        if SIM.rt is not None or any(e["kind"] == "error" for e in SIM.events):
            break
        time.sleep(1)
    return SIM


S = sim()

st.markdown("## Thali Live &nbsp;<span style='background:#7dd3fc;color:#0b0e14;font-weight:700;padding:2px 10px;border-radius:999px;font-size:12px'>two arms · voice · verifier · audit log</span>", unsafe_allow_html=True)
st.caption("Type a command or pick a preset / voice sample; both SO-101 arms execute it in the MuJoCo simulator while the plan, the verifier's "
           "verdict, every skill's checks and the sha256-chained audit records appear live. One simulator is shared by all visitors.")

left, right = st.columns([3, 2], gap="large")

with left:
    cam = st.empty()
    b1, b2, b3, _ = st.columns([1, 1, 1, 3])
    if b1.button("■ Stop (barge-in)", use_container_width=True) and S.rt is not None:
        S.rt.barge.on_partial("stop", S.rt.barge.operator); S._push({"kind": "bargein_button", "payload": {"text": "stop"}})
    if b2.button("▶ Continue", use_container_width=True) and S.rt is not None:
        S.rt.barge.on_partial("continue", S.rt.barge.operator); S._push({"kind": "bargein_button", "payload": {"text": "continue"}})
    if b3.button("⇄ Other arm", use_container_width=True) and S.rt is not None:
        S.rt.barge.on_partial("use the other arm", S.rt.barge.operator); S._push({"kind": "bargein_button", "payload": {"text": "use the other arm"}})

    st.markdown("#### Command")
    presets = {"drawer + plate": "open the top drawer, put the plate on the table with arm A", "set the table": "set the table",
               "pour a little": "hold the mug with arm B and pour a little water with arm A",
               "both arms at once": "open the top drawer with arm A and put the mug on its spot with arm B", "clear the table (follow-up)": "clear the table",
               "a bit more (follow-up)": "a bit more", "other side (follow-up)": "no, the other side"}
    pcols = st.columns(len(presets))
    for (name, text), c in zip(presets.items(), pcols):
        if c.button(name, use_container_width=True):
            st.session_state["cmd"] = text
            st.session_state["keep"] = "follow-up" in name
    cmd = st.text_input("command", key="cmd", value=st.session_state.get("cmd", presets["drawer + plate"]), label_visibility="collapsed")
    c1, c2, c3, c4 = st.columns([1, 1.4, 1.4, 1.4])
    seed = c1.number_input("seed", 0, 99, 3)
    split = c2.selectbox("layout", ["test", "train"], format_func=lambda x: "held-out layout" if x == "test" else "training layout")
    concurrent = c3.checkbox("both arms at once", value=False)
    keep = c4.checkbox("keep current scene", value=st.session_state.get("keep", False))
    samples = [p.name for p in sorted((ROOT / "voice" / "test_samples").glob("*.wav"))]
    voice = st.selectbox("or a Speechmatics voice sample", ["(none)"] + samples,
                         help="realtime transcription of the bundled recordings: normal, tired, Hindi, noisy room (operator focus ignores the other speaker)")
    run_disabled = S.busy or S.rt is None
    if st.button("Run", type="primary", disabled=run_disabled):
        job = {"command": None if voice != "(none)" else cmd, "voice": None if voice == "(none)" else voice, "planner": "rules",
               "seed": int(seed), "split": split, "concurrent": bool(concurrent), "anomaly": False, "keep_scene": bool(keep)}
        try:
            S.submit(job)
        except Exception as e:  # already busy
            st.warning(str(e))
    if not S.vlm_available:
        st.caption("This host runs the rule planner; the OpenVINO Qwen2-VL planner and the PatchCore table check need the exported models (see the README's live-site section).")

with right:
    st.markdown("#### Run")
    kpi = st.empty()
    st.markdown("#### Plan and verdict")
    plan_box = st.empty()
    st.markdown("#### Steps — oracle / camera")
    steps_box = st.empty()
    st.markdown("#### Live log — every line is a sha256-chained audit record")
    log_box = st.empty()


def render() -> None:
    if S.frame_jpeg:
        cam.image(S.frame_jpeg, use_container_width=True)
    evs = list(S.events)
    # last run only: from the last "command" event on
    start = max((i for i, e in enumerate(evs) if e["kind"] == "command"), default=0)
    run = evs[start:]
    plan = next((e["payload"] for e in reversed(run) if e["kind"] in ("plan", "replan")), None)
    if plan:
        plan_box.markdown(f"source **{plan.get('source')}** · verdict **{plan.get('verdict')}**\n\n```json\n{json.dumps(plan.get('steps'), indent=1)}\n```")
    else:
        plan_box.caption("no plan yet")
    skills = [e["payload"] for e in run if e["kind"] == "skill"]
    if skills:
        steps_box.table([{"#": p["step"], "skill": p["skill"] + (" ⇉" if p.get("concurrent") else ""), "arm": p["arm"].upper(),
                          "oracle": "✓" if p["oracle"] else "✗", "camera": "—" if p.get("camera") is None else ("✓" if p["camera"] else "✗"),
                          "s": p["seconds"]} for p in skills])
    else:
        steps_box.caption("no steps yet")
    done = next((e["payload"] for e in reversed(run) if e["kind"] == "done"), None)
    state = S.rt.state if S.rt else "STARTING"
    if done:
        lat = done.get("latency") or {}
        kpi.markdown(f"state **{state}** · success **{done['success']}** · speech end → arms moving **{lat.get('speech_end_to_arm_moves_s', '—')} s** · "
                     f"replans **{done['replans']}** · audit **{done['audit_records']}** records, head `{done['audit_head']}`")
    else:
        kpi.markdown(f"state **{state}**" + (" · running…" if S.busy else ""))
    lines = []
    for e in run[-40:]:
        p, k = e.get("payload", {}), e["kind"]
        head = f"`#{e['seq']} {e['sha256']}` " if "seq" in e else ""
        if k == "state":
            lines.append(f"{head}{p.get('from')} → **{p.get('to')}**" + (" · " + "; ".join(p["reasons"]) if p.get("reasons") else ""))
        elif k == "say":
            lines.append(f"🗣 {p['text']}")
        elif k == "skill":
            lines.append(f"{head}{p['skill']} (arm {p['arm'].upper()}) {'✓' if p['oracle'] else '✗'} · {p['seconds']} s")
        elif k in ("plan", "replan"):
            lines.append(f"{head}{k} from {p.get('source')} · verdict {p.get('verdict')}")
        elif k == "heard":
            lines.append(f"heard (operator {p.get('operator')}): “{p['text']}”" + (f" · ignored: “{p['ignored'][0]}”" if p.get("ignored") else ""))
        elif k == "listening":
            lines.append(f"🎤 Speechmatics realtime on {p['file']} ({p['language']})")
        elif k == "final_check":
            lines.append(f"final check: {', '.join(p['missing'])} not in place → redoing {len(p['redo'])} step(s)")
        elif k == "bargein_button":
            lines.append(f"🎤 (partial) “{p['text']}”")
        elif k == "error":
            lines.append(f"⚠️ {p['text']}")
        elif k == "result":
            lines.append(f"{head}result: {'success' if p['success'] else 'not all goals met'} · {p['sim_steps']} sim steps")
    log_box.markdown("\n\n".join(lines) if lines else "_idle_")


@st.fragment(run_every=0.6)
def live():
    render()


live()
