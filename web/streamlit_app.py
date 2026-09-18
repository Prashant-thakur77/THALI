"""Thali Live — the whole closed loop in a browser (Streamlit Community Cloud, ~1 GB, free).

Tabs: Live (run commands, watch both arms, step checks, audit records) · Speak (your microphone → Speechmatics → command) ·
Plan sandbox (command → parse → plan → verdict, no execution) · Verifier (paste a plan; the 20 injected unsafe plans) ·
Audit log (recompute the sha256 chain, tamper with a copy) · Results (every results file) · Replay (recorded runs + video) · About.

Deploy: share.streamlit.io → New app → repo Prashant-thakur77/THALI, branch main, main file web/streamlit_app.py
(web/requirements.txt and web/packages.txt are picked up automatically; add SPEECHMATICS_API_KEY under Secrets).
Runs locally too:  MUJOCO_GL=glfw streamlit run web/streamlit_app.py

What this host cannot do: the OpenVINO Qwen2-VL planner and the PatchCore table check (no exported models, no openvino
here) -- the page says so and uses the rule planner; everything else (physics, expert, verifier, queues, audit chain,
Speechmatics, barge-in, follow-ups, both-arms-at-once) is the real code path.
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
import time
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.environ.setdefault("MUJOCO_GL", "osmesa" if not os.environ.get("DISPLAY") and "MUJOCO_GL" not in os.environ else os.environ.get("MUJOCO_GL", "glfw"))
# a 1-CPU cloud host renders with software GL: small frames, every 10th control step, light JPEG (~5 fps, physics keeps pace)
os.environ.setdefault("THALI_STREAM_EVERY", "10")
os.environ.setdefault("THALI_STREAM_W", "320")
os.environ.setdefault("THALI_STREAM_H", "240")
os.environ.setdefault("THALI_STREAM_Q", "70")
try:  # Streamlit Cloud secrets -> environment (no secrets file locally: .env is read by web.server)
    if "SPEECHMATICS_API_KEY" in st.secrets:
        os.environ["SPEECHMATICS_API_KEY"] = st.secrets["SPEECHMATICS_API_KEY"]
except Exception:
    pass

st.set_page_config(page_title="Thali Live", page_icon="🍽️", layout="wide", menu_items={
    "Get help": "https://github.com/Prashant-thakur77/THALI", "About": "Thali: two SO-101 arms set and clear a table from voice, verified before they move."})

R = ROOT / "results"
FULL = ("drawer_open", "plate_placed", "fork_placed", "spoon_placed", "mug_placed", "poured")


def load(name: str):
    p = R / name
    return json.loads(p.read_text()) if p.exists() else None


@st.cache_resource(show_spinner="starting the simulator…")
def sim():
    from web.server import SIM   # one simulator + worker thread per process, shared by every session
    for _ in range(180):
        if SIM.rt is not None or any(e["kind"] == "error" for e in SIM.events):
            break
        time.sleep(1)
    return SIM


try:
    S = sim()
except Exception as e:   # show the real reason on the page instead of Streamlit's generic error screen
    import traceback
    st.error(f"Thali Live could not start: {type(e).__name__}: {e}")
    st.code(traceback.format_exc())
    st.stop()
ready = S.rt is not None
for _e in S.events:
    if _e["kind"] == "error":
        st.error(_e["payload"]["text"])
speech_on = bool(os.environ.get("SPEECHMATICS_API_KEY"))

# ----------------------------------------------------------------------------- header
st.markdown("""<style>
.kpi b{font-size:26px;color:#7dd3fc;display:block;line-height:1.1} .kpi span{color:#8b93a3;font-size:12px}
.ev{padding:4px 8px;border-left:3px solid #232a36;margin:3px 0;background:#0f1420;border-radius:0 6px 6px 0;font-size:13px}
.ev.ok{border-color:#34d399} .ev.bad{border-color:#f87171} .ev.plan{border-color:#fbbf24} .ev.say{border-color:#a78bfa} .ev.state{border-color:#7dd3fc}
.h{color:#8b93a3;font-size:11px;margin-right:6px;font-family:ui-monospace,Menlo,monospace}
</style>""", unsafe_allow_html=True)
hl, hr = st.columns([3, 2])
with hl:
    st.markdown("## Thali Live &nbsp;<span style='background:#7dd3fc;color:#0b0e14;font-weight:700;padding:2px 10px;border-radius:999px;font-size:12px'>two arms · voice · verifier · audit log</span>", unsafe_allow_html=True)
    st.caption("Two SO-101 arms set and clear a dinner table from spoken commands. Every plan is verified before anything moves, every step is checked, "
               "and every record is sha256-chained. One simulator is shared by all visitors; a run takes 1–4 minutes.")
with hr:
    exp = load("seeds_expert_expert_test.json") or {}
    cl = load("clear_table.json") or {}
    hm = load("heatmap_expert.json") or {}
    k1, k2, k3, k4 = st.columns(4)
    k1.markdown(f"<div class=kpi><b>{exp.get('successes', '–')}/10</b><span>full task, held-out</span></div>", unsafe_allow_html=True)
    k2.markdown(f"<div class=kpi><b>{cl.get('successes', '–')}/10</b><span>clear the table</span></div>", unsafe_allow_html=True)
    k3.markdown(f"<div class=kpi><b>{int(100 * hm.get('per_axis_success_rate', {}).get('all', 0))}%</b><span>all 6 perturbations</span></div>", unsafe_allow_html=True)
    k4.markdown(f"<div class=kpi><b>{'RUNNING' if S.busy else 'IDLE'}</b><span>simulator {'ready' if ready else 'starting'} · renderer {getattr(S, 'gl_backend', '?')}</span></div>", unsafe_allow_html=True)

tab_live, tab_speak, tab_plan, tab_verify, tab_audit, tab_results, tab_replay, tab_about = st.tabs(
    ["▶ Live", "🎤 Speak", "🧠 Plan sandbox", "🛡 Verifier", "🔗 Audit log", "📊 Results", "🎬 Replay", "ℹ About"])


# ----------------------------------------------------------------------------- helpers
def barge(text: str) -> None:
    if S.rt is not None:
        S.rt.barge.on_partial(text, S.rt.barge.operator)
        S._push({"kind": "bargein_button", "payload": {"text": text}})


def submit(job: dict) -> None:
    try:
        S.submit(job)
        st.toast("queued — watch the cameras")
    except Exception as e:
        st.warning(f"the simulator is busy: {e}")


def last_run_events() -> list[dict]:
    evs = list(S.events)
    start = max((i for i, e in enumerate(evs) if e["kind"] == "command"), default=0)
    return evs[start:]


def event_line(e: dict) -> tuple[str, str]:
    p, k = e.get("payload", {}), e["kind"]
    head = f"<span class=h>#{e['seq']} {e['sha256']}</span>" if "seq" in e else ""
    if k == "state":
        return "state", f"{head}{p.get('from')} → <b>{p.get('to')}</b>" + (" · " + "; ".join(p["reasons"]) if p.get("reasons") else "") + (" · both arms" if p.get("concurrent") else "")
    if k == "say":
        return "say", f"🗣 {p['text']}"
    if k == "skill":
        won = p["detail"].get("won_by") if isinstance(p.get("detail"), dict) else None
        return ("ok" if p["oracle"] else "bad"), f"{head}{p['skill']} (arm {p['arm'].upper()}) {'✓' if p['oracle'] else '✗'} · {p['seconds']} s" + (f" · won by {won}" if won else "")
    if k in ("plan", "replan"):
        return "plan", f"{head}{k} from <b>{p.get('source')}</b> · verdict <b>{p.get('verdict')}</b> · {len(p.get('steps', []))} steps"
    if k == "followup":
        return "plan", f"{head}correction “{p.get('kind')}” resolved against the last step ({p.get('resolved_from', {}).get('skill')})"
    if k == "heard":
        return "say", f"heard (operator {p.get('operator')}): “{p['text']}”" + (f" · ignored other speaker: “{p['ignored'][0]}”" if p.get("ignored") else "")
    if k == "listening":
        return "say", f"🎤 Speechmatics realtime on {p['file']} ({p['language']})"
    if k == "parsed":
        return "say", f"parsed → “{p.get('command')}”"
    if k == "final_check":
        return "plan", f"final check: {', '.join(p['missing'])} not in place → redoing {len(p['redo'])} step(s)"
    if k == "bargein_button":
        return "say", f"🎤 (partial) “{p['text']}”"
    if k == "result":
        return ("ok" if p["success"] else "bad"), f"{head}result: {'success' if p['success'] else 'not all goals met'} · {p['sim_steps']} sim steps · replans {p['replans']}"
    if k == "done":
        return ("ok" if p["success"] else "bad"), f"done · success {p['success']} · audit records {p.get('audit_records')}, head {p.get('audit_head')}"
    if k == "error":
        return "bad", f"⚠️ {p['text']}"
    if k == "ready":
        return "state", "simulator ready"
    if k == "gl":
        return "state", f"renderer: {p.get('backend')}" + ("" if p.get("cameras") else " — no headless GL on this host, cameras off (scene state still live)")
    return "", ""


def render_run(cam, kpi, plan_box, steps_box, log_box, scene_box) -> None:
    if S.frame_jpeg:
        cam.image(S.frame_jpeg, use_container_width=True)
    elif ready and not getattr(S, "cameras", True):
        cam.info("No headless GL renderer is available on this host, so the camera views are off here; the scene inspector below is live and the "
                 "lab-machine site (landing page, first button) streams the cameras.")
    else:
        cam.caption("the cameras appear as soon as the simulator has built the scene…")
    run = last_run_events()
    plan = next((e["payload"] for e in reversed(run) if e["kind"] in ("plan", "replan")), None)
    if plan:
        plan_box.markdown(f"source **{plan.get('source')}** · verdict **{plan.get('verdict')}**\n\n```json\n{json.dumps(plan.get('steps'), indent=1)}\n```")
    else:
        plan_box.caption("no plan yet — run a command")
    skills = [e["payload"] for e in run if e["kind"] == "skill"]
    if skills:
        steps_box.table([{"#": p["step"], "skill": p["skill"] + (" ⇉" if p.get("concurrent") else ""), "arm": p["arm"].upper(),
                          "oracle": "✓" if p["oracle"] else "✗", "camera": "—" if p.get("camera") is None else ("✓" if p["camera"] else "✗"),
                          "won by": (p.get("detail") or {}).get("won_by") or "expert", "s": p["seconds"]} for p in skills])
    else:
        steps_box.caption("no steps yet")
    done = next((e["payload"] for e in reversed(run) if e["kind"] == "done"), None)
    state = S.rt.state if S.rt else "STARTING"
    if done:
        lat = done.get("latency") or {}
        kpi.markdown(f"state **{state}** · success **{done['success']}** · speech end → arms moving **{lat.get('speech_end_to_arm_moves_s', '—')} s** · "
                     f"replans **{done['replans']}** · sim steps **{done['sim_steps']}** · audit **{done['audit_records']}** records, head `{done['audit_head']}`")
    else:
        kpi.markdown(f"state **{state}**" + (" · running…" if S.busy else ""))
    lines = []
    for e in run[-45:]:
        cls, text = event_line(e)
        if text:
            lines.append(f"<div class='ev {cls}'>{text}</div>")
    log_box.markdown("".join(lines) if lines else "<div class=ev>idle — pick a preset and press Run</div>", unsafe_allow_html=True)
    if S.env is not None:   # scene inspector: what the robot's own state says right now
        try:
            from souschef_env import oracles
            from souschef_env.scene_description import scene_state
            stt = scene_state(S.env.model, S.env.data)
            sg = oracles.subgoals(S.env.model, S.env.data)
            objs = [{"object": n, "x cm": o["x_cm"], "y cm": o["y_cm"], "z cm": o["z_cm"], "held by": o["held_by"] or "—", "in drawer": "yes" if o["in_drawer"] else ""} for n, o in stt["objects"].items()]
            goals = " · ".join(f"{'✅' if sg[g] else '⬜'} {g.replace('_', ' ')}" for g in FULL)
            scene_box[0].markdown(f"**drawer** {'open' if stt['drawer']['open'] else 'closed'} ({stt['drawer']['travel_cm']} cm) · **water in mug** {stt['water_in_mug']}/20  \n{goals}")
            scene_box[1].dataframe(objs, hide_index=True, use_container_width=True, height=250)
        except Exception as e:  # pragma: no cover
            scene_box[0].caption(f"scene inspector unavailable: {e}")
    else:
        scene_box[0].caption("simulator starting…")
        scene_box[1].empty()


# ----------------------------------------------------------------------------- Live
with tab_live:
    left, right = st.columns([3, 2], gap="large")
    with left:
        cam = st.empty()
        b1, b2, b3, _ = st.columns([1, 1, 1, 2])
        if b1.button("■ Stop (barge-in)", use_container_width=True, help="injects the partial transcript 'stop' — the arm pauses within one 20 ms control step"):
            barge("stop")
        if b2.button("▶ Continue", use_container_width=True):
            barge("continue")
        if b3.button("⇄ Other arm", use_container_width=True):
            barge("use the other arm")
        st.markdown("#### Command")
        presets = {"drawer + plate": "open the top drawer, put the plate on the table with arm A", "set the table": "set the table",
                   "pour a little": "hold the mug with arm B and pour a little water with arm A",
                   "both arms at once": "open the top drawer with arm A and put the mug on its spot with arm B",
                   "clear the table ↩": "clear the table", "a bit more ↩": "a bit more", "other side ↩": "no, the other side", "again ↩": "again"}
        cols = st.columns(4)
        for i, (name, text) in enumerate(presets.items()):
            if cols[i % 4].button(name, use_container_width=True, key=f"preset_{i}"):
                st.session_state["cmd"] = text
                st.session_state["keep"] = name.endswith("↩")
        st.caption("↩ = follow-up on the current scene (keep current scene).")
        cmd = st.text_input("command", key="cmd", value=st.session_state.get("cmd", presets["drawer + plate"]), label_visibility="collapsed",
                            placeholder="e.g. open the top drawer, put the plate on the table with arm A")
        c1, c2, c3, c4 = st.columns([1, 1.4, 1.4, 1.4])
        seed = c1.number_input("seed", 0, 99, 3)
        split = c2.selectbox("layout", ["test", "train"], format_func=lambda x: "held-out layout" if x == "test" else "training layout")
        concurrent = c3.checkbox("both arms at once", value=False, help="independent single-arm steps run as two skill threads through one physics step barrier")
        keep = c4.checkbox("keep current scene", value=st.session_state.get("keep", False), help="follow-ups and 'clear the table' act on the table as it is now")
        samples = [p.name for p in sorted((ROOT / "voice" / "test_samples").glob("*.wav"))]
        voice = st.selectbox("or a bundled Speechmatics recording", ["(none)"] + samples,
                             help="realtime transcription of the bundled recordings: normal, tired, Hindi, noisy room (the other speaker in the noisy room is ignored)")
        r1, r2 = st.columns([1, 2])
        if r1.button("Run", type="primary", disabled=(S.busy or not ready), use_container_width=True):
            submit({"command": None if voice != "(none)" else cmd, "voice": None if voice == "(none)" else voice, "planner": "rules",
                    "seed": int(seed), "split": split, "concurrent": bool(concurrent), "anomaly": False, "keep_scene": bool(keep)})
        if r2.button("🎬 Guided tour: set the table → a bit more → clear the table", disabled=(S.busy or not ready), use_container_width=True):
            S.enqueue([{"command": "set the table", "seed": int(seed), "split": split, "planner": "rules"},
                       {"command": "a bit more", "seed": int(seed), "split": split, "planner": "rules", "keep_scene": True},
                       {"command": "clear the table", "seed": int(seed), "split": split, "planner": "rules", "keep_scene": True}])
            st.toast("tour queued: three commands back to back (~6 minutes)")
        if not S.vlm_available:
            st.caption("This host runs the rule planner; the OpenVINO Qwen2-VL planner and the PatchCore table check need the exported models and run on the lab machine (see the README).")
        st.markdown("#### What the robot's state says right now")
        scene_box = (st.empty(), st.empty())   # goals line + object table, refreshed from the fragment
    with right:
        st.markdown("#### Run")
        kpi = st.empty()
        st.markdown("#### Plan and verdict")
        plan_box = st.empty()
        st.markdown("#### Steps — oracle / camera / who won")
        steps_box = st.empty()
        st.markdown("#### Live log — every line is a sha256-chained audit record")
        log_box = st.empty()
        audit_path = R / "audit_web.jsonl"
        if audit_path.exists():
            st.download_button("⬇ download this site's audit log (JSONL, hash-chained)", audit_path.read_bytes(), file_name="thali_audit_web.jsonl", mime="application/json")

    # write every placeholder once during the full run (Streamlit reserves their slots), then keep them fresh from a fragment
    render_run(cam, kpi, plan_box, steps_box, log_box, scene_box)

    @st.fragment(run_every=0.25)   # ≤4 page refreshes/s; the worker captures a frame every THALI_STREAM_EVERY control steps
    def live():
        render_run(cam, kpi, plan_box, steps_box, log_box, scene_box)

    live()


# ----------------------------------------------------------------------------- Speak
with tab_speak:
    st.markdown("#### Talk to it")
    st.caption("Record a command with your microphone; it goes through Speechmatics realtime (partials, end-of-turn, diarization) "
               "→ the parser → the planner → the verifier → the arms, exactly like the bundled samples. Try: "
               "*“open the top drawer and put the plate on the table with arm A”*, *“hold the mug with arm B and pour a little water”*, "
               "or in Hindi: *“top drawer kholo, plate ko arm A se uthao, table pe rakh do”*.")
    if not speech_on:
        st.warning("Speechmatics is not configured on this host (SPEECHMATICS_API_KEY missing); the bundled recordings in the Live tab need it too.")
    lang = st.radio("language", ["en", "hi"], horizontal=True, format_func=lambda x: {"en": "English", "hi": "Hindi (Devanagari transcript, Hinglish understood)"}[x])
    audio = st.audio_input("record your command")
    s1, s2 = st.columns([1, 2])
    mic_seed = s2.number_input("seed for this run", 0, 99, 3, key="mic_seed")
    if s1.button("Transcribe and run", type="primary", disabled=(audio is None or S.busy or not ready or not speech_on)):
        tmp = Path(tempfile.gettempdir()) / f"thali_mic_{int(time.time())}.wav"
        tmp.write_bytes(audio.getvalue())
        submit({"voice_path": str(tmp), "language": lang, "planner": "rules", "seed": int(mic_seed), "split": "test", "keep_scene": False})
    heard = next((e["payload"] for e in reversed(S.events) if e["kind"] == "heard"), None)
    if heard:
        st.markdown(f"**last transcript** (operator {heard.get('operator')}): “{heard['text']}”")
        if heard.get("ignored"):
            st.caption("ignored other speaker: " + " | ".join(heard["ignored"]))
    st.markdown("Bundled recordings, with the transcript the parser saw:")
    tx = {}
    tp = ROOT / "voice" / "test_samples" / "transcripts.txt"
    if tp.exists():
        for line in tp.read_text().splitlines():
            if "\t" in line:
                k, v = line.split("\t", 1)
                tx[k.strip()] = v.strip()
    for wav in sorted((ROOT / "voice" / "test_samples").glob("*.wav")):
        a1, a2 = st.columns([1, 3])
        a1.audio(wav.read_bytes(), format="audio/wav")
        a2.markdown(f"**{wav.name}** — {tx.get(wav.name, '')}")


# ----------------------------------------------------------------------------- Plan sandbox
with tab_plan:
    st.markdown("#### From words to a verified plan — without moving the arms")
    st.caption("The same parser, planner and verifier the runtime uses, on the current scene. Instant.")
    text = st.text_input("command", "open the drawer, put a fork and a spoon by the plate, then pour me some water", key="sandbox_cmd")
    if ready and text:
        from planner.plan import rule_plan
        from souschef_env.scene_description import describe, scene_state
        from verifier.rules import Verifier, World
        from voice.parser import intents_to_command, parse
        pr = parse(text)
        canonical = intents_to_command(pr.intents) or text
        stt = scene_state(S.env.model, S.env.data)
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**1 · parser** (ASR-tolerant, Hinglish, Devanagari)")
            st.json([i.as_dict() for i in pr.intents])
            st.markdown(f"canonical command → `{canonical}`")
            if pr.diagnostics:
                st.caption("; ".join(pr.diagnostics))
            st.markdown("**2 · what the planner is told about the scene**")
            st.code(describe(S.env.model, S.env.data), language="text")
        with c2:
            st.markdown("**3 · plan** (rule planner on this host; the VLM proposes on the lab machine)")
            try:
                plan = rule_plan(canonical, stt)
                st.json(plan)
                st.markdown("**4 · verifier**")
                v = Verifier().verify(plan, World.from_scene_state(stt))
                colour = {"ALLOW": "green", "REORDER": "orange", "BLOCK": "red"}.get(v.verdict, "gray")
                st.markdown(f"verdict :{colour}[**{v.verdict}**]")
                for i in v.issues:
                    st.markdown(f"- `{i.code}` step {i.step}: {i.message}")
                if v.verdict == "REORDER":
                    st.markdown("fixed plan:")
                    st.json(v.plan)
            except Exception as e:
                st.error(f"planner: {e}")


# ----------------------------------------------------------------------------- Verifier
with tab_verify:
    st.markdown("#### Paste any plan; the verifier decides before anything moves")
    default_plan = json.dumps({"steps": [{"skill": "pour", "arm": "a", "amount": "normal"}, {"skill": "hold_mug", "arm": "b"}], "mode": "normal"}, indent=1)
    js = st.text_area("plan JSON", default_plan, height=160)
    if ready and st.button("Verify"):
        from verifier.rules import Verifier, World
        from souschef_env.scene_description import scene_state
        try:
            plan = json.loads(js)
            v = Verifier().verify(plan, World.from_scene_state(scene_state(S.env.model, S.env.data)))
            colour = {"ALLOW": "green", "REORDER": "orange", "BLOCK": "red"}.get(v.verdict, "gray")
            st.markdown(f"verdict :{colour}[**{v.verdict}**]")
            for i in v.issues:
                st.markdown(f"- `{i.code}` step {i.step}: {i.message}")
            if v.verdict == "REORDER":
                st.json(v.plan)
        except json.JSONDecodeError as e:
            st.error(f"invalid JSON: {e}")
    inj = load("verifier_injection.json")
    if inj:
        st.markdown(f"#### The {inj['bad_plans']} injected unsafe plans — {inj['caught']} caught, {inj['good_passed']}/{inj['good_plans']} sane plans passed")
        st.dataframe([{"plan": r["name"], "verdict": r.get("verdict"), "reason": (r.get("reasons") or [""])[0], "caught": "✓" if r.get("caught") else ""} for r in inj["rows"]],
                     hide_index=True, use_container_width=True)


# ----------------------------------------------------------------------------- Audit log
with tab_audit:
    st.markdown("#### Tamper-evident by construction")
    st.caption("Every plan, verdict and skill result is a JSON line carrying the sha256 of the previous line. `make verify-log` recomputes the chain; edit, delete or reorder one record and it names the first bad one.")
    from verifier.audit import verify as verify_chain
    choices = {"this site's log (results/audit_web.jsonl)": R / "audit_web.jsonl", "the shipped evaluation log (results/audit.jsonl)": R / "audit.jsonl"}
    available = [k for k, p in choices.items() if p.exists()]
    if available:
        which = st.selectbox("log", available)
        path = choices[which]
        rep = verify_chain(path)
        st.markdown(f"chain **{'intact ✅' if rep['ok'] else 'BROKEN ❌'}** · {rep.get('records', '?')} records" + ("" if rep["ok"] else f" · first bad seq {rep.get('first_bad_seq')}: {rep.get('reason')}"))
        if st.button("Tamper with a copy (flip one character in record 3) and re-verify"):
            lines = path.read_text().splitlines()
            if len(lines) > 3:
                bad = lines[:]
                bad[3] = bad[3].replace('"kind": "', '"kind": "x', 1)
                tmp = Path(tempfile.gettempdir()) / "thali_audit_tampered.jsonl"
                tmp.write_text("\n".join(bad) + "\n")
                rep2 = verify_chain(tmp)
                st.markdown(f"tampered copy: **{'intact' if rep2['ok'] else 'BROKEN ❌'}** — first bad seq {rep2.get('first_bad_seq')}: {rep2.get('reason')}")
        recs = [json.loads(l) for l in path.read_text().splitlines()[-40:]]
        st.dataframe([{"seq": r["seq"], "kind": r["kind"], "summary": json.dumps(r["payload"])[:110], "sha256": r["sha256"][:16], "prev": r["prev_hash"][:16]} for r in recs],
                     hide_index=True, use_container_width=True, height=420)
    else:
        st.caption("no audit log yet — run a command in the Live tab")


# ----------------------------------------------------------------------------- Results
with tab_results:
    st.markdown("#### Every number, from the file that produced it")
    st.caption("The README is rendered from these files by `python -m docs.render_readme`; nothing is typed by hand.")
    c1, c2 = st.columns(2)
    with c1:
        rows = []
        for sp in ("test", "train"):
            d = load(f"seeds_expert_expert_{sp}.json")
            if d:
                rows.append({"run": f"scripted expert, {sp} split", "success": f"{d['successes']}/{d['seeds']}", **{k.replace('_', ' '): f"{int(100 * d['per_subgoal_rate'][k])}%" for k in FULL}})
        for pol in ("act", "smolvla"):
            for mode in ("policy_only", "policy_retry", "policy_fallback"):
                d = load(f"seeds_{pol}_{mode}_test.json")
                if d:
                    rows.append({"run": f"{pol} {mode.replace('_', ' ')}", "success": f"{d['successes']}/{d['seeds']}", **{k.replace('_', ' '): f"{int(100 * d['per_subgoal_rate'][k])}%" for k in FULL}})
        st.markdown("**Full task, 10 seeds**")
        st.dataframe(rows, hide_index=True, use_container_width=True)
        for tag, label in (("act_50k", "ACT, 50k steps"), ("act_plate_50k", "ACT plate, 50k steps"), ("act_1050ep", "ACT, 12k steps, 150 ep"), ("act_60ep", "ACT, 12k steps, 60 ep"), ("smolvla_14000", "SmolVLA step 14 000")):
            d = load(f"skill_eval_{tag}.json")
            if d:
                st.markdown(f"**Per skill, policy only — {label}**")
                st.dataframe([{"skill": k, "success": f"{v['successes']}/{v['total']}", "median cm from zone": v.get("median_zone_error_cm")} for k, v in d["skills"].items()], hide_index=True, use_container_width=True)
    with c2:
        if (R / "heatmap_expert.png").exists():
            st.image(str(R / "heatmap_expert.png"), caption="robustness: held-out seeds × perturbation axis (scripted expert)")
        b = load("bench.json")
        if b:
            st.markdown(f"**OpenVINO latency** — {b.get('caption', '')}")
            st.dataframe([{"precision / device": k, "mean p50 ms": v["mean_p50_ms"], "skills": v["skills"]} for k, v in b["summary"].items()], hide_index=True, use_container_width=True)
        for name, label in (("concurrency.json", "both arms at once"), ("recovery.json", "mid-task recovery"), ("followups.json", "corrections"), ("clear_table.json", "clear the table"),
                            ("pour_amount.json", "target-volume pour"), ("anomaly_diffreal.json", "Anomalib table check"), ("voice_test.json", "voice"), ("planner_eval.json", "planner"), ("instruction_swap.json", "instruction swap")):
            d = load(name)
            if d:
                head = {k: v for k, v in d.items() if isinstance(v, (int, float, str, bool))}
                st.markdown(f"**{label}** — `results/{name}`")
                st.json(head, expanded=False)


# ----------------------------------------------------------------------------- Replay
with tab_replay:
    st.video("https://huggingface.co/spaces/Prashant-77/thali/resolve/main/thali_demo.mp4")   # streamed, not served from this 1 GB host
    st.caption("The demo: the problem, the closed loop, a real noisy-room recording, Hindi, the full task, the pour, blocked plans, barge-in, clearing the table, corrections, both arms at once, the learned policies, OpenVINO, both track scorecards.")
    st.markdown("#### Recorded runs")
    for name in ("demo_seed3.json", "demo_seed0.json", "demo_bargein_stop.json"):
        d = load(name)
        if not d:
            continue
        with st.expander(f"{name} — “{str(d.get('command', ''))[:80]}” · success {d.get('success')} · latency {json.dumps(d.get('latency'))}"):
            st.dataframe([{"skill": s["step"]["skill"], "arm": s["arm"], "oracle": "✓" if s["oracle_ok"] else "✗", "camera": s.get("camera_ok"),
                           "seconds": round(s["t1"] - s["t0"], 1) if "t1" in s and "t0" in s else None} for s in d.get("steps", [])], hide_index=True, use_container_width=True)
            if d.get("barge_ins"):
                st.json(d["barge_ins"])


# ----------------------------------------------------------------------------- About
with tab_about:
    if (ROOT / "docs" / "architecture.png").exists():
        st.image(str(ROOT / "docs" / "architecture.png"))
    st.markdown("""
**Closed loop.** Speechmatics realtime speech (partials, end-of-turn 0.6 s, speaker diarization → only the operator is obeyed, custom vocabulary, Hindi) →
parser → local Qwen2-VL-2B INT4 planner on OpenVINO (overhead frame + scene text → JSON) → deterministic verifier (reachability, grasp preconditions,
drawer before cutlery, never pour unless the other arm holds the mug; ALLOW / REORDER / BLOCK; sha256-chained audit log) → per-arm queues (independent steps
run on both arms at once) → skills (ACT / SmolVLA policies → retry → scripted IK expert) → camera check + simulator oracle + Anomalib PatchCore table check →
next step, replan, or a final-state verification that redoes what moved → Speechmatics TTS.

**Hardware.** Every number was measured on an Intel Core i7-13650HX + UHD iGPU laptop with OpenVINO (no NPU). This page runs on a 1 GB CPU host with the
rule planner; the OpenVINO planner and the PatchCore check run on the lab machine.

**Links.** [GitHub](https://github.com/Prashant-thakur77/THALI) · [dataset (1050 episodes)](https://huggingface.co/datasets/Prashant-77/thali_all) ·
[SmolVLA checkpoint](https://huggingface.co/Prashant-77/thali_smolvla) · [video, deck, write-up](https://huggingface.co/spaces/Prashant-77/thali)
""")
