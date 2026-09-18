"""End-to-end check of a deployed Thali Live (Streamlit) page with a real browser.

Opens the URL, waits for the simulator to report ready, presses Run on the default command, waits for the run to finish,
and saves screenshots + a JSON report.  Exit code 0 only when the run reports success.

    python -m web.selftest https://thali-live.streamlit.app --out video/selftest
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("url")
    ap.add_argument("--out", type=Path, default=Path("video/selftest"))
    ap.add_argument("--timeout", type=int, default=420, help="seconds to wait for the run to finish")
    ap.add_argument("--verbose", action="store_true", help="print the page state while waiting")
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    from playwright.sync_api import sync_playwright

    report = {"url": a.url, "t0": time.time()}
    with sync_playwright() as pw:
        browser = pw.chromium.launch()
        page = browser.new_page(viewport={"width": 1400, "height": 1100})
        page.goto(a.url, wait_until="domcontentloaded", timeout=120_000)
        # Streamlit Community Cloud shows a wake-up page for a sleeping app: press its button if present
        try:
            page.get_by_role("button", name=re.compile("get this app back up", re.I)).click(timeout=5_000)
        except Exception:
            pass
        # Streamlit Community Cloud embeds the app in an iframe; a local server does not
        page.wait_for_timeout(3000)
        app = next((f for f in page.frames if f != page.main_frame and "streamlit" in (f.url or "")), None) or page.main_frame
        # the header KPI says "simulator ready" once the worker has built the scene
        app.wait_for_selector("text=simulator ready", timeout=240_000)
        page.screenshot(path=str(a.out / "1_ready.png"), full_page=True)
        report["ready_s"] = round(time.time() - report["t0"], 1)
        report["renderer"] = (re.search(r"renderer:? ([a-z]+)", app.inner_text("body")) or [None, None])[1]
        if "RUNNING" in app.inner_text("body"):   # someone else's run: wait for it to finish first
            app.wait_for_selector("text=IDLE", timeout=a.timeout * 1000)
        before = (re.search(r"head ([0-9a-f]{8,16})", app.inner_text("body")) or [None, None])[1]   # last run's audit head
        app.get_by_role("button", name="Run", exact=True).first.click(timeout=30_000)
        report["clicked_run"] = True
        deadline = time.time() + a.timeout
        outcome = None
        frames: list[str] = []          # stream frames seen (distinct = the page really refreshes during the run)
        t_run = None
        while time.time() < deadline:
            body = app.inner_text("body")
            running = "RUNNING" in body or bool(re.search(r"^state (?!DONE|IDLE)[A-Z_]+", body, re.M))   # header KPI or the live state line
            if running and t_run is None:
                t_run = time.time()
            try:
                src = app.locator('[data-testid="stImage"] img').first.get_attribute("src", timeout=1000) or ""   # the camera frame
                if src and (not frames or frames[-1] != src):
                    frames.append(src)
            except Exception:
                pass
            m = re.search(r"done · success (True|False)", body)
            head = (re.search(r"head ([0-9a-f]{8,16})", body) or [None, None])[1]
            if a.verbose:
                print(f"t+{time.time() - report['t0']:.0f}s running={running} head={head} done={m.group(0) if m else None} frames={len(frames)}", flush=True)
            if m and not running and head != before:      # a new run finished (not the previous run's status)
                outcome = m.group(1) == "True"
                break
            if "⚠️" in body and "simulator failed" in body:
                outcome = False
                break
            time.sleep(0.5)
        run_s = round(time.time() - t_run, 1) if t_run else None
        report["run_s"] = run_s
        report["distinct_frames"] = len(frames)
        report["frames_per_s"] = round(len(frames) / run_s, 2) if run_s else None
        page.screenshot(path=str(a.out / "2_after_run.png"), full_page=True)
        body = app.inner_text("body")
        report.update({"success": outcome, "elapsed_s": round(time.time() - report["t0"], 1),
                       "steps_seen": len(re.findall(r"\(arm [AB]\) [✓✗]", body)),
                       "plan_verdict": (re.search(r"verdict (ALLOW|REORDER|BLOCK)", body) or [None, None])[1],
                       "audit_head": (re.search(r"head ([0-9a-f]{8,16})", body) or [None, None])[1]})
        browser.close()
    (a.out / "report.json").write_text(json.dumps(report, indent=1))
    print(json.dumps(report, indent=1))
    return 0 if report.get("success") else 1


if __name__ == "__main__":
    sys.exit(main())
