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
        report["renderer"] = (re.search(r"renderer: ([a-z]+)", app.inner_text("body")) or [None, None])[1]
        if "RUNNING" in app.inner_text("body"):   # someone else's run: wait for it to finish first
            app.wait_for_selector("text=IDLE", timeout=a.timeout * 1000)
        app.get_by_role("button", name="Run", exact=True).first.click(timeout=30_000)
        report["clicked_run"] = True
        deadline = time.time() + a.timeout
        outcome = None
        while time.time() < deadline:
            body = app.inner_text("body")
            m = re.search(r"done · success (True|False)", body)
            if m:
                outcome = m.group(1) == "True"
                break
            if "⚠️" in body and "simulator failed" in body:
                outcome = False
                break
            time.sleep(5)
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
