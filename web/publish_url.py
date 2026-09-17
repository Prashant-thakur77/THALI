"""Point the free static Space (the stable landing page) at the current live URL of Thali Live.

The interactive site runs on the lab machine behind a Cloudflare quick tunnel whose hostname changes on every
restart; this script rewrites the "Open Thali Live" button on https://huggingface.co/spaces/Prashant-77/thali.

    python -m web.publish_url https://xxxx.trycloudflare.com
"""

from __future__ import annotations

import os
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def main() -> None:
    url = sys.argv[1].rstrip("/")
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env")
    from huggingface_hub import HfApi
    p = ROOT / "hosting" / "static" / "index.html"
    s = p.read_text()
    button = (f'<p id="live"><a href="{url}" style="display:inline-block;background:#7dd3fc;color:#0b0e14;font-weight:800;padding:12px 20px;'
              f'border-radius:10px;text-decoration:none;font-size:18px">▶ Open Thali Live — drive the simulator yourself</a>'
              f' <span style="color:#8b93a3;font-size:13px">served from the lab machine; if it does not answer, the machine is off — the video below shows the same loop</span></p>')
    if 'id="live"' in s:
        s = re.sub(r'<p id="live">.*?</p>', button, s, flags=re.S)
    else:
        s = s.replace("<h2>Demo (4:23)", button + "\n<h2>Demo (4:23)", 1)
    p.write_text(s)
    HfApi(token=os.environ["HF_TOKEN"]).upload_file(path_or_fileobj=str(p), path_in_repo="index.html", repo_id="Prashant-77/thali", repo_type="space")
    print("landing page now points to", url)


if __name__ == "__main__":
    main()
