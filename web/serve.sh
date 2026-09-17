#!/usr/bin/env bash
# Run Thali Live on this machine and expose it through a Cloudflare quick tunnel (no account needed), then publish the URL
# on the static landing Space.  Re-run after a reboot.
cd "$(dirname "$0")/.."
export MUJOCO_GL=${MUJOCO_GL:-glfw}
pgrep -f "uvicorn web.serve[r]" >/dev/null || (nohup .venv/bin/uvicorn web.server:app --host 127.0.0.1 --port 7861 > /tmp/thali_web.log 2>&1 &)
pgrep -f "cloudflared tunnel --url http://127.0.0.1:786[1]" >/dev/null || (nohup "$HOME/bin/cloudflared" tunnel --url http://127.0.0.1:7861 --no-autoupdate > /tmp/thali_tunnel.log 2>&1 &)
for i in $(seq 1 30); do sleep 2; URL=$(grep -oE "https://[a-z0-9-]+\.trycloudflare\.com" /tmp/thali_tunnel.log | head -1); [ -n "$URL" ] && break; done
echo "live at $URL"
.venv/bin/python -m web.publish_url "$URL"
