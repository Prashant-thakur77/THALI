"""Pick a MuJoCo rendering backend that works on this host without taking the server down with it.

A bad GL backend can segfault the whole process (which is what a hosting platform then reports as a dead app), so each
candidate is tried in a throw-away subprocess first.  Returns the backend name, or None when no headless renderer works
(the runtime then keeps the cameras off and the page shows the scene state only).

    python -m web.glprobe            # prints the chosen backend
"""

from __future__ import annotations

import os
import subprocess
import sys

PROBE = """
import mujoco, numpy as np
m = mujoco.MjModel.from_xml_string('<mujoco><worldbody><light pos="0 0 1"/><geom type="box" size=".1 .1 .1"/><camera name="c" pos="0 -1 1" xyaxes="1 0 0 0 1 1"/></worldbody></mujoco>')
d = mujoco.MjData(m); mujoco.mj_forward(m, d)
r = mujoco.Renderer(m, 64, 64); r.update_scene(d, camera="c"); img = r.render(); r.close()
assert img.shape == (64, 64, 3) and img.max() > 0
print("ok")
"""


def pick(candidates=("egl", "osmesa", "glfw")) -> str | None:
    preferred = os.environ.get("MUJOCO_GL")
    order = ([preferred] if preferred else []) + [c for c in candidates if c != preferred]
    for backend in order:
        env = dict(os.environ, MUJOCO_GL=backend, PYOPENGL_PLATFORM=backend)
        try:
            out = subprocess.run([sys.executable, "-c", PROBE], env=env, capture_output=True, text=True, timeout=90)
            if out.returncode == 0 and "ok" in out.stdout:
                return backend
        except Exception:
            pass
    return None


if __name__ == "__main__":
    print(pick())
