"""Thali environment package: two SO-101 arms setting a dinner table in MuJoCo.

Importing the package registers the gym env ids and, when LeRobot is installed,
the ``souschef`` EnvConfig (see lerobot_plugin.py) so that
``lerobot-eval --env.type=souschef --env.discover_packages_path=souschef_env`` works.
"""

from __future__ import annotations

import os

# EGL is broken on this box (see docs/BLOCKERS.md); glfw works with the local display.
os.environ.setdefault("MUJOCO_GL", "glfw")

from gymnasium.envs.registration import register

register(
    id="souschef_env/Thali-v0",
    entry_point="souschef_env.env:ThaliEnv",
    max_episode_steps=600,
    nondeterministic=True,  # rendering differs slightly run to run; physics is seeded
    kwargs={"obs_type": "pixels_agent_pos"},
)

try:  # LeRobot is optional at import time (the env works without it)
    from souschef_env import lerobot_plugin as _lerobot_plugin  # noqa: F401
except ImportError:  # pragma: no cover
    pass
