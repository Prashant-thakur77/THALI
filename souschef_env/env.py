"""Gymnasium env for the Thali dinner table (shape of gym-aloha's AlohaEnv, MuJoCo native).

Action (12,): per arm [5 absolute joint targets (rad), gripper command in 0..1] -- 1 = fully open, 0 = close
(torque sweep then hold), 0 < g < 1 = hold the jaw at angle g * 1.745 rad (P-control on the torque actuator), which
is how the expert opens just wide enough for an object so the moving jaw does not sweep into its neighbours.
Observation: ``pixels`` dict of the three cameras + ``agent_pos`` (12,) [joint angles, normalised jaw].

The jaw actuators are torque sources (gripper.py).  A close command runs
GRIP_TORQUE until the pads feel GRIP_TRIGGER_N of contact force, then eases to
HOLD_TORQUE so a light piece is neither dropped nor flicked out.
"""

from __future__ import annotations

from typing import Any

import gymnasium as gym
import mujoco
import numpy as np
from gymnasium import spaces

from souschef_env import constants as C
from souschef_env import gripper as G
from souschef_env import oracles
from souschef_env.randomize import AXES, Randomizer

_JAW_RANGE = (-0.17, 1.745)


class ThaliEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"], "render_fps": C.FPS}

    def __init__(
        self,
        obs_type: str = "pixels_agent_pos",
        render_mode: str = "rgb_array",
        observation_width: int = C.IMAGE_WIDTH,
        observation_height: int = C.IMAGE_HEIGHT,
        visualization_width: int = 640,
        visualization_height: int = 480,
        split: str = "train",
        axes: tuple[str, ...] = AXES,
        goals: tuple[str, ...] = oracles.FULL_TASK,
        task: str = "Thali-v0",
        grasp_assist: bool = True,
    ):
        super().__init__()
        self.obs_type = obs_type
        self.render_mode = render_mode
        self.observation_width, self.observation_height = observation_width, observation_height
        self.visualization_width, self.visualization_height = visualization_width, visualization_height
        self.split, self.axes, self.goals, self.task = split, tuple(axes), tuple(goals), task
        # Grasp assist: once a grasp is physically established (both pads touching, moving jaw loaded) a weld
        # pins the object to the gripper for the carry, and drops on open. It does not create grasps -- a miss
        # stays a miss -- it stops the 5-DoF wrist's orientation drift unloading a two-pad grasp mid-transit.
        self.grasp_assist = grasp_assist
        # demo recording turns this off while it runs unrecorded prerequisite skills (rendering is 3/4 of step time)
        self.render_enabled = True

        self.model = C.load_model(C.SCENE_XML)
        self.data = mujoco.MjData(self.model)
        self.randomizer = Randomizer(self.model)
        self.last_sample = None
        self._renderer: mujoco.Renderer | None = None
        self._vis_renderer: mujoco.Renderer | None = None

        m = self.model
        self._arm_qadr = {a: [m.jnt_qposadr[m.joint(C.ARM_PREFIX[a] + j).id] for j in C.ARM_JOINTS] for a in C.ARMS}
        self._arm_act = {a: [m.actuator(C.ARM_PREFIX[a] + j).id for j in C.ARM_JOINTS] for a in C.ARMS}
        self._jaw_qadr = {a: m.jnt_qposadr[m.joint(C.ARM_PREFIX[a] + "gripper").id] for a in C.ARMS}
        self._jaw_act = {a: m.actuator(C.ARM_PREFIX[a] + "gripper").id for a in C.ARMS}
        self._pads = {a: (m.geom(f"{C.ARM_PREFIX[a]}gripper_pad").id, m.geom(f"{C.ARM_PREFIX[a]}moving_jaw_so101_v1_pad").id) for a in C.ARMS}
        self._jaw_state = {a: "open" for a in C.ARMS}
        self._jaw_cmd = {a: 1.0 for a in C.ARMS}
        self._weld = {(a, o): m.equality(f"weld_{a}_{o}").id for a in C.ARMS for o in C.OBJECTS}
        self._welded: dict[str, str | None] = {a: None for a in C.ARMS}
        self._gripper_body = {a: m.body(f"{C.ARM_PREFIX[a]}gripper").id for a in C.ARMS}
        lo = np.array([m.jnt_range[m.joint(C.ARM_PREFIX[a] + j).id][0] for a in C.ARMS for j in (*C.ARM_JOINTS, "gripper")])
        hi = np.array([m.jnt_range[m.joint(C.ARM_PREFIX[a] + j).id][1] for a in C.ARMS for j in (*C.ARM_JOINTS, "gripper")])
        for a_idx in range(len(C.ARMS)):  # gripper entries are normalised 0..1
            lo[a_idx * 6 + 5], hi[a_idx * 6 + 5] = 0.0, 1.0
        self.action_space = spaces.Box(low=lo.astype(np.float32), high=hi.astype(np.float32), dtype=np.float32)

        img = spaces.Box(0, 255, (observation_height, observation_width, 3), np.uint8)
        pixels = spaces.Dict({cam: img for cam in C.CAMERAS})
        agent = spaces.Box(-np.inf, np.inf, (C.N_ACTIONS,), np.float64)
        if obs_type == "pixels":
            self.observation_space = pixels
        elif obs_type == "pixels_agent_pos":
            self.observation_space = spaces.Dict({"pixels": pixels, "agent_pos": agent})
        elif obs_type == "state":
            self.observation_space = spaces.Dict({"agent_pos": agent})
        else:
            raise ValueError(obs_type)

    # ------------------------------------------------------------------ state
    def agent_pos(self) -> np.ndarray:
        out = []
        for a in C.ARMS:
            out.extend(self.data.qpos[self._arm_qadr[a]])
            out.append(self.jaw_normalised(a))
        return np.asarray(out, dtype=np.float64)

    def jaw_normalised(self, arm: str) -> float:
        q = float(self.data.qpos[self._jaw_qadr[arm]])
        return float(np.clip((q - _JAW_RANGE[0]) / (_JAW_RANGE[1] - _JAW_RANGE[0]), 0, 1))

    def pad_forces(self, arm: str) -> tuple[float, float]:
        """Normal contact force on (fixed pad, moving pad) in N."""
        fixed, moving = self._pads[arm]
        out, f = {fixed: 0.0, moving: 0.0}, np.zeros(6)
        for i in range(self.data.ncon):
            c = self.data.contact[i]
            for pad in (fixed, moving):
                if c.geom1 == pad or c.geom2 == pad:
                    mujoco.mj_contactForce(self.model, self.data, i, f)
                    out[pad] += abs(f[0])
        return out[fixed], out[moving]

    def pad_force(self, arm: str) -> float:
        return sum(self.pad_forces(arm))

    # ------------------------------------------------------------------ gym api
    def reset(self, seed: int | None = None, options: dict[str, Any] | None = None):
        super().reset(seed=seed)
        options = options or {}
        seed = 0 if seed is None else int(seed)
        split = options.get("split", self.split)
        axes = tuple(options.get("axes", self.axes))
        self.last_sample = self.randomizer.apply(self.data, seed, split=split, axes=axes)
        self._jaw_state = {a: "open" for a in C.ARMS}
        self.data.eq_active[:] = 0
        self._welded = {a: None for a in C.ARMS}
        self.data.ctrl[:] = self.model.key_ctrl[0]
        for _ in range(25):  # let the props settle onto the table
            mujoco.mj_step(self.model, self.data)
        mujoco.mj_forward(self.model, self.data)
        return self._obs(), {"is_success": False, "sample": self.last_sample.as_dict(), "subgoals": oracles.subgoals(self.model, self.data)}

    def step(self, action: np.ndarray):
        action = np.asarray(action, dtype=np.float64).reshape(C.N_ACTIONS)
        for i, a in enumerate(C.ARMS):
            q = action[i * 6 : i * 6 + 5]
            self.data.ctrl[self._arm_act[a]] = np.clip(q, self.action_space.low[i * 6 : i * 6 + 5], self.action_space.high[i * 6 : i * 6 + 5])
            self._command_jaw(a, float(action[i * 6 + 5]))
        for _ in range(C.N_SUBSTEPS):
            mujoco.mj_step(self.model, self.data)
            for a in C.ARMS:
                self._update_jaw(a)
        sg = oracles.subgoals(self.model, self.data)
        reward = float(sum(sg[g] for g in self.goals))
        terminated = all(sg[g] for g in self.goals)
        info = {"is_success": terminated, "subgoals": sg, "task": self.task}
        return self._obs(), reward, terminated, False, info

    def _command_jaw(self, arm: str, g: float) -> None:
        self._jaw_cmd[arm] = g
        if g > 0.02:
            if self._jaw_state[arm] != "open":
                self._jaw_state[arm] = "open"
                self._release_weld(arm)
        elif self._jaw_state[arm] == "open":
            self._jaw_state[arm] = "closing"

    def welded(self, arm: str) -> str | None:
        return self._welded[arm]

    def _release_weld(self, arm: str) -> None:
        obj = self._welded[arm]
        if obj is not None:
            self.data.eq_active[self._weld[(arm, obj)]] = 0
            self._welded[arm] = None

    def _try_weld(self, arm: str) -> None:
        """Activate the weld for whatever both pads of ``arm`` are touching, at its current relative pose."""
        if not self.grasp_assist or self._welded[arm] is not None:
            return
        for obj in C.OBJECTS:
            if oracles.held_by(self.model, self.data, obj) == arm:
                m, d = self.model, self.data
                b1, b2 = self._gripper_body[arm], m.body(obj).id
                R1 = d.xmat[b1].reshape(3, 3)
                rel_pos = R1.T @ (d.xpos[b2] - d.xpos[b1])
                q1_inv, rel_quat = np.zeros(4), np.zeros(4)
                mujoco.mju_negQuat(q1_inv, d.xquat[b1])
                mujoco.mju_mulQuat(rel_quat, q1_inv, d.xquat[b2])
                eq = self._weld[(arm, obj)]
                m.eq_data[eq, 0:3] = 0.0
                m.eq_data[eq, 3:6] = rel_pos
                m.eq_data[eq, 6:10] = rel_quat
                m.eq_data[eq, 10] = 1.0
                d.eq_active[eq] = 1
                self._welded[arm] = obj
                return

    def _update_jaw(self, arm: str) -> None:
        st = self._jaw_state[arm]
        g = self._jaw_cmd[arm]
        if st == "open":
            if g >= 0.98:
                torque = G.OPEN_TORQUE
            else:  # partial open: P-control the jaw angle through the torque actuator
                qa = self._jaw_qadr[arm]
                q, qd = self.data.qpos[qa], self.data.qvel[self.model.jnt_dofadr[self.model.joint(C.ARM_PREFIX[arm] + "gripper").id]]
                torque = float(np.clip(6.0 * (g * _JAW_RANGE[1] - q) - 0.3 * qd, -G.OPEN_TORQUE, G.OPEN_TORQUE))
        elif st == "closing":
            torque = G.GRIP_TORQUE
            # the *moving* jaw arriving on the object is the grasp; the fixed pad can be touching long before
            if self.pad_forces(arm)[1] > G.GRIP_TRIGGER_N:
                self._jaw_state[arm] = "holding"
                self._try_weld(arm)
        else:  # holding
            torque = G.HOLD_TORQUE
            if self._welded[arm] is None:
                self._try_weld(arm)
            if self.pad_forces(arm)[1] < 0.05 and self.data.qpos[self._jaw_qadr[arm]] < 0.05:
                self._jaw_state[arm] = "closing"  # closed on nothing: keep sweeping
        self.data.ctrl[self._jaw_act[arm]] = torque

    def jaw_state(self, arm: str) -> str:
        return self._jaw_state[arm]

    # ------------------------------------------------------------------ obs / render
    def _renderer_for(self, w: int, h: int, vis: bool) -> mujoco.Renderer:
        attr = "_vis_renderer" if vis else "_renderer"
        r = getattr(self, attr)
        if r is None or r.width != w or r.height != h:
            r = mujoco.Renderer(self.model, height=h, width=w)
            # shadows + reflections cost 3x per frame (20 -> 6 ms at 240x320); policies do not need them
            r.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = 0
            r.scene.flags[mujoco.mjtRndFlag.mjRND_REFLECTION] = 0
            setattr(self, attr, r)
        return r

    def render_camera(self, camera: str, width: int | None = None, height: int | None = None) -> np.ndarray:
        w, h = width or self.observation_width, height or self.observation_height
        r = self._renderer_for(w, h, vis=(w != self.observation_width))
        r.update_scene(self.data, camera=camera)
        return r.render().copy()

    def _obs(self):
        if self.obs_type == "state":
            return {"agent_pos": self.agent_pos()}
        if not self.render_enabled:
            return {"pixels": None, "agent_pos": self.agent_pos()}
        pixels = {cam: self.render_camera(cam) for cam in C.CAMERAS}
        if self.obs_type == "pixels":
            return pixels
        return {"pixels": pixels, "agent_pos": self.agent_pos()}

    def render(self):
        return self.render_camera("overhead", self.visualization_width, self.visualization_height)

    def close(self):
        for attr in ("_renderer", "_vis_renderer"):
            r = getattr(self, attr)
            if r is not None:
                r.close()
                setattr(self, attr, None)
