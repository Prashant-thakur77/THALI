"""Seeded domain randomisation over six axes, with a held-out test split.

Axes (plan Phase 1.4): placement (xy + yaw), mass, friction, shape (variant +
scale), lighting, background.  ``train_ranges`` is what demos are recorded
under; ``test_ranges`` widens every continuous range by 1.5x about its centre
and adds two unseen table textures and one unseen mug shape.

Nothing here recompiles the model: sizes, masses, frictions, lights and
material ids are edited in place on the compiled MjModel and restored from a
snapshot before each new sample, so a given (seed, split, axes) always yields
byte-identical state (tests/test_randomize.py).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import mujoco
import numpy as np

from souschef_env import constants as C

AXES = ("placement", "mass", "friction", "shape", "lighting", "background")

# Per-object placement boxes (half-extents, m) about the built-in nominal position; cutlery jitters inside the drawer.
PLACEMENT_HALF = {"plate": (0.04, 0.03), "mug": (0.04, 0.04), "bottle": (0.03, 0.02),  # boxes sized so no start overlaps a target zone, the drawer path or the pour pose
                  "fork_1": (0.006, 0.01), "fork_2": (0.006, 0.01), "spoon_1": (0.006, 0.01), "spoon_2": (0.006, 0.01)}
YAW_HALF = {"plate": math.pi, "mug": math.pi, "bottle": math.pi,
            "fork_1": 0.17, "fork_2": 0.17, "spoon_1": 0.17, "spoon_2": 0.17}

# Shape variants: anisotropic (sx, sy, sz) multipliers on every geom of the body.  Index 3 for the mug is test-only.
SHAPE_VARIANTS = {
    "plate": [(1, 1, 1), (1, 1, 1.3), (1.08, 1.08, 0.8)],
    "mug": [(1, 1, 1), (1, 1, 1.2), (1, 1, 0.85), (1.1, 1.1, 1.35)],
    "bottle": [(1, 1, 1), (1, 1, 1.15), (1, 1, 0.9)],
    "fork_1": [(1, 1, 1), (1.1, 1, 1), (0.9, 1, 1)],
    "fork_2": [(1, 1, 1), (1.1, 1, 1), (0.9, 1, 1)],
    "spoon_1": [(1, 1, 1), (1.1, 1, 1), (0.9, 1, 1)],
    "spoon_2": [(1, 1, 1), (1.1, 1, 1), (0.9, 1, 1)],
}
N_TRAIN_TABLE_MATERIALS = 8
N_TABLE_MATERIALS = 10
N_BACKDROPS = 3


@dataclass(frozen=True)
class Ranges:
    placement_scale: float = 1.0       # multiplies PLACEMENT_HALF / YAW_HALF
    mass: tuple[float, float] = (0.7, 1.5)
    friction: tuple[float, float] = (0.6, 1.4)
    scale: tuple[float, float] = (0.85, 1.15)
    light_pos_jitter: float = 0.30
    light_diffuse: tuple[float, float] = (0.6, 1.2)
    n_table_materials: int = N_TRAIN_TABLE_MATERIALS
    n_shape_variants: int = 3          # mug variant 3 is held out


def _widen(lo: float, hi: float, k: float) -> tuple[float, float]:
    c, h = (lo + hi) / 2, (hi - lo) / 2 * k
    return (c - h, c + h)


train_ranges = Ranges()
test_ranges = Ranges(
    placement_scale=1.5,
    mass=_widen(*train_ranges.mass, 1.5),
    friction=_widen(*train_ranges.friction, 1.5),
    scale=_widen(*train_ranges.scale, 1.5),
    light_pos_jitter=train_ranges.light_pos_jitter * 1.5,
    light_diffuse=_widen(*train_ranges.light_diffuse, 1.5),
    n_table_materials=N_TABLE_MATERIALS,
    n_shape_variants=4,
)
SPLITS = {"train": train_ranges, "test": test_ranges}


@dataclass
class Sample:
    seed: int
    split: str
    axes: tuple[str, ...]
    placement: dict = field(default_factory=dict)
    mass: dict = field(default_factory=dict)
    friction: dict = field(default_factory=dict)
    shape: dict = field(default_factory=dict)
    lighting: dict = field(default_factory=dict)
    background: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {k: v for k, v in self.__dict__.items()}


class Randomizer:
    """Snapshot the compiled model once; every ``apply`` restores it and re-samples."""

    def __init__(self, model: mujoco.MjModel):
        self.model = model
        self._snap = {
            "geom_size": model.geom_size.copy(), "geom_pos": model.geom_pos.copy(),
            "geom_rbound": model.geom_rbound.copy(), "geom_aabb": model.geom_aabb.copy(),
            "geom_friction": model.geom_friction.copy(), "geom_matid": model.geom_matid.copy(),
            "site_size": model.site_size.copy(), "site_pos": model.site_pos.copy(),
            "body_mass": model.body_mass.copy(), "body_inertia": model.body_inertia.copy(),
            "light_pos": model.light_pos.copy(), "light_diffuse": model.light_diffuse.copy(),
        }
        self._home = np.array(model.key_qpos[0])
        self._obj_body = {o: model.body(o).id for o in C.OBJECTS}
        self._obj_geoms = {o: [g for g in range(model.ngeom) if model.geom_bodyid[g] == self._obj_body[o]] for o in C.OBJECTS}
        self._obj_qadr = {o: model.jnt_qposadr[model.joint(f"{o}_free").id] for o in C.OBJECTS}
        self._water_qadr = [model.jnt_qposadr[model.joint(f"water_{i}_free").id] for i in range(C.N_WATER)]
        self._table_geom = model.geom("table_top").id
        self._backdrop_geom = model.geom("backdrop").id
        self._table_mats = [model.material(f"mat_{n}").id for n in
                            ["wood_light", "wood_dark", "checker_grey", "checker_blue", "cloth_white", "cloth_red",
                             "marble", "slate", "test_green", "test_orange"]]
        self._backdrop_mats = [model.material(f"mat_{n}").id for n in ["room_grey", "room_warm", "room_dark"]]
        self._mug_site = model.site("mug_inside").id
        self._spout_site = model.site("bottle_spout").id

    def restore(self) -> None:
        m = self.model
        for k, v in self._snap.items():
            getattr(m, k)[:] = v

    def apply(self, data: mujoco.MjData, seed: int, split: str = "train", axes: tuple[str, ...] = AXES) -> Sample:
        """Restore nominal, reset to the home keyframe, then randomise the requested axes deterministically."""
        m = self.model
        r = SPLITS[split]
        rng = np.random.default_rng(seed)
        self.restore()
        mujoco.mj_resetDataKeyframe(m, data, 0)
        s = Sample(seed=seed, split=split, axes=tuple(axes))

        # draws happen for every axis so a seed's samples are stable no matter which axes are switched on
        draws = {o: {"dxy": rng.uniform(-1, 1, 2), "dyaw": rng.uniform(-1, 1), "mass": rng.uniform(*r.mass),
                     "fric": rng.uniform(*r.friction), "variant": int(rng.integers(0, len(SHAPE_VARIANTS[o]) if o == "mug" and r.n_shape_variants == 4 else 3)),
                     "scale": rng.uniform(*r.scale)} for o in C.OBJECTS}
        light = {"dpos": rng.uniform(-1, 1, (m.nlight, 3)) * r.light_pos_jitter, "diffuse": rng.uniform(*r.light_diffuse, m.nlight)}
        table_mat = int(rng.integers(0, r.n_table_materials))
        backdrop_mat = int(rng.integers(0, N_BACKDROPS))
        table_fric = rng.uniform(*r.friction)

        if "shape" in axes:
            for o in C.OBJECTS:
                sx, sy, sz = SHAPE_VARIANTS[o][draws[o]["variant"]]
                k = draws[o]["scale"]
                mult = np.array([sx, sy, sz]) * k
                for g in self._obj_geoms[o]:
                    m.geom_pos[g] *= mult
                    self._scale_geom_size(g, mult)
                    m.geom_rbound[g] *= mult.max()
                    m.geom_aabb[g] *= np.concatenate([mult, mult])
                if o == "mug":
                    m.site_pos[self._mug_site] *= mult
                    m.site_size[self._mug_site] *= np.array([mult[0], mult[2], 1.0])
                if o == "bottle":
                    m.site_pos[self._spout_site] *= mult
                s.shape[o] = {"variant": draws[o]["variant"], "scale": float(k)}
        if "mass" in axes:
            for o in C.OBJECTS:
                bid = self._obj_body[o]
                m.body_mass[bid] *= draws[o]["mass"]
                m.body_inertia[bid] *= draws[o]["mass"]
                s.mass[o] = float(draws[o]["mass"])
        if "friction" in axes:
            for o in C.OBJECTS:
                for g in self._obj_geoms[o]:
                    m.geom_friction[g, 0] *= draws[o]["fric"]
                s.friction[o] = float(draws[o]["fric"])
            m.geom_friction[self._table_geom, 0] *= table_fric
            s.friction["table"] = float(table_fric)
        if "placement" in axes:
            for o in C.OBJECTS:
                hx, hy = PLACEMENT_HALF[o]
                dx, dy = draws[o]["dxy"] * np.array([hx, hy]) * r.placement_scale
                dyaw = draws[o]["dyaw"] * YAW_HALF[o] * r.placement_scale
                q = self._obj_qadr[o]
                nominal = self._home[q : q + 7]
                yaw0 = 2 * math.atan2(nominal[6], nominal[3])
                yaw = yaw0 + dyaw
                data.qpos[q : q + 3] = nominal[:3] + np.array([dx, dy, 0.002])
                data.qpos[q + 3 : q + 7] = [math.cos(yaw / 2), 0, 0, math.sin(yaw / 2)]
                s.placement[o] = {"dx": float(dx), "dy": float(dy), "dyaw": float(dyaw)}
                if o == "bottle":  # water rides with the bottle
                    for qa in self._water_qadr:
                        data.qpos[qa : qa + 2] = self._home[qa : qa + 2] + np.array([dx, dy])
        if "lighting" in axes:
            m.light_pos[:] += light["dpos"]
            m.light_diffuse[:] *= light["diffuse"][:, None]
            s.lighting = {"dpos": light["dpos"].tolist(), "diffuse_scale": light["diffuse"].tolist()}
        if "background" in axes:
            m.geom_matid[self._table_geom] = self._table_mats[table_mat]
            m.geom_matid[self._backdrop_geom] = self._backdrop_mats[backdrop_mat]
            s.background = {"table_material": table_mat, "backdrop": backdrop_mat}
        mujoco.mj_forward(m, data)
        return s

    def _scale_geom_size(self, g: int, mult: np.ndarray) -> None:
        m = self.model
        t = m.geom_type[g]
        if t == mujoco.mjtGeom.mjGEOM_BOX or t == mujoco.mjtGeom.mjGEOM_ELLIPSOID:
            # box size is in the geom's own frame; for our rings the local x is radial, so use xy mean laterally
            R = np.zeros(9)
            mujoco.mju_quat2Mat(R, m.geom_quat[g])
            R = R.reshape(3, 3)
            local = np.abs(R.T @ mult)  # how the world multipliers project on local axes
            local = np.where(local < 1e-6, 1.0, local)
            m.geom_size[g] *= local
        elif t == mujoco.mjtGeom.mjGEOM_CYLINDER:
            m.geom_size[g, 0] *= (mult[0] + mult[1]) / 2
            m.geom_size[g, 1] *= mult[2]
        elif t == mujoco.mjtGeom.mjGEOM_CAPSULE:
            m.geom_size[g, 0] *= (mult[0] + mult[1]) / 2
            m.geom_size[g, 1] *= mult.max()
        elif t == mujoco.mjtGeom.mjGEOM_SPHERE:
            m.geom_size[g, 0] *= mult.max()
