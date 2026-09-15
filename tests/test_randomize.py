import numpy as np
import gymnasium as gym

import souschef_env  # noqa: F401  (registers the env)
from souschef_env import constants as C
from souschef_env.randomize import AXES, SHAPE_VARIANTS, test_ranges, train_ranges


def _env():
    return gym.make("souschef_env/Thali-v0", disable_env_checker=True, obs_type="state").unwrapped


def test_same_seed_is_byte_identical():
    env = _env()
    env.reset(seed=11)
    q1, s1 = env.data.qpos.copy(), env.last_sample.as_dict()
    env.reset(seed=12)
    env.reset(seed=11)
    assert np.array_equal(q1, env.data.qpos)
    assert s1 == env.last_sample.as_dict()


def test_different_seeds_differ():
    env = _env()
    env.reset(seed=1)
    a = env.last_sample.as_dict()
    env.reset(seed=2)
    assert a != env.last_sample.as_dict()


def test_axis_subset_only_touches_that_axis():
    env = _env()
    env.reset(seed=5, options={"axes": ("lighting",)})
    s = env.last_sample
    assert s.lighting and not s.placement and not s.mass and not s.shape and not s.background


def test_test_split_is_wider_and_has_unseen_material_and_shape():
    assert test_ranges.mass[1] > train_ranges.mass[1]
    assert test_ranges.n_table_materials == 10 and train_ranges.n_table_materials == 8
    assert len(SHAPE_VARIANTS["mug"]) == 4 and train_ranges.n_shape_variants == 3
    env = _env()
    mats_train = {env.reset(seed=s)[1]["sample"]["background"]["table_material"] for s in range(60)}
    assert max(mats_train) <= 7
    variants_train = {env.reset(seed=s)[1]["sample"]["shape"]["mug"]["variant"] for s in range(60)}
    assert 3 not in variants_train
    mats_test = {env.reset(seed=s, options={"split": "test"})[1]["sample"]["background"]["table_material"] for s in range(80)}
    assert mats_test & {8, 9}


def test_all_axes_listed():
    assert set(AXES) == {"placement", "mass", "friction", "shape", "lighting", "background"}
    env = _env()
    for s in range(3):
        env.reset(seed=s, options={"split": "test"})
        for o in C.OBJECTS:  # nothing fell off or exploded under randomisation
            assert -0.01 < env.data.xpos[env.model.body(o).id][2] < 0.05
