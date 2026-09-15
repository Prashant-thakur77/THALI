import gymnasium as gym
import mujoco
import numpy as np

import souschef_env  # noqa: F401
from souschef_env import constants as C
from souschef_env import oracles


def test_gym_make_pixels_agent_pos():
    env = gym.make("souschef_env/Thali-v0", disable_env_checker=True)
    obs, info = env.reset(seed=0)
    assert set(obs["pixels"]) == set(C.CAMERAS)
    for im in obs["pixels"].values():
        assert im.shape == (C.IMAGE_HEIGHT, C.IMAGE_WIDTH, 3) and im.dtype == np.uint8 and im.std() > 5
    assert obs["agent_pos"].shape == (C.N_ACTIONS,)
    a = env.unwrapped.agent_pos()
    obs, r, term, trunc, info = env.step(a)
    assert r == 0.0 and not term and "subgoals" in info
    assert env.render().shape == (480, 640, 3)
    env.close()


def test_jaw_closes_on_nothing_and_holds_the_mug():
    env = gym.make("souschef_env/Thali-v0", disable_env_checker=True, obs_type="state").unwrapped
    env.reset(seed=0)
    a = env.agent_pos().copy()
    a[5] = 0.0  # close arm A on empty air
    for _ in range(80):
        env.step(a)
    assert env.jaw_normalised("a") < 0.15
    assert env.jaw_state("a") == "closing"
    # now put the mug between arm B's jaws and close: both pads must touch it.
    # Mug axis along the site's approach axis (x), barrel centred on the site along x, offset 17.5 mm
    # along the opening axis (z) so the r=18 mm barrel presses 0.5 mm into the fixed pad at z=0.
    env.reset(seed=0)
    m, d = env.model, env.data
    site = m.site("arm_b_gripperframe").id
    pos, R = d.site_xpos[site].copy(), d.site_xmat[site].reshape(3, 3).copy()
    x_axis, z_axis = R[:, 0], R[:, 2]
    quat = np.zeros(4)
    mujoco.mju_quatZ2Vec(quat, x_axis)
    origin = pos + z_axis * 0.0175 - x_axis * 0.032
    q = m.jnt_qposadr[m.joint("mug_free").id]
    dof = m.jnt_dofadr[m.joint("mug_free").id]
    a = env.agent_pos().copy()
    a[11] = 0.0
    for _ in range(80):
        d.qpos[q : q + 3] = origin  # pin the mug while the jaw sweeps in
        d.qpos[q + 3 : q + 7] = quat
        d.qvel[dof : dof + 6] = 0
        env.step(a)
    assert env.jaw_state("b") == "holding"
    assert oracles.held_by(m, d, "mug") == "b"
