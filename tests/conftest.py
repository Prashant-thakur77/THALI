import os

os.environ.setdefault("MUJOCO_GL", "glfw")

import mujoco  # noqa: E402
import pytest  # noqa: E402

from souschef_env import constants as C  # noqa: E402


@pytest.fixture(scope="session")
def model() -> mujoco.MjModel:
    return mujoco.MjModel.from_xml_path(str(C.SCENE_XML))


@pytest.fixture
def data(model: mujoco.MjModel) -> mujoco.MjData:
    d = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, d, 0)
    mujoco.mj_forward(model, d)
    return d
