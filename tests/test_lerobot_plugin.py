"""The env is reachable from LeRobot the way lerobot-eval reaches it (plan 1.7)."""

import gymnasium as gym

from lerobot.configs.parser import load_plugin
from lerobot.envs.configs import EnvConfig


def test_plugin_registers_souschef_env_type():
    load_plugin("souschef_env")
    cfg = EnvConfig.get_choice_class("souschef")()
    assert cfg.type == "souschef"
    assert cfg.gym_id == "souschef_env/Thali-v0"
    assert cfg.features["action"].shape == (12,)
    assert "pixels/overhead" in cfg.features and "pixels/wrist_a" in cfg.features
    env = gym.make(cfg.gym_id, disable_env_checker=True, **cfg.gym_kwargs)
    obs, _ = env.reset(seed=0)
    assert obs["agent_pos"].shape == (12,)
    env.close()


def test_lerobot_make_env_builds_vector_env():
    from lerobot.envs.factory import make_env
    load_plugin("souschef_env")
    cfg = EnvConfig.get_choice_class("souschef")(episode_length=5)
    envs = make_env(cfg, n_envs=1)
    suite = envs["souschef"][0]
    obs, _ = suite.reset(seed=0)
    assert obs["agent_pos"].shape == (1, 12)
    suite.close()
