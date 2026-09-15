"""Register the Thali env with LeRobot's env registry (plan Phase 1.7).

LeRobot discovers env configs through ``--env.discover_packages_path=souschef_env``
(lerobot.configs.parser.load_plugin), which imports this module; the
``register_subclass`` decorator then makes ``--env.type=souschef`` valid, and
``package_name`` tells the factory which package registers the gym id.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.envs.configs import EnvConfig
from lerobot.utils.constants import ACTION, OBS_IMAGES, OBS_STATE

from souschef_env import constants as C


@EnvConfig.register_subclass("souschef")
@dataclass
class SouschefEnv(EnvConfig):
    task: str | None = "Thali-v0"
    fps: int = C.FPS
    episode_length: int = 600
    obs_type: str = "pixels_agent_pos"
    observation_height: int = C.IMAGE_HEIGHT
    observation_width: int = C.IMAGE_WIDTH
    render_mode: str = "rgb_array"
    split: str = "test"
    features: dict[str, PolicyFeature] = field(
        default_factory=lambda: {ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(C.N_ACTIONS,))}
    )
    features_map: dict[str, str] = field(
        default_factory=lambda: {
            ACTION: ACTION,
            "agent_pos": OBS_STATE,
            **{f"pixels/{cam}": f"{OBS_IMAGES}.{cam}" for cam in C.CAMERAS},
        }
    )

    def __post_init__(self):
        self.features["agent_pos"] = PolicyFeature(type=FeatureType.STATE, shape=(C.N_ACTIONS,))
        for cam in C.CAMERAS:
            self.features[f"pixels/{cam}"] = PolicyFeature(
                type=FeatureType.VISUAL, shape=(self.observation_height, self.observation_width, 3)
            )

    @property
    def package_name(self) -> str:
        return "souschef_env"

    @property
    def gym_kwargs(self) -> dict:
        return {
            "obs_type": self.obs_type,
            "render_mode": self.render_mode,
            "observation_height": self.observation_height,
            "observation_width": self.observation_width,
            "split": self.split,
            "max_episode_steps": self.episode_length,
        }
