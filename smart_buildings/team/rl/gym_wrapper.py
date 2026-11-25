
from pathlib import Path

import gymnasium as gym
from gymnasium import spaces
import numpy as np

from smart_buildings.evaluate import (
    load_hybrid_environment,
    create_building_meta_data,
)

from smart_buildings.baselines.pid import PIDPolicy

from .observation_encoder import ObservationEncoder
from .action_adapter import ActionAdapter


class HVACEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(
        self,
        config_path,
        max_episode_steps=8640,
        reward_scale=1.0,
        *,
        occupancy_scale=1.0,
    ):
        super().__init__()

        self.config_path = Path(config_path)
        self.max_episode_steps = max_episode_steps
        self.reward_scale = reward_scale

        self.simulator = load_hybrid_environment(
            self.config_path
        )

        self.simulator._metrics_path = None
        self.simulator._occupancy_normalization_constant = 125.0

        self.building_metadata = create_building_meta_data(
            self.simulator,
            self.config_path.parent,
        )

        self.pid = PIDPolicy(
            self.building_metadata,
            self.simulator.action_spec(),
        )

        self.encoder = ObservationEncoder(
            self.building_metadata,
            self.pid,
            occupancy_scale=occupancy_scale,
        )

        self.adapter = ActionAdapter(
            self.building_metadata
        )

        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(ObservationEncoder.OBS_DIM,),
            dtype=np.float32,
        )

        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(ActionAdapter.ACTION_DIM,),
            dtype=np.float32,
        )

        self.raw_obs = None
        self.current_step = 0
        self.episode_reward = 0.0

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)

        self.pid.reset()

        self.current_step = 0
        self.episode_reward = 0.0

        self.raw_obs, info = self.simulator.reset()

        observation = self.encoder.encode(
            self.raw_obs
        )

        return observation, dict(info or {})

    def step(self, action):
        if self.raw_obs is None:
            raise RuntimeError(
                "Call reset() before step()"
            )

        # Generate the original PID action.
        base_action = self.pid.get_action(
            self.raw_obs
        )

        # Apply PPO corrections.
        final_action = self.adapter.combine(
            base_action,
            action,
        )

        (
            self.raw_obs,
            reward,
            terminated,
            truncated,
            info,
        ) = self.simulator.step(final_action)

        self.current_step += 1

        raw_reward = float(reward)

        self.episode_reward += raw_reward

        if self.current_step >= self.max_episode_steps:
            truncated = True

        observation = self.encoder.encode(
            self.raw_obs
        )

        info = dict(info or {})

        info["raw_reward"] = raw_reward
        info["episode_raw_reward"] = self.episode_reward

        return (
            observation,
            raw_reward * self.reward_scale,
            bool(terminated),
            bool(truncated),
            info,
        )

    def close(self):
        close_fn = getattr(
            self.simulator,
            "close",
            None,
        )

        if callable(close_fn):
            close_fn()
