from typing import Optional

import gymnasium as gym
import numpy as np


ENV_ID = "BattleSsafyEnv-v0"


def register_env() -> None:
    gym.register(
        id=ENV_ID,
        entry_point="battle_ssafy_env:BattleSsafyEnv",
        max_episode_steps=100,
    )


class BattleSsafyEnv(gym.Env):
    metadata = {
        "render_modes": ["human"],
        "render_fps": 30,
    }

    def __init__(self, size: int = 16, render_mode: Optional[str] = None):
        self.size = size
        self.render_mode = render_mode

        # 위치 초기값 (-1, -1): reset 전 사용 방지 가드
        self._agent_location = np.array([-1, -1], dtype=np.int32)
        self._target_location = np.array([-1, -1], dtype=np.int32)

        self.observation_space = gym.spaces.Dict(
            {
                "agent": gym.spaces.Box(0, size - 1, shape=(2,), dtype=np.int32),
                "target": gym.spaces.Box(0, size - 1, shape=(2,), dtype=np.int32),
            }
        )

        self.action_space = gym.spaces.Discrete(4)

        self._action_to_direction = {
            0: np.array([1, 0]),
            1: np.array([0, 1]),
            2: np.array([-1, 0]),
            3: np.array([0, -1]),
        }

    def _get_obs(self) -> dict:
        return {"agent": self._agent_location, "target": self._target_location}

    def _get_info(self) -> dict:
        return {
            "distance": np.linalg.norm(
                self._agent_location - self._target_location, ord=1
            )
        }

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ) -> tuple[dict, dict]:
        super().reset(seed=seed)

        self._agent_location = self.np_random.integers(0, self.size, size=2, dtype=int)

        self._target_location = self._agent_location
        while np.array_equal(self._target_location, self._agent_location):
            self._target_location = self.np_random.integers(
                0, self.size, size=2, dtype=int
            )

        return self._get_obs(), self._get_info()

    def step(self, action: int) -> tuple[dict, float, bool, bool, dict]:
        direction = self._action_to_direction[action]

        self._agent_location = np.clip(
            self._agent_location + direction, 0, self.size - 1
        )

        terminated = bool(np.array_equal(self._agent_location, self._target_location))
        truncated = False
        reward = 10.0 if terminated else -0.005

        return self._get_obs(), reward, terminated, truncated, self._get_info()

    def render(self) -> None:
        if self.render_mode != "human":
            return
        for y in range(self.size - 1, -1, -1):
            row = ""
            for x in range(self.size):
                if np.array_equal([x, y], self._agent_location):
                    row += "A "
                elif np.array_equal([x, y], self._target_location):
                    row += "T "
                else:
                    row += ". "
            print(row)
        print()
