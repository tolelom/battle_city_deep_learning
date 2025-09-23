from typing import Optional
import numpy as np
import gymnasium as gym

gym.register(
    id='BattleSsafyEnv-v0',
    entry_point="battle_ssafy_env:BattleSsafyEnv",
    max_episode_steps=100,
)

class BattleSsafyEnv(gym.Env):
    metadata = {
        "render_modes": ["human", "rgb_array"],
        "render_fps": 30,
    }

    def __init__(self, size: int = 16, fixed_map: Optional[np.ndarray] = None):
        # 맵의 크기
        self.size = size
        self.num_tile_types = 6

        if fixed_map is not None:
            assert fixed_map.shape == (size, size), "fixed_map 크기가 size×size가 아닙니다"
            self._fixed_map = fixed_map.astype(np.int32)
        else:
            self._fixed_map = np.zeros((size, size), dtype=np.int32)

        # 순서대로 풀, 모래, 물, 바위, 나무, 보급
        self._can_move = np.array([True, True, False, False, False, False,], dtype=bool)

        # 위치들 초기값은 reset()에서 랜덤하게 생성
        # (-1, -1)을 초기화 되지 않은 상황으로 설정
        self._agent_location = np.array([-1, -1], dtype=np.int32)
        self._target_location = np.array([-1, -1], dtype=np.int32)

        # agent가 볼수 있는 상황 정의
        self.observation_space = gym.spaces.Dict(
            {
                "agent": gym.spaces.Box(0, size - 1, shape=(2,), dtype=np.int32), # [x, y]
                "target": gym.spaces.Box(0, size - 1, shape=(2,), dtype=np.int32), # [x, y]
                "map_onehot": gym.spaces.Box(0,1, shape=(self.num_tile_types,size,size), dtype=np.int8),
            }
        )

        # 할 수 있는 액션 정의
        self.action_space = gym.spaces.Discrete(4 + 4 + 4 + 1)

        self._action_to_direction = {
            0: np.array([1, 0]),
            1: np.array([0, 1]),
            2: np.array([-1, 0]),
            3: np.array([0, -1]),
        }

        self.seed()

    def seed(self, seed: Optional[int] = None):
        self.np_random, seed = gym.utils.seeding.np_random(seed)
        return [seed]
    
    def _encode_onehot(self):
        onehot = np.eye(self.num_tile_types, dtype=np.int8)[self._fixed_map]
        # 결과 shape: (size, size, num_tile_types) → transpose to (num_tile_types,size,size)
        return onehot.transpose(2,0,1)

    # 핼퍼 메서드
    def _get_obs(self):
        return {
            "agent": self._agent_location.copy(),
            "target": self._target_location.copy(),
            "map_onehot": self._encode_onehot(),
        }

    # 디버깅 용, 학습 알고리즘에서 사용하면 안됨
    def _get_info(self):
        return {
            "distance": np.linalg.norm(
                self._agent_location - self._target_location, ord=1
            )
        }

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)

        # agent 위치 랜덤 생성
        while True:
            pos = self.np_random.integers(0, self.size, size=2)
            if self._can_move[self._fixed_map[pos[1], pos[0]]]:
                self._agent_location = pos
                break

        # target 위치 랜덤 생성(agent와 다른 위치 보장)
        while True:
            pos = self.np_random.integers(0, self.size, size=2)
            if (self._can_move[self._fixed_map[pos[1], pos[0]]] and
                not np.array_equal(pos, self._agent_location)):
                self._target_location = pos
                break

        observation = self._get_obs()
        info = self._get_info()

        return observation, info

    def step(self, action):
        direction = self._action_to_direction[action]
        next_location = np.clip(self._agent_location + direction, 0, self.size - 1)
        tile = self._fixed_map[next_location[1], next_location[0]]

        if self._can_move[tile]:
            self._agent_location = next_location

        terminated = np.array_equal(self._agent_location, self._target_location)

        # 시간 제한 없음(추후 변경 예정)
        truncated = False

        # 보상 세팅
        reward = 10 if terminated else -0.005
        # 보상 세팅 끝

        observation = self._get_obs()
        info = self._get_info()

        return observation, reward, terminated, truncated, info

    def render(self):
        """Render the environment for human viewing."""
        if self.render_mode == "human":
            # Print a simple ASCII representation
            for y in range(self.size - 1, -1, -1):  # Top to bottom
                row = ""
                for x in range(self.size):
                    if np.array_equal([x, y], self._agent_location):
                        row += "A "  # Agent
                    elif np.array_equal([x, y], self._target_location):
                        row += "T "  # Target
                    else:
                        row += ". "  # Empty
                print(row)
            print()
