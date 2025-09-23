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

    def __init__(
            self,
            size: int = 16,
            fixed_map: Optional[np.ndarray] = None,
            target_hp: int = 100,
            agent_hp: int = 100,
    ):
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
        self._can_attack_through = np.array([True, True, True, False, False, False], dtype=bool)

        # 위치들 초기값은 reset()에서 랜덤하게 생성
        # (-1, -1)을 초기화 되지 않은 상황으로 설정
        self._agent_location = np.array([-1, -1], dtype=np.int32)
        self._target_location = np.array([-1, -1], dtype=np.int32)

        self._max_target_hp = target_hp
        self._target_hp = target_hp
        self._max_agent_hp = agent_hp
        self._agent_hp = agent_hp
        self._yellow_cards = 0
        self._max_yellow_card = 3
        self._mega_bombs = 0

        # agent가 볼수 있는 상황 정의
        self.observation_space = gym.spaces.Dict(
            {
                "agent": gym.spaces.Box(0, size - 1, shape=(2,), dtype=np.int32), # [x, y]
                "target": gym.spaces.Box(0, size - 1, shape=(2,), dtype=np.int32), # [x, y]
                "map_onehot": gym.spaces.Box(0,1, shape=(self.num_tile_types,size,size), dtype=np.int8),
                "target_hp": gym.spaces.Box(0, target_hp, shape=(1,), dtype=np.int32),
                "agent_hp": gym.spaces.Box(0, agent_hp, shape=(1,), dtype=np.int32),
                "yellow_cards": gym.spaces.Box(0, self._max_yellow_card, shape=(1,), dtype=np.int32),
                "mega_bombs":   gym.spaces.Box(0, self._mega_bombs, shape=(1,), dtype=np.int32),

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

        self._attack_direction = {
            0: np.array([1, 0]),   # 우
            1: np.array([0, 1]),   # 상
            2: np.array([-1, 0]),  # 좌
            3: np.array([0, -1]),  # 하
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
            "target_hp": np.array(self._target_hp, dtype=np.int32),
            "agent_hp":   np.array(self._agent_hp, dtype=np.int32),
            "yellow_cards": np.array(self._yellow_cards, dtype=np.int32),
            "mega_bombs":   np.array(self._mega_bombs, dtype=np.int32),
        }

    # 디버깅 용, 학습 알고리즘에서 사용하면 안됨
    def _get_info(self):
        return {
            "distance": np.linalg.norm(
                self._agent_location - self._target_location, ord=1
            ),
            "target_hp": self._target_hp,
            "agent_hp":  self._agent_hp,
            "yellow_cards": self._yellow_cards,
            "mega_bombs":   self._mega_bombs,
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

        self._agent_hp = self._max_agent_hp
        self._target_hp = self._max_target_hp
        self._yellow_cards = 0

        observation = self._get_obs()
        info = self._get_info()

        return observation, info

    def step(self, action):
        reward = 0
        terminated = False

        if action in range(4):
            direction = self._action_to_direction[action]
            next_location = np.clip(self._agent_location + direction, 0, self.size - 1)
            tile = self._fixed_map[next_location[1], next_location[0]]

            if self._can_move[tile]:
                self._agent_location = next_location
                reward -= 0.1
            else:
                reward -= 3
                self._yellow_cards += 1
                if self._yellow_cards >= self._max_yellow_card:
                    terminated = True
                    reward -= 20

        elif action in range(4, 8):
            attack_direction = self._attack_direction[action - 4]
            hit = False
            valid = False
            for dist in range(1, 4):
                pos = self._agent_location + attack_direction * dist
                if np.any(pos < 0) or np.any(pos >= self.size):
                    break
                tile = self._fixed_map[pos[1], pos[0]]
                if tile == 4: # 나무
                    self._fixed_map[pos[1], pos[0]] = 0
                    valid = True
                    break

                if not self._can_attack_through[tile]:
                    break
                if np.array_equal(pos, self._target_location):
                    valid = True
                    hit = True
                    break

            if valid:
                if hit:
                    self._target_hp -= 30
                    reward += 5
                    if self._target_hp <= 0:
                        reward += 10
                        terminated = True
            else:
                reward -= 3
                self._yellow_cards += 1
                if self._yellow_cards >= self._max_yellow_card:
                    terminated = True
                    reward -= 20

        elif action in range(8, 12):
            attack_direction = self._attack_direction[action - 8]
            hit = False
            valid = False

            if self._mega_bombs > 0:
                for dist in range(1, 4):
                    pos= self._agent_location + attack_direction * dist
                    if np.any(pos < 0) or np.any(pos >= self.size):
                        break
                    tile = self._fixed_map[pos[1], pos[0]]
                    if tile == 4:  # 나무
                        self._fixed_map[pos[1], pos[0]] = 0
                        valid = True
                        break
                    if not self._can_attack_through[tile]:
                        break
                    if np.array_equal(pos, self._target_location):
                        valid = True
                        hit = True
                        break
            if valid:
                if hit:
                    self._target_hp -= 70
                    reward += 10
                    if self._target_hp <= 0:
                        reward += 10
                        terminated = True
            else:
                reward -= 5
                self._yellow_cards += 1
                if self._yellow_cards >= self._max_yellow_card:
                    terminated = True
                    reward -= 20

        # 대기
        else:
            reward -= 3


        # 시간 제한 없음(추후 변경 예정)
        truncated = False
        observation = self._get_obs()
        info = self._get_info()
        return observation, reward, terminated, truncated, info

    def render(self, mode="human"):
        # human 모드: 컬러 ASCII + 상태 정보
        if mode == "human":
            status = (
                f"Agent HP: {self._agent_hp}/{self._max_agent_hp} | "
                f"Target HP: {self._target_hp}/{self._max_target_hp} | "
                f"Yellow Cards: {self._yellow_cards}/{self._max_yellow_card} | "
                f"Mega Bombs: {self._mega_bombs}"
            )
            print(status)
            # 타일별 심볼 정의
            symbols = {
                0: "G ",  # 풀
                1: "S ",  # 모래
                2: "W ",  # 물
                3: "R ",  # 바위
                4: "T ",  # 나무
                5: "+ "   # 보급
            }
            for y in range(self.size - 1, -1, -1):
                row = ""
                for x in range(self.size):
                    if np.array_equal([x, y], self._agent_location):
                        row += "A "
                    elif np.array_equal([x, y], self._target_location):
                        row += "X "
                    else:
                        tile = self._fixed_map[y, x]
                        row += symbols.get(tile, "? ")
                print(row)
            print()

        # rgb_array 모드: NumPy 배열 (H,W,3) 반환
        elif mode == "rgb_array":
            # 각 타일에 RGB 색 할당
            color_map = {
                0: (34, 139,  34),  # 풀 (ForestGreen)
                1: (238, 214, 175), # 모래 (SandyBrown)
                2: ( 65, 105, 225), # 물 (RoyalBlue)
                3: (128, 128, 128), # 바위 (Gray)
                4: ( 34, 139,  34), # 나무 (ForestGreen) – 같은 색으로 처리
                5: (255, 215,   0), # 보급 (Gold)
            }
            cell_size = 20
            height = self.size * cell_size
            width  = self.size * cell_size
            canvas = np.zeros((height, width, 3), dtype=np.uint8)

            # 배경 타일 그리기
            for y in range(self.size):
                for x in range(self.size):
                    color = color_map[self._fixed_map[y, x]]
                    y0, y1 = (self.size - 1 - y) * cell_size, (self.size - y) * cell_size
                    x0, x1 = x * cell_size, (x + 1) * cell_size
                    canvas[y0:y1, x0:x1] = color

            # 에이전트와 타겟 표시
            ax0, ay0 = self._agent_location * cell_size
            canvas[height - ay0 - cell_size:height - ay0, ax0:ax0 + cell_size] = (255, 0, 0)  # 에이전트(빨강)
            tx0, ty0 = self._target_location * cell_size
            canvas[height - ty0 - cell_size:height - ty0, tx0:tx0 + cell_size] = (0, 0, 0)  # 타겟(검정)

            return canvas

        else:
            raise ValueError(f"Unknown render mode: {mode}")

