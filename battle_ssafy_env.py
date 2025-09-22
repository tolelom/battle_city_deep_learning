from typing import Optional, Tuple, Dict, List
import numpy as np
import gymnasium as gym
from gymnasium import spaces

# 타일 인코딩
G, R, W, F, X, E = 0, 1, 2, 3, 4, 5

class BattleSsafyEnv(gym.Env):
    metadata = {"render_modes": ["human", "ansi"], "render_fps": 30}

    def __init__(
            self,
            size: int = 16,
            custom_map: Optional[np.ndarray] = None,
            render_mode: Optional[str] = None,
            n_enemies: int = 3):
        assert size == 16, "16x16 맵"
        self.H = self.W = size
        self.render_mode = render_mode
        self.n_enemies = n_enemies

        if custom_map is not None:
            assert custom_map.shape == (self.H, self.W)
            assert custom_map.dtype == np.int8

        # === Observation space (요청 반영) ===
        self.observation_space = spaces.Dict(
            {
                "map": spaces.Box(low=0, high=5, shape=(self.H, self.W), dtype=np.int8),
                # "agent": spaces.Dict(
                #     {
                #         # 사용자가 제시한 agent 관측 구조
                #         "pos": spaces.Box(0, self.W - 1, shape=(2,), dtype=np.int32),
                #         "hp": spaces.Box(0, 100, shape=(), dtype=np.int32),
                #         "bomb": spaces.Box(0, 99, shape=(), dtype=np.int32),
                #         "mega_bomb": spaces.Box(0, 10, shape=(), dtype=np.int32),
                #     }
                # ),
                # # 적 정보는 "체력만" 노출
                # "enemies": spaces.Dict(
                #     {
                #         # 길이가 n_enemies인 HP 벡터 (없는 적은 0으로 표시)
                #         "pos": spaces.Box(-1, self.W - 1, shape=(self.n_enemies, 2), dtype=np.int32),
                #         "hp":  spaces.Box(0, 100, shape=(self.n_enemies,), dtype=np.int32),
                #     }
                # ),

                # agent (flattened)
                "agent_pos":       spaces.Box(0, self.W - 1, shape=(2,),   dtype=np.int32),
                "agent_hp":        spaces.Box(0, 100,        shape=(1,),   dtype=np.int32),  # ← (1,)
                "agent_bomb":      spaces.Box(0, 99,         shape=(1,),   dtype=np.int32),  # ← (1,)
                "agent_mega_bomb": spaces.Box(0, 10,         shape=(1,),   dtype=np.int32),  # ← (1,)


                # enemies (flattened)
                # - positions: (-1,-1) means empty slot
                "enemies_pos": spaces.Box(-1, self.W - 1, shape=(self.n_enemies, 2), dtype=np.int32),
                "enemies_hp":  spaces.Box(0, 100,         shape=(self.n_enemies,),   dtype=np.int32),

            }
        )

        # 행동: 0→ 1↑ 2← 3↓ 4:폭탄 5:메가폭탄 6:암호풀기
        self.action_space = spaces.Discrete(13)

        self._dir = {
            0: np.array([1, 0], dtype=np.int32),
            1: np.array([0, 1], dtype=np.int32),
            2: np.array([-1, 0], dtype=np.int32),
            3: np.array([0, -1], dtype=np.int32),
        }

        # 내부 상태(포지션/HP/탄수 등)
        self._base_map = np.zeros((self.H, self.W), dtype=np.int8)  # 지형만
        self._turrets: List[Tuple[int, int]] = []  # X 좌표 리스트
        self._agent_pos = np.array([0, 0], dtype=np.int32)
        self._agent_hp = 100
        self._agent_bomb = 3
        self._agent_mega_bomb = 1

        # 적: 포지션 + HP
        self._enemies_pos: List[Optional[np.ndarray]] = [None] * self.n_enemies
        self._enemies_hp = np.zeros(self.n_enemies, dtype=np.int32)

        self._rng = None

    # ============ Gym API ============
    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        self._rng = self.np_random

        # 1) 지형 초기화
        self._base_map[:] = G
        self._random_scatter(R, count=20)  # 벽
        self._random_scatter(W, count=15)  # 물
        self._random_scatter(F, count=4)   # 보급소
        self._turrets = self._random_scatter(X, count=3, as_list=True)  # 포탑

        # 2) 에이전트 초기화
        self._agent_pos = self._sample_passable()
        self._agent_hp = 100
        self._agent_bomb = 3
        self._agent_mega_bomb = 1

        # 3) 적 초기화
        for i in range(self.n_enemies):
            self._enemies_pos[i] = self._sample_passable()
            self._enemies_hp[i] = 100

        # 시작 시 보급 체크(보급소 인접 시 메가폭탄 +1)
        self._try_refill_mega()

        return self._get_obs(), self._get_info()

    def step(self, action: int):
        reward = 0.0
        terminated = False
        truncated = False

        # 1) 이동
        if 0 <= action <= 3:
            nxt = self._agent_pos + self._dir[action]
            nxt = np.clip(nxt, [0, 0], [self.W - 1, self.H - 1])
            if self._is_passable(tuple(nxt)):
                self._agent_pos = nxt
            # 이동 페널티 완화: -0.005 → -0.002
            reward += -0.002

        # 2) 일반 폭탄
        elif 4 <= action <= 7:
            if self._agent_bomb > 0:
                self._agent_bomb -= 1
                dir_idx = action - 4
                hit_t, hit_e = self._fire_bomb(mega=False, splash=False, direction=dir_idx)
                # 포탑 적중 보상 증가: 15.0
                reward += 15.0 * hit_t + 7.0 * hit_e or -0.005
            else:
                reward += -0.005

        # 3) 메가폭탄
        elif action == 5:
            if self._agent_mega_bomb > 0:
                self._agent_mega_bomb -= 1
                hit_t, hit_e = self._fire_bomb(mega=True, splash=True)
                # 메가폭탄 보상 증가: 20.0
                reward += 20.0 * hit_t + 10.0 * hit_e
                if hit_t == 0 and hit_e == 0:
                    reward += -0.005
            else:
                reward += -0.005

        # 4) 암호풀기: 소정의 보상 추가
        elif action == 6:
            # 기존 패널티 대신 소량 보상(+0.1)으로 유도
            reward += 0.1

        # 보급 체크
        self._try_refill_mega()

        # 포탑 시야에 노출되면 사망
        if self._in_los_of_any_turret():
            self._agent_hp = 0
            # 사망 페널티 완화: -5.0 → -3.0
            reward += -3.0
            terminated = True

        # 모든 포탑 제거 시 종료
        if len(self._turrets) == 0 and not terminated:
            # 게임 클리어 보상 추가
            reward += 5.0
            terminated = True

        return self._get_obs(), float(reward), terminated, truncated, self._get_info()

    def render(self):
        grid = self._compose_visible_map()
        lines = []
        for y in range(self.H - 1, -1, -1):
            row = []
            for x in range(self.W):
                ch = grid[y, x]
                row.append(self._tile_char(ch, (x, y)))
            lines.append(" ".join(row))
        out = "\n".join(lines) + "\n"

        if self.render_mode == "ansi":
            return out
        if self.render_mode == "human":
            print(out)

    def close(self):
        pass

    # ============ Helpers ============
    def _get_obs(self):
        vmap = self._compose_visible_map()

        enemies_pos = np.full((self.n_enemies, 2), -1, dtype=np.int32)
        enemies_hp  = np.zeros(self.n_enemies, dtype=np.int32)
        for i, pos in enumerate(self._enemies_pos):
            if pos is not None and self._enemies_hp[i] > 0:
                enemies_pos[i] = pos
                enemies_hp[i]  = self._enemies_hp[i]

        return {
            "map": vmap.astype(np.int8, copy=True),

            "agent_pos":       self._agent_pos.astype(np.int32, copy=True),
            "agent_hp":        np.array([self._agent_hp],        dtype=np.int32),  # ← (1,)
            "agent_bomb":      np.array([self._agent_bomb],      dtype=np.int32),  # ← (1,)
            "agent_mega_bomb": np.array([self._agent_mega_bomb], dtype=np.int32),  # ← (1,)

            "enemies_pos": enemies_pos,
            "enemies_hp":  enemies_hp,
        }
        # vmap = self._compose_visible_map()

        # enemies_pos = np.full((self.n_enemies, 2), -1, dtype=np.int32)
        # enemies_hp = np.zeros(self.n_enemies, dtype=np.int32)

        # for i, pos in enumerate(self._enemies_pos):
        #     if pos is not None and self._enemies_hp[i] > 0:
        #         enemies_pos[i] = pos
        #         enemies_hp[i] = self._enemies_hp[i]

        # return {
        #     "map": vmap.astype(np.int8, copy=True),
        #     "agent": {
        #         "pos": self._agent_pos.astype(np.int32, copy=True),
        #         "hp": np.int32(self._agent_hp),
        #         "bomb": np.int32(self._agent_bomb),
        #         "mega_bomb": np.int32(self._agent_mega_bomb),
        #     },
        #     "enemies": {
        #         "pos": enemies_pos,
        #         "hp": enemies_hp,
        #     },
        # }

    def _get_info(self) -> Dict:
        # 에이전트와 가장 가까운 포탑까지의 맨해튼 거리(없으면 -1)
        td = (
            min(abs(self._agent_pos[0] - x) + abs(self._agent_pos[1] - y) for x, y in self._turrets)
            if self._turrets
            else -1
        )
        return {"nearest_turret_L1": td}

    def _compose_visible_map(self) -> np.ndarray:
        """지형 + 포탑 + 적을 합친 관측용 맵."""
        v = self._base_map.copy()
        for (tx, ty) in self._turrets:
            v[ty, tx] = X
        for i, pos in enumerate(self._enemies_pos):
            if pos is not None and self._enemies_hp[i] > 0:
                v[pos[1], pos[0]] = E
        return v

    def _tile_char(self, val: int, xy: Tuple[int, int]) -> str:
        if (xy[0] == self._agent_pos[0]) and (xy[1] == self._agent_pos[1]):
            return "A"
        return {G: ".", R: "R", W: "W", F: "F", X: "X", E: "E"}.get(int(val), "?")

    def _random_scatter(self, kind: int, count: int, as_list: bool = False):
        coords = []
        for _ in range(count):
            for _t in range(50):  # 충돌 시 재시도
                x = int(self._rng.integers(0, self.W))
                y = int(self._rng.integers(0, self.H))
                if self._base_map[y, x] == G:
                    self._base_map[y, x] = kind
                    coords.append((x, y))
                    break
        return coords if as_list else None

    def _sample_passable(self) -> np.ndarray:
        # 통행 가능(G) 타일 중 하나 샘플
        while True:
            x = int(self._rng.integers(0, self.W))
            y = int(self._rng.integers(0, self.H))
            if self._is_passable((x, y)):
                return np.array([x, y], dtype=np.int32)

    def _is_passable(self, xy: Tuple[int, int]) -> bool:
        t = self._base_map[xy[1], xy[0]]
        if t in (R, W, F):
            return False
        # 적/포탑 위치는 통과 불가
        if (xy in self._turrets):
            return False
        for i, p in enumerate(self._enemies_pos):
            if p is not None and self._enemies_hp[i] > 0 and p[0] == xy[0] and p[1] == xy[1]:
                return False
        return True

    def _try_refill_mega(self):
        # F(보급소)에 인접(맨해튼 1)하면 메가폭탄 +1
        x, y = self._agent_pos
        for dx, dy in ((1,0),(-1,0),(0,1),(0,-1)):
            nx, ny = x + dx, y + dy
            if 0 <= nx < self.W and 0 <= ny < self.H and self._base_map[ny, nx] == F:
                self._agent_mega_bomb += 1
                return

    # 시야(같은 행/열, W는 통과, R/F는 차단)
    def _in_los(self, src: Tuple[int, int], dst: Tuple[int, int]) -> bool:
        sx, sy = src
        dx, dy = dst
        if sx != dx and sy != dy:
            return False
        if sx == dx:
            y_step = 1 if dy > sy else -1
            for y in range(sy + y_step, dy, y_step):
                t = self._base_map[y, sx]
                if t in (R, F):  # 차단
                    return False
            return True
        else:
            x_step = 1 if dx > sx else -1
            for x in range(sx + x_step, dx, x_step):
                t = self._base_map[sy, x]
                if t in (R, F):
                    return False
            return True

    def _first_target_in_line(self) -> Tuple[int, int, Optional[int], Optional[Tuple[int,int]]]:
        """에이전트 기준 같은 행/열에서 가장 가까운 타겟(포탑/적) 반환.
        return: (turret_hits(0/1), enemy_hits(0/1), enemy_index or None, pos or None)
        """
        ax, ay = int(self._agent_pos[0]), int(self._agent_pos[1])

        candidates: List[Tuple[int, str, Optional[int], Tuple[int,int]]] = []  # (dist, kind, e_idx, pos)

        # 포탑 후보
        for (tx, ty) in self._turrets:
            if self._in_los((ax, ay), (tx, ty)):
                dist = abs(tx - ax) + abs(ty - ay)
                candidates.append((dist, "turret", None, (tx, ty)))

        # 적 후보
        for i, pos in enumerate(self._enemies_pos):
            if pos is None or self._enemies_hp[i] <= 0:
                continue
            ex, ey = int(pos[0]), int(pos[1])
            if self._in_los((ax, ay), (ex, ey)):
                dist = abs(ex - ax) + abs(ey - ay)
                candidates.append((dist, "enemy", i, (ex, ey)))

        if not candidates:
            return 0, 0, None, None

        candidates.sort(key=lambda t: t[0])
        _, kind, e_idx, pos = candidates[0]

        if kind == "turret":
            return 1, 0, None, pos
        else:
            return 0, 1, e_idx, pos

    def _fire_bomb(self, mega: bool, splash: bool, direction: int) -> Tuple[int, int]:
        """폭탄/메가폭탄 발사.
        return: (turret_kills, enemy_kills)
        """
        t_hit, e_hit, e_idx, pos = self._first_target_in_line()
        turret_kills = 0
        enemy_kills = 0

        # 메인 타겟 처리(HP를 0으로)
        if t_hit and pos is not None:
            # 포탑 제거
            if pos in self._turrets:
                self._turrets.remove(pos)
                turret_kills += 1
        elif e_hit and e_idx is not None:
            # 적 제거
            self._enemies_hp[e_idx] = 0
            self._enemies_pos[e_idx] = None
            enemy_kills += 1

        # 스플래시(3x3)
        if mega and splash and pos is not None:
            cx, cy = pos
            for y in range(max(0, cy - 1), min(self.H, cy + 2)):
                for x in range(max(0, cx - 1), min(self.W, cx + 2)):
                    # 주변 포탑
                    if (x, y) in self._turrets:
                        self._turrets.remove((x, y))
                        turret_kills += 1
                    # 주변 적
                    for i, p in enumerate(self._enemies_pos):
                        if p is not None and self._enemies_hp[i] > 0 and p[0] == x and p[1] == y:
                            self._enemies_hp[i] = 0
                            self._enemies_pos[i] = None
                            enemy_kills += 1

        return turret_kills, enemy_kills

    def _in_los_of_any_turret(self) -> bool:
        a = (int(self._agent_pos[0]), int(self._agent_pos[1]))
        for t in self._turrets:
            if self._in_los(a, t):
                return True
        return False