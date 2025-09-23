from typing import Optional, Tuple, Dict, List
import numpy as np
import gymnasium as gym
from gymnasium import spaces

gym.register(
    id='BattleSsafyEnv-v0',
    entry_point="battle_ssafy_env:BattleSsafyEnv",
    max_episode_steps=100,
)

G, R, W, F, X, E, S, T, H = 0, 1, 2, 3, 4, 5, 6, 7, 8

class BattleSsafyEnv(gym.Env):
    # metadata = {"render_modes": ["human", "ansi"], "render_fps": 30}
    metadata = {"render_modes": ["human", "ansi", "rgb_array"], "render_fps": 30}


    def __init__(self, size: int = 16, render_mode: Optional[str] = None, n_enemies: int = 3):
        """_summary_

        Args:
            size (int, optional): _description_. Defaults to 16.
            render_mode (Optional[str], optional): _description_. Defaults to None.
            n_enemies (int, optional): _description_. Defaults to 3.
            state = {
                "map": (16,16),               # 지형 + 포탑 + 적 표시
                "agent_pos":     # 에이전트 위치
                "agent_hp":       # 체력
                "agent_bomb":     # 일반 폭탄 수
                "agent_mega_bomb":  # 메가폭탄 수
                "enemies_pos": # 각 적 좌표, 없으면 [-1,-1]
                "enemies_hp":   # 각 적 체력, 없으면 0
            }
        """
        assert size == 16, "요청 사양: 16x16 맵"
        self.H = self.W = size
        self.render_mode = render_mode
        self.n_enemies = n_enemies

        # === Observation space (요청 반영) ===
        self.observation_space = spaces.Dict(
            {
                "map": spaces.Box(low=0, high=8, shape=(self.H, self.W), dtype=np.int8),
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
        self.action_space = spaces.Discrete(7)

        self._dir = {
            0: np.array([1, 0], dtype=np.int32),
            1: np.array([0, 1], dtype=np.int32),
            2: np.array([-1, 0], dtype=np.int32),
            3: np.array([0, -1], dtype=np.int32),
        }

        # 내부 상태(포지션/HP/탄수 등)
        self._base_map = np.zeros((self.H, self.W), dtype=np.int8)  # 지형만
        self._turrets: List[Tuple[int, int]] = []  # X 좌표 리스트
        self._turrets_hp: List[int] = []  # 포탑 체력 병렬 리스트  <-- 추가
        
        self._agent_pos = np.array([0, 0], dtype=np.int32)
        self._agent_hp = 100
        self._agent_bomb = 3
        self._agent_mega_bomb = 1

        # 적: 포지션 + HP
        self._enemies_pos: List[Optional[np.ndarray]] = [None] * self.n_enemies
        self._enemies_hp = np.zeros(self.n_enemies, dtype=np.int32)

        self._rng = None

        # 죽은 적과, 우리팀을 count
        self._enemy_kill_count = 0
        self._agent_death_count = 0

        # 나(우리팀) 포탑(H)도 HP 트래킹 (X와 대칭 구조)
        self._home_turrets: List[Tuple[int,int]] = []
        self._home_turrets_hp: List[int] = []


    # ============ Gym API ============
    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        self._rng = self.np_random

        # 1) 지형 초기화
        # self._base_map[:] = G
        # self._random_scatter(R, count=20)  # 벽
        # self._random_scatter(W, count=15)  # 물
        # self._random_scatter(F, count=4)   # 보급소
        # self._turrets = self._random_scatter(X, count=3, as_list=True)  # 포탑
        self._base_map = np.array([
            [R, R, R, G, G, G, F, G, G, F, G, G, G, G, G, X],
            [R, G, G, G, G, G, S, S, S, S, G, G, G, G, G, G],
            [R, G, G, G, G, G, S, R, R, S, G, G, G, G, G, G],
            [G, G, G, G, G, G, R, R, R, R, G, G, G, G, G, G],
            [G, G, G, G, G, T, G, G, G, G, T, G, G, G, G, G],
            [G, G, R, R, W, W, T, G, G, T, W, W, R, R, G, G],
            [G, G, T, G, G, G, G, G, G, G, G, G, G, T, G, G],
            [G, G, R, R, G, R, G, W, W, G, R, G, R, R, G, G],
            [G, G, R, R, G, R, G, W, W, G, R, G, R, R, G, G],
            [G, G, T, G, G, G, G, G, G, G, G, G, G, T, G, G],
            [G, G, R, R, W, W, T, G, G, T, W, W, R, R, G, G],
            [G, G, G, G, G, T, G, G, G, G, T, G, G, G, G, G],
            [G, G, G, G, G, G, R, R, R, R, G, G, G, G, G, G],
            [G, G, G, G, G, G, S, R, R, S, G, G, G, G, G, R],
            [G, G, G, G, G, G, S, S, S, S, G, G, G, G, G, R],
            [H, G, G, G, G, G, F, G, G, F, G, G, G, R, R, R],
        ], dtype=np.int8)

        # 적 포탑
        self._turrets = [(x, y)
                 for y in range(self.H)
                 for x in range(self.W)
                 if self._base_map[y, x] == X]
        self._turrets_hp = [100 for _ in range(len(self._turrets))]

        # 내 포탑
        self._home_turrets = [(x, y)
                 for y in range(self.H)
                 for x in range(self.W)
                 if self._base_map[y, x] == H]
        self._home_turrets_hp = [100 for _ in range(len(self._home_turrets))]

        # 카운터 리셋
        self._enemy_kill_count = 0
        self._agent_death_count = 0

        # 2) 에이전트 초기화
        self._agent_pos = np.array([10, 0], dtype=np.int32)
        self._agent_hp = 100
        self._agent_bomb = 99
        self._agent_mega_bomb = 0

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
        if action in (0, 1, 2, 3):
            nxt = self._agent_pos + self._dir[action]
            nxt = np.clip(nxt, [0, 0], [self.W - 1, self.H - 1])
            if self._is_passable(tuple(nxt)):
                self._agent_pos = nxt
            # 이동/대기 패널티
            reward += -0.01

        # 2) 폭탄
        elif action == 4:
            if self._agent_bomb > 0:
                self._agent_bomb -= 1
                t_kill, e_kill, tree_kill, dmg = self._fire_bomb(mega=False, splash=False)
                # 보상: 파괴 보상 + 데미지 비례 보상
                reward += 10.0 * t_kill + 5.0 * e_kill
                reward += 0.05 * dmg   # 데미지당 보상(수치 조절 가능)
                if t_kill == 0 and e_kill == 0 and tree_kill == 0:
                    reward += -3 # 빗나감 소소 패널티
            else:
                reward += -0.01

        # 3) 메가폭탄
        elif action == 5:
            if self._agent_mega_bomb > 0:
                self._agent_mega_bomb -= 1
                t_kill, e_kill, tree_kill, dmg = self._fire_bomb(mega=True, splash=False)
                reward += 10.0 * t_kill + 5.0 * e_kill
                reward += 0.05 * dmg
                if t_kill == 0 and e_kill == 0 and tree_kill == 0:
                    reward += -3
            else:
                reward += -0.01
        
        # 4) 암호풀기 (훅: 현재 no-op + 패널티)
        elif action == 6:
            reward += 0

        # 보급 체크
        self._try_refill_mega()

        # --- Enemy attack phase: 적의 공격(Agent는 오직 이 공격으로 데미지 받음) ---
        # 규칙: 적은 같은 행/열에 있고 LOS가 성립하며 거리 <= 3 이면 공격 성공
        ENEMY_ATTACK_POWER = 30  # 필요 시 조정
        for i, pos in enumerate(self._enemies_pos):
            if pos is None or self._enemies_hp[i] <= 0:
                continue
            ex, ey = int(pos[0]), int(pos[1])

            if self._enemy_can_attack_agent(ex, ey):
                self._agent_hp -= ENEMY_ATTACK_POWER
                reward += -5.0  # 맞았을 때 패널티(튜닝 가능)
                if self._agent_hp <= 0:
                    self._agent_death_count += 1
                    # if self._agent_death_count >= 2 :
                    terminated = True
                    break                         # 여러 적의 추가 타격은 무의미

        # === Termination Rules ===
        # 1) 상대 포탑(X) 제거
        if not terminated and len(self._turrets) == 0:
            reward += 50.0
            terminated = True

        # 2) 적 enemies 2개 이상 제거
        if not terminated and self._enemy_kill_count >= 2:
            reward += 50.0
            terminated = True

        # 3) 내 포탑(H) 전부 제거
        if not terminated and len(self._home_turrets) == 0 and len(self._home_turrets_hp) == 0:
            reward += -50.0
            terminated = True

        # 4) 내 agent 2회 이상 사망
        if not terminated and self._agent_death_count >= 2:
            reward += -50.0
            terminated = True

        if self.render_mode == "human":
            self.render()

        return self._get_obs(), float(reward), terminated, truncated, self._get_info()

    def render(self):
        grid = self._compose_visible_map()
        if self.render_mode == "rgb_array":
            return self._rgb_from_grid(cell=20)

        # 아래는 기존 텍스트 렌더
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
        return {
            "nearest_turret_L1": td,
            "enemy_kill_count": self._enemy_kill_count,
            "agent_death_count": self._agent_death_count,
            "num_enemy_turrets": len(self._turrets),
            "num_home_turrets": len(self._home_turrets),
        }

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
        return {
            G: ".",
            R: "R",
            W: "W",
            F: "F",
            X: "X",
            E: "E",
            T: "T",   # 나무
            S: "S",   # 모래
            H: "H",   # 내 포탑
        }.get(int(val), "?")

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
        # 못 지나는 타일: R, F, T, H, W (H는 내 포탑이라 통행 불가로 가정)
        if t in (R, F, T, H, W):
            return False
        # 나머지는 통행 가능 (G, S, 등)
        # 적/포탑 점유 칸은 별도 차단
        if (xy in self._turrets):
            return False
        for i, p in enumerate(self._enemies_pos):
            if p is not None and self._enemies_hp[i] > 0 and (p[0], p[1]) == xy:
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
    
    def _enemy_can_attack_agent(self, ex: int, ey: int) -> bool:
        """적 (ex,ey) 가 에이전트를 상하좌우 직선으로 거리 ≤3 에서 공격 가능한지 판정.
          차단 타일 R/F를 만나면 그 방향 탐색 중단. W는 통과."""
        ax, ay = int(self._agent_pos[0]), int(self._agent_pos[1])

        # 같은 행
        if ey == ay:
            dx = 1 if ax > ex else -1
            for step in range(1, 4):  # 거리 1..3
                x = ex + dx * step
                if x < 0 or x >= self.W: break
                t = self._base_map[ey, x]
                if t in (R, F):  # 차단
                    break
                if x == ax:      # 에이전트를 만났다면 성공
                    return True
            return False

        # 같은 열
        if ex == ax:
            dy = 1 if ay > ey else -1
            for step in range(1, 4):  # 거리 1..3
                y = ey + dy * step
                if y < 0 or y >= self.H: break
                t = self._base_map[y, ex]
                if t in (R, F):  # 차단
                    break
                if y == ay:      # 에이전트를 만났다면 성공
                    return True
            return False

        # 대각선/기타 불가
        return False


    def _first_target_in_line(self) -> Tuple[int, int, Optional[int], Optional[Tuple[int,int]]]:
        """에이전트 기준 상/하/좌/우로 거리 1..3만 직선 스캔.
        차단(R,F)을 만나면 그 방향 탐색을 중단.
        가장 가까운 하나의 타겟만 반환.
        return: (turret_hit(0/1), enemy_hit(0/1), enemy_index or None, pos or None)
        """
        ax, ay = int(self._agent_pos[0]), int(self._agent_pos[1])
        best: Optional[Tuple[int, str, Optional[int], Tuple[int,int]]] = None  # (dist, kind, e_idx, pos)

        for dx, dy in ((1,0), (-1,0), (0,1), (0,-1)):
            for dist in range(1, 4):
                x, y = ax + dx*dist, ay + dy*dist
                if not (0 <= x < self.W and 0 <= y < self.H):
                    break

                t = self._base_map[y, x]
                if t in (R, F):  # 차단 타일이면 이 방향 중단
                    break

                # 포탑(X) 먼저 확인
                for tx, ty in self._turrets:
                    if tx == x and ty == y:
                        cand = (dist, "turret", None, (x, y))
                        if best is None or dist < best[0]:
                            best = cand
                        break
                if best is not None and best[3] == (x, y):
                    break  # 이 방향에서 타겟 찾았으니 다음 방향으로

                # 적(E) 확인
                hit_enemy = False
                for e_idx, epos in enumerate(self._enemies_pos):
                    if epos is not None and self._enemies_hp[e_idx] > 0:
                        if int(epos[0]) == x and int(epos[1]) == y:
                            cand = (dist, "enemy", e_idx, (x, y))
                            if best is None or dist < best[0]:
                                best = cand
                            hit_enemy = True
                            break
                if hit_enemy:
                    break  # 이 방향에서 타겟 찾았으니 다음 방향으로

        if best is None:
            return 0, 0, None, None

        _, kind, e_idx, pos = best
        if kind == "turret":
            return 1, 0, None, pos
        else:
            return 0, 1, e_idx, pos



    def _fire_bomb(self, mega: bool, splash: bool) -> Tuple[int, int, int, float]:
        """
        폭탄/메가폭탄 발사.
        단일 타격, 사거리 3, splash 플래그는 무시(호환용).
        return: (turret_kills, enemy_kills, tree_kills, total_damage_dealt)
        - turret/enemy: hp 감소, hp <= 0 이면 제거(킬)
        - tree(T): 폭탄 위력 >= 1이면 제거(일반폭탄으로 제거 가능). 제거하면 tree_kills += 1
        """
        # 위력 설정
        power = 70 if mega else 30

        total_damage = 0.0
        turret_kills = 0
        enemy_kills = 0
        tree_kills = 0

        # 1) 찾은 타겟(가장 가까운 한 개)을 처리 (기존 _first_target_in_line 로 사용)
        t_hit, e_hit, e_idx, pos = self._first_target_in_line()
        if pos is None:
            # 명중 대상 없음 -> 폭발했지만 아무도 안 맞음 (자해 없음)
            return turret_kills, enemy_kills, tree_kills, total_damage

        # helper to try damage turret at pos
        def damage_turret_at(p: Tuple[int,int]):
            nonlocal total_damage, turret_kills
            # find turret index in list
            for i, tpos in enumerate(self._turrets):
                if tpos == p:
                    dmg = min(power, self._turrets_hp[i])
                    self._turrets_hp[i] -= power
                    total_damage += dmg
                    if self._turrets_hp[i] <= 0:
                        tx, ty = tpos
                        self._base_map[ty, tx] = G
                        self._turrets.pop(i)
                        self._turrets_hp.pop(i)
                        turret_kills += 1
                    return True
            return False

        # helper to damage enemy index
        def damage_enemy(idx: int):
            nonlocal total_damage, enemy_kills
            if idx is None:
                return False
            if 0 <= idx < len(self._enemies_hp) and self._enemies_pos[idx] is not None and self._enemies_hp[idx] > 0:
                dmg = min(power, self._enemies_hp[idx])
                self._enemies_hp[idx] -= power
                total_damage += dmg
                if self._enemies_hp[idx] <= 0:
                    # kill enemy: 좌표 제거 + map 갱신
                    ex, ey = int(self._enemies_pos[idx][0]), int(self._enemies_pos[idx][1])
                    self._enemies_pos[idx] = None
                    # 만약 그 칸에 적을 표시하기 위해 base_map에 다른 표식이 있었다면 평지로 바꿔줌
                    # (일반적으로 적은 base_map에 없지만 안전하게 갱신)
                    if 0 <= ex < self.W and 0 <= ey < self.H:
                        self._base_map[ey, ex] = G
                    enemy_kills += 1
                    self._enemy_kill_count += 1
                return True
            return False

        # helper to destroy a tree at p (if present)
        def destroy_tree_at(p: Tuple[int,int]):
            nonlocal total_damage, tree_kills
            x, y = p
            if self._base_map[y, x] == T:
                # 나무는 일반폭탄으로도 제거 가능 - 위력 충족하면 제거
                # (여기서는 어떤 폭탄이든 제거하도록 함)
                # 데미지는 power (단, 실질적으로 tree에는 HP 보관 안함)
                total_damage += power
                self._base_map[y, x] = G  # 제거 -> 통과 가능
                tree_kills += 1
                return True
            return False
        
        # helper to try damage HOME turret at pos (H)
        def damage_home_turret_at(p: Tuple[int,int]):
            nonlocal total_damage
            for i, tpos in enumerate(self._home_turrets):
                if tpos == p:
                    dmg = min(power, self._home_turrets_hp[i])
                    self._home_turrets_hp[i] -= power
                    total_damage += dmg
                    if self._home_turrets_hp[i] <= 0:
                        hx, hy = tpos
                        self._base_map[hy, hx] = G
                        self._home_turrets.pop(i)
                        self._home_turrets_hp.pop(i)
                    return True
            return False
        
        # 메인 타겟 처리(한 점)
        # 우선순위: X -> E -> T -> H
        if t_hit and pos is not None:
            damage_turret_at(pos)
        elif e_hit and e_idx is not None:
            damage_enemy(e_idx)
        else:
            # (타겟이 실제로 나무/우리포탑인지 확인하고 처리)
            if not destroy_tree_at(pos):
                damage_home_turret_at(pos)

        return turret_kills, enemy_kills, tree_kills, total_damage
    
    def _rgb_from_grid(self, cell=20) -> np.ndarray:
        """타일 맵을 간단한 컬러 이미지로 변환."""
        import numpy as np
        H, W = self.H, self.W
        img = np.zeros((H*cell, W*cell, 3), dtype=np.uint8)

        # 타일 팔레트 (원하는 색으로 바꿔도 됨)
        palette = {
            G: (220, 220, 220),  # 평지
            R: (70, 70, 70),     # 바위/벽
            W: (120, 170, 255),  # 물
            F: (255, 220, 0),    # 보급소
            X: (255, 80, 80),    # 적 포탑
            E: (80, 80, 255),    # 적
            T: (0, 160, 0),      # 나무
            S: (210, 180, 140),  # 모래
            H: (80, 255, 120),   # 우리 포탑
        }

        v = self._compose_visible_map()
        for y in range(H):
            for x in range(W):
                color = palette.get(int(v[y, x]), (0, 0, 0))
                ys, xs = y*cell, x*cell
                img[ys:ys+cell, xs:xs+cell] = color

        # 에이전트는 테두리로 표시(검은색)
        ax, ay = int(self._agent_pos[0]), int(self._agent_pos[1])
        ys, xs = ay*cell, ax*cell
        img[ys:ys+cell, xs:xs+cell] = (0, 0, 0)
        pad = 2
        img[ys+pad:ys+cell-pad, xs+pad:xs+cell-pad] = (255, 255, 255)  # 내부 흰색

        return img
