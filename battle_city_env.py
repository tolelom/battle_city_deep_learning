# import numpy as np
# import gymnasium as gym
# from gymnasium import spaces
#
# class BattleCityEnv(gym.Env):
#     metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 15}
#
#     def __init__(self, render_mode=None):
#         super().__init__()
#         # 1) 행동 공간: 8개 이산 행동 (이동 4 + 발사 조합 4)
#         self.action_space = spaces.Discrete(8)
#
#         # 2) 관찰 공간: 13x13 격자, 각 격자를 0~4로 인코딩된 정수형 상태로 표현
#         #    0=빈 공간, 1=벽돌, 2=강철, 3=플레이어, 4=적
#         self.observation_space = spaces.Box(
#             low=0, high=4, shape=(13, 13), dtype=np.int8
#         )
#
#         # 내부 상태 초기화
#         self.state = None
#         self.render_mode = render_mode
#         self.viewer = None
#
#     def reset(self, seed=None, options=None):
#         super().reset(seed=seed)
#         # 맵, 탱크, 기지, 아이템 위치 초기 배치
#         self.state = np.zeros((13, 13), dtype=np.int8)
#         # 예시: 가장자리에 벽돌/강철 배치
#         self.state[0, :] = 1
#         self.state[-1, :] = 1
#         self.state[:, 0] = 1
#         self.state[:, -1] = 1
#         # 플레이어 탱크 초기 위치
#         self.player_pos = [6, 6]
#         self.state[6, 6] = 3
#         # 적 탱크 초기 위치
#         self.enemy_positions = [[1, 1], [1, 11]]
#         for y, x in self.enemy_positions:
#             self.state[y, x] = 4
#         # 총알 목록, 점수, 종료 플래그 초기화
#         self.bullets = []
#         self.score = 0
#         self.done = False
#
#         if self.render_mode == "human":
#             self._init_renderer()
#         return self.state, {}
#
#     def step(self, action):
#         reward = 0
#         # 1) 행동 해석
#         move, fire = divmod(action, 5) if action < 5 else (action - 5, True)
#         # 실제 예시: 0=정지, 1=위, 2=아래, 3=왼, 4=오
#         direction = {1:(-1,0),2:(1,0),3:(0,-1),4:(0,1)}.get(move, (0,0))
#         # 2) 플레이어 이동
#         new_y = np.clip(self.player_pos[0] + direction[0], 0, 12)
#         new_x = np.clip(self.player_pos[1] + direction[1], 0, 12)
#         if self.state[new_y, new_x] in (0,):  # 빈 공간일 때만 이동
#             self.state[self.player_pos[0], self.player_pos[1]] = 0
#             self.player_pos = [new_y, new_x]
#             self.state[new_y, new_x] = 3
#
#         # 3) 발사 처리
#         if fire:
#             # 총알 초기 위치와 방향 저장
#             self.bullets.append({"pos": self.player_pos.copy(), "dir": direction})
#
#         # 4) 총알 이동 및 충돌 처리
#         remaining = []
#         for bullet in self.bullets:
#             by, bx = bullet["pos"]
#             dy, dx = bullet["dir"]
#             ny, nx = by + dy, bx + dx
#             if not (0 <= ny < 13 and 0 <= nx < 13):
#                 continue  # 화면 밖으로 나가면 제거
#             cell = self.state[ny, nx]
#             if cell == 1:  # 벽돌 부술 때
#                 reward += 1
#                 self.state[ny, nx] = 0
#             elif cell == 4:  # 적 파괴
#                 reward += 10
#                 self.enemy_positions.remove([ny, nx])
#                 self.state[ny, nx] = 0
#             else:
#                 remaining.append({"pos": [ny, nx], "dir": bullet["dir"]})
#         self.bullets = remaining
#
#         # 5) 에피소드 종료 여부
#         if not self.enemy_positions:
#             self.done = True
#             reward += 50  # 클리어 보너스
#
#         # 6) 시간 패널티
#         reward -= 0.1
#
#         # 7) 상태 반환
#         info = {"score": self.score}
#         return self.state, reward, self.done, False, info
#
#     def render(self):
#         # human 모드 렌더링 (간단한 텍스트 기반)
#         grid = self.state.copy()
#         render_map = {0:".", 1:"#", 2:"@", 3:"P", 4:"E"}
#         print("\n".join("".join(render_map[int(c)] for c in row) for row in grid))
#
#     def close(self):
#         pass
