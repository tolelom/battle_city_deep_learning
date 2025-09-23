import numpy as np
import torch
import gymnasium as gym
import matplotlib.pyplot as plt
from battle_ssafy_env import BattleSsafyEnv
from stable_baselines3 import PPO

# 1) 환경과 모델 불러오기
# 고정 맵 정의
fixed_map = np.array([
    [3, 3, 3, 0, 0, 0, 5, 0, 0, 5, 0, 0, 0, 0, 0, 0],
    [3, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0],
    [3, 0, 0, 0, 0, 0, 1, 3, 3, 1, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 3, 3, 3, 3, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 4, 0, 0, 0, 0, 4, 0, 0, 0, 0, 0],
    [0, 0, 3, 3, 2, 2, 4, 0, 0, 4, 2, 2, 3, 3, 0, 0],
    [0, 0, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4, 0, 0],
    [0, 0, 3, 3, 0, 3, 0, 2, 2, 0, 3, 0, 3, 3, 0, 0],
    [0, 0, 3, 3, 0, 3, 0, 2, 2, 0, 3, 0, 3, 3, 0, 0],
    [0, 0, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4, 0, 0],
    [0, 0, 3, 3, 2, 2, 4, 0, 0, 4, 2, 2, 3, 3, 0, 0],
    [0, 0, 0, 0, 0, 4, 0, 0, 0, 0, 4, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 3, 3, 3, 3, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 1, 3, 3, 1, 0, 0, 0, 0, 0, 3],
    [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 3],
    [0, 0, 0, 0, 0, 0, 5, 0, 0, 5, 0, 0, 0, 3, 3, 3],
], dtype=np.int32)

env = BattleSsafyEnv(size=16, fixed_map=fixed_map)
model = PPO.load("./best_model/ppo_battle_ssafy", device="cpu")

# 2) 에피소드 실행 및 렌더링
obs, info = env.reset()
plt.ion()
fig, ax = plt.subplots()

for _ in range(200):
    action, _ = model.predict(obs, deterministic=True)
    obs, _, done, _, info = env.step(action)
    frame = env.render(mode="rgb_array")
    ax.imshow(frame)
    ax.set_axis_off()
    plt.pause(1 / env.metadata["render_fps"])
    ax.clear()
    if done:
        break

plt.ioff()
plt.show()
