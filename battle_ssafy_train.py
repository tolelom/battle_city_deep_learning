import gymnasium as gym
import torch
import datetime
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor
# from sb3_contrib.common.callbacks import TqdmCallback  # 추가
from battle_ssafy_env import BattleSsafyEnv

# GPU 사용 설정
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

def make_env():
    return BattleSsafyEnv(size=16, render_mode="human")

# 로그 디렉터리
log_dir = "./tensorboard_logs/"

# 벡터 환경 및 모니터링 (학습/평가 환경 동일하게 VecMonitor 적용)
train_env = DummyVecEnv([make_env])
train_env = VecMonitor(train_env, log_dir)

eval_env = DummyVecEnv([make_env])
eval_env = VecMonitor(eval_env, log_dir)

# 평가 콜백: 평균 리턴 ≥ 0.9 시 학습 중단 및 모델 저장
stop_callback = StopTrainingOnRewardThreshold(reward_threshold=0.9, verbose=1)
eval_callback = EvalCallback(
    eval_env,
    callback_on_new_best=stop_callback,
    eval_freq=1000,
    n_eval_episodes=5,
    best_model_save_path="./best_model",
    verbose=1,
)

# 콘솔 진행률 표시용 TqdmCallback
# tqdm_callback = TqdmCallback()

model = PPO(
    policy="MultiInputPolicy",
    env=train_env,
    learning_rate=3e-4,
    n_steps=2048,
    batch_size=64,
    n_epochs=10,
    gamma=0.99,
    verbose=1,
    device=device,
    tensorboard_log=log_dir,  # TensorBoard 로그 경로 지정
    policy_kwargs=dict(
        net_arch=dict(pi=[128, 128], vf=[128, 128]),
        activation_fn=torch.nn.ReLU,
    ),
)


import os, datetime, numpy as np
import imageio.v2 as imageio
from stable_baselines3.common.callbacks import BaseCallback
from battle_ssafy_env import BattleSsafyEnv

class GifCallback(BaseCallback):
    def __init__(self, base_dir="./gifs",
                 rollout_steps=200, fps=8,
                 target_gif_count=10, verbose=1):
        super().__init__(verbose)
        self.base_dir = base_dir
        self.rollout_steps = rollout_steps
        self.fps = fps
        self.target_gif_count = max(1, int(target_gif_count))

        # 저장 디렉토리와 카운터는 _init_callback에서 세팅
        self.save_dir = None
        self.gif_counter = 0
        self._rollout_idx = 0
        self._save_every_rollout = 1

    def _on_step(self) -> bool:
        return True

    def _init_callback(self) -> None:
        # 학습 시작 시점에 타임스탬프 폴더 생성
        ts = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.save_dir = os.path.join(self.base_dir, ts)
        os.makedirs(self.save_dir, exist_ok=True)

        # 총 롤아웃 수 추정 -> 몇 번에 한 번 저장할지 결정
        n_envs = self.model.n_envs
        n_steps = self.model.n_steps
        total = self.model._total_timesteps
        total_rollouts = max(1, total // (n_steps * n_envs))
        self._save_every_rollout = max(1, total_rollouts // self.target_gif_count)

        if self.verbose:
            print(f"[GifCallback] save_dir={self.save_dir}, "
                  f"total_rollouts≈{total_rollouts}, "
                  f"save_every_rollout={self._save_every_rollout}")

    def _on_rollout_end(self) -> None:
        self._rollout_idx += 1
        if self._rollout_idx % self._save_every_rollout == 0:
            self._record_one_episode()

    def _record_one_episode(self):
        env = BattleSsafyEnv(size=16, render_mode="rgb_array")
        obs, info = env.reset(seed=np.random.randint(1_000_000))

        frames = []
        for t in range(self.rollout_steps):
            action, _ = self.model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(int(action))
            frame = env.render()
            frames.append(frame.copy())
            if terminated or truncated:
                break

        if len(frames) == 1:
            frames.append(frames[0].copy())

        # 파일명: gif1.gif, gif2.gif, …
        self.gif_counter += 1
        filename = f"gif{self.gif_counter}.gif"
        path = os.path.join(self.save_dir, filename)

        imageio.mimsave(path, frames, duration=1.0/self.fps, loop=0)
        if self.verbose:
            print(f"[GifCallback] saved: {path} ({len(frames)} frames)")

        env.close()



gif_cb = GifCallback(
    base_dir="./gifs",
    rollout_steps=200,   # 녹화 길이(프레임 수)
    fps=8,
    target_gif_count=10, # 전체 학습 동안 10개만 저장
    verbose=1,
)

model.learn(
    total_timesteps=int(1e5),
    callback=[gif_cb, eval_callback],
)

# model.save("./best_model/ppo_battle_ssafy")
# print("학습 완료 및 모델 저장됨: ppo_battle_ssafy.zip")

timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
save_path = f"./best_model/ppo_battle_ssafy_{timestamp}.pt"

# PyTorch state_dict 저장
torch.save(model.policy.state_dict(), save_path)
print(f"학습 완료 및 모델 저장됨: {save_path}")