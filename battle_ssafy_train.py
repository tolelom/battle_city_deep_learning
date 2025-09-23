import gymnasium as gym
import torch
import datetime
# from stable_baselines3 import PPO
from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker

from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor
from stable_baselines3.common.monitor import Monitor
# from sb3_contrib.common.callbacks import TqdmCallback  # 추가
from battle_ssafy_env import BattleSsafyEnv
from tqdm import tqdm
from stable_baselines3.common.callbacks import BaseCallback

class ProgressBarCallback(BaseCallback):
    def __init__(self, total_timesteps, verbose=0):
        super().__init__(verbose)
        self.total_timesteps = total_timesteps
        self.pbar = None

    def _on_training_start(self):
        self.pbar = tqdm(total=self.total_timesteps, desc="Training Progress")

    def _on_step(self):
        self.pbar.update(self.model.n_envs)  # 한 스텝마다 n_envs만큼 업데이트
        return True

    def _on_training_end(self):
        self.pbar.close()



# GPU 사용 설정
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

def mask_fn(env):
    # env 내부 상태로 마스크 계산 (obs 인자 필요 없음)
    return env.unwrapped._compute_valid_mask().astype(bool)

def make_env():
    env = BattleSsafyEnv(size=16, render_mode="human")
    env = ActionMasker(env, mask_fn)   # 여기서 래핑
    return env

# 로그 디렉터리
log_dir = "./tensorboard_logs/"

train_env = DummyVecEnv([make_env])     # ✅ ActionMasker 제거
eval_env  = ActionMasker(BattleSsafyEnv(size=16, render_mode="human"), mask_fn)

# 평가 콜백: 평균 리턴 ≥ 0.9 시 학습 중단 및 모델 저장
stop_callback = StopTrainingOnRewardThreshold(reward_threshold=0.9, verbose=1)
eval_callback = EvalCallback(
    eval_env,
    callback_on_new_best=stop_callback,
    eval_freq=10000,
    n_eval_episodes=1,
    best_model_save_path="./best_model",
    verbose=1,
)

# 콘솔 진행률 표시용 TqdmCallback
# tqdm_callback = TqdmCallback()

model = MaskablePPO(
    policy="MultiInputPolicy",
    env=train_env,
    learning_rate=3e-4,
    n_steps=2048,
    batch_size=64,
    n_epochs=10,
    gamma=0.99,
    verbose=1,
    device=device,
    tensorboard_log=log_dir,  # TensorBoard log directory
    policy_kwargs=dict(
        net_arch=dict(pi=[128, 128], vf=[128, 128]),
        activation_fn=torch.nn.ReLU,
    ),
)

progress_bar = ProgressBarCallback(total_timesteps=int(1e5))

env = BattleSsafyEnv()
obs, info = env.reset()
print("mask after reset:", env.action_masks())


model.learn(
    total_timesteps=int(1e5),
    callback=[progress_bar, eval_callback],
    # callback=[gif_cb, eval_callback],
)

timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
model.save("./best_model/ppo_battle_ssafy_{timestamp}")
print("학습 완료 및 모델 저장됨: ppo_battle_ssafy.zip")


# PyTorch state_dict 저장
# save_path = f"./best_model/ppo_battle_ssafy_{timestamp}.pt"
# torch.save(model.policy.state_dict(), save_path)
# print(f"학습 완료 및 모델 저장됨: {save_path}")