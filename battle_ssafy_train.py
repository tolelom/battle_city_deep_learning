import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold
import tensorboard

# BattleSsafyEnv 클래스를 직접 import
from battle_ssafy_env import BattleSsafyEnv

# GPU 사용 설정
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

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

# 환경 생성 함수: gym.make 대신 직접 클래스 사용
def make_env(seed: int):
    def _init():
        env = BattleSsafyEnv(size=16, fixed_map=fixed_map)
        env.seed(seed)
        env = Monitor(env, filename=None, info_keywords=("distance",))
        return env
    return _init

n_envs = 4
train_env = DummyVecEnv([make_env(i) for i in range(n_envs)])
train_env = VecMonitor(train_env, "./logs/train/")

eval_env = DummyVecEnv([make_env(i + 100) for i in range(n_envs)])
eval_env = VecMonitor(eval_env, "./logs/eval/")

stop_callback = StopTrainingOnRewardThreshold(reward_threshold=0.9, verbose=1)
eval_callback = EvalCallback(
    eval_env,
    callback_on_new_best=stop_callback,
    eval_freq=1000,
    n_eval_episodes=5,
    best_model_save_path="./best_model",
    verbose=1,
)

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
    tensorboard_log="./tensorboard_logs/",
    policy_kwargs=dict(
        net_arch=dict(pi=[128, 128], vf=[128, 128]),
        activation_fn=torch.nn.ReLU,
    ),
)

model.learn(
    total_timesteps=int(1e5),
    callback=[eval_callback]
)

model.save("./best_model/ppo_battle_ssafy")
print("학습 완료 및 모델 저장됨: ppo_battle_ssafy.zip")
