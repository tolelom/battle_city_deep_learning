import gymnasium as gym
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor
from battle_ssafy_env import BattleSsafyEnv

# GPU 사용 설정
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

def make_env():
    return BattleSsafyEnv(size=16)

# 벡터 환경 및 모니터링
train_env = DummyVecEnv([make_env])
train_env = VecMonitor(train_env)

eval_env = DummyVecEnv([make_env])

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

model = PPO(
    policy="MultiInputPolicy",
    env=train_env,
    learning_rate=3e-4,
    n_steps=2048,
    batch_size=64,
    n_epochs=10,
    gamma=0.99,
    verbose=1,
    device=device,                # GPU 혹은 CPU 자동 선택
    policy_kwargs=dict(
        # 필요 시 네트워크 구조 조정 예시
        net_arch=[dict(pi=[128, 128], vf=[128, 128])],
        activation_fn=torch.nn.ReLU,
    ),
)

model.learn(total_timesteps=int(1e5), callback=eval_callback)
model.save("ppo_battle_ssafy")

print("학습 완료 및 모델 저장됨: ppo_battle_ssafy.zip")
