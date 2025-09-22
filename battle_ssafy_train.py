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
    return BattleSsafyEnv(size=16)

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

# 학습: Tqdm + EvalCallback 동시 사용
model.learn(
    total_timesteps=int(1e5),
    # callback=[tqdm_callback, eval_callback]
)

# model.save("./best_model/ppo_battle_ssafy")
# print("학습 완료 및 모델 저장됨: ppo_battle_ssafy.zip")

timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
save_path = f"./ppo_battle_ssafy_{timestamp}.pt"

# PyTorch state_dict 저장
torch.save(model.policy.state_dict(), save_path)
print(f"학습 완료 및 모델 저장됨: {save_path}")