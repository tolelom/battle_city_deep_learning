import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import (
    EvalCallback,
    StopTrainingOnRewardThreshold,
)
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor

from battle_ssafy_env import BattleSsafyEnv, register_env


TOTAL_TIMESTEPS = int(1e5)
LEARNING_RATE = 3e-4
N_STEPS = 2048
BATCH_SIZE = 64
N_EPOCHS = 10
GAMMA = 0.99
EVAL_FREQ = 1000
N_EVAL_EPISODES = 5
REWARD_THRESHOLD = 9.0
ENV_SIZE = 16
LOG_DIR = "./tensorboard_logs/"
BEST_MODEL_DIR = "./best_model"


def make_env() -> BattleSsafyEnv:
    return BattleSsafyEnv(size=ENV_SIZE)


def main() -> None:
    register_env()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    train_env = VecMonitor(DummyVecEnv([make_env]), LOG_DIR)
    eval_env = VecMonitor(DummyVecEnv([make_env]), LOG_DIR)

    stop_callback = StopTrainingOnRewardThreshold(
        reward_threshold=REWARD_THRESHOLD, verbose=1
    )
    eval_callback = EvalCallback(
        eval_env,
        callback_on_new_best=stop_callback,
        eval_freq=EVAL_FREQ,
        n_eval_episodes=N_EVAL_EPISODES,
        best_model_save_path=BEST_MODEL_DIR,
        verbose=1,
    )

    model = PPO(
        policy="MultiInputPolicy",
        env=train_env,
        learning_rate=LEARNING_RATE,
        n_steps=N_STEPS,
        batch_size=BATCH_SIZE,
        n_epochs=N_EPOCHS,
        gamma=GAMMA,
        verbose=1,
        device=device,
        tensorboard_log=LOG_DIR,
        policy_kwargs=dict(
            net_arch=dict(pi=[128, 128], vf=[128, 128]),
            activation_fn=torch.nn.ReLU,
        ),
    )

    model.learn(
        total_timesteps=TOTAL_TIMESTEPS,
        callback=eval_callback,
    )

    save_path = f"{BEST_MODEL_DIR}/ppo_battle_ssafy"
    model.save(save_path)
    print(f"학습 완료 및 모델 저장됨: {save_path}.zip")


if __name__ == "__main__":
    main()
