import gymnasium as gym
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor

from battle_ssafy_env import ENV_ID, register_env


MODEL_PATH = "./best_model/best_model.zip"
N_EVAL_EPISODES = 100


def main() -> None:
    register_env()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = PPO.load(MODEL_PATH, device=device)

    eval_env = Monitor(gym.make(ENV_ID))

    mean_reward, std_reward = evaluate_policy(
        model,
        eval_env,
        n_eval_episodes=N_EVAL_EPISODES,
        render=False,
        reward_threshold=None,
        return_episode_rewards=False,
    )

    print(
        f"Mean reward over {N_EVAL_EPISODES} episodes: "
        f"{mean_reward:.3f} ± {std_reward:.3f}"
    )


if __name__ == "__main__":
    main()
