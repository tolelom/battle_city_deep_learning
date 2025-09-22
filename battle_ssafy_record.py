import battle_ssafy_env
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor

# 1. Load the trained model
model = PPO.load("./best_model/ppo_battle_ssafy.zip", device="cuda")

# 2. Create evaluation environment
raw_env = gym.make("BattleSsafyEnv-v0")
eval_env = Monitor(raw_env)

# 3. Run evaluation
mean_reward, std_reward = evaluate_policy(
    model,
    eval_env,
    n_eval_episodes=100,
    render=False,              # Set True to render each step
    reward_threshold=None,
    return_episode_rewards=False
)

print(f"Mean reward over 100 episodes: {mean_reward:.3f} ± {std_reward:.3f}")