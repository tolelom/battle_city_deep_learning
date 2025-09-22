from gymnasium.utils.env_checker import check_env
from battle_ssafy_env import BattleSsafyEnv
import gymnasium as gym


env = gym.make('BattleSsafyEnv-v0')

try:
    check_env(env.unwrapped)
    print("Environment passes all checks!")
except Exception as e:
    print(f"Environment has issues: {e}")
