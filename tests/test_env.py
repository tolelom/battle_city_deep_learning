from gymnasium.utils.env_checker import check_env

from battle_ssafy_env import BattleSsafyEnv, register_env


def test_env_passes_gymnasium_checker():
    register_env()
    env = BattleSsafyEnv()
    check_env(env)
