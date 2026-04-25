import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env

from battle_ssafy_env import BattleSsafyEnv, register_env


@pytest.fixture
def env():
    register_env()
    e = BattleSsafyEnv()
    yield e
    e.close()


def test_env_passes_gymnasium_checker(env):
    check_env(env)


def test_reset_returns_valid_obs(env):
    obs, info = env.reset(seed=0)
    assert set(obs.keys()) == {"agent", "target"}
    assert obs["agent"].shape == (2,)
    assert obs["target"].shape == (2,)
    assert not np.array_equal(obs["agent"], obs["target"])
    assert "distance" in info


def test_step_returns_valid_tuple(env):
    env.reset(seed=0)
    obs, reward, terminated, truncated, info = env.step(0)
    assert set(obs.keys()) == {"agent", "target"}
    assert isinstance(reward, float)
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)
    assert "distance" in info


def test_episode_can_terminate(env):
    env.reset(seed=0)
    terminated = False
    for _ in range(500):
        _, _, terminated, _, _ = env.step(env.action_space.sample())
        if terminated:
            break
    assert terminated, "에피소드가 500 스텝 내에 종료되지 않음"
