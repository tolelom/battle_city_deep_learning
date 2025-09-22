import gymnasium as gym
from battle_city_env import BattleCityEnv
from torch import tensor

env = BattleCityEnv(render_mode=None)
state_dim = env.observation_space.shape[0] * env.observation_space.shape[1]
action_dim = env.action_space.n

agent = BattleCityAgent(state_size=state_dim, action_size=action_dim)

# 학습 루프
for episode in range(1000):
    state, _ = env.reset()
    state = state.flatten()
    total_reward = 0

    while True:
        action = agent.act(state)
        next_state, reward, done, truncated, info = env.step(action)
        next_state = next_state.flatten()

        agent.memory.append((state, action, reward, next_state, done))
        agent.train()

        state = next_state
        total_reward += reward
        if done or truncated:
            break

    print(f"Episode {episode}: Reward = {total_reward:.2f}")
