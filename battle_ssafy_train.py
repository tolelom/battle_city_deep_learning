import gymnasium as gym
from battle_ssafy_env import BattleSsafyEnv
from battle_city_agent import BattleCityAgent  # 구현한 에이전트

# 환경 생성
env = gym.make("BattleSSafyEnv-v0")

# 에이전트 초기화
state_dim = sum(env.observation_space["agent"].shape)  # 2
action_dim = env.action_space.n  # 4
agent = BattleCityAgent(state_size=state_dim, action_size=action_dim)

# 학습 루프
for episode in range(1, 100001):
    obs, info = env.reset()
    state = obs["agent"]  # 혹은 필요한 전처리
    done = False
    total_reward = 0

    while not done:
        action = agent.act(state)
        next_obs, reward, terminated, truncated, info = env.step(action)
        next_state = next_obs["agent"]
        done = terminated or truncated

        agent.store_transition(state, action, reward, next_state, done)
        agent.train()

        state = next_state
        total_reward += reward

    agent.update_target_network()  # DQN 계열인 경우
    print(f"Episode {episode}: Total Reward = {total_reward:.2f}")
