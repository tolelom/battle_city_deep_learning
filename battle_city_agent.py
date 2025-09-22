# agents/battle_city_agent.py

import random
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


class DQNNetwork(nn.Module):
    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class BattleCityAgent:
    def __init__(
        self,
        state_size: int,
        action_size: int,
        lr: float = 1e-3,
        gamma: float = 0.99,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.1,
        epsilon_decay: float = 0.995,
        buffer_size: int = 10000,
        batch_size: int = 64,
        target_update_freq: int = 1000,
    ):
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = gamma

        # ε-greedy
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay

        # 경험 리플레이 버퍼
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size

        # 네트워크와 옵티마이저
        self.q_network = DQNNetwork(state_size, action_size)
        self.target_network = DQNNetwork(state_size, action_size)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)

        # 타겟 네트워크 동기화 주기
        self.target_update_freq = target_update_freq
        self.learn_step_counter = 0

        # 장치 설정 (GPU 사용 시)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.q_network.to(self.device)
        self.target_network.to(self.device)
        self._sync_target_network()

    def _sync_target_network(self):
        self.target_network.load_state_dict(self.q_network.state_dict())

    def act(self, state: np.ndarray) -> int:
        """현재 정책에 따라 행동 선택 (ε-greedy)"""
        if random.random() < self.epsilon:
            return random.randrange(self.action_size)
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        q_values = self.q_network(state_tensor)
        return q_values.argmax().item()

    def store_transition(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool
    ):
        """경험을 버퍼에 저장"""
        self.memory.append((state, action, reward, next_state, done))

    def train(self):
        """버퍼에서 샘플링하여 네트워크 학습"""
        if len(self.memory) < self.batch_size:
            return
        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states_np = np.array(states, dtype=np.float32)
        actions_np = np.array(actions, dtype=np.int64)
        rewards_np = np.array(rewards, dtype=np.float32)
        next_states_np = np.array(next_states, dtype=np.float32)
        dones_np = np.array(dones, dtype=np.bool_)

        states = torch.from_numpy(states_np).to(self.device)
        actions = torch.from_numpy(actions_np).unsqueeze(1).to(self.device)
        rewards = torch.from_numpy(rewards_np).to(self.device)
        next_states = torch.from_numpy(next_states_np).to(self.device)
        dones = torch.from_numpy(dones_np).to(self.device)

        # 현재 Q 값
        q_pred  = self.q_network(states).gather(1, actions).squeeze()

        # 다음 상태에서 최대 Q 값 (타겟 네트워크)
        q_next  = self.target_network(next_states).max(dim=1)[0].detach()
        q_target = rewards + (self.gamma * q_next * (~dones))

        # 손실 및 업데이트
        loss = nn.MSELoss()(q_pred, q_target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # ε 감소
        if self.epsilon > self.epsilon_end:
            self.epsilon *= self.epsilon_decay

        # 타겟 네트워크 동기화
        self.learn_step_counter += 1
        if self.learn_step_counter % self.target_update_freq == 0:
            self._sync_target_network()

    def update_target_network(self):
        """외부 호출용: 수동으로 타겟 네트워크 동기화"""
        self._sync_target_network()

    def save(self, filepath: str):
        torch.save(self.q_network.state_dict(), filepath)

    def load(self, filepath: str):
        self.q_network.load_state_dict(torch.load(filepath, map_location=self.device))
        self._sync_target_network()
