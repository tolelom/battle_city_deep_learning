import socket
import torch
import numpy as np
from battle_ssafy_env import BattleSsafyEnv
from stable_baselines3.common.preprocessing import preprocess_obs
from stable_baselines3.ppo.policies import MultiInputPolicy
from stable_baselines3.common.utils import FloatSchedule

HOST = '127.0.0.1'
PORT = 8747
MODEL_PATH = './ppo_battle_ssafy.pt'
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

sock = socket.socket()

# 문자열 타일 → 정수 인코딩
tile_map = {
    'G': 0,  # Grass
    'R': 1,  # Rock
    'W': 2,  # Water
    'F': 3,  # Supply
    'X': 4,  # Turret
    'E': 5,  # Enemy tank
    'A': 0,  # Agent tank treated as grass in map
    'H': 4,  # Ally turret also marked as turret
    'T': 1,  # Tree treated as rock
}

def init(nickname: str) -> str:
    sock.connect((HOST, PORT))
    return submit(f'INIT {nickname}')

def submit(cmd: str) -> str:
    sock.send((cmd + ' ').encode('utf-8'))
    return receive()

def receive() -> str:
    data = sock.recv(4096).decode()
    # 종료 또는 에러: 맨 앞 숫자가 0 이하일 때 연결 종료
    if not data or int(data[0]) <= 0:
        close()
        return None
    return data

def close():
    sock.close()

def parse_data(game_data: str) -> dict:
    """
    서버로부터 받은 문자열을 파싱하여 Gym 환경 관측(obs) 형태로 반환
    """
    # 빈 라인 제거
    rows = [r for r in game_data.strip().split('\n') if r]
    # 헤더 파싱
    h, w, n_allies, n_enemies, n_codes = map(int, rows[0].split())
    idx = 1

    # 1) 맵 타일 파싱
    raw_map = [rows[idx + i].split() for i in range(h)]
    idx += h
    grid = np.zeros((h, w), dtype=np.int8)
    for i in range(h):
        for j in range(w):
            grid[i, j] = tile_map.get(raw_map[i][j], 0)

    # 2) 아군 정보 파싱: A 하나만 사용
    a_hp = a_bomb = a_mega = 0
    agent_pos = [0, 0]
    for _ in range(n_allies):
        parts = rows[idx].split()
        idx += 1
        if parts and parts[0] == 'A' and len(parts) >= 7:
            a_hp = int(parts[1])
            # 서버가 보내는 좌표 순서: y x
            agent_pos = [int(parts[2]), int(parts[3])]
            a_bomb = int(parts[5])
            a_mega = int(parts[6])

    # 3) 적군(포탑) 정보 파싱: X만 사용
    turret_positions = []
    turret_hps = []
    for _ in range(n_enemies):
        parts = rows[idx].split()
        idx += 1
        if parts and parts[0] == 'X' and len(parts) >= 4:
            turret_hps.append(int(parts[1]))
            turret_positions.append([int(parts[2]), int(parts[3])])

    # 4) 암호문 라인 건너뛰기
    idx += n_codes

    # 5) 슬롯 맞추기 (env.n_enemies 만큼)
    while len(turret_positions) < env_dummy.n_enemies:
        turret_positions.append([-1, -1])
        turret_hps.append(0)

    # 6) 최종 obs 딕셔너리 생성
    obs = {
        "map":             grid,
        "agent_pos":       np.array(agent_pos, dtype=np.int32),
        "agent_hp":        np.array([a_hp],        dtype=np.int32),
        "agent_bomb":      np.array([a_bomb],      dtype=np.int32),
        "agent_mega_bomb": np.array([a_mega],      dtype=np.int32),
        "enemies_pos":     np.array(turret_positions, dtype=np.int32),
        "enemies_hp":      np.array(turret_hps,       dtype=np.int32),
    }
    return obs

# Stable Baselines3 정책 네트워크 로드
env_dummy = BattleSsafyEnv()
policy = MultiInputPolicy(
    env_dummy.observation_space,
    env_dummy.action_space,
    lr_schedule=FloatSchedule(3e-4),
    net_arch=dict(pi=[128, 128], vf=[128, 128]),
    activation_fn=torch.nn.ReLU,
)
state_dict = torch.load(MODEL_PATH, map_location=DEVICE)
policy.load_state_dict(state_dict)
policy.to(DEVICE)
policy.eval()

# 행동 인덱스 → 서버 커맨드 매핑
action_map = {
    0: 'D A',  # 아래 전진
    1: 'U A',  # 위 전진
    2: 'L A',  # 왼쪽 전진
    3: 'R A',  # 오른쪽 전진
    4: 'S F',  # 폭탄 투하
    5: 'S M',  # 메가폭탄 투하
    6: 'S P',  # 암호풀기
}

def select_action(obs: dict) -> str:
    obs_proc = preprocess_obs(obs, env_dummy.observation_space, DEVICE)
    with torch.no_grad():
        action_tensor, _ = policy(obs_proc, deterministic=True)
    action_idx = int(action_tensor.cpu().numpy()[0])
    return action_map[action_idx]

if __name__ == '__main__':
    NICKNAME = '짜잔내가돌아왔다'
    game_data = init(NICKNAME)
    while game_data:
        obs = parse_data(game_data)
        cmd = select_action(obs)
        game_data = submit(cmd)
    close()
