# Code Cleanup & Stabilization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 배틀시티 강화학습 프로젝트 코드의 명백한 결함을 수정하고 구조/품질을 개선한 뒤 단일 커밋으로 `origin/main`에 푸시.

**Architecture:** 환경 정의(`battle_ssafy_env.py`), 학습 진입점(`battle_ssafy_train.py`), 평가 진입점(`battle_ssafy_eval.py`, 기존 `record.py` 개명), pytest 스모크 테스트(`tests/test_env.py`) 4개 모듈로 정돈. 환경 등록은 모듈 top-level side-effect 대신 `register_env()` 함수로 분리해 각 진입점에서 명시 호출.

**Tech Stack:** Python 3.13, Gymnasium, Stable Baselines3 (PPO), PyTorch, pytest.

**Reference Spec:** `docs/superpowers/specs/2026-04-25-code-cleanup-design.md`

---

## File Structure

| 파일 | 역할 | 작업 |
|---|---|---|
| `battle_ssafy_env.py` | Gymnasium 커스텀 환경 정의 및 등록 함수 | Modify |
| `battle_ssafy_train.py` | PPO 학습 진입점 | Modify |
| `battle_ssafy_eval.py` | 학습된 모델 평가 진입점 | Create (from `record.py`) |
| `battle_ssafy_record.py` | (삭제) | Delete |
| `battle_ssafy_env_checker.py` | (삭제 — 테스트로 대체) | Delete |
| `tests/test_env.py` | pytest 스모크 테스트 | Create |
| `requirements.txt` | 의존성 명세 | Create |
| `README.md` | 실행 커맨드 업데이트 | Modify |

---

## Task 1: requirements.txt 생성

**Files:**
- Create: `requirements.txt`

- [ ] **Step 1: 의존성 명세 파일 작성**

Create `requirements.txt`:

```
gymnasium>=1.0
stable-baselines3>=2.3
torch>=2.0
numpy>=1.26
tensorboard>=2.15
pytest>=8.0
```

---

## Task 2: battle_ssafy_env.py 리팩토링

**Files:**
- Modify: `battle_ssafy_env.py` (전체 교체)

변경 사항:
- 모듈 top-level `gym.register()` 호출 제거, `register_env()` 함수로 분리
- `ENV_ID` 상수 추가
- `metadata`의 `render_modes`에서 `rgb_array` 제거 (미구현)
- 타입 힌트 보강 (`reset`, `step`, `_get_obs`, `_get_info`, `render`)
- `reward`를 `10.0` (float)로 통일, `terminated`에 `bool()` 명시 캐스트
- 중복 주석 정리

- [ ] **Step 1: battle_ssafy_env.py 전체 교체**

Replace entire file contents:

```python
from typing import Optional

import gymnasium as gym
import numpy as np


ENV_ID = "BattleSsafyEnv-v0"


def register_env() -> None:
    gym.register(
        id=ENV_ID,
        entry_point="battle_ssafy_env:BattleSsafyEnv",
        max_episode_steps=100,
    )


class BattleSsafyEnv(gym.Env):
    metadata = {
        "render_modes": ["human"],
        "render_fps": 30,
    }

    def __init__(self, size: int = 16):
        self.size = size

        # 위치 초기값 (-1, -1): reset 전 사용 방지 가드
        self._agent_location = np.array([-1, -1], dtype=np.int32)
        self._target_location = np.array([-1, -1], dtype=np.int32)

        self.observation_space = gym.spaces.Dict(
            {
                "agent": gym.spaces.Box(0, size - 1, shape=(2,), dtype=int),
                "target": gym.spaces.Box(0, size - 1, shape=(2,), dtype=int),
            }
        )

        self.action_space = gym.spaces.Discrete(4)

        self._action_to_direction = {
            0: np.array([1, 0]),
            1: np.array([0, 1]),
            2: np.array([-1, 0]),
            3: np.array([0, -1]),
        }

    def _get_obs(self) -> dict:
        return {"agent": self._agent_location, "target": self._target_location}

    def _get_info(self) -> dict:
        return {
            "distance": np.linalg.norm(
                self._agent_location - self._target_location, ord=1
            )
        }

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ) -> tuple[dict, dict]:
        super().reset(seed=seed)

        self._agent_location = self.np_random.integers(0, self.size, size=2, dtype=int)

        self._target_location = self._agent_location
        while np.array_equal(self._target_location, self._agent_location):
            self._target_location = self.np_random.integers(
                0, self.size, size=2, dtype=int
            )

        return self._get_obs(), self._get_info()

    def step(self, action: int) -> tuple[dict, float, bool, bool, dict]:
        direction = self._action_to_direction[action]

        self._agent_location = np.clip(
            self._agent_location + direction, 0, self.size - 1
        )

        terminated = bool(np.array_equal(self._agent_location, self._target_location))
        truncated = False
        reward = 10.0 if terminated else -0.005

        return self._get_obs(), reward, terminated, truncated, self._get_info()

    def render(self) -> None:
        if self.render_mode != "human":
            return
        for y in range(self.size - 1, -1, -1):
            row = ""
            for x in range(self.size):
                if np.array_equal([x, y], self._agent_location):
                    row += "A "
                elif np.array_equal([x, y], self._target_location):
                    row += "T "
                else:
                    row += ". "
            print(row)
        print()
```

---

## Task 3: tests/test_env.py 생성 (TDD)

**Files:**
- Create: `tests/test_env.py`
- Delete: `battle_ssafy_env_checker.py`

- [ ] **Step 1: 테스트 파일 작성**

Create `tests/test_env.py`:

```python
from gymnasium.utils.env_checker import check_env

from battle_ssafy_env import BattleSsafyEnv, register_env


def test_env_passes_gymnasium_checker():
    register_env()
    env = BattleSsafyEnv()
    check_env(env)
```

- [ ] **Step 2: 테스트 실행하여 통과 확인**

Run:
```bash
pytest tests/ -v
```

Expected: `1 passed`. 만약 실패하면 `battle_ssafy_env.py` 수정(Task 2)을 점검.

- [ ] **Step 3: 기존 체커 스크립트 삭제**

Run:
```bash
rm battle_ssafy_env_checker.py
```

---

## Task 4: battle_ssafy_train.py 리팩토링

**Files:**
- Modify: `battle_ssafy_train.py` (전체 교체)

변경 사항:
- 하이퍼파라미터를 파일 상단 상수로 분리
- `register_env()` 호출 추가
- `model.learn(callback=eval_callback)` 연결 (버그 수정)
- `reward_threshold=0.9` → `REWARD_THRESHOLD=9.0` (스케일 일치)
- `main()` 함수 + `if __name__ == "__main__":` 가드 추가
- 주석 처리된 `TqdmCallback` 데드코드 제거
- 미사용 `import gymnasium as gym` 제거

- [ ] **Step 1: battle_ssafy_train.py 전체 교체**

Replace entire file contents:

```python
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
```

- [ ] **Step 2: import 정상 확인**

Run:
```bash
python -c "import battle_ssafy_train; print('import OK')"
```

Expected: `import OK` (의존성이 설치된 환경에서만 유효. torch/sb3 미설치 시 ModuleNotFoundError는 환경 이슈이며 코드 문제 아님 — 사용자에게 보고 후 pass)

---

## Task 5: battle_ssafy_eval.py 생성 및 record.py 삭제

**Files:**
- Create: `battle_ssafy_eval.py`
- Delete: `battle_ssafy_record.py`

변경 사항:
- 파일명 `record.py` → `eval.py` (실제 기능인 evaluate_policy와 일치)
- `MODEL_PATH`, `N_EVAL_EPISODES` 상수화
- `register_env()` 호출 추가
- `main()` + `if __name__ == "__main__":` 가드

- [ ] **Step 1: battle_ssafy_eval.py 생성**

Create `battle_ssafy_eval.py`:

```python
import gymnasium as gym
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor

from battle_ssafy_env import ENV_ID, register_env


MODEL_PATH = "./best_model/ppo_battle_ssafy.zip"
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
```

- [ ] **Step 2: 기존 record.py 삭제**

Run:
```bash
rm battle_ssafy_record.py
```

---

## Task 6: README.md 업데이트

**Files:**
- Modify: `README.md` (전체 교체)

변경 사항:
- 설치 섹션 추가 (`pip install -r requirements.txt`)
- `battle_ssafy_record.py` → `battle_ssafy_eval.py` 반영
- `battle_ssafy_env_checker.py` → `pytest tests/` 반영

- [ ] **Step 1: README.md 전체 교체**

Replace entire file contents:

````markdown
# Battle City Deep Learning

배틀시티 게임을 위한 강화학습 프로젝트.
Gymnasium 커스텀 환경 + Stable Baselines3 PPO로 에이전트 학습.

## Tech Stack

- **환경**: Gymnasium (커스텀 BattleSsafyEnv)
- **학습**: Stable Baselines3 (PPO)
- **시각화**: TensorBoard

## 설치

```bash
pip install -r requirements.txt
```

## 실행

```bash
# 학습
python battle_ssafy_train.py

# 평가
python battle_ssafy_eval.py

# 테스트
pytest tests/
```
````

---

## Task 7: 최종 검증

**Files:** (검증만 — 파일 변경 없음)

- [ ] **Step 1: pytest 최종 실행**

Run:
```bash
pytest tests/ -v
```

Expected: `1 passed`.

- [ ] **Step 2: git status 확인**

Run:
```bash
git status
```

Expected (파일 목록):
```
modified:   README.md
modified:   battle_ssafy_env.py
modified:   battle_ssafy_train.py
deleted:    battle_ssafy_env_checker.py
deleted:    battle_ssafy_record.py

Untracked files:
  battle_ssafy_eval.py
  requirements.txt
  tests/test_env.py
```

만약 `docs/superpowers/plans/...` 나 `__pycache__` 가 untracked로 보이면 의도된 것이므로 무시.

- [ ] **Step 3: pytest cache 정리 확인**

`.pytest_cache/`가 생성되었을 수 있음. `.gitignore`에 `__pycache__/`는 있지만 `.pytest_cache/`는 없으므로 확인:

Run:
```bash
ls -la | grep pytest_cache
```

있으면 `.gitignore`에 추가:

Edit `.gitignore` — 마지막 줄에 추가:
```
.pytest_cache/
```

---

## Task 8: 단일 커밋 생성

**Files:** (git 작업 — 파일 변경 없음)

- [ ] **Step 1: 변경 파일 스테이징**

Run:
```bash
git add battle_ssafy_env.py battle_ssafy_train.py battle_ssafy_eval.py README.md requirements.txt tests/test_env.py .gitignore
git rm battle_ssafy_env_checker.py battle_ssafy_record.py
```

- [ ] **Step 2: 플랜 문서 스테이징**

Run:
```bash
git add docs/superpowers/plans/2026-04-25-code-cleanup.md
```

- [ ] **Step 3: 단일 커밋 생성**

Run:
```bash
git commit -m "$(cat <<'EOF'
refactor: 코드 정리 및 안정화

- battle_ssafy_env.py: register_env() 분리, 타입 힌트, rgb_array metadata 제거
- battle_ssafy_train.py: EvalCallback 연결, reward_threshold 스케일 수정(0.9→9.0), 하이퍼파라미터 상수화
- battle_ssafy_record.py → battle_ssafy_eval.py (실제 기능과 파일명 일치)
- battle_ssafy_env_checker.py → tests/test_env.py (pytest 스모크)
- requirements.txt 추가
- README 업데이트

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

- [ ] **Step 4: 커밋 확인**

Run:
```bash
git log -1 --stat
```

Expected: 위 파일들이 변경/추가/삭제되어 있음.

---

## Task 9: 푸시 (사용자 최종 확인 필수)

**Files:** (git 작업)

- [ ] **Step 1: 사용자에게 푸시 확인 요청**

커밋 로그를 출력하고 "origin/main 으로 푸시해도 될까요?" 라고 사용자에게 확인받는다. 명시적 승인 없이는 다음 단계로 진행하지 않음.

- [ ] **Step 2: 푸시 실행 (승인 후)**

Run:
```bash
git push origin main
```

Expected: 푸시 성공 출력.

---

## 완료 조건 (Spec 5번 대응)

1. ✅ `pytest tests/` 통과 (Task 7 Step 1)
2. ✅ `battle_ssafy_train.py` 가 import 에러 없이 로드 (Task 4 Step 2)
3. ✅ 단일 커밋 작성 (Task 8)
4. ✅ 사용자 최종 확인 후 `origin/main` 푸시 (Task 9)
