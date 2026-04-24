# Code Cleanup & Stabilization — Design Spec

**Date:** 2026-04-25
**Scope:** 전체 프로젝트 코드 정리 및 안정화 (단일 커밋 → `origin/main` 푸시)

## 1. 목적

현재 배틀시티 강화학습 프로젝트는 초기 상태로, 여러 명백한 결함과 구조적 개선 여지가 있다:

- `battle_ssafy_train.py`의 `EvalCallback`이 `model.learn()`에 연결되지 않아 동작하지 않음
- `reward_threshold=0.9`가 실제 reward 스케일(성공 시 +10, 스텝 -0.005)과 맞지 않음
- `battle_ssafy_record.py`는 실제로는 evaluation만 수행하며 파일명과 기능이 불일치
- `gym.register()`가 환경 모듈 import 시점에 실행되는 side-effect
- `rgb_array` render 모드가 metadata에만 선언되고 미구현
- 의존성 명세(`requirements.txt`) 부재
- 테스트 파일이 `check_env` 호출 스크립트 1개에 불과

학습 동작(reward 함수, PPO 하이퍼파라미터)은 이번 작업 대상에서 제외한다 — 별도 튜닝 작업으로 분리.

## 2. 변경 범위

### 2.1 파일별 변경

| 파일 | 변경 내용 |
|---|---|
| `battle_ssafy_env.py` | 타입 힌트 보강, `register_env()` 함수로 분리, `render()` 정리 (`rgb_array` 제거), 상수 정돈 |
| `battle_ssafy_train.py` | callback 연결, `reward_threshold` 수정, 하이퍼파라미터 상단 상수화, 주석 데드코드 제거 |
| `battle_ssafy_record.py` → `battle_ssafy_eval.py` | 파일명 변경, 타입 힌트 보강, `register_env()` 호출 |
| `battle_ssafy_env_checker.py` → `tests/test_env.py` | pytest 포맷으로 변환 (smoke test 1개) |
| `requirements.txt` | 신규 생성 |
| `README.md` | 실행 커맨드 업데이트 (파일명 변경, `pytest` 실행 추가) |

### 2.2 버그/안정화 상세 (behavior 영향 있음)

**`train.py` — callback 연결**
```python
# Before
model.learn(
    total_timesteps=int(1e5),
    # callback=[tqdm_callback, eval_callback]
)

# After
model.learn(
    total_timesteps=TOTAL_TIMESTEPS,
    callback=eval_callback,
)
```

**`train.py` — reward_threshold 스케일 일치**
- Reward 스케일: 성공 에피소드 reward ≈ `10 - 0.005 * steps` (≈ 9 이상)
- `reward_threshold=0.9` → `reward_threshold=9.0`

### 2.3 구조 개선 상세 (behavior 영향 없음)

**`battle_ssafy_env.py` — `register_env()` 분리**
```python
# Before: 모듈 top-level에서 즉시 register()
gym.register(id='BattleSsafyEnv-v0', ...)

# After: 함수로 감싸서 호출자가 명시적으로 호출
def register_env() -> None:
    gym.register(
        id='BattleSsafyEnv-v0',
        entry_point='battle_ssafy_env:BattleSsafyEnv',
        max_episode_steps=100,
    )
```

`train.py`, `eval.py`, `tests/test_env.py`에서 필요 시 명시 호출.

**`render()` 정리**
- `metadata`의 `render_modes`에서 `rgb_array` 제거 (미구현)
- `human` 모드만 유지

**파일 개명**
- `battle_ssafy_record.py` → `battle_ssafy_eval.py` (실제로 `evaluate_policy` 호출하므로 `record`는 오해의 소지)

### 2.4 품질 강화 상세

**타입 힌트**
- `reset`, `step`, `_get_obs`, `_get_info`, `render`, `register_env` 시그니처에 반환 타입 포함
- 예: `def step(self, action: int) -> tuple[dict, float, bool, bool, dict]:`

**하이퍼파라미터 상수화 (train.py 상단)**
```python
TOTAL_TIMESTEPS = int(1e5)
LEARNING_RATE = 3e-4
N_STEPS = 2048
BATCH_SIZE = 64
N_EPOCHS = 10
GAMMA = 0.99
EVAL_FREQ = 1000
N_EVAL_EPISODES = 5
REWARD_THRESHOLD = 9.0
LOG_DIR = "./tensorboard_logs/"
BEST_MODEL_DIR = "./best_model"
```

argparse/CLI 추가는 scope 벗어남 — 상수로만 정리.

**`requirements.txt`**
```
gymnasium>=1.0
stable-baselines3>=2.3
torch>=2.0
numpy>=1.26
tensorboard>=2.15
pytest>=8.0
```
엄격 pinning은 과함 — 하한만 지정.

**`tests/test_env.py`**
```python
from gymnasium.utils.env_checker import check_env
from battle_ssafy_env import BattleSsafyEnv, register_env

def test_env_passes_gymnasium_checker():
    register_env()
    env = BattleSsafyEnv()
    check_env(env)
```
`pytest tests/` 로 실행.

## 3. 최종 디렉토리 구조

```
battle_city_deep_learning/
├── battle_ssafy_env.py
├── battle_ssafy_train.py
├── battle_ssafy_eval.py        # renamed from record.py
├── tests/
│   └── test_env.py             # converted from env_checker.py
├── requirements.txt            # new
├── README.md                   # updated
├── docs/
│   └── superpowers/
│       └── specs/
│           └── 2026-04-25-code-cleanup-design.md
├── best_model/                 # gitignored
└── tensorboard_logs/           # gitignored
```

## 4. Non-Goals (명시적 제외)

- Reward shaping 변경 (distance 기반 보상 등)
- PPO 하이퍼파라미터 튜닝
- 배틀시티 요소(탱크, 총알, 적) 추가
- argparse/CLI 인터페이스 추가
- Env 내부 동작 상세 테스트 (reward 값, clip, action 매핑 등)
- `rgb_array` render 모드 구현 (제거하는 방향)

## 5. 완료 조건

1. `python battle_ssafy_train.py` 가 import 에러 없이 학습 시작 (몇 스텝만 확인)
2. `pytest tests/` 통과
3. 단일 커밋 작성 (메시지: "코드 정리 및 안정화")
4. 사용자 최종 확인 후 `origin/main` 푸시

## 6. 리스크 및 대응

- **기존 저장 모델과 호환성**: 환경 observation/action space를 변경하지 않으므로 기존 `best_model/ppo_battle_ssafy.zip`과 호환 유지
- **`register_env()` 분리로 인한 미호출 가능성**: `train.py`, `eval.py`에서 `BattleSsafyEnv`를 직접 `DummyVecEnv`에 넣거나, `gym.make("BattleSsafyEnv-v0")` 사용 시 `register_env()` 선행 호출 필수 — 각 진입점에서 명시적으로 호출하도록 수정
- **Python 3.14 호환**: 실행 환경은 PyCharm 설정상 Python 3.13. `tuple[dict, ...]` PEP 585 문법은 3.9+ 지원이므로 문제 없음
