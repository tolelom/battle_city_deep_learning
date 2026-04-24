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
