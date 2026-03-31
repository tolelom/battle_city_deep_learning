# Battle City Deep Learning

배틀시티 게임을 위한 강화학습 프로젝트.
Gymnasium 커스텀 환경 + Stable Baselines3 PPO로 에이전트 학습.

## Tech Stack

- **환경**: Gymnasium (커스텀 BattleSsafyEnv)
- **학습**: Stable Baselines3 (PPO)
- **시각화**: TensorBoard

## 실행

```bash
# 학습
python battle_ssafy_train.py

# 환경 검증
python battle_ssafy_env_checker.py

# 녹화
python battle_ssafy_record.py
```
