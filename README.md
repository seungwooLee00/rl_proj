# 강화학습 기반 MCU 대상 Neural Architecture Search

MCU 환경에서 동작 가능한 효율적 이미지 분류 모델을 찾기 위해  
제약 조건(Flash, SRAM, Latency)을 동시에 고려하는 Neural Architecture Search(NAS)를 설계·구현한 프로젝트이다.

이 프로젝트는 두 가지 탐색 방식을 비교·분석하는 데 초점을 둔다.

1. 유전자 알고리즘(GA)을 이용한 서브넷 탐색  
2. PPO(Proximal Policy Optimization)를 이용한 정책 기반 강화학습 NAS  

---

## 1. 프로젝트 주제와 목표

### 1.1 프로젝트 주제

- MCU 환경에서 동작 가능한 효율적 이미지 분류 모델을 찾기 위한  
  강화학습 기반 Neural Architecture Search(NAS) 설계 및 평가

### 1.2 프로젝트 목표

- Flash, SRAM, Latency 제약을 동시에 만족하는 서브넷(subnet) 탐색  
- 기존 유전자 알고리즘(GA) 기반 탐색 대비 더 나은 정확도–지연시간–자원 사용량 trade-off 확보  
- PPO를 활용한 정책 기반 NAS가 제약 기반 모델 탐색에서 실질적인 성능 향상을 달성할 수 있는지 검증  
- 보상 설계, 탐색 절차, 예측기 기반 환경 구성 등 강화학습 탐색이 성공적으로 작동하기 위한 조건 분석  

---

## 2. 프로젝트 구조

```text
.  
|-- README.md  
|-- requirements.txt  
|-- run_ga.py  
|-- run_rl.py  
|  
|-- assets  
|   `-- predictors  
|       |-- accuracy_predictor.pth  
|       |-- flash_predictor.pth  
|       |-- latency_predictor.pth  
|       |-- sram_predictor.pth  
|       `-- violation_predictor.pth  
|  
|-- common  
|   |-- encoder.py  
|   |-- predictor.py  
|   `-- subnet_config.py  
|  
|-- ga  
|   |-- ga_config.py  
|   `-- ga_searcher.py  
|  
|-- ga_results  
|  
|-- rl  
|   |-- ppo_agent.py  
|   |-- rl_config.py  
|   `-- subnet_env.py  
|  
|-- rl_results  
```

---

## 3. 방법론 개요

### 3.1 검색 공간과 예측기

- `common/subnet_config.py`  
  서브넷 아키텍처 검색 공간 정의  

- `common/encoder.py`  
  아키텍처를 벡터로 인코딩  

- `common/predictor.py (MLPPredictor)`  
  정확도 / Latency / Flash / SRAM을 예측  
  MCU 실측 대신 예측기로 빠른 탐색 수행  

---

### 3.2 GA 기반 NAS

- `ga/ga_searcher.py` : GA 기반 탐색 알고리즘을 정의하는 GASearcher 클래스
- `ga/ga_config.py` : GA 탐색에 필요한 하이퍼파라미터 dataclass
- `run_ga.py` : GA 탐색 실행 스크립트

---

### 3.3 RL(PPO) 기반 NAS

- `rl/subnet_env.py`: 환경 정의 (state, action, reward 정의)
- `rl/ppo_agent.py`: SB3 PPO 에이전트 래핑  
- `rl/rl_config.py`: PPOConfig, SubnetEnvConfig 및 환경 설정 클래스  
- `run_rl.py` : PPO 하이퍼파라미터 그리드 서치, 보상 파라미터 그리드 서치

---

## 4. 실행 환경

### 4.1 Docker 기반 실행 환경

- 베이스 이미지: PyTorch + CUDA 12.1 (torch-cuda12.1)  
- apt 설치 패키지: git  
- Python 의존성: requirements.txt  

```bash
apt update && apt install -y git
cd /workspace/project
pip install -r requirements.txt
```

---

### 4.2 모델 가중치

예측기 가중치는 이미 프로젝트 내부에 포함됨:

```text
assets/predictors/
    accuracy_predictor.pth
    latency_predictor.pth
    flash_predictor.pth
    sram_predictor.pth
```

---

## 5. 코드 실행 방법

### 5.1 GA 기반 NAS 실행

```bash
python3 run_ga.py
```

결과 저장 위치:

```text
ga_results/ga_latency_sweep.json
```

---

### 5.2 PPO 기반 NAS 실행

```bash
python3 run_rl.py
```

결과 저장 위치:

- Stage 1: `rl_results/grid_summary_stage1.json`  
- Stage 2: `rl_results/grid_summary_stage2.json`  
- 최종 선택 결과: `rl_results/best_configs.json`  

TensorBoard 로그는 각 stage 폴더의 `tb/`에 생성됨.

---

## 6. TensorBoard 시각화

```bash
tensorboard --logdir ./rl_results --host 0.0.0.0 --port 6006
```

브라우저에서 확인:

```
http://localhost:6006
```

---

## 7. 재현성과 실험 설정

- 고정 SEED: 120250336  
- GA는 latency·trial index 기반 seed 설정  
- CUDA 사용 시 자동 디바이스 선택  
- RL 실험은 cuDNN deterministic/bemchmark 옵션 설정으로 재현성 강화  

---

## 8. 정리

이 프로젝트는 예측기 기반 NAS 환경 위에서  

- GA 탐색  
- PPO 기반 강화학습 탐색  

을 비교하여, MCU 제약을 만족하는 서브넷을 효율적으로 탐색하는 방법을 실험적으로 검증한다.

보상 설계, 하이퍼파라미터 튜닝, 제약 기반 탐색 전략 등  
강화학습이 NAS 문제에서 어떤 강점을 가질 수 있는지 분석하는 것을 목표로 한다.