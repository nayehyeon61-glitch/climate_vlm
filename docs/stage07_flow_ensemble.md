# Stage 07: 디코더 뒤의 flow matching 앙상블 생성기

소유자 요청(2026-11-07): 지금은 앙상블인데 이것을 빼먹었다. VLA처럼, 다만 action이 아니라 flow matching으로 앙상블을
만드는 구조, 즉 downscaling → vision encoder → LLM → decoder → flow matching(앙상블 생성기)은 어떤가?
(참고 자료: Sapkota 외, VLA 개관 2505.04769; Lipman 외, Flow Matching 2210.02747)

가설: VLM 디코더 메모리를 보는 flow matching 생성기가 만든 멤버는 (1) 격자점별 가우시안 폭보다 CRPS가 낮고 spread/RMSE가
1에 가까우며, (2) 같은 생성기에서 VLM 메모리를 빼면 이 이득이 줄어든다.

> 상태: 코드와 테스트 완료. 시험 실행(stage 04 이류 모델)은 stage 06이 CPU를 비우면 시작하고, 본 실행은 stage 05 이류
> 모델로 합니다.

## 왜 필요한가

- 지금 디코더의 불확실성은 격자점마다 따로 낸 가우시안 폭입니다. 공간과 리드가 이어진 멤버를 만들지 못해서 pg-flow의
  멤버 형식(E_minus_x0)을 채울 수 없고, evaluate.py의 `--members`도 격자점마다 따로 뽑은 표본일 뿐입니다.
- VLA의 action expert(π0: VLM 토큰을 보는 작은 transformer가 flow matching으로 행동 덩어리를 10번의 적분으로 생성)와
  같은 자리에 "예보 전문가"를 둡니다. 행동 덩어리 대신 6개 리드 × 5개 변수의 예보 덩어리를 한 번에 생성하므로, 멤버 하나가
  공간과 리드에 걸쳐 이어진 궤적입니다.
- "결정론 평균 + 생성 잔차"는 D에 쓰는 CorrDiff(회귀 + 확산 잔차)와 같은 구조입니다.

## 설계

- VLM(D, E, P, LLM, G, F)은 고정합니다. CPU로는 FM 반복마다 VLM을 순전파·역전파할 수 없고, 고정하면 생성기 손실이 VLM을
  흔들지 않습니다. 표본마다 한 번 VLM을 돌려 저장합니다(`scripts/stage07_cache.py`):
  - F가 읽는 메모리 토큰(날씨 토큰, 예보 질의 토큰, G의 토큰),
  - F의 평균 μ(ClimODE 5변수, WB2 64×32 격자로 평균), 가우시안 폭 σ(시험 해만),
  - WB2 64×32 정답(리드별)과 발행 시각 정답 x0.
- 생성기(`climate_vlm/flow.py` FlowExpert, 약 1.6 M 매개변수): 리드마다 4×4 패치(8×16 토큰) × 6리드 = 768 토큰의
  transformer(4층, 폭 128). 토큰 입력은 x_t 패치와 μ 패치, x0 패치(갈래에 따라), 패치 위치와 리드 임베딩입니다. flow
  시간은 adaLN-Zero(DiT)로, VLM 메모리는 층마다 cross-attention으로 들어갑니다(게이트 0에서 시작).
- 목표는 잔차 r = (y − μ)/s입니다(s는 학습 표본의 리드·변수별 면적 가중 RMS). Lipman의 OT 경로
  x_t = (1 − (1 − σ_min)t)x_0 + t r와 목표 속도 r − (1 − σ_min)x_0를 cos 위도 가중 MSE로 학습합니다(식 22, 23).
  생성은 Euler 10 step, 멤버는 8개입니다(pg-flow와 같은 수).
- 학습: AdamW 3e-4, warmup 100, cosine, batch 16, EMA 0.999. 시험 실행 2,000 step, 본 실행 3,000 step.
  검증(2016 ClimODE IC)마다 이어 하기 파일을 씁니다.

## 갈래

| 갈래 | 잔차 기준 | 보는 것 | 묻는 것 |
|---|---|---|---|
| gaussian_head | | μ, σ (지금의 디코더) | 기준 |
| fm_vla | μ | μ + VLM 메모리 | 소유자의 구조 그대로 |
| fm_vla_x0 | μ | μ + VLM 메모리 + x0 | π0처럼 상태도 직접 줄 때 |
| fm_mu_x0 | μ | μ + x0 | VLM 메모리가 평균 이상의 정보를 주는가 |
| fm_x0 | x0 | x0 | VLM 없이 FM만(pg-flow와 비슷한 대조군) |

주의: 생성기가 x0을 보면 VLM 없이도 혼자 예보를 배울 수 있습니다. 같은 2012~15년으로 CPU에서 학습한 pg-flow FM
(`blend_4y`)은 2017~18년 z500 RMSE가 189 / 451 / 719(24 / 72 / 144 h)인데, VLM 이류 갈래는 2020년 516 / 748 / 827
입니다. 그래서 점수가 좋아지더라도 VLM 덕인지는 fm_mu_x0와 fm_x0와의 차이로만 말할 수 있습니다.

## 채점

2020 ClimODE 규약(IC 179개, WB2 64×32, `compare_climode.py`와 같은 정의): 앙상블 평균의 RMSE와 ACC, 멤버 평균과
퍼짐으로 계산한 가우시안 CRPS(ClimODE 식), 같은 단위의 공정한 앙상블 CRPS, spread/RMSE((M+1)/M 보정), 정답이 멤버 범위
밖에 있는 비율(보정된 8멤버면 2/9 = 0.22). persistence를 다시 계산해 기준 파일과 맞는지 확인합니다. 멤버는
`<out>/<갈래>/members_2020.npz`에 pg-flow 형식으로 남습니다.

## 결과

(실행 전)

## 재현

```bash
HF_HOME=/home/user/hf HF_HUB_OFFLINE=1 bash scripts/run_stage07.sh                                   # 시험 실행
CONFIG=configs/stage07_flow_ensemble_s05.yaml OMP_NUM_THREADS=4 HF_HOME=/home/user/hf HF_HUB_OFFLINE=1 \
    bash scripts/run_stage07.sh                                                                         # 본 실행
```

CPU 1스레드에서 VLM 저장은 표본당 약 1.8 s(학습 2,900개, 2016과 2020 ClimODE IC), 생성기 학습은 step당 약 2 s,
멤버 생성은 IC당 약 3 s(8멤버, 10 step)입니다.
