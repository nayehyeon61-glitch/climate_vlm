# Stage 04: neural operator로 vision encoder 앞에서 분해하고 마지막에 결합

소유자 요청(2026-11-06): neuraloperator(https://github.com/neuraloperator/neuraloperator)로 vision encoder 앞에서 한 번
분해하고 마지막에 결합하는 방식, 즉 ClimateLLM(arXiv 2502.11059)과 ClimODE 같은 방식을 쓰면 어떤가.

가설: 기상장을 공간 규모별로 나눠서 넣고 규모별로 다시 결합하면, 같은 학습량에서 stage 01 모델(규모 구분 없이 영상으로 넣고
격자에서 바로 출력)보다 예측이 좋아진다. 특히 오차가 가장 큰 종관 규모에서 좋아진다.

> 상태: 코드, 설정, 테스트 완료. 학습은 stage 03이 끝난 뒤 CPU에서 돌립니다. 결과 절은 그때 채웁니다.

## 두 논문에서 가져온 것

- ClimateLLM: 정규화한 장을 2D FFT로 주파수 영역에 보내고, 주파수별 전문가(MoE)와 게이트를 거쳐 LLM에 넣은 뒤 역 FFT로
  되돌립니다. 다만 6~24 h만 보고했고, 6 h z RMSE는 ClimODE보다 나쁩니다(143.2 대 112.3). 또 위경도 격자에서 2D FFT는
  위도 방향을 주기적으로 취급하므로 구면에서는 맞지 않습니다.
- Neural Operator(JMLR 2022): 격자 해상도와 무관한 연산자 학습. 구면에서는 neuraloperator의 SFNO(구면조화 기저의
  spectral convolution, torch-harmonics 사용)가 맞는 형태입니다. physicsnemo 클론에는 FNO/AFNO만 있고 SFNO는 없어서
  neuraloperator 2.0.0과 torch-harmonics 0.8.0을 씁니다.
- ClimODE: 연속 방정식 du/dt = −∇·(u v) + s 를 학습한 속도장 v와 소스 s로 적분합니다.

## 설계

1. 분해 (vision encoder 앞, `climate_vlm/spectral.py` SphericalBands):
   - D_μ가 만든 2.8125° 장(64×128)을 구면조화함수로 세 대역으로 나눕니다: l ≤ 6(행성 규모, 파장 약 6,700 km 이상),
     7 ≤ l ≤ 15(종관 규모, 약 2,700~5,700 km), l > 15(그보다 작은 규모).
   - 경도는 rfft, 위도는 차수 m별 Legendre 투영이고, 적분은 극을 포함하지 않는 칸 중심 격자의 규칙인 Fejér 첫째 규칙입니다.
     이 격자에서 l + l' < n_lat 이면 정확하므로, 경계 15는 64×32 채점 격자에서도 정확한 투영입니다.
   - 마지막 대역은 나머지(원래 장 − 아래 대역들)라서 분해와 결합에 손실이 없습니다.
   - 입력이 5.625°(위도 32줄)라서 l > 31 성분은 관측이 아니라 D가 만든 디테일입니다.
2. 인코더:
   - 갈래 (a) `bands`: 대역 장 세 개와 경향(tendency)을 각각 영상으로 만들어 기존의 동결된 Qwen vision tower에 넣습니다
     (stage 01의 lr, mean, tendency 대신 band0, band1, band2, tendency).
   - 갈래 (b) `sfno`: 같은 대역 장을 neuraloperator SFNO(은닉 64, 4층, 학습함)에 넣고, 나온 잠재장을 8×8 patch 토큰
     128개로 잘라 Perceiver resampler P가 LLM 토큰 64개로 읽습니다. vision tower를 쓰지 않습니다.
3. LLM: stage 01과 같습니다(Qwen3.5-0.8B 동결, 학습 단계 2).
4. 결합 (마지막, F):
   - (a), (b): F가 채널마다 대역별 지도 세 개를 내고, 각 지도를 자기 대역에 투영해 더한 뒤 기반장(D_μ의 발행 시각 장)에
     더합니다. 대역별 출력이 서로 섞이지 않습니다.
   - 갈래 (c) `sfno_advect`: (b)에서 z500, t850, t2m은 F가 리드 구간마다 속도장(격자 칸/시간, tanh로 1칸/시간 이하)과
     소스를 내고, 반-라그랑주 연속 방정식 적분(6 h 간격, 경도 주기, 압축 항 포함)으로 발행 시각 장에서 24 h부터 144 h까지
     이어서 옮깁니다. u10, v10은 입력 상태에 없어서 대역 결합을 그대로 씁니다. ClimODE와 달리 속도장은 2계 ODE가 아니라
     LLM 기억을 읽은 F가 구간마다 정하고, 구 계량 없이 격자 단위로 계산합니다(ClimODE와 같음).

## 비교 방법

- 대조군: stage 01 학습 단계 2(`runs/stage01/stage2.pt`). 같은 D(`runs/stage01/stage1.pt`), 같은 자료, 같은 학습량
  (900 step, batch 4, lr 3e-4, 진단 손실 0.1)으로 세 갈래를 학습합니다.
- 채점: stage 01과 같은 ClimODE 규약(2020년 IC 179개, WB2 64×32, RMSE, ACC, Gaussian CRPS)에 더해, 오차를 같은 세 대역으로
  나눈 RMSE(`scripts/compare_climode.py --bands 6,15`)를 봅니다. persistence 검사는 그대로 통과합니다(상대 차이 9.7e-7).

대조군의 대역별 RMSE(z500, m²/s², l ≤ 6 / 7–15 / > 15):

| 리드 | stage 01 학습 단계 2 | persistence |
|---|---|---|
| 1 d | 234 / 472 / 234 | 228 / 474 / 239 |
| 3 d | 489 / 744 / 247 | 487 / 749 / 253 |
| 6 d | 621 / 800 / 248 | 620 / 805 / 253 |

오차의 대부분은 종관 대역(7~15)에 있고, stage 01 모델은 모든 대역에서 persistence와 거의 같습니다. 이 stage가 줄여야 할
것이 바로 이 종관 대역 오차입니다.

## 실행

```bash
pip install neuraloperator==2.0.0 torch-harmonics==0.8.0    # (b), (c)
bash scripts/run_stage04.sh                                  # 대조군 대역 채점, 세 갈래 학습, 평가, ClimODE 비교
```

실제 Qwen으로 확인한 비용(batch 2, CPU 1스레드, stage 03과 함께 실행): 한 step 9~11 s, 최대 메모리 약 5.3 GB.

## 결과

(학습 후 채웁니다.)
