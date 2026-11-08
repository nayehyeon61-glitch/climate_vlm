# Stage 01: CPU 작은 규모 학습과 ClimODE 비교 (WeatherBench2, 2020)

소유자 요청(2026-11-06): small scale로 WeatherBench2 규약에 따라 ClimODE와 비교하고, 클라우드 환경에서 학습부터 시험까지
작게 한 번 돌린다.

## 설정

- 자료: WeatherBench2 ERA5 128×64(2.8125°)를 캐시로 만들고(`scripts/wb2_to_cache.py`), 모델 입력은 그것을 5.625°로 줄인
  64×32입니다. WB2 128×64 격자는 0°에 중심이 있어 경도 방향은 1/4, 1/2, 1/4 가중으로 줄입니다(`data.coarsen`, lon0).
- 기간: 학습 2012–2015, 검증 2016, 시험 2020. 2019년은 2020년 첫 IC들의 과거 입력으로만 씁니다.
- 모델: Qwen3.5-0.8B 실제 가중치(동결, 학습 단계 3에서 LoRA), CorrDiff 회귀 UNet(32 채널, [1, 2, 2]), 이미지 토큰 64개,
  짧은 프롬프트(87 토큰), Perceiver IO decoder(dim 256, patch 8, Gaussian 출력).
- 학습량: 학습 단계 1 / 2 / 3 = 800 / 900 / 300 step(batch 8 / 4 / 4). CPU 4코어에서 약 40 / 78 / 44분.
  검증 손실은 0.0231 / 0.5120 / 0.5268입니다.
- 채점: ClimODE 규약입니다(`scripts/compare_climode.py`).
  - IC 179개: 2020년 12 UTC 이틀 간격이고, 970번은 제외합니다.
  - 정답은 WB2 64×32 conservative입니다.
  - RMSE와 ACC는 ClimODE 정의를 따르고, CRPS는 2006–2018 min-max 단위의 Gaussian CRPS입니다.
  - ClimODE와 pg-flow 행은 pg-flow의 2020 비교 JSON에서 가져왔습니다.
  - 우리 persistence는 그 JSON의 persistence와 최대 상대 차이 9.7e-7로 일치합니다.

## 결과 (학습 단계 2 checkpoint, RMSE, 1 d / 3 d / 6 d)

| 변수 | climate_vlm | persistence | ClimODE | pg-flow (blend) |
|---|---|---|---|---|
| z500 (m²/s²) | 574 / 919 / 1040 | 576 / 924 / 1040 | 225 / 509 / 874 | 144 / 353 / 608 |
| t850 (K) | 2.68 / 3.87 / 4.20 | 2.71 / 3.99 / 4.33 | 1.32 / 2.46 / 3.94 | 1.35 / 2.01 / 2.78 |
| t2m (K) | 2.22 / 2.95 / 3.27 | 1.74 / 2.65 / 3.01 | 2.73 / 4.41 / 5.87 | 1.17 / 1.68 / 2.22 |
| u10 (m/s) | 3.60 / 3.59 / 3.58 | 3.31 / 4.37 / 4.64 | 1.55 / 2.74 / 3.90 | 1.58 / 2.42 / 3.17 |
| v10 (m/s) | 3.76 / 3.76 / 3.75 | 4.06 / 4.94 / 5.02 | 1.60 / 2.98 / 4.46 | 1.69 / 2.56 / 3.36 |

z500 ACC는 0.837 / 0.577 / 0.464로 persistence와 소수 셋째 자리까지 같습니다. 전체 표(RMSE, ACC, CRPS, 모든 리드)는
[docs/stage01/table_phase2.md](stage01/table_phase2.md)(학습 단계 2)와
[docs/stage01/table_phase3.md](stage01/table_phase3.md)(학습 단계 3)에 있습니다.

## 해석

- 이 학습량에서는 예측 head가 persistence 기반 위에 거의 아무것도 더하지 못했습니다.
  - z500은 persistence와 같습니다.
  - t850만 조금 낫습니다(1 d 2.68 대 2.71, 6 d 4.20 대 4.33).
  - t2m은 persistence보다 나쁩니다(1 d 2.22 대 1.74).
- u10과 v10은 입력에 없는 출력 전용 채널이라 기반값이 0입니다. 모든 리드에서 RMSE가 같고 ACC가 0.1–0.19라서, 기후값에
  가까운 평균장을 내고 있습니다. 긴 리드에서 persistence보다 RMSE가 낮은 것은 그 때문이고, 예측 기술은 아닙니다.
- ClimODE보다 z500, t850, u10, v10 모두 크게 나쁩니다(z500 1 d RMSE 574 대 225).
  - t2m은 1 d와 3–6 d에서 RMSE가 ClimODE보다 낮습니다. 다만 ClimODE의 t2m이 리드에 따라 2.3–6.3 K로 흔들리기 때문이고,
    우리 t2m이 persistence보다도 나쁘므로 의미 있는 우위로 보지 않습니다.
- 학습 단계 3(LoRA, 300 step)은 검증 손실이 0.512에서 0.527로 나빠졌고, t850 1 d RMSE도 2.68에서 3.06이 되었습니다.
  그래서 stage 02의 기반 예측기는 학습 단계 2 checkpoint로 정했습니다.
- 원인은 학습량으로 봅니다(검증하지 않은 추정). 학습 단계 2는 batch 4로 900 step이라 5,816개 학습 창의 0.6 epoch밖에
  보지 못했습니다.

결론: 이 stage는 자료 준비, 학습 세 단계, 2020 예측, ClimODE 규약 채점이 끝까지 돌아간다는 것과 채점이 pg-flow 결과와
맞는다는 것을 확인했습니다. 성능 비교로서는 CPU 학습량이 너무 작아 결론을 낼 수 없습니다. ClimODE와 겨룰 수 있는지는
GPU에서 학습량을 늘려 다시 봐야 합니다.

## 재현

```bash
python scripts/wb2_to_cache.py --config configs/stage01_cpu_small.yaml   # WB2 128x64 캐시와 64x32 정답
python scripts/prepare_data.py --config configs/stage01_cpu_small.yaml   # 학습 기간 통계, 월별 평년값
bash scripts/run_stage01.sh                                              # 학습 단계 1 → 2 → 3, 2020 예측, 비교 표
```

산출물: `runs/stage01/`(checkpoint, 예측, 로그; 저장소에는 넣지 않음), `docs/stage01/`(점수 JSON, 표, 검증 손실 기록).
