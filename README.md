# climate_vlm

연구 계획서 「기후 VLM 미래 예측 연구 통합 설계」(2026.11.06)의 모델 구현입니다.
Downscaling → 사전학습 Vision Encoder 토큰화 → Qwen3.5-0.8B(LoRA) → 물리 진단 head와 수치 decoder로
미래 기상장(24·48·72·120 h)을 예측합니다. 입력과 출력은 `physical_guide_flowmatching`(pg-flow)과 같은 형식입니다.

```
과거 기상장 x_lr (t-72 … t, pg-flow 5.625°)
  │ ① D: physicsnemo CorrDiff (회귀 UNet D_mu [+ EDM 잔차 diffusion D_res])
  ▼
HR 지도: LR 보간 / Mean / Tendency(/Δt) [/ Spread]           (2.8125°, 64×128)
  │ ② E: Qwen3.5 자체 vision tower(동결) → P: Perceiver resampler (+수치 토큰)
  ▼
[메타데이터 프롬프트 | 기상 토큰 256 | 예측 query 16 | (설명 텍스트)]
  │ ③ Qwen3.5-0.8B (Gated DeltaNet + attention hybrid) + LoRA
  ▼
hidden(기상 토큰, query) ─► G: 물리 진단 head ─► ẑ_t, ẑ_{t+h}, Z_phys
                        └► F: q = E_lead(h) + E_geo(lat, lon), CrossAttn(q, [H, Z_phys]) → MLP → unpatchify → CNN
                              → Ŷ (B, 4, C, 64, 128)  (+ D_mu(x_t) 기준장, 선택적 Gaussian scale)
```

## 입력과 출력 (pg-flow 호환)

| | 형식 |
|---|---|
| 입력 | pg-flow `scripts/local_era5_to_kai.py --grid 2.8125 --out data/era5_2p8125`의 결과: `kai6h/<year>.nc` (`fields (time, 67, lat, lon)`, KAI 채널, 6시간, ERA5 단위)와 `era5_data/`(u10, v10). 모델 입력 LR은 HR을 면적 가중 평균(factor 2)한 것이라 pg-flow의 5.625° 상태와 정확히 같습니다. |
| 출력 | `<tag>_ens_<year>.npz`: pg-flow member 형식 (`E_minus_x0 (n_ic, M, L, 5, H, W)`, `O`, `X0`, `ics`, `leads`), ClimODE 5변수(z500, t850, t2m, u10, v10). pg-flow의 `score_step13.py`, `compare_step13_climode.py`로 바로 채점하고, `coarsen_ens.py`로 5.625°에서 채점할 수 있습니다. 모든 출력 채널은 `<tag>_all_<year>.npz`에 있습니다. |
| 시험 IC | pg-flow/ClimODE와 같은 6 h index 2 + 8e (12 UTC, 이틀 간격). 과거 이력이 같은 split의 이전 연도로 넘어가는 것은 허용합니다. |

## 설치와 실행

```bash
pip install -r requirements.txt            # physicsnemo는 pip install -e <physicsnemo clone> 도 됩니다
# 0. pg-flow 쪽에서 2.8125° 자료 생성 (예: lustre 0.25° 원자료)
#    python scripts/local_era5_to_kai.py --src $SRC --grid 2.8125 --out data/era5_2p8125 --years $(seq 1979 2018)
C=configs/qwen35_0p8b.yaml
python scripts/prepare_data.py --config $C                                   # 캐시 + train 기간 통계
python scripts/train.py --config $C --stage 1 --out runs/v1                  # ① D_mu
python scripts/train.py --config $C --stage 1res --out runs/v1 --init runs/v1/stage1.pt   # (선택) D_res
python scripts/train.py --config $C --stage 2 --out runs/v1 --init runs/v1/stage1.pt      # ② P, F, G
python scripts/train.py --config $C --stage 3 --out runs/v1 --init runs/v1/stage2.pt      # ③ + LoRA
python scripts/evaluate.py --config $C --ckpt runs/v1/stage3.pt --out runs/v1/eval
```

CPU 확인용: `python scripts/make_synthetic.py --out data/synthetic` 뒤 `configs/tiny_synthetic.yaml`로 같은 명령을
실행합니다(임의 가중치의 작은 Qwen3.5, 숫자는 의미 없음). 테스트는 `python -m pytest tests`입니다.

## Stage 기록

가설이나 모델이 크게 바뀌면 stage를 나눕니다. 각 stage는 `docs/stageNN_<이름>.md`(목적, 설정, 결과, 해석),
`docs/stageNN/`(점수와 표), `configs/stageNN_<이름>.yaml`, `runs/stageNN/`(저장소 밖)을 가지며, 이전 stage는 덮어쓰지 않습니다.

| stage | 문서 | 내용 |
|---|---|---|
| 00 | [docs/stage00_initial_implementation.md](docs/stage00_initial_implementation.md) | 첫 구현, 합성 자료로 구조 검사 |
| 01 | [docs/stage01_cpu_small_climode.md](docs/stage01_cpu_small_climode.md) | CPU 작은 규모 학습, 2020년 ClimODE 규약 비교 |
| 02 | [docs/stage02_laya_router.md](docs/stage02_laya_router.md) | Laya를 기후 전문가 선택 과제에 학습시켜 라우터로 시험 |
| 03 | [docs/stage03_lead_questions.md](docs/stage03_lead_questions.md) | 프롬프트에서 리드(24/72/96/120 h)를 묻고 물리 상태를 답하게 학습: 수치 예측 변화 없음, 답은 형식만 학습 |
| 04 | [docs/stage04_neural_operator.md](docs/stage04_neural_operator.md) | 구면조화 대역 분해와 SFNO(neuraloperator), 대역 결합과 ClimODE식 이류 출력: 분해는 효과 없음, 이류 출력이 z500 3 d RMSE 919에서 748로 |
| 05 | [docs/stage05_long_training.md](docs/stage05_long_training.md) | 대조군과 stage 04 이류 갈래를 5 epoch 학습: 이류 갈래 z500 RMSE 475 / 728 / 800(1 / 3 / 6 d)으로 6 d에서 ClimODE(874)를 앞서지만 1~3 d는 크게 뒤짐, 대조군은 여전히 persistence, 둘 다 2.5~3.5 epoch에서 평탄 |
| 06 | [docs/stage06_laya_questions.md](docs/stage06_laya_questions.md) | 예측 뒤에서 Laya가 typed 질문(T850 하강, Z500 변화, 가장 식는 사분면)에 답하기: head + 보정 정확도 0.55로 같은 숫자의 로지스틱 0.67보다 낮음, zero-shot은 한 답으로 붕괴 |
| 07 | [docs/stage07_flow_ensemble.md](docs/stage07_flow_ensemble.md) | 디코더 뒤 flow matching 앙상블 생성기(VLA의 action expert 자리): 시험 실행에서 CRPS는 가우시안 폭보다 낮지만 VLM 메모리의 몫은 없음, 구조 잡음으로 멤버가 리드·공간에 걸쳐 이어짐 (본 실행 보류) |

## 작은 규모 실행과 ClimODE 비교 (WeatherBench2, CPU)

pg-flow의 2020 hold-out 비교(`physical_guide_flowmatching/docs/results_2026_11`)와 같은 IC 179개(12 UTC, 2일 간격),
같은 정답(WB2 ERA5 64×32 conservative), 같은 점수식(ClimODE의 RMSE, ACC, min–max 단위 Gaussian CRPS), lead 24~144 h로
채점합니다. pg-flow의 KAI 파일 없이 공개 WB2 zarr만으로 돌아가서 클라우드 CPU에서도 됩니다.

```bash
C=configs/stage01_cpu_small.yaml       # 학습 2012–2015, 검증 2016, 시험 2020 (2019는 첫 IC의 과거 입력으로만)
python scripts/wb2_to_cache.py --config $C      # WB2 128x64 → 캐시, 64x32 채점 정답 2006–2018·2020, ClimODE min–max
python scripts/prepare_data.py --config $C      # train 기간 통계
bash scripts/run_stage01.sh                   # 1 → 2 → 3단계, 2020 예측, ClimODE 비교 표
```

- `scripts/compare_climode.py`가 2.8125° 예측을 64×32로 평균해(가우시안 scale도 평균: 한 5.625° 칸 안의 오차가 완전 상관이라는
  가정) 채점하고, pg-flow의 결과 JSON에서 ClimODE와 pg-flow 모델의 평균을 가져와 표(`table.md`)를 만듭니다.
  지속성 예보를 다시 계산해 그 JSON과 비교하므로(상대 차이 1e-6 이하) 정답·IC·점수식이 같다는 것이 확인됩니다.
- ClimODE의 IC별 예측(`--climode-pred`, 서버의 `climode_pred_2020_valid.npz`)을 주면 짝 차이의 95 % 부트스트랩 구간도 냅니다.
  주지 않으면 같은 IC에서의 평균끼리 비교입니다.
- WB2 128×64 격자는 0°에 칸 중심이 있어 2.8125° 칸이 5.625° 칸을 정확히 나누지 않습니다. 이 경우 경도 방향 1/4, 1/2, 1/4
  가중 평균(`data.coarsen(..., lon0=True)`)과 그에 맞춘 보간(`downscaler.interp`)을 씁니다. 서버의 pg-flow `--grid 2.8125`
  자료는 칸이 정확히 나뉘므로 이 근사가 필요 없습니다.

## 모듈

| 파일 | 계획서 | 내용 |
|---|---|---|
| `climate_vlm/contract.py` | — | pg-flow KAI 채널·단위·상수, 격자 가중치 |
| `climate_vlm/data.py` | p.9 | KAI 파일 → 캐시, 과거·미래 window (split 안에서만), 정확한 면적 평균 coarsening |
| `climate_vlm/downscaler.py` | p.3 ① | physicsnemo `CorrDiffRegressionUNet` / `EDMPrecondSuperResolution` / `RegressionLoss` / `ResidualLoss` / `stochastic_sampler`. 전구 격자라 경도 순환 padding(`LonPeriodic`)만 추가 |
| `climate_vlm/vision.py` | p.4 ② | 고정 train 통계의 색 변환(이미지별 min/max 없음, clipping 비율 기록), Qwen 이미지 프로세서와 같은 patch 순서, resampler P, 수치 보조 토큰 |
| `climate_vlm/llm.py` | p.10 ③ | Qwen3.5 로딩, LoRA(peft, LLM에만), 메타데이터 프롬프트, 진단 기반 설명 템플릿 |
| `climate_vlm/decoder.py` | p.5 | G(물리 진단 head)와 F(Perceiver IO 수치 decoder) |
| `climate_vlm/physics.py` | p.6 | 구면 격자 미분, 진단(850 hPa 온도·수증기 이류, 수분 flux 수렴, 500–850 hPa 층후), 정역학 residual, q ≥ 0 |
| `climate_vlm/losses.py` | p.7 | L_forecast(면적 가중 MSE 또는 CRPS) + λ_d L_diag + λ_p L_phys + λ_c L_cons + λ_t L_text |
| `climate_vlm/metrics.py`, `io.py` | p.9 | RMSE / ACC / CRPS (pg-flow 정의), pg-flow 형식 출력 |
| `scripts/` | | `make_synthetic`, `prepare_data`, `train`(단계 1, 1res, 2, 3), `evaluate`, `wb2_to_cache`(공개 WB2 자료), `compare_climode`(ClimODE 규약 채점) |

## 계획서에서 정한 것을 이렇게 구현했습니다

- **단계별 동결과 gradient 경로**: 2단계는 D, E, LLM 가중치를 동결하지만 LLM forward를 `no_grad`로 감싸지 않아
  손실이 LLM을 거쳐 P까지 갑니다(`tests/test_model.py`에서 resampler query의 gradient로 확인). 3단계는 LLM 안에서
  LoRA만 학습합니다. 모델 선택은 validation의 미래 예측 오차로 합니다.
- **정보 누출 방지**: 설명 텍스트는 query 토큰 뒤에 있어서, G와 F가 읽는 hidden state는 teacher-forcing된 텍스트를
  보지 못합니다(텍스트를 바꿔도 예측이 같은지 테스트). 미래 진단은 손실의 정답으로만 씁니다.
- **Tendency**는 결정론적 평균 D_mu의 차분을 실제 Δt(일)로 나눈 값이고, Spread는 D_res 앙상블의 위치별 표준편차입니다.
- **물리 손실**: 정역학 관계는 출력 기압면(850–700, 700–500 hPa)의 층 residual로, ERA5 자체의 이산화 residual을 넘는
  부분만 벌점을 주는 약한 제약입니다. 수증기 식 R_q는 ω와 원천항 S_q가 자료에 없어 residual로 쓰지 않고(계획서의
  "우변을 무조건 0으로 두지 않는다"), 이류·수렴을 진단 감독으로만 씁니다.
- **손실 순서**: 2단계는 forecast + 진단, 3단계에서 consistency와 작은 가중치의 물리 residual을 더합니다(설정 파일).

## 검증 상태

- 확인한 것: 합성 자료(pg-flow 형식)와 임의 가중치의 작은 Qwen3.5(실제 hybrid 구조)로 1·1res·2·3단계 학습과
  평가 전체 경로, pg-flow 형식 출력, 단위 테스트(`tests/`).
- 확인하지 못한 것: 공개 Qwen3.5-0.8B 체크포인트 로딩(이 작업 환경에서 Hugging Face 접속이 막혀 있었습니다), 실제
  ERA5 성능, GPU 메모리. 계획서 표현대로 "구조 검사"이지 성능 검증이 아닙니다.
- 아직 없는 것: 경로 B(Prithvi WxC 등 기상 encoder), 텍스트 생성 추론, 극한현상 점수·bootstrap, H1~H6 비교 실험 스크립트.
