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
| `scripts/` | | `make_synthetic`, `prepare_data`, `train`(단계 1, 1res, 2, 3), `evaluate` |

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
