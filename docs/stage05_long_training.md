# Stage 05: 학습량을 늘렸을 때 (5 epoch)

소유자 질문(2026-11-06): "아직 학습량이 적은 것 같은데 어떻게 생각하시나요?" 소유자는 긴 학습을 클라우드 CPU에서 돌리기로
정했습니다.

가설: stage 01~04의 결과가 persistence 근처에 머문 것은 주로 학습량(학습 단계 2가 0.6 epoch) 때문이다. 같은 모델을
5 epoch(약 8배) 학습하면 검증 손실이 계속 내려가고 2020 점수가 persistence보다 분명히 좋아진다. 그렇지 않으면 병목은
학습량보다 구조(동결된 영상 인코더, 토큰 수, 출력 방식) 쪽이다.

> 상태: 설정과 실행 스크립트 완료. stage 04가 끝나면 자동으로 시작합니다. 결과 절은 그때 채웁니다.

## 설정

- 갈래 두 개:
  - control: stage 01 모델(vision tower, 격자 직접 출력), `configs/stage05_long_control.yaml`.
  - stage 04에서 2016 검증 손실이 가장 낮은 갈래, `configs/stage05_long_<갈래>.yaml`. 2020 시험 연도는 고르는 데 쓰지
    않습니다(`scripts/run_stage05.sh`가 `runs/stage04/<갈래>/stage2_val.json`으로 고릅니다).
- 학습(`configs/stage05_long_train.yaml`): 학습 단계 2(P, F, G; D는 stage 01의 `runs/stage01/stage1.pt` 그대로),
  7,270 step × batch 4 = 5 epoch, lr 3e-4(warm-up 2 %, cosine), 진단 손실 0.1. 검증은 반 epoch마다 2016 전체(179개 창).
- 중단 대비: 검증 때마다 `stage2_resume.pt`(모델, optimizer, 일정, step)를 쓰고, `--resume`으로 그 지점부터 이어갑니다.
- 비용: 이 CPU에서 한 step 약 5 s, 한 갈래 약 10.5시간, 평가 포함 두 갈래 약 22시간.
- 채점: stage 01, 04와 같은 ClimODE 규약과 대역별 RMSE(`--bands 6,15`).

## 볼 것

1. 검증 손실 곡선: 0.6 epoch 지점(stage 01의 0.512)을 지나서도 계속 내려가는가.
2. 2020 점수: z500, t850 RMSE와 ACC가 persistence(z500 1 d 576)에서 얼마나 멀어지고 ClimODE(225)에 얼마나 다가가는가.
3. 대역별 RMSE: 줄어드는 오차가 종관 대역(7 ≤ l ≤ 15)인가.
4. 짧은 학습(stage 04)과 긴 학습에서 갈래 순위가 같은가.

## 실행

```bash
bash scripts/run_stage05.sh            # ARM=<stage 04 갈래>로 직접 고를 수도 있음
```

## 결과

(학습 후 채웁니다.)
