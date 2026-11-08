# Stage 00: 첫 구현 (구조 검사)

소유자 요청(2026-11-06): 연구 계획서 「기후 VLM 미래 예측 연구 통합 설계」를 토대로 climate VLM 모델을 작성하고,
다운스케일링은 physicsnemo 저장소를 따른다. 입력·출력은 pg-flow(physical_guide_flowmatching)와 같게 한다.

## 구조

```
pg-flow 5.625° 과거장 → D: physicsnemo CorrDiff (회귀 UNet [+ EDM 잔차]) → 2.8125° Mean / Tendency / (Spread)
 → E: Qwen3.5 vision tower(동결) → P: Perceiver resampler (+ 수치 토큰)
 → Qwen3.5-0.8B (+ LoRA, 학습 단계 3) → G: 물리 진단 head → F: Perceiver IO 수치 decoder → 미래장
```

학습 단계: 1 (D_μ), 1res (D_res, 선택), 2 (P, G, F; LLM 동결이지만 gradient는 통과), 3 (+ LoRA).
손실은 L_f + λ_d L_diag + λ_p L_phys + λ_c L_cons + λ_t L_text입니다. 자세한 설명은 README의 모듈 표와
"계획서에서 정한 것을 이렇게 구현했습니다" 절에 있습니다.

## 검증

- 합성 자료(pg-flow 형식)와 임의 가중치의 작은 Qwen3.5로 학습 단계 1, 1res, 2, 3과 평가 전체 경로를 확인했습니다.
- patch 순서가 Qwen 이미지 프로세서와 같고, 설명 텍스트가 수치 예측으로 새지 않으며, 2단계 gradient가 동결 LLM을 지나
  resampler까지 가는 것을 테스트로 확인했습니다.
- 실제 체크포인트와 ERA5 성능은 이 stage에서 확인하지 않았습니다(Hugging Face 접속 차단). 이것은 stage 01에서 다룹니다.

코드: PR #1의 첫 커밋(`bc70c2d`).
