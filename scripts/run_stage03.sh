#!/usr/bin/env bash
# Stage 03 (docs/stage03_lead_questions.md): lead questions in text vs a control with the same steps, all from the
# stage-01 checkpoint of training phase 2; 2020 forecasts scored with the ClimODE protocol; answers scored.
#   bash scripts/run_stage03.sh            (ARMS="control ask ask_textlead" by default)
set -euo pipefail
cd "$(dirname "$0")/.."
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4}
INIT=${INIT:-runs/stage01/stage2.pt}
REF=${REF:-../physical_guide_flowmatching/docs/results_2026_11/scores/raw/blend_ngcm_2020_climode_paired.json}
for arm in ${ARMS:-control ask ask_textlead}; do
    C=configs/stage03_$arm.yaml; O=runs/stage03/$arm
    python scripts/train.py --config "$C" --stage 3 --init "$INIT" --out "$O"
    python scripts/evaluate.py --config "$C" --ckpt "$O/stage3.pt" --out "$O/eval"
    python scripts/compare_climode.py --config "$C" --eval "$O/eval" --reference "$REF" --out "$O/eval/climode_compare"
    if [ "$arm" != control ]; then
        python scripts/stage03_answers.py --config "$C" --ckpt "$O/stage3.pt" --out "$O/answers" --n-ics 24
    fi
done
echo STAGE03_DONE
