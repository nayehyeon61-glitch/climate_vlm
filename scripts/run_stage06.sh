#!/usr/bin/env bash
# Stage 06 (docs/stage06_laya_questions.md): typed questions about the forecast of stage 04's advection arm, answered
# by Laya's vega reader (zero-shot, calibrated, fitted heads) and by baselines. CPU, one thread for vega.
# Needs laya with vegaml (pip install -U laya) and the vega-U files (HF_HUB_DISABLE_XET=1 if the download stalls).
#   bash scripts/run_stage06.sh            # re-run after an interruption: finished steps are skipped, reads cached
set -euo pipefail
cd "$(dirname "$0")/.."
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}
C=configs/stage06_laya_questions.yaml
O=runs/stage06
step() { local out=$1; shift; if [ -e "$out" ]; then echo "done already: $out"; else "$@"; fi; }
for p in router calib test; do
    step "$O/fc_$p.npz" python scripts/stage02_forecasts.py --config "$C" --periods "$p" --batch 4
done
step "$O/rows.json" python scripts/stage06_questions.py --config "$C" rows
step "$O/pred_lr_all.json" python scripts/stage06_questions.py --config "$C" baselines
python scripts/stage06_questions.py --config "$C" report  # baselines alone first
step "$O/pred_laya_zero.json" python scripts/stage06_questions.py --config "$C" laya --arms zero
step "$O/pred_laya_cal.json" python scripts/stage06_questions.py --config "$C" laya --arms cal
step "$O/pred_laya_fit.json" python scripts/stage06_questions.py --config "$C" laya --arms fit
python scripts/stage06_questions.py --config "$C" report
echo STAGE06_DONE
