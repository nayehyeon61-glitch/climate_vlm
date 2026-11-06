#!/usr/bin/env bash
# Stage 04 (docs/stage04_neural_operator.md): spherical-harmonic bands before the encoder, SFNO encoder, band
# recombination or ClimODE-style advection at F. Training phase 2 from the stage-01 downscaler with stage 01's
# phase-2 budget; that run (runs/stage01/eval_stage2) is the control and is scored here with the same band split.
# Needs neuraloperator and torch-harmonics (pip install neuraloperator torch-harmonics) for the SFNO arms.
#   bash scripts/run_stage04.sh            # ARMS="bands sfno sfno_advect" by default
set -euo pipefail
cd "$(dirname "$0")/.."
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4}
REF=${REF:-../physical_guide_flowmatching/docs/results_2026_11/scores/raw/blend_ngcm_2020_climode_paired.json}
O=${OUT:-runs/stage04}
B=${BANDS:-6,15}
# skip a step whose output exists, so the script can simply be run again after an interruption
step() { local out=$1; shift; if [ -e "$out" ]; then echo "done already: $out"; else "$@"; fi; }
step "$O/control_stage01/climode_compare/scores.json" python scripts/compare_climode.py \
    --config configs/stage01_cpu_small.yaml --eval runs/stage01/eval_stage2 --reference "$REF" --bands "$B" \
    --out "$O/control_stage01/climode_compare"
for arm in ${ARMS:-bands sfno sfno_advect}; do
    C=configs/stage04_$arm.yaml
    step "$O/$arm/stage2_val.json" python scripts/train.py --config "$C" --stage 2 --init runs/stage01/stage1.pt \
        --out "$O/$arm" --resume
    step "$O/$arm/eval/scores.json" python scripts/evaluate.py --config "$C" --ckpt "$O/$arm/stage2.pt" --out "$O/$arm/eval"
    step "$O/$arm/eval/climode_compare/scores.json" python scripts/compare_climode.py --config "$C" --eval "$O/$arm/eval" \
        --reference "$REF" --bands "$B" --out "$O/$arm/eval/climode_compare"
    rm -f "$O/$arm/eval/"*_all_*.npz  # 2.2 GB of fields: the scores are kept (disk allowance)
done
echo STAGE04_DONE
