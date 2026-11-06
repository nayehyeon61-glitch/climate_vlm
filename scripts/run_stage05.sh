#!/usr/bin/env bash
# Stage 05 (docs/stage05_long_training.md): the control and stage 04's best arm, trained for 5 epochs.
# The arm is the stage 04 arm with the lowest best 2016 validation loss (runs/stage04/<arm>/stage2_val.json), so the
# 2020 test year plays no part in the choice; ARM=<name> overrides it. Every run resumes if it was stopped.
#   bash scripts/run_stage05.sh
set -euo pipefail
cd "$(dirname "$0")/.."
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4}
REF=${REF:-../physical_guide_flowmatching/docs/results_2026_11/scores/raw/blend_ngcm_2020_climode_paired.json}
O=${OUT:-runs/stage05}
# skip a step whose output exists, so the script can simply be run again after an interruption
step() { local out=$1; shift; if [ -e "$out" ]; then echo "done already: $out"; else "$@"; fi; }
if [ -z "${ARM:-}" ]; then
    ARM=$(python -c 'import json, os
v = {a: json.load(open(f"runs/stage04/{a}/stage2_val.json"))["best"] for a in ("bands", "sfno", "sfno_advect")
     if os.path.exists(f"runs/stage04/{a}/stage2_val.json")}
print(min(v, key=v.get))')
fi
echo "stage 05 arms: control $ARM"
for arm in control "$ARM"; do
    C=configs/stage05_long_$arm.yaml
    step "$O/$arm/stage2_val.json" python scripts/train.py --config "$C" --stage 2 --init runs/stage01/stage1.pt \
        --out "$O/$arm" --resume
    step "$O/$arm/eval/scores.json" python scripts/evaluate.py --config "$C" --ckpt "$O/$arm/stage2.pt" --out "$O/$arm/eval"
    step "$O/$arm/eval/climode_compare/scores.json" python scripts/compare_climode.py --config "$C" --eval "$O/$arm/eval" \
        --reference "$REF" --bands 6,15 --out "$O/$arm/eval/climode_compare"
    rm -f "$O/$arm/eval/"*_all_*.npz  # 2.2 GB of fields: the scores are kept (disk allowance)
done
echo STAGE05_DONE
