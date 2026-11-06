#!/usr/bin/env bash
# Small-scale run (configs/stage01_cpu_small.yaml): stage 1 -> 2 -> 3, 2020 forecasts, ClimODE-protocol scores.
# Needs the cache from scripts/wb2_to_cache.py + scripts/prepare_data.py and pg-flow next to this repository
# (REF: its 2020 ClimODE comparison JSON). Stage 2 is scored too, to see what LoRA adds.
#   OUT=runs/stage01 bash scripts/run_stage01.sh
set -euo pipefail
cd "$(dirname "$0")/.."
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4}
C=${CONFIG:-configs/stage01_cpu_small.yaml}; O=${OUT:-runs/stage01}
REF=${REF:-../physical_guide_flowmatching/docs/results_2026_11/scores/raw/blend_ngcm_2020_climode_paired.json}
python scripts/train.py --config "$C" --stage 1 --out "$O"
python scripts/train.py --config "$C" --stage 2 --init "$O/stage1.pt" --out "$O"
python scripts/train.py --config "$C" --stage 3 --init "$O/stage2.pt" --out "$O"
for s in 3 2; do
    python scripts/evaluate.py --config "$C" --ckpt "$O/stage$s.pt" --out "$O/eval_stage$s"
    python scripts/compare_climode.py --config "$C" --eval "$O/eval_stage$s" --reference "$REF" \
        --out "$O/eval_stage$s/climode_compare"
done
echo PIPELINE_DONE
