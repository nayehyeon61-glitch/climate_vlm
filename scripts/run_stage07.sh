#!/usr/bin/env bash
# Stage 07 (docs/stage07_flow_ensemble.md): flow-matching ensemble generator after the frozen VLM's decoder.
# Cache the decoder memory / mean, train each arm's expert, draw 8 members per 2020 IC, score against the Gaussian
# head, persistence and ClimODE, then the paired differences and the members' structure (scripts/stage07_analysis.py).
# Every step resumes after an interruption (cache by batch, training by validation).
#   bash scripts/run_stage07.sh                                              # pilot: stage 04 advection arm
#   CONFIG=configs/stage07_flow_ensemble_s05.yaml bash scripts/run_stage07.sh  # stage 05's advection arm
set -euo pipefail
cd "$(dirname "$0")/.."
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}
C=${CONFIG:-configs/stage07_flow_ensemble.yaml}
O=$(python -c "import yaml; print(yaml.safe_load(open('$C'))['out'])" 2>/dev/null || true)
[ -n "$O" ] || O=$(python -c "from climate_vlm import build; print(build.load_config('$C')['out'])")
step() { local out=$1; shift; if [ -e "$out" ]; then echo "done already: $out"; else "$@"; fi; }
for s in train val test; do
    python scripts/stage07_cache.py --config "$C" --split "$s"
done
# the config's run_arms if it has them (the main run trains a subset), else every arm
A=$(python -c "from climate_vlm import build; c = build.load_config('$C'); print(' '.join(c.get('run_arms') or c['arms']))" | tail -1)
for arm in ${ARMS:-$A}; do
    step "$O/$arm/flow.pt" python scripts/stage07_flow.py --config "$C" train --arm "$arm"
    step "$O/$arm/members_2020.npz" python scripts/stage07_flow.py --config "$C" sample --arm "$arm"
done
python scripts/stage07_flow.py --config "$C" score
python scripts/stage07_analysis.py --config "$C"
echo STAGE07_DONE
