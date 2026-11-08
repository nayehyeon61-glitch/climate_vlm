#!/usr/bin/env bash
# Stage 02 (configs/stage02_laya_router.yaml): Laya as an expert-selecting router on top of the frozen stage-01 model.
# Needs runs/stage01/stage2.pt, the stage-01 cache (2012-2016, 2019, 2020) and truth_5p625 for those years, and
# laya (pip install -e <laya clone>) with its checkpoint in the Hugging Face cache (HF_HOME).
#   bash scripts/run_stage02.sh
set -euo pipefail
cd "$(dirname "$0")/.."
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4}
C=${CONFIG:-configs/stage02_laya_router.yaml}
FROM=${FROM:-forecasts}   # forecasts | experts | dataset | laya | router: resume from that step
case $FROM in forecasts) python scripts/stage02_forecasts.py --config "$C" ;; esac
case $FROM in forecasts|experts) python scripts/stage02_experts.py --config "$C" ;; esac
case $FROM in forecasts|experts|dataset) python scripts/stage02_dataset.py --config "$C" ;; esac
if [ "$FROM" != router ]; then
python scripts/stage02_laya.py dry-run --config "$C"
python scripts/stage02_laya.py predict --config "$C" --model base   # zero-shot and the option states (own process)
python scripts/stage02_laya.py train --config "$C"                  # encoder frozen, split B
python scripts/stage02_laya.py predict --config "$C" --model tuned --splits C D
fi
python scripts/stage02_router.py --config "$C"
echo STAGE02_DONE
