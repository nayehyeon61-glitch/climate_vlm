#!/usr/bin/env bash
# Stages 03 -> 04 -> 05 in a row on one machine, each appending to runs/stageNN/pipeline.log. Every step is skipped
# when its output exists and training resumes from the last validation, so after an interruption (the cloud
# container can be stopped) the same command continues where it was.
#   nohup setsid bash scripts/run_stages_03_05.sh > /dev/null 2>&1 &
cd "$(dirname "$0")/.."
mkdir -p runs/stage03 runs/stage04 runs/stage05
bash scripts/run_stage03.sh >> runs/stage03/pipeline.log 2>&1 || echo "stage 03 stopped with an error" >> runs/stage03/pipeline.log
bash scripts/run_stage04.sh >> runs/stage04/pipeline.log 2>&1 && bash scripts/run_stage05.sh >> runs/stage05/pipeline.log 2>&1
echo QUEUE_DONE >> runs/stage05/pipeline.log
