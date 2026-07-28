#!/usr/bin/env bash
set -euo pipefail
source /usr/local/Ascend/ascend-toolkit/set_env.sh

python tools/train.py \
  --config configs/npu_1p.yaml \
  train.max_steps=100 \
  train.steps_per_epoch=100 \
  evaluation.quick_test_every_steps=50 \
  evaluation.full_test_every_steps=100 \
  checkpoint.save_last_every_steps=50 \
  experiment.output_dir=runs/npu_1p_smoke
