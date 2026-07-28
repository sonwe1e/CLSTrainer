#!/usr/bin/env bash
set -euo pipefail
source /usr/local/Ascend/ascend-toolkit/set_env.sh

torchrun --standalone --nnodes=1 --nproc_per_node=8 \
  tools/train.py \
  --config configs/npu_8p.yaml \
  train.max_steps=500 \
  train.steps_per_epoch=500 \
  evaluation.quick_test_every_steps=100 \
  evaluation.full_test_every_steps=500 \
  checkpoint.save_last_every_steps=100 \
  experiment.output_dir=runs/npu_8p_smoke
