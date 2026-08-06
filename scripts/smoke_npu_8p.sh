#!/usr/bin/env bash
set -euo pipefail
source /usr/local/Ascend/ascend-toolkit/set_env.sh

torchrun --standalone --nnodes=1 --nproc_per_node=8 \
  tools/train.py --run-mode fixed \
  --config configs/npu_8p.yaml \
  train.max_steps=100 \
  train.steps_per_epoch=100 \
  train.local_batch_size=8 \
  train.log_every_steps=10 \
  dataloader.train.num_workers=1 \
  dataloader.eval.num_workers=1 \
  evaluation.val_quick_every_steps=50 \
  evaluation.val_quick_pairs_per_video=2 \
  evaluation.val_full_every_steps=100 \
  evaluation.val_full_at_end=false \
  checkpoint.save_last_every_steps=100 \
  experiment.output_dir=runs/npu_8p_spawn_smoke
