#!/usr/bin/env bash
set -euo pipefail
source /usr/local/Ascend/ascend-toolkit/set_env.sh

unset RANK LOCAL_RANK WORLD_SIZE MASTER_ADDR MASTER_PORT
export ASCEND_RT_VISIBLE_DEVICES="${ASCEND_RT_VISIBLE_DEVICES:-0}"
unset ASCEND_LAUNCH_BLOCKING || true

# Establish that the model and NPU operators work without worker processes.
python -u tools/train.py \
  --config configs/npu_1p.yaml \
  train.max_steps=5 \
  train.steps_per_epoch=5 \
  train.local_batch_size=2 \
  train.log_every_steps=1 \
  dataloader.train.num_workers=0 \
  dataloader.eval.num_workers=0 \
  dataloader.train.pin_memory=false \
  dataloader.eval.pin_memory=false \
  augmentation.enabled=false \
  evaluation.quick_test_every_steps=0 \
  evaluation.full_test_every_steps=0 \
  evaluation.full_test_at_end=false \
  checkpoint.save_last_every_steps=0 \
  experiment.output_dir=runs/npu_single_process_baseline

# Start with one spawned worker before increasing worker concurrency.
python -u tools/train.py \
  --config configs/npu_1p.yaml \
  train.max_steps=10 \
  train.steps_per_epoch=10 \
  train.local_batch_size=2 \
  train.log_every_steps=1 \
  dataloader.train.num_workers=1 \
  dataloader.eval.num_workers=0 \
  augmentation.enabled=false \
  evaluation.quick_test_every_steps=0 \
  evaluation.full_test_every_steps=0 \
  evaluation.full_test_at_end=false \
  checkpoint.save_last_every_steps=0 \
  experiment.output_dir=runs/npu_spawn_1worker

# Validate spawn worker startup and the augmented training path.
python -u tools/train.py \
  --config configs/npu_1p.yaml \
  train.max_steps=50 \
  train.steps_per_epoch=50 \
  train.local_batch_size=8 \
  train.log_every_steps=5 \
  dataloader.train.num_workers=2 \
  dataloader.eval.num_workers=0 \
  augmentation.enabled=true \
  evaluation.quick_test_every_steps=0 \
  evaluation.full_test_every_steps=0 \
  evaluation.full_test_at_end=false \
  checkpoint.save_last_every_steps=0 \
  experiment.output_dir=runs/npu_spawn_2workers

# Add quick evaluation with non-persistent eval workers.
python -u tools/train.py \
  --config configs/npu_1p.yaml \
  train.max_steps=20 \
  train.steps_per_epoch=20 \
  train.local_batch_size=8 \
  train.log_every_steps=5 \
  dataloader.train.num_workers=2 \
  dataloader.eval.num_workers=1 \
  evaluation.quick_test_every_steps=10 \
  evaluation.quick_test_pairs_per_video=2 \
  evaluation.full_test_every_steps=0 \
  evaluation.full_test_at_end=false \
  checkpoint.save_last_every_steps=0 \
  experiment.output_dir=runs/npu_spawn_quick_eval

# Run full evaluation only after the preceding stages have passed.
python -u tools/train.py \
  --config configs/npu_1p.yaml \
  train.max_steps=20 \
  train.steps_per_epoch=20 \
  train.local_batch_size=8 \
  train.log_every_steps=5 \
  dataloader.train.num_workers=2 \
  dataloader.eval.num_workers=1 \
  evaluation.quick_test_every_steps=0 \
  evaluation.full_test_every_steps=20 \
  evaluation.full_test_at_end=false \
  checkpoint.save_last_every_steps=20 \
  experiment.output_dir=runs/npu_spawn_full_eval
