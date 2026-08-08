#!/usr/bin/env bash
set -euo pipefail
source /usr/local/Ascend/ascend-toolkit/set_env.sh

# Audit P0-10: the config and its placeholder fields are overridable, so a
# provisioned runner can point this at a real recipe/factory/checkpoint without
# editing the script. Defaults reproduce the previous hard-coded behaviour, so
# existing local invocations keep working unchanged.
#
#   CONFIG=configs/recipes/my_prod.yaml \
#   EXTRA_OVERRIDES="model.factory=pkg.mod:build model.checkpoint_path=/w.pt" \
#     bash scripts/smoke_npu_1p.sh
CONFIG="${CONFIG:-configs/recipes/game_cls_production.yaml}"
PROFILE="${PROFILE:-npu_1p}"
# Word-split on purpose: EXTRA_OVERRIDES carries several key=value pairs.
read -r -a EXTRA <<<"${EXTRA_OVERRIDES:-}"

unset RANK LOCAL_RANK WORLD_SIZE MASTER_ADDR MASTER_PORT
export ASCEND_RT_VISIBLE_DEVICES="${ASCEND_RT_VISIBLE_DEVICES:-0}"
unset ASCEND_LAUNCH_BLOCKING || true

# Establish that the model and NPU operators work without worker processes.
python -u tools/train.py --run-mode fixed \
  --config "$CONFIG" "profile=$PROFILE" "${EXTRA[@]}" \
  experiment.smoke_mode=true \
  train.max_steps=5 \
  train.steps_per_epoch=5 \
  train.local_batch_size=2 \
  train.log_every_steps=1 \
  dataloader.train.num_workers=0 \
  dataloader.eval.num_workers=0 \
  dataloader.train.pin_memory=false \
  dataloader.eval.pin_memory=false \
  augmentation.enabled=false \
  evaluation.val_quick_every_steps=0 \
  evaluation.val_full_every_steps=0 \
  evaluation.val_full_at_end=false \
  checkpoint.save_last_every_steps=0 \
  experiment.output_dir=runs/npu_single_process_baseline

# Start with one spawned worker before increasing worker concurrency.
python -u tools/train.py --run-mode fixed \
  --config "$CONFIG" "profile=$PROFILE" "${EXTRA[@]}" \
  experiment.smoke_mode=true \
  train.max_steps=10 \
  train.steps_per_epoch=10 \
  train.local_batch_size=2 \
  train.log_every_steps=1 \
  dataloader.train.num_workers=1 \
  dataloader.eval.num_workers=0 \
  augmentation.enabled=false \
  evaluation.val_quick_every_steps=0 \
  evaluation.val_full_every_steps=0 \
  evaluation.val_full_at_end=false \
  checkpoint.save_last_every_steps=0 \
  experiment.output_dir=runs/npu_spawn_1worker

# Validate spawn worker startup and the augmented training path.
python -u tools/train.py --run-mode fixed \
  --config "$CONFIG" "profile=$PROFILE" "${EXTRA[@]}" \
  experiment.smoke_mode=true \
  train.max_steps=50 \
  train.steps_per_epoch=50 \
  train.local_batch_size=8 \
  train.log_every_steps=5 \
  dataloader.train.num_workers=2 \
  dataloader.eval.num_workers=0 \
  augmentation.enabled=true \
  evaluation.val_quick_every_steps=0 \
  evaluation.val_full_every_steps=0 \
  evaluation.val_full_at_end=false \
  checkpoint.save_last_every_steps=0 \
  experiment.output_dir=runs/npu_spawn_2workers

# Add quick evaluation with non-persistent eval workers.
python -u tools/train.py --run-mode fixed \
  --config "$CONFIG" "profile=$PROFILE" "${EXTRA[@]}" \
  experiment.smoke_mode=true \
  train.max_steps=20 \
  train.steps_per_epoch=20 \
  train.local_batch_size=8 \
  train.log_every_steps=5 \
  dataloader.train.num_workers=2 \
  dataloader.eval.num_workers=1 \
  evaluation.val_quick_every_steps=10 \
  evaluation.val_quick_pairs_per_video=2 \
  evaluation.val_full_every_steps=0 \
  evaluation.val_full_at_end=false \
  checkpoint.save_last_every_steps=0 \
  experiment.output_dir=runs/npu_spawn_quick_eval

# Run full evaluation only after the preceding stages have passed.
python -u tools/train.py --run-mode fixed \
  --config "$CONFIG" "profile=$PROFILE" "${EXTRA[@]}" \
  experiment.smoke_mode=true \
  train.max_steps=20 \
  train.steps_per_epoch=20 \
  train.local_batch_size=8 \
  train.log_every_steps=5 \
  dataloader.train.num_workers=2 \
  dataloader.eval.num_workers=1 \
  evaluation.val_quick_every_steps=0 \
  evaluation.val_full_every_steps=20 \
  evaluation.val_full_at_end=false \
  checkpoint.save_last_every_steps=20 \
  experiment.output_dir=runs/npu_spawn_full_eval
