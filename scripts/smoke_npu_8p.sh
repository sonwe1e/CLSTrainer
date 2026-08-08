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
#     bash scripts/smoke_npu_8p.sh
CONFIG="${CONFIG:-configs/recipes/game_cls_production.yaml}"
PROFILE="${PROFILE:-npu_8p}"
# Word-split on purpose: EXTRA_OVERRIDES carries several key=value pairs.
read -r -a EXTRA <<<"${EXTRA_OVERRIDES:-}"

torchrun --standalone --nnodes=1 --nproc_per_node=8 \
  tools/train.py --run-mode fixed \
  --config "$CONFIG" "profile=$PROFILE" "${EXTRA[@]}" \
  experiment.smoke_mode=true \
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
