#!/usr/bin/env bash
set -euo pipefail
source /usr/local/Ascend/ascend-toolkit/set_env.sh
torchrun --standalone --nnodes=1 --nproc_per_node=8 \
  tools/train.py --config configs/recipes/game_cls_production.yaml

