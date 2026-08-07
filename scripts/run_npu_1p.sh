#!/usr/bin/env bash
set -euo pipefail
source /usr/local/Ascend/ascend-toolkit/set_env.sh
python tools/train.py --config configs/recipes/game_cls_production.yaml profile=npu_1p experiment.output_dir=runs/game_cls_1p

