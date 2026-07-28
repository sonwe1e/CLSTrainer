#!/usr/bin/env bash
set -euo pipefail
source /usr/local/Ascend/ascend-toolkit/set_env.sh
python tools/train.py --config configs/npu_1p.yaml

