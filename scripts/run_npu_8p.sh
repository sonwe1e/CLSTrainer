#!/usr/bin/env bash
set -euo pipefail

CONFIG="${1:-configs/example_npu_8p.yaml}"

# Load your Ascend environment before this script if your server does not do so globally, e.g.:
# source /usr/local/Ascend/ascend-toolkit/set_env.sh

# The package intentionally does not install/replace torch or torch_npu.
# Use the versions matched to your CANN stack.
torchrun --standalone --nnodes=1 --nproc_per_node=8 \
  -m clstrainer_lite.cli train --config "$CONFIG"
