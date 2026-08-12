#!/usr/bin/env bash
set -euo pipefail
CONFIG="${1:-configs/example_cpu.yaml}"
torchrun --standalone --nnodes=1 --nproc_per_node=2 \
  -m clstrainer_lite.cli train --config "$CONFIG" runtime.accelerator=cpu runtime.backend=gloo
