#!/usr/bin/env bash
set -euo pipefail
python tools/train.py --config configs/cuda_debug.yaml train.max_steps=100

