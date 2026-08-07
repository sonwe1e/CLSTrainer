#!/usr/bin/env bash
set -euo pipefail
python tools/train.py --config configs/recipes/example_debug.yaml profile=cuda_1p train.max_steps=100

