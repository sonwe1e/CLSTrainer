#!/usr/bin/env bash
set -euo pipefail
CONFIG="${1:-configs/example_cpu.yaml}"
python -m clstrainer_lite.cli train --config "$CONFIG"
