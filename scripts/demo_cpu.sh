#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
python tools/make_demo_data.py --out demo_data
python -m clstrainer_lite.cli check --config configs/demo_cpu.yaml
python -m clstrainer_lite.cli train --config configs/demo_cpu.yaml
