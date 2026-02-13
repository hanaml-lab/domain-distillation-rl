#!/usr/bin/env bash
# Train GATE and the three baselines on the toy task (3 seeds each). Runs on CPU in well under a minute.
set -euo pipefail
cd "$(dirname "$0")/.."
PYTHONPATH=src python -m gate_distill.cli --config configs/toy.json "$@"
