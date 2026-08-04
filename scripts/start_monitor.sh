#!/usr/bin/env bash
# Runs the live performance watchdog that auto-switches the champion model
# if its real Sharpe ratio on fresh data drops below the configured threshold.
set -e

cd "$(dirname "$0")/.."
source venv/bin/activate 2>/dev/null || true

export PYTHONPATH="$(pwd):$PYTHONPATH"

python src/monitoring/monitor.py "$@"
