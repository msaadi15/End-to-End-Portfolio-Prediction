#!/usr/bin/env bash
# Serves the current champion model via FastAPI on http://127.0.0.1:8000
set -e

cd "$(dirname "$0")/.."
source venv/bin/activate 2>/dev/null || true

export PYTHONPATH="$(pwd):$PYTHONPATH"

echo "🌐 Starting API at http://127.0.0.1:8000 (docs at /docs) ..."
uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload
