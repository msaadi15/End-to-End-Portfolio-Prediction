#!/usr/bin/env bash
# One-time setup: virtualenv, dependencies, ZenML init + dashboard.
set -e

cd "$(dirname "$0")/.."

echo "🔧 Creating virtual environment..."
python3 -m venv venv
source venv/bin/activate

echo "📦 Installing dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

echo "🧬 Initializing ZenML..."
zenml init || true

echo "🚀 Starting ZenML dashboard (http://127.0.0.1:8237) ..."
zenml up

echo ""
echo "✅ Setup complete."
echo "   -> Next: edit config.yaml, then run: bash scripts/run_pipeline.sh"
