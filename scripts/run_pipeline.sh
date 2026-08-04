#!/usr/bin/env bash
# Downloads real market data, builds features, trains all candidate models,
# evaluates them, and promotes a champion. Watch it live in the ZenML dashboard
# at http://127.0.0.1:8237 (started by scripts/setup.sh).
set -e



python -m src.pipelines.training_pipeline

echo ""
echo "✅ Done. Open the ZenML dashboard to inspect the run: http://127.0.0.1:8237"
echo "   -> Next: bash scripts/start_api.sh"
