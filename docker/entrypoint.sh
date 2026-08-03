#!/usr/bin/env bash
set -euo pipefail

# Build artifacts on first boot (idempotent: skip if the model already exists).
if [ ! -f "${FLEET_MODEL_ROOT:-/app/models}/risk_model.txt" ]; then
  echo "No model found — running the pipeline (generate -> train)..."
  python -m fleet_adas.cli all
fi

echo "Starting API on :8000"
exec uvicorn fleet_adas.api.main:app --host 0.0.0.0 --port 8000
