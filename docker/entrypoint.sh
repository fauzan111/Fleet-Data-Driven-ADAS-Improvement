#!/usr/bin/env bash
set -euo pipefail

# Artifacts are normally baked into the image at build time. As a fallback
# (e.g. an empty mounted volume), build them on first boot if the model is
# missing — idempotent.
if [ ! -f "${FLEET_MODEL_ROOT:-/app/models}/risk_model.txt" ]; then
  echo "No model found — running the pipeline (generate -> train)..."
  python -m fleet_adas.cli all
fi

PORT="${PORT:-8000}"
echo "Starting API on :${PORT}"
exec uvicorn fleet_adas.api.main:app --host 0.0.0.0 --port "${PORT}"
