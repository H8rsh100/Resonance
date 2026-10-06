#!/usr/bin/env bash
# Bootstrap for the Citizen Fraud Shield demo (macOS / Linux / bash / WSL).
set -euo pipefail
cd "$(dirname "$0")"

echo "=== 1/4  Building the labelled corpus ==="
python -m src.data.build_dataset

echo "=== 2/4  Downloading the public cross-domain corpus ==="
python -m src.data.fetch_public

echo "=== 3/4  Training and evaluating ==="
python -m src.evaluate

echo "=== 4/4  Launching the Citizen Fraud Shield ==="
echo "Opening http://localhost:8501 - press Ctrl+C to stop."
streamlit run app.py