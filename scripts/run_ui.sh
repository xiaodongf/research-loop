#!/bin/bash
# Launch the Research Loop Web Dashboard Control Tower
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"

cd "$ROOT_DIR"
exec ./venv/bin/streamlit run src/ui/dashboard.py --server.port 8501 --server.headless true
