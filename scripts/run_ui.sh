#!/bin/bash
# Launch the Research Loop Web Dashboard Control Tower
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"
PORT="${1:-8503}"
cd "$ROOT_DIR"
exec ./venv/bin/python -m streamlit run src/ui/dashboard.py --server.port "$PORT" --server.headless true
