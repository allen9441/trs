#!/usr/bin/env bash
# Start backend (FastAPI :8000) and frontend (Streamlit :8501); Ctrl+C stops both.
set -euo pipefail
cd "$(dirname "$0")"

uv run uvicorn backend.api:app --host 127.0.0.1 --port 8000 &
BACKEND_PID=$!
trap 'kill $BACKEND_PID 2>/dev/null' EXIT

XRD_BACKEND=http://127.0.0.1:8000 uv run streamlit run frontend/app.py --server.port 8501
