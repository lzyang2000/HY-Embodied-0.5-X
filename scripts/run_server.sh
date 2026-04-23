#!/usr/bin/env bash
# ============================================================
# Launch HY-Embodied-0.5-X as an OpenAI-compatible API server
# ============================================================
#
# Usage:
#   bash scripts/run_server.sh                          # defaults
#   bash scripts/run_server.sh --port 8080              # custom port
#   CUDA_VISIBLE_DEVICES=0 bash scripts/run_server.sh   # select GPU
#
# After startup, call from any OpenAI SDK client:
#   from openai import OpenAI
#   client = OpenAI(base_url="http://localhost:8080/v1", api_key="any")
#   resp = client.chat.completions.create(
#       model="HY-Embodied-0.5-X",
#       messages=[{"role":"user","content":"Hello!"}],
#   )
# ============================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# Default arguments (can be overridden by passing CLI flags)
MODEL="${MODEL:-${PROJECT_ROOT}/ckpts/HY-Embodied-0.5-X}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8080}"
DTYPE="${DTYPE:-bfloat16}"

cd "$PROJECT_ROOT"

echo "============================================="
echo "  HY-Embodied-0.5-X  OpenAI API Server"
echo "============================================="
echo "  Model : ${MODEL}"
echo "  Host  : ${HOST}"
echo "  Port  : ${PORT}"
echo "  Dtype : ${DTYPE}"
echo "============================================="

python -m hy_embodied.cli.server \
    --model "${MODEL}" \
    --host "${HOST}" \
    --port "${PORT}" \
    --dtype "${DTYPE}" \
    "$@"
