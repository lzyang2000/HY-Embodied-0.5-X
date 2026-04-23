#!/usr/bin/env bash
# HY-Embodied-0.5-X SFT — single-GPU launcher (no torchrun / accelerate needed)
#
# Usage:
#   bash scripts/run_sft_single_gpu.sh                    # default: GPU 0
#   GPU_ID=2 bash scripts/run_sft_single_gpu.sh           # use GPU 2
#   CONFIG=configs/sft/my_config.yaml bash scripts/run_sft_single_gpu.sh
#
# This script is the simplest way to validate the full training pipeline on a
# single GPU.  It does NOT require torchrun, accelerate, or DeepSpeed — just
# plain ``python -m``.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_DIR"
export PYTHONPATH="$PROJECT_DIR/src${PYTHONPATH:+:$PYTHONPATH}"

GPU_ID=${GPU_ID:-0}
CONFIG=${CONFIG:-configs/sft/example_small_single_gpu.yaml}
TRAIN_MODULE=${TRAIN_MODULE:-hy_embodied.cli.train}

export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4}
export PYTHONUNBUFFERED=${PYTHONUNBUFFERED:-1}

echo "[$(date '+%Y-%m-%d %H:%M:%S')] HY-Embodied-0.5-X SFT — single GPU (GPU ${GPU_ID})"
echo "  Config : ${CONFIG}"
echo "  Module : python -m ${TRAIN_MODULE}"
echo ""

CUDA_VISIBLE_DEVICES=${GPU_ID} python -m "${TRAIN_MODULE}" --config "${CONFIG}" "$@"
