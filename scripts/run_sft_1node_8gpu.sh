#!/usr/bin/env bash
# HY-Embodied-0.5-X SFT — single-node multi-GPU launcher
#
# Usage:
#   bash scripts/run_sft_1node_8gpu.sh                          # default config, 8 GPUs, torchrun
#   NUM_GPUS=4 bash scripts/run_sft_1node_8gpu.sh               # 4 GPUs
#   CONFIG=configs/sft/example_small.yaml bash scripts/run_sft_1node_8gpu.sh
#   LAUNCHER=accelerate bash scripts/run_sft_1node_8gpu.sh      # accelerate launcher
#
# Environment variables:
#   CONFIG                   default configs/sft/example_small.yaml
#   NUM_GPUS                 default 8
#   LAUNCHER                 default torchrun (alt: accelerate)
#   ACCELERATE_CONFIG_FILE   default configs/accelerate/sft_1node_8gpu_zero2.yaml
#   TRAIN_MODULE             default hy_embodied.cli.train

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_DIR"
export PYTHONPATH="$PROJECT_DIR/src${PYTHONPATH:+:$PYTHONPATH}"

NUM_GPUS=${NUM_GPUS:-8}
CONFIG=${CONFIG:-configs/sft/example_small.yaml}
LAUNCHER=${LAUNCHER:-torchrun}
ACCELERATE_CONFIG_FILE=${ACCELERATE_CONFIG_FILE:-configs/accelerate/sft_1node_8gpu_zero2.yaml}
TRAIN_MODULE=${TRAIN_MODULE:-hy_embodied.cli.train}

export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4}
export NCCL_IB_DISABLE=${NCCL_IB_DISABLE:-0}
export NCCL_DEBUG=${NCCL_DEBUG:-WARN}
export PYTHONUNBUFFERED=${PYTHONUNBUFFERED:-1}

timestamp() { date '+%Y-%m-%d %H:%M:%S'; }

resolve_output_dir() {
    local config_path="$1"
    local resolved_output

    resolved_output=$(awk '
        /^[[:space:]]*train:[[:space:]]*$/ {in_train=1; next}
        in_train && /^[^[:space:]]/ {in_train=0}
        in_train && /^[[:space:]]*output_dir:[[:space:]]*/ {
            sub(/^[[:space:]]*output_dir:[[:space:]]*/, "", $0)
            gsub(/^["'\''"]|["'\''"]$/, "", $0)
            print $0
            exit
        }
    ' "$config_path" 2>/dev/null || true)

    if [ -n "${resolved_output}" ]; then
        printf '%s\n' "${resolved_output}"
    else
        printf 'outputs/train_logs\n'
    fi
}

log() {
    local message="$1"
    printf '[%s] %s\n' "$(timestamp)" "${message}" | tee -a "${LOG_FILE}"
}

run_with_logging() {
    local -a cmd=("$@")
    local cmd_str=""
    local start_epoch end_epoch status

    printf -v cmd_str '%q ' "${cmd[@]}"
    log "Run command: ${cmd_str}"

    start_epoch=$(date +%s)
    if command -v script >/dev/null 2>&1; then
        script -aqefc "${cmd_str}" "${LOG_FILE}"
        status=$?
    else
        "${cmd[@]}" 2>&1 | tee -a "${LOG_FILE}"
        status=${PIPESTATUS[0]}
    fi
    end_epoch=$(date +%s)

    log "Command exit code: ${status}"
    log "Elapsed seconds: $((end_epoch - start_epoch))"
    return "${status}"
}

START_TS=$(date '+%Y%m%d_%H%M%S')
LOG_DIR=${LOG_DIR:-"$(resolve_output_dir "${CONFIG}")/logs"}
mkdir -p "${LOG_DIR}"
LOG_FILE=${LOG_FILE:-"${LOG_DIR}/sft_1node_${NUM_GPUS}gpu_${START_TS}.txt"}

on_exit() {
    local status=$?
    if [ "${status}" -eq 0 ]; then
        log "Training finished successfully."
    else
        log "Training failed with exit code ${status}."
    fi
    log "Log file: ${LOG_FILE}"
}
trap on_exit EXIT

log "=============================="
log "HY-Embodied-0.5-X SFT — 1 node x ${NUM_GPUS} GPUs"
log "Config    : ${CONFIG}"
log "Launcher  : ${LAUNCHER}"
log "Module    : python -m ${TRAIN_MODULE}"
log "PYTHONPATH: ${PYTHONPATH}"
log "Log file  : ${LOG_FILE}"
log "=============================="

if [ "${LAUNCHER}" = "accelerate" ]; then
    if ! command -v accelerate >/dev/null 2>&1; then
        log "accelerate is not installed. Please install it first, e.g. 'pip install accelerate'."
        exit 127
    fi

    log "Accelerate config: ${ACCELERATE_CONFIG_FILE}"
    run_with_logging \
        accelerate launch \
        --config_file "${ACCELERATE_CONFIG_FILE}" \
        --num_processes "${NUM_GPUS}" \
        -m "${TRAIN_MODULE}" \
        --config "${CONFIG}" \
        "$@"
else
    run_with_logging \
        torchrun \
        --standalone \
        --nproc_per_node="${NUM_GPUS}" \
        -m "${TRAIN_MODULE}" \
        --config "${CONFIG}" \
        "$@"
fi
