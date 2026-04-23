#!/usr/bin/env bash
# HY-Embodied-0.5-X SFT — multi-node multi-GPU launcher (4 nodes x 8 GPUs by default)
#
# ============================================================
# Mode 1: SLURM
# ============================================================
#   sbatch scripts/run_sft_4node_8gpu.sh
#   # or
#   srun --nodes=4 --ntasks-per-node=8 --gpus-per-node=8 \
#        bash scripts/run_sft_4node_8gpu.sh
#
# ============================================================
# Mode 2: Manual multi-node torchrun
# ============================================================
#   # On each node:
#   MASTER_ADDR=<node0_ip> MASTER_PORT=29500 NODE_RANK=<0-3> \
#     bash scripts/run_sft_4node_8gpu.sh
#
# ============================================================
# Mode 3: Taiji / Jizhi style platforms
# ============================================================
#   If NODE_IP / NODE_IP_LIST / HOST_NUM / HOST_GPU_NUM are injected by the
#   platform, MASTER_ADDR and NODE_RANK are inferred automatically.
#
# Environment variables:
#   CONFIG / TRAINING_CONFIG   default configs/sft/example_small.yaml
#   NUM_NODES                  default 4
#   NUM_GPUS                   default 8
#   LAUNCHER                   default accelerate (alt: torchrun)
#   ACCELERATE_CONFIG / ACCELERATE_CONFIG_FILE
#                              default configs/accelerate/sft_4node_8gpu_zero2.yaml
#   TRAIN_MODULE               default hy_embodied.cli.train

#SBATCH --job-name=hy-embodied-sft-4x8
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=8
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=8
#SBATCH --time=72:00:00
#SBATCH --output=outputs/hy-embodied-sft-4node/slurm-%j.out
#SBATCH --error=outputs/hy-embodied-sft-4node/slurm-%j.err

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
WORK_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${WORK_DIR}"
export PYTHONPATH="${WORK_DIR}/src${PYTHONPATH:+:$PYTHONPATH}"

NUM_NODES=${NUM_NODES:-${HOST_NUM:-${TAIJI_HOST_NUM:-4}}}
NUM_GPUS=${NUM_GPUS:-${HOST_GPU_NUM:-8}}
CONFIG=${CONFIG:-${TRAINING_CONFIG:-configs/sft/example_small.yaml}}
TRAIN_MODULE=${TRAIN_MODULE:-hy_embodied.cli.train}
LAUNCHER=${LAUNCHER:-accelerate}
ACCELERATE_CONFIG=${ACCELERATE_CONFIG:-${ACCELERATE_CONFIG_FILE:-configs/accelerate/sft_4node_8gpu_zero2.yaml}}
ACCELERATE_CONFIG_FILE=${ACCELERATE_CONFIG}
TORCHRUN_LOCAL_ADDR=""

export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4}
# 当前集群多机阶段更稳的默认值：禁用 IB/RDMA，走 Socket。
# 若后续确认 RDMA 正常，再显式传 NCCL_IB_DISABLE=0 覆盖。
export NCCL_IB_DISABLE=${NCCL_IB_DISABLE:-1}
export NCCL_DEBUG=${NCCL_DEBUG:-WARN}
export NCCL_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME:-eth0}
export GLOO_SOCKET_IFNAME=${GLOO_SOCKET_IFNAME:-${NCCL_SOCKET_IFNAME}}
export NCCL_IB_GID_INDEX=${NCCL_IB_GID_INDEX:-3}
export NCCL_ASYNC_ERROR_HANDLING=${NCCL_ASYNC_ERROR_HANDLING:-1}
export TORCH_NCCL_ASYNC_ERROR_HANDLING=${TORCH_NCCL_ASYNC_ERROR_HANDLING:-1}
export NCCL_NET=${NCCL_NET:-Socket}
export PYTHONUNBUFFERED=${PYTHONUNBUFFERED:-1}

timestamp() {
    date '+%Y-%m-%d %H:%M:%S'
}

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
    local start_epoch
    local end_epoch
    local status

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
HOSTNAME_SAFE=$(hostname | tr -c '[:alnum:]._-' '_')
NODE_RANK_TAG=${NODE_RANK:-${SLURM_NODEID:-${INDEX:-unknown}}}
LOG_DIR=${LOG_DIR:-"$(resolve_output_dir "${CONFIG}")/logs"}
mkdir -p "${LOG_DIR}"
LOG_FILE=${LOG_FILE:-"${LOG_DIR}/sft_${NUM_NODES}node_${NUM_GPUS}gpu_node${NODE_RANK_TAG}_${HOSTNAME_SAFE}_${START_TS}.txt"}

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

if [ -n "${SLURM_JOB_ID:-}" ]; then
    MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n1)
    MASTER_PORT=${MASTER_PORT:-29500}
    NODE_RANK=${SLURM_NODEID:-0}
    log "[SLURM] Job ${SLURM_JOB_ID}, node ${NODE_RANK}/${NUM_NODES}"
elif [ -n "${NODE_IP_LIST:-}" ] && [ -n "${NODE_IP:-}" ]; then
    MASTER_ADDR=${MASTER_ADDR:-${CHIEF_IP:-}}
    if [ -z "${MASTER_ADDR}" ]; then
        MASTER_ADDR=$(printf '%s' "$NODE_IP_LIST" | awk -F'[,:]' '{print $1}')
    fi
    MASTER_PORT=${MASTER_PORT:-29500}
    NODE_RANK=${NODE_RANK:-${INDEX:-}}
    if [ -z "${NODE_RANK}" ]; then
        NODE_RANK=$(printf '%s' "$NODE_IP_LIST" | awk -v RS=, -v ip="$NODE_IP" -F: '$1==ip{print NR-1}')
    fi
    TORCHRUN_LOCAL_ADDR=${LOCAL_ADDR:-${LOCAL_IP:-${NODE_IP:-}}}

    if [ -z "${NODE_RANK:-}" ] && [ -n "${POD_NAME:-}" ]; then
        case "${POD_NAME}" in
            *-launcher)
                NODE_RANK=0
                ;;
            *-worker-*)
                NODE_RANK=$(printf '%s' "${POD_NAME##*-worker-}" | awk '{print $1 + 1}')
                ;;
        esac
    fi

    MASTER_ADDR=${MASTER_ADDR:?"Failed to infer MASTER_ADDR from NODE_IP_LIST"}
    NODE_RANK=${NODE_RANK:?"Failed to infer NODE_RANK from NODE_IP/NODE_IP_LIST/POD_NAME"}
    log "[TAIJI] Pod ${POD_NAME:-unknown}, node ${NODE_RANK}/${NUM_NODES}, ip ${NODE_IP}, chief ${MASTER_ADDR}, local ${TORCHRUN_LOCAL_ADDR:-n/a}"
else
    MASTER_ADDR=${MASTER_ADDR:?"Set MASTER_ADDR (node0 ip) for manual multi-node launch"}
    MASTER_PORT=${MASTER_PORT:-29500}
    NODE_RANK=${NODE_RANK:?"Set NODE_RANK (0..NUM_NODES-1) for manual multi-node launch"}
fi

export MASTER_ADDR MASTER_PORT NODE_RANK

log "=============================="
log "HY-Embodied-0.5-X SFT — ${NUM_NODES} nodes x ${NUM_GPUS} GPUs"
log "Config : ${CONFIG}"
log "Launcher: ${LAUNCHER}"
log "Train  : python -m ${TRAIN_MODULE}"
log "Accelerate: ${ACCELERATE_CONFIG_FILE}"
log "Master : ${MASTER_ADDR}:${MASTER_PORT}"
log "Node   : ${NODE_RANK}"
log "Pod    : ${POD_NAME:-n/a}"
log "NCCL   : IB_DISABLE=${NCCL_IB_DISABLE} NET=${NCCL_NET} IFNAME=${NCCL_SOCKET_IFNAME} GLOO_IFNAME=${GLOO_SOCKET_IFNAME}"
log "PYTHONPATH: ${PYTHONPATH}"
log "Log file: ${LOG_FILE}"
log "=============================="

if [ "${LAUNCHER}" = "accelerate" ]; then
    if ! command -v accelerate >/dev/null 2>&1; then
        log "accelerate is not installed. Please install it first, e.g. `pip install accelerate`."
        exit 127
    fi

    TOTAL_PROCESSES=$((NUM_NODES * NUM_GPUS))
    log "Accelerate total processes: ${TOTAL_PROCESSES}"

    ACCELERATE_ARGS=(
        launch
        --config_file "${ACCELERATE_CONFIG_FILE}"
        --num_machines "${NUM_NODES}"
        --num_processes "${TOTAL_PROCESSES}"
        --machine_rank "${NODE_RANK}"
        --main_process_ip "${MASTER_ADDR}"
        --main_process_port "${MASTER_PORT}"
    )

    run_with_logging \
        accelerate \
        "${ACCELERATE_ARGS[@]}" \
        -m "${TRAIN_MODULE}" \
        --config "${CONFIG}" \
        "$@"
else
    TORCHRUN_ARGS=(
        --nnodes="${NUM_NODES}"
        --nproc_per_node="${NUM_GPUS}"
        --node_rank="${NODE_RANK}"
        --master_addr="${MASTER_ADDR}"
        --master_port="${MASTER_PORT}"
    )

    if [ -n "${TORCHRUN_LOCAL_ADDR:-}" ] && torchrun --help 2>&1 | rg -q -- '--local-addr'; then
        TORCHRUN_ARGS+=(--local-addr="${TORCHRUN_LOCAL_ADDR}")
    fi

    run_with_logging \
        torchrun \
        "${TORCHRUN_ARGS[@]}" \
        -m "${TRAIN_MODULE}" \
        --config "${CONFIG}" \
        "$@"
fi
