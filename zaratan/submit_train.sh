#!/usr/bin/env bash
#
# Submit main STAR-LDM training to Zaratan / SLURM.
#
# FINEWEB_LOCAL_PATH and C4_LOCAL_PATH are required — run scripts/download_assets.py
# on a login node first to populate them.
#
# Example:
#   STARLDM_SCRATCH=/home/mbezick/scratch/starLDM
#
#   VENV_PATH="${STARLDM_SCRATCH}/.venv" \
#   HF_HOME="${STARLDM_SCRATCH}/.hf_cache" \
#   PARTITION=gpu \
#   FINEWEB_LOCAL_PATH="${STARLDM_SCRATCH}/datasets/fineweb-10BT" \
#   C4_LOCAL_PATH="${STARLDM_SCRATCH}/datasets/c4-validation" \
#     zaratan/submit_train.sh
#
#   CONFIG_PATH=configs/train_fineweb.yaml \
#   EXTRA_ARGS="train.learning_rate=1e-4" \
#     zaratan/submit_train.sh

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
LOG_DIR="${REPO_ROOT}/logs/slurm"

CONFIG_PATH="${CONFIG_PATH:-configs/train_fineweb.yaml}"
FINEWEB_LOCAL_PATH="${FINEWEB_LOCAL_PATH:?FINEWEB_LOCAL_PATH not set}"
C4_LOCAL_PATH="${C4_LOCAL_PATH:?C4_LOCAL_PATH not set}"
HF_HOME="${HF_HOME:-${REPO_ROOT}/.hf_cache}"
EXTRA_ARGS="${EXTRA_ARGS:-}"
CUDA_DEVICE="${CUDA_DEVICE:-0}"

ACCOUNT="${ACCOUNT:-}"
PARTITION="${PARTITION:-}"
QOS="${QOS:-}"
TIME_LIMIT="${TIME_LIMIT:-24:00:00}"
NUM_GPUS="${NUM_GPUS:-1}"
GPU_TYPE="${GPU_TYPE:-a100}"
CPUS_PER_TASK="${CPUS_PER_TASK:-8}"
MEMORY="${MEMORY:-64G}"

if [[ ! -f "${REPO_ROOT}/${CONFIG_PATH}" && ! -f "${CONFIG_PATH}" ]]; then
    echo "ERROR: config not found: ${CONFIG_PATH}" >&2
    exit 1
fi

if [[ ! -d "${FINEWEB_LOCAL_PATH}" ]]; then
    echo "ERROR: FINEWEB_LOCAL_PATH not found: ${FINEWEB_LOCAL_PATH}" >&2
    echo "       Run scripts/download_assets.py on a login node first." >&2
    exit 1
fi

if [[ ! -d "${C4_LOCAL_PATH}" ]]; then
    echo "ERROR: C4_LOCAL_PATH not found: ${C4_LOCAL_PATH}" >&2
    echo "       Run scripts/download_assets.py on a login node first." >&2
    exit 1
fi

export REPO_ROOT
export CONFIG_PATH
export FINEWEB_LOCAL_PATH
export C4_LOCAL_PATH
export HF_HOME
export EXTRA_ARGS
export CUDA_DEVICE

mkdir -p "${LOG_DIR}"

sbatch_args=(
    "--job-name=train-star-ldm"
    "--nodes=1" "--ntasks=1"
    "--chdir=${REPO_ROOT}"
    "--cpus-per-task=${CPUS_PER_TASK}"
    "--time=${TIME_LIMIT}"
    "--mem=${MEMORY}"
    "--output=${LOG_DIR}/%x-%j.out"
    "--export=ALL"
)

if [[ -n "${ACCOUNT}" ]]; then sbatch_args+=("--account=${ACCOUNT}"); fi
if [[ -n "${PARTITION}" ]]; then sbatch_args+=("--partition=${PARTITION}"); fi
if [[ -n "${QOS}" ]]; then sbatch_args+=("--qos=${QOS}"); fi
if [[ -n "${GPU_TYPE}" ]]; then
    sbatch_args+=("--gres=gpu:${GPU_TYPE}:${NUM_GPUS}")
else
    sbatch_args+=("--gres=gpu:${NUM_GPUS}")
fi

echo "Submitting STAR-LDM training..."
echo "  config: ${CONFIG_PATH}"
echo "  fineweb_local_path: ${FINEWEB_LOCAL_PATH}"
echo "  c4_local_path: ${C4_LOCAL_PATH}"
echo "  hf_home: ${HF_HOME}"
echo "  extra_args: ${EXTRA_ARGS:-none}"
sbatch "${sbatch_args[@]}" "${SCRIPT_DIR}/run_train.sbatch"
