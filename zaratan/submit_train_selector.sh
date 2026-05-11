#!/usr/bin/env bash
#
# Submit selector training to Zaratan / SLURM.
#
# Defaults train with configs/selector_train.yaml. Override the dataset,
# output directory, config, or extra train_selector.py flags via env vars:
#
#   STARLDM_SCRATCH=/home/mbezick/scratch/starLDM
#
#   VENV_PATH="${STARLDM_SCRATCH}/.venv" \
#   HF_HOME="${STARLDM_SCRATCH}/.hf_cache" \
#   PARTITION=gpu \
#   DATA_PATH="${STARLDM_SCRATCH}/data/selector_dataset.pt" \
#   OUTPUT_DIR="${STARLDM_SCRATCH}/checkpoints/selector" \
#   TIME_LIMIT=12:00:00 \
#     zaratan/submit_train_selector.sh
#
#   CONFIG_PATH=configs/selector_train.yaml \
#   EXTRA_ARGS="--num_steps 5000 --batch_size 128" \
#     zaratan/submit_train_selector.sh

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
LOG_DIR="${REPO_ROOT}/logs/slurm"

CONFIG_PATH="${CONFIG_PATH:-configs/selector_train.yaml}"
DATA_PATH="${DATA_PATH:-}"
OUTPUT_DIR="${OUTPUT_DIR:-}"
EXTRA_ARGS="${EXTRA_ARGS:-}"
CUDA_DEVICE="${CUDA_DEVICE:-0}"

ACCOUNT="${ACCOUNT:-}"
PARTITION="${PARTITION:-}"
QOS="${QOS:-}"
TIME_LIMIT="${TIME_LIMIT:-12:00:00}"
NUM_GPUS="${NUM_GPUS:-1}"
GPU_TYPE="${GPU_TYPE:-a100}"
CPUS_PER_TASK="${CPUS_PER_TASK:-8}"
MEMORY="${MEMORY:-64G}"

if [[ ! -f "${REPO_ROOT}/${CONFIG_PATH}" ]]; then
    echo "ERROR: config not found: ${CONFIG_PATH}" >&2
    exit 1
fi

if [[ -n "${DATA_PATH}" && ! -f "${REPO_ROOT}/${DATA_PATH}" && ! -f "${DATA_PATH}" ]]; then
    echo "ERROR: selector dataset not found: ${DATA_PATH}" >&2
    exit 1
fi

export REPO_ROOT
export CONFIG_PATH
export DATA_PATH
export OUTPUT_DIR
export EXTRA_ARGS
export CUDA_DEVICE

mkdir -p "${LOG_DIR}"

sbatch_args=(
    "--job-name=train-selector"
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

echo "Submitting selector training..."
echo "  config: ${CONFIG_PATH}"
echo "  data_path: ${DATA_PATH:-from config}"
echo "  output_dir: ${OUTPUT_DIR:-from config}"
echo "  extra_args: ${EXTRA_ARGS:-none}"
sbatch "${sbatch_args[@]}" "${SCRIPT_DIR}/run_train_selector.sbatch"
