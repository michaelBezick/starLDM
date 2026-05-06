#!/usr/bin/env bash
#SBATCH --output=logs/slurm/%x-%j.out

set -euo pipefail

REPO_ROOT="${REPO_ROOT:?REPO_ROOT not set; submit via submit_train_selector.sh}"
CONFIG_PATH="${CONFIG_PATH:-configs/selector_train.yaml}"
DATA_PATH="${DATA_PATH:-}"
OUTPUT_DIR="${OUTPUT_DIR:-}"
EXTRA_ARGS="${EXTRA_ARGS:-}"
CUDA_DEVICE="${CUDA_DEVICE:-0}"
VENV_PATH="${VENV_PATH:-${REPO_ROOT}/.venv}"

mkdir -p "${REPO_ROOT}/logs/slurm"
cd "${REPO_ROOT}"

echo "hostname: $(hostname)"
echo "repo_root: ${REPO_ROOT}"
echo "config_path: ${CONFIG_PATH}"
echo "data_path: ${DATA_PATH:-from config}"
echo "output_dir: ${OUTPUT_DIR:-from config}"
echo "extra_args: ${EXTRA_ARGS:-none}"
echo "venv_path: ${VENV_PATH}"
echo "cuda_device: ${CUDA_DEVICE}"
echo "slurm_job_id: ${SLURM_JOB_ID:-unknown}"
echo "slurm_time_limit: ${SLURM_TIMELIMIT:-unknown} minutes"
echo "started_at: $(date --iso-8601=seconds)"

if [[ ! -f "${VENV_PATH}/bin/activate" ]]; then
    echo "Virtual environment not found at ${VENV_PATH}/bin/activate" >&2
    exit 1
fi

source "${VENV_PATH}/bin/activate"

export CUDA_VISIBLE_DEVICES="${CUDA_DEVICE}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1

cmd=(
    python -m scripts.train_selector
    --config "${CONFIG_PATH}"
)

if [[ -n "${DATA_PATH}" ]]; then
    cmd+=(--data_path "${DATA_PATH}")
fi

if [[ -n "${OUTPUT_DIR}" ]]; then
    cmd+=(--output_dir "${OUTPUT_DIR}")
fi

if [[ -n "${EXTRA_ARGS}" ]]; then
    read -r -a extra_args <<< "${EXTRA_ARGS}"
    cmd+=("${extra_args[@]}")
fi

printf 'Running: %s\n' "${cmd[*]}"
echo "command_started_at: $(date --iso-8601=seconds)"
srun "${cmd[@]}"
echo "finished_at: $(date --iso-8601=seconds)"
