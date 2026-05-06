#!/usr/bin/env bash
#SBATCH --output=logs/slurm/%x-%j.out

set -euo pipefail

REPO_ROOT="${REPO_ROOT:?REPO_ROOT not set; submit via submit_collect_selector_data.sh}"
MODEL_PATH="${MODEL_PATH:?MODEL_PATH not set; submit via submit_collect_selector_data.sh}"
PROMPTS_PATH="${PROMPTS_PATH:?PROMPTS_PATH not set; submit via submit_collect_selector_data.sh}"
OUTPUT_PATH="${OUTPUT_PATH:?OUTPUT_PATH not set; submit via submit_collect_selector_data.sh}"
NUM_SAMPLES_PER_PROMPT="${NUM_SAMPLES_PER_PROMPT:-16}"
VERIFIER="${VERIFIER:-exact_match}"
SAVE_NOISY_TIMESTEPS="${SAVE_NOISY_TIMESTEPS:-0}"
NOISY_PER_CLEAN="${NOISY_PER_CLEAN:-4}"
SAMPLING_TIMESTEPS="${SAMPLING_TIMESTEPS:-50}"
SAMPLER="${SAMPLER:-ddpm}"
COSINE_SCALE="${COSINE_SCALE:-3.0}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-64}"
SEED="${SEED:-0}"
EXTRA_ARGS="${EXTRA_ARGS:-}"
CUDA_DEVICE="${CUDA_DEVICE:-0}"
VENV_PATH="${VENV_PATH:-${REPO_ROOT}/.venv}"

mkdir -p "${REPO_ROOT}/logs/slurm"
cd "${REPO_ROOT}"

echo "hostname: $(hostname)"
echo "repo_root: ${REPO_ROOT}"
echo "model_path: ${MODEL_PATH}"
echo "prompts_path: ${PROMPTS_PATH}"
echo "output_path: ${OUTPUT_PATH}"
echo "num_samples_per_prompt: ${NUM_SAMPLES_PER_PROMPT}"
echo "verifier: ${VERIFIER}"
echo "save_noisy_timesteps: ${SAVE_NOISY_TIMESTEPS}"
echo "noisy_per_clean: ${NOISY_PER_CLEAN}"
echo "sampling_timesteps: ${SAMPLING_TIMESTEPS}"
echo "sampler: ${SAMPLER}"
echo "cosine_scale: ${COSINE_SCALE}"
echo "max_new_tokens: ${MAX_NEW_TOKENS}"
echo "seed: ${SEED}"
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
    python -m scripts.collect_selector_data
    --model_path "${MODEL_PATH}"
    --prompts_path "${PROMPTS_PATH}"
    --output_path "${OUTPUT_PATH}"
    --num_samples_per_prompt "${NUM_SAMPLES_PER_PROMPT}"
    --verifier "${VERIFIER}"
    --noisy_per_clean "${NOISY_PER_CLEAN}"
    --sampling_timesteps "${SAMPLING_TIMESTEPS}"
    --sampler "${SAMPLER}"
    --cosine_scale "${COSINE_SCALE}"
    --max_new_tokens "${MAX_NEW_TOKENS}"
    --device cuda
    --seed "${SEED}"
)

if [[ "${SAVE_NOISY_TIMESTEPS}" == "1" || "${SAVE_NOISY_TIMESTEPS}" == "true" || "${SAVE_NOISY_TIMESTEPS}" == "TRUE" ]]; then
    cmd+=(--save_noisy_timesteps)
fi

if [[ -n "${EXTRA_ARGS}" ]]; then
    read -r -a extra_args <<< "${EXTRA_ARGS}"
    cmd+=("${extra_args[@]}")
fi

printf 'Running: %s\n' "${cmd[*]}"
echo "command_started_at: $(date --iso-8601=seconds)"
srun "${cmd[@]}"
echo "finished_at: $(date --iso-8601=seconds)"
