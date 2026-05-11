#!/usr/bin/env bash
#SBATCH --output=logs/slurm/%x-%j.out

set -euo pipefail

REPO_ROOT="${REPO_ROOT:?REPO_ROOT not set; submit via submit_evaluate_gsm8k.sh}"
MODEL_PATH="${MODEL_PATH:?MODEL_PATH not set; submit via submit_evaluate_gsm8k.sh}"
SELECTOR_PATH="${SELECTOR_PATH:?SELECTOR_PATH not set; submit via submit_evaluate_gsm8k.sh}"
GSM8K_PATH="${GSM8K_PATH:?GSM8K_PATH not set; submit via submit_evaluate_gsm8k.sh}"
OUTPUT_PATH="${OUTPUT_PATH:?OUTPUT_PATH not set; submit via submit_evaluate_gsm8k.sh}"
SPLIT="${SPLIT:-test}"
LIMIT="${LIMIT:-}"
SAMPLING_TIMESTEPS="${SAMPLING_TIMESTEPS:-50}"
SAMPLER="${SAMPLER:-ddpm}"
CLS_FREE_GUIDANCE="${CLS_FREE_GUIDANCE:-1.0}"
SELECTOR_GUIDANCE="${SELECTOR_GUIDANCE:-1.0}"
NUM_PLAN_BRANCHES="${NUM_PLAN_BRANCHES:-8}"
REPULSION_SCALE="${REPULSION_SCALE:-0.0}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-256}"
DO_SAMPLE="${DO_SAMPLE:-0}"
TOP_P="${TOP_P:-0.9}"
REPETITION_PENALTY="${REPETITION_PENALTY:-1.2}"
SEED="${SEED:-0}"
EXTRA_ARGS="${EXTRA_ARGS:-}"
CUDA_DEVICE="${CUDA_DEVICE:-0}"
VENV_PATH="${VENV_PATH:-${REPO_ROOT}/.venv}"
SRUN_CPU_BIND="${SRUN_CPU_BIND:-none}"

mkdir -p "${REPO_ROOT}/logs/slurm"
cd "${REPO_ROOT}"

echo "hostname: $(hostname)"
echo "repo_root: ${REPO_ROOT}"
echo "model_path: ${MODEL_PATH}"
echo "selector_path: ${SELECTOR_PATH}"
echo "gsm8k_path: ${GSM8K_PATH}"
echo "output_path: ${OUTPUT_PATH}"
echo "split: ${SPLIT}"
echo "limit: ${LIMIT:-none}"
echo "sampling_timesteps: ${SAMPLING_TIMESTEPS}"
echo "sampler: ${SAMPLER}"
echo "cls_free_guidance: ${CLS_FREE_GUIDANCE}"
echo "selector_guidance: ${SELECTOR_GUIDANCE}"
echo "num_plan_branches: ${NUM_PLAN_BRANCHES}"
echo "repulsion_scale: ${REPULSION_SCALE}"
echo "max_new_tokens: ${MAX_NEW_TOKENS}"
echo "do_sample: ${DO_SAMPLE}"
echo "top_p: ${TOP_P}"
echo "repetition_penalty: ${REPETITION_PENALTY}"
echo "seed: ${SEED}"
echo "extra_args: ${EXTRA_ARGS:-none}"
echo "venv_path: ${VENV_PATH}"
echo "cuda_device: ${CUDA_DEVICE}"
echo "srun_cpu_bind: ${SRUN_CPU_BIND}"
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

# --- Offline / no-network settings ---
export HF_HOME="${HF_HOME:-${REPO_ROOT}/.hf_cache}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export WANDB_MODE=offline

cmd=(
    python -m scripts.evaluate_gsm8k
    --model_path "${MODEL_PATH}"
    --selector_path "${SELECTOR_PATH}"
    --gsm8k_path "${GSM8K_PATH}"
    --split "${SPLIT}"
    --output_path "${OUTPUT_PATH}"
    --sampling_timesteps "${SAMPLING_TIMESTEPS}"
    --sampler "${SAMPLER}"
    --cls_free_guidance "${CLS_FREE_GUIDANCE}"
    --selector_guidance "${SELECTOR_GUIDANCE}"
    --num_plan_branches "${NUM_PLAN_BRANCHES}"
    --repulsion_scale "${REPULSION_SCALE}"
    --max_new_tokens "${MAX_NEW_TOKENS}"
    --top_p "${TOP_P}"
    --repetition_penalty "${REPETITION_PENALTY}"
    --seed "${SEED}"
    --device cuda
)

if [[ -n "${LIMIT}" ]]; then
    cmd+=(--limit "${LIMIT}")
fi

if [[ "${DO_SAMPLE}" == "1" || "${DO_SAMPLE}" == "true" || "${DO_SAMPLE}" == "TRUE" ]]; then
    cmd+=(--do_sample)
fi

if [[ -n "${EXTRA_ARGS}" ]]; then
    read -r -a extra_args <<< "${EXTRA_ARGS}"
    cmd+=("${extra_args[@]}")
fi

printf 'Running: %s\n' "${cmd[*]}"
echo "command_started_at: $(date --iso-8601=seconds)"
srun --cpu-bind="${SRUN_CPU_BIND}" "${cmd[@]}"
echo "finished_at: $(date --iso-8601=seconds)"
