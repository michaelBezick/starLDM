#!/usr/bin/env bash
#
# Submit selector dataset collection to Zaratan / SLURM.
#
# Required:
#   MODEL_PATH=checkpoints/star-ldm
#   PROMPTS_PATH=data/selector_prompts.jsonl
#   OUTPUT_PATH=data/selector_dataset.pt
#
# Example:
#   MODEL_PATH=checkpoints/star-ldm \
#   PROMPTS_PATH=data/selector_prompts.jsonl \
#   OUTPUT_PATH=data/selector_dataset.pt \
#   SAVE_NOISY_TIMESTEPS=1 \
#   TIME_LIMIT=24:00:00 \
#     zaratan/submit_collect_selector_data.sh
#
# Extra collect_selector_data.py flags can be appended with:
#   EXTRA_ARGS="--num_samples_per_prompt 32 --sampling_timesteps 100"

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
LOG_DIR="${REPO_ROOT}/logs/slurm"

MODEL_PATH="${MODEL_PATH:-}"
PROMPTS_PATH="${PROMPTS_PATH:-}"
OUTPUT_PATH="${OUTPUT_PATH:-}"
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

ACCOUNT="${ACCOUNT:-}"
PARTITION="${PARTITION:-}"
QOS="${QOS:-}"
TIME_LIMIT="${TIME_LIMIT:-24:00:00}"
NUM_GPUS="${NUM_GPUS:-1}"
GPU_TYPE="${GPU_TYPE:-a100}"
CPUS_PER_TASK="${CPUS_PER_TASK:-8}"
MEMORY="${MEMORY:-64G}"

if [[ -z "${MODEL_PATH}" ]]; then
    echo "ERROR: MODEL_PATH is required" >&2
    exit 1
fi
if [[ -z "${PROMPTS_PATH}" ]]; then
    echo "ERROR: PROMPTS_PATH is required" >&2
    exit 1
fi
if [[ -z "${OUTPUT_PATH}" ]]; then
    echo "ERROR: OUTPUT_PATH is required" >&2
    exit 1
fi

if [[ ! -d "${REPO_ROOT}/${MODEL_PATH}" && ! -f "${REPO_ROOT}/${MODEL_PATH}" && ! -d "${MODEL_PATH}" && ! -f "${MODEL_PATH}" ]]; then
    echo "ERROR: model path not found: ${MODEL_PATH}" >&2
    exit 1
fi
if [[ ! -f "${REPO_ROOT}/${PROMPTS_PATH}" && ! -f "${PROMPTS_PATH}" ]]; then
    echo "ERROR: prompts file not found: ${PROMPTS_PATH}" >&2
    exit 1
fi
if [[ "${SAMPLER}" != "ddpm" && "${SAMPLER}" != "ddim" ]]; then
    echo "ERROR: SAMPLER must be ddpm or ddim" >&2
    exit 1
fi

export REPO_ROOT
export MODEL_PATH
export PROMPTS_PATH
export OUTPUT_PATH
export NUM_SAMPLES_PER_PROMPT
export VERIFIER
export SAVE_NOISY_TIMESTEPS
export NOISY_PER_CLEAN
export SAMPLING_TIMESTEPS
export SAMPLER
export COSINE_SCALE
export MAX_NEW_TOKENS
export SEED
export EXTRA_ARGS
export CUDA_DEVICE

mkdir -p "${LOG_DIR}"

sbatch_args=(
    "--job-name=collect-selector-data"
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

echo "Submitting selector dataset collection..."
echo "  model_path: ${MODEL_PATH}"
echo "  prompts_path: ${PROMPTS_PATH}"
echo "  output_path: ${OUTPUT_PATH}"
echo "  num_samples_per_prompt: ${NUM_SAMPLES_PER_PROMPT}"
echo "  save_noisy_timesteps: ${SAVE_NOISY_TIMESTEPS}"
echo "  extra_args: ${EXTRA_ARGS:-none}"
sbatch "${sbatch_args[@]}" "${SCRIPT_DIR}/run_collect_selector_data.sbatch"
