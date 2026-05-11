#!/usr/bin/env bash
#
# Submit GSM8K generation evaluation to Zaratan / SLURM.
#
# Required:
#   MODEL_PATH=checkpoints/star-ldm
#   SELECTOR_PATH=/path/to/selector/checkpoint-dir
#   GSM8K_PATH=/path/to/local/gsm8k
#
# Example:
#   STARLDM_SCRATCH=/home/mbezick/scratch/starLDM
#
#   VENV_PATH="${STARLDM_SCRATCH}/.venv" \
#   HF_HOME="${STARLDM_SCRATCH}/.hf_cache" \
#   PARTITION=gpu \
#   MODEL_PATH=checkpoints/star-ldm \
#   SELECTOR_PATH="${STARLDM_SCRATCH}/checkpoints/selector-gsm8k-1k" \
#   GSM8K_PATH="${STARLDM_SCRATCH}/datasets/gsm8k" \
#   OUTPUT_PATH="${STARLDM_SCRATCH}/eval/gsm8k_selector_1k_test.jsonl" \
#   LIMIT=100 \
#   SAMPLER=ddim \
#   SAMPLING_TIMESTEPS=25 \
#   MAX_NEW_TOKENS=128 \
#   TIME_LIMIT=12:00:00 \
#       zaratan/submit_evaluate_gsm8k.sh
#
# Extra evaluate_gsm8k.py flags can be appended with:
#   EXTRA_ARGS="--num_plan_branches 4 --selector_guidance 0.5"

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
LOG_DIR="${REPO_ROOT}/logs/slurm"

MODEL_PATH="${MODEL_PATH:-}"
SELECTOR_PATH="${SELECTOR_PATH:-}"
GSM8K_PATH="${GSM8K_PATH:-}"
OUTPUT_PATH="${OUTPUT_PATH:-}"
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

ACCOUNT="${ACCOUNT:-}"
PARTITION="${PARTITION:-}"
QOS="${QOS:-}"
TIME_LIMIT="${TIME_LIMIT:-12:00:00}"
NUM_GPUS="${NUM_GPUS:-1}"
GPU_TYPE="${GPU_TYPE:-a100}"
CPUS_PER_TASK="${CPUS_PER_TASK:-8}"
MEMORY="${MEMORY:-64G}"

if [[ -z "${MODEL_PATH}" ]]; then
    echo "ERROR: MODEL_PATH is required" >&2
    exit 1
fi
if [[ -z "${SELECTOR_PATH}" ]]; then
    echo "ERROR: SELECTOR_PATH is required" >&2
    exit 1
fi
if [[ -z "${GSM8K_PATH}" ]]; then
    echo "ERROR: GSM8K_PATH is required" >&2
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
if [[ ! -d "${REPO_ROOT}/${SELECTOR_PATH}" && ! -f "${REPO_ROOT}/${SELECTOR_PATH}" && ! -d "${SELECTOR_PATH}" && ! -f "${SELECTOR_PATH}" ]]; then
    echo "ERROR: selector path not found: ${SELECTOR_PATH}" >&2
    exit 1
fi
if [[ ! -d "${REPO_ROOT}/${GSM8K_PATH}" && ! -d "${GSM8K_PATH}" ]]; then
    echo "ERROR: GSM8K_PATH not found: ${GSM8K_PATH}" >&2
    exit 1
fi
if [[ "${SPLIT}" != "train" && "${SPLIT}" != "test" ]]; then
    echo "ERROR: SPLIT must be train or test" >&2
    exit 1
fi
if [[ "${SAMPLER}" != "ddpm" && "${SAMPLER}" != "ddim" ]]; then
    echo "ERROR: SAMPLER must be ddpm or ddim" >&2
    exit 1
fi

export REPO_ROOT
export MODEL_PATH
export SELECTOR_PATH
export GSM8K_PATH
export OUTPUT_PATH
export SPLIT
export LIMIT
export SAMPLING_TIMESTEPS
export SAMPLER
export CLS_FREE_GUIDANCE
export SELECTOR_GUIDANCE
export NUM_PLAN_BRANCHES
export REPULSION_SCALE
export MAX_NEW_TOKENS
export DO_SAMPLE
export TOP_P
export REPETITION_PENALTY
export SEED
export EXTRA_ARGS
export CUDA_DEVICE

mkdir -p "${LOG_DIR}"

sbatch_args=(
    "--job-name=evaluate-gsm8k"
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

echo "Submitting GSM8K generation evaluation..."
echo "  model_path: ${MODEL_PATH}"
echo "  selector_path: ${SELECTOR_PATH}"
echo "  gsm8k_path: ${GSM8K_PATH}"
echo "  output_path: ${OUTPUT_PATH}"
echo "  split: ${SPLIT}"
echo "  limit: ${LIMIT:-none}"
echo "  sampler: ${SAMPLER}"
echo "  sampling_timesteps: ${SAMPLING_TIMESTEPS}"
echo "  selector_guidance: ${SELECTOR_GUIDANCE}"
echo "  num_plan_branches: ${NUM_PLAN_BRANCHES}"
echo "  extra_args: ${EXTRA_ARGS:-none}"
sbatch "${sbatch_args[@]}" "${SCRIPT_DIR}/run_evaluate_gsm8k.sbatch"
