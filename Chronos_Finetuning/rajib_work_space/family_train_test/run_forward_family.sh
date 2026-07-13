#!/usr/bin/env bash
#
# Score a model on EVERY held-out dataset (100% of each one's *test.csv files) and report
# VUS-PR per dataset. forward.py is byte-identical to the copy in TOTAL_RUN_maskloss_v2_HS,
# so the scoring procedure is the same as every other arm -- only the test SET differs
# (100% of a dataset the model never saw, not a 50% half of one it trained on).
#
# The held-out datasets are discovered automatically: run/prepared/per_dataset holds a
# test_model_inputs.pkl for exactly the 6 held-out datasets and none of the 9 training
# ones, so there is nothing to name and nothing to get wrong.
#
# Usage:
#   bash run_forward_family.sh              # the fine-tuned pooled model  (the experiment)
#   ZERO_SHOT=1 bash run_forward_family.sh  # the base model               (the baseline)
#   DATASETS="SVDB MSL" bash run_forward_family.sh    # subset
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FORWARD_PY="${SCRIPT_DIR}/forward.py"

RUN="${RUN:-run}"
ZERO_SHOT="${ZERO_SHOT:-0}"

PREPARED_DIR="${PREPARED_DIR:-${SCRIPT_DIR}/${RUN}/prepared}"
PER_DATASET_DIR="${PREPARED_DIR}/per_dataset"

if [ ! -d "${PER_DATASET_DIR}" ]; then
    echo "ERROR: ${PER_DATASET_DIR} not found. Run ./run_prepare_family.sh first."; exit 1
fi

if [ "${ZERO_SHOT}" = "1" ]; then
    CHECKPOINT=""
    MODEL_TAG="zeroshot"
else
    CHECKPOINT="${CHECKPOINT:-${SCRIPT_DIR}/${RUN}/ckpt/finetuned-ckpt}"
    MODEL_TAG="family_ft"
    if [ ! -d "${CHECKPOINT}" ]; then
        echo "ERROR: no checkpoint at ${CHECKPOINT}"
        echo "       Train it first:  bash run_finetune_family.sh"
        exit 1
    fi
fi

RESULTS_DIR="${RESULTS_DIR:-${SCRIPT_DIR}/results/${MODEL_TAG}}"
mkdir -p "${RESULTS_DIR}"

MODEL_ID="${MODEL_ID:-amazon/chronos-2}"
DEVICE="${DEVICE:-cuda}"

# Sequence layout -- must match prep and fine-tuning
NORMAL_SIGNAL_LENGTH="${NORMAL_SIGNAL_LENGTH:-256}"
CONTEXT_LENGTH="${CONTEXT_LENGTH:-512}"
PREDICTION_LENGTH="${PREDICTION_LENGTH:-64}"
INPUT_PATCH_SIZE="${INPUT_PATCH_SIZE:-16}"

# Mode-aware normal-prefix default: the fine-tuned model was trained WITH the [SEP]-
# separated normal prefix; the base model has never seen one.
if [ -n "${CHECKPOINT}" ]; then _DEFAULT_PREFIX=1; else _DEFAULT_PREFIX=0; fi
USE_NORMAL_PREFIX="${USE_NORMAL_PREFIX:-${_DEFAULT_PREFIX}}"

BATCH_WINDOWS="${BATCH_WINDOWS:-32}"
PREDICT_BATCH_SIZE="${PREDICT_BATCH_SIZE:-160}"
SCORE_METHOD="${SCORE_METHOD:-mse}"
AGG_METHOD="${AGG_METHOD:-l2}"
TOPK="${TOPK:-4}"
SMOOTH_WINDOW="${SMOOTH_WINDOW:-5}"
SLIDING_WINDOW_VUS="${SLIDING_WINDOW_VUS:-100}"
VUS_VERSION="${VUS_VERSION:-opt}"
VUS_THRE="${VUS_THRE:-250}"

export PYTHONPATH="${SCRIPT_DIR}${PYTHONPATH:+:${PYTHONPATH}}"
find -L "$SCRIPT_DIR/chronos" -name '__pycache__' -type d -exec rm -rf {} + 2>/dev/null || true

# Held-out datasets = the folders carrying a test pkl. The 9 training datasets carry only
# a train pkl, so they cannot appear here.
if [ -n "${DATASETS:-}" ]; then
    DS_LIST=(${DATASETS})
else
    DS_LIST=()
    for d in "${PER_DATASET_DIR}"/*/; do
        if [ -f "${d}/test_model_inputs.pkl" ]; then DS_LIST+=("$(basename "$d")"); fi
    done
fi
if [ ${#DS_LIST[@]} -eq 0 ]; then
    echo "ERROR: no test_model_inputs.pkl under ${PER_DATASET_DIR}"; exit 1
fi

if [ -n "${CHECKPOINT}" ]; then
    MODE="FINE-TUNED on the pooled family-sibling datasets"
else
    MODE="ZERO-SHOT (${MODEL_ID}) -- the baseline"
fi

echo "======================================================"
echo "  Family transfer evaluation"
echo "  MODEL             = ${MODE}"
echo "  HELD-OUT datasets = ${DS_LIST[*]}"
echo "                      (100% of files; zero training windows from any of them)"
echo "  USE_NORMAL_PREFIX = ${USE_NORMAL_PREFIX}"
echo "  RESULTS_DIR       = ${RESULTS_DIR}"
echo "======================================================"

for ds in "${DS_LIST[@]}"; do
    TEST_PKL="${PER_DATASET_DIR}/${ds}/test_model_inputs.pkl"
    META_PKL="${PER_DATASET_DIR}/${ds}/test_series_meta.pkl"
    OUT_CSV="${RESULTS_DIR}/${ds}_results.csv"
    if [ ! -f "${TEST_PKL}" ]; then
        echo "  skip ${ds}: no test pkl"; continue
    fi
    echo ""
    echo "------ ${ds} ------"

    ARGS=(
        --test_pkl             "${TEST_PKL}"
        --meta_pkl             "${META_PKL}"
        --model_id             "${MODEL_ID}"
        --device               "${DEVICE}"
        --normal_signal_length "${NORMAL_SIGNAL_LENGTH}"
        --context_length       "${CONTEXT_LENGTH}"
        --prediction_length    "${PREDICTION_LENGTH}"
        --input_patch_size     "${INPUT_PATCH_SIZE}"
        --batch_windows        "${BATCH_WINDOWS}"
        --predict_batch_size   "${PREDICT_BATCH_SIZE}"
        --score_method         "${SCORE_METHOD}"
        --agg_method           "${AGG_METHOD}"
        --topk                 "${TOPK}"
        --smooth_window        "${SMOOTH_WINDOW}"
        --sliding_window_VUS   "${SLIDING_WINDOW_VUS}"
        --vus_version          "${VUS_VERSION}"
        --vus_thre             "${VUS_THRE}"
        --out_csv              "${OUT_CSV}"
    )
    [ -n "${CHECKPOINT}" ] && ARGS+=(--checkpoint "${CHECKPOINT}")
    [ "${USE_NORMAL_PREFIX}" = "0" ] && ARGS+=(--no_normal_prefix)

    python -u "${FORWARD_PY}" "${ARGS[@]}"
done

echo ""
echo "======================================================"
python -u "${SCRIPT_DIR}/aggregate_results.py" \
    --results_dir "${RESULTS_DIR}" \
    --out_summary "${RESULTS_DIR}/per_dataset_summary.csv"
echo "  Deliverable : ${RESULTS_DIR}/per_dataset_summary.csv"
echo "======================================================"
