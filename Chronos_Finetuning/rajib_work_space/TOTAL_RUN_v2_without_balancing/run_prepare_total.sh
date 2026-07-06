#!/usr/bin/env bash
#
# Whole-mTSBench data preparation — v2 (SMD strategy, NO data balancing).
#
#   * Combined train_model_inputs.pkl = EVERY sliding window from the TRAIN half
#     (50% file-based split) of every multi-file dataset, pooled + shuffled.
#     No class balancing, no reservoir, no per-dataset window cap — we train on
#     the entirety of the *test.csv train half, exactly like SMD_run did.
#   * Per-dataset test set under prepared_total/per_dataset/<DATASET>/ for the
#     forward.py / run_forward_total.sh evaluation (one dataset at a time).
#
# Single-file datasets (CalIt2, creditcard, GECCO, Genesis, metro, PSM, swan) are
# TEST-ONLY: their one file goes entirely to test, nothing to train.
#
# Usage:
#   ./run_prepare_total.sh
#   DATASETS="SMD MSL SMAP" ./run_prepare_total.sh          # subset
#   TEST_FRACTION=0.5 STRIDE=64 ./run_prepare_total.sh
#
set -euo pipefail
cd "$(dirname "$0")"

DATA_ROOT="${DATA_ROOT:-/home/rajib/mTSBench/Datasets/mTSBench}"
OUTPUT_DIR="${OUTPUT_DIR:-./prepared_total}"

CONTEXT_LENGTH="${CONTEXT_LENGTH:-512}"
PREDICTION_LENGTH="${PREDICTION_LENGTH:-64}"
STRIDE="${STRIDE:-64}"
TEST_STRIDE="${TEST_STRIDE:-64}"      # MUST equal PREDICTION_LENGTH (contiguous test tiling)
MIN_LENGTH="${MIN_LENGTH:-50}"

TEST_FRACTION="${TEST_FRACTION:-0.5}"
SEED="${SEED:-42}"

# Used ONLY for logging the anomalous/normal split of the (unbalanced) train set;
# must match --anomaly_threshold in run_finetune_total.sh.
ANOMALY_THRESHOLD="${ANOMALY_THRESHOLD:-10}"

echo "Data root         : ${DATA_ROOT}"
echo "Output dir        : ${OUTPUT_DIR}"
echo "Test fraction     : ${TEST_FRACTION}  (file-based, per dataset)"
echo "Context/Pred      : ${CONTEXT_LENGTH} / ${PREDICTION_LENGTH}"
echo "Stride train/test : ${STRIDE} / ${TEST_STRIDE}"
echo "Balancing         : NONE (every train-half window kept)"
echo "Anomaly threshold : ${ANOMALY_THRESHOLD}  (stats only)"
echo

ARGS=(
    --data_root           "${DATA_ROOT}"
    --output_dir          "${OUTPUT_DIR}"
    --context_length      "${CONTEXT_LENGTH}"
    --prediction_length   "${PREDICTION_LENGTH}"
    --stride              "${STRIDE}"
    --test_stride         "${TEST_STRIDE}"
    --min_length          "${MIN_LENGTH}"
    --test_fraction       "${TEST_FRACTION}"
    --seed                "${SEED}"
    --anomaly_threshold   "${ANOMALY_THRESHOLD}"
)
[ -n "${DATASETS:-}" ] && ARGS+=(--datasets ${DATASETS})

python -u prepare_total.py "${ARGS[@]}"

echo
echo "Done. Combined train -> ${OUTPUT_DIR}/train_model_inputs.pkl"
echo "      Per-dataset test -> ${OUTPUT_DIR}/per_dataset/<DATASET>/"
