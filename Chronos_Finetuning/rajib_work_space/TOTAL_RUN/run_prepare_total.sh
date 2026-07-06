#!/usr/bin/env bash
#
# Whole-mTSBench data preparation.
#   * Combined train_model_inputs.pkl pooled from the TRAIN half (50% file-based
#     split) of every multi-file dataset, class-balanced + shuffled.
#   * Per-dataset test set under prepared_total/per_dataset/<DATASET>/ for the
#     forward.py / run_forward_total.sh evaluation (one dataset at a time).
#
# Single-file datasets (CalIt2, creditcard, GECCO, Genesis, metro, PSM, swan) are
# TEST-ONLY: their one file goes entirely to test, nothing to train.
#
# Usage:
#   ./run_prepare_total.sh
#   BALANCE_SCOPE=global NORMAL_RATIO=1.0 ./run_prepare_total.sh
#   DATASETS="SMD MSL SMAP" ./run_prepare_total.sh          # subset
#   NORMAL_RATIO=3.0 PER_DATASET_ANOM_CAP=2000 ./run_prepare_total.sh
#
set -euo pipefail
cd "$(dirname "$0")"

DATA_ROOT="${DATA_ROOT:-/home/rajib/mTSBench/Datasets/mTSBench}"
OUTPUT_DIR="${OUTPUT_DIR:-./prepared_total}"

CONTEXT_LENGTH="${CONTEXT_LENGTH:-512}"
PREDICTION_LENGTH="${PREDICTION_LENGTH:-64}"
STRIDE="${STRIDE:-64}"
TEST_STRIDE="${TEST_STRIDE:-64}"
MIN_LENGTH="${MIN_LENGTH:-50}"

TEST_FRACTION="${TEST_FRACTION:-0.5}"
SEED="${SEED:-42}"

ANOMALY_THRESHOLD="${ANOMALY_THRESHOLD:-10}"
BALANCE_SCOPE="${BALANCE_SCOPE:-per_dataset_cap}"   # global | per_dataset | per_dataset_cap
NORMAL_RATIO="${NORMAL_RATIO:-2.0}"
PER_DATASET_ANOM_CAP="${PER_DATASET_ANOM_CAP:-median}"
NORMAL_RESERVOIR_CAP="${NORMAL_RESERVOIR_CAP:-200000}"
PURE_NORMAL_ONLY="${PURE_NORMAL_ONLY:-0}"

# Validation PROBE: a COPY of a subset of train (no data removed) to track training
# loss. 0 disables. e.g. VAL_FRACTION=0.02 keeps a 2% probe (capped at VAL_MAX).
VAL_FRACTION="${VAL_FRACTION:-0.02}"
VAL_MAX="${VAL_MAX:-5000}"

echo "Data root         : ${DATA_ROOT}"
echo "Output dir        : ${OUTPUT_DIR}"
echo "Test fraction     : ${TEST_FRACTION}  (file-based, per dataset)"
echo "Context/Pred      : ${CONTEXT_LENGTH} / ${PREDICTION_LENGTH}"
echo "Stride train/test : ${STRIDE} / ${TEST_STRIDE}"
echo "Balance scope     : ${BALANCE_SCOPE}  (normal_ratio=${NORMAL_RATIO}, anom_cap=${PER_DATASET_ANOM_CAP})"
echo "Anomaly threshold : ${ANOMALY_THRESHOLD}"
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
    --balance_scope       "${BALANCE_SCOPE}"
    --normal_ratio        "${NORMAL_RATIO}"
    --per_dataset_anom_cap "${PER_DATASET_ANOM_CAP}"
    --normal_reservoir_cap "${NORMAL_RESERVOIR_CAP}"
    --val_fraction        "${VAL_FRACTION}"
    --val_max             "${VAL_MAX}"
)
[ -n "${DATASETS:-}" ] && ARGS+=(--datasets ${DATASETS})
[ "${PURE_NORMAL_ONLY}" = "1" ] && ARGS+=(--pure_normal_only)

python -u prepare_total.py "${ARGS[@]}"

echo
echo "Done. Combined train -> ${OUTPUT_DIR}/train_model_inputs.pkl"
echo "      Per-dataset test -> ${OUTPUT_DIR}/per_dataset/<DATASET>/"
