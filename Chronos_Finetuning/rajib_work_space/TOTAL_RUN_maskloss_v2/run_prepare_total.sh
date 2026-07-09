#!/usr/bin/env bash
#
# Whole-mTSBench data prep — DATASET-BALANCED, THRESHOLDLESS (for the sampler).
#
#   * Combined train_model_inputs.pkl = each dataset's train-half windows, CAPPED
#     to PER_DATASET_CAP so no dataset (MITDB/SVDB are 88% of the raw pool) dominates.
#     Both classes are guaranteed per dataset (anomalies <= MAX_ANOM_FRAC of the cap).
#     NO class balancing here — the count-weighted sampler does that at runtime.
#   * Per-dataset TEST sets under prepared_total/per_dataset/<DATASET>/ (UNCAPPED)
#     for run_forward_total.sh (one dataset at a time).
#
# Single-file datasets (CalIt2, creditcard, GECCO, Genesis, metro, PSM, swan) are
# TEST-ONLY: their one file goes entirely to test, nothing to train.
#
# Usage:
#   ./run_prepare_total.sh
#   PER_DATASET_CAP=3000 ./run_prepare_total.sh              # smaller/faster pool
#   DATASETS="SMD MSL SMAP" ./run_prepare_total.sh           # subset
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

# Dataset de-domination — the ONLY balancing done at prep. Class balance is the
# sampler's job (SAMPLING_TARGET in run_finetune_total.sh).
PER_DATASET_CAP="${PER_DATASET_CAP:-5000}"   # max train windows per dataset (<=0 disables)
MAX_ANOM_FRAC="${MAX_ANOM_FRAC:-0.5}"        # anomalies take <= this fraction of each cap

# Optional EVAL_VAL monitoring probe (a COPY of a train subset). 0 = off (default;
# we run NO_VALIDATION=1). EVAL_TEST stays MANUAL — pass TEST_DATA to run_finetune_total.sh.
VAL_FRACTION="${VAL_FRACTION:-0}"            # e.g. 0.02 to write val_model_inputs.pkl
VAL_MAX="${VAL_MAX:-5000}"

echo "Data root         : ${DATA_ROOT}"
echo "Output dir        : ${OUTPUT_DIR}"
echo "Test fraction     : ${TEST_FRACTION}  (file-based, per dataset)"
echo "Context/Pred      : ${CONTEXT_LENGTH} / ${PREDICTION_LENGTH}"
echo "Stride train/test : ${STRIDE} / ${TEST_STRIDE}"
echo "Per-dataset cap   : ${PER_DATASET_CAP}  (anomalies <= ${MAX_ANOM_FRAC} of cap; both classes kept)"
echo "Balancing         : DATASET only (class balance = runtime sampler)"
echo "Val probe         : ${VAL_FRACTION}  (0 = no val pkl; EVAL_TEST stays manual via TEST_DATA)"
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
    --per_dataset_cap     "${PER_DATASET_CAP}"
    --max_anom_frac       "${MAX_ANOM_FRAC}"
    --val_fraction        "${VAL_FRACTION}"
    --val_max             "${VAL_MAX}"
)
[ -n "${DATASETS:-}" ] && ARGS+=(--datasets ${DATASETS})

python -u prepare_total.py "${ARGS[@]}"

echo
echo "Done. Combined train -> ${OUTPUT_DIR}/train_model_inputs.pkl"
echo "      Per-dataset test -> ${OUTPUT_DIR}/per_dataset/<DATASET>/"
