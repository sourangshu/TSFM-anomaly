#!/usr/bin/env bash
#
# Data prep for the FAMILY transfer study.
#
# Carves every dataset named in families.json ONCE, whole, into a shared pool/, then
# assembles run/prepared as symlinks into it. Nothing is duplicated on disk.
#
# What makes this different from run_prepare_total.sh: there is NO within-dataset file
# split. Each dataset is used entirely as TRAIN or entirely as TEST, so every held-out
# dataset contributes zero training windows and its test set is 100% of its files.
#
# Consequence: the test halves CANNOT be symlinked from TOTAL_RUN_maskloss_v2 (those
# are 50% halves). They are carved fresh here. This also means the existing results_ZS
# is NOT a valid baseline for this study -- zero-shot must be re-run on these 100% test
# sets. run_study.sh does that for you.
#
# Usage:
#   ./run_prepare_family.sh
#   ONLY="ecg server" ./run_prepare_family.sh      # subset of families
#   PER_FAMILY=1 ./run_prepare_family.sh           # + per-family dirs (attribution variant)
#   VAL_ONLY=1 ./run_prepare_family.sh             # add the val set to an existing pool
#                                                  # (train/test pkls left untouched)
#
set -euo pipefail
cd "$(dirname "$0")"

DATA_ROOT="${DATA_ROOT:-/home/rajib/mTSBench/Datasets/mTSBench}"
OUTPUT_DIR="${OUTPUT_DIR:-./pool}"

CONTEXT_LENGTH="${CONTEXT_LENGTH:-512}"
PREDICTION_LENGTH="${PREDICTION_LENGTH:-64}"
STRIDE="${STRIDE:-64}"
TEST_STRIDE="${TEST_STRIDE:-64}"      # MUST equal PREDICTION_LENGTH (contiguous tiling)
MIN_LENGTH="${MIN_LENGTH:-50}"

# Reporting only -- the sampler reads its own P_ANOM in run_finetune_family.sh.
P_ANOM="${P_ANOM:-0.3333333333333333}"
MIN_ANOM_WINDOWS="${MIN_ANOM_WINDOWS:-50}"
MIN_ANOM_STEPS_TEST="${MIN_ANOM_STEPS_TEST:-500}"

# Validation — from mTSBench's own *val.csv (this study otherwise reads *test.csv only),
# carved for the TRAIN-role datasets ONLY. A held-out dataset gets no val windows: eval_loss
# selects the checkpoint, so validating on a held-out dataset would leak it into the very
# transfer number the study reports. Within the train pool the carve is hierarchical: an
# equal budget per dataset (level 1), a VAL_P_ANOM anomalous share (level 2), n_anom-weighted
# inside the anomalous kind (level 3). NO_VAL=1 restores the old no-validation behaviour.
NO_VAL="${NO_VAL:-0}"
VAL_PER_DATASET="${VAL_PER_DATASET:-200}"
VAL_STRIDE="${VAL_STRIDE:-16}"        # < STRIDE: small val files can't fill 200 at 64
VAL_P_ANOM="${VAL_P_ANOM:-0.3333333333333333}"   # match P_ANOM: eval the mix we train on
SEED="${SEED:-42}"

echo "Data root      : ${DATA_ROOT}"
echo "Pool dir       : ${OUTPUT_DIR}"
echo "Split          : NONE -- whole datasets, 100% of *test.csv, train XOR test per fold"
echo "Context/Pred   : ${CONTEXT_LENGTH} / ${PREDICTION_LENGTH}"
echo "Stride tr/te   : ${STRIDE} / ${TEST_STRIDE}"
echo "Balancing      : none at prep -- the HS sampler handles it at train time"
if [ "${NO_VAL}" = "1" ]; then
echo "Val set        : none (fine-tune with NO_VALIDATION=1)"
else
echo "Val set        : mTSBench *val.csv, hierarchical -- ${VAL_PER_DATASET} windows per"
echo "                 TRAIN dataset (stride ${VAL_STRIDE}, anomalous target ${VAL_P_ANOM});"
echo "                 held-out datasets contribute NOTHING"
fi
echo

ARGS=(
    --data_root           "${DATA_ROOT}"
    --output_dir          "${OUTPUT_DIR}"
    --context_length      "${CONTEXT_LENGTH}"
    --prediction_length   "${PREDICTION_LENGTH}"
    --stride              "${STRIDE}"
    --test_stride         "${TEST_STRIDE}"
    --min_length          "${MIN_LENGTH}"
    --p_anom              "${P_ANOM}"
    --min_anom_windows    "${MIN_ANOM_WINDOWS}"
    --min_anom_steps_test "${MIN_ANOM_STEPS_TEST}"
    --val_per_dataset     "${VAL_PER_DATASET}"
    --val_stride          "${VAL_STRIDE}"
    --val_p_anom          "${VAL_P_ANOM}"
    --seed                "${SEED}"
)
[ "${NO_VAL}" = "1" ] && ARGS+=(--no_val)
[ "${VAL_ONLY:-0}" = "1" ] && ARGS+=(--val_only)
[ -n "${ONLY:-}" ] && ARGS+=(--only ${ONLY})

python -u prepare_family.py "${ARGS[@]}"

echo
MK=(--seed "${SEED}"); [ "${PER_FAMILY:-0}" = "1" ] && MK+=(--per_family)
python -u make_folds.py "${MK[@]}"
