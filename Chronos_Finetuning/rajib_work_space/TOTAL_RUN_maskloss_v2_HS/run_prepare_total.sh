#!/usr/bin/env bash
#
# Whole-mTSBench data prep — UNCAPPED, THRESHOLDLESS, PER-DATASET (for the HS sampler).
#
#   * Per-dataset train pools under prepared_total/per_dataset/<DATASET>/train_model_inputs.pkl
#     Every train-half window of every dataset is kept: no cap, no threshold, no class
#     balancing. Both imbalances (dataset and class) are handled at TRAIN time by the
#     hierarchical sampler in finetune_anomaly_simple.py.
#   * Per-dataset TEST sets under prepared_total/per_dataset/<DATASET>/ (UNCAPPED)
#     for run_forward_total.sh (one dataset at a time).
#
# The test halves are a pure function of (files, geometry, seed), and the file split is
# seeded identically to TOTAL_RUN / TOTAL_RUN_maskloss_v2, so by default we SYMLINK them
# from the v2 run rather than re-carving 6.3 GB. That makes "inference is identical across
# all arms" a fact rather than a claim, and halves prep time. Set LINK_TEST_FROM="" to
# carve them here instead (the bytes will match either way).
#
# Single-file datasets (CalIt2, creditcard, GECCO, Genesis, metro, PSM, swan) are
# TEST-ONLY: their one file goes entirely to test, nothing to train. They are absent from
# the train pool and therefore invisible to the sampler's level 1.
#
# Usage:
#   ./run_prepare_total.sh
#   LINK_TEST_FROM="" ./run_prepare_total.sh                 # re-carve the test halves
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

# Reuse the byte-identical test halves from the 2x arm. Set to "" to re-carve.
LINK_TEST_FROM="${LINK_TEST_FROM-../TOTAL_RUN_maskloss_v2/prepared_total}"

# Reporting only: the manifest records the anomaly-step fraction each dataset will
# yield under the sampler. The sampler reads its own P_ANOM in run_finetune_total.sh.
P_ANOM="${P_ANOM:-0.3333333333333333}"
MIN_ANOM_WINDOWS="${MIN_ANOM_WINDOWS:-50}"   # warn below this; hard-fail only at zero

# Validation set — carved from mTSBench's own *val.csv (one per dataset, disjoint from
# the *test.csv files split into our train/test halves), HIERARCHICALLY: an equal budget
# for every trained dataset (level 1), split VAL_P_ANOM / 1-VAL_P_ANOM across anomalous
# and normal kinds (level 2), n_anom-weighted inside the anomalous kind (level 3).
# Fixed seed -> a deterministic val_model_inputs.pkl. NO_VAL=1 restores the old behaviour.
# VAL_ONLY=1 adds the val set to an existing prepared_total without re-carving train/test.
NO_VAL="${NO_VAL:-0}"
VAL_PER_DATASET="${VAL_PER_DATASET:-200}"    # windows per dataset; short files give all they have
VAL_STRIDE="${VAL_STRIDE:-16}"               # < STRIDE: the small val files can't fill 200 at 64
VAL_P_ANOM="${VAL_P_ANOM:-0.3333333333333333}"   # match P_ANOM: eval the mix we train on

echo "Data root         : ${DATA_ROOT}"
echo "Output dir        : ${OUTPUT_DIR}"
echo "Test fraction     : ${TEST_FRACTION}  (file-based, per dataset)"
echo "Context/Pred      : ${CONTEXT_LENGTH} / ${PREDICTION_LENGTH}"
echo "Stride train/test : ${STRIDE} / ${TEST_STRIDE}"
echo "Cap / threshold   : NONE / NONE  (every train window kept)"
echo "Balancing         : none at prep — dataset AND class balance are the HS sampler's job"
echo "Test halves       : ${LINK_TEST_FROM:-carved here}"
if [ "${NO_VAL}" = "1" ]; then
echo "Val set           : none (run finetune with NO_VALIDATION=1; EVAL_TEST stays manual)"
else
echo "Val set           : mTSBench *val.csv, hierarchical — ${VAL_PER_DATASET} windows/dataset"
echo "                    (stride ${VAL_STRIDE}, target anomalous share ${VAL_P_ANOM})"
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
    --test_fraction       "${TEST_FRACTION}"
    --seed                "${SEED}"
    --p_anom              "${P_ANOM}"
    --min_anom_windows    "${MIN_ANOM_WINDOWS}"
    --val_per_dataset     "${VAL_PER_DATASET}"
    --val_stride          "${VAL_STRIDE}"
    --val_p_anom          "${VAL_P_ANOM}"
)
[ "${NO_VAL}" = "1" ] && ARGS+=(--no_val)
[ "${VAL_ONLY:-0}" = "1" ] && ARGS+=(--val_only)
[ -n "${LINK_TEST_FROM}" ] && ARGS+=(--link_test_from "${LINK_TEST_FROM}")
[ -n "${DATASETS:-}" ] && ARGS+=(--datasets ${DATASETS})

python -u prepare_total.py "${ARGS[@]}"

echo
echo "Done. Per-dataset train -> ${OUTPUT_DIR}/per_dataset/<DATASET>/train_model_inputs.pkl"
echo "      Per-dataset test  -> ${OUTPUT_DIR}/per_dataset/<DATASET>/test_model_inputs.pkl"
[ "${NO_VAL}" = "1" ] || \
echo "      Val set           -> ${OUTPUT_DIR}/val_model_inputs.pkl   (finetune: NO_VALIDATION=0)"
