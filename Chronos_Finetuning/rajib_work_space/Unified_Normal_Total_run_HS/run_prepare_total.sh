#!/usr/bin/env bash
#
# Whole-mTSBench data prep — UNCAPPED, THRESHOLDLESS, PER-DATASET, with a
# UNIFIED (global) normal prefix per dataset (for the HS sampler).
#
#   * Per-dataset train pools under prepared_total/per_dataset/<DATASET>/train_model_inputs.pkl
#     Every train-half window of every dataset is kept: no cap, no threshold, no class
#     balancing. Both imbalances (dataset and class) are handled at TRAIN time by the
#     hierarchical sampler in finetune_anomaly_simple.py.
#   * ONE unified normal prefix per dataset: the medoid of that dataset's TRAINING
#     files (phase-robust FFT similarity), z-normalized, re-scaled into each series'
#     own per-channel normal-zone units. Saved as per_dataset/<DS>/global_normal_signal.npz.
#   * Per-dataset TEST sets under prepared_total/per_dataset/<DATASET>/ (UNCAPPED)
#     for run_forward_total.sh. Test series receive the TRAINING medoid re-scaled by
#     their own normal-zone stats (no leakage into the shared reference).
#
# NOTE: unlike TOTAL_RUN_maskloss_v2_HS, the test pkls of TRAINED datasets can NOT be
# symlinked from another arm — their 256-step normal prefix differs. Only the 7
# single-file TEST-ONLY datasets (CalIt2, creditcard, GECCO, Genesis, metro, PSM, swan)
# keep the per-series prefix (byte-identical to the other arms); LINK_TEST_FROM, if
# set, symlinks just those. Default is empty = carve everything here (standalone).
#
# Usage:
#   ./run_prepare_total.sh
#   DATASETS="SMD MSL SMAP" ./run_prepare_total.sh                    # subset
#   LINK_TEST_FROM=../TOTAL_RUN_maskloss_v2_HS/prepared_total ./run_prepare_total.sh
#
set -euo pipefail
cd "$(dirname "$0")"

# Server path; on WSL use e.g.
# DATA_ROOT="/mnt/c/Files/MTP Code Local Files/MTP_SEM_3_LOCAL_FILES/mTSBench"
DATA_ROOT="${DATA_ROOT:-/home/rajib/mTSBench/Datasets/mTSBench}"
OUTPUT_DIR="${OUTPUT_DIR:-./prepared_total}"

CONTEXT_LENGTH="${CONTEXT_LENGTH:-512}"
PREDICTION_LENGTH="${PREDICTION_LENGTH:-64}"
STRIDE="${STRIDE:-64}"
TEST_STRIDE="${TEST_STRIDE:-64}"      # MUST equal PREDICTION_LENGTH (contiguous test tiling)
MIN_LENGTH="${MIN_LENGTH:-50}"

TEST_FRACTION="${TEST_FRACTION:-0.5}"
SEED="${SEED:-42}"

# Unified normal: similarity metric for per-dataset medoid selection
METRIC="${METRIC:-fft}"

# Optional: symlink the TEST-ONLY datasets' test pkls from an existing prepared_total
# (their per-series-prefix bytes are identical across arms). Trained datasets are
# ALWAYS carved here. Default empty = fully standalone.
LINK_TEST_FROM="${LINK_TEST_FROM:-}"

# Reporting only: the manifest records the anomaly-step fraction each dataset will
# yield under the sampler. The sampler reads its own P_ANOM in run_finetune_total.sh.
P_ANOM="${P_ANOM:-0.3333333333333333}"
MIN_ANOM_WINDOWS="${MIN_ANOM_WINDOWS:-50}"   # warn below this; hard-fail only at zero

echo "Data root         : ${DATA_ROOT}"
echo "Output dir        : ${OUTPUT_DIR}"
echo "Test fraction     : ${TEST_FRACTION}  (file-based, per dataset)"
echo "Context/Pred      : ${CONTEXT_LENGTH} / ${PREDICTION_LENGTH}"
echo "Stride train/test : ${STRIDE} / ${TEST_STRIDE}"
echo "Cap / threshold   : NONE / NONE  (every train window kept)"
echo "Balancing         : none at prep — dataset AND class balance are the HS sampler's job"
echo "Normal prefix     : UNIFIED per dataset (training-file medoid, metric=${METRIC});"
echo "                    per-series for test-only datasets"
echo "Test halves       : ${LINK_TEST_FROM:+test-only linked from ${LINK_TEST_FROM}; }carved here"
echo "Val probe         : none (run finetune with NO_VALIDATION=1; EVAL_TEST stays manual)"
echo

ARGS=(
    --data_root           "${DATA_ROOT}"
    --output_dir          "${OUTPUT_DIR}"
    --metric              "${METRIC}"
    --context_length      "${CONTEXT_LENGTH}"
    --prediction_length   "${PREDICTION_LENGTH}"
    --stride              "${STRIDE}"
    --test_stride         "${TEST_STRIDE}"
    --min_length          "${MIN_LENGTH}"
    --test_fraction       "${TEST_FRACTION}"
    --seed                "${SEED}"
    --p_anom              "${P_ANOM}"
    --min_anom_windows    "${MIN_ANOM_WINDOWS}"
)
[ -n "${LINK_TEST_FROM}" ] && ARGS+=(--link_test_from "${LINK_TEST_FROM}")
[ -n "${DATASETS:-}" ] && ARGS+=(--datasets ${DATASETS})

python -u prepare_total.py "${ARGS[@]}"

echo
echo "Done. Per-dataset train  -> ${OUTPUT_DIR}/per_dataset/<DATASET>/train_model_inputs.pkl"
echo "      Per-dataset test   -> ${OUTPUT_DIR}/per_dataset/<DATASET>/test_model_inputs.pkl"
echo "      Per-dataset medoid -> ${OUTPUT_DIR}/per_dataset/<DATASET>/global_normal_signal.npz"
