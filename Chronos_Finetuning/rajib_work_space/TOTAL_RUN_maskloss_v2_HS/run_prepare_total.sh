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
# TEST-ONLY by default: their one file goes entirely to test, nothing to train. They are
# absent from the train pool and therefore invisible to the sampler's level 1.
# USE_SYN_DATA=1 changes that — see below.
#
# Real *test.csv files are discovered at the dataset's TOP LEVEL only. The previous
# recursive glob also swept in Exathlon/data_not_used/ (three byte-identical copies of
# real Exathlon files, absent from upstream PLAN-Lab/mTSBench), which put
# Exathlon_1_5_1000000_86 in the train half AND the test half at once. Dropping it
# re-seeds Exathlon's file split, so its test half moves and can no longer be linked from
# the v2 prep — prepare_total.py detects the mismatch and re-carves Exathlon rather than
# linking a stale test half. Exathlon's forward numbers are therefore not comparable with
# the arms that ran before this fix; every other dataset is untouched.
#
# Usage:
#   ./run_prepare_total.sh
#   USE_SYN_DATA=1 ./run_prepare_total.sh                    # + synthetic data into TRAIN
#   LINK_TEST_FROM="" ./run_prepare_total.sh                 # re-carve the test halves
#   DATASETS="SMD MSL SMAP" ./run_prepare_total.sh           # subset
#
set -euo pipefail
cd "$(dirname "$0")"

DATA_ROOT="${DATA_ROOT:-/home/rajib/mTSBench/Datasets/mTSBench}"
OUTPUT_DIR="${OUTPUT_DIR:-./prepared_total}"

# ── Synthetic data — TRAIN ONLY ──────────────────────────────────────────────
# Synthetic *test.csv files live in <DATA_ROOT>/<DATASET>/<SYN_DIR>/. With USE_SYN_DATA=1
# they are appended to that dataset's TRAIN half and go NOWHERE else: never the test half
# (evaluation must stay real), never the val set (that is the dataset's own *val.csv).
#
# The point is the 7 datasets that ship a single *test.csv and are therefore untrainable:
# synthetic files give them a train half while their one real *test.csv remains their
# whole, unchanged test half. So LINK_TEST_FROM and forward.py stay valid, but the train
# pool grows 12 -> 19 datasets and level 1 draws each 5.3% of the time instead of 8.3%.
# Those datasets also start contributing val windows from their *val.csv, so
# val_model_inputs.pkl changes and eval_loss is NOT comparable across this flag.
#
# Default 0 — the old experiment reproduces byte-for-byte (Exathlon aside).
USE_SYN_DATA="${USE_SYN_DATA:-1}"         # set 1 to use 
SYN_DIR="${SYN_DIR:-syn_data}"

# ── Per-file dissimilarity weights ───────────────────────────────────────────
# Always computed and written (per_dataset/<DS>/train_files.json). They are only USED when
# run_finetune_total.sh sets FILE_DIVERSITY_DATASETS, but writing them unconditionally
# means that flag can be flipped without re-prepping.
#
# Within a dataset, many *test.csv files are near duplicates (SVDB: 39 ECG records) and
# level 3's count weighting is blind to which file a window came from, so the longest files
# dominate and rare recordings are drowned out. Each file gets a 22-number signature
# (per-channel shape stats pooled across channels + its anomaly profile), and the weight is
# an inverse kernel density over those: a file in a tight cluster of near-duplicates is
# down-weighted, a lone unusual file is up-weighted.
FILE_WEIGHT_ALPHA="${FILE_WEIGHT_ALPHA:-1.0}"   # 0 = uniform over files (ablation baseline)
FILE_WEIGHT_CAP="${FILE_WEIGHT_CAP:-5.0}"       # no file above 5x / below 1/5x uniform

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
NO_VAL="${NO_VAL:-1}"                 # 1 = no val set (finetune: NO_VALIDATION=1)
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
if [ "${USE_SYN_DATA}" = "1" ]; then
echo "Synthetic data    : ON  — <DATASET>/${SYN_DIR}/*test.csv appended to TRAIN only"
echo "                    (never test, never val; the 7 one-file datasets join the pool)"
else
echo "Synthetic data    : off (USE_SYN_DATA=1 to fold ${SYN_DIR}/ into the train halves)"
fi
echo "File weights      : alpha=${FILE_WEIGHT_ALPHA} cap=${FILE_WEIGHT_CAP}x  (inverse kernel density;"
echo "                    used only if run_finetune_total.sh sets FILE_DIVERSITY_DATASETS)"
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
    --syn_dir             "${SYN_DIR}"
    --file_weight_alpha   "${FILE_WEIGHT_ALPHA}"
    --file_weight_cap     "${FILE_WEIGHT_CAP}"
)
[ "${USE_SYN_DATA}" = "1" ] && ARGS+=(--use_syn_data)
[ "${NO_VAL}" = "1" ] && ARGS+=(--no_val)
[ "${VAL_ONLY:-0}" = "1" ] && ARGS+=(--val_only)
[ -n "${LINK_TEST_FROM}" ] && ARGS+=(--link_test_from "${LINK_TEST_FROM}")
[ -n "${DATASETS:-}" ] && ARGS+=(--datasets ${DATASETS})

python -u prepare_total.py "${ARGS[@]}"

echo
echo "Done. Per-dataset train -> ${OUTPUT_DIR}/per_dataset/<DATASET>/train_model_inputs.pkl"
echo "      Per-dataset test  -> ${OUTPUT_DIR}/per_dataset/<DATASET>/test_model_inputs.pkl"
echo "      Per-file weights  -> ${OUTPUT_DIR}/per_dataset/<DATASET>/train_files.json"
echo "                           (finetune: FILE_DIVERSITY_DATASETS=\"MITDB SVDB\" to use them)"
[ "${NO_VAL}" = "1" ] || \
echo "      Val set           -> ${OUTPUT_DIR}/val_model_inputs.pkl   (finetune: NO_VALIDATION=0)"
