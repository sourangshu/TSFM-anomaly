#!/usr/bin/env bash
#
# Evaluate Chronos-2 (zero-shot or a fine-tuned checkpoint) on the per-dataset
# mTSBench test sets produced by run_prepare_total.sh. Runs forward.py once per
# dataset (each dataset's per_dataset/<DS>/test_model_inputs.pkl + test_series_meta.pkl),
# writes results/<DS>_results.csv (one row per series), then calls
# aggregate_results.py to produce the DELIVERABLE: one VUS-PR value per dataset.
#
# forward.py is byte-identical to every other arm's copy (md5
# a226d1f5e899d7ae332112e3f29d076f). The test pkls differ from the other arms in
# EXACTLY ONE place: the 256-step normal prefix is the UNIFIED per-dataset medoid
# (re-scaled per series) instead of a per-series signal. Context, future and labels
# are byte-identical by construction. Test-only datasets keep per-series prefixes.
#
# Test sets are UNCAPPED — no arm's balancing ever touched TRAIN's test half, so
# evaluation sees every test window, exactly like run_forward_smd.sh did for SMD.
#
# Usage:
#   ./run_forward_total.sh                       # all datasets, fine-tuned ckpt
#   DATASETS="SMD MSL" ./run_forward_total.sh    # subset
#   CHECKPOINT="" ./run_forward_total.sh         # zero-shot baseline
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"   # .../rajib_work_space/Unified_Normal_Total_run_HS
WORK_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"                  # .../rajib_work_space
FORWARD_PY="${SCRIPT_DIR}/forward.py"                        # local copy (self-contained)

PREPARED_DIR="${PREPARED_DIR:-${SCRIPT_DIR}/prepared_total}"
PER_DATASET_DIR="${PREPARED_DIR}/per_dataset"

# Fine-tuned by default; set CHECKPOINT="" for the zero-shot baseline. Results dir
# is auto-tagged FT vs ZS so the two never overwrite each other.
CHECKPOINT="${CHECKPOINT:-${SCRIPT_DIR}/chronos2-single-stage_mtsbench_maskLossv2_HS_UN_v1/finetuned-ckpt}"
if [ -n "${CHECKPOINT}" ]; then _TAG=FT; else _TAG=ZS; fi
RESULTS_DIR="${RESULTS_DIR:-${SCRIPT_DIR}/results_${_TAG}}"
mkdir -p "${RESULTS_DIR}"

MODEL_ID="${MODEL_ID:-amazon/chronos-2}"
DEVICE="${DEVICE:-cuda}"

# Sequence layout — must match data prep / fine-tuning
NORMAL_SIGNAL_LENGTH="${NORMAL_SIGNAL_LENGTH:-256}"
CONTEXT_LENGTH="${CONTEXT_LENGTH:-512}"
PREDICTION_LENGTH="${PREDICTION_LENGTH:-64}"
INPUT_PATCH_SIZE="${INPUT_PATCH_SIZE:-16}"

# Mode-aware normal-prefix default (fine-tuned -> use prefix; zero-shot -> context only)
if [ -n "${CHECKPOINT:-}" ]; then _DEFAULT_PREFIX=1; else _DEFAULT_PREFIX=0; fi
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

export PYTHONPATH="${WORK_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
find "$WORK_ROOT/chronos" -name '__pycache__' -type d -exec rm -rf {} + 2>/dev/null || true

if [ -n "${CHECKPOINT}" ]; then MODE="FINE-TUNED (${CHECKPOINT})"; else MODE="ZERO-SHOT (${MODEL_ID})"; fi
echo "======================================================"
echo "  mTSBench Chronos-2 Anomaly Evaluation (maskloss_v2 + HS + UNIFIED NORMAL, per dataset)"
echo "  MODE              = ${MODE}"
echo "  PER_DATASET_DIR   = ${PER_DATASET_DIR}"
echo "  USE_NORMAL_PREFIX = ${USE_NORMAL_PREFIX}"
echo "  RESULTS_DIR       = ${RESULTS_DIR}"
echo "======================================================"

# Datasets to evaluate: explicit DATASETS env, else every per_dataset subfolder.
if [ -n "${DATASETS:-}" ]; then
    DS_LIST=(${DATASETS})
else
    DS_LIST=()
    for d in "${PER_DATASET_DIR}"/*/; do
        [ -f "${d}/test_model_inputs.pkl" ] && DS_LIST+=("$(basename "$d")")
    done
fi

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
        --test_pkl            "${TEST_PKL}"
        --meta_pkl            "${META_PKL}"
        --model_id            "${MODEL_ID}"
        --device              "${DEVICE}"
        --normal_signal_length "${NORMAL_SIGNAL_LENGTH}"
        --context_length      "${CONTEXT_LENGTH}"
        --prediction_length   "${PREDICTION_LENGTH}"
        --input_patch_size    "${INPUT_PATCH_SIZE}"
        --batch_windows       "${BATCH_WINDOWS}"
        --predict_batch_size  "${PREDICT_BATCH_SIZE}"
        --score_method        "${SCORE_METHOD}"
        --agg_method          "${AGG_METHOD}"
        --topk                "${TOPK}"
        --smooth_window       "${SMOOTH_WINDOW}"
        --sliding_window_VUS  "${SLIDING_WINDOW_VUS}"
        --vus_version         "${VUS_VERSION}"
        --vus_thre            "${VUS_THRE}"
        --out_csv             "${OUT_CSV}"
    )
    [ -n "${CHECKPOINT}" ] && ARGS+=(--checkpoint "${CHECKPOINT}")
    [ "${USE_NORMAL_PREFIX}" = "0" ] && ARGS+=(--no_normal_prefix)

    python -u "${FORWARD_PY}" "${ARGS[@]}"
done

# ─────────────────────────────────────────────────────────────────────────────
#  Aggregate -> per-dataset single value (the deliverable) + macro/micro summary
# ─────────────────────────────────────────────────────────────────────────────
echo ""
echo "======================================================"
python -u "${SCRIPT_DIR}/aggregate_results.py" \
    --results_dir "${RESULTS_DIR}" \
    --out_summary "${RESULTS_DIR}/per_dataset_summary.csv"
echo "  Per-dataset CSVs : ${RESULTS_DIR}/<DATASET>_results.csv"
echo "  Deliverable      : ${RESULTS_DIR}/per_dataset_summary.csv"
echo "======================================================"
