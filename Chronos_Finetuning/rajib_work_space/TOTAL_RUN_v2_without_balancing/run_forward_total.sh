#!/usr/bin/env bash
#
# Evaluate Chronos-2 (zero-shot or a fine-tuned checkpoint) on the per-dataset
# mTSBench test sets produced by run_prepare_total.sh. Reuses SMD_run/forward.py
# unchanged — one invocation per dataset (each dataset's test_model_inputs.pkl +
# test_series_meta.pkl), exactly like run_forward_smd.sh did for SMD alone.
#
# Each test window carries its OWN per-series normal prefix (built by data prep from
# that *test.csv file's normal zones), and the fine-tuned model is fed [normal|context].
#
# Writes per-dataset results to results/<DATASET>_results.csv and concatenates
# them into results/ALL_results.csv (one row per test series).
#
# Usage:
#   ./run_forward_total.sh                       # all datasets, fine-tuned ckpt
#   DATASETS="SMD MSL" ./run_forward_total.sh    # subset
#   CHECKPOINT="" ./run_forward_total.sh         # zero-shot baseline
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"   # .../rajib_work_space/TOTAL_RUN_v2
WORK_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"                  # .../rajib_work_space
FORWARD_PY="${WORK_ROOT}/SMD_run/forward.py"                 # reuse the SMD forward script

PREPARED_DIR="${PREPARED_DIR:-${SCRIPT_DIR}/prepared_total}"
PER_DATASET_DIR="${PREPARED_DIR}/per_dataset"
RESULTS_DIR="${RESULTS_DIR:-${SCRIPT_DIR}/results_finetuned}"
mkdir -p "${RESULTS_DIR}"

MODEL_ID="${MODEL_ID:-amazon/chronos-2}"
# Fine-tuned by default; set CHECKPOINT="" for the zero-shot baseline.
CHECKPOINT="${CHECKPOINT:-${SCRIPT_DIR}/chronos2-single-stage_TOTAL_v2/finetuned-ckpt}"
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
echo "  mTSBench Chronos-2 Anomaly Evaluation — v2 (per dataset)"
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

ALL_CSV="${RESULTS_DIR}/ALL_results.csv"
rm -f "${ALL_CSV}"

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

    # Append to the aggregate CSV (prefix a dataset column; header once).
    if [ -f "${OUT_CSV}" ]; then
        if [ ! -f "${ALL_CSV}" ]; then
            head -1 "${OUT_CSV}" | sed 's/^/dataset,/' > "${ALL_CSV}"
        fi
        tail -n +2 "${OUT_CSV}" | sed "s/^/${ds},/" >> "${ALL_CSV}"
    fi
done

echo ""
echo "======================================================"
echo "  Done. Per-dataset CSVs in ${RESULTS_DIR}/"
echo "  Aggregate: ${ALL_CSV}"
echo "  Mean VUS-PR across all series:"
python - "${ALL_CSV}" <<'PY' || true
import sys, csv
p = sys.argv[1]
try:
    rows = list(csv.DictReader(open(p)))
except FileNotFoundError:
    print("   (no aggregate csv)"); raise SystemExit
if not rows:
    print("   (empty)"); raise SystemExit
import statistics as st
for k in ("VUS-PR","VUS-ROC","AUC-PR","AUC-ROC"):
    vals=[float(r[k]) for r in rows if r.get(k) not in (None,"","nan")]
    if vals: print(f"   {k:<10}: {st.mean(vals):.4f}  (n={len(vals)} series)")
PY
echo "======================================================"
