#!/usr/bin/env bash
#
# Evaluate Chronos-2 (zero-shot or a fine-tuned checkpoint) on the per-dataset
# mTSBench test sets produced by run_prepare_total.sh. Runs forward.py once per
# dataset (each dataset's per_dataset/<DS>/test_model_inputs.pkl + test_series_meta.pkl),
# writes results/<DS>_results.csv (one row per series), then calls
# aggregate_results.py to produce the DELIVERABLE: one VUS-PR value per dataset.
#
# Inference is IDENTICAL across all methods by construction: the test pkls here are
# symlinks to the very bytes TOTAL_RUN_maskloss_v2 evaluated on, and the only line that
# differs from that arm's copy of this script is the default CHECKPOINT path.
#
# NOTE (2026-07-20): this arm's forward.py is NO LONGER byte-identical to the other arms'
# (was md5 a226d1f5e899d7ae332112e3f29d076f) — the optional --cross_learning flag was added
# here. The flag is OFF by default and the predict() call is otherwise untouched, so the
# default code path still produces the same numbers as the other arms. family_train_test/
# symlinks to this file and inherits the flag.
#
# Test sets are UNCAPPED — no arm's balancing ever touched TRAIN's test half, so
# evaluation sees every test window, exactly like run_forward_smd.sh did for SMD.
#
# Usage:
#   ./run_forward_total.sh                       # all datasets, fine-tuned ckpt
#   DATASETS="SMD MSL" ./run_forward_total.sh    # subset
#   CHECKPOINT="" ./run_forward_total.sh         # zero-shot baseline
#   CROSS_LEARNING=1 ./run_forward_total.sh      # + Chronos-2 cross-learning (see below)
#   EVAL_SPLIT=full ./run_forward_total.sh       # 100% of data (train+test files) — NOT held out
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"   # .../rajib_work_space/TOTAL_RUN_maskloss_v2_HS
WORK_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"                  # .../rajib_work_space
FORWARD_PY="${SCRIPT_DIR}/forward.py"                        # local copy (self-contained)

PREPARED_DIR="${PREPARED_DIR:-${SCRIPT_DIR}/prepared_total}"
PER_DATASET_DIR="${PREPARED_DIR}/per_dataset"

# Fine-tuned by default; set CHECKPOINT="" for the zero-shot baseline. Results dir
# is auto-tagged FT vs ZS so the two never overwrite each other.
CHECKPOINT="${CHECKPOINT:-${SCRIPT_DIR}/chronos2-single-stage_mtsbench_maskLossv2_HS_v1/finetuned-ckpt}"
if [ -n "${CHECKPOINT}" ]; then _TAG=FT; else _TAG=ZS; fi

# ─────────────────────────────────────────────────────────────────────────────
#  Chronos-2 CROSS-LEARNING  (default: 0 = off)
# ─────────────────────────────────────────────────────────────────────────────
# 1 -> pass cross_learning=True to pipeline.predict(). Internally this zeros the
# batch's group_ids, so every task in a predict() batch is forecast JOINTLY and the
# model shares information across them, instead of each window being an independent
# task. Read before flipping it:
#
#   * Results become BATCH-DEPENDENT. Which windows land together is decided by
#     BATCH_WINDOWS and PREDICT_BATCH_SIZE below; change either and the scores move.
#   * Windows here are chunked contiguously across the whole test pkl, so a batch can
#     straddle two different SERIES — they will attend to each other. If you want
#     cross-learning strictly within a series, that needs a change to how forward.py
#     chunks `windows`, not just this flag.
#   * Upstream recommends a batch of ~100 TASKS (the Chronos-2 report's setting).
#     PREDICT_BATCH_SIZE counts SERIES ROWS (= windows * n_variates), not tasks, so on
#     a wide dataset 160 rows is only a handful of tasks. Tune BATCH_WINDOWS if you
#     want to land near 100.
#   * Upstream is explicit that it does not always help — it is most useful when each
#     series has little history. Here each window carries 768 context steps, so treat
#     this as an experiment, not an expected win.
#
# Results dir is tagged _XL when on, so a cross-learning run never overwrites the
# baseline numbers in results_FT / results_ZS.
CROSS_LEARNING="${CROSS_LEARNING:-0}"
if [ "${CROSS_LEARNING}" = "1" ]; then _TAG="${_TAG}_XL"; fi

# ─────────────────────────────────────────────────────────────────────────────
#  EVAL SPLIT  (default: test = the held-out half)
# ─────────────────────────────────────────────────────────────────────────────
#   test  -> per_dataset/<DS>/test_model_inputs.pkl   (the 50% test-file half; DEFAULT)
#   full  -> per_dataset/<DS>/full_model_inputs.pkl   (100% of the data: train + test files)
#
# The `full` pkls are NOT built by run_prepare_total.sh. Build them once with:
#     python prepare_full_eval.py
# That script ignores the 50/50 file split and windows every real *test.csv (synthetic
# syn_data/ files are excluded — they are train-only fabrications). It writes only NEW
# filenames, so the existing split is untouched.
#
# ⚠ EVAL_SPLIT=full IS NOT A HELD-OUT SCORE. The checkpoint was fine-tuned on windows
# from the train-half files, and those files are in the full set. Roughly half of each
# multi-file dataset's eval windows are therefore training data. See
# prepared_total/manifest_full_eval.json -> contaminated_window_frac for the exact
# per-dataset share. Report these as TRAIN+TEST numbers next to the held-out results_FT
# ones, never instead of them.
#
# (The 7 one-file datasets — CalIt2, GECCO, Genesis, PSM, creditcard, metro, swan — are
# test-only, so their full set equals their test set and their contamination is 0%.)
#
# Results dir is tagged _FULL, so this never overwrites the held-out numbers.
EVAL_SPLIT="${EVAL_SPLIT:-test}"
case "${EVAL_SPLIT}" in
    test) _PKL_PREFIX=test ;;
    full) _PKL_PREFIX=full; _TAG="${_TAG}_FULL" ;;
    *) echo "ERROR: EVAL_SPLIT must be 'test' or 'full', got '${EVAL_SPLIT}'"; exit 1 ;;
esac

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
echo "  mTSBench Chronos-2 Anomaly Evaluation (maskloss_v2 + HS, per dataset)"
echo "  MODE              = ${MODE}"
echo "  PER_DATASET_DIR   = ${PER_DATASET_DIR}"
if [ "${EVAL_SPLIT}" = "full" ]; then
echo "  EVAL_SPLIT        = full  ⚠ 100% of data (train+test files) — NOT a held-out score"
else
echo "  EVAL_SPLIT        = test  (the held-out 50% test-file half)"
fi
echo "  USE_NORMAL_PREFIX = ${USE_NORMAL_PREFIX}"
if [ "${CROSS_LEARNING}" = "1" ]; then
echo "  CROSS_LEARNING    = 1  (ON — batch-dependent; BATCH_WINDOWS=${BATCH_WINDOWS}, PREDICT_BATCH_SIZE=${PREDICT_BATCH_SIZE})"
else
echo "  CROSS_LEARNING    = 0  (off — each window forecast independently)"
fi
echo "  RESULTS_DIR       = ${RESULTS_DIR}"
echo "======================================================"

# Datasets to evaluate: explicit DATASETS env, else every per_dataset subfolder.
if [ -n "${DATASETS:-}" ]; then
    DS_LIST=(${DATASETS})
else
    DS_LIST=()
    for d in "${PER_DATASET_DIR}"/*/; do
        [ -f "${d}/${_PKL_PREFIX}_model_inputs.pkl" ] && DS_LIST+=("$(basename "$d")")
    done
fi

if [ "${#DS_LIST[@]}" -eq 0 ]; then
    echo "ERROR: no dataset under ${PER_DATASET_DIR} has ${_PKL_PREFIX}_model_inputs.pkl"
    [ "${EVAL_SPLIT}" = "full" ] && echo "       Build the full-data eval set first:  python prepare_full_eval.py"
    exit 1
fi

for ds in "${DS_LIST[@]}"; do
    TEST_PKL="${PER_DATASET_DIR}/${ds}/${_PKL_PREFIX}_model_inputs.pkl"
    META_PKL="${PER_DATASET_DIR}/${ds}/${_PKL_PREFIX}_series_meta.pkl"
    OUT_CSV="${RESULTS_DIR}/${ds}_results.csv"
    if [ ! -f "${TEST_PKL}" ]; then
        echo "  skip ${ds}: no ${_PKL_PREFIX} pkl"; continue
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
    if [ "${CROSS_LEARNING}" = "1" ]; then ARGS+=(--cross_learning); fi

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
