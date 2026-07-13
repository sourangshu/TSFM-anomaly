#!/usr/bin/env bash
# Fine-tune ONE Chronos-2 model on the pooled TRAIN datasets of every family.
#
# HIERARCHICAL SAMPLING is used exactly as in TOTAL_RUN_maskloss_v2_HS -- same sampler,
# same code, same knobs. It needs no reconfiguration here: it discovers its datasets from
# per_dataset/*/train_model_inputs.pkl, so pointing it at the unified pool simply makes
#   level 1  dataset ~ Uniform(K)          K = 9 train datasets (was 12 there)
#   level 2  kind    ~ Bernoulli(P_ANOM)   P_ANOM = 1/3, so 2:1 normal:anomalous
#   level 3  window  ~ count-weighted WITHIN that dataset (n_anom / 64 - n_anom)
# Every dataset in the pool gets 1/9 of the draws regardless of its size, which is the
# whole point of HS: MITDB (382k windows) does not drown out room-occupancy (206).
#
# Every hyperparameter below is copied from TOTAL_RUN_maskloss_v2_HS/run_finetune_total.sh
# unchanged -- per-step masked margin loss, relative margin M=5, LoRA r=32, lr 1e-5,
# 4000 steps. The ONLY thing this study changes is WHICH datasets are in the pool: nine
# whole datasets whose family siblings are held out entirely, instead of the train halves
# of all twelve. One variable.
#
# Usage:
#   bash run_finetune_family.sh                          # the unified model
#   DEBUG=1 bash run_finetune_family.sh                  # 50 windows/dataset, smoke test
#   RUN=per_family/ecg bash run_finetune_family.sh       # attribution variant

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# RUN names the directory built by make_folds.py. Default = the unified pooled model.
RUN="${RUN:-run}"

# `chronos` and `VUS_ROC_VUS_PR` are symlinked into this dir, so it is self-contained.
PREPARED_DIR="${PREPARED_DIR:-${SCRIPT_DIR}/${RUN}/prepared}"
OUTPUT_DIR="${OUTPUT_DIR:-${SCRIPT_DIR}/${RUN}/ckpt}"

MODEL_ID="${MODEL_ID:-amazon/chronos-2}"
DEVICE="${DEVICE:-cuda}"
FINETUNE_MODE="${FINETUNE_MODE:-lora}"

PREDICTION_LENGTH="${PREDICTION_LENGTH:-64}"
# N + C: normal-signal prefix (256) + actual context (512). Must match data prep.
CONTEXT_LENGTH="${CONTEXT_LENGTH:-768}"
NUM_STEPS="${NUM_STEPS:-4000}"
LR="${LR:-1e-5}"
BATCH_SIZE="${BATCH_SIZE:-160}"                 # CHANNEL ROWS, not windows
GRAD_ACCUM="${GRAD_ACCUM:-2}"
LOGGING_STEPS="${LOGGING_STEPS:-50}"
EVAL_STEPS="${EVAL_STEPS:-50}"
WARMUP_RATIO="${WARMUP_RATIO:-0.03}"
LR_SCHEDULER="${LR_SCHEDULER:-cosine}"
FP16="${FP16:-1}"
# 0 = validate on ${PREPARED_DIR}/val_model_inputs.pkl, which make_folds.py assembles from
# the TRAIN datasets of THIS fold only (an equal window budget each, ~1/3 anomalous where
# the val file allows). Held-out datasets contribute no val window, so best-model selection
# cannot see them. Set to 1 only when the prep ran with NO_VAL=1.
NO_VALIDATION="${NO_VALIDATION:-0}"
DEBUG="${DEBUG:-0}"

HINGE_MODE="${HINGE_MODE:-per_step}"
MARGIN_MODE="${MARGIN_MODE:-relative}"
MARGIN_M="${MARGIN_M:-5}"
MARGIN_TAU="${MARGIN_TAU:-6}"
MARGIN_LAMBDA="${MARGIN_LAMBDA:-1.0}"
P_ANOM="${P_ANOM:-0.3333333333333333}"
AGG_MODE="${AGG_MODE:-batch_global}"

LORA_R="${LORA_R:-32}"
LORA_ALPHA="${LORA_ALPHA:-32}"
LORA_DROPOUT="${LORA_DROPOUT:-0.01}"

ENABLE_SEP_TOKEN="${ENABLE_SEP_TOKEN:-1}"
NORMAL_SIGNAL_LENGTH="${NORMAL_SIGNAL_LENGTH:-256}"
INPUT_PATCH_SIZE="${INPUT_PATCH_SIZE:-16}"

# ── Validation ───────────────────────────────────────────────────────────────
if [ "${ENABLE_SEP_TOKEN}" = "1" ]; then
    if [ $(( NORMAL_SIGNAL_LENGTH % INPUT_PATCH_SIZE )) -ne 0 ]; then
        echo "ERROR: NORMAL_SIGNAL_LENGTH ($NORMAL_SIGNAL_LENGTH) must be a multiple of INPUT_PATCH_SIZE ($INPUT_PATCH_SIZE)"; exit 1
    fi
    if [ "${CONTEXT_LENGTH}" -le "${NORMAL_SIGNAL_LENGTH}" ]; then
        echo "ERROR: CONTEXT_LENGTH ($CONTEXT_LENGTH) must equal NORMAL_SIGNAL_LENGTH + actual context"; exit 1
    fi
fi

if [ ! -d "${PREPARED_DIR}/per_dataset" ]; then
    echo "ERROR: ${PREPARED_DIR}/per_dataset not found."
    echo "       Run ./run_prepare_family.sh first (it also builds the fold dirs)."
    exit 1
fi

if [ "${NO_VALIDATION}" != "1" ] && [ ! -f "${PREPARED_DIR}/val_model_inputs.pkl" ]; then
    echo "ERROR: ${PREPARED_DIR}/val_model_inputs.pkl not found, but NO_VALIDATION=0."
    echo "       Add the val set WITHOUT re-carving the pool:"
    echo "           VAL_ONLY=1 ./run_prepare_family.sh"
    echo "       Or train without validation:  NO_VALIDATION=1 ./run_finetune_family.sh"
    exit 1
fi

# NB: use `if`, not `[ -f ] && basename`. Under `set -o pipefail` a trailing && whose
# test fails on the LAST iteration makes the loop return 1, which the pipe propagates
# and `set -e` turns into a silent exit. The last dir here is often the one WITHOUT the
# artifact being looked for, so that fires routinely.
TRAIN_DS=$(for d in "${PREPARED_DIR}"/per_dataset/*/; do
    if [ -f "${d}/train_model_inputs.pkl" ]; then basename "$d"; fi
done | paste -sd+ -)
TEST_DS=$(for d in "${PREPARED_DIR}"/per_dataset/*/; do
    if [ -f "${d}/test_model_inputs.pkl" ]; then basename "$d"; fi
done | paste -sd, -)

if [ -z "${TRAIN_DS}" ]; then
    echo "ERROR: no train_model_inputs.pkl under ${PREPARED_DIR}/per_dataset"; exit 1
fi

N_TRAIN=$(echo "${TRAIN_DS}" | tr '+' '\n' | wc -l)

echo "======================================================"
echo "  FAMILY TRANSFER  --  ONE pooled model  [${RUN}]"
echo "======================================================"
echo "  TRAIN pool (${N_TRAIN})   = ${TRAIN_DS}"
echo "                      (100% of their *test.csv)"
echo "  HELD OUT          = ${TEST_DS}"
echo "                      (zero training windows from these)"
echo "  HS level 1        = uniform over ${N_TRAIN} datasets"
echo "------------------------------------------------------"
echo "  PREPARED_DIR      = $PREPARED_DIR"
echo "  OUTPUT_DIR        = $OUTPUT_DIR"
echo "  MODEL_ID          = $MODEL_ID"
echo "  FINETUNE_MODE     = $FINETUNE_MODE   (LoRA r=$LORA_R alpha=$LORA_ALPHA)"
echo "  NUM_STEPS         = $NUM_STEPS"
echo "  LR                = $LR"
echo "  BATCH_SIZE        = $BATCH_SIZE  (channel ROWS)  x GRAD_ACCUM $GRAD_ACCUM"
echo "  HINGE/MARGIN      = $HINGE_MODE / $MARGIN_MODE (M=$MARGIN_M, lambda=$MARGIN_LAMBDA)"
echo "  P_ANOM            = $P_ANOM  (HS level 2)"
echo "  AGG_MODE          = $AGG_MODE"
echo "  DEBUG             = $DEBUG"
echo "======================================================"
echo ""

export PYTHONPATH="${SCRIPT_DIR}${PYTHONPATH:+:${PYTHONPATH}}"

FINETUNE_ARGS=(
    --model_id                    "$MODEL_ID"
    --device                      "$DEVICE"
    --data_dir                    "$PREPARED_DIR"
    --output_dir                  "$OUTPUT_DIR"
    --finetune_mode               "$FINETUNE_MODE"
    --prediction_length           "$PREDICTION_LENGTH"
    --context_length              "$CONTEXT_LENGTH"
    --num_steps                   "$NUM_STEPS"
    --batch_size                  "$BATCH_SIZE"
    --gradient_accumulation_steps "$GRAD_ACCUM"
    --logging_steps               "$LOGGING_STEPS"
    --eval_steps                  "$EVAL_STEPS"
    --warmup_ratio                "$WARMUP_RATIO"
    --lr_scheduler_type           "$LR_SCHEDULER"
    --hinge_mode                  "$HINGE_MODE"
    --margin_mode                 "$MARGIN_MODE"
    --margin_m                    "$MARGIN_M"
    --margin_tau                  "$MARGIN_TAU"
    --margin_lambda               "$MARGIN_LAMBDA"
    --p_anom                      "$P_ANOM"
    --agg_mode                    "$AGG_MODE"
)

[ -n "${LR}" ] && FINETUNE_ARGS+=(--lr "$LR")
[ "${NO_VALIDATION}" = "1" ] && FINETUNE_ARGS+=(--no_validation)
[ "${DEBUG}" = "1" ] && FINETUNE_ARGS+=(--debug)
[ "${FP16}" = "1" ] && FINETUNE_ARGS+=(--fp16) || FINETUNE_ARGS+=(--no_fp16)

if [ "$FINETUNE_MODE" = "lora" ]; then
    FINETUNE_ARGS+=(--lora_r "$LORA_R" --lora_alpha "$LORA_ALPHA" --lora_dropout "$LORA_DROPOUT")
fi

if [ "${ENABLE_SEP_TOKEN}" = "1" ]; then
    FINETUNE_ARGS+=(
        --enable_sep_token
        --normal_signal_length "$NORMAL_SIGNAL_LENGTH"
        --input_patch_size     "$INPUT_PATCH_SIZE"
    )
fi

find -L "$SCRIPT_DIR/chronos" -name '__pycache__' -type d -exec rm -rf {} + 2>/dev/null || true

python -u "$SCRIPT_DIR/finetune_anomaly_simple.py" "${FINETUNE_ARGS[@]}"

echo ""
echo "======================================================"
echo "  Pooled model trained."
echo "  Checkpoint : $OUTPUT_DIR/finetuned-ckpt"
echo "  Score it   : bash run_forward_family.sh    (all held-out datasets)"
echo "======================================================"
