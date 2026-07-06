#!/usr/bin/env bash
# Single-stage anomaly-aware fine-tuning for Chronos-2 on the COMBINED mTSBench
# train set (prepared_total/train_model_inputs.pkl from run_prepare_total.sh).
#
# Identical objective/layout to SMD_run/run_finetune_smd.sh — only PREPARED_DIR and
# OUTPUT_DIR differ. The combined train pkl mixes datasets with DIFFERENT feature
# counts (2..72); Chronos2Dataset flattens each window into per-variate rows tagged
# by group_ids, so variable-variate windows batch together fine.
#
# Usage:
#   bash run_finetune_total.sh
#   FINETUNE_MODE=full bash run_finetune_total.sh
#   NUM_STEPS=8000 BATCH_SIZE=128 bash run_finetune_total.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"   # .../rajib_work_space/TOTAL_RUN
WORK_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"                  # .../rajib_work_space (code root)

# Paths — combined train pkl (no validation; NO_VALIDATION=1 by default)
PREPARED_DIR="${PREPARED_DIR:-${SCRIPT_DIR}/prepared_total}"
OUTPUT_DIR="${OUTPUT_DIR:-${SCRIPT_DIR}/chronos2-single-stage_TOTAL_v1}"
TEST_DATA="${TEST_DATA:-}"

# Model
MODEL_ID="${MODEL_ID:-amazon/chronos-2}"
DEVICE="${DEVICE:-cuda}"

# Fine-tuning mode
FINETUNE_MODE="${FINETUNE_MODE:-lora}"          # "lora" or "full"

# Training hyperparameters
PREDICTION_LENGTH="${PREDICTION_LENGTH:-64}"
# CONTEXT_LENGTH = NORMAL_SIGNAL_LENGTH (256) + actual context (512) = 768
CONTEXT_LENGTH="${CONTEXT_LENGTH:-768}"
NUM_STEPS="${NUM_STEPS:-2400}"
LR="${LR:-1e-5}"
BATCH_SIZE="${BATCH_SIZE:-160}"
GRAD_ACCUM="${GRAD_ACCUM:-2}"
LOGGING_STEPS="${LOGGING_STEPS:-50}"
EVAL_STEPS="${EVAL_STEPS:-100}"
WARMUP_RATIO="${WARMUP_RATIO:-0.03}"
LR_SCHEDULER="${LR_SCHEDULER:-cosine}"
FP16="${FP16:-1}"
NO_VALIDATION="${NO_VALIDATION:-0}" # set 1 to disable validation
DEBUG="${DEBUG:-0}"

# Window-level anomaly threshold — MUST match --anomaly_threshold used in data prep
ANOMALY_THRESHOLD="${ANOMALY_THRESHOLD:-10}"

# Margin (hinge) loss
MARGIN_TAU="${MARGIN_TAU:-8.0}" # increased from 6 to 8
MARGIN_LAMBDA="${MARGIN_LAMBDA:-1.0}"

# LoRA
LORA_R="${LORA_R:-16}"
LORA_ALPHA="${LORA_ALPHA:-32}"
LORA_DROPOUT="${LORA_DROPOUT:-0.01}"

# [SEP] token between normal signal and context
ENABLE_SEP_TOKEN="${ENABLE_SEP_TOKEN:-1}"
NORMAL_SIGNAL_LENGTH="${NORMAL_SIGNAL_LENGTH:-256}"
INPUT_PATCH_SIZE="${INPUT_PATCH_SIZE:-16}"

if [ "${ENABLE_SEP_TOKEN}" = "1" ]; then
    if [ $(( NORMAL_SIGNAL_LENGTH % INPUT_PATCH_SIZE )) -ne 0 ]; then
        echo "ERROR: NORMAL_SIGNAL_LENGTH ($NORMAL_SIGNAL_LENGTH) must be a multiple of INPUT_PATCH_SIZE ($INPUT_PATCH_SIZE)"; exit 1
    fi
    if [ "${CONTEXT_LENGTH}" -le "${NORMAL_SIGNAL_LENGTH}" ]; then
        echo "ERROR: CONTEXT_LENGTH ($CONTEXT_LENGTH) must be > NORMAL_SIGNAL_LENGTH ($NORMAL_SIGNAL_LENGTH)"; exit 1
    fi
fi

echo "======================================================"
echo "  Chronos-2 Single-Stage Fine-Tuning — TOTAL (mTSBench)"
echo "======================================================"
echo "  PREPARED_DIR      = $PREPARED_DIR"
echo "  OUTPUT_DIR        = $OUTPUT_DIR"
echo "  FINETUNE_MODE     = $FINETUNE_MODE"
echo "  CONTEXT/PRED      = $CONTEXT_LENGTH / $PREDICTION_LENGTH"
echo "  NUM_STEPS         = $NUM_STEPS   BATCH=$BATCH_SIZE x GA=$GRAD_ACCUM"
echo "  ANOMALY_THRESHOLD = $ANOMALY_THRESHOLD"
echo "  MARGIN_TAU/LAMBDA = $MARGIN_TAU / $MARGIN_LAMBDA"
echo "  [SEP] token       = $ENABLE_SEP_TOKEN  (normal_len=$NORMAL_SIGNAL_LENGTH)"
echo "======================================================"
echo

export PYTHONPATH="${WORK_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

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
    --anomaly_threshold           "$ANOMALY_THRESHOLD"
    --margin_tau                  "$MARGIN_TAU"
    --margin_lambda               "$MARGIN_LAMBDA"
)
[ -n "${LR}" ] && FINETUNE_ARGS+=(--lr "$LR")
[ -n "${TEST_DATA}" ] && FINETUNE_ARGS+=(--test_data "$TEST_DATA")
[ "${NO_VALIDATION}" = "1" ] && FINETUNE_ARGS+=(--no_validation)
[ "${DEBUG}" = "1" ] && FINETUNE_ARGS+=(--debug)
[ "${FP16}" = "1" ] && FINETUNE_ARGS+=(--fp16) || FINETUNE_ARGS+=(--no_fp16)

if [ "$FINETUNE_MODE" = "lora" ]; then
    FINETUNE_ARGS+=(--lora_r "$LORA_R" --lora_alpha "$LORA_ALPHA" --lora_dropout "$LORA_DROPOUT")
fi
if [ "${ENABLE_SEP_TOKEN}" = "1" ]; then
    FINETUNE_ARGS+=(--enable_sep_token --normal_signal_length "$NORMAL_SIGNAL_LENGTH" --input_patch_size "$INPUT_PATCH_SIZE")
fi

find "$WORK_ROOT/chronos" -name '__pycache__' -type d -exec rm -rf {} + 2>/dev/null || true

python -u "$WORK_ROOT/finetune_anomaly_simple.py" "${FINETUNE_ARGS[@]}"

echo
echo "Done. Checkpoint: $OUTPUT_DIR/finetuned-ckpt"
