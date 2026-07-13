#!/usr/bin/env bash
# Single-stage anomaly-aware fine-tuning for Chronos-2 — HIERARCHICAL SAMPLING (HS).
#
# The loss is a PER-STEP masked margin (hinge) objective. The per-timestep labels
# (future_labels, 0=normal / 1=anomaly) mask the model's per-step pinball loss:
#   L = L_good + lambda * L_bad_term
#   normal steps (label 0)  → L_good     : minimised (predict normal well)
#   anomaly steps (label 1) → L_bad_term : pushed UP toward a margin, then saturates
#
# This arm is TOTAL_RUN_maskloss_v2 with EXACTLY ONE variable changed: the sampler.
#   1. HINGE_MODE=per_step   → hinge every anomaly step, not just the pooled mean  [same]
#   2. MARGIN_MODE=relative  → margin = MARGIN_M * per-window normal loss          [same]
#   3. P_ANOM=1/3            → HIERARCHICAL sampler (replaces v2's SAMPLING_TARGET)
#
# The hierarchical sampler draws each window in three levels:
#   level 1  dataset ~ Uniform(K)                 → every dataset gets 1/K of the draws
#   level 2  kind    ~ Bernoulli(P_ANOM)          → 2:1 normal:anomalous in expectation
#   level 3  window  ~ count-weighted WITHIN that dataset, by n_anom (anomalous kind)
#                      or 64 - n_anom (normal kind)
# Both level-3 branches are thresholdless: the anomalous branch is self-gating (a
# pure-normal window has weight 0), and the normal branch merely down-weights an
# anomaly-bearing window rather than excluding it. No `future_labels.sum() >= 10`
# threshold anywhere, so nothing contradicts the per-step objective.
#
# v2 used ONE global weight vector for both jobs, which cannot separate them: solving
# eps for a 40% anomaly-step target also pulled MITDB/SVDB/cicids to 1.35x their pool
# share and pushed GHL/OPPORTUNITY to 0.68x, undoing v2's own per-dataset cap.
#
# Assumes the prepared data dir contains the PER-DATASET train pools:
#   <PREPARED_DIR>/per_dataset/<DATASET>/train_model_inputs.pkl
# Run ./run_prepare_total.sh first if they don't exist. A flat train_model_inputs.pkl
# (the TOTAL_RUN / TOTAL_RUN_maskloss_v2 layout) will NOT work — it carries no dataset
# identity, so level 1 has nothing to group by.
#
# Usage examples:
#   bash run_finetune_total.sh                               # LoRA, all defaults
#   P_ANOM=0.5 bash run_finetune_total.sh                    # 1:1 normal:anomalous kinds
#   BATCH_SIZE=80 GRAD_ACCUM=4 bash run_finetune_total.sh    # tighter memory budget
#   DEBUG=1 bash run_finetune_total.sh                       # 50 windows PER DATASET, smoke test

set -euo pipefail

# ─────────────────────────────────────────────────────────────────────────────
#  Configuration — edit here or export before running
# ─────────────────────────────────────────────────────────────────────────────

# This script lives in rajib_work_space/TOTAL_RUN_maskloss_v2_HS. The training
# entrypoint (finetune_anomaly_simple.py) sits alongside it; `chronos` is one level up.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"   # .../rajib_work_space/TOTAL_RUN_maskloss_v2_HS
WORK_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"                  # .../rajib_work_space (code root)

# Paths
# PREPARED_DIR must contain per_dataset/<DS>/train_model_inputs.pkl, and — unless
# NO_VALIDATION=1 — val_model_inputs.pkl. Both are built by run_prepare_total.sh in this
# same folder (uncapped per-dataset train pools; hierarchically carved val set).
PREPARED_DIR="${PREPARED_DIR:-${SCRIPT_DIR}/prepared_total}"
OUTPUT_DIR="${OUTPUT_DIR:-${SCRIPT_DIR}/chronos2-single-stage_mtsbench_maskLossv2_HS_val_v2}"

# Optional third dataset (full path to a .pkl). When set, it is evaluated and
# logged every eval step exactly like validation (no weight updates), as eval_test_*.
TEST_DATA="${TEST_DATA:-}"

# Model
MODEL_ID="${MODEL_ID:-amazon/chronos-2}"
DEVICE="${DEVICE:-cuda}"

# Fine-tuning mode
FINETUNE_MODE="${FINETUNE_MODE:-lora}"          # "lora" or "full"

# Training hyperparameters — identical to TOTAL_RUN_maskloss_v2 so the arms are comparable
PREDICTION_LENGTH="${PREDICTION_LENGTH:-64}"
# NOTE: CONTEXT_LENGTH here is N + C (normal signal + actual context).
#       It must equal data-prep NORMAL_SIGNAL_LENGTH + data-prep CONTEXT_LENGTH
#       e.g.  CONTEXT_LENGTH (768) = NORMAL_SIGNAL_LENGTH (256) + CONTEXT_LENGTH (512)
CONTEXT_LENGTH="${CONTEXT_LENGTH:-768}"
# "Epoch" is not meaningful here: the hierarchical sampler draws WITH REPLACEMENT from
# an uncapped 340k-window pool, so it never enumerates it. BATCH_SIZE counts channel
# ROWS, and mean F across the 12 train datasets is ~24.7, so a 160-row batch holds ~6-7
# windows. 4000 steps x GRAD_ACCUM 2 ≈ 52k window draws ≈ 4.3k per dataset. NUM_STEPS is
# held at 4000 to match TOTAL_RUN_maskloss_v2 exactly — this arm changes the sampler only.
NUM_STEPS="${NUM_STEPS:-4000}"
LR="${LR:-1e-5}"                                # blank → script default (1e-5 lora / 1e-6 full)
BATCH_SIZE="${BATCH_SIZE:-160}"                 # CHANNEL ROWS, not windows
GRAD_ACCUM="${GRAD_ACCUM:-2}"                   # effective batch = BATCH_SIZE * GRAD_ACCUM
LOGGING_STEPS="${LOGGING_STEPS:-100}"
EVAL_STEPS="${EVAL_STEPS:-100}"      # validate + log eval_loss every N steps (must divide 100)
WARMUP_RATIO="${WARMUP_RATIO:-0.03}"
LR_SCHEDULER="${LR_SCHEDULER:-cosine}"
FP16="${FP16:-1}"                               # 1 = fp16 mixed precision, 0 = disable
# 0 = validate on PREPARED_DIR/val_model_inputs.pkl (hierarchical: an equal window budget
# from every trained dataset's mTSBench *val.csv, ~1/3 anomalous where the file allows).
# Set to 1 only when the prep ran with NO_VAL=1 and no val pkl exists.
NO_VALIDATION="${NO_VALIDATION:-0}"
DEBUG="${DEBUG:-0}"                             # 1 = truncate to 50 windows PER DATASET (smoke test)

# Margin (hinge) loss: L_good + MARGIN_LAMBDA * L_bad_term.
#
#   HINGE_MODE  (Suggestion 1: hinge each anomaly step, not the pooled mean)
#     per_step  -> L_bad_term = mean over anomaly steps of max(0, margin - per_step_i)
#                  (DEFAULT; pushes EVERY anomalous timestep — matches per-step VUS)
#     pooled    -> hinge only the mean anomaly loss (v1 behaviour)
#
#   MARGIN_MODE (Suggestion 2: relative, not absolute, margin)
#     relative  -> margin = MARGIN_M * L_good_w.detach() per window (DEFAULT;
#                  self-scales across quiet/busy series)
#     absolute  -> margin = MARGIN_TAU, a fixed constant (v1 behaviour)
HINGE_MODE="${HINGE_MODE:-per_step}"
MARGIN_MODE="${MARGIN_MODE:-relative}"
# Relative-margin multiplier (MARGIN_MODE=relative). 5 matches TOTAL_RUN_maskloss_v2,
# so this arm isolates the sampling change.
MARGIN_M="${MARGIN_M:-5}"
# MARGIN_TAU is used only when MARGIN_MODE=absolute. Must sit ABOVE the normal-point
# loss (~3-4 here) to matter — use ~10-15, NOT 2.
MARGIN_TAU="${MARGIN_TAU:-6}"
MARGIN_LAMBDA="${MARGIN_LAMBDA:-1.0}"

# Hierarchical sampler, level 2: probability that a draw asks for an ANOMALOUS-kind
# window. 1/3 gives the 2:1 normal:anomalous window ratio in expectation, PER DATASET.
# Because level 3 renormalizes its count weights within the chosen dataset, this holds
# for anomaly-sparse datasets (GHL, 1.7% anomaly steps) and anomaly-dense ones
# (room-occupancy, 18.9%) alike: both land at ~30-33% anomaly steps per batch.
# Level 1 (dataset) is uniform and has no knob — every dataset gets 1/K of the draws.
P_ANOM="${P_ANOM:-0.3333333333333333}"

# Per-step masked-loss aggregation:
#   batch_global -> pool all steps in the batch (each step weighted equally; most
#                   stable across datasets with different anomaly densities)
#   per_window   -> masked means within each window, then average over windows
#                   (each window weighted equally regardless of step count)
# NOTE: batch_global means a dataset's share of the GRADIENT is F_d / sum(F), not 1/K —
# level 1 balances draws, not channel rows (cicids F=72 -> 24% of the gradient; MITDB
# F=2 -> 0.7%). This is inherited from v2/3x, not introduced by HS. Kept at batch_global
# so HS-vs-2x stays a one-variable ablation; per_window would equalize it.
AGG_MODE="${AGG_MODE:-batch_global}"

# LoRA hyperparameters (ignored when FINETUNE_MODE=full)
LORA_R="${LORA_R:-32}"
LORA_ALPHA="${LORA_ALPHA:-32}"
LORA_DROPOUT="${LORA_DROPOUT:-0.01}"

# [SEP] token between normal signal and context
# When enabled, every target must be laid out as
#   [normal (NORMAL_SIGNAL_LENGTH) | context | future]
# and CONTEXT_LENGTH must equal NORMAL_SIGNAL_LENGTH + actual_context_length.
ENABLE_SEP_TOKEN="${ENABLE_SEP_TOKEN:-1}"            # set to 1 to enable
NORMAL_SIGNAL_LENGTH="${NORMAL_SIGNAL_LENGTH:-256}"  # length of normal signal prefix
INPUT_PATCH_SIZE="${INPUT_PATCH_SIZE:-16}"           # model's input_patch_size

# ─────────────────────────────────────────────────────────────────────────────
#  Validation
# ─────────────────────────────────────────────────────────────────────────────

if [ "${ENABLE_SEP_TOKEN}" = "1" ]; then
    if [ "${NORMAL_SIGNAL_LENGTH}" -le 0 ]; then
        echo "ERROR: NORMAL_SIGNAL_LENGTH must be > 0 when ENABLE_SEP_TOKEN=1"
        exit 1
    fi
    if [ $(( NORMAL_SIGNAL_LENGTH % INPUT_PATCH_SIZE )) -ne 0 ]; then
        echo "ERROR: NORMAL_SIGNAL_LENGTH ($NORMAL_SIGNAL_LENGTH) must be a multiple of INPUT_PATCH_SIZE ($INPUT_PATCH_SIZE)"
        exit 1
    fi
    if [ "${CONTEXT_LENGTH}" -le "${NORMAL_SIGNAL_LENGTH}" ]; then
        echo "ERROR: CONTEXT_LENGTH ($CONTEXT_LENGTH) must be greater than NORMAL_SIGNAL_LENGTH ($NORMAL_SIGNAL_LENGTH)"
        echo "       It must equal NORMAL_SIGNAL_LENGTH + actual_context_length."
        exit 1
    fi
fi

if [ ! -d "${PREPARED_DIR}/per_dataset" ]; then
    echo "ERROR: ${PREPARED_DIR}/per_dataset not found."
    echo "       The hierarchical sampler needs the per-dataset train pool."
    echo "       Run ./run_prepare_total.sh first."
    exit 1
fi

if [ "${NO_VALIDATION}" != "1" ] && [ ! -f "${PREPARED_DIR}/val_model_inputs.pkl" ]; then
    echo "ERROR: ${PREPARED_DIR}/val_model_inputs.pkl not found, but NO_VALIDATION=0."
    echo "       Add the val set WITHOUT re-carving train/test (safe on prepared data"
    echo "       a run is already reading):"
    echo "           VAL_ONLY=1 ./run_prepare_total.sh"
    echo "       Or train without validation:  NO_VALIDATION=1 ./run_finetune_total.sh"
    exit 1
fi

# ─────────────────────────────────────────────────────────────────────────────
#  Print config
# ─────────────────────────────────────────────────────────────────────────────

echo "======================================================"
echo "  Chronos-2 Single-Stage Anomaly Fine-Tuning  [HS]"
echo "======================================================"
echo "  PREPARED_DIR      = $PREPARED_DIR"
echo "  OUTPUT_DIR        = $OUTPUT_DIR"
echo "  MODEL_ID          = $MODEL_ID"
echo "  DEVICE            = $DEVICE"
echo "  FINETUNE_MODE     = $FINETUNE_MODE"
echo "  PREDICTION_LENGTH = $PREDICTION_LENGTH"
echo "  CONTEXT_LENGTH    = $CONTEXT_LENGTH"
echo "  NUM_STEPS         = $NUM_STEPS"
echo "  LR                = ${LR:-<default>}"
echo "  BATCH_SIZE        = $BATCH_SIZE  (channel ROWS, not windows)"
echo "  GRAD_ACCUM        = $GRAD_ACCUM  (effective batch = $((BATCH_SIZE * GRAD_ACCUM)))"
echo "  LOGGING_STEPS     = $LOGGING_STEPS"
echo "  EVAL_STEPS        = $EVAL_STEPS"
echo "  WARMUP_RATIO      = $WARMUP_RATIO"
echo "  LR_SCHEDULER      = $LR_SCHEDULER"
echo "  FP16              = $FP16"
echo "  NO_VALIDATION     = $NO_VALIDATION"
echo "  DEBUG             = $DEBUG  (1 = truncate to 50 windows per dataset)"
echo "  HINGE_MODE        = $HINGE_MODE  (Suggestion 1: per_step | pooled)"
echo "  MARGIN_MODE       = $MARGIN_MODE  (Suggestion 2: relative | absolute)"
echo "  MARGIN_M          = $MARGIN_M  (relative-margin multiplier; MARGIN_MODE=relative)"
echo "  MARGIN_TAU        = $MARGIN_TAU  (absolute margin; MARGIN_MODE=absolute)"
echo "  MARGIN_LAMBDA     = $MARGIN_LAMBDA"
echo "  P_ANOM            = $P_ANOM  (HS level 2: P(anomalous kind); level 1 = uniform datasets)"
echo "  AGG_MODE          = $AGG_MODE  (per-step masked loss aggregation)"
if [ "$FINETUNE_MODE" = "lora" ]; then
echo "------------------------------------------------------"
echo "  LORA_R            = $LORA_R"
echo "  LORA_ALPHA        = $LORA_ALPHA"
echo "  LORA_DROPOUT      = $LORA_DROPOUT"
fi
echo "------------------------------------------------------"
if [ "${ENABLE_SEP_TOKEN}" = "1" ]; then
echo "  [SEP] token       = ENABLED"
echo "  NORMAL_SIGNAL_LEN = $NORMAL_SIGNAL_LENGTH  (sep_patch_index = $(( NORMAL_SIGNAL_LENGTH / INPUT_PATCH_SIZE )))"
echo "  INPUT_PATCH_SIZE  = $INPUT_PATCH_SIZE"
echo "  Sequence layout   = [normal][SEP][context][REG][future]"
else
echo "  [SEP] token       = disabled"
echo "  Sequence layout   = [context][REG][future]"
fi
echo "======================================================"
echo ""

# ─────────────────────────────────────────────────────────────────────────────
#  Ensure the local rajib_work_space/chronos is used, not any installed package
# ─────────────────────────────────────────────────────────────────────────────

export PYTHONPATH="${WORK_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

# ─────────────────────────────────────────────────────────────────────────────
#  Build argument list and run
# ─────────────────────────────────────────────────────────────────────────────

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

# Learning rate — omit entirely to use the script's built-in default
[ -n "${LR}" ] && FINETUNE_ARGS+=(--lr "$LR")

# Optional third (test) dataset — evaluated like validation, logged as eval_test_*
[ -n "${TEST_DATA}" ] && FINETUNE_ARGS+=(--test_data "$TEST_DATA")

# Flags
[ "${NO_VALIDATION}" = "1" ] && FINETUNE_ARGS+=(--no_validation)
[ "${DEBUG}" = "1" ] && FINETUNE_ARGS+=(--debug)
[ "${FP16}" = "1" ] && FINETUNE_ARGS+=(--fp16) || FINETUNE_ARGS+=(--no_fp16)

# LoRA hyperparameters
if [ "$FINETUNE_MODE" = "lora" ]; then
    FINETUNE_ARGS+=(
        --lora_r       "$LORA_R"
        --lora_alpha   "$LORA_ALPHA"
        --lora_dropout "$LORA_DROPOUT"
    )
fi

# [SEP] token
if [ "${ENABLE_SEP_TOKEN}" = "1" ]; then
    FINETUNE_ARGS+=(
        --enable_sep_token
        --normal_signal_length "$NORMAL_SIGNAL_LENGTH"
        --input_patch_size     "$INPUT_PATCH_SIZE"
    )
fi

# Clear stale bytecode so edits made on Windows are always picked up in WSL
find "$WORK_ROOT/chronos" -name '__pycache__' -type d -exec rm -rf {} + 2>/dev/null || true

# Use the HS training entrypoint that lives alongside this script.
python -u "$SCRIPT_DIR/finetune_anomaly_simple.py" "${FINETUNE_ARGS[@]}"

# ─────────────────────────────────────────────────────────────────────────────
#  Done
# ─────────────────────────────────────────────────────────────────────────────

echo ""
echo "======================================================"
echo "  Single-stage fine-tuning complete!"
echo "  Checkpoint : $OUTPUT_DIR/finetuned-ckpt  ← use for inference"
echo ""
echo "  Load the final model with:"
echo "    from chronos import BaseChronosPipeline"
echo "    pipeline = BaseChronosPipeline.from_pretrained("
echo "        '$OUTPUT_DIR/finetuned-ckpt', device_map='cuda')"
echo "======================================================"
