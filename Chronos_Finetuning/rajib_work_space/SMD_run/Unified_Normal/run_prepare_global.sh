#!/usr/bin/env bash
# Build Chronos-2 fine-tuning data with a UNIFIED (global) normal signal shared
# across every series in a dataset (see README.md).
#
# Each prepared window is laid out as:
#   [ normal_signal (N) | context (C) | future (P) ]
# and the normal prefix is one global shape (from --reference), re-scaled into
# each series' own per-channel units.
#
# Split is FILE-BASED (like SMD_run/run_prepare_smd.sh): a fraction of CSV files
# go to TEST, the rest to the training pool, with an optional file-based val set.
# The global medoid is derived from the TRAINING files only (no test leakage).
#
# Usage examples:
#   bash run_prepare_global.sh                                # 70/30 train/test, no val
#   TEST_FRACTION=0.6 bash run_prepare_global.sh              # 40/60 train/test
#   CREATE_VAL=1 bash run_prepare_global.sh                   # also carve a val set
#   CREATE_VAL=1 VAL_FRACTION=0.15 bash run_prepare_global.sh
#   TEST_FRACTION=0 bash run_prepare_global.sh                # no test split (all train)
#   DATA_DIR=/path/to/SMD OUTPUT_DIR=./prepared_global bash run_prepare_global.sh
#   REFERENCE=machine-3-3 bash run_prepare_global.sh         # force a training reference
#   GLOBAL_NORMAL_NPZ=/path/global.npz bash run_prepare_global.sh   # skip --reference

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"      # .../Unified_Normal

# ── Configuration — edit here or export before running ────────────────────────
# DATA_DIR="${DATA_DIR:-/home/rajib/mTSBench/Datasets/mTSBench/SMD}"
DATA_DIR="${DATA_DIR:-/mnt/c/Files/MTP Code Local Files/MTP_SEM_3_LOCAL_FILES/mTSBench/SMD}"

OUTPUT_DIR="${OUTPUT_DIR:-${SCRIPT_DIR}/prepared_global}"
REFERENCE="${REFERENCE:-auto}"               # 'auto' = derive dataset medoid; or a filename substring
METRIC="${METRIC:-fft}"                       # similarity for 'auto' medoid: fft (phase-robust) | pearson
GLOBAL_NORMAL_NPZ="${GLOBAL_NORMAL_NPZ:-}"   # optional precomputed normalized (F,N) signal

NORMAL_SIGNAL_LENGTH="${NORMAL_SIGNAL_LENGTH:-256}"   # N
CONTEXT_LENGTH="${CONTEXT_LENGTH:-512}"              # C
PREDICTION_LENGTH="${PREDICTION_LENGTH:-64}"         # P
STRIDE="${STRIDE:-64}"
MIN_LENGTH="${MIN_LENGTH:-50}"
TEST_FRACTION="${TEST_FRACTION:-0.3}"        # file-based test hold-out (0 = no test split)
SEED="${SEED:-42}"

# Validation toggle (file-based, carved from the training pool):
#   CREATE_VAL=0 (default) -> no validation data (full train pool used for training)
#   CREATE_VAL=1           -> carve VAL_FRACTION of the train-pool files into a val set
CREATE_VAL="${CREATE_VAL:-0}"
VAL_FRACTION="${VAL_FRACTION:-0.1}"
if [ "${CREATE_VAL}" = "1" ]; then
    EFFECTIVE_VAL_FRACTION="${VAL_FRACTION}"
else
    EFFECTIVE_VAL_FRACTION="0"
fi

PYTHON="${PYTHON:-python}"                    # override e.g. PYTHON=/path/to/conda/envs/.../bin/python

# ── Print config ─────────────────────────────────────────────────────────────
echo "======================================================"
echo "  Unified (Global) Normal-Signal Data Preparation"
echo "======================================================"
echo "  DATA_DIR             = $DATA_DIR"
echo "  OUTPUT_DIR           = $OUTPUT_DIR"
if [ -n "${GLOBAL_NORMAL_NPZ}" ]; then
echo "  GLOBAL_NORMAL_NPZ    = $GLOBAL_NORMAL_NPZ  (reference ignored)"
else
echo "  REFERENCE            = $REFERENCE"
[ "$REFERENCE" = "auto" ] && echo "  METRIC               = $METRIC  (auto medoid selection)"
fi
echo "  NORMAL_SIGNAL_LENGTH = $NORMAL_SIGNAL_LENGTH"
echo "  CONTEXT_LENGTH       = $CONTEXT_LENGTH"
echo "  PREDICTION_LENGTH    = $PREDICTION_LENGTH"
echo "  STRIDE               = $STRIDE"
echo "  MIN_LENGTH           = $MIN_LENGTH"
echo "  TEST_FRACTION        = $TEST_FRACTION  (file-based)"
echo "  CREATE_VAL           = $CREATE_VAL  (val_fraction=$EFFECTIVE_VAL_FRACTION)"
echo "  SEED                 = $SEED"
echo "  (fine-tune CONTEXT_LENGTH must = N + C = $((NORMAL_SIGNAL_LENGTH + CONTEXT_LENGTH)))"
echo "======================================================"
echo ""

# ── Build argument list and run ──────────────────────────────────────────────
PREP_ARGS=(
    --data_dir             "$DATA_DIR"
    --output_dir           "$OUTPUT_DIR"
    --normal_signal_length "$NORMAL_SIGNAL_LENGTH"
    --context_length       "$CONTEXT_LENGTH"
    --prediction_length    "$PREDICTION_LENGTH"
    --stride               "$STRIDE"
    --min_length           "$MIN_LENGTH"
    --test_fraction        "$TEST_FRACTION"
    --val_fraction         "$EFFECTIVE_VAL_FRACTION"
    --seed                 "$SEED"
)

if [ -n "${GLOBAL_NORMAL_NPZ}" ]; then
    PREP_ARGS+=(--global_normal_npz "$GLOBAL_NORMAL_NPZ")
else
    PREP_ARGS+=(--reference "$REFERENCE" --metric "$METRIC")
fi

"$PYTHON" -u "${SCRIPT_DIR}/prepare_global_normal.py" "${PREP_ARGS[@]}"

echo ""
echo "======================================================"
echo "  Data preparation complete!"
echo "  Train inputs : $OUTPUT_DIR/train_model_inputs.pkl"
echo "  Val inputs   : $OUTPUT_DIR/val_model_inputs.pkl   (only if CREATE_VAL=1)"
echo "  Test inputs  : $OUTPUT_DIR/test_model_inputs.pkl  (only if TEST_FRACTION>0)"
echo "  Test meta    : $OUTPUT_DIR/test_series_meta.pkl"
echo "  Global signal: $OUTPUT_DIR/global_normal_signal.npz"
echo ""
echo "  Point PREPARED_DIR in SMD_run/run_finetune_smd.sh at:"
echo "    $OUTPUT_DIR"
echo "======================================================"
