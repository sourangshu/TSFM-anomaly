#!/usr/bin/env bash
#
# Drive the family-transfer study end to end.
#
#   prepare   -> pool/ + run/prepared          (whole datasets, no within-dataset split)
#   finetune  -> ONE model on the pooled 9 training datasets, hierarchical sampling
#   score     -> that model on all 6 held-out datasets        (the experiment)
#   score     -> the BASE model on the same 6                 (the baseline)
#   summarize
#
# Zero-shot MUST be re-run here rather than lifted from TOTAL_RUN_maskloss_v2_HS/results_ZS:
# that one is scored on 50% test halves, this study evaluates 100% of each held-out
# dataset's files. Different test set, non-comparable numbers.
#
# The existing results_FT is NOT a valid reference column either: that checkpoint trained
# on the train-half files of every dataset, and under this protocol those files are part
# of the test set. It is leaky here by construction. Do not put it in the table.
#
# Usage:
#   bash run_study.sh                # prepare -> finetune -> score -> summarize
#   SKIP_PREPARE=1 bash run_study.sh # pool/ already built
#   SKIP_TRAIN=1 bash run_study.sh   # re-score the existing checkpoint only
#
set -euo pipefail
cd "$(dirname "$0")"

SKIP_PREPARE="${SKIP_PREPARE:-0}"
SKIP_TRAIN="${SKIP_TRAIN:-0}"

echo "######################################################"
echo "#  FAMILY TRANSFER STUDY -- one pooled model"
echo "######################################################"
echo

if [ "${SKIP_PREPARE}" != "1" ]; then
    echo "### PREPARE ###"
    bash ./run_prepare_family.sh
    echo
fi

if [ "${SKIP_TRAIN}" != "1" ]; then
    echo "### FINETUNE (one model, pooled train datasets, HS sampler) ###"
    bash ./run_finetune_family.sh
    echo
fi

echo "### SCORE: fine-tuned model on the held-out datasets (the experiment) ###"
bash ./run_forward_family.sh
echo

echo "### SCORE: zero-shot base model on the same held-out datasets (the baseline) ###"
ZERO_SHOT=1 bash ./run_forward_family.sh
echo

echo "### SUMMARY ###"
python -u summarize_study.py
