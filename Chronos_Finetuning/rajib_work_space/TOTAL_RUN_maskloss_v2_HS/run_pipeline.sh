#!/usr/bin/env bash
#
# ═══════════════════════════════════════════════════════════════════════════════
#  ONE EXPERIMENT, TOP TO BOTTOM:  data prep  →  fine-tune  →  inference
#  Submit THIS file as a job. Pick an experiment below, submit. Pick another, submit.
# ═══════════════════════════════════════════════════════════════════════════════
#
# Experiments differ in ONE thing: which datasets get the sampler's level 1.5 — the
# dissimilarity-weighted FILE draw. Everything else (the prepared data, P_ANOM, the loss,
# LoRA, the step count, the test halves) is identical, which is what makes the comparison
# an ablation of the file level and not of the data.
#
# Level 1.5 recap: after the dataset is drawn uniformly, ONE of its source *test.csv files
# is drawn from that file's dissimilarity weight, and only THEN the kind and the window.
# Near-duplicate files are down-weighted and unusual ones promoted, so a dataset's budget
# is spent across its distinct PATTERNS rather than across its raw minutes of recording.
# Datasets not named keep the plain 3-level path.
#
# Outputs, all tagged by EXP_TAG so two jobs can never overwrite each other:
#   chronos2-HS_<TAG>/finetuned-ckpt          the model
#   results_<TAG>/per_dataset_summary.csv     the deliverable (one VUS-PR per dataset)
#   logs_<TAG>/{1_prep,2_finetune,3_forward}.log
#
# Zero-shot inference is NOT run: forward.py is unchanged from the previous arms, so the
# zero-shot baseline you already have still stands.
#
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"


# ═════════════════════════════════════════════════════════════════════════════
#  1. PICK THE EXPERIMENT — uncomment ONE block, comment the rest.
# ═════════════════════════════════════════════════════════════════════════════

# ── EXP A: baseline — no file level anywhere (the plain 3-level sampler) ──────
# This is the arm the other two need to be compared against. Without it you can only
# say "MSL-only differs from all", not "the file level helps".
# FILE_DIVERSITY_DATASETS=""
# EXP_TAG="base"

# ── EXP B: file level on MSL ONLY ────────────────────────────────────────────
# The single-dataset probe. Every other dataset keeps the plain 3-level path, so any
# change in MSL's number is attributable to the file level alone.
FILE_DIVERSITY_DATASETS="MSL"
EXP_TAG="fdiv_MSL"

# ── EXP C: file level on EVERY multi-file dataset ────────────────────────────
# 'all' resolves at load time to every dataset in the train pool with >1 train file
# (single-file datasets are skipped automatically — the file level is a no-op there).
# FILE_DIVERSITY_DATASETS="all"
# EXP_TAG="fdiv_all"

# ── EXP D: your own subset ───────────────────────────────────────────────────
# FILE_DIVERSITY_DATASETS="MITDB SVDB Exathlon Daphnet SMAP GutenTAG"
# EXP_TAG="fdiv_manyfile"


# ═════════════════════════════════════════════════════════════════════════════
#  2. PICK THE STAGES — uncomment ONE block. Normally leave this alone.
# ═════════════════════════════════════════════════════════════════════════════
#
# Every job preps its OWN copy of the data (see PREPARED_DIR below), so jobs are fully
# independent: submit as many as you like, whenever you like, in any order. No shared
# state, no races, nothing to sequence.
#
# The other blocks exist only for resuming a job that died partway.

# ── Full pipeline: prep, then fine-tune, then evaluate. THE NORMAL CASE. ─────
STAGES="prep finetune forward"

# ── Resume: skip prep, reuse this experiment's already-prepared data ─────────
# STAGES="finetune forward"

# ── Resume: evaluate only, from this experiment's existing checkpoint ────────
# STAGES="forward"


# ═════════════════════════════════════════════════════════════════════════════
#  3. Shared configuration — normally leave alone
# ═════════════════════════════════════════════════════════════════════════════

# Each experiment preps into its OWN directory, tagged like everything else. That costs
# ~7 GB per experiment (the test halves are symlinks, so they are not duplicated) and buys
# total independence between jobs.
#
# This is safe for the ablation ONLY because prep is deterministic: same DATA_ROOT, same
# SEED, same geometry -> the same file split and byte-identical windows every time. So two
# separately-prepped dirs contain the same data, and any difference in the final numbers is
# the sampler, not the data. The guard below checks that assumption rather than trusting it:
# if a sibling prepared_total_* was carved with a different config, this job stops.
PREPARED_DIR="${PREPARED_DIR:-${SCRIPT_DIR}/prepared_total_${EXP_TAG}}"

# Synthetic files -> TRAIN half only, so the 7 one-file datasets (CalIt2, creditcard,
# GECCO, Genesis, metro, PSM, swan) join the train pool. Their real *test.csv stays their
# whole, untouched test half, so inference is unaffected. Level 1 grows 12 -> 19 datasets.
USE_SYN_DATA="${USE_SYN_DATA:-1}"

# Validation. NO_VAL=1 skips carving the val set at prep, which REQUIRES NO_VALIDATION=1
# at fine-tune time — otherwise the fine-tuner aborts looking for a val pkl that was never
# written. Chained here so the two cannot drift apart.
NO_VAL="${NO_VAL:-1}"
NO_VALIDATION="${NO_VALIDATION:-${NO_VAL}}"

DEVICE="${DEVICE:-cuda}"

# Set to 1 to print the plan and exit without running anything.
DRY_RUN="${DRY_RUN:-0}"

# Derived — do not edit.
OUTPUT_DIR="${SCRIPT_DIR}/chronos2-HS_${EXP_TAG}"
CKPT="${OUTPUT_DIR}/finetuned-ckpt"
RESULTS_DIR="${SCRIPT_DIR}/results_${EXP_TAG}"
LOG_DIR="${SCRIPT_DIR}/logs_${EXP_TAG}"


# ═════════════════════════════════════════════════════════════════════════════
#  Runner
# ═════════════════════════════════════════════════════════════════════════════

banner() {
    echo ""
    echo "═══════════════════════════════════════════════════════════════════════"
    echo "  $*"
    echo "  $(date '+%Y-%m-%d %H:%M:%S')"
    echo "═══════════════════════════════════════════════════════════════════════"
}
has_stage() { [[ " ${STAGES} " == *" $1 "* ]]; }

mkdir -p "${LOG_DIR}"

banner "PLAN  —  ${EXP_TAG}"
echo "  level 1.5 (FILE) on : ${FILE_DIVERSITY_DATASETS:-<none — plain 3-level sampler>}"
echo "  STAGES              : ${STAGES}"
echo "  PREPARED_DIR        : ${PREPARED_DIR}   (this experiment's own copy)"
echo "  USE_SYN_DATA        : ${USE_SYN_DATA}   (synthetic files -> TRAIN half only)"
echo "  NO_VALIDATION       : ${NO_VALIDATION}  (from NO_VAL=${NO_VAL})"
echo "  DEVICE              : ${DEVICE}"
echo "  checkpoint -> ${CKPT}"
echo "  results    -> ${RESULTS_DIR}/per_dataset_summary.csv"
echo "  logs       -> ${LOG_DIR}/"

# ── Sanity checks, before hours of GPU time are spent ─────────────────────────
for st in ${STAGES}; do
    case "${st}" in
        prep|finetune|forward) ;;
        *) echo "ERROR: unknown stage '${st}' (want: prep, finetune, forward)" >&2; exit 1 ;;
    esac
done

# This experiment's prepared dir already exists — don't silently re-carve it (that would
# throw away hours and, if a fine-tune were reading it, corrupt that run too).
if has_stage prep && [ -f "${PREPARED_DIR}/manifest.json" ]; then
    echo "" >&2
    echo "ERROR: ${PREPARED_DIR} is already prepared (manifest.json exists)." >&2
    echo "       Refusing to re-carve it." >&2
    echo "" >&2
    echo "       To reuse it (e.g. resuming a crashed job), switch to the resume block:" >&2
    echo "           STAGES=\"finetune forward\"" >&2
    echo "       To rebuild from scratch, move it aside first:" >&2
    echo "           mv ${PREPARED_DIR} ${PREPARED_DIR}.old" >&2
    exit 1
fi

if ! has_stage prep && [ ! -f "${PREPARED_DIR}/manifest.json" ]; then
    echo "ERROR: ${PREPARED_DIR}/manifest.json not found and 'prep' is not in STAGES." >&2
    echo "       manifest.json is written LAST, so its absence means prep never ran or never" >&2
    echo "       finished. Submit with the full-pipeline block." >&2
    exit 1
fi

if [ "${DRY_RUN}" = "1" ]; then
    banner "DRY_RUN=1 — plan printed above, nothing executed"
    exit 0
fi

# ── Stage 1: data prep (shared; only the first job runs this) ─────────────────
if has_stage prep; then
    banner "STAGE  PREP  ->  ${PREPARED_DIR}"
    echo "  Carves per-dataset train pools, test halves, and the per-file dissimilarity weights."
    echo "  Expect a warning that Exathlon is RE-CARVED rather than linked: that is the"
    echo "  data_not_used fix, and it means Exathlon's test half legitimately changed."
    OUTPUT_DIR="${PREPARED_DIR}" \
    USE_SYN_DATA="${USE_SYN_DATA}" \
    NO_VAL="${NO_VAL}" \
        bash "${SCRIPT_DIR}/run_prepare_total.sh" 2>&1 | tee "${LOG_DIR}/1_prep.log"
fi

# ── Cross-check: every experiment's prepared dir must describe the SAME data ──
# Per-experiment prepped dirs are only sound because prep is deterministic. If someone
# changes SEED, DATA_ROOT, USE_SYN_DATA, the geometry — anything — between two submissions,
# the two experiments quietly end up trained on different windows and the comparison is
# worthless. prepare_total.py records its full config in manifest.json, so compare this
# job's against every sibling prepared_total_* and stop if they disagree.
python - "${PREPARED_DIR}" "${SCRIPT_DIR}" <<'PYCHK'
import glob, json, os, sys
mine_dir, root = sys.argv[1], sys.argv[2]
IGNORE = {"output_dir"}          # differs by construction; nothing else may
def cfg(p):
    with open(p) as f:
        return {k: v for k, v in (json.load(f).get("config") or {}).items() if k not in IGNORE}
mine_path = os.path.join(mine_dir, "manifest.json")
if not os.path.exists(mine_path):
    sys.exit(0)                  # nothing prepped yet (dry run / resume) — nothing to check
mine, bad = cfg(mine_path), []
for other_path in sorted(glob.glob(os.path.join(root, "prepared_total_*", "manifest.json"))):
    if os.path.samefile(os.path.dirname(other_path), mine_dir):
        continue
    theirs = cfg(other_path)
    diff = {k: (mine.get(k), theirs.get(k)) for k in set(mine) | set(theirs)
            if mine.get(k) != theirs.get(k)}
    if diff:
        bad.append((os.path.dirname(other_path), diff))
if bad:
    print("\nERROR: this experiment's prepared data was carved with a DIFFERENT config than")
    print("       a sibling experiment's. They are not comparable — the difference in your")
    print("       final numbers would be the DATA, not the sampler.\n")
    for d, diff in bad:
        print(f"  vs {d}:")
        for k, (m, t) in sorted(diff.items()):
            print(f"      {k}: this={m!r}  other={t!r}")
    print("\n  Re-prep them with identical settings (move the odd one out aside and rerun),")
    print("  or, if the difference is intentional, delete the stale sibling.")
    sys.exit(1)
print(f"  prep config matches all sibling prepared_total_* dirs — experiments are comparable")
PYCHK

# ── Stage 2: fine-tune ───────────────────────────────────────────────────────
if has_stage finetune; then
    banner "STAGE  FINE-TUNE [${EXP_TAG}]  —  level 1.5 on: ${FILE_DIVERSITY_DATASETS:-<none>}"
    # Exported, so they win over whichever block is uncommented in run_finetune_total.sh's
    # own selector — this file is the single source of truth for the job.
    FILE_DIVERSITY_DATASETS="${FILE_DIVERSITY_DATASETS}" \
    EXP_TAG="${EXP_TAG}" \
    PREPARED_DIR="${PREPARED_DIR}" \
    OUTPUT_DIR="${OUTPUT_DIR}" \
    NO_VALIDATION="${NO_VALIDATION}" \
    DEVICE="${DEVICE}" \
        bash "${SCRIPT_DIR}/run_finetune_total.sh" 2>&1 | tee "${LOG_DIR}/2_finetune.log"
fi

# ── Stage 3: inference on the per-dataset test halves ────────────────────────
if has_stage forward; then
    if [ ! -d "${CKPT}" ]; then
        echo "ERROR: no checkpoint at ${CKPT}. Did fine-tuning run and succeed?" >&2
        echo "       See ${LOG_DIR}/2_finetune.log" >&2
        exit 1
    fi
    banner "STAGE  INFERENCE [${EXP_TAG}]  ->  ${RESULTS_DIR}"
    PREPARED_DIR="${PREPARED_DIR}" \
    CHECKPOINT="${CKPT}" \
    RESULTS_DIR="${RESULTS_DIR}" \
    DEVICE="${DEVICE}" \
        bash "${SCRIPT_DIR}/run_forward_total.sh" 2>&1 | tee "${LOG_DIR}/3_forward.log"
fi

# ── Done ─────────────────────────────────────────────────────────────────────
banner "DONE  —  ${EXP_TAG}"
echo "  Deliverable : ${RESULTS_DIR}/per_dataset_summary.csv"
echo "  Checkpoint  : ${CKPT}"
echo "  Logs        : ${LOG_DIR}/"
echo ""
echo "  Check the sampler did what you asked (file level ON for exactly the datasets you"
echo "  named, and the realized dataset mix still uniform):"
echo "    grep -E '\[hs\] level1.5|\[hs\] realized' ${LOG_DIR}/2_finetune.log"
