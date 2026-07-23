#!/usr/bin/env bash
#
# Run mTSBench baseline detectors on the SAME held-out test-file half, over the SAME
# per-series index range, with the SAME metric code and aggregation as the Chronos-2
# arm (TOTAL_RUN_maskloss_v2_HS/run_forward_total.sh).
#
# What is held identical to the Chronos run:
#   * SERIES        — the 178 held-out series frozen in covered_regions.json, which is
#                     derived from that arm's prepared_total/manifest.json (seed 42,
#                     50/50 file split over the real *test.csv files).
#   * REGION        — each series is scored on [lo, hi) = the exact slice Chronos
#                     forecast (leading 512 context steps and the trailing len%64
#                     remainder are excluded, as in forward.py).
#   * METRIC CODE   — VUS_ROC_VUS_PR.get_metrics, imported from the Chronos work root
#                     (not a copy), called with slidingWindow=100, version='opt',
#                     thre=250, pred=None.
#   * SKIP RULES    — a series with 0 positives or 100% positives inside [lo, hi) is
#                     skipped, so n_series per dataset matches results_FT.
#   * AGGREGATION   — TOTAL_RUN_maskloss_v2_HS/aggregate_results.py, unmodified:
#                     per-dataset mean over series, then MACRO mean over datasets.
#
# What legitimately differs: semi-supervised detectors fit on each series' own
# *_train.csv (the mTSBench protocol). That file is normal-only reference data and is
# independent of our 50/50 split of the *test.csv files — it is not leakage into the
# held-out half.
#
# The dataset -> detector mapping comes from ./config.py. Nothing runs unless it is in
# that mapping (or you name it explicitly).
#
# Usage:
#   ./run_baseline_total.sh                              # every (dataset, model) in config.py
#   DATASETS="SMD MSL" ./run_baseline_total.sh           # those datasets, their config.py models
#   MODELS="PCA IForest" ./run_baseline_total.sh         # those models, wherever config.py lists them
#   DATASETS="SMD" MODELS="PCA" ./run_baseline_total.sh  # one pair
#   OVERWRITE=1 ./run_baseline_total.sh                  # recompute instead of resuming
#   LIST=1 ./run_baseline_total.sh                       # print the job list and exit
#
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="${PYTHON:-${HOME}/miniconda3/envs/mtsbench/bin/python}"
# Dataset tree (~4.8 GB, deliberately not vendored). Leave DATA_ROOT unset and
# run_baseline.py resolves it from $MTSBENCH_DATA or locations near the repo.
DATA_ROOT="${DATA_ROOT:-${MTSBENCH_DATA:-}}"
COVERED_JSON="${COVERED_JSON:-${SCRIPT_DIR}/covered_regions.json}"
RESULTS_ROOT="${RESULTS_ROOT:-${SCRIPT_DIR}/results_baselines}"
CONFIG_PY="${CONFIG_PY:-${SCRIPT_DIR}/config.py}"
# Byte-identical copy of the Chronos arm's aggregator — see VENDORED.md.
AGGREGATE_PY="${AGGREGATE_PY:-${SCRIPT_DIR}/aggregate_results.py}"

SLIDING_WINDOW_VUS="${SLIDING_WINDOW_VUS:-100}"
VUS_VERSION="${VUS_VERSION:-opt}"
VUS_THRE="${VUS_THRE:-250}"
SEED="${SEED:-2024}"

if [ ! -x "${PYTHON}" ]; then
    echo "ERROR: interpreter not found: ${PYTHON}"
    echo "       create it with:  conda create -n mtsbench python=3.11 -y && \\"
    echo "                        ~/miniconda3/envs/mtsbench/bin/pip install -r mTSBench/requirements.txt numba"
    exit 1
fi
if [ ! -f "${COVERED_JSON}" ]; then
    echo "ERROR: ${COVERED_JSON} not found. Build it first:"
    echo "       ${PYTHON} ${SCRIPT_DIR}/dump_covered_regions.py"
    exit 1
fi

# ── Build the (dataset, model) job list from config.py, normalizing its names ──
JOBS="$("${PYTHON}" - "${CONFIG_PY}" "${SCRIPT_DIR}" <<'PYEOF'
import sys, os, importlib.util
cfg_path, here = sys.argv[1], sys.argv[2]
sys.path.insert(0, here)
from run_baseline import norm_dataset, norm_model

spec = importlib.util.spec_from_file_location("cfg", cfg_path)
cfg = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cfg)

only_ds = {norm_dataset(d) for d in os.environ.get("DATASETS", "").split()}
only_md = {norm_model(m) for m in os.environ.get("MODELS", "").split()}

for raw_ds, models in cfg.MERGED_MODELS.items():
    ds = norm_dataset(raw_ds)
    if only_ds and ds not in only_ds:
        continue
    seen = set()
    for raw_m in models:
        m = norm_model(raw_m)
        if m in seen:                      # config.py repeats a name after normalization
            continue
        seen.add(m)
        if only_md and m not in only_md:
            continue
        print(f"{ds}\t{m}")
PYEOF
)"

if [ -z "${JOBS}" ]; then
    echo "No (dataset, model) pairs matched. DATASETS='${DATASETS:-}' MODELS='${MODELS:-}'"
    exit 1
fi

N_JOBS="$(echo "${JOBS}" | wc -l)"
echo "======================================================================"
echo "  mTSBench baselines on the Chronos-2 held-out split"
echo "  PYTHON       = ${PYTHON}"
echo "  DATA_ROOT    = ${DATA_ROOT:-<auto-detected by run_baseline.py>}"
echo "  COVERED      = ${COVERED_JSON}"
echo "  RESULTS_ROOT = ${RESULTS_ROOT}"
echo "  JOBS         = ${N_JOBS}  (dataset, model) pairs"
echo "======================================================================"
echo "${JOBS}" | awk -F'\t' '{printf "    %-16s %s\n", $1, $2}'
if [ -n "${LIST:-}" ]; then exit 0; fi

FAILED=""
while IFS=$'\t' read -r ds model; do
    [ -z "${ds}" ] && continue
    echo ""
    ARGS=(
        --dataset            "${ds}"
        --model              "${model}"
        --covered_json       "${COVERED_JSON}"
        --results_dir        "${RESULTS_ROOT}/${model}"
        --sliding_window_VUS "${SLIDING_WINDOW_VUS}"
        --vus_version        "${VUS_VERSION}"
        --vus_thre           "${VUS_THRE}"
        --seed               "${SEED}"
    )
    [ -n "${DATA_ROOT}" ] && ARGS+=(--data_root "${DATA_ROOT}")
    [ -n "${OVERWRITE:-}" ] && ARGS+=(--overwrite)

    if ! "${PYTHON}" -u "${SCRIPT_DIR}/run_baseline.py" "${ARGS[@]}"; then
        echo "  !! ${model} on ${ds} exited non-zero — continuing with the rest"
        FAILED="${FAILED}\n    ${ds}\t${model}"
    fi
done <<< "${JOBS}"

# ── Per-model aggregation, using the Chronos arm's aggregator unmodified ──────
echo ""
echo "======================================================================"
for d in "${RESULTS_ROOT}"/*/; do
    [ -d "${d}" ] || continue
    compgen -G "${d}/*_results.csv" > /dev/null || continue
    echo ""
    echo "###### $(basename "${d}") ######"
    "${PYTHON}" -u "${AGGREGATE_PY}" \
        --results_dir "${d}" \
        --out_summary "${d}/per_dataset_summary.csv"
done

if [ -n "${FAILED}" ]; then
    echo ""
    echo "  Jobs that exited non-zero:"
    echo -e "${FAILED}"
fi
echo ""
echo "  Per-model deliverable : ${RESULTS_ROOT}/<MODEL>/per_dataset_summary.csv"
echo "  Cross-model table     : ${PYTHON} ${SCRIPT_DIR}/summarize_baselines.py"
echo "======================================================================"
