"""
Score ONE mTSBench baseline detector on ONE dataset's held-out series, on exactly
the index range the Chronos-2 arm scored.

Why this script exists
----------------------
`TOTAL_RUN_maskloss_v2_HS/forward.py` evaluates Chronos-2 per series on the slice
[lo, hi) = [context_length, context_length + n_windows*prediction_length): the leading
512 context steps are never forecast and the trailing `len % 64` steps fall outside the
last window. mTSBench's detectors, by contrast, emit a score for every timestep.

To compare them series-for-series we therefore:
  1. run the detector on the FULL test.csv — it needs all of its input, exactly as
     mTSBench itself runs it (nothing about the detector's fit changes);
  2. slice its per-timestep score to Chronos's [lo, hi) (frozen in covered_regions.json
     by dump_covered_regions.py);
  3. call the SAME get_metrics with the SAME arguments Chronos used
     (slidingWindow=100, version='opt', thre=250, pred=None -> oracle threshold).

We import get_metrics from the vendored `./VUS_ROC_VUS_PR` package rather than from
`Detectors.evaluation.metrics`. The two are equivalent for these 9 metrics — mTSBench's
`basic_metrics.py` is identical to `VUS_ROC_VUS_PR/basic_metrics.py` once comments and
whitespace are normalized, and its `get_metrics` only APPENDS extra metrics after
computing the same AUC_ROC / AUC_PR / generate_curve core. Using the Chronos arm's own
copy makes "same metric code" a fact rather than a claim, and avoids pulling in the
extra `range_metrics` dependency.

`VUS_ROC_VUS_PR/` here is a byte-identical copy of
`Chronos_Finetuning/rajib_work_space/VUS_ROC_VUS_PR/`, vendored so this directory runs
standalone with no path dependency on the Chronos workspace. See VENDORED.md for the
provenance and the md5 check that verifies it has not drifted.

Series selection and the two skip rules also mirror forward.py exactly, so `n_series`
per dataset matches the Chronos results table:
  * only series present in covered_regions.json are scored (= the held-out 50% file
    half that produced at least one window);
  * a series whose covered region has 0 anomalies is skipped (PR/ROC undefined);
  * a series whose covered region is 100% anomalous is skipped (no negatives; also
    crashes range_convers_new).

Output: results_dir/<DATASET>_results.csv, one row per scored series, with the same 9
metric columns forward.py writes — so aggregate_results.py consumes it unchanged.

Usage:
    python run_baseline.py --dataset SMD --model PCA
    python run_baseline.py --dataset MSL --model CNN --results_dir results_baselines/CNN
"""

import argparse
import csv
import glob
import json
import os
import random
import sys
import time

import numpy as np
import pandas as pd
from tqdm import tqdm

_HERE = os.path.dirname(os.path.abspath(__file__))
_MTSBENCH = os.path.join(_HERE, "mTSBench")

# Detectors is a package under mTSBench/; VUS_ROC_VUS_PR is vendored beside this file.
sys.path.insert(0, _MTSBENCH)
sys.path.insert(0, _HERE)

# Columns written, in forward.py's order.
METRIC_KEYS = ("VUS-PR", "VUS-ROC", "AUC-PR", "AUC-ROC",
               "Standard-F1", "PA-F1", "Event-based-F1", "R-based-F1", "Affiliation-F")


# ──────────────────────────────────────────────────────────────────────────────
#  Name normalization — config.py's keys/values are hand-written and misspelled
# ──────────────────────────────────────────────────────────────────────────────
DATASET_ALIASES = {
    "calit2": "CalIt2", "gecco": "GECCO", "ghl": "GHL",
    "room-occu": "room-occupancy", "roomoccupancy": "room-occupancy",
    "swan": "swan", "metro": "metro", "cicids": "cicids", "creditcard": "creditcard",
    "daphnet": "Daphnet", "exathlon": "Exathlon", "genesis": "Genesis",
    "gutentag": "GutenTAG", "mitdb": "MITDB", "msl": "MSL",
    "opportunity": "OPPORTUNITY", "psm": "PSM", "smap": "SMAP", "smd": "SMD",
    "svdb": "SVDB",
}

MODEL_ALIASES = {
    "omnianoomaly": "OmniAnomaly", "onmianomaly": "OmniAnomaly",
    "omnianomaly": "OmniAnomaly",
    "oaf": "OFA", "ofa": "OFA",
    "kmeansad": "KMeansAD", "kmeanad": "KMeansAD",
    "donut": "Donut", "iforest": "IForest", "lof": "LOF", "knn": "KNN",
    "pca": "PCA", "mcd": "MCD", "ocsvm": "OCSVM", "copod": "COPOD",
    "cblof": "CBLOF", "hbos": "HBOS", "eif": "EIF", "robustpca": "RobustPCA",
    "autoencoder": "AutoEncoder", "cnn": "CNN", "lstmad": "LSTMAD",
    "tranad": "TranAD", "usad": "USAD", "allm4ts": "ALLM4TS",
    "anomalytransformer": "AnomalyTransformer", "timesnet": "TimesNet",
    "fits": "FITS",
}


def norm_dataset(name):
    return DATASET_ALIASES.get(name.strip().lower().replace("_", "-"), name.strip())


def norm_model(name):
    return MODEL_ALIASES.get(name.strip().lower(), name.strip())


# ──────────────────────────────────────────────────────────────────────────────
#  Data loading — must match the Chronos prep's column selection EXACTLY
# ──────────────────────────────────────────────────────────────────────────────
def load_csv(path):
    """Return (data (T, n_features) float64, labels (T,) int).

    Column selection mirrors prepare_smd_split.load_csv_as_multivariate: every column
    that is not `timestamp` or `is_anomaly`. Rows are NOT dropped — no *test.csv in
    mTSBench has a NaN (verified against Datasets/data_summary.csv), and dropping rows
    would shift the indices that covered_regions.json refers to.
    """
    df = pd.read_csv(path)
    feature_cols = [c for c in df.columns if c not in ("timestamp", "is_anomaly")]
    data = df[feature_cols].values.astype(float)
    labels = (df["is_anomaly"].values.astype(int) if "is_anomaly" in df.columns
              else np.zeros(len(df), dtype=int))
    return data, labels


def minmax(x):
    """Scale to [0,1]. Monotone, so it does not change any metric here (all 9 are
    rank/threshold-sweep based) — applied only to match mTSBench's own convention."""
    x = np.asarray(x, dtype=float).ravel()
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    lo, hi = x.min(), x.max()
    return np.zeros_like(x) if hi - lo < 1e-12 else (x - lo) / (hi - lo)


def looks_like_data_root(path):
    """True only if `path` actually holds mTSBench data.

    Checking `isdir` alone is not enough: the upstream clone ships an EMPTY
    `mTSBench/Datasets/mTSBench/` containing just a download script, and picking
    that up silently yields 'MISSING file' for every single series. Require real
    dataset sub-directories with *test.csv inside.
    """
    if not path or not os.path.isdir(path):
        return False
    hits = 0
    for entry in os.listdir(path):
        d = os.path.join(path, entry)
        if os.path.isdir(d) and glob.glob(os.path.join(d, "*test.csv")):
            hits += 1
            if hits >= 3:
                return True
    return False


def find_data_root():
    """Locate the mTSBench CSV tree without hardcoding one machine's layout.

    The dataset is ~4.8 GB, so it is NOT vendored into this repo — it is the one
    thing a fresh clone must be pointed at. Resolution order:

        1. --data_root on the command line
        2. $MTSBENCH_DATA
        3. common locations relative to this file (clone-anywhere friendly)

    Every candidate must pass looks_like_data_root(), so an empty placeholder
    directory is skipped rather than silently used.
    """
    env = os.environ.get("MTSBENCH_DATA")
    if env:
        return env                      # explicit wins; validated by the caller
    for cand in (os.path.join(_HERE, "Datasets", "mTSBench"),
                 os.path.join(_HERE, "mTSBench", "Datasets", "mTSBench"),
                 os.path.join(_HERE, "..", "Datasets", "mTSBench"),
                 os.path.join(_HERE, "..", "..", "mTSBench", "Datasets", "mTSBench"),
                 os.path.expanduser("~/mTSBench/Datasets/mTSBench")):
        if looks_like_data_root(cand):
            return os.path.normpath(cand)
    return None


def seed_everything(seed=2024):
    """Same seeding block as mTSBench's Detectors/main.py."""
    import torch
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--data_root", default=None,
                    help="mTSBench CSV tree. Default: $MTSBENCH_DATA, else auto-detected "
                         "near this file (the ~4.8 GB dataset is not vendored).")
    ap.add_argument("--covered_json", default=os.path.join(_HERE, "covered_regions.json"))
    ap.add_argument("--results_dir", default=None,
                    help="Default: results_baselines/<MODEL>")
    ap.add_argument("--sliding_window_VUS", type=int, default=100)
    ap.add_argument("--vus_version", default="opt", choices=["opt", "opt_mem"])
    ap.add_argument("--vus_thre", type=int, default=250)
    ap.add_argument("--seed", type=int, default=2024)
    ap.add_argument("--overwrite", action="store_true",
                    help="Recompute series already present in the output CSV "
                         "(default: resume, skipping them)")
    args = ap.parse_args()

    ds = norm_dataset(args.dataset)
    model = norm_model(args.model)

    args.data_root = args.data_root or find_data_root()
    if not looks_like_data_root(args.data_root):
        sys.exit(f"ERROR: no mTSBench data at {args.data_root!r}\n"
                 "       (a directory counts only if it holds dataset sub-dirs with\n"
                 "       *test.csv inside — the upstream clone ships an EMPTY\n"
                 "       mTSBench/Datasets/mTSBench/ that is deliberately skipped).\n"
                 "       Point at the real tree:\n"
                 "         export MTSBENCH_DATA=/path/to/Datasets/mTSBench\n"
                 "       or pass --data_root. It is ~4.8 GB and not vendored here.")
    print(f"  data_root = {args.data_root}")

    from Detectors.model_wrapper import (run_Unsupervise_AD, run_Semisupervise_AD,
                                         Unsupervise_AD_Pool, Semisupervise_AD_Pool)
    from Detectors.HP_list import Optimal_Multi_algo_HP_dict
    from VUS_ROC_VUS_PR.metrics import get_metrics

    if model in Semisupervise_AD_Pool:
        kind = "semi"
    elif model in Unsupervise_AD_Pool:
        kind = "unsup"
    else:
        sys.exit(f"ERROR: '{model}' is in neither AD pool of Detectors/model_wrapper.py")

    hp = Optimal_Multi_algo_HP_dict.get(model, {})
    if model not in Optimal_Multi_algo_HP_dict:
        print(f"  NOTE: no entry for {model} in Optimal_Multi_algo_HP_dict — "
              f"using the wrapper's default hyper-parameters.")

    with open(args.covered_json) as f:
        covered = json.load(f)
    if ds not in covered["datasets"]:
        sys.exit(f"ERROR: dataset '{ds}' not in {args.covered_json} "
                 f"(have: {', '.join(sorted(covered['datasets']))})")
    series = covered["datasets"][ds]

    results_dir = args.results_dir or os.path.join(_HERE, "results_baselines", model)
    os.makedirs(results_dir, exist_ok=True)
    out_csv = os.path.join(results_dir, f"{ds}_results.csv")
    log_json = os.path.join(results_dir, f"{ds}_runlog.json")

    # ── Resume: keep rows already computed ───────────────────────────────────
    done_rows, done_ids = [], set()
    if os.path.exists(out_csv) and not args.overwrite:
        with open(out_csv) as f:
            done_rows = list(csv.DictReader(f))
        done_ids = {r["series_id"] for r in done_rows}
        if done_ids:
            print(f"  resuming: {len(done_ids)} series already in {out_csv}")

    runlog = {"dataset": ds, "model": model, "kind": kind, "hp": hp,
              "sliding_window_VUS": args.sliding_window_VUS,
              "vus_version": args.vus_version, "vus_thre": args.vus_thre,
              "skipped": [], "failed": [], "runtime_s": {}}

    print("=" * 70)
    print(f"  {model}  ({kind}-supervised)  on  {ds}   [{len(series)} held-out series]")
    print(f"  HP        = {hp}")
    print(f"  region    = Chronos covered slice [lo, hi) from {os.path.basename(args.covered_json)}")
    print(f"  metrics   = VUS_ROC_VUS_PR.get_metrics(slidingWindow={args.sliding_window_VUS}, "
          f"version='{args.vus_version}', thre={args.vus_thre}, pred=None)")
    print(f"  out       = {out_csv}")
    print("=" * 70)

    # Series-level progress. The deep detectors print their own per-batch tqdm bars,
    # but the classical ones (PCA, IForest, MCD, OCSVM, EIF, ...) are completely
    # silent for the whole fit — on a 520K-step MITDB series that is a long time with
    # no output at all. The bar goes to stderr, result lines to stdout, so redirecting
    # stdout to a log keeps the log clean and still shows progress on the terminal.
    todo = [s for s in sorted(series) if s not in done_ids]
    rows = list(done_rows)
    bar = tqdm(todo, desc=f"{model}/{ds}", unit="series", dynamic_ncols=True)
    for sid in bar:
        info = series[sid]
        lo, hi = info["lo"], info["hi"]
        test_path = os.path.join(args.data_root, ds, sid)
        if not os.path.exists(test_path):
            tqdm.write(f"  {sid}: MISSING file, skipping")
            runlog["failed"].append({"series": sid, "reason": "test csv not found"})
            continue

        data, labels = load_csv(test_path)
        if len(data) != info["length"]:
            tqdm.write(f"  {sid}: length {len(data)} != {info['length']} recorded by the "
                  f"Chronos prep — refusing to score (index alignment unsafe)")
            runlog["failed"].append({"series": sid, "reason":
                                     f"length mismatch {len(data)} vs {info['length']}"})
            continue

        # ── Run the detector on the FULL series ──────────────────────────────
        # Announce BEFORE the call — for a silent detector this is the only sign of
        # life until it returns, which can be an hour on MITDB.
        bar.set_postfix_str(f"{sid[:28]} T={len(data)} d={data.shape[1]}", refresh=True)
        seed_everything(args.seed)
        t0 = time.time()
        if kind == "semi":
            train_path = os.path.join(args.data_root, ds, sid.replace("_test.csv", "_train.csv"))
            if not os.path.exists(train_path):
                tqdm.write(f"  {sid}: no matching _train.csv for a semi-supervised detector, skipping")
                runlog["failed"].append({"series": sid, "reason": "train csv not found"})
                continue
            data_train, _ = load_csv(train_path)
            output = run_Semisupervise_AD(model, data_train, data, **hp)
        else:
            output = run_Unsupervise_AD(model, data, **hp)
        elapsed = time.time() - t0

        if not isinstance(output, np.ndarray):
            tqdm.write(f"  {sid}: detector returned an error -> {output}")
            runlog["failed"].append({"series": sid, "reason": str(output)})
            continue

        score_full = np.asarray(output).ravel()
        if len(score_full) != len(data):
            # Every detector here pads its own windowed score back to n_samples; a
            # mismatch means an unexpected code path, and guessing where to pad would
            # silently misalign the labels. Record and move on rather than fabricate.
            tqdm.write(f"  {sid}: score length {len(score_full)} != series length {len(data)} "
                  f"— refusing to guess the alignment, skipping")
            runlog["failed"].append({"series": sid, "reason":
                                     f"score len {len(score_full)} vs data len {len(data)}"})
            continue

        # ── Slice to Chronos's covered region, then evaluate ─────────────────
        y_score = minmax(score_full[lo:hi])
        y_true = labels[lo:hi].astype(int)
        n_pos = int(y_true.sum())
        if n_pos == 0:
            tqdm.write(f"  {sid}: no anomalies in covered region, skipping")
            runlog["skipped"].append({"series": sid, "reason": "0 positives in [lo,hi)"})
            continue
        if n_pos == len(y_true):
            tqdm.write(f"  {sid}: covered region is 100% anomalous ({n_pos} steps), skipping")
            runlog["skipped"].append({"series": sid, "reason": "100% anomalous in [lo,hi)"})
            continue

        res = get_metrics(y_score, y_true,
                          slidingWindow=args.sliding_window_VUS,
                          version=args.vus_version, thre=args.vus_thre)
        tqdm.write(f"  {sid:<34} VUS-PR={res['VUS-PR']:.4f}  VUS-ROC={res['VUS-ROC']:.4f}  "
              f"AUC-PR={res['AUC-PR']:.4f}  AUC-ROC={res['AUC-ROC']:.4f}  ({elapsed:.1f}s)")

        rows.append({"series_id": sid, **{k: res[k] for k in METRIC_KEYS}})
        runlog["runtime_s"][sid] = round(elapsed, 2)

        # Write after every series so a crash / kill never loses completed work.
        with open(out_csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["series_id"] + list(METRIC_KEYS))
            w.writeheader()
            w.writerows(rows)
        with open(log_json, "w") as f:
            json.dump(runlog, f, indent=1)

    if not rows:
        print(f"\n  No series scored for {model} on {ds}.")
        with open(log_json, "w") as f:
            json.dump(runlog, f, indent=1)
        return

    print(f"\n  ---- MEAN OVER {len(rows)} SERIES ({model} / {ds}) ----")
    for k in METRIC_KEYS:
        print(f"    {k:<16}: {np.mean([float(r[k]) for r in rows]):.4f}")
    print(f"\n  Wrote {out_csv}")


if __name__ == "__main__":
    main()
