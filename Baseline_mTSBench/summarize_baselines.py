"""
Cross-model view of the baseline results, next to the Chronos-2 numbers — ALL 9 METRICS.

Reads every results_baselines/<MODEL>/per_dataset_summary.csv produced by
aggregate_results.py, plus the Chronos-2 reference summaries vendored under
chronos_reference/, and reports every metric the pipeline computes:

    VUS-PR, VUS-ROC, AUC-PR, AUC-ROC,
    Standard-F1, PA-F1, Event-based-F1, R-based-F1, Affiliation-F

Every number is a per-dataset mean over the same held-out series, computed on the same
index range by the same metric code, so the columns are directly comparable.

Output:
  * printed — one dataset x {detectors, Chronos} table per metric, each with a MACRO
    row; then an all-metric leaderboard (best config.py detector per dataset vs Chronos).
  * --out    — LONG-format CSV: one row per (dataset, source) carrying all 9 metrics,
    where `source` is a detector name or a Chronos run. Pivot it however the paper needs.

All paths are relative to this file, so a clone works anywhere.

Usage:
    python summarize_baselines.py                           # all 9 metrics
    python summarize_baselines.py --metrics VUS-PR VUS-ROC  # a subset
    python summarize_baselines.py --out baseline_vs_chronos.csv
"""

import argparse
import csv
import glob
import importlib.util
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
from run_baseline import norm_dataset, norm_model  # noqa: E402

# Chronos results are vendored under ./chronos_reference/ so this runs standalone.
# Pass --chronos_dirs to point at a live Chronos workspace for fresher numbers
# (e.g. once results_ZS exists).
_CHRONOS_DIR = os.path.join(_HERE, "chronos_reference")

ALL_METRICS = ["VUS-PR", "VUS-ROC", "AUC-PR", "AUC-ROC",
               "Standard-F1", "PA-F1", "Event-based-F1", "R-based-F1", "Affiliation-F"]


def read_summary(path, metrics):
    """{dataset: {"n_series": int, metric: float|None, ...}} from a per_dataset_summary.csv."""
    out = {}
    if not os.path.exists(path):
        return out
    with open(path) as f:
        for r in csv.DictReader(f):
            rec = {}
            try:
                rec["n_series"] = int(r["n_series"])
            except (KeyError, TypeError, ValueError):
                rec["n_series"] = None
            for m in metrics:
                try:
                    rec[m] = float(r[m])
                except (KeyError, TypeError, ValueError):
                    rec[m] = None
            out[r["dataset"]] = rec
    return out


def load_config(path):
    spec = importlib.util.spec_from_file_location("cfg", path)
    cfg = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cfg)
    mapping = {}
    for raw_ds, models in cfg.MERGED_MODELS.items():
        seen, keep = set(), []
        for m in models:
            n = norm_model(m)
            if n not in seen:
                seen.add(n)
                keep.append(n)
        mapping[norm_dataset(raw_ds)] = keep
    return mapping


def fmt(v, w=14):
    return f"{v:>{w}.4f}" if isinstance(v, float) else f"{'-':>{w}}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--metrics", nargs="*", default=ALL_METRICS,
                    help=f"Subset to report (default: all 9 — {', '.join(ALL_METRICS)})")
    ap.add_argument("--results_root", default=os.path.join(_HERE, "results_baselines"))
    ap.add_argument("--config", default=os.path.join(_HERE, "config.py"))
    ap.add_argument("--chronos_dirs", nargs="*",
                    default=[os.path.join(_CHRONOS_DIR, "results_ZS"),
                             os.path.join(_CHRONOS_DIR, "results_FT")],
                    help="Chronos results dirs to show as reference columns")
    ap.add_argument("--out", default=None, help="Long-format CSV carrying every metric")
    args = ap.parse_args()

    metrics = [m for m in args.metrics if m in ALL_METRICS]
    bad = [m for m in args.metrics if m not in ALL_METRICS]
    if bad:
        print(f"  ignoring unknown metric(s): {', '.join(bad)}")
    if not metrics:
        sys.exit(f"No valid metrics. Choose from: {', '.join(ALL_METRICS)}")

    cfg = load_config(args.config)

    baselines = {}
    for d in sorted(glob.glob(os.path.join(args.results_root, "*", ""))):
        model = os.path.basename(os.path.dirname(d))
        s = read_summary(os.path.join(d, "per_dataset_summary.csv"), metrics)
        if s:
            baselines[model] = s
    if not baselines:
        print(f"No per_dataset_summary.csv under {args.results_root}. "
              f"Run ./run_baseline_total.sh first.")
        return

    chronos = {}
    for cd in args.chronos_dirs:
        s = read_summary(os.path.join(cd, "per_dataset_summary.csv"), metrics)
        if s:
            chronos[f"Chronos2-{os.path.basename(cd).replace('results_', '')}"] = s

    datasets = sorted({ds for s in baselines.values() for ds in s})
    models = sorted(baselines)
    ch_cols = sorted(chronos)

    # ── one table per metric ─────────────────────────────────────────────────
    for metric in metrics:
        print(f"\n{'=' * 78}\n  {metric}   "
              f"(per-dataset mean over held-out series, Chronos covered region)\n")
        hdr = "  {:<16}{:>4}".format("dataset", "n")
        hdr += "".join(f"{m:>14}" for m in models)
        hdr += "".join(f"{c:>16}" for c in ch_cols)
        print(hdr)
        print("  " + "-" * (len(hdr) - 2))
        acc = {k: [] for k in models + ch_cols}
        for ds in datasets:
            n = next((baselines[m][ds]["n_series"] for m in models if ds in baselines[m]), None)
            line = "  {:<16}{:>4}".format(ds, n if n is not None else "-")
            for m in models:
                # blank unless config.py actually asks for this detector on this dataset
                v = baselines[m].get(ds, {}).get(metric) if m in cfg.get(ds, []) else None
                line += fmt(v)
                if isinstance(v, float):
                    acc[m].append(v)
            for c in ch_cols:
                v = chronos[c].get(ds, {}).get(metric)
                line += fmt(v, 16)
                if isinstance(v, float):
                    acc[c].append(v)
            print(line)
        print("  " + "-" * (len(hdr) - 2))
        line = "  {:<16}{:>4}".format("MACRO (mean/ds)", "")
        for m in models:
            line += fmt(sum(acc[m]) / len(acc[m]) if acc[m] else None)
        for c in ch_cols:
            line += fmt(sum(acc[c]) / len(acc[c]) if acc[c] else None, 16)
        print(line)
        print("  (blank = config.py does not list that detector for that dataset, "
              "or it has not been run yet)")

    # ── all-metric leaderboard ───────────────────────────────────────────────
    print(f"\n{'=' * 78}\n  BEST config.py DETECTOR PER DATASET vs CHRONOS-2 — all metrics\n")
    hdr = "  {:<16}{:>14}{:>10}".format("metric", "best (macro)", "datasets")
    hdr += "".join(f"{c:>18}" for c in ch_cols)
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    for metric in metrics:
        best_vals, wins, n_cmp = [], {c: 0 for c in ch_cols}, {c: 0 for c in ch_cols}
        for ds in datasets:
            cand = [(baselines[m][ds][metric], m) for m in cfg.get(ds, [])
                    if isinstance(baselines.get(m, {}).get(ds, {}).get(metric), float)]
            if not cand:
                continue
            bv, _ = max(cand)
            best_vals.append(bv)
            for c in ch_cols:
                v = chronos[c].get(ds, {}).get(metric)
                if isinstance(v, float):
                    n_cmp[c] += 1
                    if v > bv:
                        wins[c] += 1
        if not best_vals:
            continue
        line = "  {:<16}".format(metric) + fmt(sum(best_vals) / len(best_vals))
        line += "{:>10}".format(len(best_vals))
        for c in ch_cols:
            line += "{:>18}".format(f"{wins[c]}/{n_cmp[c]} win" if n_cmp[c] else "-")
        print(line)
    print("\n  'win' = Chronos beats the best config.py detector on that dataset, "
          "for that metric.")

    # ── long-format CSV: one row per (dataset, source), all metrics ───────────
    if args.out:
        rows = []
        for ds in datasets:
            for m in cfg.get(ds, []):
                rec = baselines.get(m, {}).get(ds)
                if rec:
                    rows.append({"dataset": ds, "source": m, "kind": "baseline",
                                 "n_series": rec["n_series"],
                                 **{k: rec.get(k) for k in metrics}})
            for c in ch_cols:
                rec = chronos[c].get(ds)
                if rec:
                    rows.append({"dataset": ds, "source": c, "kind": "chronos",
                                 "n_series": rec["n_series"],
                                 **{k: rec.get(k) for k in metrics}})
        with open(args.out, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["dataset", "source", "kind", "n_series"] + metrics)
            w.writeheader()
            w.writerows(rows)
        print(f"\n  Wrote {args.out}  ({len(rows)} rows x {len(metrics)} metrics, long format)")


if __name__ == "__main__":
    main()
