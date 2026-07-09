"""
Aggregate per-dataset forward.py outputs into the reported deliverable.

forward.py writes one CSV per dataset (results_dir/<DATASET>_results.csv), with ONE
ROW PER SERIES. The number we report for each dataset is the MEAN over that
dataset's series — SMD -> one value, Exathlon -> one value, ... (exactly how the
per-dataset VUS-PR was being read from the logs before).

This script produces:
  * per_dataset_summary.csv  — THE DELIVERABLE: one row per dataset, one value per
    metric (mean over that dataset's series) + n_series.
  * a printed MACRO average  — mean of the per-dataset values (every dataset weighted
    equally; matches the dataset-balanced training objective). This is the single
    headline number for comparing runs.
  * a printed MICRO average  — mean over ALL series pooled (kept only for reference;
    biased toward big datasets like MITDB, so NOT the headline).

Usage:
  python aggregate_results.py --results_dir results_FT --out_summary results_FT/per_dataset_summary.csv
"""

import argparse
import csv
import glob
import os
import statistics as st

# Metrics to summarise, in report order. Missing columns are skipped gracefully.
METRICS = [
    "VUS-PR", "VUS-ROC", "AUC-PR", "AUC-ROC",
    "Standard-F1", "PA-F1", "Event-based-F1", "R-based-F1", "Affiliation-F",
]


def _fnum(x):
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return v if v == v else None   # drop NaN


def load_dataset_rows(results_dir):
    """Return {dataset: [row_dict, ...]} from every <DATASET>_results.csv."""
    out = {}
    for path in sorted(glob.glob(os.path.join(results_dir, "*_results.csv"))):
        base = os.path.basename(path)
        if base in ("ALL_results.csv",):
            continue
        ds = base[: -len("_results.csv")]
        with open(path) as f:
            rows = list(csv.DictReader(f))
        if rows:
            out[ds] = rows
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results_dir", required=True)
    ap.add_argument("--out_summary", required=True)
    args = ap.parse_args()

    per_ds = load_dataset_rows(args.results_dir)
    if not per_ds:
        print(f"  (no <DATASET>_results.csv found in {args.results_dir})")
        return

    present = [m for m in METRICS if any(m in r for rows in per_ds.values() for r in rows)]

    # ── per-dataset means (the deliverable) ─────────────────────────────────────
    summary_rows = []
    for ds in sorted(per_ds):
        rows = per_ds[ds]
        rec = {"dataset": ds, "n_series": len(rows)}
        for m in present:
            vals = [_fnum(r.get(m)) for r in rows]
            vals = [v for v in vals if v is not None]
            rec[m] = round(st.mean(vals), 6) if vals else ""
        summary_rows.append(rec)

    os.makedirs(os.path.dirname(os.path.abspath(args.out_summary)), exist_ok=True)
    with open(args.out_summary, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["dataset", "n_series"] + present)
        w.writeheader()
        w.writerows(summary_rows)

    # ── macro (mean of per-dataset values) + micro (all series pooled) ─────────
    def macro(m):
        vals = [r[m] for r in summary_rows if isinstance(r.get(m), (int, float))]
        return st.mean(vals) if vals else None

    def micro(m):
        vals = [_fnum(r.get(m)) for rows in per_ds.values() for r in rows]
        vals = [v for v in vals if v is not None]
        return st.mean(vals) if vals else None

    n_ds = len(summary_rows)
    n_series = sum(r["n_series"] for r in summary_rows)

    print(f"  Per-dataset single-value summary ({n_ds} datasets, {n_series} series total):")
    print(f"    -> {args.out_summary}\n")
    hdr = "    {:<16}".format("dataset") + "".join(f"{m:>16}" for m in present)
    print(hdr)
    print("    " + "-" * (len(hdr) - 4))
    for r in summary_rows:
        line = "    {:<16}".format(r["dataset"][:16])
        for m in present:
            line += f"{r[m]:>16.4f}" if isinstance(r.get(m), (int, float)) else f"{'-':>16}"
        print(line)
    print("    " + "-" * (len(hdr) - 4))
    macro_line = "    {:<16}".format("MACRO (mean/ds)")
    micro_line = "    {:<16}".format("micro (all ser)")
    for m in present:
        mv, uv = macro(m), micro(m)
        macro_line += f"{mv:>16.4f}" if mv is not None else f"{'-':>16}"
        micro_line += f"{uv:>16.4f}" if uv is not None else f"{'-':>16}"
    print(macro_line)
    print(micro_line)

    mp = macro("VUS-PR")
    if mp is not None:
        print(f"\n  HEADLINE  MACRO VUS-PR = {mp:.4f}  (mean of {n_ds} per-dataset values)")


if __name__ == "__main__":
    main()
