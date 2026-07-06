"""
Compare zero-shot vs fine-tuned per-dataset evaluation results.

Reads the two ALL_results.csv files (default: results/ = zero-shot,
results_finetuned/ = fine-tuned), joins on (dataset, series_id), and reports
per-series and per-dataset deltas, split by whether the dataset actually
contributed windows to the combined training set (from manifest.json).

    zero-shot  : results/ALL_results.csv          (context-only, --no_normal_prefix)
    fine-tuned : results_finetuned/ALL_results.csv (with normal prefix)

Usage:
    python compare_results.py
    python compare_results.py --metric VUS-PR
    python compare_results.py --base results/ALL_results.csv --ft results_finetuned/ALL_results.csv
"""
import argparse, json, os
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))

ap = argparse.ArgumentParser()
ap.add_argument("--base", default=os.path.join(HERE, "results", "ALL_results.csv"),
                help="zero-shot ALL_results.csv")
ap.add_argument("--ft", default=os.path.join(HERE, "results_finetuned", "ALL_results.csv"),
                help="fine-tuned ALL_results.csv")
ap.add_argument("--manifest", default=os.path.join(HERE, "prepared_total", "manifest.json"))
ap.add_argument("--metric", default="VUS-PR",
                help="primary metric for the per-dataset summary")
ap.add_argument("--out", default=os.path.join(HERE, "comparison.csv"))
args = ap.parse_args()

METRICS = ["VUS-PR", "VUS-ROC", "AUC-PR", "AUC-ROC",
           "Standard-F1", "PA-F1", "Event-based-F1", "R-based-F1", "Affiliation-F"]

b = pd.read_csv(args.base)
f = pd.read_csv(args.ft)

# which datasets contributed to training (file_split w/ kept windows) vs test_only
in_train = {}
if os.path.exists(args.manifest):
    m = json.load(open(args.manifest))
    bal = m.get("balance", {}).get("datasets", {})
    for name, info in m["datasets"].items():
        kept = bal.get(name, {}).get("anomalous_kept", 0) + bal.get(name, {}).get("normal_kept", 0)
        in_train[name] = (info.get("mode") == "file_split") and kept > 0

key = ["dataset", "series_id"]
j = b.merge(f, on=key, suffixes=("_base", "_ft"))
if j.empty:
    print("No overlapping (dataset, series_id) rows yet — let both runs finish.")
    print(f"  base has {len(b)} rows across {b['dataset'].nunique()} datasets")
    print(f"  ft   has {len(f)} rows across {f['dataset'].nunique()} datasets")
    raise SystemExit(0)

for mt in METRICS:
    if f"{mt}_base" in j and f"{mt}_ft" in j:
        j[f"d_{mt}"] = j[f"{mt}_ft"] - j[f"{mt}_base"]

j["role"] = j["dataset"].map(lambda d: "train" if in_train.get(d) else "test_only")

# ---- per-dataset summary on the primary metric ----
mt = args.metric
grp = (j.groupby(["role", "dataset"])
         .agg(n=("series_id", "size"),
              base=(f"{mt}_base", "mean"),
              ft=(f"{mt}_ft", "mean"))
         .reset_index())
grp["delta"] = grp["ft"] - grp["base"]

print(f"\n=== Per-dataset mean {mt}  (fine-tuned − zero-shot) ===")
print(f"{'role':<10}{'dataset':<16}{'n':>4}{'zero-shot':>11}{'fine-tuned':>12}{'delta':>10}")
for role in ["train", "test_only"]:
    sub = grp[grp.role == role].sort_values("delta", ascending=False)
    if sub.empty:
        continue
    print("-" * 63)
    for _, r in sub.iterrows():
        flag = "  ▲" if r.delta > 0 else ("  ▼" if r.delta < 0 else "")
        print(f"{role:<10}{r.dataset:<16}{int(r.n):>4}{r.base:>11.4f}{r.ft:>12.4f}{r.delta:>+10.4f}{flag}")

# ---- headline: macro-avg over datasets, per role, for every metric ----
print(f"\n=== Macro-avg over datasets (mean per dataset, then mean over datasets) ===")
print(f"{'metric':<16}{'role':<11}{'zero-shot':>11}{'fine-tuned':>12}{'delta':>10}")
for mt in METRICS:
    if f"{mt}_base" not in j:
        continue
    per_ds = j.groupby(["role", "dataset"])[[f"{mt}_base", f"{mt}_ft"]].mean().reset_index()
    for role in ["train", "test_only"]:
        sub = per_ds[per_ds.role == role]
        if sub.empty:
            continue
        bb, ff = sub[f"{mt}_base"].mean(), sub[f"{mt}_ft"].mean()
        print(f"{mt:<16}{role:<11}{bb:>11.4f}{ff:>12.4f}{ff-bb:>+10.4f}")

j.to_csv(args.out, index=False)
print(f"\nper-series join written -> {args.out}  ({len(j)} rows)")
