#!/usr/bin/env python
"""
Build the study's deliverable table from the two scored runs:

    results/family_ft/per_dataset_summary.csv   the ONE pooled model, fine-tuned on the
                                                training datasets of every family
    results/zeroshot/per_dataset_summary.csv    the base model, no fine-tuning

Both are scored on the SAME thing: 100% of every held-out dataset's *test.csv files, none
of which contributed a single training window.

The question the table answers:

    Does fine-tuning on a pool of datasets -- in which every held-out dataset has at least
    one same-family sibling -- beat zero-shot on datasets the model has never seen?

Read the 'sibling' column: it names what the model saw from that dataset's own family.
That is the only thing standing between the pool and the held-out dataset.

Usage:
    python summarize_study.py
    python summarize_study.py --metric VUS-ROC
"""

import argparse
import csv
import json
import os

_HERE = os.path.dirname(os.path.abspath(__file__))


def holdout_list(fam):
    """A family's holdout as a list. Accepts the historical string form too."""
    h = fam["holdout"]
    return list(h) if isinstance(h, (list, tuple)) else [h]


def read_summary(path, metric):
    """-> {dataset: value} for the requested metric."""
    if not os.path.exists(path):
        return {}
    out = {}
    with open(path) as f:
        for r in csv.DictReader(f):
            try:
                out[r["dataset"]] = float(r[metric])
            except (KeyError, TypeError, ValueError):
                pass
    return out


def fmt(v, width):
    return f"{v:>{width}.4f}" if isinstance(v, float) else f"{'-':>{width}}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results_dir", default=os.path.join(_HERE, "results"))
    ap.add_argument("--families", default=os.path.join(_HERE, "families.json"))
    ap.add_argument("--metric", default="VUS-PR")
    ap.add_argument("--out_csv", default=os.path.join(_HERE, "results", "study_summary.csv"))
    args = ap.parse_args()

    with open(args.families) as f:
        fams = json.load(f)["families"]

    zs = read_summary(os.path.join(args.results_dir, "zeroshot",
                                   "per_dataset_summary.csv"), args.metric)
    ft = read_summary(os.path.join(args.results_dir, "family_ft",
                                   "per_dataset_summary.csv"), args.metric)
    if not zs and not ft:
        raise SystemExit(
            f"No scored results under {args.results_dir}.\n"
            f"Expected results/zeroshot/ and results/family_ft/per_dataset_summary.csv")

    pool = []
    for fam in fams.values():
        for ds in fam["train"]:
            if ds not in pool:
                pool.append(ds)

    print(f"\n{'=' * 104}")
    print(f"  FAMILY TRANSFER -- ONE pooled model, {args.metric} on datasets it never saw")
    print(f"  TRAIN pool ({len(pool)}): {' '.join(pool)}")
    print(f"{'=' * 104}")
    hdr = (f"  {'family':<13}{'held out':<14}{'same-family sibling(s) in pool':<32}"
           f"{'zero-shot':>10}{'family-FT':>11}{'delta':>9}{'rel %':>9}  tier")
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))

    rows = []
    for name, fam in fams.items():
        tier = fam.get("tier", 1)
        # One row per held-out dataset. A family may hold out more than one (env_iot,
        # server); each is credited to that family's whole train list as its sibling(s).
        for ds in holdout_list(fam):
            a, b = zs.get(ds), ft.get(ds)
            delta = (b - a) if isinstance(a, float) and isinstance(b, float) else None
            rel = (100.0 * delta / a) if isinstance(delta, float) and a else None
            rel_s = f"{rel:>+8.1f}%" if isinstance(rel, float) else f"{'-':>9}"
            print(f"  {name:<13}{ds:<14}{'+'.join(fam['train']):<32}"
                  f"{fmt(a, 10)}{fmt(b, 11)}{fmt(delta, 9)}{rel_s}"
                  f"  {tier}{'' if tier == 1 else '  <- underpowered'}")
            rows.append({
                "family": name, "domain": fam.get("family", ""),
                "held_out": ds, "siblings_in_pool": "+".join(fam["train"]), "tier": tier,
                f"zeroshot_{args.metric}": a, f"family_ft_{args.metric}": b,
                "delta": delta,
                "relative_pct": round(rel, 2) if isinstance(rel, float) else None,
            })

    def macro(subset, label):
        ok = [r for r in subset if isinstance(r["delta"], float)]
        if not ok:
            return
        ma = sum(r[f"zeroshot_{args.metric}"] for r in ok) / len(ok)
        mb = sum(r[f"family_ft_{args.metric}"] for r in ok) / len(ok)
        rel = 100.0 * (mb - ma) / ma if ma else float("nan")
        print(f"  {label:<59}{fmt(ma, 10)}{fmt(mb, 11)}{fmt(mb - ma, 9)}{rel:>+8.1f}%")

    print("  " + "-" * (len(hdr) - 2))
    macro([r for r in rows if r["tier"] == 1], f"MACRO -- tier 1 only ({sum(1 for r in rows if r['tier'] == 1)} families)")
    macro(rows, f"MACRO -- all families ({len(rows)})")

    print()
    print("  Tier 2 (industrial/Genesis ~28 anomalous timesteps, env_iot/CalIt2 ~138) is")
    print("  NOISE-DOMINATED. A bad number there does not mean transfer failed -- it means")
    print("  the measurement cannot resolve the question. Report it separately.")

    os.makedirs(os.path.dirname(os.path.abspath(args.out_csv)), exist_ok=True)
    with open(args.out_csv, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"\n  -> {args.out_csv}")


if __name__ == "__main__":
    main()
