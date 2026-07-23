"""
Freeze Chronos's evaluated index range for every held-out mTSBench series.

The Chronos arm (TOTAL_RUN_maskloss_v2_HS) does NOT score a whole test.csv. Per
series it tiles windows of `prediction_length` starting after `context_length`, so
the scored region is

    [ context_length , context_length + n_windows * prediction_length )

i.e. the leading 512 context steps are never forecast and the trailing
`len % 64` steps fall outside the last window. `forward.py` computes its metrics on
exactly that slice (`covered[sid] = [min future_start, max future_end]`).

For the mTSBench baseline detectors to be comparable series-for-series, they must be
scored on the SAME slice. This script reads the Chronos test pkls once and writes

    covered_regions.json
    {
      "config":  {...provenance...},
      "datasets": {
         "SMD": {
            "SMD_machine-1-2_test.csv": {"lo": 512, "hi": 4928, "length": 4930,
                                         "n_features": 38, "n_pos_covered": 123},
            ...
         }, ...
      }
    }

Only series that actually appear in a test pkl are listed — a held-out file that
produced no windows (too short) is absent here exactly as it is absent from the
Chronos results, so the two runs agree on the denominator.

Usage:
    python dump_covered_regions.py
    python dump_covered_regions.py --prepared_dir <...>/prepared_total --out covered_regions.json
"""

import argparse
import json
import os
import pickle
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))


def find_prepared_dir():
    """Locate the Chronos arm's prepared_total, or None.

    This is only needed to REGENERATE covered_regions.json. That file is committed,
    so a fresh clone never needs the Chronos workspace at all — which is the whole
    point of vendoring. Resolution order: $CHRONOS_PREPARED, then the usual sibling
    layout relative to this file.
    """
    env = os.environ.get("CHRONOS_PREPARED")
    if env:
        return env
    for cand in (
        os.path.join(_HERE, "..", "Chronos_Finetuning", "rajib_work_space",
                     "TOTAL_RUN_maskloss_v2_HS", "prepared_total"),
        os.path.join(_HERE, "..", "..", "Chronos_Finetuning", "rajib_work_space",
                     "TOTAL_RUN_maskloss_v2_HS", "prepared_total"),
    ):
        if os.path.isdir(cand):
            return os.path.normpath(cand)
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prepared_dir", default=None,
                    help="Chronos prepared_total dir (holds manifest.json + per_dataset/). "
                         "Default: $CHRONOS_PREPARED, else auto-detected. Only needed to "
                         "REGENERATE covered_regions.json, which is committed.")
    ap.add_argument("--split", default="test", choices=["test", "full"],
                    help="Which pkl prefix to read; 'test' = the held-out 50%% half")
    ap.add_argument("--out", default=os.path.join(_HERE, "covered_regions.json"))
    args = ap.parse_args()

    args.prepared_dir = args.prepared_dir or find_prepared_dir()
    if not args.prepared_dir or not os.path.isdir(args.prepared_dir):
        sys.exit("ERROR: could not locate the Chronos prepared_total dir.\n"
                 "       You only need it to REGENERATE covered_regions.json — the\n"
                 "       committed copy is what the pipeline actually reads.\n"
                 "       To regenerate anyway:\n"
                 "         export CHRONOS_PREPARED=/path/to/TOTAL_RUN_maskloss_v2_HS/prepared_total")

    per_ds_dir = os.path.join(args.prepared_dir, "per_dataset")
    with open(os.path.join(args.prepared_dir, "manifest.json")) as f:
        manifest = json.load(f)

    out = {"config": {"prepared_dir": args.prepared_dir,
                      "split": args.split,
                      "context_length": manifest["config"]["context_length"],
                      "prediction_length": manifest["config"]["prediction_length"],
                      "test_fraction": manifest["config"]["test_fraction"],
                      "seed": manifest["config"]["seed"],
                      "data_root": manifest["config"]["data_root"]},
           "datasets": {}}

    n_series_total = 0
    for ds in sorted(manifest["datasets"]):
        win_pkl = os.path.join(per_ds_dir, ds, f"{args.split}_model_inputs.pkl")
        meta_pkl = os.path.join(per_ds_dir, ds, f"{args.split}_series_meta.pkl")
        if not (os.path.exists(win_pkl) and os.path.exists(meta_pkl)):
            print(f"  skip {ds}: no {args.split} pkl")
            continue

        with open(win_pkl, "rb") as f:
            windows = pickle.load(f)
        with open(meta_pkl, "rb") as f:
            meta = pickle.load(f)

        cov = {}
        for w in windows:
            sid, fs, fe = w["series_id"], int(w["future_start"]), int(w["future_end"])
            if sid in cov:
                cov[sid][0] = min(cov[sid][0], fs)
                cov[sid][1] = max(cov[sid][1], fe)
            else:
                cov[sid] = [fs, fe]

        entry = {}
        for sid in sorted(cov):
            lo, hi = cov[sid]
            m = meta[sid]
            labels = np.asarray(m["labels"]).astype(int)
            entry[sid] = {"lo": lo, "hi": hi,
                          "length": int(m["length"]),
                          "n_features": int(m["n_features"]),
                          "n_pos_covered": int(labels[lo:hi].sum())}
        out["datasets"][ds] = entry
        n_series_total += len(entry)
        print(f"  {ds:<16} {len(entry):>3} series   "
              f"(from {len(manifest['datasets'][ds]['test_files'])} held-out files)")

    with open(args.out, "w") as f:
        json.dump(out, f, indent=1)
    print(f"\nWrote {args.out}: {len(out['datasets'])} datasets, {n_series_total} series.")


if __name__ == "__main__":
    main()
