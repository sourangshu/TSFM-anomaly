#!/usr/bin/env python
"""
Build a 100%-OF-DATA eval set: every real *test.csv of each dataset, IGNORING the
50/50 file split that prepare_total.py carved.

WHY THIS IS A SEPARATE SCRIPT
    prepare_total.py owns the train/test/val split that every arm's numbers rest on.
    This script never calls into that ownership: it imports prepare_total's helpers
    read-only and writes to NEW filenames. No train, test or val pkl is opened for
    writing, so this is safe to run while a fine-tune is reading prepared_total.

WHAT IT WRITES
    per_dataset/<DATASET>/full_model_inputs.pkl     ← same layout/dtypes as the test half
    per_dataset/<DATASET>/full_series_meta.pkl
    manifest_full_eval.json

    The windowing is IDENTICAL to the test half's (test_stride, include_meta=True), so
    forward.py reads these with no code changes. The only difference is which files go
    in: split_files() is never called, so train_files AND test_files are both windowed.
    Synthetic files are excluded — they are train-only fabrications with no place in an
    eval set.

⚠ THE RESULT IS NOT A HELD-OUT SCORE.
    The checkpoint was fine-tuned on windows drawn from train_files, and train_files are
    in here. Per-dataset contamination — the share of eval windows coming from files the
    model trained on — is logged and stored in manifest_full_eval.json as
    `contaminated_window_frac`. Report these as TRAIN+TEST numbers, never as test
    numbers, and keep the results_FT/ held-out numbers alongside them.

USAGE
    python prepare_full_eval.py                          # all datasets
    python prepare_full_eval.py --datasets MSL SMD       # subset
    EVAL_SPLIT=full ./run_forward_total.sh               # then evaluate

    Geometry flags default to prepare_total.py's values and MUST match them; they are
    exposed only so an unusual prep can be mirrored.
"""

import argparse
import json
import logging
import os
import pickle
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

# Read-only reuse of the prep's own primitives, so file discovery, the split maths and
# the window construction are BY CONSTRUCTION the same code the test half went through.
from prepare_total import (                          # noqa: E402
    anomaly_step_counts,
    build_pairs_for_files,
    list_datasets,
    pairs_to_model_inputs,
    real_test_csvs,
    split_files,
)

logger = logging.getLogger("prepare_full_eval")


def prepare_full_eval(args) -> None:
    per_ds_root = os.path.join(args.output_dir, "per_dataset")
    if not os.path.isdir(per_ds_root):
        raise SystemExit(
            f"{per_ds_root} not found. This script adds a full-data eval set to an "
            f"EXISTING prepared dir — run ./run_prepare_total.sh first."
        )

    datasets = list_datasets(args.data_root, args.datasets)
    if not datasets:
        raise SystemExit(f"No datasets with *test.csv found under {args.data_root}")

    logger.info(f"FULL-EVAL: building 100%-of-data eval sets for {len(datasets)} datasets.")
    logger.info("The 50/50 file split is IGNORED (train_files + test_files both windowed).")
    logger.info("Existing train/test/val pkls are NOT touched.")
    logger.info("These scores are NOT held out — see contaminated_window_frac per dataset.")

    min_req = max(args.min_length, args.context_length + args.prediction_length)
    manifest: dict = {"datasets": {}, "config": vars(args).copy()}
    tot_win = tot_contam = 0

    for ds in datasets:
        ddir = os.path.join(args.data_root, ds)
        all_files = real_test_csvs(ddir)          # ALL of them — no split_files() gate

        # Recover the split ONLY to report contamination. Same (test_fraction, seed) as
        # the real prep, so train_files here is exactly the set the checkpoint saw.
        train_files, test_files, mode = split_files(all_files, args.test_fraction, args.seed)
        train_bases = {os.path.basename(f) for f in train_files}

        logger.info("=" * 78)
        logger.info(f"{ds}: {len(all_files)} real *test.csv  [{mode}]  "
                    f"(train_files={len(train_files)} + test_files={len(test_files)})")

        ds_out = os.path.join(per_ds_root, ds)
        os.makedirs(ds_out, exist_ok=True)

        pairs, _, _, series_meta = build_pairs_for_files(
            all_files, args.context_length, args.prediction_length,
            args.test_stride, min_req, f"{ds}/full")
        if not pairs:
            logger.info(f"  no windows survived min_length={min_req} — skipping {ds}")
            continue

        # series_id is the file BASENAME (set by build_pairs_for_files), so this is
        # exactly the share of eval windows the checkpoint was fine-tuned on.
        n_contam = sum(1 for p in pairs if p["series_id"] in train_bases)
        contam_frac = n_contam / len(pairs)

        inputs = pairs_to_model_inputs(pairs, include_meta=True)
        del pairs

        with open(os.path.join(ds_out, "full_model_inputs.pkl"), "wb") as f:
            pickle.dump(inputs, f)
        with open(os.path.join(ds_out, "full_series_meta.pkl"), "wb") as f:
            pickle.dump(series_meta, f)

        n_anom = int((anomaly_step_counts(inputs) >= 1).sum())
        tot_win += len(inputs)
        tot_contam += n_contam
        logger.info(f"  FULL  -> {len(inputs)} windows ({len(series_meta)} series), "
                    f"{n_anom} anomalous")
        logger.info(f"  CONTAMINATION: {n_contam}/{len(inputs)} windows ({contam_frac:.1%}) "
                    f"come from files the checkpoint trained on")

        manifest["datasets"][ds] = {
            "mode": mode,
            "n_test_csv": len(all_files),
            "train_files": [os.path.basename(f) for f in train_files],
            "test_files": [os.path.basename(f) for f in test_files],
            "full_windows": len(inputs),
            "full_series": len(series_meta),
            "full_anomalous": n_anom,
            "contaminated_windows": n_contam,
            "contaminated_window_frac": round(contam_frac, 4),
        }

    manifest["totals"] = {
        "full_windows": tot_win,
        "contaminated_windows": tot_contam,
        "contaminated_window_frac": round(tot_contam / tot_win, 4) if tot_win else None,
    }

    mpath = os.path.join(args.output_dir, "manifest_full_eval.json")
    with open(mpath, "w") as f:
        json.dump(manifest, f, indent=2, default=str)

    logger.info("=" * 78)
    if tot_win:
        logger.info(f"TOTAL: {tot_win} eval windows, {tot_contam} ({tot_contam / tot_win:.1%}) "
                    f"from files the checkpoint trained on — NOT a held-out score.")
    logger.info(f"Full-eval manifest -> {mpath}")
    logger.info("Evaluate with:  EVAL_SPLIT=full ./run_forward_total.sh")


def main():
    p = argparse.ArgumentParser(
        description="Build a 100%-of-data (train+test files) eval set alongside an "
                    "existing prepared_total. Does not modify the existing split.")
    p.add_argument("--data_root", default="/home/rajib/mTSBench/Datasets/mTSBench",
                   help="Root containing one sub-folder per dataset")
    p.add_argument("--output_dir", default=os.path.join(_HERE, "prepared_total"),
                   help="Existing prepared dir to add full_*.pkl into")
    p.add_argument("--datasets", nargs="*", default=None,
                   help="Optional subset of dataset names (default: all discovered)")

    # Window geometry — MUST match prepare_total.py / run_forward_total.sh
    p.add_argument("--context_length", type=int, default=512)
    p.add_argument("--prediction_length", type=int, default=64)
    p.add_argument("--test_stride", type=int, default=64,
                   help="Sliding-window stride (MUST equal prediction_length so windows "
                        "tile contiguously for forward.py)")
    p.add_argument("--min_length", type=int, default=50)

    # Split parameters — used ONLY to report contamination, never to filter files.
    # Must match the prep that produced the checkpoint, or the contamination numbers lie.
    p.add_argument("--test_fraction", type=float, default=0.5)
    p.add_argument("--seed", type=int, default=42)

    args = p.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    log_path = os.path.join(args.output_dir, "prepare_full_eval.log")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.StreamHandler(), logging.FileHandler(log_path)],
    )
    logger.info(f"Config: {vars(args)}")
    prepare_full_eval(args)
    logger.info("Done.")


if __name__ == "__main__":
    main()
