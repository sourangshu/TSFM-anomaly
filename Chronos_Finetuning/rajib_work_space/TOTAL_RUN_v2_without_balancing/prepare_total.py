"""
Whole-mTSBench sliding-window data preparation for Chronos-2 anomaly fine-tuning.

This is the SMD_run/prepare_smd_split.py pipeline scaled from one dataset to ALL
mTSBench datasets, keeping the per-window layout BYTE-FOR-BYTE identical:

    [ normal_signal (256) | context (512) | future (64) ]   + future_labels (64,)

with the PER-SERIES normal prefix (each *test.csv file carves its own normal zone
via extract_normal_signal — this is the per-series approach that gave the good SMD
results, NOT the global/unified signal). The carving / loading / pairing primitives
are imported directly from prepare_smd_split.py so behaviour matches SMD exactly.

WHAT'S DIFFERENT FROM ../TOTAL_RUN/prepare_total.py
---------------------------------------------------
That version class-balanced the combined train set (kept all rare anomalous windows,
reservoir-subsampled normals to normal_ratio*anom, and capped each dataset's
anomalous contribution). Those results were not convincing. This v2 removes ALL of
that: NO class balancing, NO reservoir, NO per-dataset window cap. We keep EVERY
sliding window from the train half of EVERY dataset — exactly like SMD did — pool
them, shuffle, and dump. This is the "train on the entirety of the *test.csv train
half" behaviour requested.

Two products
------------
1. ONE combined train_model_inputs.pkl — every window (anomalous + normal, no
   subsampling) pooled from the TRAIN half of every multi-file dataset, shuffled.
   Feed straight into run_finetune_total.sh (PREPARED_DIR -> this folder).

2. Per-dataset test sets — each dataset's TEST half written as its own
   per_dataset/<DATASET>/{test_model_inputs.pkl, test_series_meta.pkl}, so the
   existing forward.py / run_forward_total.sh evaluates one dataset at a time
   unchanged (point TEST_PKL / META_PKL at a per-dataset folder). Test windows tile
   each series contiguously (stride == prediction_length) so forward.py can
   reassemble per-timestamp scores for VUS-PR.

Per-dataset split
-----------------
  * >=2 *test.csv files -> FILE-BASED 50/50 (seeded): half the files -> train pool,
    half -> that dataset's test set. No window from a CSV appears in two splits.
  * exactly 1 *test.csv file -> TEST-ONLY: the single file goes entirely to that
    dataset's test set and contributes NOTHING to the combined train pkl (a
    file-based split is impossible with one file).

Usage
-----
    python prepare_total.py                                   # all datasets, 50/50
    python prepare_total.py --datasets SMD MSL SMAP          # subset
    python prepare_total.py --test_fraction 0.5 --stride 64
"""

import argparse
import glob
import json
import logging
import os
import pickle
import sys

import numpy as np

# ── Reuse the SMD per-series primitives verbatim ─────────────────────────────
_HERE = os.path.dirname(os.path.abspath(__file__))
_SMD_RUN = os.path.join(os.path.dirname(_HERE), "SMD_run")
sys.path.insert(0, _SMD_RUN)
from prepare_smd_split import (                      # noqa: E402
    NORMAL_SIGNAL_LENGTH,
    build_pairs_for_files,
    pairs_to_model_inputs,
)

logger = logging.getLogger("prepare_total")


# ─────────────────────────────────────────────────────────────────────────────
#  Per-dataset file discovery + 50/50 file-based split
# ─────────────────────────────────────────────────────────────────────────────

def list_datasets(data_root: str, only):
    """Dataset = any sub-directory of data_root containing >=1 *test.csv file."""
    names = []
    for entry in sorted(os.listdir(data_root)):
        d = os.path.join(data_root, entry)
        if not os.path.isdir(d):
            continue
        if only and entry not in only:
            continue
        if glob.glob(os.path.join(d, "**", "*test.csv"), recursive=True):
            names.append(entry)
    return names


def split_files(test_csvs, test_fraction: float, seed: int):
    """File-based split. Returns (train_files, test_files, mode).

    >=2 files -> seeded file split. Exactly 1 file -> TEST-ONLY (train empty)."""
    if len(test_csvs) == 1:
        return [], list(test_csvs), "test_only"
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(test_csvs))
    n_test = max(1, min(len(test_csvs) - 1, int(round(len(test_csvs) * test_fraction))))
    test_idx = set(perm[:n_test].tolist())
    train_files = [test_csvs[i] for i in range(len(test_csvs)) if i not in test_idx]
    test_files = [test_csvs[i] for i in sorted(test_idx)]
    return train_files, test_files, "file_split"


# ─────────────────────────────────────────────────────────────────────────────
#  Pipeline
# ─────────────────────────────────────────────────────────────────────────────

def prepare(args) -> None:
    os.makedirs(args.output_dir, exist_ok=True)
    per_ds_root = os.path.join(args.output_dir, "per_dataset")
    os.makedirs(per_ds_root, exist_ok=True)

    datasets = list_datasets(args.data_root, args.datasets)
    if not datasets:
        raise SystemExit(f"No datasets with *test.csv found under {args.data_root}")
    logger.info(f"Datasets ({len(datasets)}): {datasets}")

    min_req = max(args.min_length, args.context_length + args.prediction_length)

    # Pool EVERY train-half window from EVERY dataset here (no balancing / no cap).
    all_train_inputs: list = []
    manifest: dict = {"datasets": {}, "config": vars(args).copy()}
    grand_train_anom = grand_train_norm = 0

    for ds in datasets:
        ddir = os.path.join(args.data_root, ds)
        test_csvs = sorted(glob.glob(os.path.join(ddir, "**", "*test.csv"), recursive=True))
        train_files, test_files, mode = split_files(test_csvs, args.test_fraction, args.seed)

        logger.info("=" * 78)
        logger.info(f"{ds}: {len(test_csvs)} *test.csv  [{mode}]  "
                    f"train_files={len(train_files)} test_files={len(test_files)}")

        # ── TEST half -> per-dataset pkls (ordered + metadata, stride=test_stride) ──
        ds_out = os.path.join(per_ds_root, ds)
        os.makedirs(ds_out, exist_ok=True)
        test_pairs, _, _, test_meta = build_pairs_for_files(
            test_files, args.context_length, args.prediction_length,
            args.test_stride, min_req, f"{ds}/test")
        test_inputs = pairs_to_model_inputs(test_pairs, include_meta=True)
        with open(os.path.join(ds_out, "test_model_inputs.pkl"), "wb") as f:
            pickle.dump(test_inputs, f)
        with open(os.path.join(ds_out, "test_series_meta.pkl"), "wb") as f:
            pickle.dump(test_meta, f)
        t_anom = sum(int(d["future_labels"].sum() >= args.anomaly_threshold) for d in test_inputs)
        logger.info(f"  TEST  -> {len(test_inputs)} windows ({len(test_meta)} series), "
                    f"{t_anom} anomalous -> {ds_out}")

        # ── TRAIN half -> KEEP EVERY WINDOW (no balancing, no subsampling, no cap) ──
        train_anom = train_norm = 0
        if train_files:
            train_pairs, _, _, _ = build_pairs_for_files(
                train_files, args.context_length, args.prediction_length,
                args.stride, min_req, f"{ds}/train")
            train_inputs = pairs_to_model_inputs(train_pairs, include_meta=False)
            for d in train_inputs:
                if int(d["future_labels"].sum()) >= args.anomaly_threshold:
                    train_anom += 1
                else:
                    train_norm += 1
            all_train_inputs.extend(train_inputs)

        grand_train_anom += train_anom
        grand_train_norm += train_norm
        logger.info(f"  TRAIN -> {train_anom + train_norm} windows kept "
                    f"(anomalous={train_anom}, normal={train_norm})  [ALL kept, no balancing]")
        manifest["datasets"][ds] = {
            "mode": mode,
            "n_test_csv": len(test_csvs),
            "train_files": [os.path.basename(f) for f in train_files],
            "test_files": [os.path.basename(f) for f in test_files],
            "test_windows": len(test_inputs),
            "test_series": len(test_meta),
            "train_windows": train_anom + train_norm,
            "train_anomalous": train_anom,
            "train_normal": train_norm,
        }

    # ── Shuffle + write the combined train pkl ──────────────────────────────────
    logger.info("=" * 78)
    shuf = np.random.default_rng(args.seed)
    shuf.shuffle(all_train_inputs)
    train_path = os.path.join(args.output_dir, "train_model_inputs.pkl")
    with open(train_path, "wb") as f:
        pickle.dump(all_train_inputs, f)

    tot = grand_train_anom + grand_train_norm
    logger.info("COMBINED TRAIN (NO BALANCING — every train-half window kept)")
    logger.info(f"  anomalous windows : {grand_train_anom}")
    logger.info(f"  normal windows    : {grand_train_norm}")
    logger.info(f"  total train windows: {tot}  "
                f"({100 * grand_train_anom / max(1, tot):.1f}% anomalous / "
                f"{100 * grand_train_norm / max(1, tot):.1f}% normal) -> {train_path}")
    if all_train_inputs:
        logger.info(f"  per-window target shape: {all_train_inputs[0]['target'].shape}  "
                    f"(= [{NORMAL_SIGNAL_LENGTH} normal | {args.context_length} context "
                    f"| {args.prediction_length} future]; F varies per dataset)")

    manifest["train_totals"] = {
        "anomalous": grand_train_anom, "normal": grand_train_norm, "total": tot,
    }
    with open(os.path.join(args.output_dir, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2, default=str)
    logger.info(f"Manifest -> {os.path.join(args.output_dir, 'manifest.json')}")


def main():
    p = argparse.ArgumentParser(
        description="Whole-mTSBench data prep for Chronos-2 anomaly fine-tuning "
                    "(per-series normal signal, NO class balancing).")
    p.add_argument("--data_root", default="/home/rajib/mTSBench/Datasets/mTSBench",
                   help="Root containing one sub-folder per dataset")
    p.add_argument("--output_dir", default=os.path.join(_HERE, "prepared_total"))
    p.add_argument("--datasets", nargs="*", default=None,
                   help="Optional subset of dataset names (default: all discovered)")

    # Window geometry — must match run_finetune / run_forward
    p.add_argument("--context_length", type=int, default=512)
    p.add_argument("--prediction_length", type=int, default=64)
    p.add_argument("--stride", type=int, default=64, help="Train sliding-window stride")
    p.add_argument("--test_stride", type=int, default=64,
                   help="Test sliding-window stride (MUST equal prediction_length so "
                        "test windows tile contiguously for forward.py)")
    p.add_argument("--min_length", type=int, default=50)

    # Split
    p.add_argument("--test_fraction", type=float, default=0.5,
                   help="Fraction of each dataset's *test.csv files held out for testing")
    p.add_argument("--seed", type=int, default=42)

    # Window classification (for logging only — NO balancing is applied)
    p.add_argument("--anomaly_threshold", type=int, default=10,
                   help="Window counted as anomalous iff >= this many anomalous steps "
                        "(matches the trainer's derive_future_type; used ONLY for stats).")
    args = p.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    log_path = os.path.join(args.output_dir, "prepare_total.log")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.StreamHandler(), logging.FileHandler(log_path)],
    )
    logger.info(f"Config: {vars(args)}")
    prepare(args)
    logger.info("Done.")


if __name__ == "__main__":
    main()
