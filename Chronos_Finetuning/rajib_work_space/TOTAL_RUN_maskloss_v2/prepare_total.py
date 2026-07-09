"""
Whole-mTSBench sliding-window data prep for Chronos-2 anomaly fine-tuning — the
DATASET-BALANCED, THRESHOLDLESS variant used by the count-weighted sampler.

Per-window layout is BYTE-FOR-BYTE identical to SMD_run / TOTAL_RUN_v2:

    [ normal_signal (256) | context (512) | future (64) ]   + future_labels (64,)

The carving / loading / pairing primitives are imported verbatim from
SMD_run/prepare_smd_split.py so behaviour matches SMD and the test sets exactly.

WHY THIS VERSION EXISTS
-----------------------
There are TWO independent imbalances in combined mTSBench, and they need two
different tools:

  * CLASS imbalance (normal vs anomaly STEPS) — handled at RUNTIME by the
    count-weighted sampler in finetune_anomaly_simple.py (thresholdless, per-step).
    So we do NOT class-balance here.

  * DATASET imbalance (MITDB+SVDB are ~88% of all windows) — the sampler does
    NOT touch this, so it MUST be handled here, at prep. Without it the combined
    model just becomes an ECG model.

This file therefore does exactly ONE thing the two neighbours don't: a per-dataset
CAP that equalises how much of each dataset enters the combined train pool, while
GUARANTEEING that both classes survive for every dataset.

  ../TOTAL_RUN/prepare_total.py             : class-balanced (normal_ratio+threshold) — obsolete now
  ../TOTAL_RUN_v2_without_balancing/...     : keeps EVERY window (88% ECG) — dataset-dominated
  THIS file                                 : per-dataset cap only; sampler does class balance

THE CAP (per dataset, keep <= PER_DATASET_CAP windows)
------------------------------------------------------
  * "anomaly window" = >=1 anomaly step  (thresholdless — matches the per-step view).
  * anomalies take at most MAX_ANOM_FRAC of the budget:  n_anom = min(#anom, floor(f*cap))
  * the remainder is filled with normal windows:         n_norm = min(#normal, cap - n_anom)
  * if a dataset is normal-poor, anomalies top up the slack so the cap is still used.
  * datasets that already fit under the cap are kept whole.
  * natural class ratio is otherwise preserved — we do NOT class-balance here.

Products (identical structure to TOTAL_RUN_v2):
  1. ONE combined train_model_inputs.pkl (dataset-capped, shuffled) -> run_finetune_total.sh
  2. per_dataset/<DATASET>/{test_model_inputs.pkl, test_series_meta.pkl}  (TEST half,
     UNCAPPED — evaluation must see every test window) -> run_forward_total.sh

Usage
-----
    python prepare_total.py                                   # all datasets, cap=5000
    python prepare_total.py --per_dataset_cap 3000            # smaller/faster pool
    python prepare_total.py --datasets SMD MSL SMAP           # subset
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
#  Per-dataset file discovery + 50/50 file-based split  (verbatim from v2)
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
#  The one new operation: per-dataset cap (dataset de-domination)
# ─────────────────────────────────────────────────────────────────────────────

def _is_anom(d) -> bool:
    """Thresholdless: a window is anomalous iff it has >= 1 anomaly step."""
    return int(np.asarray(d["future_labels"]).sum()) >= 1


def cap_dataset(train_inputs, cap: int, max_anom_frac: float, rng):
    """Subsample ONE dataset's train windows to <= cap, keeping BOTH classes.

    Anomaly windows take at most max_anom_frac*cap slots; normals fill the rest.
    Returns (kept_list, stats_dict). If everything fits under cap, keep all."""
    n = len(train_inputs)
    anom_idx = np.array([i for i, d in enumerate(train_inputs) if _is_anom(d)], dtype=int)
    norm_idx = np.array([i for i, d in enumerate(train_inputs) if not _is_anom(d)], dtype=int)

    if cap <= 0 or n <= cap:
        return train_inputs, {
            "available": n, "avail_anom": int(len(anom_idx)), "avail_norm": int(len(norm_idx)),
            "kept": n, "kept_anom": int(len(anom_idx)), "kept_norm": int(len(norm_idx)),
            "capped": False,
        }

    n_anom = int(min(len(anom_idx), np.floor(max_anom_frac * cap)))
    n_norm = int(min(len(norm_idx), cap - n_anom))
    # Normal-poor dataset: let anomalies use the leftover budget so cap is filled.
    if n_norm < (cap - n_anom) and len(anom_idx) > n_anom:
        n_anom = int(min(len(anom_idx), cap - n_norm))

    sel_anom = rng.choice(anom_idx, size=n_anom, replace=False) if n_anom > 0 else np.array([], dtype=int)
    sel_norm = rng.choice(norm_idx, size=n_norm, replace=False) if n_norm > 0 else np.array([], dtype=int)
    keep = sorted(sel_anom.tolist() + sel_norm.tolist())
    kept = [train_inputs[i] for i in keep]
    return kept, {
        "available": n, "avail_anom": int(len(anom_idx)), "avail_norm": int(len(norm_idx)),
        "kept": len(kept), "kept_anom": n_anom, "kept_norm": n_norm,
        "capped": True,
    }


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
    logger.info(f"Per-dataset cap = {args.per_dataset_cap}  "
                f"(anomalies <= {args.max_anom_frac:.2f} of cap; both classes guaranteed)")

    min_req = max(args.min_length, args.context_length + args.prediction_length)
    cap_rng = np.random.default_rng(args.seed + 1234)   # dedicated stream for the cap

    all_train_inputs: list = []
    manifest: dict = {"datasets": {}, "config": vars(args).copy()}
    grand_anom = grand_norm = 0

    for ds in datasets:
        ddir = os.path.join(args.data_root, ds)
        test_csvs = sorted(glob.glob(os.path.join(ddir, "**", "*test.csv"), recursive=True))
        train_files, test_files, mode = split_files(test_csvs, args.test_fraction, args.seed)

        logger.info("=" * 78)
        logger.info(f"{ds}: {len(test_csvs)} *test.csv  [{mode}]  "
                    f"train_files={len(train_files)} test_files={len(test_files)}")

        # ── TEST half -> per-dataset pkls (UNCAPPED, ordered + metadata) ──
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
        t_anom = sum(1 for d in test_inputs if _is_anom(d))
        logger.info(f"  TEST  -> {len(test_inputs)} windows ({len(test_meta)} series), "
                    f"{t_anom} anomalous -> {ds_out}  [UNCAPPED]")

        # ── TRAIN half -> CAP to equalise dataset footprint (both classes kept) ──
        cap_stats = {"available": 0, "kept": 0, "kept_anom": 0, "kept_norm": 0, "capped": False}
        if train_files:
            train_pairs, _, _, _ = build_pairs_for_files(
                train_files, args.context_length, args.prediction_length,
                args.stride, min_req, f"{ds}/train")
            train_inputs = pairs_to_model_inputs(train_pairs, include_meta=False)
            train_inputs, cap_stats = cap_dataset(
                train_inputs, args.per_dataset_cap, args.max_anom_frac, cap_rng)
            all_train_inputs.extend(train_inputs)

        grand_anom += cap_stats["kept_anom"]
        grand_norm += cap_stats["kept_norm"]
        logger.info(
            f"  TRAIN -> kept {cap_stats['kept']}/{cap_stats['available']} windows "
            f"(anom={cap_stats['kept_anom']}, normal={cap_stats['kept_norm']})"
            f"{'  [CAPPED]' if cap_stats['capped'] else '  [all kept, under cap]'}")
        manifest["datasets"][ds] = {
            "mode": mode,
            "n_test_csv": len(test_csvs),
            "train_files": [os.path.basename(f) for f in train_files],
            "test_files": [os.path.basename(f) for f in test_files],
            "test_windows": len(test_inputs),
            "test_series": len(test_meta),
            "test_anomalous": t_anom,
            **{f"train_{k}": v for k, v in cap_stats.items()},
        }

    # ── Shuffle + write the combined train pkl ──────────────────────────────────
    logger.info("=" * 78)
    shuf = np.random.default_rng(args.seed)
    shuf.shuffle(all_train_inputs)
    train_path = os.path.join(args.output_dir, "train_model_inputs.pkl")
    with open(train_path, "wb") as f:
        pickle.dump(all_train_inputs, f)

    tot = grand_anom + grand_norm
    logger.info("COMBINED TRAIN (dataset-capped; class balance is the sampler's job)")
    logger.info(f"  anomaly windows : {grand_anom}")
    logger.info(f"  normal windows  : {grand_norm}")
    logger.info(f"  total windows   : {tot}  "
                f"({100 * grand_anom / max(1, tot):.1f}% anom / "
                f"{100 * grand_norm / max(1, tot):.1f}% normal) -> {train_path}")
    if all_train_inputs:
        logger.info(f"  per-window target shape: {all_train_inputs[0]['target'].shape}  "
                    f"(= [{NORMAL_SIGNAL_LENGTH} normal | {args.context_length} context "
                    f"| {args.prediction_length} future]; F varies per dataset)")

    # ── Optional EVAL_VAL probe (off by default) ────────────────────────────────
    # A COPY of a random subset of the combined train pool — a stable monitoring set
    # for the training-loss curves (every train window is STILL trained on; nothing
    # is removed). This is NOT a held-out generalization estimate (the per-dataset
    # test sets are). To use it, set VAL_FRACTION>0 here AND NO_VALIDATION=0 in
    # run_finetune_total.sh. EVAL_TEST is separate and stays MANUAL (pass TEST_DATA).
    if args.val_fraction > 0 and all_train_inputs:
        vrng = np.random.default_rng(args.seed + 9)
        n_val = int(round(args.val_fraction * len(all_train_inputs)))
        n_val = max(1, min(n_val, args.val_max, len(all_train_inputs)))
        vidx = vrng.choice(len(all_train_inputs), size=n_val, replace=False)
        val_inputs = [all_train_inputs[i] for i in vidx]      # references, train kept intact
        val_path = os.path.join(args.output_dir, "val_model_inputs.pkl")
        with open(val_path, "wb") as f:
            pickle.dump(val_inputs, f)
        v_anom = sum(1 for d in val_inputs if _is_anom(d))
        logger.info(f"VAL PROBE -> {len(val_inputs)} windows (subset of train, train kept intact); "
                    f"{v_anom} anom / {len(val_inputs) - v_anom} normal -> {val_path}")
        manifest["val_probe"] = {"windows": len(val_inputs), "anomaly": v_anom,
                                 "subset_of_train": True, "val_fraction": args.val_fraction}
    else:
        logger.info("val_fraction=0 -> no val_model_inputs.pkl "
                    "(run finetune with NO_VALIDATION=1; EVAL_TEST stays manual via TEST_DATA)")

    manifest["train_totals"] = {"anomaly": grand_anom, "normal": grand_norm, "total": tot}
    with open(os.path.join(args.output_dir, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2, default=str)
    logger.info(f"Manifest -> {os.path.join(args.output_dir, 'manifest.json')}")


def main():
    p = argparse.ArgumentParser(
        description="Dataset-balanced (per-dataset cap), thresholdless mTSBench data "
                    "prep for the count-weighted sampler.")
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

    # The per-dataset cap (dataset de-domination). Class balance is NOT done here.
    p.add_argument("--per_dataset_cap", type=int, default=5000,
                   help="Max train windows kept per dataset (<=0 disables capping). "
                        "Equalises dataset footprint in the combined pool.")
    p.add_argument("--max_anom_frac", type=float, default=0.5,
                   help="Within a dataset's cap, anomalies take at most this fraction "
                        "of the budget; the rest are normal windows (guarantees both "
                        "classes survive). Only binds for anomaly-heavy datasets (ECG).")

    # Optional EVAL_VAL monitoring probe (off by default). EVAL_TEST stays manual.
    p.add_argument("--val_fraction", type=float, default=0.0,
                   help="If >0, write val_model_inputs.pkl = a COPY of this fraction of "
                        "the combined train pool (a monitoring probe; no data removed "
                        "from train). 0 = no val pkl. Pair with NO_VALIDATION=0.")
    p.add_argument("--val_max", type=int, default=5000,
                   help="Hard cap on the validation-probe size (keeps eval cheap).")
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
