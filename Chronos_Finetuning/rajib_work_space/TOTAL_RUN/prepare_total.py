"""
Whole-mTSBench sliding-window data preparation for Chronos-2 anomaly fine-tuning.

This scales SMD_run/prepare_smd_split.py from a single dataset to ALL mTSBench
datasets, keeping the per-window layout BYTE-FOR-BYTE identical:

    [ normal_signal (256) | context (512) | future (64) ]   + future_labels (64,)

with the PER-SERIES normal prefix (each series carves its own normal zone — this is
the per-series approach, NOT the global/unified signal). The carving / loading /
pairing primitives are imported directly from prepare_smd_split.py so behaviour
matches the SMD pipeline exactly.

Two products
------------
1. ONE combined train_model_inputs.pkl  — windows pooled from the TRAIN half of
   every multi-file dataset, class-balanced and shuffled. Feed straight into
   run_finetune (PREPARED_DIR -> this folder).

2. Per-dataset test sets — each dataset's TEST half written as its own
   per_dataset/<DATASET>/{test_model_inputs.pkl, test_series_meta.pkl}, so the
   existing forward.py / run_forward_smd.sh evaluates one dataset at a time
   unchanged (point TEST_PKL / META_PKL at a per-dataset folder).

Per-dataset split
-----------------
  * >=2 *test.csv files -> FILE-BASED 50/50 (seeded): half the files -> train pool,
    half -> that dataset's test set. No window from a CSV appears in two splits.
  * exactly 1 *test.csv file -> TEST-ONLY: the single file goes entirely to that
    dataset's test set and contributes NOTHING to the combined train pkl (a
    file-based split is impossible with one file).

Class balancing of the combined train pkl
------------------------------------------
A window is anomalous iff  future_labels.sum() >= --anomaly_threshold  (the SAME
rule the trainer applies in derive_future_type), else normal. Anomalous windows are
RARE, so we keep them and subsample the abundant normal windows to a target ratio.

  --balance_scope:
    global           : keep ALL anomalous windows from all datasets; subsample
                       normals globally to round(normal_ratio * total_anomalous).
    per_dataset      : within each dataset, keep all its anomalous windows and
                       subsample its normals to round(normal_ratio * anom_d).
                       Fixes class balance but anomaly-rich datasets still dominate.
    per_dataset_cap  : (default) cap each dataset's anomalous contribution to
                       --per_dataset_anom_cap (default = median anomalous count
                       across contributing datasets) BEFORE matching normals to
                       round(normal_ratio * kept_anom_d). Equalises each dataset's
                       vote so big datasets (SVDB/MITDB) can't swamp small ones,
                       while never starving rare-anomaly datasets (the cap only
                       trims datasets that have MORE anomalies than the cap).

  --normal_ratio: normals per anomalous window in the final train set (default 2.0:
                  keeps a normal majority so the forecasting prior stays strong while
                  still ~10x over-representing anomalies vs their natural <5% rate).

  --pure_normal_only: draw the normal pool only from anomaly-free windows
                      (future_labels.sum()==0) instead of "< threshold" windows.

Memory is bounded by reservoir-sampling the (abundant) normal windows per dataset;
the (rare) anomalous windows are kept in full.

Usage
-----
    python prepare_total.py                                   # all defaults
    python prepare_total.py --balance_scope global --normal_ratio 1.0
    python prepare_total.py --datasets SMD MSL SMAP          # subset
    python prepare_total.py --stride 64 --test_stride 64
"""

import argparse
import glob
import json
import logging
import os
import pickle
import sys

import numpy as np
from tqdm import tqdm

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
#  Reservoir sampler — uniform random sample of a stream, bounded memory
# ─────────────────────────────────────────────────────────────────────────────

class Reservoir:
    """Uniform reservoir sample of at most `cap` items from an arbitrarily long
    stream. `seen` records the true population size so callers can warn when the
    sample is smaller than what they need."""

    def __init__(self, cap: int, rng: np.random.Generator):
        self.cap = cap
        self.rng = rng
        self.items: list = []
        self.seen = 0

    def add(self, item) -> None:
        self.seen += 1
        if len(self.items) < self.cap:
            self.items.append(item)
        else:
            j = int(self.rng.integers(0, self.seen))
            if j < self.cap:
                self.items[j] = item


# ─────────────────────────────────────────────────────────────────────────────
#  Per-dataset file discovery + 50/50 file-based split
# ─────────────────────────────────────────────────────────────────────────────

def list_datasets(data_root: str, only: list[str] | None) -> list[str]:
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


def split_files(test_csvs: list[str], test_fraction: float, seed: int):
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
#  Window classification (matches the trainer's derive_future_type)
# ─────────────────────────────────────────────────────────────────────────────

def is_anom_window(pair: dict, threshold: int) -> bool:
    return int(pair["future_labels"].sum()) >= threshold


def is_pure_normal(pair: dict) -> bool:
    return int(pair["future_labels"].sum()) == 0


# ─────────────────────────────────────────────────────────────────────────────
#  Balancing (global / per_dataset / per_dataset_cap)
# ─────────────────────────────────────────────────────────────────────────────

def _subsample(items: list, k: int, rng: np.random.Generator) -> list:
    """Uniform sample of k items without replacement (k clamped to len)."""
    if k >= len(items):
        return list(items)
    idx = rng.choice(len(items), size=k, replace=False)
    return [items[i] for i in idx]


def balance_train(per_ds: dict, scope: str, normal_ratio: float,
                  anom_cap, seed: int) -> tuple[list, dict]:
    """Combine per-dataset (anom list, normal Reservoir) into one balanced train
    list. Returns (train_pairs, stats)."""
    rng = np.random.default_rng(seed + 777)
    stats = {"scope": scope, "normal_ratio": normal_ratio, "datasets": {}}

    # Resolve the per-dataset anomalous cap for per_dataset_cap mode.
    cap = None
    if scope == "per_dataset_cap":
        anom_counts = [len(v["anom"]) for v in per_ds.values() if len(v["anom"]) > 0]
        if anom_cap in (None, "median"):
            cap = int(np.median(anom_counts)) if anom_counts else 0
        else:
            cap = int(anom_cap)
        cap = max(1, cap)
        stats["per_dataset_anom_cap"] = cap
        logger.info(f"per_dataset_cap: anomalous cap = {cap} "
                    f"(median of per-dataset anomalous counts unless overridden)")

    train: list = []

    if scope == "global":
        all_anom, normal_pool, normal_seen = [], [], 0
        for ds, v in per_ds.items():
            all_anom.extend(v["anom"])
            normal_pool.extend(v["normal"].items)
            normal_seen += v["normal"].seen
        target_norm = int(round(normal_ratio * len(all_anom)))
        sel_norm = _subsample(normal_pool, target_norm, rng)
        if target_norm > len(normal_pool):
            logger.warning(
                f"[global] wanted {target_norm} normals but only {len(normal_pool)} "
                f"in the reservoir (seen {normal_seen}); raise --normal_reservoir_cap "
                f"if this is below target. Achieved ratio "
                f"{len(sel_norm) / max(1, len(all_anom)):.2f}:1")
        train = all_anom + sel_norm
        stats["total_anomalous"] = len(all_anom)
        stats["total_normal"] = len(sel_norm)

    else:  # per_dataset or per_dataset_cap
        for ds, v in sorted(per_ds.items()):
            anom = v["anom"]
            if scope == "per_dataset_cap":
                anom = _subsample(anom, min(len(anom), cap), rng)
            kept_anom = len(anom)
            target_norm = int(round(normal_ratio * kept_anom))
            sel_norm = _subsample(v["normal"].items, target_norm, rng)
            train.extend(anom)
            train.extend(sel_norm)
            stats["datasets"][ds] = {
                "anomalous_kept": kept_anom,
                "normal_kept": len(sel_norm),
                "normal_seen": v["normal"].seen,
                "anomalous_available": len(v["anom"]),
            }
            if target_norm > len(v["normal"].items):
                logger.warning(
                    f"[{ds}] wanted {target_norm} normals but reservoir holds "
                    f"{len(v['normal'].items)} (seen {v['normal'].seen}); "
                    f"achieved {len(sel_norm) / max(1, kept_anom):.2f}:1")
        stats["total_anomalous"] = sum(d["anomalous_kept"] for d in stats["datasets"].values())
        stats["total_normal"] = sum(d["normal_kept"] for d in stats["datasets"].values())

    stats["final_windows"] = len(train)
    return train, stats


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
    res_rng = np.random.default_rng(args.seed + 123)

    # per_ds[ds] = {"anom": [pairs...], "normal": Reservoir}
    per_ds: dict[str, dict] = {}
    manifest: dict = {"datasets": {}, "config": vars(args).copy()}

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

        # ── TRAIN half -> per-dataset anom list + normal reservoir ──────────────
        ds_entry = {"anom": [], "normal": Reservoir(args.normal_reservoir_cap, res_rng)}
        train_anom = train_normal = 0
        if train_files:
            train_pairs, _, _, _ = build_pairs_for_files(
                train_files, args.context_length, args.prediction_length,
                args.stride, min_req, f"{ds}/train")
            for p in train_pairs:
                if is_anom_window(p, args.anomaly_threshold):
                    ds_entry["anom"].append(p)
                    train_anom += 1
                else:
                    if args.pure_normal_only and not is_pure_normal(p):
                        continue
                    ds_entry["normal"].add(p)
                    train_normal += 1
        per_ds[ds] = ds_entry

        logger.info(f"  TRAIN -> anomalous={train_anom}, normal_pool={train_normal} "
                    f"(reservoir holds {len(ds_entry['normal'].items)})")
        manifest["datasets"][ds] = {
            "mode": mode,
            "n_test_csv": len(test_csvs),
            "train_files": [os.path.basename(f) for f in train_files],
            "test_files": [os.path.basename(f) for f in test_files],
            "test_windows": len(test_inputs),
            "test_series": len(test_meta),
            "train_anomalous": train_anom,
            "train_normal_pool": train_normal,
        }

    # ── Balance + write the combined train pkl ──────────────────────────────────
    logger.info("=" * 78)
    train_pairs, bstats = balance_train(
        per_ds, args.balance_scope, args.normal_ratio,
        args.per_dataset_anom_cap, args.seed)
    shuf = np.random.default_rng(args.seed)
    train_inputs = pairs_to_model_inputs(train_pairs, include_meta=False)
    shuf.shuffle(train_inputs)
    train_path = os.path.join(args.output_dir, "train_model_inputs.pkl")
    with open(train_path, "wb") as f:
        pickle.dump(train_inputs, f)

    # ── Validation PROBE (a copy of a subset of train — NO data is removed) ──────
    # This is a fixed monitoring set drawn FROM the train windows, so every training
    # window is still trained on. It tracks training loss on a stable sample (clean
    # normal_loss / anomaly_loss / mse curves every eval step) — it is NOT a held-out
    # generalization estimate (the per-dataset test sets are). Set --val_fraction 0
    # to skip it.
    if args.val_fraction > 0 and train_inputs:
        vrng = np.random.default_rng(args.seed + 9)
        n_val = int(round(args.val_fraction * len(train_inputs)))
        n_val = max(1, min(n_val, args.val_max, len(train_inputs)))
        vidx = vrng.choice(len(train_inputs), size=n_val, replace=False)
        val_inputs = [train_inputs[i] for i in vidx]          # references, not removed from train
        val_path = os.path.join(args.output_dir, "val_model_inputs.pkl")
        with open(val_path, "wb") as f:
            pickle.dump(val_inputs, f)
        v_anom = sum(int(d["future_labels"].sum() >= args.anomaly_threshold) for d in val_inputs)
        logger.info(f"VAL PROBE -> {len(val_inputs)} windows (subset of train, train kept intact); "
                    f"{v_anom} anomalous / {len(val_inputs) - v_anom} normal -> {val_path}")
        manifest["val_probe"] = {"windows": len(val_inputs), "anomalous": v_anom,
                                 "subset_of_train": True, "val_fraction": args.val_fraction}
    else:
        logger.info("val_fraction=0 -> no val_model_inputs.pkl (run finetune with NO_VALIDATION=1)")

    n_anom = bstats["total_anomalous"]
    n_norm = bstats["total_normal"]
    tot = n_anom + n_norm
    logger.info("=" * 78)
    logger.info("COMBINED TRAIN BALANCE")
    logger.info(f"  scope                 : {args.balance_scope}")
    logger.info(f"  normal_ratio          : {args.normal_ratio}")
    if "per_dataset_anom_cap" in bstats:
        logger.info(f"  per-dataset anom cap  : {bstats['per_dataset_anom_cap']}")
    logger.info(f"  anomalous windows kept: {n_anom}")
    logger.info(f"  normal windows kept   : {n_norm}")
    logger.info(f"  final train windows   : {tot}  "
                f"({100 * n_anom / max(1, tot):.1f}% anomalous / "
                f"{100 * n_norm / max(1, tot):.1f}% normal) -> {train_path}")
    if train_inputs:
        logger.info(f"  per-window target shape: {train_inputs[0]['target'].shape}  "
                    f"(= [{NORMAL_SIGNAL_LENGTH} normal | {args.context_length} context "
                    f"| {args.prediction_length} future]; F varies per dataset)")

    manifest["balance"] = bstats
    with open(os.path.join(args.output_dir, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2, default=str)
    logger.info(f"Manifest -> {os.path.join(args.output_dir, 'manifest.json')}")


def main():
    p = argparse.ArgumentParser(description="Whole-mTSBench data prep for Chronos-2 anomaly fine-tuning.")
    p.add_argument("--data_root", default="/home/rajib/mTSBench/Datasets/mTSBench",
                   help="Root containing one sub-folder per dataset")
    p.add_argument("--output_dir", default=os.path.join(_HERE, "prepared_total"))
    p.add_argument("--datasets", nargs="*", default=None,
                   help="Optional subset of dataset names (default: all discovered)")

    # Window geometry — must match run_finetune / run_forward
    p.add_argument("--context_length", type=int, default=512)
    p.add_argument("--prediction_length", type=int, default=64)
    p.add_argument("--stride", type=int, default=64, help="Train sliding-window stride")
    p.add_argument("--test_stride", type=int, default=64, help="Test sliding-window stride")
    p.add_argument("--min_length", type=int, default=50)

    # Split
    p.add_argument("--test_fraction", type=float, default=0.5,
                   help="Fraction of each dataset's *test.csv files held out for testing")
    p.add_argument("--seed", type=int, default=42)

    # Balancing
    p.add_argument("--anomaly_threshold", type=int, default=10,
                   help="Window is anomalous iff >= this many anomalous steps (matches trainer)")
    p.add_argument("--balance_scope", default="per_dataset_cap",
                   choices=["global", "per_dataset", "per_dataset_cap"])
    p.add_argument("--normal_ratio", type=float, default=2.0,
                   help="Normal windows per anomalous window in the combined train pkl")
    p.add_argument("--per_dataset_anom_cap", default="median",
                   help="'median' (default) or an int: cap on each dataset's anomalous "
                        "contribution (per_dataset_cap scope only)")
    p.add_argument("--pure_normal_only", action="store_true",
                   help="Draw normals only from anomaly-free windows (sum==0)")
    p.add_argument("--val_fraction", type=float, default=0.0,
                   help="Validation PROBE size as a fraction of the combined train windows. "
                        "The probe is a COPY of a random subset of train (no data removed from "
                        "training); it tracks training loss on a fixed sample. 0 = disabled.")
    p.add_argument("--val_max", type=int, default=5000,
                   help="Hard cap on the validation-probe size (keeps eval cheap on big sets)")
    p.add_argument("--normal_reservoir_cap", type=int, default=200000,
                   help="Per-dataset reservoir size bounding normal-window memory")
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
