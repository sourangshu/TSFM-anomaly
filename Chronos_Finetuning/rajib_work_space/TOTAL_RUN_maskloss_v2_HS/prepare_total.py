"""
Whole-mTSBench sliding-window data prep for Chronos-2 anomaly fine-tuning — the
UNCAPPED, PER-DATASET variant used by the HIERARCHICAL SAMPLER (HS).

Per-window layout is BYTE-FOR-BYTE identical to SMD_run / TOTAL_RUN / TOTAL_RUN_maskloss_v2:

    [ normal_signal (256) | context (512) | future (64) ]   + future_labels (64,)

The carving / loading / pairing primitives are imported verbatim from
SMD_run/prepare_smd_split.py so behaviour matches SMD and the test sets exactly.

WHY THIS VERSION EXISTS
-----------------------
There are TWO independent imbalances in combined mTSBench:

  * DATASET imbalance — MITDB + SVDB are ~88% of all windows.
  * CLASS imbalance   — anomaly STEPS are ~17% of forecast steps.

Each previous arm handled these with the wrong number of tools:

  ../TOTAL_RUN/prepare_total.py          (FT) : one prep-time knob (window-level
      anomaly threshold `future_labels.sum() >= 10` + 2:1 normal:anomaly ratio).
      The threshold contradicts a per-step loss.
  ../TOTAL_RUN_maskloss_v2/prepare_total.py (2x) : per-dataset cap at prep +
      ONE global count-weighted sampler at train time. A single weight vector
      cannot control two imbalances at once — in practice the sampler silently
      undoes the cap (MITDB/SVDB/cicids end up drawn at 1.35x their pool share,
      GHL/OPPORTUNITY at 0.68x).
  TOTAL_RUN_maskloss_v2 + FT's data       (3x) : both of the above, stacked. Best
      results so far, but it inherits FT's window-level threshold.

THIS file does the prep half of the fix: it does NOTHING except carve windows.
No cap, no threshold, no class balancing, nothing discarded — every window of
every dataset's train half enters the pool. All balancing moves to train time,
into the hierarchical sampler in finetune_anomaly_simple.py, which decomposes
the two imbalances into two independent levels:

    level 1   draw a DATASET uniformly                     -> kills dataset imbalance
    level 2   draw a KIND: anomalous w.p. p_anom (=1/3)    -> kills class imbalance
    level 3   draw a WINDOW from that dataset, count-weighted by
              n_anom (anomalous kind) or 64 - n_anom (normal kind)

Level 3 is thresholdless in both branches: the anomalous branch is self-gating
(a pure-normal window has n_anom = 0, hence weight 0), and the normal branch
weights an anomaly-bearing window down in proportion to how anomalous it is.

WHY PER-DATASET PKLs (not one combined pkl)
-------------------------------------------
Level 1 needs the pool grouped by dataset. Writing it grouped means the sampler
gets its index groups for free instead of reconstructing them from a `dataset`
key that would have to be smuggled past Chronos2Dataset's input validation.
It also caps prep-time RAM at one dataset (MITDB, ~1.2 GB) instead of the whole
6.9 GB pool, and makes prep resumable and subsettable.

Products:
  1. per_dataset/<DATASET>/train_model_inputs.pkl   (UNCAPPED train half)
     per_dataset/<DATASET>/train_n_anom.npy         (int16 (N,) sidecar: anomaly
         steps per window — lets the sampler's expected balance be audited
         without loading 6.9 GB of windows)
  2. per_dataset/<DATASET>/{test_model_inputs.pkl, test_series_meta.pkl}
     (TEST half, UNCAPPED — evaluation must see every test window)
  3. manifest.json

Datasets with exactly one *test.csv are TEST-ONLY: they contribute no train
windows, so they are absent from the train pool and invisible to level 1.

Usage
-----
    python prepare_total.py                                  # carve everything
    python prepare_total.py --link_test_from ../TOTAL_RUN_maskloss_v2/prepared_total
                                                             # reuse existing test pkls
    python prepare_total.py --datasets SMD MSL SMAP          # subset
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

    >=2 files -> seeded file split. Exactly 1 file -> TEST-ONLY (train empty).

    Identical to TOTAL_RUN / TOTAL_RUN_maskloss_v2: same seed, same permutation,
    so the train/test file assignment is the same across all four arms."""
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
#  Window bookkeeping (no balancing — HS does that at train time)
# ─────────────────────────────────────────────────────────────────────────────

def anomaly_step_counts(inputs) -> np.ndarray:
    """Per-window anomaly-step count n_anom in [0, prediction_length]."""
    return np.asarray(
        [int(np.asarray(d["future_labels"]).sum()) for d in inputs], dtype=np.int16
    )


def hs_expected_anom_step_frac(n_anom: np.ndarray, H: int, p_anom: float):
    """Expected anomaly-STEP fraction the HS sampler delivers for ONE dataset.

    Level 3 draws window i with probability
        (1 - p_anom) * n_norm_i / sum(n_norm)  +  p_anom * n_anom_i / sum(n_anom)
    so the expected anomaly steps per drawn window is the same mixture applied to
    n_anom_i. Returns (expected_fraction, natural_fraction) or (nan, nat) when the
    dataset has no anomalies at all."""
    n = n_anom.astype(np.float64)
    n_norm = H - n
    natural = float(n.mean() / H) if len(n) else float("nan")
    s_a, s_n = n.sum(), n_norm.sum()
    if s_a <= 0 or s_n <= 0:
        return float("nan"), natural
    e_anom = float((n * n).sum() / s_a)        # E[n_anom | anomalous-kind draw]
    e_norm = float((n_norm * n).sum() / s_n)   # E[n_anom | normal-kind draw]
    return float((p_anom * e_anom + (1.0 - p_anom) * e_norm) / H), natural


# ─────────────────────────────────────────────────────────────────────────────
#  Pipeline
# ─────────────────────────────────────────────────────────────────────────────

def _link_or_build_test(ds, ds_out, test_files, args, min_req, src_manifest):
    """Symlink the TEST pkls from an existing prepared_total, or carve them.

    Linking is safe because the test half is a pure function of (test_files,
    context_length, prediction_length, test_stride, min_length) and the file split
    is seeded identically — the bytes are the same. Linking guarantees, rather than
    merely asserts, that every arm evaluates on identical windows."""
    test_pkl = os.path.join(ds_out, "test_model_inputs.pkl")
    meta_pkl = os.path.join(ds_out, "test_series_meta.pkl")

    if args.link_test_from:
        src = os.path.join(os.path.abspath(args.link_test_from), "per_dataset", ds)
        src_test, src_meta = os.path.join(src, "test_model_inputs.pkl"), \
                             os.path.join(src, "test_series_meta.pkl")
        if not (os.path.exists(src_test) and os.path.exists(src_meta)):
            raise SystemExit(
                f"--link_test_from given but {src_test} is missing. Either point it at a "
                f"complete prepared_total, or drop the flag to carve the test half here."
            )
        for dst, s in ((test_pkl, src_test), (meta_pkl, src_meta)):
            if os.path.islink(dst) or os.path.exists(dst):
                os.remove(dst)
            os.symlink(s, dst)
        stats = (src_manifest.get("datasets", {}) or {}).get(ds, {})
        n_win, n_series, n_anom = (stats.get("test_windows"), stats.get("test_series"),
                                   stats.get("test_anomalous"))
        if n_win is None:               # source manifest incomplete -> count for real
            with open(src_test, "rb") as f:
                ti = pickle.load(f)
            with open(src_meta, "rb") as f:
                tm = pickle.load(f)
            n_win, n_series = len(ti), len(tm)
            n_anom = int((anomaly_step_counts(ti) >= 1).sum())
        logger.info(f"  TEST  -> LINKED {n_win} windows ({n_series} series), "
                    f"{n_anom} anomalous <- {src}")
        return n_win, n_series, n_anom

    test_pairs, _, _, test_meta = build_pairs_for_files(
        test_files, args.context_length, args.prediction_length,
        args.test_stride, min_req, f"{ds}/test")
    test_inputs = pairs_to_model_inputs(test_pairs, include_meta=True)
    with open(test_pkl, "wb") as f:
        pickle.dump(test_inputs, f)
    with open(meta_pkl, "wb") as f:
        pickle.dump(test_meta, f)
    n_anom = int((anomaly_step_counts(test_inputs) >= 1).sum())
    logger.info(f"  TEST  -> {len(test_inputs)} windows ({len(test_meta)} series), "
                f"{n_anom} anomalous -> {ds_out}  [UNCAPPED]")
    return len(test_inputs), len(test_meta), n_anom


def prepare(args) -> None:
    os.makedirs(args.output_dir, exist_ok=True)
    per_ds_root = os.path.join(args.output_dir, "per_dataset")
    os.makedirs(per_ds_root, exist_ok=True)

    datasets = list_datasets(args.data_root, args.datasets)
    if not datasets:
        raise SystemExit(f"No datasets with *test.csv found under {args.data_root}")
    logger.info(f"Datasets ({len(datasets)}): {datasets}")
    logger.info("NO CAP, NO THRESHOLD, NO CLASS BALANCING — every train window is kept. "
                "Both imbalances are handled at train time by the hierarchical sampler.")

    src_manifest = {}
    if args.link_test_from:
        mpath = os.path.join(os.path.abspath(args.link_test_from), "manifest.json")
        if os.path.exists(mpath):
            with open(mpath) as f:
                src_manifest = json.load(f)
        logger.info(f"TEST halves will be SYMLINKED from {args.link_test_from} "
                    "(identical bytes -> identical evaluation across arms)")

    min_req = max(args.min_length, args.context_length + args.prediction_length)

    manifest: dict = {"datasets": {}, "config": vars(args).copy()}
    train_datasets: list = []          # only those that contribute to the pool
    grand_anom = grand_norm = 0
    grand_anom_steps = grand_total_steps = 0

    for ds in datasets:
        ddir = os.path.join(args.data_root, ds)
        test_csvs = sorted(glob.glob(os.path.join(ddir, "**", "*test.csv"), recursive=True))
        train_files, test_files, mode = split_files(test_csvs, args.test_fraction, args.seed)

        logger.info("=" * 78)
        logger.info(f"{ds}: {len(test_csvs)} *test.csv  [{mode}]  "
                    f"train_files={len(train_files)} test_files={len(test_files)}")

        ds_out = os.path.join(per_ds_root, ds)
        os.makedirs(ds_out, exist_ok=True)

        # ── TEST half (UNCAPPED, ordered + metadata) ─────────────────────────
        t_win, t_series, t_anom = _link_or_build_test(
            ds, ds_out, test_files, args, min_req, src_manifest)

        # ── TRAIN half — carve and keep EVERYTHING ───────────────────────────
        entry = {
            "mode": mode,
            "n_test_csv": len(test_csvs),
            "train_files": [os.path.basename(f) for f in train_files],
            "test_files": [os.path.basename(f) for f in test_files],
            "test_windows": t_win,
            "test_series": t_series,
            "test_anomalous": t_anom,
            "train_windows": 0,
            "train_anom_windows": 0,
            "train_norm_windows": 0,
            "in_train_pool": False,
        }

        if train_files:
            train_pairs, _, _, _ = build_pairs_for_files(
                train_files, args.context_length, args.prediction_length,
                args.stride, min_req, f"{ds}/train")
            train_inputs = pairs_to_model_inputs(train_pairs, include_meta=False)
            del train_pairs

            n_anom = anomaly_step_counts(train_inputs)
            n_anom_win = int((n_anom >= 1).sum())
            n_norm_win = len(train_inputs) - n_anom_win

            # Level 2's anomalous branch draws from weights proportional to n_anom.
            # A dataset with no anomaly windows has an all-zero weight vector, which
            # the sampler cannot draw from. Fail here rather than at step 3000.
            if n_anom_win == 0:
                raise SystemExit(
                    f"{ds}: train half has {len(train_inputs)} windows but ZERO anomaly "
                    f"windows. The hierarchical sampler's anomalous branch would have an "
                    f"all-zero weight vector for this dataset. Exclude it with --datasets, "
                    f"or re-split so its train half contains at least one anomaly."
                )
            if n_anom_win < args.min_anom_windows:
                logger.warning(
                    f"  [{ds}] only {n_anom_win} anomaly windows. Level 1 draws every "
                    f"dataset uniformly, so these few windows will be revisited heavily "
                    f"(~1/3 of this dataset's draws land on them). Watch for overfitting."
                )

            with open(os.path.join(ds_out, "train_model_inputs.pkl"), "wb") as f:
                pickle.dump(train_inputs, f)
            np.save(os.path.join(ds_out, "train_n_anom.npy"), n_anom)

            exp_frac, nat_frac = hs_expected_anom_step_frac(
                n_anom, args.prediction_length, args.p_anom)
            mean_anom_steps = float(n_anom[n_anom >= 1].mean())

            entry.update({
                "train_windows": len(train_inputs),
                "train_anom_windows": n_anom_win,
                "train_norm_windows": n_norm_win,
                "train_anom_steps": int(n_anom.sum()),
                "train_total_steps": len(train_inputs) * args.prediction_length,
                "train_mean_anom_steps_per_anom_window": round(mean_anom_steps, 2),
                "natural_anom_step_frac": round(nat_frac, 4),
                "hs_expected_anom_step_frac": round(exp_frac, 4),
                "in_train_pool": True,
                "channels": int(train_inputs[0]["target"].shape[0]),
            })
            train_datasets.append(ds)
            grand_anom += n_anom_win
            grand_norm += n_norm_win
            grand_anom_steps += int(n_anom.sum())
            grand_total_steps += len(train_inputs) * args.prediction_length

            logger.info(
                f"  TRAIN -> kept ALL {len(train_inputs)} windows "
                f"(anom={n_anom_win}, normal={n_norm_win}); "
                f"F={entry['channels']}, anomaly steps {100.0 * nat_frac:.1f}% natural "
                f"-> {100.0 * exp_frac:.1f}% under HS (p_anom={args.p_anom:.3f})"
            )
            del train_inputs, n_anom
        else:
            logger.info("  TRAIN -> none (test-only dataset; invisible to level 1)")

        manifest["datasets"][ds] = entry

    # ── Summary ──────────────────────────────────────────────────────────────
    logger.info("=" * 78)
    tot = grand_anom + grand_norm
    logger.info(f"TRAIN POOL (uncapped; both imbalances are the sampler's job)")
    logger.info(f"  datasets in pool : {len(train_datasets)}/{len(datasets)}  {train_datasets}")
    logger.info(f"  anomaly windows  : {grand_anom}")
    logger.info(f"  normal windows   : {grand_norm}")
    logger.info(f"  total windows    : {tot}  "
                f"({100.0 * grand_anom / max(1, tot):.1f}% anomaly-bearing)")
    logger.info(f"  anomaly steps    : {grand_anom_steps}/{grand_total_steps} "
                f"({100.0 * grand_anom_steps / max(1, grand_total_steps):.1f}% natural)")
    logger.info(f"  under HS each dataset is drawn {100.0 / max(1, len(train_datasets)):.1f}% "
                f"of the time regardless of size")
    logger.info(f"  per-window target: [{NORMAL_SIGNAL_LENGTH} normal | {args.context_length} "
                f"context | {args.prediction_length} future]; F varies per dataset")
    logger.info("  no val_model_inputs.pkl -> run finetune with NO_VALIDATION=1 "
                "(EVAL_TEST stays manual via TEST_DATA)")

    manifest["train_pool"] = {
        "datasets": train_datasets,
        "n_datasets": len(train_datasets),
        "anomaly_windows": grand_anom,
        "normal_windows": grand_norm,
        "total_windows": tot,
        "anomaly_steps": grand_anom_steps,
        "total_steps": grand_total_steps,
        "natural_anom_step_frac": round(grand_anom_steps / max(1, grand_total_steps), 4),
        "per_dataset_draw_prob": round(1.0 / max(1, len(train_datasets)), 4),
        "capped": False,
        "thresholded": False,
    }
    with open(os.path.join(args.output_dir, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2, default=str)
    logger.info(f"Manifest -> {os.path.join(args.output_dir, 'manifest.json')}")


def main():
    p = argparse.ArgumentParser(
        description="Uncapped, thresholdless, per-dataset mTSBench data prep for the "
                    "hierarchical sampler (HS).")
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

    # Split — same seed as TOTAL_RUN / TOTAL_RUN_maskloss_v2, so the file assignment matches
    p.add_argument("--test_fraction", type=float, default=0.5,
                   help="Fraction of each dataset's *test.csv files held out for testing")
    p.add_argument("--seed", type=int, default=42)

    # Evaluation reuse: symlink the (byte-identical) test halves instead of re-carving.
    p.add_argument("--link_test_from", default=None,
                   help="Path to an existing prepared_total whose per_dataset/<DS>/"
                        "test_*.pkl should be SYMLINKED instead of regenerated. Makes "
                        "'inference is identical across arms' a fact, not a claim, and "
                        "halves prep time. E.g. ../TOTAL_RUN_maskloss_v2/prepared_total")

    # Reporting only — the sampler owns this knob; mirrored here so the manifest can
    # record the expected per-dataset balance the pool will produce.
    p.add_argument("--p_anom", type=float, default=1.0 / 3.0,
                   help="Level-2 probability of drawing an ANOMALOUS-kind window. Used "
                        "here only to report the expected anomaly-step fraction per "
                        "dataset in the manifest; the sampler reads its own --p_anom.")
    p.add_argument("--min_anom_windows", type=int, default=50,
                   help="Warn (do not fail) when a dataset's train half has fewer than "
                        "this many anomaly windows — they will be revisited heavily.")
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
