"""
Whole-mTSBench sliding-window data prep for Chronos-2 anomaly fine-tuning — the
UNIFIED-NORMAL variant of the UNCAPPED, PER-DATASET prep used by the
HIERARCHICAL SAMPLER (HS).

Per-window layout is the same as every other arm:

    [ normal_signal (256) | context (512) | future (64) ]   + future_labels (64,)

EXACTLY ONE thing differs from ../TOTAL_RUN_maskloss_v2_HS/prepare_total.py:
the 256-step normal prefix. There it is carved PER SERIES from that series' own
normal zones. Here every series of a dataset shares ONE global normal SHAPE —
the dataset's TRAINING-file medoid (build_unified_signal.py), z-normalized —
re-scaled into each series' own per-channel normal-zone mean/std so it sits in
the same units as that series' context. Context, future, labels, metadata and
the file split are byte-identical by construction (same primitives, same seed).

  IMPORTANT — no leakage: each dataset's medoid is derived from its TRAINING
  files only. Test series never influence the shared reference; at test time
  they receive the training medoid re-scaled by their own normal-zone stats
  (exactly the SMD_run/Unified_Normal recipe).

  TEST-ONLY datasets (a single *test.csv → no training files) have no
  leakage-free medoid, so they keep the PER-SERIES prefix — their test pkls are
  byte-identical to the HS arm's, isolating the unified-normal effect to the
  12 trained datasets.

Everything else mirrors ../TOTAL_RUN_maskloss_v2_HS/prepare_total.py verbatim:
no cap, no threshold, no class balancing, nothing discarded — every window of
every dataset's train half enters the pool, grouped per dataset so the
hierarchical sampler (finetune_anomaly_simple.py) gets its level-1 groups for
free:

    level 1   draw a DATASET uniformly                     -> kills dataset imbalance
    level 2   draw a KIND: anomalous w.p. p_anom (=1/3)    -> kills class imbalance
    level 3   draw a WINDOW from that dataset, count-weighted by
              n_anom (anomalous kind) or 64 - n_anom (normal kind)

Products:
  1. per_dataset/<DATASET>/train_model_inputs.pkl   (UNCAPPED train half, unified prefix)
     per_dataset/<DATASET>/train_n_anom.npy         (int16 (N,) sidecar)
     per_dataset/<DATASET>/global_normal_signal.npz (the dataset's normalized medoid shape)
  2. per_dataset/<DATASET>/{test_model_inputs.pkl, test_series_meta.pkl}
     (TEST half, UNCAPPED, unified prefix — or per-series prefix for test-only datasets)
  3. manifest.json  (adds unified_reference / unified_mean_similarity per dataset)

NOTE: unlike the HS arm, the test pkls of TRAINED datasets can NOT be symlinked
from ../TOTAL_RUN_maskloss_v2 — their normal prefix differs. --link_test_from
therefore applies to TEST-ONLY datasets only (whose bytes are identical anyway).

Usage
-----
    python prepare_total.py --data_root /path/to/mTSBench    # carve everything
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

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)                        # local, standalone primitives
from prep_common import (                        # noqa: E402
    NORMAL_SIGNAL_LENGTH,
    build_pairs_for_files,
    pairs_to_model_inputs,
)
from build_unified_signal import build_unified_signal   # noqa: E402

logger = logging.getLogger("prepare_total")


# ─────────────────────────────────────────────────────────────────────────────
#  Per-dataset file discovery + 50/50 file-based split  (verbatim from HS)
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

    Identical to TOTAL_RUN / TOTAL_RUN_maskloss_v2 / _HS: same seed, same
    permutation, so the train/test file assignment is the same across all arms."""
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
#  Unified normal signal — one medoid per dataset, from TRAINING files only
# ─────────────────────────────────────────────────────────────────────────────

def derive_global_normal(ds, ds_out, train_files, metric: str):
    """Derive the dataset's medoid shape from its TRAINING files and save it.

    Returns the normalized (F, NORMAL_SIGNAL_LENGTH) shape."""
    res = build_unified_signal(csv_files=train_files, length=NORMAL_SIGNAL_LENGTH,
                               metric=metric)
    sig = res["signal"].astype(np.float32)
    np.savez_compressed(
        os.path.join(ds_out, "global_normal_signal.npz"),
        signal=sig, reference=res["reference"], metric=metric,
        mean_similarity=res["mean_similarity"],
        normal_signal_length=NORMAL_SIGNAL_LENGTH,
    )
    logger.info(f"  UNIFIED normal: medoid={res['reference']} of {len(train_files)} train "
                f"files (mean {metric}-sim to others={res['mean_similarity']:.3f}) "
                f"-> shape {sig.shape}")
    return sig, res["reference"], res["mean_similarity"]


# ─────────────────────────────────────────────────────────────────────────────
#  Pipeline
# ─────────────────────────────────────────────────────────────────────────────

def _link_or_build_test(ds, ds_out, test_files, args, min_req, src_manifest,
                        global_norm, mode):
    """Build the TEST pkls (unified prefix for trained datasets, per-series for
    test-only), or — TEST-ONLY datasets only — symlink them from an existing
    prepared_total.

    Linking a trained dataset's test half would be WRONG here: its normal prefix
    differs from every other arm. Test-only datasets keep the per-series prefix,
    so their bytes are identical and linking is safe."""
    test_pkl = os.path.join(ds_out, "test_model_inputs.pkl")
    meta_pkl = os.path.join(ds_out, "test_series_meta.pkl")

    if args.link_test_from and mode == "test_only":
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
                    f"{n_anom} anomalous <- {src}  [per-series prefix, test-only]")
        return n_win, n_series, n_anom

    test_pairs, _, _, test_meta = build_pairs_for_files(
        test_files, args.context_length, args.prediction_length,
        args.test_stride, min_req, f"{ds}/test", global_norm=global_norm)
    test_inputs = pairs_to_model_inputs(test_pairs, include_meta=True)
    with open(test_pkl, "wb") as f:
        pickle.dump(test_inputs, f)
    with open(meta_pkl, "wb") as f:
        pickle.dump(test_meta, f)
    n_anom = int((anomaly_step_counts(test_inputs) >= 1).sum())
    prefix = "per-series (test-only)" if global_norm is None else "UNIFIED"
    logger.info(f"  TEST  -> {len(test_inputs)} windows ({len(test_meta)} series), "
                f"{n_anom} anomalous -> {ds_out}  [UNCAPPED, {prefix} prefix]")
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
    logger.info("UNIFIED NORMAL: one medoid per dataset, derived from its TRAINING files "
                f"only (metric={args.metric}); test-only datasets keep per-series prefixes.")

    src_manifest = {}
    if args.link_test_from:
        mpath = os.path.join(os.path.abspath(args.link_test_from), "manifest.json")
        if os.path.exists(mpath):
            with open(mpath) as f:
                src_manifest = json.load(f)
        logger.info(f"TEST halves of TEST-ONLY datasets will be SYMLINKED from "
                    f"{args.link_test_from} (identical bytes — per-series prefix). "
                    "Trained datasets are always carved here (their prefix differs).")

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

        # ── UNIFIED normal shape (trained datasets only; TRAIN files only) ───
        global_norm, uni_ref, uni_sim = None, None, None
        if train_files:
            global_norm, uni_ref, uni_sim = derive_global_normal(
                ds, ds_out, train_files, args.metric)

        # ── TEST half (UNCAPPED, ordered + metadata) ─────────────────────────
        t_win, t_series, t_anom = _link_or_build_test(
            ds, ds_out, test_files, args, min_req, src_manifest, global_norm, mode)

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
            "unified_reference": uni_ref,
            "unified_mean_similarity": uni_sim,
        }

        if train_files:
            train_pairs, _, _, _ = build_pairs_for_files(
                train_files, args.context_length, args.prediction_length,
                args.stride, min_req, f"{ds}/train", global_norm=global_norm)
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
                f"-> {100.0 * exp_frac:.1f}% under HS (p_anom={args.p_anom:.3f})  "
                f"[UNIFIED prefix: {uni_ref}]"
            )
            del train_inputs, n_anom
        else:
            logger.info("  TRAIN -> none (test-only dataset; invisible to level 1; "
                        "per-series prefix)")

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
    logger.info("  normal prefix    : UNIFIED per dataset (training-file medoid, re-scaled "
                "per series); per-series for test-only datasets")
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
        "normal_prefix": "unified_per_dataset_medoid",
        "unified_metric": args.metric,
    }
    with open(os.path.join(args.output_dir, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2, default=str)
    logger.info(f"Manifest -> {os.path.join(args.output_dir, 'manifest.json')}")


def main():
    p = argparse.ArgumentParser(
        description="Uncapped, thresholdless, per-dataset mTSBench data prep for the "
                    "hierarchical sampler (HS) with a UNIFIED per-dataset normal prefix.")
    p.add_argument("--data_root", default="/home/rajib/mTSBench/Datasets/mTSBench",
                   help="Root containing one sub-folder per dataset")
    p.add_argument("--output_dir", default=os.path.join(_HERE, "prepared_total"))
    p.add_argument("--datasets", nargs="*", default=None,
                   help="Optional subset of dataset names (default: all discovered)")

    # Unified normal signal
    p.add_argument("--metric", default="fft", choices=["fft", "pearson"],
                   help="Similarity used for per-dataset medoid selection (fft = phase-robust)")

    # Window geometry — must match run_finetune / run_forward
    p.add_argument("--context_length", type=int, default=512)
    p.add_argument("--prediction_length", type=int, default=64)
    p.add_argument("--stride", type=int, default=64, help="Train sliding-window stride")
    p.add_argument("--test_stride", type=int, default=64,
                   help="Test sliding-window stride (MUST equal prediction_length so "
                        "test windows tile contiguously for forward.py)")
    p.add_argument("--min_length", type=int, default=50)

    # Split — same seed as TOTAL_RUN / TOTAL_RUN_maskloss_v2 / _HS, so the file
    # assignment matches across all arms
    p.add_argument("--test_fraction", type=float, default=0.5,
                   help="Fraction of each dataset's *test.csv files held out for testing")
    p.add_argument("--seed", type=int, default=42)

    # Evaluation reuse — TEST-ONLY datasets only (their per-series-prefix bytes are
    # identical to every other arm's; trained datasets' prefixes differ and are
    # always carved here).
    p.add_argument("--link_test_from", default=None,
                   help="Path to an existing prepared_total whose TEST-ONLY datasets' "
                        "per_dataset/<DS>/test_*.pkl should be SYMLINKED instead of "
                        "regenerated. E.g. ../TOTAL_RUN_maskloss_v2_HS/prepared_total")

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
