"""
Data preparation for Chronos-2 anomaly fine-tuning with a UNIFIED (global) normal
signal shared across every series in a dataset, with a FILE-BASED train/test split.

Difference from the per-series prep (inst_data_prepare_labeled.py): instead of
carving a separate normal-signal prefix from each series, we build ONE global
normal-signal SHAPE for the whole dataset and prepend it to every window. The
shape is taken from a single reference series (the dataset "medoid") in
z-normalized space, then RE-SCALED into each target series' own per-channel
mean/std so it sits in the same units as that series' context.

Split (mirrors SMD_run/prepare_smd_split.py): the split is FILE-BASED — a fraction
of the *test.csv files go to the TEST set and the rest to the training pool, with
an optional validation set carved (still file-based) from the training pool. No
window from a given CSV ever appears in two splits.

  IMPORTANT — no leakage: the global medoid is derived from the TRAINING files
  only (test/val files never influence which series becomes the shared reference).

Final per-window layout (drop-in for run_finetune_smd.sh):

    [ normal_signal (N) | context (C) | future (P) ]   + future_labels (P,)

Outputs (written to --output_dir):
    train_model_inputs.pkl   ← training-pool files minus val   [SHUFFLED]
    val_model_inputs.pkl     ← val_fraction of the train pool  [SHUFFLED, omitted if 0]
    test_model_inputs.pkl    ← the test files                  [ORDERED + metadata, omitted if test_fraction=0]
    test_series_meta.pkl     ← per-series ground truth for the test files
    global_normal_signal.npz ← the normalized global shape, for reference

TEST entries are NOT shuffled: windows stay grouped by series and in temporal
order, each carrying series_id / future_start / future_end / series_length so the
per-window predictions can be scattered back onto the series timeline for
series-based metrics (VUS-PR etc.). test_series_meta.pkl maps
series_id -> {length, labels (full per-timestamp), n_features, context_length}.

Usage:
    python prepare_global_normal.py \
        --data_dir /home/rajib/mTSBench/Datasets/mTSBench/SMD \
        --output_dir ./prepared_global \
        --reference auto --test_fraction 0.3 --val_fraction 0.0 \
        --context_length 512 --prediction_length 64 \
        --normal_signal_length 256 --stride 64
"""

import argparse
import glob
import logging
import os
import pickle
import sys

import numpy as np
from tqdm import tqdm

# ── Reuse the pure carving / loading primitives from the per-series prep ───────
HERE = os.path.dirname(os.path.abspath(__file__))


def _find_work_root(start: str, marker: str = "inst_data_prepare_labeled.py") -> str:
    """Walk up from `start` until the dir containing `marker` is found.

    Lets this script live at any depth (e.g. moved into SMD_run/Unified_Normal)
    while still importing the per-series prep helpers from rajib_work_space.
    """
    d = start
    while True:
        if os.path.exists(os.path.join(d, marker)):
            return d
        parent = os.path.dirname(d)
        if parent == d:
            raise RuntimeError(f"Could not locate {marker} above {start}")
        d = parent


WORK_ROOT = _find_work_root(HERE)                          # .../rajib_work_space
sys.path.insert(0, WORK_ROOT)
sys.path.insert(0, HERE)                                   # for build_unified_signal
from inst_data_prepare_labeled import (                    # noqa: E402
    load_csv_as_multivariate,
    extract_anomaly_boundaries,
    get_normal_zones,
    extract_normal_signal,
)
from build_unified_signal import build_unified_signal      # noqa: E402

logger = logging.getLogger("prepare_global_normal")
EPS = 1e-8


# ─────────────────────────────────────────────────────────────────────────────
#  Per-channel normal statistics
# ─────────────────────────────────────────────────────────────────────────────

def normal_zone_stats(data: np.ndarray, zones: list[tuple[int, int]]) -> tuple[np.ndarray, np.ndarray]:
    """Per-channel (mean, std) over all of a series' normal-zone samples.

    NaN-safe. A zero/degenerate std is clamped to 1.0 so re-scaling never blows up.
    """
    cols = np.concatenate([data[:, s:e] for s, e in zones], axis=1) if zones else data
    mean = np.nan_to_num(np.nanmean(cols, axis=1), nan=0.0)
    std = np.nan_to_num(np.nanstd(cols, axis=1), nan=1.0)
    std = np.where(std < EPS, 1.0, std)
    return mean.astype(np.float32), std.astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
#  Global normal signal (normalized shape) — derived from TRAINING files only
# ─────────────────────────────────────────────────────────────────────────────

def build_global_signal(
    derive_files: list[str],
    reference: str,
    length: int,
    global_npz: str | None,
    metric: str = "fft",
) -> np.ndarray:
    """Return the (F, length) global normal-signal SHAPE in z-normalized space.

    `derive_files` is the pool the shape may be selected from — pass the TRAINING
    files only so the test/val series cannot leak into the shared reference.

    Resolution order:
      1. `--global_normal_npz`  -> load a precomputed normalized `signal` array.
      2. `reference == "auto"`  -> derive the dataset medoid from `derive_files`.
      3. otherwise              -> use the training file whose name contains `reference`.
    """
    if global_npz:
        d = np.load(global_npz, allow_pickle=True)
        sig = d["signal"].astype(np.float32)
        logger.info(f"Loaded global normal shape from {global_npz} -> {sig.shape}")
        if sig.shape[1] != length:
            raise ValueError(f"global signal length {sig.shape[1]} != normal_signal_length {length}")
        return sig

    if reference == "auto":
        res = build_unified_signal(csv_files=derive_files, length=length, metric=metric)
        logger.info(f"Auto-derived global shape from {len(derive_files)} TRAIN files: "
                    f"medoid={res['reference']} "
                    f"(mean {metric}-sim to others={res['mean_similarity']:.3f}) -> {res['signal'].shape}")
        return res["signal"].astype(np.float32)

    matches = [p for p in derive_files if reference in os.path.basename(p)]
    if len(matches) != 1:
        raise ValueError(
            f"--reference '{reference}' matched {len(matches)} TRAINING files; expected exactly 1. "
            f"(A test/val file cannot be the reference — that would leak.) "
            f"Matches: {[os.path.basename(m) for m in matches]}"
        )
    ref_path = matches[0]
    feat, lbl = load_csv_as_multivariate(ref_path)
    if feat is None:
        raise ValueError(f"Reference series has no usable features: {ref_path}")
    zones = get_normal_zones(extract_anomaly_boundaries(lbl), len(lbl))
    raw = extract_normal_signal(feat, zones, length)          # (F, length) raw units
    if raw is None:
        raise ValueError(f"Reference series has no normal zones: {ref_path}")
    mean, std = normal_zone_stats(feat, zones)
    norm = (raw - mean[:, None]) / std[:, None]
    norm = np.nan_to_num(norm, nan=0.0).astype(np.float32)    # NaN pad -> channel mean
    logger.info(f"Built global normal shape from reference {os.path.basename(ref_path)} -> {norm.shape}")
    return norm


def rescale_to_series(global_norm: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    """Map the normalized global shape into a target series' raw units (option 2a)."""
    return (global_norm * std[:, None] + mean[:, None]).astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
#  Pair construction (metadata-aware, so the TEST split can be reassembled)
# ─────────────────────────────────────────────────────────────────────────────

def create_pairs(data, labels, context_length, prediction_length, stride):
    """Slide a window; keep each future's positional metadata for reconstruction."""
    pairs, total = [], data.shape[1]
    for t in range(context_length, total, stride):
        fut_end = t + prediction_length
        if fut_end > total:
            break
        pairs.append({
            "context": {"target": data[:, t - context_length:t].astype(np.float32, copy=False)},
            "future":  {"target": data[:, t:fut_end].astype(np.float32, copy=False)},
            "future_labels": labels[t:fut_end].astype(np.int32, copy=False),
            "future_start": int(t),
            "future_end":   int(fut_end),
        })
    return pairs


def attach_normal_signal(pairs, normal_sig):
    """In-place: attach the (re-scaled global) normal_signal to every pair of a series."""
    for p in pairs:
        p["normal_signal"] = normal_sig


def pairs_to_model_inputs(pairs, length: int, include_meta: bool = False):
    """Convert pairs to fixed-length [normal | context | future] targets."""
    out = []
    for p in pairs:
        ctx, fut = p["context"]["target"], p["future"]["target"]
        normal = p.get("normal_signal")
        if normal is None:
            normal = np.full((ctx.shape[0], length), np.nan, dtype=np.float32)
        entry = {"target": np.concatenate([normal, ctx, fut], axis=1),
                 "future_labels": p["future_labels"]}
        if include_meta:
            entry["series_id"]     = p.get("series_id")
            entry["future_start"]  = p.get("future_start")
            entry["future_end"]    = p.get("future_end")
            entry["series_length"] = p.get("series_length")
        out.append(entry)
    return out


# ─────────────────────────────────────────────────────────────────────────────
#  Per-file pair building (attaches the RE-SCALED global signal)
# ─────────────────────────────────────────────────────────────────────────────

def build_pairs_for_files(files, global_norm, context_length, prediction_length,
                          stride, min_req, n_feat, tag):
    """Build windows for a list of files; attach the global signal re-scaled into
    each series' own units. Returns (all_pairs, series_meta)."""
    all_pairs, series_meta = [], {}
    for path in tqdm(files, desc=f"Building {tag} pairs", unit="file"):
        feat, lbl = load_csv_as_multivariate(path)
        if feat is None or feat.shape[1] < min_req:
            length = feat.shape[1] if feat is not None else "None"
            logger.info(f"  skip {os.path.basename(path)} (length={length} < {min_req})")
            continue
        if feat.shape[0] != n_feat:
            logger.warning(f"  skip channel mismatch ({feat.shape[0]}!={n_feat}): {os.path.basename(path)}")
            continue
        sid = os.path.basename(path)
        pairs = create_pairs(feat, lbl, context_length, prediction_length, stride)
        zones = get_normal_zones(extract_anomaly_boundaries(lbl), len(lbl))
        mean, std = normal_zone_stats(feat, zones)
        attach_normal_signal(pairs, rescale_to_series(global_norm, mean, std))
        for p in pairs:
            p["series_id"] = sid
            p["series_length"] = int(feat.shape[1])
        all_pairs.extend(pairs)
        series_meta[sid] = {
            "length": int(feat.shape[1]),
            "labels": lbl.astype(np.int32, copy=False),
            "n_features": int(feat.shape[0]),
            "context_length": int(context_length),
        }
    return all_pairs, series_meta


def dump_split(pairs, output_dir, fname, length, rng=None, include_meta=False):
    """Convert pairs → model inputs, pickle, log distribution. rng!=None → shuffle."""
    model_inputs = pairs_to_model_inputs(pairs, length, include_meta=include_meta)
    if rng is not None:
        rng.shuffle(model_inputs)
    path = os.path.join(output_dir, fname)
    with open(path, "wb") as f:
        pickle.dump(model_inputs, f)
    anom = sum(int(d["future_labels"].sum()) for d in model_inputs)
    total = sum(d["future_labels"].size for d in model_inputs)
    pct = (anom / total * 100) if total else 0.0
    order = "shuffled" if rng is not None else "ordered"
    logger.info(f"{fname:<26} {len(model_inputs):>8} windows ({order})  "
                f"(future steps: normal={total - anom}, anomaly={anom} [{pct:.1f}%]) -> {path}")
    return len(model_inputs)


# ─────────────────────────────────────────────────────────────────────────────
#  Pipeline
# ─────────────────────────────────────────────────────────────────────────────

def prepare(args) -> None:
    csv_files = sorted(glob.glob(os.path.join(args.data_dir, "**", "*test.csv"), recursive=True))
    logger.info(f"Found {len(csv_files)} *test.csv files under {args.data_dir}")
    if not csv_files:
        raise SystemExit("No *test.csv files found.")

    # 1) FILE-BASED split (deterministic via seed): test pool, train pool, optional val.
    rng = np.random.default_rng(args.seed)
    perm = rng.permutation(len(csv_files))
    n_test = max(1, int(round(len(csv_files) * args.test_fraction))) if args.test_fraction > 0 else 0
    test_idx = set(perm[:n_test].tolist())
    train_pool = [csv_files[i] for i in range(len(csv_files)) if i not in test_idx]
    test_files = [csv_files[i] for i in sorted(test_idx)]

    if args.val_fraction > 0 and len(train_pool) > 1:
        perm_tp = rng.permutation(len(train_pool))
        n_val = min(max(1, int(round(len(train_pool) * args.val_fraction))), len(train_pool) - 1)
        val_set = set(perm_tp[:n_val].tolist())
        val_files = [train_pool[i] for i in sorted(val_set)]
        train_files = [train_pool[i] for i in range(len(train_pool)) if i not in val_set]
    else:
        val_files, train_files = [], train_pool

    logger.info("=" * 70)
    logger.info("FILE-BASED SPLIT")
    logger.info(f"  Total files : {len(csv_files)}")
    logger.info(f"  Train files : {len(train_files)} -> {[os.path.basename(f) for f in train_files]}")
    logger.info(f"  Val files   : {len(val_files)} -> {[os.path.basename(f) for f in val_files]}")
    logger.info(f"  Test files  : {len(test_files)} -> {[os.path.basename(f) for f in test_files]}")
    logger.info("=" * 70)
    if not train_files:
        raise ValueError("No training files after split. Lower --test_fraction/--val_fraction.")

    # 2) Global normal-signal shape (normalized) — derived from TRAIN files only.
    global_norm = build_global_signal(train_files, args.reference, args.normal_signal_length,
                                      args.global_normal_npz, metric=args.metric)
    n_feat = global_norm.shape[0]

    # 3) Build windows per split; attach the re-scaled global signal.
    min_req = max(args.min_length, args.context_length + args.prediction_length)
    train_pairs, _ = build_pairs_for_files(train_files, global_norm, args.context_length,
                                           args.prediction_length, args.stride, min_req, n_feat, "train")
    val_pairs, _ = build_pairs_for_files(val_files, global_norm, args.context_length,
                                         args.prediction_length, args.stride, min_req, n_feat, "val")
    test_pairs, test_series_meta = build_pairs_for_files(test_files, global_norm, args.context_length,
                                                         args.prediction_length, args.stride, min_req,
                                                         n_feat, "test")
    logger.info(f"Pairs — train: {len(train_pairs)} | val: {len(val_pairs)} | test: {len(test_pairs)}")

    # 4) Write outputs. train/val SHUFFLED; test ORDERED + metadata for VUS-PR.
    os.makedirs(args.output_dir, exist_ok=True)
    np.savez_compressed(os.path.join(args.output_dir, "global_normal_signal.npz"),
                        signal=global_norm, reference=args.reference,
                        normal_signal_length=args.normal_signal_length)

    shuf = np.random.default_rng(args.seed)
    L = args.normal_signal_length
    dump_split(train_pairs, args.output_dir, "train_model_inputs.pkl", L, rng=shuf)
    if val_pairs:
        dump_split(val_pairs, args.output_dir, "val_model_inputs.pkl", L, rng=shuf)
    else:
        logger.info("val_fraction=0 → no val_model_inputs.pkl (run finetune with NO_VALIDATION=1)")
    if test_pairs:
        dump_split(test_pairs, args.output_dir, "test_model_inputs.pkl", L, rng=None, include_meta=True)
        meta_path = os.path.join(args.output_dir, "test_series_meta.pkl")
        with open(meta_path, "wb") as f:
            pickle.dump(test_series_meta, f)
        logger.info(f"test_series_meta.pkl       {len(test_series_meta):>8} series "
                    f"(per-timestamp ground-truth labels) -> {meta_path}")
    else:
        logger.info("test_fraction=0 → no test_model_inputs.pkl / test_series_meta.pkl")

    if train_pairs:
        shp = pairs_to_model_inputs(train_pairs[:1], L)[0]["target"].shape
        logger.info(f"Per-window target shape: {shp}  "
                    f"(= [{L} normal | {args.context_length} context | {args.prediction_length} future])")


def main():
    p = argparse.ArgumentParser(
        description="Unified (global) normal-signal data prep for Chronos-2 with a file-based split.")
    p.add_argument("--data_dir", default="/home/rajib/mTSBench/Datasets/mTSBench/SMD",
                   help="Directory containing the dataset's *test.csv files")
    p.add_argument("--output_dir", default=os.path.join(HERE, "prepared_global"),
                   help="Where the pkls and the global signal are written")
    p.add_argument("--reference", default="auto",
                   help="'auto' (default) derives the dataset medoid from the TRAINING files; "
                        "otherwise a filename substring selecting a training reference series")
    p.add_argument("--metric", default="fft", choices=["fft", "pearson"],
                   help="Similarity used for 'auto' medoid selection (fft = phase-robust)")
    p.add_argument("--global_normal_npz", default=None,
                   help="Optional: load a precomputed normalized global signal (key 'signal') instead of --reference")
    p.add_argument("--normal_signal_length", type=int, default=256, help="Length N of the normal-signal prefix")
    p.add_argument("--context_length", type=int, default=512, help="Length C of the context window")
    p.add_argument("--prediction_length", type=int, default=64, help="Length P of the future window")
    p.add_argument("--stride", type=int, default=64, help="Sliding-window stride")
    p.add_argument("--min_length", type=int, default=50, help="Discard series shorter than this")
    p.add_argument("--test_fraction", type=float, default=0.3,
                   help="Fraction of files held out for testing (file-based). 0 = no test split.")
    p.add_argument("--val_fraction", type=float, default=0.0,
                   help="Fraction of the TRAINING-POOL files used for validation (file-based). 0 = none.")
    p.add_argument("--seed", type=int, default=42, help="RNG seed for split + shuffle")
    args = p.parse_args()

    log_dir = os.path.join(args.output_dir, "log")
    os.makedirs(log_dir, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.StreamHandler(),
                  logging.FileHandler(os.path.join(log_dir, "prepare_global.log"))],
    )
    logger.info(f"Config: {vars(args)}")
    prepare(args)
    logger.info("Done.")


if __name__ == "__main__":
    main()
