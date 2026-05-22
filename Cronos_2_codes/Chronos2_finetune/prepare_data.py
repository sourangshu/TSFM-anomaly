"""
Simple sliding-window data preparation for Chronos-2 anomaly fine-tuning on mTSBench.

For each series we slide a window across every timestamp. At each start position the
context is the preceding `context_length` steps and the future is the next
`prediction_length` steps. The future is kept PURE: it is truncated at the first
normal/anomaly transition and the remainder is right-padded with NaN, so a future
window is either fully-normal or fully-anomalous (never mixed).

  pair = [CONTEXT][FUTURE]            future type ∈ {normal, anomaly}

Rules:
  - Start at t = context_length, so the context is always a full `context_length`
    real steps (no context padding).
  - Future window starts at t, length `prediction_length`, truncated at the first
    label change and right-padded with NaN.
  - The very end of the series (where no future steps remain) is excluded.
  - A `normal_signal` reference of NORMAL_SIGNAL_LENGTH steps is prepended to each
    pair as an instruction prefix:

        [normal_signal (256) | context (C) | future (P)]

Usage:
    python inst_data_prepare_simple.py [--data_dir ...] [--output_dir ...]
"""

import argparse
import glob
import logging
import os
import pickle

import numpy as np
import pandas as pd
from tqdm import tqdm

log_path = os.path.join("./prepared_data_simple/log", "prepare_data.log")
os.makedirs(os.path.dirname(log_path), exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[logging.StreamHandler(), logging.FileHandler(log_path)],
)
logger = logging.getLogger(__name__)

NORMAL_SIGNAL_LENGTH = 256   # instruction prefix length (NaN-padded if too few normal steps)


# ─────────────────────────────────────────────────────────────────────────────
#  Data Loading
# ─────────────────────────────────────────────────────────────────────────────

def load_csv_as_multivariate(csv_path: str) -> tuple[np.ndarray | None, np.ndarray | None]:
    """
    Load one *test.csv file.

    Returns
    -------
    features : float32 array (n_variates, time_steps) — timestamp/is_anomaly excluded.
    labels   : int32 array (time_steps,), 1=anomaly 0=normal (all-zero if column absent).
    """
    df = pd.read_csv(csv_path)
    feature_cols = [c for c in df.columns if c not in ("timestamp", "is_anomaly")]
    if not feature_cols:
        return None, None
    try:
        features = df[feature_cols].values.T.astype(np.float32)
        labels = df["is_anomaly"].values.astype(np.int32) if "is_anomaly" in df.columns \
            else np.zeros(df.shape[0], dtype=np.int32)
        return features, labels
    except Exception as e:
        logger.warning(f"Error processing {csv_path}: {e}")
        return None, None


# ─────────────────────────────────────────────────────────────────────────────
#  Anomaly Boundary / Normal Zone Helpers
# ─────────────────────────────────────────────────────────────────────────────

def extract_anomaly_boundaries(labels: np.ndarray) -> list[tuple[int, int]]:
    """Contiguous anomaly regions as (start, end) with end EXCLUSIVE."""
    boundaries, in_anom, start = [], False, 0
    for i, v in enumerate(labels):
        if v == 1 and not in_anom:
            in_anom, start = True, i
        elif v == 0 and in_anom:
            in_anom = False
            boundaries.append((start, i))
    if in_anom:
        boundaries.append((start, len(labels)))
    return boundaries


def get_normal_zones(boundaries: list[tuple[int, int]], total: int) -> list[tuple[int, int]]:
    """Normal (non-anomaly) zones as (start, end) pairs."""
    zones, prev = [], 0
    for s, e in boundaries:
        if s > prev:
            zones.append((prev, s))
        prev = e
    if prev < total:
        zones.append((prev, total))
    return zones


def extract_normal_signal(
    data: np.ndarray,
    normal_zones: list[tuple[int, int]],
    length: int,
) -> np.ndarray | None:
    """
    Return a (F, length) reference normal signal sampled from the series' normal zones.

      1. If a single normal zone is long enough, take its last `length` timesteps.
      2. Otherwise concatenate normal zones (longest first) until enough.
      3. If still short, left-pad with NaN.

    Returns None if there are no normal zones at all.
    """
    if not normal_zones:
        return None

    sorted_zones = sorted(normal_zones, key=lambda z: z[1] - z[0], reverse=True)
    s, e = sorted_zones[0]
    if e - s >= length:
        return data[:, e - length:e].astype(np.float32, copy=False)

    chunks, collected = [], 0
    for s, e in sorted_zones:
        chunks.append(data[:, s:e])
        collected += e - s
        if collected >= length:
            break

    combined = np.concatenate(chunks, axis=1).astype(np.float32, copy=False)
    if combined.shape[1] >= length:
        return combined[:, -length:]

    F = combined.shape[0]
    pad = np.full((F, length - combined.shape[1]), np.nan, dtype=np.float32)
    return np.concatenate([pad, combined], axis=1)


def _pad_right(arr: np.ndarray, length: int) -> np.ndarray:
    """Right-pad (or trim) a (F, T) array along time to exactly `length` steps with NaN."""
    T = arr.shape[1]
    if T == length:
        return arr.astype(np.float32, copy=False)
    if T > length:
        return arr[:, :length].astype(np.float32, copy=False)
    pad = np.full((arr.shape[0], length - T), np.nan, dtype=np.float32)
    return np.concatenate([arr, pad], axis=1).astype(np.float32, copy=False)


# ─────────────────────────────────────────────────────────────────────────────
#  Pair Construction
# ─────────────────────────────────────────────────────────────────────────────

def create_pairs(
    data: np.ndarray,
    labels: np.ndarray,
    context_length: int,
    prediction_length: int,
    stride: int,
) -> list[dict]:
    """
    Slide a window over the series. For each start t (from context_length to the end):

      context = data[t - context_length : t]            (always full, real steps)
      future  = data[t : t + prediction_length]         truncated at first label change,
                                                          right-padded with NaN.

    The future type is decided by the label at t: 0 → "normal", 1 → "anomaly".
    Windows with no future steps left are skipped.
    """
    pairs = []
    total = data.shape[1]
    for t in range(context_length, total, stride):
        fut_end = min(t + prediction_length, total)
        if fut_end <= t:
            continue                                   # no future steps remain

        fut_labels = labels[t:fut_end]
        first = fut_labels[0]
        # truncate at first transition so the future is a single signal type
        change = np.nonzero(fut_labels != first)[0]
        keep = int(change[0]) if change.size else len(fut_labels)

        ctx = data[:, t - context_length:t].astype(np.float32, copy=False)
        fut = _pad_right(data[:, t:t + keep], prediction_length)

        pairs.append({
            "context": {"target": ctx},
            "future":  {"target": fut},
            "type": "anomaly" if first == 1 else "normal",
        })
    return pairs


# ─────────────────────────────────────────────────────────────────────────────
#  Model-Ready Input Conversion
# ─────────────────────────────────────────────────────────────────────────────

def _attach_normal_signal(pairs: list[dict], normal_sig: np.ndarray | None) -> None:
    """In-place: attach the same per-series normal_signal reference to every pair."""
    for p in pairs:
        p["normal_signal"] = normal_sig


def pairs_to_model_inputs(pairs: list[dict]) -> list[dict]:
    """
    Convert pairs to fixed-length model inputs:

        [normal_signal (256) | context (C) | future (P)]

    Each output dict carries a `future_type` int (0=normal, 1=anomaly) so a
    single-stage trainer can flip the loss sign per sample.
    """
    out = []
    for p in pairs:
        ctx, fut = p["context"]["target"], p["future"]["target"]
        normal = p.get("normal_signal")
        if normal is None:
            normal = np.full((ctx.shape[0], NORMAL_SIGNAL_LENGTH), np.nan, dtype=np.float32)
        target = np.concatenate([normal, ctx, fut], axis=1)
        future_type = 1 if p.get("type") == "anomaly" else 0
        out.append({"target": target, "future_type": future_type})
    return out


# ─────────────────────────────────────────────────────────────────────────────
#  Preparation Pipeline
# ─────────────────────────────────────────────────────────────────────────────

def prepare_inputs(
    data_dir: str,
    min_length: int,
    val_fraction: float,
    context_length: int,
    prediction_length: int,
    stride: int,
    seed: int = 42,
):
    """
    Build a single set of [CONTEXT][FUTURE] pairs (future type = normal or anomaly),
    split into train/val by series, each pair carrying a 256-step normal prefix.

    Returns
    -------
    train_inputs, val_inputs, train_pairs, val_pairs, train_model_inputs, val_model_inputs
    """
    rng = np.random.default_rng(seed)

    csv_files = sorted(glob.glob(os.path.join(data_dir, "**", "*test.csv"), recursive=True))
    logger.info(f"Found {len(csv_files)} *test.csv files under {data_dir}")

    # Need full context plus at least one future step.
    min_req = max(min_length, context_length + 1)
    all_inputs, all_labels, skipped = [], [], 0
    for path in tqdm(csv_files, desc="Loading CSVs", unit="file"):
        try:
            feat, lbl = load_csv_as_multivariate(path)
            if feat is None or feat.shape[1] < min_req:
                logger.debug(
                    f"Skipping {os.path.basename(path)}: "
                    f"length={feat.shape[1] if feat is not None else 'None'}, required={min_req}"
                )
                skipped += 1
                continue
            all_inputs.append({"target": feat})
            all_labels.append(lbl)
        except Exception as exc:
            logger.warning(f"Skipping {path}: {exc}")
            skipped += 1
    logger.info(f"Usable series: {len(all_inputs)}  (skipped {skipped})")
    if not all_inputs:
        raise ValueError("No usable series found. Check data_dir and min_length.")

    idx     = rng.permutation(len(all_inputs))
    n_val   = max(1, int(len(all_inputs) * val_fraction))
    val_set = set(idx[:n_val].tolist())
    train_inputs = [all_inputs[i] for i in range(len(all_inputs)) if i not in val_set]
    val_inputs   = [all_inputs[i] for i in val_set]
    train_labels = [all_labels[i] for i in range(len(all_inputs)) if i not in val_set]
    val_labels   = [all_labels[i] for i in val_set]
    logger.info(f"Train series: {len(train_inputs)} | Val series: {len(val_inputs)}")

    def build(series_list, label_list, tag):
        out = []
        for series, lbl in tqdm(zip(series_list, label_list),
                                total=len(series_list), desc=f"Building {tag} pairs",
                                unit="series"):
            pairs = create_pairs(series["target"], lbl, context_length, prediction_length, stride)
            zones = get_normal_zones(extract_anomaly_boundaries(lbl), len(lbl))
            normal_sig = extract_normal_signal(series["target"], zones, NORMAL_SIGNAL_LENGTH)
            _attach_normal_signal(pairs, normal_sig)
            out.extend(pairs)
        return out

    train_pairs = build(train_inputs, train_labels, "Train")
    val_pairs   = build(val_inputs, val_labels, "Val")
    logger.info(f"Pairs — Train: {len(train_pairs)} | Val: {len(val_pairs)}")

    logger.info("Converting to fixed-length model inputs...")
    train_model_inputs = pairs_to_model_inputs(train_pairs)
    val_model_inputs   = pairs_to_model_inputs(val_pairs)
    logger.info(f"Model inputs — Train: {len(train_model_inputs)} | Val: {len(val_model_inputs)}")

    return train_inputs, val_inputs, train_pairs, val_pairs, train_model_inputs, val_model_inputs


# ─────────────────────────────────────────────────────────────────────────────
#  Statistics Logging
# ─────────────────────────────────────────────────────────────────────────────

def log_statistics(train_inputs, val_inputs, train_pairs, val_pairs) -> None:
    """Log shape and future-type distribution statistics."""
    series   = train_inputs + val_inputs
    lengths  = [s["target"].shape[1] for s in series]
    variates = [s["target"].shape[0] for s in series]

    logger.info("=" * 60)
    logger.info("RAW SERIES STATISTICS")
    logger.info(f"  Total series : {len(series)}")
    logger.info(f"  Time steps   : min={min(lengths)}, max={max(lengths)}, mean={np.mean(lengths):.0f}")
    logger.info(f"  Num features : min={min(variates)}, max={max(variates)}, mean={np.mean(variates):.1f}")

    all_pairs = train_pairs + val_pairs
    if all_pairs:
        counts: dict[str, int] = {}
        for p in all_pairs:
            t = p.get("type", "?")
            counts[t] = counts.get(t, 0) + 1
        logger.info("=" * 60)
        logger.info("PAIR STATISTICS")
        logger.info(f"  Train: {len(train_pairs)}  Val: {len(val_pairs)}  Total: {len(all_pairs)}")
        logger.info(f"  Avg per series : {len(all_pairs) / len(series):.1f}")
        logger.info("  Future-type distribution:")
        for type_name, count in sorted(counts.items()):
            logger.info(f"    {type_name:<12} : {count:>8}  ({count / len(all_pairs) * 100:.1f}%)")
    logger.info("=" * 60)


# ─────────────────────────────────────────────────────────────────────────────
#  Entry Point
# ─────────────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(
        description="Simple sliding-window data prep for Chronos-2 anomaly fine-tuning."
    )
    p.add_argument("--data_dir",          default="/home/rajib/mTSBench/Datasets/mTSBench",
                   help="Root directory of the mTSBench dataset")
    p.add_argument("--output_dir",        default="./prepared_data_simple",
                   help="Output directory for pairs and model inputs")
    p.add_argument("--min_length",        type=int,   default=50,
                   help="Minimum series length; shorter series are discarded")
    p.add_argument("--val_fraction",      type=float, default=0.1,
                   help="Fraction of series held out for validation")
    p.add_argument("--context_length",    type=int,   default=512,
                   help="Number of past time steps used as context")
    p.add_argument("--prediction_length", type=int,   default=64,
                   help="Number of future time steps to predict")
    p.add_argument("--stride",            type=int,   default=1,
                   help="Sliding-window stride (1 = every timestamp)")
    args = p.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    (train_inputs, val_inputs,
     train_pairs, val_pairs,
     _train_model_inputs, _val_model_inputs) = prepare_inputs(
        data_dir=args.data_dir,
        min_length=args.min_length,
        val_fraction=args.val_fraction,
        context_length=args.context_length,
        prediction_length=args.prediction_length,
        stride=args.stride,
    )

    # ── Model inputs (normal + anomaly mixed, shuffled) ──────────────────────
    # Used by the single-stage trainer where loss sign is flipped per sample.
    rng_combined = np.random.default_rng(42)
    combined_train = pairs_to_model_inputs(train_pairs)
    combined_val   = pairs_to_model_inputs(val_pairs)
    rng_combined.shuffle(combined_train)
    rng_combined.shuffle(combined_val)

    for fname, data in [
        ("train_model_inputs.pkl", combined_train),
        ("val_model_inputs.pkl",   combined_val),
    ]:
        path = os.path.join(args.output_dir, fname)
        with open(path, "wb") as f:
            pickle.dump(data, f)
        n_anom = sum(d["future_type"] for d in data)
        logger.info(
            f"combined — {len(data):>8} entries "
            f"(normal={len(data)-n_anom}, anomaly={n_anom}) → {path}"
        )

    log_statistics(train_inputs, val_inputs, train_pairs, val_pairs)


if __name__ == "__main__":
    main()
