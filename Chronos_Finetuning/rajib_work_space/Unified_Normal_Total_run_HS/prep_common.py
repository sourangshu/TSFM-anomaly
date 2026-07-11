"""
Standalone windowing / carving primitives for the UNIFIED-NORMAL HS data prep.

Every function here is copied VERBATIM from SMD_run/prepare_smd_split.py (the
canonical primitives used by TOTAL_RUN / TOTAL_RUN_maskloss_v2 / _HS), plus the
two unified-normal helpers (normal_zone_stats, rescale_to_series) copied from
SMD_run/Unified_Normal/prepare_global_normal.py. They live here so this folder
has NO import dependency on any sibling folder in rajib_work_space (only the
`chronos` package is imported by the training/inference entrypoints).

Because the functions are verbatim, windows built with global_norm=None are
byte-identical to those of TOTAL_RUN_maskloss_v2_HS: same context, same future,
same per-series normal prefix, same metadata. With a global_norm, only the
256-step normal prefix changes — the global shape re-scaled into each series'
own per-channel normal-zone units.
"""

import logging
import os
import pickle

import numpy as np
import pandas as pd
from tqdm import tqdm

# Same instruction-prefix length as inst_data_prepare_labeled.py / all other arms
NORMAL_SIGNAL_LENGTH = 256

EPS = 1e-8

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
#  Data Loading  (identical to SMD_run/prepare_smd_split.py)
# ─────────────────────────────────────────────────────────────────────────────

def load_csv_as_multivariate(csv_path: str):
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
#  Anomaly Boundary / Normal Zone Helpers  (identical)
# ─────────────────────────────────────────────────────────────────────────────

def extract_anomaly_boundaries(labels: np.ndarray):
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


def get_normal_zones(boundaries, total: int):
    """Normal (non-anomaly) zones as (start, end) pairs."""
    zones, prev = [], 0
    for s, e in boundaries:
        if s > prev:
            zones.append((prev, s))
        prev = e
    if prev < total:
        zones.append((prev, total))
    return zones


def extract_normal_signal(data: np.ndarray, normal_zones, length: int):
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


# ─────────────────────────────────────────────────────────────────────────────
#  Unified-normal helpers  (identical to SMD_run/Unified_Normal/prepare_global_normal.py)
# ─────────────────────────────────────────────────────────────────────────────

def normal_zone_stats(data: np.ndarray, zones) -> tuple:
    """Per-channel (mean, std) over all of a series' normal-zone samples.

    NaN-safe. A zero/degenerate std is clamped to 1.0 so re-scaling never blows up.
    Empty zones (fully-anomalous series) fall back to whole-series stats.
    """
    cols = np.concatenate([data[:, s:e] for s, e in zones], axis=1) if zones else data
    mean = np.nan_to_num(np.nanmean(cols, axis=1), nan=0.0)
    std = np.nan_to_num(np.nanstd(cols, axis=1), nan=1.0)
    std = np.where(std < EPS, 1.0, std)
    return mean.astype(np.float32), std.astype(np.float32)


def rescale_to_series(global_norm: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    """Map the normalized global shape into a target series' raw units."""
    return (global_norm * std[:, None] + mean[:, None]).astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
#  Pair Construction  (identical)
# ─────────────────────────────────────────────────────────────────────────────

def create_pairs(data, labels, context_length, prediction_length, stride):
    """
    Slide a window over the series. For each start t (from context_length onward):

      context        = data[:, t - context_length : t]      (always full, real steps)
      future         = data[:, t : t + prediction_length]   (full window only)
      future_labels  = labels[t : t + prediction_length]    (one label per future step)

    Windows with fewer than `prediction_length` future steps remaining are skipped.
    """
    pairs = []
    total = data.shape[1]
    for t in range(context_length, total, stride):
        fut_end = t + prediction_length
        if fut_end > total:
            break
        ctx = data[:, t - context_length:t].astype(np.float32, copy=False)
        fut = data[:, t:fut_end].astype(np.float32, copy=False)
        fut_labels = labels[t:fut_end].astype(np.int32, copy=False)
        pairs.append({
            "context": {"target": ctx},
            "future":  {"target": fut},
            "future_labels": fut_labels,
            "future_start": int(t),         # global index of first future step in the series
            "future_end":   int(fut_end),   # exclusive
        })
    return pairs


def _attach_normal_signal(pairs, normal_sig):
    """In-place: attach the same normal_signal reference to every pair of a series."""
    for p in pairs:
        p["normal_signal"] = normal_sig


def pairs_to_model_inputs(pairs, include_meta: bool = False):
    """
    Convert pairs to fixed-length model inputs:

        [normal_signal (256) | context (C) | future (P)]

    Each output dict carries `future_labels` (P,) int array (0=normal, 1=anomaly).

    When `include_meta=True` (used for the TEST split) each entry also carries the
    positional metadata needed to scatter the 64 per-step predictions back onto the
    original series timeline for series-based metrics (VUS-PR etc.):
        series_id, future_start, future_end, series_length
    """
    out = []
    for p in pairs:
        ctx, fut = p["context"]["target"], p["future"]["target"]
        normal = p.get("normal_signal")
        if normal is None:
            normal = np.full((ctx.shape[0], NORMAL_SIGNAL_LENGTH), np.nan, dtype=np.float32)
        target = np.concatenate([normal, ctx, fut], axis=1)
        entry = {"target": target, "future_labels": p["future_labels"]}
        if include_meta:
            entry["series_id"]     = p.get("series_id")
            entry["future_start"]  = p.get("future_start")
            entry["future_end"]    = p.get("future_end")
            entry["series_length"] = p.get("series_length")
        out.append(entry)
    return out


# ─────────────────────────────────────────────────────────────────────────────
#  Per-file pair building
# ─────────────────────────────────────────────────────────────────────────────

def build_pairs_for_files(files, context_length, prediction_length, stride, min_req, tag,
                          global_norm: np.ndarray | None = None):
    """
    Load each CSV (in the given order) and build pairs. Each pair is tagged with
    its series_id and series_length, so the windows of a series stay grouped and
    temporally ordered.

    Normal prefix:
      global_norm is None  -> PER-SERIES prefix carved from that series' own
                              normal zones (verbatim TOTAL_RUN_maskloss_v2_HS
                              behaviour — byte-identical windows).
      global_norm (F, 256) -> UNIFIED prefix: the dataset's global normalized
                              shape re-scaled into each series' own per-channel
                              normal-zone mean/std. Files whose channel count
                              doesn't match global_norm are skipped (warned).

    Also returns `series_meta`: series_id -> {length, labels (full per-timestamp
    ground truth), n_features, context_length}. This is the ground truth used to
    compute series-based metrics (VUS-PR), including the per-step labels for the
    leading `context_length` steps that are never a forecast target.
    """
    all_pairs, used, skipped, series_meta = [], [], 0, {}
    for path in tqdm(files, desc=f"Building {tag} pairs", unit="file"):
        feat, lbl = load_csv_as_multivariate(path)
        if feat is None or feat.shape[1] < min_req:
            length = feat.shape[1] if feat is not None else "None"
            logger.info(f"  skip {os.path.basename(path)} (length={length} < {min_req})")
            skipped += 1
            continue
        if global_norm is not None and feat.shape[0] != global_norm.shape[0]:
            logger.warning(f"  skip channel mismatch ({feat.shape[0]}!={global_norm.shape[0]}): "
                           f"{os.path.basename(path)}")
            skipped += 1
            continue
        sid = os.path.basename(path)
        pairs = create_pairs(feat, lbl, context_length, prediction_length, stride)
        zones = get_normal_zones(extract_anomaly_boundaries(lbl), len(lbl))
        if global_norm is None:
            normal_sig = extract_normal_signal(feat, zones, NORMAL_SIGNAL_LENGTH)
        else:
            mean, std = normal_zone_stats(feat, zones)
            normal_sig = rescale_to_series(global_norm, mean, std)
        _attach_normal_signal(pairs, normal_sig)
        for p in pairs:
            p["series_id"] = sid
            p["series_length"] = int(feat.shape[1])
        all_pairs.extend(pairs)
        used.append(sid)
        series_meta[sid] = {
            "length": int(feat.shape[1]),
            "labels": lbl.astype(np.int32, copy=False),
            "n_features": int(feat.shape[0]),
            "context_length": int(context_length),
        }
    return all_pairs, used, skipped, series_meta
