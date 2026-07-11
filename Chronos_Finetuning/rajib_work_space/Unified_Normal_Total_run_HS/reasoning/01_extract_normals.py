"""
Phase 1 — Extract + normalize per-series normal signals for a dataset.

For every *test.csv in DATA_DIR we reuse the EXISTING logic from
inst_data_prepare_labeled.py (no duplication) to:
  1. load the multivariate series + per-step labels,
  2. find the non-anomalous (normal) zones,
  3. carve a (F, L) reference normal waveform (L = NORMAL_SIGNAL_LENGTH).

We then z-normalize that waveform PER CHANNEL using statistics computed from ALL
of the series' normal-zone samples (more robust than the 256-window alone). The
per-series, per-channel (mean, std) are stored so the chosen GLOBAL signal can
later be re-scaled into any target series' own normalized space (plan option 2a).

Output: <OUT>/normals.npz with
    names      : (N,)        str  — csv basenames
    raw        : (N, F, L)   f32  — raw normal waveforms (NaN-padded if too short)
    norm       : (N, F, L)   f32  — per-channel z-normalized waveforms
    means      : (N, F)      f32  — per-series per-channel normal mean
    stds       : (N, F)      f32  — per-series per-channel normal std (>=eps)
"""

import os
import sys
import glob
import numpy as np

# ── Make the local code root importable, then reuse its pure helpers ──────────
HERE = os.path.dirname(os.path.abspath(__file__))


def _find_work_root(start: str, marker: str = "inst_data_prepare_labeled.py") -> str:
    """Walk up from `start` to the dir containing `marker` (rajib_work_space)."""
    d = start
    while True:
        if os.path.exists(os.path.join(d, marker)):
            return d
        parent = os.path.dirname(d)
        if parent == d:
            raise RuntimeError(f"Could not locate {marker} above {start}")
        d = parent


WORK_ROOT = _find_work_root(HERE)                      # .../rajib_work_space
sys.path.insert(0, WORK_ROOT)
from inst_data_prepare_labeled import (                # noqa: E402
    load_csv_as_multivariate,
    extract_anomaly_boundaries,
    get_normal_zones,
    extract_normal_signal,
    NORMAL_SIGNAL_LENGTH,
)

# ── Config ────────────────────────────────────────────────────────────────────
DATA_DIR = os.environ.get("DATA_DIR", "/home/rajib/mTSBench/Datasets/mTSBench/SMD")
OUT_DIR  = os.environ.get("OUT_DIR",  os.path.join(HERE, "artifacts"))
LENGTH   = int(os.environ.get("NORMAL_SIGNAL_LENGTH", NORMAL_SIGNAL_LENGTH))
EPS      = 1e-8


def normal_zone_stats(data: np.ndarray, zones: list[tuple[int, int]]) -> tuple[np.ndarray, np.ndarray]:
    """Per-channel (mean, std) over all normal-zone samples. NaN-safe; std>=EPS."""
    if zones:
        cols = np.concatenate([data[:, s:e] for s, e in zones], axis=1)
    else:
        cols = data
    mean = np.nanmean(cols, axis=1)
    std = np.nanstd(cols, axis=1)
    mean = np.nan_to_num(mean, nan=0.0)
    std = np.nan_to_num(std, nan=1.0)
    std = np.where(std < EPS, 1.0, std)
    return mean.astype(np.float32), std.astype(np.float32)


def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    csv_files = sorted(glob.glob(os.path.join(DATA_DIR, "**", "*test.csv"), recursive=True))
    print(f"[extract] DATA_DIR = {DATA_DIR}")
    print(f"[extract] found {len(csv_files)} *test.csv files; LENGTH = {LENGTH}")
    if not csv_files:
        raise SystemExit("No *test.csv files found.")

    names, raws, norms, means_l, stds_l = [], [], [], [], []
    n_feat = None
    for path in csv_files:
        feat, lbl = load_csv_as_multivariate(path)
        if feat is None:
            print(f"  skip (no features): {os.path.basename(path)}")
            continue
        zones = get_normal_zones(extract_anomaly_boundaries(lbl), len(lbl))
        sig = extract_normal_signal(feat, zones, LENGTH)        # (F, L) or None
        if sig is None:
            print(f"  skip (no normal zones): {os.path.basename(path)}")
            continue

        # Enforce consistent channel count within a dataset.
        if n_feat is None:
            n_feat = sig.shape[0]
        elif sig.shape[0] != n_feat:
            print(f"  skip (channel mismatch {sig.shape[0]}!={n_feat}): {os.path.basename(path)}")
            continue

        mean, std = normal_zone_stats(feat, zones)
        norm = (sig - mean[:, None]) / std[:, None]
        # Any residual NaN (padding) -> 0 in normalized space (= the channel mean).
        norm = np.nan_to_num(norm, nan=0.0).astype(np.float32)

        names.append(os.path.basename(path))
        raws.append(sig.astype(np.float32))
        norms.append(norm)
        means_l.append(mean)
        stds_l.append(std)
        print(f"  ok  {os.path.basename(path):35s} F={sig.shape[0]} L={sig.shape[1]}")

    raw = np.stack(raws)          # (N, F, L)
    norm = np.stack(norms)        # (N, F, L)
    means = np.stack(means_l)     # (N, F)
    stds = np.stack(stds_l)       # (N, F)
    out = os.path.join(OUT_DIR, "normals.npz")
    np.savez_compressed(out, names=np.array(names), raw=raw, norm=norm, means=means, stds=stds)
    print(f"[extract] N={len(names)} F={raw.shape[1]} L={raw.shape[2]} -> {out}")


if __name__ == "__main__":
    main()
