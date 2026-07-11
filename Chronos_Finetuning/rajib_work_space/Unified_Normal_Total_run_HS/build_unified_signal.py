"""
Build the UNIFIED (global) normal-signal SHAPE for *any* dataset.

Given a directory of `*test.csv` files, this derives one common normal shape that
represents the whole dataset, with no per-dataset hardcoding:

  1. Carve each series' normal signal (F, L) and z-normalize it per channel.
  2. Score how representative each series is of all the others, under a
     phase-robust similarity (correlation of magnitude spectra by default).
  3. Pick the MEDOID — the real series most similar to all the rest — and return
     its normalized shape as the global signal.

The medoid is preferred over a synthetic average because averaging weakly-related
normals collapses to a near-flat, structureless signal.

Importable: `build_unified_signal(data_dir=..., length=256)` -> dict.
Standalone:  writes the chosen shape to <out>/global_normal_signal.npz.
"""

import argparse
import glob
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))


def _find_work_root(start: str, marker: str = "inst_data_prepare_labeled.py") -> str:
    """Walk up from `start` to the dir containing `marker` (rajib_work_space),
    so this script works regardless of how deep the folder is moved."""
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
from inst_data_prepare_labeled import (                    # noqa: E402
    load_csv_as_multivariate,
    extract_anomaly_boundaries,
    get_normal_zones,
    extract_normal_signal,
)

EPS = 1e-8


# ─────────────────────────────────────────────────────────────────────────────
#  Per-series extraction + normalization
# ─────────────────────────────────────────────────────────────────────────────

def _normal_zone_stats(data: np.ndarray, zones: list[tuple[int, int]]) -> tuple[np.ndarray, np.ndarray]:
    cols = np.concatenate([data[:, s:e] for s, e in zones], axis=1) if zones else data
    mean = np.nan_to_num(np.nanmean(cols, axis=1), nan=0.0)
    std = np.nan_to_num(np.nanstd(cols, axis=1), nan=1.0)
    std = np.where(std < EPS, 1.0, std)
    return mean.astype(np.float32), std.astype(np.float32)


def extract_all_normals(csv_files: list[str], length: int):
    """Return (names, norm (N,F,L)) of z-normalized per-series normal signals."""
    names, norms, n_feat = [], [], None
    for path in csv_files:
        feat, lbl = load_csv_as_multivariate(path)
        if feat is None:
            continue
        zones = get_normal_zones(extract_anomaly_boundaries(lbl), len(lbl))
        sig = extract_normal_signal(feat, zones, length)
        if sig is None:
            continue
        if n_feat is None:
            n_feat = sig.shape[0]
        elif sig.shape[0] != n_feat:
            continue                                       # channel mismatch -> skip
        mean, std = _normal_zone_stats(feat, zones)
        norm = np.nan_to_num((sig - mean[:, None]) / std[:, None], nan=0.0).astype(np.float32)
        names.append(os.path.basename(path))
        norms.append(norm)
    if not norms:
        raise ValueError("No usable normal signals could be extracted.")
    return names, np.stack(norms)


# ─────────────────────────────────────────────────────────────────────────────
#  Similarity + medoid selection
# ─────────────────────────────────────────────────────────────────────────────

def _corr(a, b):
    a = a - a.mean(); b = b - b.mean()
    da, db = np.sqrt((a * a).sum()), np.sqrt((b * b).sum())
    return 0.0 if da < EPS or db < EPS else float((a * b).sum() / (da * db))


def similarity_matrix(norm: np.ndarray, metric: str = "fft") -> np.ndarray:
    """Channel-averaged (N,N) similarity. 'fft' = magnitude-spectrum corr (phase
    robust); 'pearson' = raw waveform corr (phase sensitive)."""
    N, F, _ = norm.shape
    feats = np.abs(np.fft.rfft(norm, axis=2))[:, :, 1:] if metric == "fft" else norm
    S = np.eye(N)
    for i in range(N):
        for j in range(i + 1, N):
            S[i, j] = S[j, i] = np.mean([_corr(feats[i, f], feats[j, f]) for f in range(F)])
    return S


def select_medoid(sim: np.ndarray) -> int:
    """Index of the series with the highest mean similarity to all others."""
    M = sim.copy()
    np.fill_diagonal(M, np.nan)
    return int(np.nanargmax(np.nanmean(M, axis=1)))


# ─────────────────────────────────────────────────────────────────────────────
#  Public API
# ─────────────────────────────────────────────────────────────────────────────

def build_unified_signal(
    data_dir: str | None = None,
    csv_files: list[str] | None = None,
    length: int = 256,
    metric: str = "fft",
) -> dict:
    """Derive the dataset's global normal shape (the medoid), with no hardcoding.

    Returns dict with: signal (F,L) normalized, reference (str), mean_similarity
    (medoid's mean sim to others), names (list), similarity (N,N).
    """
    if csv_files is None:
        if data_dir is None:
            raise ValueError("Provide data_dir or csv_files.")
        csv_files = sorted(glob.glob(os.path.join(data_dir, "**", "*test.csv"), recursive=True))
    if not csv_files:
        raise ValueError(f"No *test.csv files found under {data_dir}")

    names, norm = extract_all_normals(csv_files, length)
    sim = similarity_matrix(norm, metric=metric)
    mi = select_medoid(sim)
    off = sim[mi].copy(); off[mi] = np.nan
    return {
        "signal": norm[mi],
        "reference": names[mi],
        "mean_similarity": float(np.nanmean(off)),
        "names": names,
        "similarity": sim,
        "metric": metric,
    }


def main():
    p = argparse.ArgumentParser(description="Derive a dataset's unified (global) normal shape (medoid).")
    p.add_argument("--data_dir", required=True, help="Directory with the dataset's *test.csv files")
    p.add_argument("--output_dir", default=None, help="Where to write global_normal_signal.npz (default: data_dir basename here)")
    p.add_argument("--normal_signal_length", type=int, default=256)
    p.add_argument("--metric", default="fft", choices=["fft", "pearson"])
    args = p.parse_args()

    res = build_unified_signal(data_dir=args.data_dir, length=args.normal_signal_length, metric=args.metric)
    out_dir = args.output_dir or os.path.join(HERE, f"unified_{os.path.basename(os.path.normpath(args.data_dir))}")
    os.makedirs(out_dir, exist_ok=True)
    out = os.path.join(out_dir, "global_normal_signal.npz")
    np.savez_compressed(out, signal=res["signal"], reference=res["reference"],
                        normal_signal_length=args.normal_signal_length, metric=args.metric)
    print(f"[build] series={len(res['names'])} metric={args.metric}")
    print(f"[build] medoid reference = {res['reference']}  (mean sim to others = {res['mean_similarity']:.3f})")
    print(f"[build] global shape {res['signal'].shape} -> {out}")


if __name__ == "__main__":
    main()
