"""
Phase 2 — Validation / EDA: is ONE global normal signal justified, or K?

For every pair of series we compute a per-channel similarity and average over the
38 channels, under THREE complementary notions of "same kind of normal":

  pearson : raw Pearson corr of the z-normalized waveforms (phase-SENSITIVE).
            Two unaligned realizations of the same process score ~0, so this is
            the fragile metric your original plan relied on — shown for contrast.
  fft     : Pearson corr of magnitude spectra (phase-ROBUST; captures periodicity
            regardless of where each cycle starts).
  dtw     : soft-DTW / DTW distance on the waveforms (alignment-aware), reported
            as a SIMILARITY = 1/(1+normalized_distance).

Outputs (in <OUT>/similarity_report/):
  sim_<metric>.npy            (N,N) matrices
  heatmap_<metric>.png        heatmaps
  offdiag_hist.png            distribution of off-diagonal similarities
  cluster_silhouette.png      silhouette vs K (on the phase-robust fft metric)
  summary.txt                 stats + 1-vs-K recommendation

Decision rule printed at the end: high & tight off-diagonal fft similarity ->
one global signal; bimodal/low -> K cluster medoids (K from best silhouette).
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score
from tslearn.metrics import dtw as ts_dtw

HERE = os.path.dirname(os.path.abspath(__file__))
ART = os.environ.get("OUT_DIR", os.path.join(HERE, "artifacts"))
REPORT = os.path.join(ART, "similarity_report")
EPS = 1e-8


def _corr(a: np.ndarray, b: np.ndarray) -> float:
    a = a - a.mean(); b = b - b.mean()
    da, db = np.sqrt((a * a).sum()), np.sqrt((b * b).sum())
    if da < EPS or db < EPS:
        return 0.0
    return float((a * b).sum() / (da * db))


def pairwise(norm: np.ndarray):
    """Return three (N,N) similarity matrices averaged over channels."""
    N, F, L = norm.shape
    mags = np.abs(np.fft.rfft(norm, axis=2))          # (N, F, L//2+1) magnitude spectra
    P = np.eye(N); Q = np.eye(N); D = np.zeros((N, N))

    for i in range(N):
        for j in range(i + 1, N):
            pc = np.mean([_corr(norm[i, f], norm[j, f]) for f in range(F)])
            fc = np.mean([_corr(mags[i, f, 1:], mags[j, f, 1:]) for f in range(F)])  # drop DC
            dd = np.mean([ts_dtw(norm[i, f], norm[j, f]) for f in range(F)])
            P[i, j] = P[j, i] = pc
            Q[i, j] = Q[j, i] = fc
            D[i, j] = D[j, i] = dd

    # DTW distance -> similarity in [0,1], normalized by the median off-diagonal dist.
    off = D[~np.eye(N, dtype=bool)]
    scale = np.median(off) if off.size else 1.0
    S = 1.0 / (1.0 + D / (scale + EPS))
    np.fill_diagonal(S, 1.0)
    return {"pearson": P, "fft": Q, "dtw": S}, D


def heatmap(M, names, title, path):
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(M, vmin=min(0, M.min()), vmax=1, cmap="viridis")
    ax.set_xticks(range(len(names))); ax.set_yticks(range(len(names)))
    short = [n.replace("SMD_machine-", "").replace("_test.csv", "") for n in names]
    ax.set_xticklabels(short, rotation=90, fontsize=6)
    ax.set_yticklabels(short, fontsize=6)
    ax.set_title(title); fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


def main():
    d = np.load(os.path.join(ART, "normals.npz"), allow_pickle=True)
    names, norm = list(d["names"]), d["norm"]
    N = len(names)
    os.makedirs(REPORT, exist_ok=True)
    print(f"[validate] N={N} F={norm.shape[1]} L={norm.shape[2]}")

    sims, D = pairwise(norm)
    for k, M in sims.items():
        np.save(os.path.join(REPORT, f"sim_{k}.npy"), M)
        heatmap(M, names, f"{k} similarity (channel-averaged)", os.path.join(REPORT, f"heatmap_{k}.png"))
    np.save(os.path.join(REPORT, "dtw_dist.npy"), D)

    # Off-diagonal distributions
    mask = ~np.eye(N, dtype=bool)
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.5))
    lines = []
    for ax, (k, M) in zip(axes, sims.items()):
        vals = M[mask]
        ax.hist(vals, bins=20, color="steelblue", edgecolor="k", alpha=0.8)
        ax.set_title(k); ax.set_xlabel("similarity")
        lines.append(f"  {k:8s}  mean={vals.mean():.3f}  median={np.median(vals):.3f}  "
                     f"min={vals.min():.3f}  max={vals.max():.3f}  std={vals.std():.3f}")
    fig.suptitle("Off-diagonal similarity distribution")
    fig.tight_layout(); fig.savefig(os.path.join(REPORT, "offdiag_hist.png"), dpi=130); plt.close(fig)

    # Clustering suggestion on the phase-robust fft metric (distance = 1 - sim)
    fft = sims["fft"]
    dist = np.clip(1.0 - fft, 0, None)
    np.fill_diagonal(dist, 0.0)
    ks, scores = [], []
    for K in range(2, min(7, N)):
        lab = AgglomerativeClustering(n_clusters=K, metric="precomputed",
                                      linkage="average").fit_predict(dist)
        if len(set(lab)) < 2:
            continue
        s = silhouette_score(dist, lab, metric="precomputed")
        ks.append(K); scores.append(s)
    best_k = ks[int(np.argmax(scores))] if scores else 1

    if scores:
        fig, ax = plt.subplots(figsize=(5, 3.2))
        ax.plot(ks, scores, "o-"); ax.set_xlabel("K"); ax.set_ylabel("silhouette")
        ax.set_title(f"Best K = {best_k} (fft metric)")
        fig.tight_layout(); fig.savefig(os.path.join(REPORT, "cluster_silhouette.png"), dpi=130); plt.close(fig)

    # Recommendation heuristic on the fft metric
    fft_off = fft[mask]
    fc_mean, fc_min = fft_off.mean(), fft_off.min()
    one_ok = (fc_mean >= 0.5) and (fc_min >= 0.2) and (not scores or max(scores) < 0.5)
    rec = "ONE global signal looks justified" if one_ok else \
          f"PREFER K cluster-medoids (suggested K={best_k}); a single signal may be a poor fit"

    txt = [f"N = {N}", "", "Off-diagonal similarity stats:", *lines, "",
           f"fft silhouette by K: " + ", ".join(f"K{k}={s:.3f}" for k, s in zip(ks, scores)),
           f"best K = {best_k}", "", f"RECOMMENDATION: {rec}"]
    with open(os.path.join(REPORT, "summary.txt"), "w") as f:
        f.write("\n".join(txt) + "\n")
    print("\n".join(txt))
    print(f"\n[validate] artifacts -> {REPORT}")


if __name__ == "__main__":
    main()
