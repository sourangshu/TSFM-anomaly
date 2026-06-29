"""
Phase 3 — Construct global normal signal candidates and compare.

For the whole dataset (K=1) AND for the silhouette-suggested clustering (K>=2),
we build two kinds of reference per group:

  medoid : the REAL series whose normalized waveform is most central (max mean
           phase-robust fft-similarity to the other group members). Preserves
           genuine periodic structure -> your plan option 1, done right.
  dba    : per-channel DTW Barycenter Average (tslearn) over the group's members.
           A SYNTHETIC signal that averages while preserving periodicity, instead
           of the flat mush a plain mean produces -> your plan option 2, done right.

Each candidate is scored by its mean fft-similarity to the members it must
represent (higher = better reference). Candidates are stored in NORMALIZED space
(per-channel z-scored); Phase 4 re-scales into each target series (option 2a).

Outputs (in <OUT>/candidates/):
  global_medoid.npz / global_dba.npz                 (F,L) + metadata (K=1)
  cluster_medoids.npz / cluster_dba.npz              (K,F,L) + assignments (K>=2)
  compare_channels.png                               medoid vs dba vs real overlays
  scores.txt                                         mean-sim-to-members per candidate
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.cluster import AgglomerativeClustering
from tslearn.barycenters import dtw_barycenter_averaging

HERE = os.path.dirname(os.path.abspath(__file__))
ART = os.environ.get("OUT_DIR", os.path.join(HERE, "artifacts"))
CAND = os.path.join(ART, "candidates")
REPORT = os.path.join(ART, "similarity_report")
EPS = 1e-8
DBA_ITERS = int(os.environ.get("DBA_ITERS", 10))


def fft_sim_to_group(cand: np.ndarray, members: np.ndarray) -> float:
    """Mean channel-averaged fft-magnitude correlation of cand to each member."""
    cm = np.abs(np.fft.rfft(cand, axis=1))[:, 1:]          # (F, K/2)
    mm = np.abs(np.fft.rfft(members, axis=2))[:, :, 1:]    # (M, F, K/2)
    cm = cm - cm.mean(1, keepdims=True)
    mm = mm - mm.mean(2, keepdims=True)
    cn = np.sqrt((cm ** 2).sum(1)) + EPS
    mn = np.sqrt((mm ** 2).sum(2)) + EPS
    corr = (cm[None] * mm).sum(2) / (cn[None] * mn)        # (M, F)
    return float(corr.mean())


def build_medoid(norm, idxs, sim):
    """Index (into full array) of the most central member by mean fft-sim."""
    sub = sim[np.ix_(idxs, idxs)].copy()
    np.fill_diagonal(sub, np.nan)
    best_local = int(np.nanargmax(np.nanmean(sub, axis=1)))
    return idxs[best_local]


def build_dba(norm, idxs):
    """Per-channel DTW barycenter over members -> (F, L)."""
    F, L = norm.shape[1], norm.shape[2]
    out = np.zeros((F, L), dtype=np.float32)
    for f in range(F):
        series = norm[idxs, f, :][:, :, None]              # (M, L, 1)
        bar = dtw_barycenter_averaging(series, max_iter=DBA_ITERS)
        out[f] = bar[:, 0]
    return out


def main():
    os.makedirs(CAND, exist_ok=True)
    d = np.load(os.path.join(ART, "normals.npz"), allow_pickle=True)
    names, norm = list(d["names"]), d["norm"]
    N, F, L = norm.shape
    sim = np.load(os.path.join(REPORT, "sim_fft.npy"))     # phase-robust similarity
    print(f"[construct] N={N} F={F} L={L}")

    score_lines = []
    all_idx = np.arange(N)

    # ── K = 1 : single global signal ─────────────────────────────────────────
    med_i = build_medoid(norm, all_idx, sim)
    med = norm[med_i]
    dba = build_dba(norm, all_idx)
    s_med = fft_sim_to_group(med, norm)
    s_dba = fft_sim_to_group(dba, norm)
    np.savez_compressed(os.path.join(CAND, "global_medoid.npz"),
                        signal=med, source=names[med_i], score=s_med)
    np.savez_compressed(os.path.join(CAND, "global_dba.npz"),
                        signal=dba, score=s_dba)
    score_lines += [f"K=1 medoid  source={names[med_i]:30s} mean_fft_sim={s_med:.3f}",
                    f"K=1 dba     (synthetic)                       mean_fft_sim={s_dba:.3f}"]

    # ── K >= 2 : cluster medoids / dba ───────────────────────────────────────
    K = int(os.environ.get("K", 2))
    dist = np.clip(1.0 - sim, 0, None); np.fill_diagonal(dist, 0.0)
    labels = AgglomerativeClustering(n_clusters=K, metric="precomputed",
                                     linkage="average").fit_predict(dist)
    cmed = np.zeros((K, F, L), np.float32); cdba = np.zeros((K, F, L), np.float32)
    cmed_src = []
    for k in range(K):
        idxs = all_idx[labels == k]
        mi = build_medoid(norm, idxs, sim) if len(idxs) > 1 else idxs[0]
        cmed[k] = norm[mi]; cmed_src.append(names[mi])
        cdba[k] = build_dba(norm, idxs)
        sm = fft_sim_to_group(cmed[k], norm[idxs])
        sd = fft_sim_to_group(cdba[k], norm[idxs])
        members = ",".join(n.replace("SMD_machine-", "").replace("_test.csv", "") for n in
                           [names[i] for i in idxs])
        score_lines += [f"K={K} c{k} medoid src={names[mi]:24s} sim={sm:.3f} (m={len(idxs)}) [{members}]",
                        f"K={K} c{k} dba    (synthetic)            sim={sd:.3f} (m={len(idxs)})"]
    np.savez_compressed(os.path.join(CAND, "cluster_medoids.npz"),
                        signals=cmed, sources=np.array(cmed_src), labels=labels, names=np.array(names))
    np.savez_compressed(os.path.join(CAND, "cluster_dba.npz"),
                        signals=cdba, labels=labels, names=np.array(names))

    # ── Comparison plot: a few channels, medoid vs dba vs real members ───────
    chans = [0, F // 4, F // 2, 3 * F // 4][:4]
    fig, axes = plt.subplots(len(chans), 1, figsize=(11, 2.4 * len(chans)))
    for ax, f in zip(np.atleast_1d(axes), chans):
        for i in all_idx:
            ax.plot(norm[i, f], color="0.8", lw=0.6, zorder=1)
        ax.plot(med[f], color="tab:blue", lw=1.8, label="K1 medoid", zorder=3)
        ax.plot(dba[f], color="tab:red", lw=1.8, label="K1 dba", zorder=3)
        ax.set_ylabel(f"ch {f}"); ax.set_xlim(0, L)
    np.atleast_1d(axes)[0].legend(loc="upper right", fontsize=8)
    np.atleast_1d(axes)[0].set_title("Global candidates vs all real normals (grey)")
    fig.tight_layout(); fig.savefig(os.path.join(CAND, "compare_channels.png"), dpi=130); plt.close(fig)

    with open(os.path.join(CAND, "scores.txt"), "w") as fh:
        fh.write("\n".join(score_lines) + "\n")
    print("\n".join(score_lines))
    print(f"\n[construct] candidates -> {CAND}")


if __name__ == "__main__":
    main()
