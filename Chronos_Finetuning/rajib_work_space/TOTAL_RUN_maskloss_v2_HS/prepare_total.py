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

SYNTHETIC DATA  (--use_syn_data, OFF by default)
------------------------------------------------
Seven datasets (CalIt2, creditcard, GECCO, Genesis, metro, PSM, swan) ship exactly
one *test.csv, so the 50/50 file split has nothing to split: they are TEST-ONLY and
invisible to level 1. Synthetic files generated for them live in

    <DATA_ROOT>/<DATASET>/syn_data/*test.csv

With --use_syn_data those files are appended to the dataset's TRAIN half and NOWHERE
else. They never enter the test half (evaluation must stay real) and never enter the
val set (that comes from mTSBench's own *val.csv). The real *test.csv of a one-file
dataset therefore remains its complete, unchanged test half — so --link_test_from
still holds and forward.py's numbers stay comparable — while the dataset joins the
train pool and level 1 grows from 12 to 19 groups.

Real files are discovered at the dataset's TOP LEVEL only (<DATASET>/*test.csv), not
recursively. This is deliberate: a recursive glob also swept up Exathlon/data_not_used/,
three byte-identical copies of real Exathlon files that are not part of upstream
mTSBench (PLAN-Lab/mTSBench has no such folder). One of them, Exathlon_1_5_1000000_86,
landed in the train half AND the test half at once. Top-level discovery drops the
folder, which changes Exathlon's seeded file split — _link_or_build_test() detects
that against the source manifest and re-carves Exathlon rather than linking a stale
test half.

FILE-LEVEL DIVERSITY WEIGHTS
----------------------------
A dataset's train windows are pooled from many *test.csv files that are often near
duplicates (SVDB: 39 ECG records; MITDB: 23), and level 3's count weighting is blind
to which file a window came from — so the longest files dominate and a rare, unusual
recording is drowned out. This prep therefore also emits, per dataset, the file each
window came from plus a per-file DISSIMILARITY weight, so that the sampler can insert
an optional file level between levels 1 and 2 (see --file_diversity_datasets in
finetune_anomaly_simple.py). Weights come from an inverse kernel density over a
robust per-file signature: a file sitting in a tight cluster of near-duplicates is
down-weighted, a lone unusual file is up-weighted. Computing them here (rather than
at train time) costs no extra I/O and lets the sampler flag be flipped without a
re-prep.

Products:
  1. per_dataset/<DATASET>/train_model_inputs.pkl   (UNCAPPED train half)
     per_dataset/<DATASET>/train_n_anom.npy         (int16 (N,) sidecar: anomaly
         steps per window — lets the sampler's expected balance be audited
         without loading 6.9 GB of windows)
     per_dataset/<DATASET>/train_file_index.npy     (int32 (N,) sidecar: which
         source file each window came from, indexing train_files.json's "files")
     per_dataset/<DATASET>/train_files.json         (file names, per-file window /
         anomaly counts, dissimilarity weights, effective sample size)
  2. per_dataset/<DATASET>/{test_model_inputs.pkl, test_series_meta.pkl}
     (TEST half, UNCAPPED — evaluation must see every test window)
  3. manifest.json

Datasets with exactly one *test.csv and no usable syn_data are TEST-ONLY: they
contribute no train windows, so they are absent from the train pool and invisible
to level 1.

Usage
-----
    python prepare_total.py                                  # carve everything
    python prepare_total.py --use_syn_data                   # + syn_data into TRAIN only
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
import zlib

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

def real_test_csvs(ddir: str):
    """The dataset's real *test.csv files — TOP LEVEL ONLY, never recursive.

    Non-recursive on purpose. A recursive glob also picks up stray sub-directories
    that are not part of upstream mTSBench (Exathlon/data_not_used/ held byte-identical
    copies of three real files, one of which then sat in the train half and the test
    half simultaneously), and it would sweep syn_data/ in as if it were real data.
    Synthetic files are discovered separately by syn_test_csvs() and are TRAIN-ONLY."""
    return sorted(glob.glob(os.path.join(ddir, "*test.csv")))


def syn_test_csvs(ddir: str, syn_dir: str):
    """The dataset's SYNTHETIC *test.csv files, or [] when it has none.

    These exist for the datasets that ship a single real *test.csv and would otherwise
    be test-only. They are appended to the TRAIN half and go nowhere else: never the
    test half (evaluation stays real), never the val set (that is the dataset's own
    *val.csv)."""
    return sorted(glob.glob(os.path.join(ddir, syn_dir, "*test.csv")))


def list_datasets(data_root: str, only):
    """Dataset = any sub-directory of data_root containing >=1 real *test.csv file."""
    names = []
    for entry in sorted(os.listdir(data_root)):
        d = os.path.join(data_root, entry)
        if not os.path.isdir(d):
            continue
        if only and entry not in only:
            continue
        if real_test_csvs(d):
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
#  Per-file dissimilarity weights (the sampler's optional file level)
# ─────────────────────────────────────────────────────────────────────────────
#
# Goal: inside one dataset, stop near-duplicate files from dominating and give the
# unusual ones a real chance of being drawn, so training sees the dataset's whole
# repertoire of patterns rather than its most common one repeated.
#
#   1. signature   a fixed-length, channel-count-agnostic descriptor per file
#   2. standardize robustly across the dataset's files (median / IQR)
#   3. density     s_f = sum over the OTHER files of a Gaussian kernel on the
#                  pairwise distance, bandwidth = median pairwise distance
#   4. weight      w_f  ∝ (1 + s_f)^(-alpha)
#
# Inverse DENSITY, not mean distance: with mean distance a single far-flung outlier
# collects most of the mass and the rest stay uniform, which is not what we want. With
# inverse density every member of a tight cluster is discounted (they explain each
# other) while an isolated file — however mildly isolated — is promoted. alpha=0 gives
# uniform-over-files (already a large change from level 3's count weighting, which is
# dominated by the longest files) and is the natural ablation baseline.

_FEAT_NAMES = ("mean", "std", "skew", "kurtosis", "acf1", "acf10", "acf50",
               "roughness", "spectral_entropy")


def _channel_signature(x: np.ndarray) -> np.ndarray:
    """9 shape descriptors for ONE channel (a 1-D float array)."""
    x = x[np.isfinite(x)]
    if x.size < 8:
        return np.zeros(len(_FEAT_NAMES), dtype=np.float64)
    mu, sd = float(x.mean()), float(x.std())
    z = (x - mu) / sd if sd > 1e-12 else np.zeros_like(x)   # shape stats are scale-free
    skew = float((z ** 3).mean())
    kurt = float((z ** 4).mean()) - 3.0
    acfs = []
    for lag in (1, 10, 50):
        if z.size > lag + 1:
            a, b = z[:-lag], z[lag:]
            denom = float(np.sqrt((a * a).mean() * (b * b).mean()))
            acfs.append(float((a * b).mean() / denom) if denom > 1e-12 else 0.0)
        else:
            acfs.append(0.0)
    roughness = float(np.abs(np.diff(z)).mean()) if z.size > 1 else 0.0
    # Spectral entropy of the normalized power spectrum: low = one dominant periodicity,
    # high = broadband/noisy. Separates e.g. a clean ECG from a bursty server metric.
    p = np.abs(np.fft.rfft(z - z.mean())) ** 2
    tot = p.sum()
    if tot > 1e-12 and p.size > 1:
        p = p / tot
        nz = p[p > 0]
        spec_ent = float(-(nz * np.log(nz)).sum() / np.log(p.size))
    else:
        spec_ent = 0.0
    return np.array([mu, sd, skew, kurt, *acfs, roughness, spec_ent], dtype=np.float64)


def file_signature(sig: np.ndarray, labels: np.ndarray, n_windows: int) -> np.ndarray:
    """One file -> a fixed-length descriptor, independent of its channel count.

    `sig` is (F, T) — the file's signal, `labels` its (T,) per-step 0/1 ground truth.
    Per-channel descriptors are pooled by mean AND std across channels (the std says
    how heterogeneous the channels are, which itself distinguishes files), then the
    file's anomaly profile is appended: anomalies are a pattern to be sampled too, so
    two files with identical dynamics but different anomaly regimes must not collapse
    onto the same point."""
    per_ch = np.stack([_channel_signature(sig[c]) for c in range(sig.shape[0])])  # (F, 9)
    pooled = np.concatenate([per_ch.mean(axis=0), per_ch.std(axis=0)])            # (18,)

    lab = np.asarray(labels, dtype=np.int8).ravel()
    rate = float(lab.mean()) if lab.size else 0.0
    d = np.diff(np.concatenate([[0], lab, [0]]))
    starts, ends = np.flatnonzero(d == 1), np.flatnonzero(d == -1)
    n_seg = len(starts)
    seg_len = float((ends - starts).mean()) if n_seg else 0.0
    seg_rate = 1000.0 * n_seg / max(lab.size, 1)          # segments per 1000 steps
    anom = np.array([rate, seg_rate, np.log1p(seg_len), np.log1p(n_windows)])
    return np.concatenate([pooled, anom])                                          # (22,)


def dissimilarity_weights(feats: np.ndarray, alpha: float, cap: float):
    """Per-file sampling weights from an inverse kernel density. Returns (w, ess).

    `feats` is (n_files, D). Weights sum to 1. `cap` bounds any file at cap x (and
    1/cap x) the uniform weight 1/n, so one weird file can never eat the dataset."""
    n = len(feats)
    if n <= 1:
        return np.ones(max(n, 1), dtype=np.float64) / max(n, 1), float(n)

    # Robust standardization: median/IQR, not mean/std — a single extreme file must not
    # define the scale of the axis it is extreme on. Constant features get zero spread
    # and are dropped (they carry no information about which files differ).
    med = np.median(feats, axis=0)
    q75, q25 = np.percentile(feats, 75, axis=0), np.percentile(feats, 25, axis=0)
    iqr = q75 - q25
    keep = iqr > 1e-9
    if not keep.any():                       # every file identical under the signature
        return np.ones(n) / n, float(n)
    z = (feats[:, keep] - med[keep]) / iqr[keep]
    z = np.clip(z, -5.0, 5.0)                # a lone outlier informs, it does not dictate

    d2 = np.maximum(((z[:, None, :] - z[None, :, :]) ** 2).sum(-1), 0.0)
    off = d2[~np.eye(n, dtype=bool)]
    sigma2 = float(np.median(off[off > 0])) if (off > 0).any() else 1.0
    if sigma2 <= 1e-12:
        sigma2 = 1.0

    kern = np.exp(-d2 / (2.0 * sigma2))
    np.fill_diagonal(kern, 0.0)
    density = kern.sum(axis=1)               # how much "company" each file keeps

    w = (1.0 + density) ** (-float(alpha))
    w = w / w.sum()

    if cap and cap > 1.0:
        lo, hi = 1.0 / (cap * n), cap / n
        for _ in range(50):                  # clip + renormalize until it settles
            w_new = np.clip(w, lo, hi)
            w_new = w_new / w_new.sum()
            if np.allclose(w_new, w, atol=1e-9):
                w = w_new
                break
            w = w_new

    ess = float(1.0 / (w ** 2).sum())        # 1 = one file owns everything, n = uniform
    return w, ess


def build_file_weights(train_inputs, file_index, file_names, args):
    """Signature + weight every train file of one dataset. Returns a JSON-able dict.

    The per-file signal is reconstructed from the windows already in memory — the train
    stride equals the prediction length, so consecutive windows' `future` blocks tile the
    series contiguously and concatenating them replays it (minus the leading context).
    No CSV is re-read. Long files are subsampled to `--file_feat_max_windows` evenly
    spaced windows, which is plenty for a 22-number descriptor."""
    H = args.prediction_length
    feats, per_file = [], []
    for f, name in enumerate(file_names):
        idx = np.flatnonzero(file_index == f)
        if idx.size == 0:                    # file yielded no window (too short)
            feats.append(np.zeros(22))
            per_file.append({"file": name, "windows": 0, "anom_windows": 0})
            continue
        take = idx
        if idx.size > args.file_feat_max_windows:
            take = idx[np.linspace(0, idx.size - 1, args.file_feat_max_windows).astype(int)]
        sig = np.concatenate([np.asarray(train_inputs[i]["target"])[:, -H:] for i in take],
                             axis=1).astype(np.float64)
        lab = np.concatenate([np.asarray(train_inputs[i]["future_labels"]) for i in take])
        feats.append(file_signature(sig, lab, n_windows=int(idx.size)))
        n_a = sum(1 for i in idx if int(np.asarray(train_inputs[i]["future_labels"]).sum()) >= 1)
        per_file.append({"file": name, "windows": int(idx.size), "anom_windows": int(n_a)})

    w, ess = dissimilarity_weights(np.stack(feats), args.file_weight_alpha,
                                   args.file_weight_cap)

    # A file with no windows cannot be drawn; hand its mass back to the others.
    live = np.array([e["windows"] > 0 for e in per_file])
    if live.any() and not live.all():
        w = np.where(live, w, 0.0)
        w = w / w.sum()
        ess = float(1.0 / (w[w > 0] ** 2).sum())

    for e, wi in zip(per_file, w):
        e["weight"] = round(float(wi), 6)
    return {
        "files": list(file_names),
        "weights": [float(x) for x in w],
        "per_file": per_file,
        "n_files": len(file_names),
        "ess": round(ess, 2),
        "alpha": args.file_weight_alpha,
        "cap": args.file_weight_cap,
        "method": "inverse kernel density over a robust 22-d per-file signature",
    }


# ─────────────────────────────────────────────────────────────────────────────
#  Pipeline
# ─────────────────────────────────────────────────────────────────────────────

def _link_or_build_test(ds, ds_out, test_files, args, min_req, src_manifest):
    """Symlink the TEST pkls from an existing prepared_total, or carve them.

    Linking is safe ONLY while the test half is a pure function of (test_files,
    context_length, prediction_length, test_stride, min_length) and the file split is
    seeded identically — then the bytes are the same, and linking guarantees rather
    than merely asserts that every arm evaluates on identical windows.

    That premise is now checkable and can fail: dropping Exathlon/data_not_used/ from
    discovery shrinks Exathlon from 33 files to 30, which re-seeds its permutation and
    moves its test half. So compare our test_files against the source manifest's and
    CARVE instead of linking whenever they disagree — silently linking a test half that
    belongs to a different file split is the one failure this whole scheme must not have."""
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
        stats_chk = (src_manifest.get("datasets", {}) or {}).get(ds, {})
        src_files = stats_chk.get("test_files")
        ours = [os.path.basename(f) for f in test_files]
        if src_files is not None and sorted(src_files) != sorted(ours):
            logger.warning(
                f"  [{ds}] test-half file split DIFFERS from {args.link_test_from} "
                f"({len(src_files)} files there vs {len(ours)} here) — NOT linking; "
                f"carving it fresh. Its forward.py numbers are not comparable with the "
                f"arms that used the old split."
            )
            logger.warning(f"           there: {sorted(src_files)}")
            logger.warning(f"           here : {sorted(ours)}")
        else:
            return _link_test(src, test_pkl, meta_pkl, src_test, src_meta, stats_chk)

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


def _link_test(src, test_pkl, meta_pkl, src_test, src_meta, stats):
    """Symlink one dataset's TEST pkls from a source prep whose file split we matched."""
    for dst, s in ((test_pkl, src_test), (meta_pkl, src_meta)):
        if os.path.islink(dst) or os.path.exists(dst):
            os.remove(dst)
        os.symlink(s, dst)
    n_win, n_series, n_anom = (stats.get("test_windows"), stats.get("test_series"),
                               stats.get("test_anomalous"))
    if n_win is None:                   # source manifest incomplete -> count for real
        with open(src_test, "rb") as f:
            ti = pickle.load(f)
        with open(src_meta, "rb") as f:
            tm = pickle.load(f)
        n_win, n_series = len(ti), len(tm)
        n_anom = int((anomaly_step_counts(ti) >= 1).sum())
    logger.info(f"  TEST  -> LINKED {n_win} windows ({n_series} series), "
                f"{n_anom} anomalous <- {src}")
    return n_win, n_series, n_anom


# ─────────────────────────────────────────────────────────────────────────────
#  Validation half — mTSBench's own *val.csv, carved hierarchically
# ─────────────────────────────────────────────────────────────────────────────

def find_val_csv(ddir: str):
    """The dataset's own validation file, or None.

    mTSBench ships exactly one *val.csv per dataset, disjoint from the *test.csv
    files that split_files() divides into our train/test halves — so nothing here
    perturbs the seeded file split shared by every arm."""
    hits = sorted(glob.glob(os.path.join(ddir, "**", "*val.csv"), recursive=True))
    if not hits:
        return None
    if len(hits) > 1:
        logger.warning(f"  {len(hits)} *val.csv found; using {os.path.basename(hits[0])}")
    return hits[0]


def hs_carve_val(n_anom: np.ndarray, budget: int, p_anom: float, rng):
    """Hierarchically pick ONE dataset's val windows. Returns sorted indices.

    Level 1 is the equal `budget` itself (the caller hands every dataset the same
    one). Level 2 splits that budget p_anom / 1 - p_anom across the anomalous and
    normal kinds. Level 3 draws inside a kind, count-weighted by n_anom exactly as
    the train sampler does — the normal kind is n_anom = 0 throughout, so its
    weights are uniform.

    Unlike the sampler this draws WITHOUT replacement (a val window must not be
    scored twice), so a kind that cannot fill its share is drained completely and
    the remainder is handed to the other kind. The equal budget therefore always
    holds; the mix bends. Both extremes occur in mTSBench: GutenTAG's val file has
    2 anomaly windows, cicids' has 86 normal ones."""
    anom_idx = np.flatnonzero(n_anom >= 1)
    norm_idx = np.flatnonzero(n_anom == 0)
    budget = int(min(budget, len(n_anom)))

    want_a = min(int(round(p_anom * budget)), len(anom_idx))
    want_n = min(budget - want_a, len(norm_idx))
    want_a = min(want_a + (budget - want_a - want_n), len(anom_idx))   # top up if normals ran out

    w = n_anom[anom_idx].astype(np.float64)                            # level 3, anomalous kind
    sel_a = rng.choice(anom_idx, size=want_a, replace=False, p=w / w.sum()) \
        if want_a else np.empty(0, dtype=np.int64)
    sel_n = rng.choice(norm_idx, size=want_n, replace=False) \
        if want_n else np.empty(0, dtype=np.int64)
    return np.sort(np.concatenate([sel_a, sel_n])).astype(np.int64)


def build_val_windows(ds, ddir, args, min_req):
    """Carve, subsample and dataset-tag this dataset's val windows.

    The 256-step normal prefix is carved PER SERIES from the val series' own normal
    zones — the same recipe this arm uses for its train and test windows.

    Returns (selected_inputs, stats) — stats is None when the dataset ships no
    *val.csv or the file is too short to yield a single window."""
    val_csv = find_val_csv(ddir)
    if val_csv is None:
        logger.warning(f"  VAL   -> no *val.csv under {ddir}; {ds} will be ABSENT from "
                       f"the val set (its level-1 group is unrepresented)")
        return [], None

    val_pairs, _, _, _ = build_pairs_for_files(
        [val_csv], args.context_length, args.prediction_length,
        args.val_stride, min_req, f"{ds}/val")
    val_inputs = pairs_to_model_inputs(val_pairs, include_meta=False)
    del val_pairs
    if not val_inputs:
        logger.warning(f"  VAL   -> {os.path.basename(val_csv)} yields no window at "
                       f"context+horizon {min_req}; {ds} ABSENT from the val set")
        return [], None

    n_anom = anomaly_step_counts(val_inputs)
    n_avail, n_anom_avail = len(val_inputs), int((n_anom >= 1).sum())
    if n_anom_avail == 0:
        logger.warning(f"  [{ds}] val file has ZERO anomaly windows — its val slice can "
                       f"only measure normal-regime loss.")

    # Seed off the dataset NAME, not its position, so the draw is stable under
    # --datasets subsetting and reordering.
    rng = np.random.default_rng([args.seed, zlib.crc32(ds.encode())])
    keep = hs_carve_val(n_anom, args.val_per_dataset, args.val_p_anom, rng)

    selected = [dict(val_inputs[i], dataset=ds) for i in keep]
    kept_anom = int((n_anom[keep] >= 1).sum())
    stats = {
        "val_file": os.path.basename(val_csv),
        "val_windows_available": n_avail,
        "val_anom_windows_available": n_anom_avail,
        "val_windows": len(keep),
        "val_anom_windows": kept_anom,
        "val_norm_windows": len(keep) - kept_anom,
        "val_anom_win_frac": round(kept_anom / max(1, len(keep)), 4),
        "val_anom_step_frac": round(
            float(n_anom[keep].sum()) / max(1, len(keep) * args.prediction_length), 4),
    }
    short = " [BUDGET SHORT]" if len(keep) < args.val_per_dataset else ""
    logger.info(
        f"  VAL   -> {len(keep)}/{n_avail} windows from {os.path.basename(val_csv)} "
        f"(anom={kept_anom}, normal={len(keep) - kept_anom}; "
        f"{100.0 * stats['val_anom_win_frac']:.1f}% anomaly-bearing vs target "
        f"{100.0 * args.val_p_anom:.1f}%, {n_anom_avail} anomaly windows existed){short}"
    )
    return selected, stats


def write_val_pool(args, val_pool, val_datasets, train_datasets, manifest) -> None:
    """Write val_model_inputs.pkl and record the pool in the manifest (in place)."""
    if not val_pool:
        logger.warning("VAL POOL: empty (no *val.csv found for any trained dataset) "
                       "-> run finetune with NO_VALIDATION=1")
        return

    val_anom = sum(1 for d in val_pool if int(np.asarray(d["future_labels"]).sum()) >= 1)

    # Interleave the datasets (seeded -> still a fixed pkl). Eval batches sequentially,
    # so a dataset-grouped pool would make every batch single-dataset and would hand
    # --debug's first-50 truncation one dataset's windows only.
    order = np.random.default_rng(args.seed).permutation(len(val_pool))
    val_pool = [val_pool[i] for i in order]

    val_path = os.path.join(args.output_dir, "val_model_inputs.pkl")
    with open(val_path, "wb") as f:
        pickle.dump(val_pool, f)

    missing = [d for d in train_datasets if d not in val_datasets]
    logger.info("=" * 78)
    logger.info("VAL POOL (hierarchical: equal budget per dataset, "
                f"p_anom={args.val_p_anom:.3f} best-effort)")
    logger.info(f"  datasets in val  : {len(val_datasets)}/{len(train_datasets)} of the "
                f"train pool" + (f"  MISSING: {missing}" if missing else ""))
    logger.info(f"  budget/dataset   : {args.val_per_dataset} windows "
                f"(stride {args.val_stride})")
    logger.info(f"  total windows    : {len(val_pool)}  "
                f"(anom={val_anom}, normal={len(val_pool) - val_anom}, "
                f"{100.0 * val_anom / max(1, len(val_pool)):.1f}% anomaly-bearing)")
    logger.info(f"  -> {val_path}  (run finetune with NO_VALIDATION=0)")

    manifest["val_pool"] = {
        "datasets": val_datasets,
        "n_datasets": len(val_datasets),
        "missing_from_train_pool": missing,
        "per_dataset_budget": args.val_per_dataset,
        "stride": args.val_stride,
        "target_p_anom": args.val_p_anom,
        "anomaly_windows": val_anom,
        "normal_windows": len(val_pool) - val_anom,
        "total_windows": len(val_pool),
        "realized_anom_win_frac": round(val_anom / max(1, len(val_pool)), 4),
        "source": "mTSBench *val.csv (disjoint from the *test.csv train/test split)",
        "sampling": "hierarchical, without replacement: equal per-dataset budget "
                    "(level 1) -> p_anom split (level 2) -> n_anom-weighted draw "
                    "within the anomalous kind (level 3)",
    }


def prepare_val_only(args) -> None:
    """Add the val set to an EXISTING prepared_total without re-carving train/test.

    Safe to run against a prepared_total a fine-tune is already reading: no train or
    test pkl is opened for writing.
    """
    per_ds_root = os.path.join(args.output_dir, "per_dataset")
    mpath = os.path.join(args.output_dir, "manifest.json")
    if not os.path.isdir(per_ds_root) or not os.path.exists(mpath):
        raise SystemExit(
            f"--val_only needs an existing prepared_total: {per_ds_root} and {mpath} "
            f"must both exist. Run a full prep first (drop --val_only)."
        )
    with open(mpath) as f:
        manifest = json.load(f)

    train_datasets = [ds for ds, e in manifest.get("datasets", {}).items()
                      if e.get("in_train_pool")]
    if args.datasets:
        train_datasets = [ds for ds in train_datasets if ds in args.datasets]
    logger.info(f"VAL-ONLY: adding a val set to {args.output_dir} for "
                f"{len(train_datasets)} trained datasets. Train/test pkls untouched.")

    min_req = max(args.min_length, args.context_length + args.prediction_length)
    val_pool, val_datasets = [], []

    for ds in train_datasets:
        logger.info("=" * 78)
        logger.info(f"{ds}")
        sel, vstats = build_val_windows(ds, os.path.join(args.data_root, ds),
                                        args, min_req)
        if vstats is not None:
            val_pool.extend(sel)
            val_datasets.append(ds)
            manifest["datasets"][ds].update(vstats)

    for k in ("val_per_dataset", "val_stride", "val_p_anom"):
        manifest.setdefault("config", {})[k] = getattr(args, k)
    write_val_pool(args, val_pool, val_datasets, train_datasets, manifest)
    with open(mpath, "w") as f:
        json.dump(manifest, f, indent=2, default=str)
    logger.info(f"Manifest updated -> {mpath}")


def prepare(args) -> None:
    if args.val_only:
        return prepare_val_only(args)

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
    val_pool: list = []                # flat, dataset-tagged, hierarchically balanced
    val_datasets: list = []

    for ds in datasets:
        ddir = os.path.join(args.data_root, ds)
        test_csvs = real_test_csvs(ddir)
        train_files, test_files, mode = split_files(test_csvs, args.test_fraction, args.seed)

        # Synthetic files join the TRAIN half and nothing else. The real *test.csv of a
        # one-file dataset therefore stays its whole, unchanged test half (so the link
        # and forward.py stay valid) while the dataset finally becomes trainable.
        syn_files = syn_test_csvs(ddir, args.syn_dir) if args.use_syn_data else []
        if syn_files:
            train_files = list(train_files) + syn_files
            mode = f"{mode}+syn"

        logger.info("=" * 78)
        logger.info(f"{ds}: {len(test_csvs)} real *test.csv"
                    + (f" + {len(syn_files)} synthetic" if syn_files else "")
                    + f"  [{mode}]  train_files={len(train_files)} "
                      f"test_files={len(test_files)}")

        ds_out = os.path.join(per_ds_root, ds)
        os.makedirs(ds_out, exist_ok=True)

        # ── TEST half (UNCAPPED, ordered + metadata) ─────────────────────────
        t_win, t_series, t_anom = _link_or_build_test(
            ds, ds_out, test_files, args, min_req, src_manifest)

        # ── TRAIN half — carve and keep EVERYTHING ───────────────────────────
        entry = {
            "mode": mode,
            "n_test_csv": len(test_csvs),
            "n_syn_files": len(syn_files),
            "syn_files": [os.path.basename(f) for f in syn_files],
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
            # build_pairs_for_files keys every pair by series_id = the file's BASENAME, so
            # two files sharing a basename would collapse into one series (and one file id).
            # Top-level + syn_data discovery makes that impossible today; assert it anyway,
            # because the file-level sampler and series_meta both silently depend on it.
            bases = [os.path.basename(f) for f in train_files]
            if len(set(bases)) != len(bases):
                dups = sorted({b for b in bases if bases.count(b) > 1})
                raise SystemExit(
                    f"{ds}: duplicate basenames among the train files ({dups}). series_id "
                    f"is the basename, so these would be merged into one series. Rename or "
                    f"remove the duplicates."
                )

            train_pairs, _, _, _ = build_pairs_for_files(
                train_files, args.context_length, args.prediction_length,
                args.stride, min_req, f"{ds}/train")
            # Which file each window came from. Capture BEFORE pairs_to_model_inputs drops
            # series_id (it only keeps it for the test half); the conversion preserves order,
            # so this indexes train_inputs 1:1.
            file_names = bases
            name_to_id = {b: i for i, b in enumerate(file_names)}
            file_index = np.asarray([name_to_id[p["series_id"]] for p in train_pairs],
                                    dtype=np.int32)
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

            # ── File identity + dissimilarity weights (the sampler's optional level) ──
            # Always emitted, even when no run uses them: they are cheap, and writing them
            # unconditionally means --file_diversity_datasets can be flipped on an existing
            # prepared_total without a re-prep.
            np.save(os.path.join(ds_out, "train_file_index.npy"), file_index)
            fw = build_file_weights(train_inputs, file_index, file_names, args)
            with open(os.path.join(ds_out, "train_files.json"), "w") as f:
                json.dump(fw, f, indent=2)

            if fw["n_files"] > 1:
                order = np.argsort(fw["weights"])[::-1]
                top = "  ".join(f"{fw['files'][i].replace('_test.csv', '')}="
                                f"{100.0 * fw['weights'][i]:.1f}%" for i in order[:3])
                bot = "  ".join(f"{fw['files'][i].replace('_test.csv', '')}="
                                f"{100.0 * fw['weights'][i]:.1f}%" for i in order[-2:])
                logger.info(
                    f"  FILES -> {fw['n_files']} files, ESS {fw['ess']:.1f} "
                    f"(uniform would be {fw['n_files']}); heaviest: {top} | lightest: {bot}"
                )

            exp_frac, nat_frac = hs_expected_anom_step_frac(
                n_anom, args.prediction_length, args.p_anom)
            mean_anom_steps = float(n_anom[n_anom >= 1].mean())

            entry.update({
                "train_n_files": fw["n_files"],
                "train_file_ess": fw["ess"],
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
            del train_inputs, n_anom, file_index

            # ── VAL half — mTSBench's own *val.csv, per-series prefix ────────
            # Only train-pool datasets get one: the val set exists to track the
            # training objective, and a test-only dataset has no level-1 group.
            if not args.no_val:
                sel, vstats = build_val_windows(ds, ddir, args, min_req)
                if vstats is not None:
                    val_pool.extend(sel)
                    val_datasets.append(ds)
                    entry.update(vstats)
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
    syn_pool = [ds for ds in train_datasets
                if manifest["datasets"][ds].get("n_syn_files", 0) > 0]
    if syn_pool:
        logger.info(f"  synthetic (TRAIN only, never test/val): {syn_pool}")
        only_syn = [ds for ds in syn_pool if manifest["datasets"][ds]["mode"].startswith("test_only")]
        if only_syn:
            logger.info(f"  of which trainable ONLY because of syn_data: {only_syn}")
    elif args.use_syn_data:
        logger.warning(f"  --use_syn_data given but no <DATASET>/{args.syn_dir}/*test.csv "
                       f"was found under {args.data_root}. Nothing synthetic was added.")
    logger.info(f"  per-window target: [{NORMAL_SIGNAL_LENGTH} normal | {args.context_length} "
                f"context | {args.prediction_length} future]; F varies per dataset")

    # ── VAL pool ─────────────────────────────────────────────────────────────
    if args.no_val:
        logger.info("VAL POOL: skipped (--no_val) -> run finetune with NO_VALIDATION=1")
    else:
        write_val_pool(args, val_pool, val_datasets, train_datasets, manifest)

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
    # Synthetic data — TRAIN ONLY, opt-in. Default OFF so the previous experiment
    # reproduces exactly.
    p.add_argument("--use_syn_data", action="store_true",
                   help="Append <DATASET>/<syn_dir>/*test.csv to the dataset's TRAIN half. "
                        "Synthetic files never enter the test half or the val set. This is "
                        "what lets the 7 one-file datasets (CalIt2, creditcard, GECCO, "
                        "Genesis, metro, PSM, swan) join the train pool: level 1 grows from "
                        "12 to 19 groups, each drawn 5.3%% of the time instead of 8.3%%. "
                        "Their real *test.csv stays their whole test half, so evaluation is "
                        "unchanged. OFF by default.")
    p.add_argument("--syn_dir", default="syn_data",
                   help="Sub-directory of each dataset holding its synthetic *test.csv.")

    # Per-file dissimilarity weights (consumed by the sampler's optional file level).
    p.add_argument("--file_weight_alpha", type=float, default=1.0,
                   help="Exponent on the inverse kernel density: w_f ∝ (1 + density_f)^-alpha. "
                        "0 = uniform over files (already a big change from level 3's count "
                        "weighting, which the longest files dominate — the natural ablation "
                        "baseline); 1 = full dissimilarity tilt. Higher = more aggressive.")
    p.add_argument("--file_weight_cap", type=float, default=5.0,
                   help="Bound any single file at cap x (and 1/cap x) the uniform weight 1/n, "
                        "so one unusual file cannot monopolise its dataset's draws. <=1 to "
                        "disable clipping.")
    p.add_argument("--file_feat_max_windows", type=int, default=200,
                   help="Windows subsampled per file when computing its signature. The "
                        "descriptor is 22 numbers; 200 windows (12.8k steps) is ample.")

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

    # Validation set — carved from mTSBench's own *val.csv (never from the *test.csv
    # files the train/test split divides), one equal budget per trained dataset.
    p.add_argument("--no_val", action="store_true",
                   help="Skip the val set entirely (then run finetune with NO_VALIDATION=1).")
    p.add_argument("--val_only", action="store_true",
                   help="Add ONLY the val set to an existing prepared_total. Train/test "
                        "pkls are never rewritten, so this is safe to run against data a "
                        "fine-tune is already reading.")
    p.add_argument("--val_per_dataset", type=int, default=200,
                   help="Level 1: windows taken from EVERY trained dataset's val file, so "
                        "each contributes equally to eval_loss. A dataset with fewer "
                        "windows than this hands over all it has.")
    p.add_argument("--val_stride", type=int, default=16,
                   help="Val sliding-window stride. Smaller than --stride on purpose: the "
                        "val files are single series and the small ones (MSL, SMD, "
                        "GutenTAG) cannot fill their budget at stride 64. Val windows may "
                        "overlap — they are only scored, never tiled like the test half.")
    p.add_argument("--val_p_anom", type=float, default=1.0 / 3.0,
                   help="Level 2: target anomalous share inside each dataset's val budget. "
                        "Matches the sampler's --p_anom so eval_loss is measured on the "
                        "mix the model is trained on. Best-effort: datasets short of "
                        "anomaly windows fill the rest with normal ones.")
    args = p.parse_args()

    if not 0.0 <= args.val_p_anom <= 1.0:
        raise SystemExit(f"--val_p_anom must be in [0, 1], got {args.val_p_anom}")

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
