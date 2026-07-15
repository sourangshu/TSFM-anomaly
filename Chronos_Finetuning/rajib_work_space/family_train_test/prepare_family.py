#!/usr/bin/env python
"""
Data prep for the FAMILY transfer study.

Protocol (differs from TOTAL_RUN / TOTAL_RUN_maskloss_v2* on exactly one axis):

    There is NO within-dataset file split. A dataset is used WHOLE, in one of two
    roles, decided per fold by families.json:

        TRAIN role -> all of its *test.csv files become train windows
        TEST  role -> all of its *test.csv files become test windows

    Because a dataset never holds both roles inside one fold, "train on family
    siblings, test on the held-out sibling" is leak-free by construction: the
    evaluated dataset contributes nothing to training, not one window.

    This is why the test halves CANNOT be symlinked from TOTAL_RUN_maskloss_v2 the
    way run_prepare_total.sh does -- those are 50% halves, and here we evaluate on
    100% of the held-out dataset's files. They must be carved fresh.

Only *test.csv files are ever read, in either role: in mTSBench the *train.csv files
carry no anomaly labels, and the per-step masked margin loss needs labels on BOTH
sides of the mask.

VALIDATION comes from mTSBench's own *val.csv — a file this study otherwise never
reads (it reads *test.csv only) — and is carved ONLY for TRAIN-role datasets. A
held-out dataset must contribute nothing to the training loop, and eval_loss steers
training through best-model selection, so a val window from SVDB would leak SVDB into
the very thing the study measures. Held-out datasets therefore get no val artifact,
exactly as they get no train artifact.

Within the train pool the carve is HIERARCHICAL, mirroring the sampler:

    level 1   an EQUAL budget (--val_per_dataset) for every TRAIN-role dataset
    level 2   an anomalous share of --val_p_anom (=1/3) inside each
    level 3   within the anomalous kind, count-weighted by n_anom

drawn WITHOUT replacement under a per-dataset seed, so the val set is deterministic.
Level 2 is best-effort: a dataset short of anomaly windows hands over every one it has
and fills the rest with normal windows, so the EQUAL per-dataset count always holds.

Output (a shared pool; folds are assembled from it by make_folds.py as symlinks, so
no dataset is ever carved twice):

    pool/per_dataset/<DS>/train_model_inputs.pkl   (TRAIN-role datasets)
    pool/per_dataset/<DS>/train_n_anom.npy
    pool/per_dataset/<DS>/train_file_index.npy     (level 1.5: file each window is from)
    pool/per_dataset/<DS>/train_files.json         (level 1.5: per-file dissimilarity weights)
    pool/per_dataset/<DS>/val_model_inputs.pkl     (TRAIN-role datasets ONLY)
    pool/per_dataset/<DS>/test_model_inputs.pkl    (TEST-role datasets)
    pool/per_dataset/<DS>/test_series_meta.pkl
    pool/manifest.json

The two train_file* sidecars feed the sampler's OPTIONAL level 1.5 (the dissimilarity-
weighted file draw): they are always written, and run_finetune_family.sh's
FILE_DIVERSITY_DATASETS decides which datasets actually use them. See build_file_weights.

make_folds.py concatenates the per-dataset val pkls of a fold's train datasets into
that fold's prepared/val_model_inputs.pkl — so the unified run validates on its 9
pool datasets, and a --per_family run validates on that family's siblings only.

Usage:
    python prepare_family.py                       # every dataset named in families.json
    python prepare_family.py --only ecg server     # just those families' datasets
    python prepare_family.py --val_only            # add val to an existing pool, leaving
                                                   # train/test pkls untouched
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

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
from prepare_smd_split import (                      # noqa: E402
    NORMAL_SIGNAL_LENGTH,
    build_pairs_for_files,
    pairs_to_model_inputs,
)

logger = logging.getLogger("prepare_family")


def anomaly_step_counts(inputs) -> np.ndarray:
    """Per-window anomaly-step count n_anom in [0, prediction_length]."""
    return np.asarray(
        [int(np.asarray(d["future_labels"]).sum()) for d in inputs], dtype=np.int16
    )


def hs_expected_anom_step_frac(n_anom: np.ndarray, H: int, p_anom: float):
    """Expected anomaly-STEP fraction the HS sampler delivers for ONE dataset.

    Verbatim from prepare_total.py so the manifest's reporting is comparable across
    arms. Reporting only -- the sampler reads its own --p_anom."""
    n = n_anom.astype(np.float64)
    n_norm = H - n
    natural = float(n.mean() / H) if len(n) else float("nan")
    s_a, s_n = n.sum(), n_norm.sum()
    if s_a <= 0 or s_n <= 0:
        return float("nan"), natural
    e_anom = float((n * n).sum() / s_a)
    e_norm = float((n_norm * n).sum() / s_n)
    return float((p_anom * e_anom + (1.0 - p_anom) * e_norm) / H), natural


# ─────────────────────────────────────────────────────────────────────────────
#  Per-file dissimilarity weights (the sampler's OPTIONAL level 1.5)
# ─────────────────────────────────────────────────────────────────────────────
#
# Ported VERBATIM from TOTAL_RUN_maskloss_v2_HS/prepare_total.py so the file each
# window came from, and its per-file dissimilarity weight, are written in exactly the
# format the (symlinked) finetune_anomaly_simple.py already reads for its level 1.5.
#
# Goal: inside one train dataset, stop near-duplicate *test.csv files from dominating
# and give unusual ones a real chance of being drawn, so training sees the dataset's
# whole repertoire of patterns rather than its longest recording repeated. Level 3's
# count weighting is blind to which file a window came from; drawing the FILE first
# (level 1.5) spends a dataset's budget across its distinct PATTERNS instead of its
# raw minutes of recording. A no-op on single-file datasets (CalIt2, GECCO here).
#
#   1. signature   a fixed-length, channel-count-agnostic descriptor per file
#   2. standardize robustly across the dataset's files (median / IQR)
#   3. density     s_f = sum over the OTHER files of a Gaussian kernel on the
#                  pairwise distance, bandwidth = median pairwise distance
#   4. weight      w_f  ∝ (1 + s_f)^(-alpha)

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

    `sig` is (F, T) -- the file's signal, `labels` its (T,) per-step 0/1 ground truth.
    Per-channel descriptors are pooled by mean AND std across channels, then the file's
    anomaly profile is appended: anomalies are a pattern to be sampled too, so two files
    with identical dynamics but different anomaly regimes must not collapse to one point."""
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

    # Robust standardization: median/IQR, not mean/std. Constant features carry no
    # information about which files differ and are dropped.
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

    The per-file signal is reconstructed from the windows already in memory -- the train
    stride equals the prediction length, so consecutive windows' `future` blocks tile the
    series contiguously and concatenating them replays it. No CSV is re-read. Long files
    are subsampled to `--file_feat_max_windows` evenly spaced windows.

    The returned dict's 'files' / 'weights' are exactly what the symlinked
    finetune_anomaly_simple.py's load_train_pool() reads for level 1.5."""
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


def find_val_csv(ddir: str):
    """The dataset's own validation file, or None.

    mTSBench ships exactly one *val.csv per dataset. This study reads *test.csv only,
    so the val file is untouched by either role — no window in it can already be a
    train or a test window."""
    hits = sorted(glob.glob(os.path.join(ddir, "**", "*val.csv"), recursive=True))
    if not hits:
        return None
    if len(hits) > 1:
        logger.warning(f"  {len(hits)} *val.csv found; using {os.path.basename(hits[0])}")
    return hits[0]


def hs_carve_val(n_anom: np.ndarray, budget: int, p_anom: float, rng):
    """Hierarchically pick ONE dataset's val windows. Returns sorted indices.

    Level 1 is the equal `budget` itself (every train-role dataset gets the same one).
    Level 2 splits it p_anom / 1 - p_anom across the anomalous and normal kinds. Level 3
    draws inside a kind, count-weighted by n_anom exactly as the train sampler does — the
    normal kind is n_anom = 0 throughout, so its weights are uniform.

    Unlike the sampler this draws WITHOUT replacement (a val window must not be scored
    twice), so a kind that cannot fill its share is drained completely and the remainder
    goes to the other kind: the equal budget always holds, the mix bends."""
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


def build_val(ds, ddir, ds_out, args, min_req):
    """Carve this TRAIN-role dataset's val windows into pool/per_dataset/<DS>/.

    Never called for a held-out dataset: eval_loss selects the checkpoint, so a val
    window from a held-out dataset would leak it into the result the study reports.
    Returns the manifest stats, or None when the dataset ships no usable *val.csv."""
    val_csv = find_val_csv(ddir)
    if val_csv is None:
        logger.warning(f"  VAL   -> no *val.csv under {ddir}; {ds} will be ABSENT from "
                       f"the val set (its level-1 group is unrepresented)")
        return None

    pairs, _, _, _ = build_pairs_for_files(
        [val_csv], args.context_length, args.prediction_length,
        args.val_stride, min_req, f"{ds}/val")
    inputs = pairs_to_model_inputs(pairs, include_meta=False)
    del pairs
    if not inputs:
        logger.warning(f"  VAL   -> {os.path.basename(val_csv)} yields no window at "
                       f"context+horizon {min_req}; {ds} ABSENT from the val set")
        return None

    n_anom = anomaly_step_counts(inputs)
    n_avail, n_anom_avail = len(inputs), int((n_anom >= 1).sum())
    if n_anom_avail == 0:
        logger.warning(f"  !! {ds} val file has ZERO anomaly windows — its val slice can "
                       f"only measure normal-regime loss.")

    # Seed off the dataset NAME, not its position, so the draw is stable under --only
    # and identical whichever fold assembles it.
    rng = np.random.default_rng([args.seed, zlib.crc32(ds.encode())])
    keep = hs_carve_val(n_anom, args.val_per_dataset, args.val_p_anom, rng)

    selected = [dict(inputs[i], dataset=ds) for i in keep]
    with open(os.path.join(ds_out, "val_model_inputs.pkl"), "wb") as f:
        pickle.dump(selected, f)

    kept_anom = int((n_anom[keep] >= 1).sum())
    short = " [BUDGET SHORT]" if len(keep) < args.val_per_dataset else ""
    logger.info(
        f"  VAL   -> {len(keep)}/{n_avail} windows from {os.path.basename(val_csv)} "
        f"({kept_anom} anomalous / {len(keep) - kept_anom} normal; "
        f"{kept_anom / max(1, len(keep)):.1%} anomaly-bearing vs target "
        f"{args.val_p_anom:.1%}, {n_anom_avail} anomaly windows existed){short}"
    )
    return {
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


def holdout_list(fam):
    """A family's holdout as a list. Accepts the historical string form too."""
    h = fam["holdout"]
    return list(h) if isinstance(h, (list, tuple)) else [h]


def load_families(path: str, only):
    with open(path) as f:
        spec = json.load(f)
    fams = spec["families"]
    if only:
        missing = [k for k in only if k not in fams]
        if missing:
            raise SystemExit(f"Unknown family/families {missing}. Known: {sorted(fams)}")
        fams = {k: fams[k] for k in only}

    for name, f in fams.items():
        clash = sorted(set(holdout_list(f)) & set(f["train"]))
        if clash:
            raise SystemExit(
                f"Family '{name}' lists {clash} as both a train dataset and a holdout. "
                f"That leaks: the evaluated dataset would contribute training windows."
            )
    return fams


def dataset_roles(fams):
    """dataset -> role. The shared TRAIN pool is the union of every family's train list;
    the TEST set is every family's holdout.

    A dataset must not appear in both, globally: there is only ONE model here, trained on
    the whole pool, so a dataset that trains for family A and is held out for family B
    would be evaluated by a model that saw it. That is the one thing that would silently
    invalidate the study, so it is a hard error, not a warning."""
    roles: dict[str, set] = {}
    for f in fams.values():
        for ds in f["train"]:
            roles.setdefault(ds, set()).add("train")
        for ds in holdout_list(f):
            roles.setdefault(ds, set()).add("test")

    both = sorted(ds for ds, r in roles.items() if len(r) > 1)
    if both:
        raise SystemExit(
            f"Dataset(s) {both} are BOTH in the shared train pool and held out. With a "
            f"single pooled model that is leakage: the model would be evaluated on data "
            f"it trained on. Fix families.json."
        )
    return roles


def real_test_csvs(ddir: str):
    """The dataset's REAL *test.csv files -- TOP LEVEL ONLY, never recursive.

    Non-recursive on purpose (matches TOTAL_RUN_maskloss_v2_HS/prepare_total.py). A
    recursive glob would sweep in <DATASET>/syn_data/*test.csv as if it were real data,
    which for a HELD-OUT dataset (metro, Genesis, PSM here) would put synthetic windows
    into the TEST set -- evaluation must stay 100% real. Synthetic files are discovered
    separately by syn_test_csvs() and appended to the TRAIN half only."""
    return sorted(glob.glob(os.path.join(ddir, "*test.csv")))


def syn_test_csvs(ddir: str, syn_dir: str):
    """The dataset's SYNTHETIC *test.csv files (top level of <DATASET>/<syn_dir>/), or [].

    These are appended to the TRAIN half and go nowhere else: never a held-out dataset's
    test set (evaluation stays real), never the val set (that is the dataset's *val.csv).
    In this study they matter for the single-file train datasets (CalIt2, GECCO): they
    turn a 1-file dataset into a multi-file one, which both grows its thin train pool and
    activates level 1.5 for it (one real recording + several synthetic patterns)."""
    return sorted(glob.glob(os.path.join(ddir, syn_dir, "*test.csv")))


def build_dataset(ds, roles, args, min_req, per_ds_root):
    ddir = os.path.join(args.data_root, ds)
    if not os.path.isdir(ddir):
        raise SystemExit(f"{ds}: no such directory under {args.data_root}")

    # REAL *test.csv only, top level (the *train.csv files carry no anomaly labels, and a
    # recursive glob would wrongly sweep syn_data/ into the real set -- see real_test_csvs).
    files = real_test_csvs(ddir)
    if not files:
        raise SystemExit(f"{ds}: no *test.csv found at the top level of {ddir}")

    # Synthetic files join the TRAIN half only. A held-out (test-role) dataset never gets
    # them: its role builds only a test set, which must stay 100% real.
    syn_files = (syn_test_csvs(ddir, args.syn_dir)
                 if (args.use_syn_data and "train" in roles) else [])

    ds_out = os.path.join(per_ds_root, ds)
    os.makedirs(ds_out, exist_ok=True)

    logger.info("=" * 78)
    logger.info(f"{ds}: {len(files)} real *test.csv"
                + (f" + {len(syn_files)} synthetic (TRAIN only)" if syn_files else "")
                + f", 100% used  [roles: {sorted(roles)}]")

    entry = {
        "roles": sorted(roles),
        "n_test_csv": len(files),
        "files": [os.path.basename(f) for f in files],
        "n_syn_files": len(syn_files),
        "syn_files": [os.path.basename(f) for f in syn_files],
    }

    # ── VAL (TRAIN-role datasets only) ───────────────────────────────────────
    # Held-out datasets are skipped deliberately: eval_loss drives best-model
    # selection, so validating on a held-out dataset would leak it into training.
    if "train" in roles and not args.no_val:
        vstats = build_val(ds, ddir, ds_out, args, min_req)
        if vstats:
            entry.update(vstats)
    if args.val_only:
        return entry

    if "train" in roles:
        # Real files + (optional) synthetic files, in that order.
        train_files = list(files) + list(syn_files)

        # build_pairs_for_files keys every pair by series_id = the file's BASENAME, so two
        # files sharing a basename would collapse into one series (and one file id). The
        # file-level sampler (level 1.5) indexes by that basename, so assert uniqueness.
        bases = [os.path.basename(f) for f in train_files]
        if len(set(bases)) != len(bases):
            dups = sorted({b for b in bases if bases.count(b) > 1})
            raise SystemExit(
                f"{ds}: duplicate basenames among the train files ({dups}). series_id is "
                f"the basename, so these would be merged into one series. Rename or remove.")

        pairs, _, _, _ = build_pairs_for_files(
            train_files, args.context_length, args.prediction_length,
            args.stride, min_req, f"{ds}/train")
        # Which file each window came from. Capture BEFORE pairs_to_model_inputs drops
        # series_id (it keeps it only for the test half); the conversion preserves order,
        # so this indexes `inputs` 1:1.
        name_to_id = {b: i for i, b in enumerate(bases)}
        file_index = np.asarray([name_to_id[p["series_id"]] for p in pairs], dtype=np.int32)
        inputs = pairs_to_model_inputs(pairs, include_meta=False)
        del pairs

        n_anom = anomaly_step_counts(inputs)
        n_anom_win = int((n_anom >= 1).sum())

        # The HS sampler's level-3 anomalous branch draws with weights proportional to
        # n_anom. An all-zero vector makes that branch undrawable for this dataset.
        if n_anom_win == 0:
            raise SystemExit(
                f"{ds}: ZERO anomaly windows in the train pool. The hierarchical "
                f"sampler's anomalous branch has an all-zero weight vector for it. "
                f"Drop it from the fold's train list in families.json."
            )
        if n_anom_win < args.min_anom_windows:
            logger.warning(
                f"  !! {ds} has only {n_anom_win} anomaly windows (<{args.min_anom_windows}). "
                f"The sampler will revisit them heavily -- overfitting risk.")

        exp_frac, nat_frac = hs_expected_anom_step_frac(
            n_anom, args.prediction_length, args.p_anom)

        with open(os.path.join(ds_out, "train_model_inputs.pkl"), "wb") as f:
            pickle.dump(inputs, f)
        np.save(os.path.join(ds_out, "train_n_anom.npy"), n_anom)

        # ── Level 1.5 sidecars — file identity + per-file dissimilarity weights ──
        # Always emitted (cheap), so FILE_DIVERSITY_DATASETS can be flipped at fine-tune
        # time without a re-prep. The symlinked finetune_anomaly_simple.py reads exactly
        # train_file_index.npy (which file each window is from) and train_files.json
        # ('files' + 'weights'). A no-op for single-file datasets: one file, weight 1.
        np.save(os.path.join(ds_out, "train_file_index.npy"), file_index)
        fw = build_file_weights(inputs, file_index, bases, args)
        with open(os.path.join(ds_out, "train_files.json"), "w") as f:
            json.dump(fw, f, indent=2)
        if fw["n_files"] > 1:
            order = np.argsort(fw["weights"])[::-1]
            top = "  ".join(f"{fw['files'][i].replace('_test.csv', '')}="
                            f"{100.0 * fw['weights'][i]:.1f}%" for i in order[:3])
            bot = "  ".join(f"{fw['files'][i].replace('_test.csv', '')}="
                            f"{100.0 * fw['weights'][i]:.1f}%" for i in order[-2:])
            logger.info(f"  FILES -> {fw['n_files']} files, ESS {fw['ess']:.1f} "
                        f"(uniform would be {fw['n_files']}); heaviest: {top} | "
                        f"lightest: {bot}")

        logger.info(f"  TRAIN -> {len(inputs)} windows "
                    f"({n_anom_win} anomalous / {len(inputs) - n_anom_win} normal), "
                    f"anomaly steps: natural {nat_frac:.1%} -> under HS {exp_frac:.1%}")
        entry.update(train_windows=len(inputs), train_anom_windows=n_anom_win,
                     train_norm_windows=len(inputs) - n_anom_win,
                     train_n_files=fw["n_files"], train_file_ess=fw["ess"],
                     natural_anom_step_frac=round(nat_frac, 4),
                     hs_expected_anom_step_frac=None if np.isnan(exp_frac)
                     else round(exp_frac, 4))
        del inputs

    if "test" in roles:
        # test_stride MUST equal prediction_length so windows tile the series
        # contiguously -- forward.py scatters the 64 per-step predictions back onto
        # the original timeline using the meta, and gaps/overlaps would corrupt it.
        pairs, _, _, meta = build_pairs_for_files(
            files, args.context_length, args.prediction_length,
            args.test_stride, min_req, f"{ds}/test")
        inputs = pairs_to_model_inputs(pairs, include_meta=True)
        del pairs

        with open(os.path.join(ds_out, "test_model_inputs.pkl"), "wb") as f:
            pickle.dump(inputs, f)
        with open(os.path.join(ds_out, "test_series_meta.pkl"), "wb") as f:
            pickle.dump(meta, f)

        n_anom_win = int((anomaly_step_counts(inputs) >= 1).sum())
        total_anom_steps = int(sum(int(np.asarray(m["labels"]).sum())
                                   for m in meta.values()))
        logger.info(f"  TEST  -> {len(inputs)} windows ({len(meta)} series), "
                    f"{n_anom_win} anomalous windows, "
                    f"{total_anom_steps} anomalous timesteps  [100% of files]")
        if total_anom_steps < args.min_anom_steps_test:
            logger.warning(
                f"  !! {ds} has only {total_anom_steps} anomalous timesteps in the test "
                f"set. VUS-PR will be noise-dominated -- treat this fold as underpowered.")
        entry.update(test_windows=len(inputs), test_series=len(meta),
                     test_anom_windows=n_anom_win,
                     test_anom_steps=total_anom_steps)
        del inputs, meta

    return entry


def main():
    p = argparse.ArgumentParser(
        description="Family-transfer data prep: whole datasets, no within-dataset split.")
    p.add_argument("--data_root", default="/home/rajib/mTSBench/Datasets/mTSBench")
    p.add_argument("--output_dir", default=os.path.join(_HERE, "pool"))
    p.add_argument("--families", default=os.path.join(_HERE, "families.json"))
    p.add_argument("--only", nargs="*", default=None,
                   help="Subset of family names from families.json (default: all)")

    # Window geometry -- MUST match run_finetune_family.sh / run_forward_family.sh
    p.add_argument("--context_length", type=int, default=512)
    p.add_argument("--prediction_length", type=int, default=64)
    p.add_argument("--stride", type=int, default=64, help="Train sliding-window stride")
    p.add_argument("--test_stride", type=int, default=64,
                   help="MUST equal prediction_length (contiguous test tiling)")
    p.add_argument("--min_length", type=int, default=50)

    p.add_argument("--p_anom", type=float, default=1.0 / 3.0,
                   help="Reporting only: the anomaly-step fraction each train pool will "
                        "yield under the HS sampler. The sampler reads its own --p_anom.")
    p.add_argument("--min_anom_windows", type=int, default=50,
                   help="Warn when a TRAIN pool has fewer anomaly windows than this.")
    p.add_argument("--min_anom_steps_test", type=int, default=500,
                   help="Warn when a TEST set has fewer anomalous timesteps than this "
                        "(VUS-PR becomes noise-dominated).")

    # Synthetic data — TRAIN-role datasets ONLY. Appends <DATASET>/<syn_dir>/*test.csv to
    # a train dataset's windows; never touches a held-out dataset's (real) test set or the
    # val set. Here it applies to CalIt2 and GECCO (the two single-file train datasets).
    p.add_argument("--use_syn_data", action="store_true",
                   help="Append <DATASET>/<syn_dir>/*test.csv to TRAIN-role datasets. "
                        "Synthetic files never enter a held-out test set or the val set.")
    p.add_argument("--syn_dir", default="syn_data",
                   help="Sub-directory of each dataset holding its synthetic *test.csv.")

    # Level 1.5 — per-file dissimilarity weights (consumed by the sampler's file level).
    # Always written; --file_diversity_datasets at fine-tune time decides which are used.
    p.add_argument("--file_weight_alpha", type=float, default=1.0,
                   help="Exponent on the inverse kernel density: w_f ∝ (1 + density_f)^-alpha. "
                        "0 = uniform over files (the ablation baseline); 1 = full "
                        "dissimilarity tilt. Higher = more aggressive.")
    p.add_argument("--file_weight_cap", type=float, default=5.0,
                   help="Bound any single train file at cap x (and 1/cap x) the uniform "
                        "weight 1/n, so one unusual file cannot monopolise its dataset's "
                        "draws. <=1 disables clipping.")
    p.add_argument("--file_feat_max_windows", type=int, default=200,
                   help="Windows subsampled per file when computing its 22-number "
                        "signature. 200 (12.8k steps) is ample.")

    # Validation — mTSBench's own *val.csv, TRAIN-role datasets only.
    p.add_argument("--no_val", action="store_true",
                   help="Skip the val set (then run the fine-tune with NO_VALIDATION=1).")
    p.add_argument("--val_only", action="store_true",
                   help="Add ONLY the val pkls to an existing pool. Train/test pkls are "
                        "never rewritten, so this is safe to run against a pool a "
                        "fine-tune is already reading.")
    p.add_argument("--val_per_dataset", type=int, default=200,
                   help="Level 1: windows taken from EVERY train-role dataset's val file, "
                        "so each contributes equally to eval_loss. A dataset with fewer "
                        "windows than this hands over all it has.")
    p.add_argument("--val_stride", type=int, default=16,
                   help="Val sliding-window stride. Smaller than --stride on purpose: the "
                        "val files are single series and the small ones (SMD, SMAP, "
                        "room-occupancy) cannot fill their budget at stride 64. Val "
                        "windows may overlap — they are scored, never tiled like test.")
    p.add_argument("--val_p_anom", type=float, default=1.0 / 3.0,
                   help="Level 2: target anomalous share inside each dataset's val budget. "
                        "Matches the sampler's --p_anom so eval_loss is measured on the "
                        "mix the model trains on. Best-effort where anomalies are scarce.")
    p.add_argument("--seed", type=int, default=42,
                   help="Seeds the val draw (with the dataset name). Nothing else in this "
                        "study is random: there is no file split.")
    args = p.parse_args()

    if not 0.0 <= args.val_p_anom <= 1.0:
        raise SystemExit(f"--val_p_anom must be in [0, 1], got {args.val_p_anom}")
    if args.val_only and args.no_val:
        raise SystemExit("--val_only and --no_val are contradictory.")

    if args.test_stride != args.prediction_length:
        raise SystemExit(
            f"--test_stride ({args.test_stride}) must equal --prediction_length "
            f"({args.prediction_length}) so test windows tile contiguously.")

    os.makedirs(args.output_dir, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.StreamHandler(),
                  logging.FileHandler(os.path.join(args.output_dir, "prepare_family.log"))],
    )

    fams = load_families(args.families, args.only)
    roles = dataset_roles(fams)
    per_ds_root = os.path.join(args.output_dir, "per_dataset")
    os.makedirs(per_ds_root, exist_ok=True)

    train_pool = sorted(ds for ds, r in roles.items() if "train" in r)
    heldout = sorted(ds for ds, r in roles.items() if "test" in r)

    logger.info(f"Config: {vars(args)}")
    logger.info(f"Families ({len(fams)}): {sorted(fams)}")
    logger.info(f"Shared TRAIN pool ({len(train_pool)}): {train_pool}")
    logger.info(f"HELD OUT ({len(heldout)}): {heldout}")
    logger.info(f"  -> ONE model on the pool; HS level 1 = uniform over "
                f"{len(train_pool)} datasets ({100.0 / len(train_pool):.1f}% of draws each)")
    logger.info(f"Normal-signal prefix: {NORMAL_SIGNAL_LENGTH}  "
                f"(model context = {NORMAL_SIGNAL_LENGTH + args.context_length})")
    if args.use_syn_data:
        logger.info(f"SYN: <DATASET>/{args.syn_dir}/*test.csv appended to TRAIN-role "
                    f"datasets ONLY (never a held-out test set, never val). Applies where "
                    f"a train dataset ships syn_data -- here CalIt2, GECCO.")
    else:
        logger.info("SYN: off (pass --use_syn_data to fold syn_data/ into train datasets).")
    if args.no_val:
        logger.info("VAL: skipped (--no_val) -> fine-tune with NO_VALIDATION=1")
    else:
        logger.info(f"VAL: mTSBench *val.csv, hierarchical — {args.val_per_dataset} windows "
                    f"per TRAIN dataset (stride {args.val_stride}, target anomalous share "
                    f"{args.val_p_anom:.1%}). Held-out datasets get NO val: eval_loss picks "
                    f"the checkpoint, so validating on one would leak it.")
    if args.val_only:
        logger.info("VAL-ONLY: train/test pkls will NOT be rewritten.")

    min_req = max(args.min_length, args.context_length + args.prediction_length)
    mpath = os.path.join(args.output_dir, "manifest.json")

    # --val_only patches the existing manifest: build_dataset returns val stats only, and
    # clobbering the file would drop the train/test counts the earlier full prep recorded.
    manifest = {"config": vars(args), "families": fams,
                "train_pool": train_pool, "heldout": heldout, "datasets": {}}
    if args.val_only and os.path.exists(mpath):
        with open(mpath) as f:
            manifest = json.load(f)
        manifest.setdefault("datasets", {})
        for k in ("val_per_dataset", "val_stride", "val_p_anom", "seed"):
            manifest.setdefault("config", {})[k] = getattr(args, k)

    for ds in sorted(roles):
        entry = build_dataset(ds, roles[ds], args, min_req, per_ds_root)
        if args.val_only:
            manifest["datasets"].setdefault(ds, {}).update(entry)
        else:
            manifest["datasets"][ds] = entry

    with open(mpath, "w") as f:
        json.dump(manifest, f, indent=2)

    logger.info("=" * 78)
    logger.info(f"Pool written to {args.output_dir}")
    logger.info("Next: python make_folds.py   (assembles run/ as symlinks into this pool, "
                "and concatenates the train datasets' val pkls into the fold's val set)")


if __name__ == "__main__":
    main()
