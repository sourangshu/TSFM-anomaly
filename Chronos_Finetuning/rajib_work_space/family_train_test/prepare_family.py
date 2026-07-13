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
    pool/per_dataset/<DS>/val_model_inputs.pkl     (TRAIN-role datasets ONLY)
    pool/per_dataset/<DS>/test_model_inputs.pkl    (TEST-role datasets)
    pool/per_dataset/<DS>/test_series_meta.pkl
    pool/manifest.json

make_folds.py concatenates the per-dataset val pkls of a fold's train datasets into
that fold's prepared/val_model_inputs.pkl — so the unified run validates on its 9
pool datasets, and a --per_family run validates on that family's siblings only.

Usage:
    python prepare_family.py                       # every dataset named in families.json
    python prepare_family.py --folds ecg server    # just those folds' datasets
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
        if f["holdout"] in f["train"]:
            raise SystemExit(
                f"Family '{name}' lists {f['holdout']} as both a train dataset and the "
                f"holdout. That leaks: the evaluated dataset would contribute training "
                f"windows."
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
        roles.setdefault(f["holdout"], set()).add("test")

    both = sorted(ds for ds, r in roles.items() if len(r) > 1)
    if both:
        raise SystemExit(
            f"Dataset(s) {both} are BOTH in the shared train pool and held out. With a "
            f"single pooled model that is leakage: the model would be evaluated on data "
            f"it trained on. Fix families.json."
        )
    return roles


def build_dataset(ds, roles, args, min_req, per_ds_root):
    ddir = os.path.join(args.data_root, ds)
    if not os.path.isdir(ddir):
        raise SystemExit(f"{ds}: no such directory under {args.data_root}")

    # Only *test.csv -- the *train.csv files carry no anomaly labels.
    files = sorted(glob.glob(os.path.join(ddir, "**", "*test.csv"), recursive=True))
    if not files:
        raise SystemExit(f"{ds}: no *test.csv found under {ddir}")

    ds_out = os.path.join(per_ds_root, ds)
    os.makedirs(ds_out, exist_ok=True)

    logger.info("=" * 78)
    logger.info(f"{ds}: {len(files)} *test.csv, 100% used  [roles: {sorted(roles)}]")

    entry = {
        "roles": sorted(roles),
        "n_test_csv": len(files),
        "files": [os.path.basename(f) for f in files],
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
        pairs, _, _, _ = build_pairs_for_files(
            files, args.context_length, args.prediction_length,
            args.stride, min_req, f"{ds}/train")
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

        logger.info(f"  TRAIN -> {len(inputs)} windows "
                    f"({n_anom_win} anomalous / {len(inputs) - n_anom_win} normal), "
                    f"anomaly steps: natural {nat_frac:.1%} -> under HS {exp_frac:.1%}")
        entry.update(train_windows=len(inputs), train_anom_windows=n_anom_win,
                     train_norm_windows=len(inputs) - n_anom_win,
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
