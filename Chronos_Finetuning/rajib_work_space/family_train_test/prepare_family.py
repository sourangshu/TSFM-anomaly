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

Output (a shared pool; folds are assembled from it by make_folds.py as symlinks, so
no dataset is ever carved twice):

    pool/per_dataset/<DS>/train_model_inputs.pkl   (TRAIN-role datasets)
    pool/per_dataset/<DS>/train_n_anom.npy
    pool/per_dataset/<DS>/test_model_inputs.pkl    (TEST-role datasets)
    pool/per_dataset/<DS>/test_series_meta.pkl
    pool/manifest.json

Usage:
    python prepare_family.py                       # every dataset named in families.json
    python prepare_family.py --folds ecg server    # just those folds' datasets
"""

import argparse
import glob
import json
import logging
import os
import pickle
import sys

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
    args = p.parse_args()

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

    min_req = max(args.min_length, args.context_length + args.prediction_length)
    manifest = {"config": vars(args), "families": fams,
                "train_pool": train_pool, "heldout": heldout, "datasets": {}}

    for ds in sorted(roles):
        manifest["datasets"][ds] = build_dataset(
            ds, roles[ds], args, min_req, per_ds_root)

    with open(os.path.join(args.output_dir, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)

    logger.info("=" * 78)
    logger.info(f"Pool written to {args.output_dir}")
    logger.info("Next: python make_folds.py   (assembles run/ as symlinks into this pool)")


if __name__ == "__main__":
    main()
