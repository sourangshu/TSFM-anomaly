#!/usr/bin/env python
"""
Assemble the study's run directory out of the shared pool, using symlinks only.

ONE model is trained on the pooled training datasets of every family, then evaluated on
each family's held-out dataset. So there is ONE directory, and it holds both sides:

    run/prepared/per_dataset/<TRAIN_DS>/train_model_inputs.pkl -> pool/...   (x9)
                                        train_n_anom.npy       -> pool/...
                            /<HELDOUT_DS>/test_model_inputs.pkl -> pool/...  (x6)
                                          test_series_meta.pkl  -> pool/...

That single directory drives BOTH stages with no flags, because of how the two
entrypoints discover data:

    finetune_anomaly_simple.py : load_train_pool() walks per_dataset/*/ and SKIPS any
                                 folder without train_model_inputs.pkl
                                 -> it sees exactly the 9 TRAIN datasets, and the
                                    hierarchical sampler's level 1 becomes uniform over
                                    those 9 (identical to TOTAL_RUN_maskloss_v2_HS, which
                                    was uniform over its 12).
    forward.py (run_forward)   : walks per_dataset/*/ and SKIPS any folder without
                                 test_model_inputs.pkl
                                 -> it sees exactly the 6 HELD-OUT datasets.

Neither stage can touch the other's data, because no dataset carries both artifacts.

--per_family additionally builds one directory per family (train = that family's siblings
only, test = its holdout). Those are for the ATTRIBUTION study -- 6 separate models that
can say *which* sibling did the work -- and are not needed for the single pooled run.

Usage:
    python make_folds.py                # the unified run (default)
    python make_folds.py --per_family   # + per-family dirs for the attribution variant
"""

import argparse
import json
import os
import pickle

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))

TRAIN_ARTIFACTS = ("train_model_inputs.pkl", "train_n_anom.npy")
TEST_ARTIFACTS = ("test_model_inputs.pkl", "test_series_meta.pkl")
VAL_ARTIFACT = "val_model_inputs.pkl"


def link(src: str, dst: str) -> None:
    if not os.path.exists(src):
        raise SystemExit(
            f"Missing pool artifact: {src}\nRun  bash run_prepare_family.sh  first.")
    if os.path.islink(dst) or os.path.exists(dst):
        os.remove(dst)
    os.symlink(os.path.abspath(src), dst)


def build_val(prepared_dir: str, pool_root: str, train, seed: int) -> None:
    """Concatenate the fold's TRAIN datasets' val pkls into prepared/val_model_inputs.pkl.

    Only the fold's own train datasets contribute — the unified run validates on its 9
    pool datasets, a --per_family run on that family's siblings alone, and a held-out
    dataset never appears in either (it has no val pkl in the pool at all). eval_loss
    steers best-model selection, so this is the same leak-free boundary the train/test
    artifact split already enforces.

    This one file is a real concatenation, not a symlink: the trainer wants a single
    val pkl, and which datasets belong in it is a property of the FOLD, not of the pool.
    It is ~9 x 200 windows, so the copy is cheap.
    """
    pool = []
    missing = []
    for ds in sorted(train):
        src = os.path.join(pool_root, ds, VAL_ARTIFACT)
        if not os.path.exists(src):
            missing.append(ds)
            continue
        with open(src, "rb") as f:
            items = pickle.load(f)
        for d in items:
            d["dataset"] = ds            # level-1 group id; the trainer strips it
        pool.extend(items)

    if not pool:
        print(f"  VAL        : none in the pool -> fine-tune with NO_VALIDATION=1")
        return

    # Interleave the datasets: eval batches sequentially, so a dataset-grouped pool would
    # make every eval batch single-dataset. Seeded -> the val pkl stays deterministic.
    order = np.random.default_rng(seed).permutation(len(pool))
    pool = [pool[i] for i in order]

    dst = os.path.join(prepared_dir, VAL_ARTIFACT)
    with open(dst, "wb") as f:
        pickle.dump(pool, f)

    n_anom = sum(1 for d in pool if int(np.asarray(d["future_labels"]).sum()) >= 1)
    print(f"  VAL        : {len(pool)} windows from {len(train) - len(missing)} TRAIN "
          f"datasets ({n_anom} anomalous, {n_anom / len(pool):.0%}) -> {dst}")
    if missing:
        print(f"               NO val pkl in the pool for {missing} — absent from the val "
              f"set. Re-run the prep (VAL_ONLY=1 is enough) to add them.")


def assemble(per_ds_out: str, pool_root: str, train, holdouts) -> None:
    os.makedirs(per_ds_out, exist_ok=True)
    for ds in train:
        d = os.path.join(per_ds_out, ds)
        os.makedirs(d, exist_ok=True)
        for a in TRAIN_ARTIFACTS:
            link(os.path.join(pool_root, ds, a), os.path.join(d, a))
    for ds in holdouts:
        d = os.path.join(per_ds_out, ds)
        os.makedirs(d, exist_ok=True)
        for a in TEST_ARTIFACTS:
            link(os.path.join(pool_root, ds, a), os.path.join(d, a))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--pool", default=os.path.join(_HERE, "pool"))
    p.add_argument("--run_dir", default=os.path.join(_HERE, "run"))
    p.add_argument("--per_family_dir", default=os.path.join(_HERE, "per_family"))
    p.add_argument("--families", default=os.path.join(_HERE, "families.json"))
    p.add_argument("--per_family", action="store_true",
                   help="Also build one dir per family (the attribution variant).")
    p.add_argument("--seed", type=int, default=42,
                   help="Shuffle seed for the assembled val set (keeps the pkl fixed).")
    args = p.parse_args()

    with open(args.families) as f:
        fams = json.load(f)["families"]

    pool_root = os.path.join(args.pool, "per_dataset")

    train_pool, holdouts = [], []
    for name, fam in fams.items():
        for ds in fam["train"]:
            if ds not in train_pool:
                train_pool.append(ds)
        holdouts.append(fam["holdout"])

    clash = sorted(set(train_pool) & set(holdouts))
    if clash:
        raise SystemExit(
            f"Dataset(s) {clash} are both in the train pool and held out. With a single "
            f"pooled model that is leakage. Fix families.json.")

    # ── the unified run: one model, one directory ────────────────────────────
    prepared = os.path.join(args.run_dir, "prepared")
    assemble(os.path.join(prepared, "per_dataset"), pool_root, train_pool, holdouts)

    print("UNIFIED RUN  (one model)")
    print(f"  TRAIN pool ({len(train_pool)}): {' '.join(sorted(train_pool))}")
    print(f"  HELD OUT   ({len(holdouts)}): {' '.join(sorted(holdouts))}")
    build_val(prepared, pool_root, train_pool, args.seed)
    print(f"  -> {prepared}")
    print()
    print("  family        train on                             held out        tier")
    print("  " + "-" * 74)
    for name, fam in fams.items():
        tier = fam.get("tier", 1)
        mark = "" if tier == 1 else "  <- underpowered"
        print(f"  {name:<13} {'+'.join(fam['train']):<36} {fam['holdout']:<15} "
              f"{tier}{mark}")

    if args.per_family:
        print("\nPER-FAMILY RUNS  (attribution variant -- one model each)")
        for name, fam in fams.items():
            fam_prepared = os.path.join(args.per_family_dir, name, "prepared")
            assemble(os.path.join(fam_prepared, "per_dataset"),
                     pool_root, fam["train"], [fam["holdout"]])
            # This fold's val = its own train siblings only, never its holdout.
            build_val(fam_prepared, pool_root, fam["train"], args.seed)
            print(f"  {name:<13} -> {fam_prepared}")

    print("\nNext:  bash run_study.sh")


if __name__ == "__main__":
    main()
