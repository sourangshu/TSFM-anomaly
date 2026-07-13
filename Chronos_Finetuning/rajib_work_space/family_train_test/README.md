# Family transfer study

**Question.** Train **one** model on a pool of datasets in which every held-out dataset has
a same-family sibling. Does it detect anomalies in the held-out datasets — which it has
**never seen** — better than zero-shot?

This is strictly harder than `TOTAL_RUN*`. Those arms train on the train-half of every
dataset and test on the other half, so the model has always seen the target dataset. Here
each evaluated dataset contributes **zero** training windows.

## Protocol

Per family, the higher-content dataset(s) go into a **shared training pool**; the remaining
one is **held out**. Then **one** model is fine-tuned on the whole pool and evaluated on
every held-out dataset.

| role | datasets | what is used |
|---|---|---|
| **TRAIN pool (9)** | MITDB, SMAP, Exathlon, SMD, Daphnet, GHL, GECCO, metro, room-occupancy | 100% of their `*test.csv` |
| **HELD OUT (6)** | SVDB, MSL, PSM, OPPORTUNITY, Genesis, CalIt2 | 100% of their `*test.csv` |

There is **no within-dataset file split anywhere**. A dataset is entirely in the pool or
entirely held out, so the study is leak-free by construction — not by a seeded split that
has to be argued about. (`prepare_family.py` hard-errors if any dataset lands in both.)

Only `*test.csv` is ever read, in either role: in mTSBench the `*train.csv` files carry no
anomaly labels, and the per-step masked margin loss needs labels on both sides of the mask.

| family | in the pool | held out | tier |
|---|---|---|---|
| ECG / physiological | MITDB (47 f, 24.4M steps) | **SVDB** (78 f, 14.3M) | 1 |
| Spacecraft telemetry | SMAP (51 f, 408k) | **MSL** (26 f, 71k) | 1 |
| Server / cloud | Exathlon + SMD (51 f, 934k) | **PSM** (1 f, 70k) | 1 |
| Human activity | Daphnet (26 f, 613k) | **OPPORTUNITY** (13 f, 310k) | 1 |
| Industrial process | GHL (14 f, 1.15M) | **Genesis** (1 f, 9.3k) | 2 |
| Environmental / building IoT | GECCO + metro + room-occupancy (4 f, 121k) | **CalIt2** (1 f, 4k) | 2 |

Excluded: **cicids, creditcard, swan, GutenTAG** — singleton families, no sibling, so
same-family transfer is undefined. They are in neither the pool nor the held-out set.

## Hierarchical sampling

Unchanged from `TOTAL_RUN_maskloss_v2_HS` — same sampler, same code, same knobs. It
discovers its datasets from `per_dataset/*/train_model_inputs.pkl`, so pointing it at the
unified pool simply makes:

```
level 1   dataset ~ Uniform(K)            K = 9   (was 12 there)
level 2   kind    ~ Bernoulli(P_ANOM)     P_ANOM = 1/3  -> 2:1 normal:anomalous
level 3   window  ~ count-weighted WITHIN that dataset  (n_anom  or  64 - n_anom)
```

Every dataset gets 1/9 of the draws **regardless of size**, which is the whole point:
MITDB (~382k windows) does not drown out room-occupancy (206).

## Read the tiers

**Tier 1 is the result.** Tier 2 is reported separately and is **underpowered**, not merely
noisy — these are measured, not estimated:

- **Genesis**: ~28 anomalous timesteps in total (0.30% of 9,332).
- **CalIt2**: 55 test windows, ~138 anomalous timesteps (3.4% of 4,032). `metro`, in the
  *pool*, has ~31.

At those counts VUS-PR is dominated by sampling noise. **A bad tier-2 number does not mean
transfer failed** — it means the measurement cannot resolve the question.

Two tier-1 caveats to state in the writeup:

- **Human activity**: Daphnet and OPPORTUNITY share a sensor modality (body-worn IMU) but
  *not* anomaly semantics (freezing-of-gait vs. activity transition). A null result there
  is ambiguous between domain mismatch and label mismatch.
- **ECG is the flagship**: same modality, same channel count (F=3), same anomaly semantics
  (arrhythmic beats), ~13% anomalies on both sides.

## Baselines: what you may and may not reuse

- **Zero-shot must be re-run here.** `TOTAL_RUN_maskloss_v2_HS/results_ZS` is scored on 50%
  test *halves*; this study evaluates 100% of each held-out dataset. Different test set,
  non-comparable. `run_study.sh` re-runs it.
- **`results_FT` is NOT a valid reference column.** That checkpoint trained on the
  train-half files of *every* dataset, and under this protocol those files are part of the
  test set. It is leaky here **by construction**. Do not put it in the table.

## Relation to F2A, and the one caveat to state

This is essentially the protocol of
[F2A (arXiv 2511.03149)](https://arxiv.org/abs/2511.03149) — one jointly fine-tuned model,
evaluated on datasets excluded from training — **with its confound removed**. F2A holds out
6 datasets (GECCO, PSM, Genesis, Daphnet, SWaT, CreditCard) because TSB-AD-M's benchmark
protocol says to, not for any scientific reason, and never groups datasets by domain. But
4 of those 6 *did* have a same-family sibling in its training set (Daphnet↔OPPORTUNITY,
PSM↔SMD/Exathlon, Genesis↔GHL, SWaT↔GHL) while CreditCard had **none** — and the paper
never varies this. Here, sibling coverage is guaranteed by design.

**The caveat, and it belongs in the writeup:** a single pooled model **cannot attribute** a
gain on SVDB to MITDB specifically — the model also saw SMAP, SMD, GHL and the rest. This
design answers *"does a family-sibling-covered pool transfer to unseen datasets"*, **not**
*"which sibling did the work"*. Attribution needs one model per family:

```bash
PER_FAMILY=1 bash run_prepare_family.sh     # builds per_family/<fam>/prepared
RUN=per_family/ecg bash run_finetune_family.sh
RUN=per_family/ecg bash run_forward_family.sh
```

That is 6 models (~27 GPU-h) instead of 1 (~4.5 h), so it is a follow-up, not the first pass.

## Usage

```bash
conda activate chronos_clean

bash run_study.sh                  # prepare -> finetune -> score -> zero-shot -> summarize
SKIP_PREPARE=1 bash run_study.sh   # pool/ already built
SKIP_TRAIN=1   bash run_study.sh   # re-score the existing checkpoint only
```

Or a stage at a time:

```bash
bash run_prepare_family.sh              # -> pool/ + run/prepared
bash run_finetune_family.sh             # -> run/ckpt/finetuned-ckpt   (~4.5 h)
bash run_forward_family.sh              # fine-tuned -> all 6 held-out   (the experiment)
ZERO_SHOT=1 bash run_forward_family.sh  # base model -> all 6 held-out   (the baseline)
python summarize_study.py
```

**Cost:** ~4.0 s/step measured, so 4000 steps ≈ **4.5 h** for the single fine-tune, plus
forward passes. Five of the six held-out sets are cheap to score (MSL 71k, PSM 70k,
Genesis 9k, CalIt2 4k, OPPORTUNITY 310k steps); only SVDB (14.3M) is expensive.

## How the plumbing works

Nothing in the trainer or `forward.py` was modified. The run is one directory of symlinks,
and the two entrypoints partition it by themselves:

```
run/prepared/per_dataset/<TRAIN_DS>/train_model_inputs.pkl -> pool/...   (x9)
                                    train_n_anom.npy       -> pool/...
                        /<HELDOUT>/test_model_inputs.pkl   -> pool/...   (x6)
                                   test_series_meta.pkl    -> pool/...
```

- `finetune_anomaly_simple.py :: load_train_pool()` walks `per_dataset/*/` and **skips any
  folder without `train_model_inputs.pkl`** → it sees exactly the 9 pool datasets.
- `forward.py` (via `run_forward_family.sh`) walks `per_dataset/*/` and **skips any folder
  without `test_model_inputs.pkl`** → it sees exactly the 6 held-out datasets.

No dataset carries both artifacts, so neither stage can touch the other's data. Each dataset
is carved **once** into `pool/` (7.5 GB) and pointed at from there.

## Files

| file | role |
|---|---|
| `families.json` | family definitions + train/holdout roles — **edit this to change the study** |
| `prepare_family.py` | carves each dataset whole into `pool/`; hard-errors on train/holdout overlap |
| `make_folds.py` | assembles `run/prepared` (and optionally `per_family/*`) as symlinks |
| `run_prepare_family.sh` | prep driver |
| `run_finetune_family.sh` | fine-tune the pooled model |
| `run_forward_family.sh` | score a model on all held-out datasets |
| `run_study.sh` | end-to-end driver |
| `summarize_study.py` | the deliverable table (zero-shot vs family-FT, per family) |

Everything else in this directory is a **symlink**, not a copy — nothing shared is duplicated:

| symlink | target |
|---|---|
| `finetune_anomaly_simple.py`, `forward.py`, `aggregate_results.py` | `../TOTAL_RUN_maskloss_v2_HS/` |
| `prepare_smd_split.py` | `../SMD_run/` (window primitives) |
| `chronos`, `VUS_ROC_VUS_PR` | `../` |

Linking rather than copying makes *"this study runs the same trainer and the same scorer as
`TOTAL_RUN_maskloss_v2_HS`"* **a fact rather than a claim** — the same argument that arm's
own `run_forward_total.sh` makes for symlinking its test pkls. The flip side, worth knowing:
editing those files in `TOTAL_RUN_maskloss_v2_HS` **changes this study too**. That is the
intended coupling (the only variable here is the training pool), but it does mean this
directory is not frozen against upstream edits. If you ever need a frozen arm, replace the
symlinks with copies at that point.

Hyperparameters in `run_finetune_family.sh` are copied unchanged from
`TOTAL_RUN_maskloss_v2_HS` (per-step masked margin loss, relative margin M=5, HS sampler
P_ANOM=1/3, LoRA r=32, lr 1e-5, 4000 steps). **The only variable this study changes is which
datasets are in the pool.**
