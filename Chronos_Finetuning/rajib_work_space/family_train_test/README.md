# Family transfer study

**Question.** Train **one** model on a pool of datasets in which every held-out dataset has
a same-family sibling. Does it detect anomalies in the held-out datasets — which it has
**never seen** — better than zero-shot?

This is strictly harder than `TOTAL_RUN*`. Those arms train on the train-half of every
dataset and test on the other half, so the model has always seen the target dataset. Here
each evaluated dataset contributes **zero** training windows.

## Protocol

Per family, one or more datasets go into a **shared training pool**; the rest are
**held out**. Then **one** model is fine-tuned on the whole pool and evaluated on every
held-out dataset. A family may hold out **more than one** dataset (`holdout` is a list).

| role | datasets | what is used |
|---|---|---|
| **TRAIN pool (8)** | MITDB, SMAP, CalIt2, GECCO, GHL, SMD, Exathlon, Daphnet | 100% of their `*test.csv` (+ syn) |
| **HELD OUT (8)** | SVDB, MSL, metro, room-occupancy, Genesis, PSM, cicids, OPPORTUNITY | 100% of their real `*test.csv` |

There is **no within-dataset file split anywhere**. A dataset is entirely in the pool or
entirely held out, so the study is leak-free by construction — not by a seeded split that
has to be argued about. (`prepare_family.py` hard-errors if any dataset lands in both.)

Only `*test.csv` is ever read, in either role: in mTSBench the `*train.csv` files carry no
anomaly labels, and the per-step masked margin loss needs labels on both sides of the mask.

| family | in the pool | held out | tier |
|---|---|---|---|
| ECG / physiological | MITDB (47 f) | **SVDB** (78 f) | 1 |
| Spacecraft telemetry | SMAP (51 f) | **MSL** (26 f) | 1 |
| Environmental / building IoT | CalIt2 + GECCO (1+1 f, +syn) | **metro** (1 f), **room-occupancy** (2 f) | 2 |
| Industrial process | GHL (14 f) | **Genesis** (1 f) | 2 |
| Server / cloud | SMD + Exathlon (18+30 f) | **PSM** (1 f), **cicids\*** (6 f) | 1 |
| Human activity | Daphnet (26 f) | **OPPORTUNITY** (13 f) | 1 |

**cicids\*** is a *stretch* of "server": it is network-intrusion traffic, not host/cloud
metrics. Its row carries the asterisk — a null there may be sub-domain mismatch, not failed
transfer. PSM (anomaly-dense) is the clean server-family holdout.

Excluded: **creditcard, swan, GutenTAG** — singleton families, no sibling, so same-family
transfer is undefined. (`cicids` used to be here; it now serves as a server-family holdout
with the asterisk above.)

## Hierarchical sampling — now 1.5-level

Same sampler, same code, same knobs as `TOTAL_RUN_maskloss_v2_HS`, **including its optional
file level (1.5)**. The sampler discovers its datasets from
`per_dataset/*/train_model_inputs.pkl`, so pointing it at the unified pool makes:

```
level 1    dataset ~ Uniform(K)              K = 8 train datasets  (was 12 there)
level 1.5  file    ~ Categorical(w_f)        multi-file train datasets only
level 2    kind    ~ Bernoulli(P_ANOM)       P_ANOM = 1/3  -> 2:1 normal:anomalous
level 3    window  ~ count-weighted within (dataset, file, kind)
```

Every dataset gets 1/8 of the draws **regardless of size** (MITDB's ~382k windows do not
drown out room-occupancy). **Level 1.5** then splits a dataset's own budget across its
*distinct patterns* rather than its longest recordings: each train file gets a
dissimilarity weight `w_f` (inverse kernel density over a 22-number per-file signature),
so a near-duplicate file is down-weighted and a lone unusual one promoted. It is written
per train dataset by the prep (`train_files.json` + `train_file_index.npy`), toggled by
`FILE_DIVERSITY_DATASETS` at fine-tune time (default `all`), and is a no-op on single-file
train datasets. Set `FILE_DIVERSITY_DATASETS=""` for the plain 3-level ablation baseline.

## Synthetic data (train only)

`USE_SYN_DATA=1` (default) appends each **train** dataset's `syn_data/*test.csv` to its
train windows — never a held-out dataset's test set (evaluation stays 100% real), never the
val set. Here it applies to the two single-file train datasets, **CalIt2** and **GECCO**:
it turns each into a 5-file dataset, which both grows its thin train pool (CalIt2 rises
from ~3 to 81 anomaly windows) *and* activates level 1.5 for it. Discovery of real files is
**top-level only** so `syn_data/` is never mistaken for real data — the earlier recursive
glob would have leaked synthetic windows into the metro/Genesis/PSM test sets.

## Read the tiers

**Tier 1 is the result.** Tier 2 is reported separately and is **underpowered**, not merely
noisy — these are measured, not estimated:

- **Genesis**: ~28 anomalous timesteps in total (0.30% of 9,332).
- **metro**: ~31 anomalous timesteps (368 test windows). `room-occupancy`, its env_iot
  co-holdout, is denser (~2,765 anomalous steps) and better conditioned.

At those counts VUS-PR is dominated by sampling noise. **A bad tier-2 number does not mean
transfer failed** — it means the measurement cannot resolve the question. Note CalIt2 is now
a *train* dataset (anomaly-sparse itself, so its level-3 anomalous branch is thin even with
syn data).

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
several of those *did* have a same-family sibling in its training set (PSM↔SMD/Exathlon,
Genesis↔GHL, …) while CreditCard had **none** — and the paper never varies this. Here,
sibling coverage is guaranteed by design.

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
bash run_forward_family.sh              # fine-tuned -> all 8 held-out   (the experiment)
ZERO_SHOT=1 bash run_forward_family.sh  # base model -> all 8 held-out   (the baseline)
python summarize_study.py
```

**Cost:** ~4.0 s/step measured, so 4000 steps ≈ **4.5 h** for the single fine-tune, plus
forward passes. Seven of the eight held-out sets are cheap to score (MSL 71k, PSM 70k,
cicids, metro, room-occupancy, Genesis 9k, OPPORTUNITY 310k steps); only SVDB (14.3M) is
expensive.

## How the plumbing works

Nothing in the trainer or `forward.py` was modified. The run is one directory of symlinks,
and the two entrypoints partition it by themselves:

```
run/prepared/per_dataset/<TRAIN_DS>/train_model_inputs.pkl -> pool/...   (x8)
                                    train_n_anom.npy       -> pool/...
                                    train_file_index.npy   -> pool/...   (level 1.5)
                                    train_files.json       -> pool/...   (level 1.5)
                        /<HELDOUT>/test_model_inputs.pkl   -> pool/...   (x8)
                                   test_series_meta.pkl    -> pool/...
```

- `finetune_anomaly_simple.py :: load_train_pool()` walks `per_dataset/*/` and **skips any
  folder without `train_model_inputs.pkl`** → it sees exactly the 8 pool datasets, and reads
  the two `train_file*` sidecars beside each for level 1.5.
- `forward.py` (via `run_forward_family.sh`) walks `per_dataset/*/` and **skips any folder
  without `test_model_inputs.pkl`** → it sees exactly the 8 held-out datasets.

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
