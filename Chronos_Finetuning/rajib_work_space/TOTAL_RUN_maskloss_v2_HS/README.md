# TOTAL_RUN_maskloss_v2_HS — hierarchical-sampling mTSBench fine-tuning

Combined-mTSBench Chronos-2 anomaly fine-tuning that fixes **both** imbalances in the
data with **one** mechanism, at **train time**, with **no threshold and nothing discarded**.

| Imbalance | Fixed where | How |
|---|---|---|
| **Dataset** (MITDB+SVDB are ~88% of raw windows) | **runtime** | HS **level 1**: draw a dataset uniformly |
| **Class** (anomaly *steps* are ~17% of forecast steps) | **runtime** | HS **levels 2–3**: draw a kind (`P_ANOM`), then count-weight windows within that dataset |

Prep does nothing but carve windows. Every train window of every dataset is kept.

## The sampler

Each window in a train batch is drawn in three independent levels:

```
level 1   k    ~ Uniform(K)                       K = 12 train datasets
level 2   kind ~ Bernoulli(p_anom = 1/3)          2:1 normal:anomalous, per dataset
level 3   i    ~ Categorical over dataset k's windows, weight
                    n_anom_i           if kind == anomalous
                    64 - n_anom_i      if kind == normal
```

Equivalently, the per-draw marginal within the chosen dataset is

```
P(i | k) = (2/3) · n_norm_i / Σ_k n_norm   +   (1/3) · n_anom_i / Σ_k n_anom
```

Three properties fall out of this, and they are the whole point:

**1. Level 3 is thresholdless in both branches.** The anomalous branch is *self-gating*:
a pure-normal window has `n_anom = 0`, hence weight 0, so it can never be drawn — no
`future_labels.sum() >= 10` cutoff is needed. (Verified: 0 pure-normal windows drawn from
the anomalous branch in 200,000 draws.) The normal branch weights an anomaly-bearing
window by its *remaining* normal steps `64 - n_anom`, i.e. down in proportion to how
anomalous it is, rather than excluding it. This is the inconsistency that FT's — and
therefore 3x's — window-level threshold introduced into an otherwise per-step objective.

**2. Level 3's weights are renormalized *within* the chosen dataset.** So an
anomaly-sparse dataset still yields anomalies (its rare anomaly windows carry all of the
anomalous branch's mass) and an anomaly-dense dataset still yields normals. Measured on
the actual train pool, this maps natural anomaly-step rates spanning **1.0% to 27.0%**
onto a near-uniform **27–33%**:

| dataset | anomaly windows | natural anomaly-step % | under HS |
|---|---|---|---|
| GHL | 95 / 8,694 | 1.0% | 31.8% |
| OPPORTUNITY | 53 / 2,622 | 1.7% | 31.5% |
| GutenTAG | 101 / 2,220 | 3.4% | 29.4% |
| SMD | 162 / 2,204 | 3.4% | 27.1% |
| MITDB | 22,477 / 186,691 | 10.5% | 32.1% |
| Daphnet | 849 / 5,612 | 13.5% | 32.4% |
| Exathlon | 793 / 5,557 | 13.5% | 33.0% |
| MSL | 62 / 337 | 13.7% | 30.7% |
| SMAP | 472 / 3,025 | 14.9% | 32.9% |
| SVDB | 25,144 / 111,288 | 15.5% | 30.0% |
| room-occupancy | 29 / 113 | 21.9% | 32.2% |
| cicids | 6,486 / 12,160 | 27.0% | 32.6% |

Pool-wide: **12.47% natural → 31.31% expected** (Monte-Carlo over 600k draws: 31.34%).

There is no `SAMPLING_TARGET` to tune. The ~31% is what `p_anom = 1/3` *implies*, given
that mTSBench anomalies are long contiguous segments (an anomaly-bearing window averages
40–60 of its 64 steps anomalous). Note cicids, at 53% anomaly *windows*, is the dataset
that needs level 2's **normal**-kind protection rather than its anomalous one — which is
exactly the contrastive case the two-sided draw was designed for.

**3. The two levels don't fight each other.** This is what `../TOTAL_RUN_maskloss_v2/`
could not do. It used a **single global** weight vector `w_i = n_anom_i + eps·H` for the
class job and a per-dataset cap at prep for the dataset job — but the global vector
reaches across datasets and silently re-skews the mix the cap had just fixed. Solving
`eps` for a 40% anomaly-step target on v2's own capped pool gives:

| dataset | share of capped pool | effective draw share | ratio |
|---|---|---|---|
| MITDB / SVDB / cicids | 12.3% each | 16.7% each | **1.35×** |
| GHL | 12.3% | 8.4% | 0.68× |
| OPPORTUNITY | 12.3% | 4.4% | **0.36×** |
| GutenTAG / SMD | 12.3% each | 3.9% / 4.1% | ~0.72× |

Under HS every dataset gets exactly `1/K` of the draws. Measured over 600k draws on the
real pool (uniform target 8.33%):

| dataset | share of pool | realized draw share | ratio |
|---|---|---|---|
| MITDB | 54.82% | 8.37% | 1.00× |
| SVDB | 32.68% | 8.35% | 1.00× |
| cicids | 3.57% | 8.38% | 1.01× |
| GHL | 2.55% | 8.35% | 1.00× |
| MSL | 0.10% | 8.30% | 1.00× |
| room-occupancy | 0.03% | 8.27% | 0.99× |

A pool that is 88% ECG is drawn from as though it were 16.7% ECG.

## Why this exists (vs the neighbouring dirs)

| arm | dataset balance | class balance | threshold? | discards data? |
|---|---|---|---|---|
| `../TOTAL_RUN/` (**FT**) | per-dataset anomaly cap at prep | `normal_ratio=2.0` at prep | **yes**, `≥10` steps | yes |
| `../TOTAL_RUN_maskloss_v2/` (**2x**) | `PER_DATASET_CAP` at prep (*undone by the sampler*) | global count-weighted sampler | no | yes (cap) |
| **3x** (FT's data + 2x's trainer) | both of the above, stacked | both, stacked | **yes**, inherited from FT | yes |
| **this dir** (**HS**) | HS level 1 | HS levels 2–3 | **no** | **no** |

3x is the current best result, but it is double-balanced: FT's prep had already lifted
anomalies to ~33% of windows *and* capped anomalies at the per-dataset median, so 2x's
runtime sampler was re-balancing an already small, already distorted pool. HS reproduces
3x's 2:1 window guarantee from a clean, complete pool and without the threshold.

## Two things to know about the batch

**`BATCH_SIZE` counts channel ROWS, not windows.** `Chronos2Dataset._generate_train_batches`
fills a batch until `Σ group_size >= batch_size`, and `group_size` is the window's channel
count `F`. Mean `F` across the 12 train datasets is 296/12 ≈ 24.7, so a 160-row batch holds
**~7 windows** drawn from **~5.6 distinct datasets**. Measured on the real pool: **7.31
windows/batch, 180.1 rows** (the last draw overshoots the 160-row budget; max observed 231).
Resampling the dataset per *window* rather than per 3-window group is what buys that
diversity — a per-group draw would give only ~2.3 distinct datasets per batch — and it
matters because `AGG_MODE=batch_global` pools the loss across the whole batch.

**Level 1 balances *draws*, not *gradient*.** Each drawn window contributes `F_d` rows of
per-step loss, so under `batch_global` a dataset's share of the gradient is `F_d / ΣF`
(ΣF = 296), **not** `1/K`:

| | F | gradient share | vs. uniform (8.3%) |
|---|---|---|---|
| cicids | 72 | 24.3% | 2.9× |
| MSL | 55 | 18.6% | 2.2× |
| MITDB / SVDB | 2 | 0.68% each | **0.08×** |

This is **inherited from 2x and 3x, not introduced by HS** — their dataset draw shares are
non-uniform *and then* scaled by `F_d`. `AGG_MODE=batch_global` is kept so that HS-vs-2x is
a strict one-variable ablation. `AGG_MODE=per_window` would equalize it (each drawn window
contributes equally, dataset gradient share → `1/K`) and is the natural follow-up arm.

A secondary consequence of level 2 being a Bernoulli rather than an exact 2:1 group: the
anomalous-kind count per batch is `Binomial(~7, 1/3)`, so some micro-batches contain no
anomaly steps at all — **measured 4.5%** (lower than the ~6% you'd get from the Bernoulli
alone, because soft-normal draws still carry anomaly steps).
`L_bad_term = step_hinge.sum() / n_anom_steps.clamp(min=eps)` handles that cleanly: the
batch simply contributes `L_good`, no NaN. With `GRAD_ACCUM=2` only ~0.2% of optimizer
steps see it. This is visible in the training log as a step with no `anomaly_loss` key.

## Files

| File | Role | Relation to `../TOTAL_RUN_maskloss_v2/` |
|---|---|---|
| `prepare_total.py` / `run_prepare_total.sh` | uncapped, thresholdless, **per-dataset** prep | rewritten (cap logic removed) |
| `finetune_anomaly_simple.py` | HS sampler + per-step masked hinge trainer | **only `AnomalyChronos2Dataset` differs**; loss/model/LoRA untouched |
| `run_finetune_total.sh` | `P_ANOM` replaces `SAMPLING_TARGET`; all else identical | |
| `forward.py` | anomaly evaluator | **byte-identical** (md5 `a226d1f5e899d7ae332112e3f29d076f`) |
| `run_forward_total.sh` | loops `forward.py` over every per-dataset test set | only the default `CHECKPOINT` path differs |
| `aggregate_results.py` | per-series CSVs → one VUS-PR value per dataset | **byte-identical** (md5 `9b591ef8fb3e67c02637875615263fd4`) |

`prepare_total.py` imports the windowing primitives from `../SMD_run/prepare_smd_split.py`,
so train/test windows are byte-identical to how SMD and the other arms built them.

### Inference is identical across arms, by construction

Not asserted — *enforced*. `run_prepare_total.sh` **symlinks** each
`prepared_total/per_dataset/<DS>/test_model_inputs.pkl` to the file
`../TOTAL_RUN_maskloss_v2/prepared_total/` already evaluated, and `forward.py` /
`aggregate_results.py` are byte-identical copies. Set `LINK_TEST_FROM=""` to re-carve them
instead; the bytes match either way (same seeded file split, same geometry), but linking
removes the possibility of drift and halves prep time.

## Data layout

```
prepared_total/
  manifest.json
  per_dataset/<DATASET>/
    train_model_inputs.pkl     # UNCAPPED train half  (absent for test-only datasets)
    train_n_anom.npy           # int16 (N,) anomaly-step count per window
    test_model_inputs.pkl  ->  symlink into ../TOTAL_RUN_maskloss_v2/prepared_total/...
    test_series_meta.pkl   ->  symlink
```

The train pool is **340,523 windows / 56,723 anomaly-bearing (16.7%) / 6.6 GB** float32.
12 of the 19 datasets contribute; the other 7 (CalIt2, GECCO, Genesis, PSM, creditcard,
metro, swan) have exactly one `*test.csv`, so they are **test-only** — they contribute no
train windows and are invisible to level 1. Biggest contributors: MITDB 186,691 windows
(1.3 GB), SVDB 111,288 (742 MB), cicids 12,160 (2.8 GB — only 3.6% of the windows but
`F=72`).

`train_n_anom.npy` exists so the sampler's expected balance can be audited without loading
6.6 GB of windows.

**Startup cost.** Loading the pool takes ~12 s; `AnomalyChronos2Dataset.__init__` (the
parent's tensor conversion, then the 12 pairs of cumulative distributions) takes ~50 s and
settles at **16.0 GB RSS**. Batch generation is 2.5 ms/batch — the sampler is not the
bottleneck. Budget ~1 min of startup and ~16 GB of RAM before the first training step.

The sampler needs the pool grouped by dataset, so a flat `train_model_inputs.pkl` (the
`TOTAL_RUN` / `TOTAL_RUN_maskloss_v2` layout) **will not work here** — it carries no
dataset identity and level 1 would have nothing to group by. `run_finetune_total.sh`
fails fast if `per_dataset/` is missing.

## Environment

Use **`debug_chronos`** for all three stages — pandas (prep), torch 2.10 + transformers +
peft (train), numpy + chronos (forward):

```bash
export PATH="/home/rajib/miniconda3/envs/debug_chronos/bin:$PATH"
```

(The runners set `PYTHONPATH` to `rajib_work_space` themselves so the local `chronos`
package is used.)

## How to run

### 1. Prepare (uncapped per-dataset train pools; test halves symlinked)

```bash
cd TOTAL_RUN_maskloss_v2_HS
bash run_prepare_total.sh                        # links test pkls from ../TOTAL_RUN_maskloss_v2
LINK_TEST_FROM="" bash run_prepare_total.sh      # re-carve the test halves instead
DATASETS="SMD MSL SMAP" bash run_prepare_total.sh
```

Prep **fails loudly** if any dataset's train half has zero anomaly windows (level 2's
anomalous branch would have an all-zero weight vector), and **warns** below
`MIN_ANOM_WINDOWS=50`. All 12 train datasets clear the hard check; room-occupancy (29
anomaly windows) trips the warning.

No `val_model_inputs.pkl` is written — train with `NO_VALIDATION=1`. `EVAL_TEST` stays
manual via `TEST_DATA=/path/to/eval_test.pkl`.

### 2. Fine-tune

```bash
bash run_finetune_total.sh                       # P_ANOM=1/3, MARGIN_M=5, NO_VALIDATION=1
P_ANOM=0.5 bash run_finetune_total.sh            # 1:1 normal:anomalous kinds
DEBUG=1 bash run_finetune_total.sh               # 50 windows PER DATASET, smoke test
```

Checkpoint → `chronos2-single-stage_mtsbench_maskLossv2_HS_v1/finetuned-ckpt`.

At startup the log prints the per-dataset natural → HS anomaly-step fractions and the
pool-wide expectation. Every 500 batches it prints the **realized** anomaly-step fraction
and the **realized dataset mix**:

```
[hs] realized over 500 batches: anomaly-step fraction 30.0% (expected 30.8%);
     dataset mix (uniform target 33.3%): GutenTAG=32.7%  MSL=35.2%  room-occupancy=32.1%
```

Both lines should track their targets. If the dataset mix drifts, level 1 is broken; if the
anomaly-step fraction drifts, level 3 is.

`DEBUG=1` truncates to 50 windows **per dataset** (not the first 50 of the flat pool),
so every level-1 group survives the smoke test.

### 3. Evaluate (per-dataset → the deliverable)

```bash
bash run_forward_total.sh                        # fine-tuned, all datasets -> results_FT/
CHECKPOINT="" bash run_forward_total.sh          # zero-shot baseline    -> results_ZS/
DATASETS="SMD MSL" bash run_forward_total.sh     # subset
```

Produces, per run:
- `results_<FT|ZS>/<DATASET>_results.csv` — one row per series (raw).
- **`results_<FT|ZS>/per_dataset_summary.csv`** — **the deliverable**: one VUS-PR (and
  VUS-ROC/AUC/F1) value per dataset = mean over that dataset's series.
- printed **MACRO** VUS-PR (mean of the per-dataset values; the headline for comparing
  runs) and **micro** (all series pooled; reference only, biased to big datasets).

## Choosing NUM_STEPS

"Epoch" is **not meaningful** here. The sampler draws **with replacement** from an uncapped
340k-window pool and never enumerates it. At `BATCH_SIZE=160` rows × `GRAD_ACCUM=2` and a
measured 7.31 windows/batch, 4000 steps ≈ **58,000 window draws** ≈ 4,900 per dataset — so
MITDB sees ~2.6% of its 186,691 windows while room-occupancy's 113 windows are each
revisited ~43×. That asymmetry is inherent to uniform level-1 sampling and is the price of
killing dataset dominance; watch the small datasets for overfitting.

`NUM_STEPS=4000` is held **identical to `../TOTAL_RUN_maskloss_v2/`** so this arm changes
the sampler and nothing else. With `NO_VALIDATION=1` the *final*-step weights are kept (no
best-checkpoint restore), so raising it is a real overtraining risk on the small datasets.

## Config summary

| Knob | Default | Where | Note |
|---|---|---|---|
| — | — | prep | **no cap, no threshold, no class balancing** |
| `LINK_TEST_FROM` | `../TOTAL_RUN_maskloss_v2/prepared_total` | prep | symlink test pkls; `""` to re-carve |
| `P_ANOM` | `1/3` | finetune | HS level 2; level 1 is uniform and has no knob |
| `MARGIN_M` | 5 | finetune | identical to 2x |
| `AGG_MODE` | `batch_global` | finetune | identical to 2x; see the gradient-share caveat above |
| `NUM_STEPS` | 4000 | finetune | identical to 2x |
| `BATCH_SIZE` | 160 | finetune | channel **rows**, ≈6–7 windows |
| `NO_VALIDATION` | 1 | finetune | no val pkl is written |
| `CONTEXT_LENGTH` | 768 | finetune | = 256 normal + 512 context |
