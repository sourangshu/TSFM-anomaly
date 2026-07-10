# Chronos-2 Anomaly Fine-Tuning — Methodology (FT, 2x, 3x, HS)

This document describes the four anomaly-aware fine-tuning methods for Chronos-2 on the
combined **mTSBench** benchmark. All four share the same backbone, the same window
layout, and the **same inference/scoring stage**; they differ on only **two axes**:

1. **Data preparation** — how the combined train pool is balanced.
2. **Training objective** — the loss function and the training-time sampler.

| Method | Data prep script | Training script | Inference |
|--------|------------------|-----------------|-----------|
| **FT** | `TOTAL_RUN/prepare_total.py` | `finetune_anomaly_simple.py` (repo root) | `SMD_run/forward.py` |
| **2x** | `TOTAL_RUN_maskloss_v2/prepare_total.py` | `TOTAL_RUN_maskloss_v2/finetune_anomaly_simple.py` | `forward.py` (identical) |
| **3x** | `TOTAL_RUN/prepare_total.py` (**= FT's**) | `SMD_Maskloss_v2/finetune_anomaly_simple.py` (**= 2x's, byte-for-byte**) | `forward.py` (identical) |
| **HS** | `TOTAL_RUN_maskloss_v2_HS/prepare_total.py` | `TOTAL_RUN_maskloss_v2_HS/finetune_anomaly_simple.py` (**= 2x's except the sampler**) | `forward.py` (identical) |

> **Key structural fact:** `SMD_run/forward.py`, `SMD_Maskloss_v2/forward.py`,
> `TOTAL_RUN_maskloss_v2/forward.py`, and `TOTAL_RUN_maskloss_v2_HS/forward.py` are
> byte-for-byte identical (md5 `a226d1f5e899d7ae332112e3f29d076f`), the 2x and 3x training
> scripts are byte-for-byte identical, and HS's training script differs from 2x's only in
> the `AnomalyChronos2Dataset` sampler. HS additionally **symlinks** the per-dataset test
> pkls, so all four arms provably evaluate on identical bytes. The *only* real differences
> are the data-prep balancing and the loss/sampler.

---

## 0. Shared components (identical across FT, 2x, 3x, HS)

### Backbone and sequence layout
- **Model:** `amazon/chronos-2`, fine-tuned in **LoRA** mode (`lora_r=16`, `lora_alpha=32`,
  `lora_dropout=0.01`) by default; `full` mode available.
- **Window layout** (matched across prep, training, and inference):
  - `[SEP]` token enabled (`ENABLE_SEP_TOKEN=1`).
  - **Per-series normal prefix** of `NORMAL_SIGNAL_LENGTH=256` steps — each series
    carves its *own* normal zone (per-series, not a global/unified normal signal).
  - Context length `512`, prediction length `64`, input patch size `16`.
  - Training `CONTEXT_LENGTH=768` = normal prefix (256) + context (512).
- **Per-window data layout is byte-for-byte identical** across all prep variants; the
  carving/loading/pairing primitives are imported from `SMD_run/prepare_smd_split.py`.

### Inference and scoring (`forward.py`)
Run one invocation per dataset over each dataset's `test_model_inputs.pkl` +
`test_series_meta.pkl`. Test sets are **uncapped/unbalanced** (honest evaluation).
- **Anomaly score:** per-step forecast error, `SCORE_METHOD=mse`.
- **Multivariate aggregation:** `AGG_METHOD=l2`, `TOPK=4`, smoothing window `5`.
- **Metrics:** VUS-family (`SLIDING_WINDOW_VUS=100`, `VUS_VERSION=opt`, `VUS_THRE=250`),
  written to `results*/<DATASET>_results.csv` and concatenated into `ALL_results.csv`.
- Fine-tuned runs use the normal prefix (`USE_NORMAL_PREFIX=1`); zero-shot baselines
  use context only.

### Per-dataset split
- Datasets with ≥2 `*test.csv` files → **file-based 50/50** (seeded): half the files
  seed the combined train pool, half become that dataset's test set (no window leaks
  across splits).
- Datasets with exactly one `*test.csv` → **test-only**: the file goes entirely to
  the test set and contributes nothing to the combined train pool.

---

## 1. Method FT — thresholded class-balancing + window-level hinge loss

### Data preparation (`TOTAL_RUN/prepare_total.py`)
- **Anomaly thresholding:** a window is *anomalous* iff
  `future_labels.sum() >= ANOMALY_THRESHOLD` (`=10`), else *normal*.
- **Class balancing (2:1):** keep all anomalous windows, subsample the abundant normals
  to `NORMAL_RATIO=2.0` normals per anomalous window.
- **Dataset cap (de-domination):** `BALANCE_SCOPE=per_dataset_cap` with
  `PER_DATASET_ANOM_CAP=median` — cap each dataset's anomalous contribution to the
  median anomalous count across datasets *before* matching normals at 2:1, so
  anomaly-rich datasets (SVDB/MITDB) can't swamp small ones.
- Normal windows are reservoir-sampled (`NORMAL_RESERVOIR_CAP=200000`) to bound memory.

### Training (`finetune_anomaly_simple.py`, repo root)
**Window-level margin (hinge) loss** conditioned on each window's `future_type`:

```
L_total = L_good + lambda * max(0, tau - L_bad)
  future_type == 0 (normal)  -> L_good : minimise (predict the normal future well)
  future_type == 1 (anomaly) -> L_bad  : push UP toward margin tau, then stop
```

- The hinge **self-saturates**: once `L_bad >= tau` it contributes no gradient, so
  training cannot diverge.
- Key hyperparameters: `MARGIN_TAU=8.0`, `MARGIN_LAMBDA=1.0`, `NUM_STEPS=2400`,
  `BATCH_SIZE=160`, `GRAD_ACCUM=2`, `LR=1e-5`, cosine schedule, fp16.
- **No count-weighted sampler** — the batch composition comes purely from the
  prep-time 2:1 class balance.

---

## 2. Method 2x — dataset-balanced, thresholdless prep + per-step masked loss + sampler

### Data preparation (`TOTAL_RUN_maskloss_v2/prepare_total.py`)
Motivation: mTSBench has **two independent imbalances** needing two different tools.
- **Class imbalance** (normal vs anomaly *steps*) → handled at **runtime** by the
  count-weighted sampler, **not** at prep. So **no class balancing / no thresholding
  is done here.**
- **Dataset imbalance** (MITDB+SVDB ≈ 88% of raw windows) → **must** be handled at prep:
  - `PER_DATASET_CAP=5000` — hard cap on the number of train windows per dataset.
  - `MAX_ANOM_FRAC=0.5` — anomalies take at most 50% of each dataset's cap; both
    classes are guaranteed present per dataset.
  - Result: a dataset-de-dominated pool where the full natural class mix is retained
    (no windows thrown away for class reasons).

### Training (`TOTAL_RUN_maskloss_v2/finetune_anomaly_simple.py`)
**Per-step masked margin (hinge) loss** — per-timestep `future_labels` (0=normal /
1=anomaly) mask the model's per-step pinball loss:

```
L_total = L_good + lambda * L_bad_term
  normal steps (label 0) -> L_good     : minimise
  anomaly steps (label 1) -> L_bad_term : push UP toward a margin, then saturate
```

Three "team suggestions" (the maskloss-v2 additions), with the settings used:
1. **`HINGE_MODE=per_step`** — hinge *each* anomaly step
   `mean_i max(0, margin - per_step_i)` instead of hinging only the pooled mean. Because
   VUS is per-timestep, this stops a few huge-error steps satisfying the margin for the
   rest.
2. **`MARGIN_MODE=relative`** — `margin = margin_m * L_good_w.detach()` per window
   (`MARGIN_M=5` for combined mTSBench), so the bad-target self-scales to each series'
   own normal error instead of a fixed `tau` that over-pushes quiet series and is
   pre-satisfied on busy ones.
3. **`SAMPLING_TARGET=0.4`** — **count-weighted train sampler**. Anomaly steps are rare
   (~14% of forecast steps), which starves the bad term under uniform sampling. Each
   window is drawn with probability proportional to its anomaly-**step** count:
   `weight_i = n_anom_i + eps * H` (`H` = prediction length). `eps` is not hand-tuned;
   it is solved once (binary search) so the expected anomaly-step fraction per batch
   equals `sampling_target` (0.4). The `eps*H` floor keeps every window — including
   pure-normal ones — reachable, so `L_good` still trains. **Thresholdless**, no data
   discarded, no balanced pkl needed. **Validation uses sequential (unbalanced)
   batching**, so eval metrics stay honest.

- **Aggregation:** `AGG_MODE=batch_global` — per-step masked losses are pooled across
  the whole batch (vs `per_window` = per-window means then averaged).
- Hyperparameters: `NUM_STEPS=4000`, `BATCH_SIZE=160`, `GRAD_ACCUM=2`, `LR=1e-5`,
  cosine, LoRA.

---

## 3. Method 3x — FT's data + 2x's loss (both balancing mechanisms stacked)

3x is the **cross / ablation**: it takes **FT's data preparation** and **2x's training
objective**, to isolate the effect of the loss change from the effect of the data change.

- **Data preparation:** `TOTAL_RUN/prepare_total.py` — **identical to FT**
  (anomaly thresholding `=10`, 2:1 class balance `NORMAL_RATIO=2.0`, dataset cap via
  `balance_scope=per_dataset_cap`, `per_dataset_anom_cap=median`).
- **Training:** `SMD_Maskloss_v2/finetune_anomaly_simple.py` — **byte-for-byte identical
  to 2x's** training script (per-step masked hinge, `HINGE_MODE=per_step`,
  `MARGIN_MODE=relative`, `SAMPLING_TARGET=0.4`, `AGG_MODE=batch_global`,
  **`MARGIN_M=5`** — the mTSBench override, matching 2x). The prepared dir was pointed
  at `TOTAL_RUN`'s `prepared_total`.
- **Inference:** the shared `forward.py`.

**What makes 3x distinct from 2x:** 3x **stacks both balancing mechanisms** —
prep-time 2:1 class balancing (from FT's prep) **and** the runtime count-weighted
sampler — whereas 2x relies on the runtime sampler alone (no prep class balancing).
The loss/sampler settings are otherwise identical to 2x (including `MARGIN_M=5`), so
3x vs 2x is a clean data-prep-only comparison.

---

## 4. Method HS — hierarchical sampling (no threshold, no cap, nothing discarded)

3x gives the best overall results, but it obtains its 2:1 normal-to-anomalous *window*
guarantee from FT's **window-level anomaly threshold** (`future_labels.sum() >= 10`). A
threshold that collapses a 64-step label vector into one bit is exactly what the per-step
masked loss was introduced to avoid — so 3x's data and 3x's objective disagree about what
an "anomaly" is. HS removes the threshold while keeping (and strengthening) the guarantee.

### The two imbalances, and why one weight vector cannot fix both

- **Dataset imbalance:** MITDB + SVDB are 87.5% of all raw train windows.
- **Class imbalance:** anomaly *steps* are 12.5% of all forecast steps.

2x attacked the class problem with a **single global** weight vector
`w_i = n_anom_i + eps·H` and the dataset problem with a prep-time cap. But the global
vector reaches across dataset boundaries and silently re-skews the mix the cap had just
fixed. Solving `eps` for `sampling_target=0.4` on 2x's own capped pool yields effective
draw shares of **1.35×** pool share for MITDB / SVDB / cicids and **0.68× / 0.36×** for
GHL / OPPORTUNITY. 3x inherits the same sampler, applied to a pool FT had *already*
class-balanced — so 3x is double-balanced, on a small, pre-distorted pool.

### Data preparation (`TOTAL_RUN_maskloss_v2_HS/prepare_total.py`)

Does nothing but carve windows. **No cap, no threshold, no class balancing, nothing
discarded.** Writes the pool **grouped per dataset** —
`prepared_total/per_dataset/<DS>/train_model_inputs.pkl` plus a `train_n_anom.npy`
sidecar — because the sampler needs level-1 index groups, and a flat pkl carries no
dataset identity. Pool: **340,523 windows, 56,723 anomaly-bearing (16.7%), 6.6 GB**, over
the same **12** train-contributing datasets. Test halves are **symlinked** from
`TOTAL_RUN_maskloss_v2/prepared_total` (byte-identical: same seeded file split, same
geometry), which makes identical evaluation a fact rather than a claim.

Prep hard-fails if any dataset's train half has zero anomaly windows (level 2's anomalous
branch would be undrawable) and warns below 50. All 12 clear the hard check; room-occupancy
(29) trips the warning.

### Training (`TOTAL_RUN_maskloss_v2_HS/finetune_anomaly_simple.py`)

Identical to 2x in **every** respect — loss, margin, LoRA, optimizer, `NUM_STEPS=4000`,
`BATCH_SIZE=160`, `GRAD_ACCUM=2`, `MARGIN_M=5`, `AGG_MODE=batch_global` — **except** that
`AnomalyChronos2Dataset._generate_train_batches` draws each window in three levels:

```
level 1   k    ~ Uniform(K = 12 datasets)              → dataset imbalance
level 2   kind ~ Bernoulli(p_anom = 1/3)               → class imbalance, 2:1 in expectation
level 3   i    ~ Categorical within dataset k, weight
                    n_anom_i        if kind == anomalous
                    64 - n_anom_i   if kind == normal
```

`SAMPLING_TARGET` and its binary-searched `eps` are **gone**; `P_ANOM` is the only knob,
and level 1 has none.

**Level 3 is thresholdless in both branches.** The anomalous branch is *self-gating* — a
pure-normal window has `n_anom = 0`, hence weight 0 (verified: 0 pure-normal windows drawn
from the anomalous branch in 200,000 draws). The normal branch *down-weights* an
anomaly-bearing window by its remaining `64 - n_anom` normal steps rather than excluding
it. This is the "soft" reading: nothing is discarded and no cutoff exists anywhere.

**Level 3 renormalizes within the chosen dataset**, which is what makes the guarantee hold
per dataset rather than only on average. Measured on the real pool, natural anomaly-step
rates spanning **1.0% (GHL) → 27.0% (cicids)** all map onto **27–33%**; pool-wide
**12.47% → 31.31%** analytic, 31.34% Monte-Carlo over 600k draws. Realized level-1 draw
shares are 8.27–8.38% against a uniform target of 8.33%, for a pool whose natural shares
run from 54.82% (MITDB) to 0.03% (room-occupancy).

**Level 1 is redrawn per window, not per group.** `BATCH_SIZE` counts channel *rows*
(`Σ group_size`), and mean `F` ≈ 24.7, so a 160-row batch holds ~7.3 windows from ~5.6
distinct datasets. Drawing the dataset once per 3-window group would give ~2.3 — and
`AGG_MODE=batch_global` pools the loss across the whole batch, so batch diversity is what
keeps the gradient from being single-dataset.

### Known caveat (shared with 2x and 3x, not introduced by HS)

Level 1 balances **draws**, not **gradient**. Each drawn window contributes `F_d` rows of
per-step loss, so under `batch_global` a dataset's share of the gradient is `F_d / ΣF`
(ΣF = 296) — cicids (`F=72`) is 24.3% of the gradient, MITDB and SVDB (`F=2`) are 0.68%
each. 2x and 3x have the same structure (non-uniform draw shares, *then* scaled by `F_d`).
`AGG_MODE=batch_global` is retained so that **HS vs 2x is a strict one-variable ablation**;
`AGG_MODE=per_window` would equalize it and is the natural follow-up arm.

### Inference

The shared `forward.py`, byte-identical (md5 `a226d1f5e899d7ae332112e3f29d076f`), run over
symlinks to the same test pkls 2x and FT evaluated on (md5-verified per dataset).

---

## 5. Summary comparison

| Axis | **FT** | **2x** | **3x** | **HS** |
|------|--------|--------|--------|--------|
| Class balancing (prep) | ✅ thresholded 2:1 (`thr=10`, `ratio=2.0`) | ❌ (runtime sampler instead) | ✅ thresholded 2:1 (`thr=10`, `ratio=2.0`) | ❌ (runtime, levels 2–3) |
| Dataset cap (prep) | ✅ anom → median (`per_dataset_cap`) | ✅ 5000/ds, anom ≤ 50% (`PER_DATASET_CAP`) | ✅ anom → median (`per_dataset_cap`) | ❌ **none — all 340,523 windows kept** |
| Window-level anomaly threshold | ✅ `≥10` steps | ❌ | ✅ `≥10` steps (inherited) | ❌ **none anywhere** |
| Loss | window-level hinge | per-step masked hinge | per-step masked hinge | per-step masked hinge |
| Margin | absolute `tau=8.0` | relative `margin_m=5` | relative `margin_m=5` | relative `margin_m=5` |
| Hinge granularity | pooled (window) | `per_step` | `per_step` | `per_step` |
| Sampler | ❌ uniform | ✅ global count-weighted, `target=0.4` | ✅ global count-weighted, `target=0.4` | ✅ **hierarchical**, `p_anom=1/3` |
| Dataset balance mechanism | prep cap | prep cap (*undone by sampler*: 1.35× / 0.68×) | prep cap (*undone by sampler*) | **level 1, exact** (8.27–8.38% vs 8.33%) |
| Loss aggregation | — | `batch_global` | `batch_global` | `batch_global` |
| Train steps | 2400 | 4000 | 4000 | 4000 |
| Train pool | ~471 MB | 40,521 win / 2.9 GB | ~471 MB | 340,523 win / 6.6 GB |
| Inference | shared `forward.py` | shared `forward.py` | shared `forward.py` | shared `forward.py` (symlinked test pkls) |

**Interpretation of the ablation:**
- **FT → 3x** isolates the effect of the **loss/sampler** change (same data).
- **2x → 3x** isolates the effect of the **data-prep** change (same loss/sampler):
  i.e. whether adding prep-time 2:1 class balancing on top of the runtime sampler helps.
- **2x → HS** isolates the effect of the **sampler** change (same loss, same
  hyperparameters, only `AnomalyChronos2Dataset` differs): whether decomposing the two
  imbalances into independent levels beats a single global weight vector.
- **3x → HS** asks whether the 2:1 window guarantee survives the removal of the
  window-level threshold that produced it — i.e. whether 3x's gain came from the balance
  or from the thresholding.

---

## 6. Reproducibility — how each method is launched

```bash
# ---- FT ----
bash TOTAL_RUN/run_prepare_total.sh        # thresholded 2:1, dataset cap=median
bash TOTAL_RUN/run_finetune_total.sh       # window-level hinge, tau=8, no sampler
bash TOTAL_RUN/run_forward_total.sh        # eval fine-tuned ckpt per dataset

# ---- 2x ----
bash TOTAL_RUN_maskloss_v2/run_prepare_total.sh    # dataset cap 5000, thresholdless
bash TOTAL_RUN_maskloss_v2/run_finetune_total.sh   # per-step maskloss + sampler 0.4
bash TOTAL_RUN_maskloss_v2/run_forward_total.sh    # eval (identical forward.py)

# ---- 3x ----
bash TOTAL_RUN/run_prepare_total.sh                # FT's data prep (reused)
PREPARED_DIR=.../TOTAL_RUN/prepared_total \
  bash SMD_Maskloss_v2/run_finetune_smd.sh         # 2x's maskloss training on FT data
bash TOTAL_RUN/run_forward_total.sh                # eval (identical forward.py)

# ---- HS ----
bash TOTAL_RUN_maskloss_v2_HS/run_prepare_total.sh   # uncapped per-dataset pool;
                                                     # test pkls symlinked from 2x
bash TOTAL_RUN_maskloss_v2_HS/run_finetune_total.sh  # per-step maskloss + HS sampler (p_anom=1/3)
bash TOTAL_RUN_maskloss_v2_HS/run_forward_total.sh   # eval (identical forward.py)
```

All four use `debug_chronos`: `export PATH="/home/rajib/miniconda3/envs/debug_chronos/bin:$PATH"`.

Results land in `TOTAL_RUN/results_finetuned/`, `TOTAL_RUN_maskloss_v2/results_FT/`,
`TOTAL_RUN_maskloss_v2_HS/results_FT/`, and the corresponding 3x output — aggregated per
method into `ALL_results.csv`.

**Watching an HS run.** The startup log prints each dataset's natural → HS anomaly-step
fraction. Every 500 batches the sampler prints its *realized* anomaly-step fraction and its
*realized dataset mix*:

```
[hs] realized over 500 batches: anomaly-step fraction 31.2% (expected 31.3%);
     dataset mix (uniform target 8.3%): MITDB=8.4%  SVDB=8.3%  cicids=8.4%  ...
```

If the dataset mix drifts, level 1 is broken; if the anomaly-step fraction drifts, level 3
is. A step with no `anomaly_loss` key is expected in ~4.5% of micro-batches (a batch with
no anomaly steps at all) — `n_anom_steps.clamp(min=eps)` handles it, the batch just
contributes `L_good`.
