# TOTAL_RUN_maskloss_v2 — dataset-balanced, sampler-class-balanced mTSBench fine-tuning

Combined-mTSBench Chronos-2 anomaly fine-tuning that fixes **both** imbalances in the
data, each with the right tool:

| Imbalance | Fixed where | How |
|---|---|---|
| **Class** (normal vs anomaly *steps*) | **runtime** | count-weighted sampler in `finetune_anomaly_simple.py` (`SAMPLING_TARGET`) — thresholdless, per-step |
| **Dataset** (MITDB+SVDB are ~88% of raw windows) | **prep** | per-dataset **cap** in `prepare_total.py` (`PER_DATASET_CAP`) |

These two axes are **decoupled**: prep equalises how much of each dataset enters the
pool; the sampler equalises classes within that pool. Neither tool touches the
other's job.

## Why this exists (vs the neighbouring dirs)

- `../TOTAL_RUN/` — class-balanced at prep via `normal_ratio` + `anomaly_threshold≥10`.
  Now **obsolete**: the per-step sampler does class balance better (no hard threshold).
- `../TOTAL_RUN_v2_without_balancing/` — keeps **every** window → 88% ECG →
  the combined model degenerates into an ECG model. Fixes class (via sampler) but
  **not** dataset dominance.
- **this dir** — per-dataset cap (dataset de-domination) **+** sampler (class). Both axes.

## The per-dataset cap (the only new algorithm)

For each dataset, keep at most `PER_DATASET_CAP` train windows, **guaranteeing both
classes survive**:

- "anomaly window" = **≥1 anomaly step** (thresholdless — matches the per-step view).
- anomalies take at most `MAX_ANOM_FRAC` (default 0.5) of the budget:
  `n_anom = min(#anom, floor(0.5·cap))`
- the rest is filled with normals: `n_norm = min(#normal, cap − n_anom)`
- normal-poor datasets top up with more anomalies so the cap is still used
- datasets already under the cap are kept whole; **natural class ratio otherwise preserved**

Only the two ECG giants actually get anomalies trimmed; every other dataset keeps all
its anomalies plus a large block of normals. Test sets are **never** capped.

With `PER_DATASET_CAP=5000, MAX_ANOM_FRAC=0.5` the combined pool is ~40k windows
(~25% anomaly-windows, ECG down from 88% → ~25%), and the runtime sampler then lifts
anomaly **steps** to `SAMPLING_TARGET` (40%) per batch.

## Files

| File | Role |
|---|---|
| `prepare_total.py` / `run_prepare_total.sh` | dataset-capped, thresholdless prep (new cap logic) |
| `finetune_anomaly_simple.py` | **verbatim copy** of `SMD_Maskloss_v2/` sampler trainer (unchanged) |
| `run_finetune_total.sh` | copy of the SMD runner, retargeted (paths, `MARGIN_M=5`, `SAMPLING_TARGET=0.4`, `NO_VALIDATION=1`) |
| `forward.py` | **verbatim copy** of the SMD forward evaluator (unchanged) |
| `run_forward_total.sh` | loops `forward.py` over every per-dataset test set, then aggregates |
| `aggregate_results.py` | per-series CSVs → **one VUS-PR value per dataset** (+ macro/micro) |

`prepare_total.py` imports the windowing primitives from `../SMD_run/prepare_smd_split.py`
so train/test windows are byte-identical to how SMD and the test sets were built.

## Environment

Use **`debug_chronos`** for all three stages — it has pandas (prep), torch 2.10 +
transformers + peft (train), numpy + chronos (forward):

```bash
export PATH="/home/rajib/miniconda3/envs/debug_chronos/bin:$PATH"
```

(The runners set `PYTHONPATH` to `rajib_work_space` themselves so the local `chronos`
package is used.)

## How to run

### 1. Prepare (dataset-capped combined pool + per-dataset test sets)

```bash
cd TOTAL_RUN_maskloss_v2
bash run_prepare_total.sh                       # PER_DATASET_CAP=5000, MAX_ANOM_FRAC=0.5
# PER_DATASET_CAP=3000 bash run_prepare_total.sh # smaller/faster pool
```

Writes `prepared_total/train_model_inputs.pkl` and `prepared_total/per_dataset/<DS>/`.

**Validation (optional, off by default).** Two independent monitoring sets, exactly
like the old `TOTAL_RUN`:
- **EVAL_VAL** — a *copy* of a random subset of the combined train pool (a loss-curve
  probe; no train data removed). Generate it at prep with `VAL_FRACTION>0`, then train
  with `NO_VALIDATION=0`:
  ```bash
  VAL_FRACTION=0.02 bash run_prepare_total.sh    # writes val_model_inputs.pkl
  ```
- **EVAL_TEST** — stays **manual**: build/point a pkl yourself and pass it as
  `TEST_DATA=/path/to/eval_test.pkl` to `run_finetune_total.sh` (logged as `eval_test_*`).

We run with neither by default (`VAL_FRACTION=0`, `NO_VALIDATION=1`).

### 2. Fine-tune (sampler does class balance; no validation)

```bash
bash run_finetune_total.sh                      # SAMPLING_TARGET=0.4, MARGIN_M=5, NO_VALIDATION=1
# NUM_STEPS=<N> bash run_finetune_total.sh       # see "Choosing NUM_STEPS" below
```

Checkpoint → `chronos2-single-stage_mtsbench_maskLossv2_v1/finetuned-ckpt`.
Watch the startup log for the solved `eps` + expected fraction, and the
`[sampler] realized anomaly-step fraction … (target 40%)` lines (should settle ~40%).

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
- printed **MACRO** VUS-PR (mean of the per-dataset values; the single headline for
  comparing runs) and **micro** (all series pooled; reference only, biased to big datasets).

## Choosing NUM_STEPS

For the default cap (`PER_DATASET_CAP=5000`) the capped pool is **40,521 windows /
911,781 channel-rows** (avg 22.5 channels). With effective batch = `BATCH_SIZE·GRAD_ACCUM`
= 160·2 = 320, **1 epoch ≈ 2,850 steps**.

| NUM_STEPS | ≈ Epochs | Note |
|---|---|---|
| 2,850 | 1.0 | conservative floor / fastest |
| **4,000** | **~1.4** | **default — recommended** |
| 5,700 | 2.0 | upper bound; only if loss still improving |
| 8,550+ | 3+ | **only with the EVAL_VAL probe** (best-ckpt selection) |

With `NO_VALIDATION=1` the *final*-step weights are kept (no best-checkpoint restore),
so overtraining the margin loss is a real risk — stay ≤ ~2 epochs unless you enable
`VAL_FRACTION`. Target 40% anomaly steps is reachable on this pool (baseline 18.4%,
max 73.5%).

## Config summary

| Knob | Default | Where |
|---|---|---|
| `PER_DATASET_CAP` | 5000 | prep (dataset balance) |
| `MAX_ANOM_FRAC` | 0.5 | prep (both classes guaranteed) |
| `SAMPLING_TARGET` | 0.4 | finetune (class balance, anomaly-step fraction/batch) |
| `MARGIN_M` | 5 | finetune (relative-margin multiplier, mtsbench) |
| `NO_VALIDATION` | 1 | finetune (combined pool has no val split) |
| `CONTEXT_LENGTH` | 768 | finetune (= 256 normal + 512 context) |
