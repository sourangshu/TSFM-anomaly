# TOTAL_RUN — whole-mTSBench Chronos-2 anomaly fine-tuning

Scales the single-dataset SMD pipeline (`../SMD_run/`) to **all of mTSBench** while
keeping the per-window format byte-for-byte identical, so training and the simple
forward-inference evaluation work unchanged.

* **One combined train set** pooled from the *train half* of every multi-file dataset.
* **Per-dataset test sets**, so evaluation runs one dataset at a time exactly like
  the old SMD `run_forward_smd.sh`.

The carving / sliding-window / per-series-normal-signal primitives are **imported
verbatim** from `../SMD_run/prepare_smd_split.py` — behaviour matches SMD exactly.

---

## Per-window layout (unchanged from SMD)

```
[ normal_signal (256) | context (512) | future (64) ]   + future_labels (64,)
```

* `normal_signal` is the **per-series** normal prefix — each series carves its own
  normal zone (NOT the global/unified signal from `../SMD_run/Unified_Normal`).
* `future_labels` is the per-timestep 0/1 ground truth for the 64-step future.
* A window is **anomalous** iff `future_labels.sum() >= anomaly_threshold` (default
  10) — the SAME rule the trainer applies in `derive_future_type`.

---

## Data split (per dataset)

mTSBench has 19 datasets under `/home/rajib/mTSBench/Datasets/mTSBench`. We only use
the `*test.csv` files (the `*train.csv` files have ~no anomalies).

* **≥ 2 `*test.csv` files** → file-based **50/50** split (seeded). Half the files →
  train pool, half → that dataset's test set. No window from a CSV appears in two
  splits. *(12 datasets: cicids, Daphnet, Exathlon, GHL, GutenTAG, MITDB, MSL,
  OPPORTUNITY, room-occupancy, SMAP, SMD, SVDB.)*
* **Exactly 1 `*test.csv` file** → **TEST-ONLY**: the single file goes entirely to
  that dataset's test set and contributes **nothing** to the combined train set (a
  file-based split needs ≥ 2 files). *(7 datasets: CalIt2, creditcard, GECCO,
  Genesis, metro, PSM, swan.)*

---

## Class balancing of the combined train set

Anomalous windows are rare, so we keep them and subsample the abundant normal
windows. Controlled by `--balance_scope` / `--normal_ratio`:

| scope | what it does |
|---|---|
| `global` | keep ALL anomalous windows; subsample normals globally to `ratio × total_anom`. |
| `per_dataset` | per dataset, keep all its anomalous windows; subsample its normals to `ratio × anom_d`. Fixes class balance but anomaly-rich datasets still dominate. |
| **`per_dataset_cap`** (default) | cap each dataset's anomalous contribution to `--per_dataset_anom_cap` (default = median anomalous count across datasets) BEFORE matching normals to `ratio × kept_anom_d`. **Equalises each dataset's vote** so big datasets (SVDB/MITDB) can't swamp small ones, while never starving rare-anomaly datasets (the cap only trims datasets that have MORE anomalies than it). |

* **`--normal_ratio`** (default **2.0**): normals per anomalous window. 2:1 keeps a
  normal majority (strong forecasting prior) while still ~10× over-representing
  anomalies vs their natural <5% rate. `L_good`/`L_bad` are per-class **means**, so
  changing this ratio does **not** require retuning λ.
* Memory is bounded by reservoir-sampling the (abundant) normals; the (rare)
  anomalous windows are kept in full.

---

## Outputs (`prepared_total/`)

```
prepared_total/
  train_model_inputs.pkl            # COMBINED, balanced, shuffled  {target,(F,832); future_labels,(64,)}
  val_model_inputs.pkl              # OPTIONAL probe (only if VAL_FRACTION>0) — a COPY of a subset of train
  manifest.json                     # per-dataset split + window counts + balance stats
  prepare_total.log
  per_dataset/
    <DATASET>/
      test_model_inputs.pkl         # ORDERED + metadata (series_id, future_start/end, series_length)
      test_series_meta.pkl          # {length, labels (per-timestep), n_features, context_length}
```

The combined train mixes datasets with **different feature counts (2..72)**. This is
fine: `Chronos2Dataset` flattens each window into per-variate rows tagged by
`group_ids` and pads only along time (all contexts are 768), so variable-variate
windows batch together.

---

## How to run

Use the `debug_chronos` conda env (has pandas/numpy/tqdm/torch/chronos):

```bash
conda activate debug_chronos
cd .../rajib_work_space/TOTAL_RUN
```

### 1. Prepare data

```bash
./run_prepare_total.sh
# common overrides:
VAL_FRACTION=0.02 ./run_prepare_total.sh           # + a 2% training-loss probe (capped at VAL_MAX)
BALANCE_SCOPE=global NORMAL_RATIO=1.0 ./run_prepare_total.sh
DATASETS="SMD MSL SMAP" ./run_prepare_total.sh     # subset (debugging)
```

### 2. Pick the number of training steps

Training samples windows **uniformly at random with replacement** from an infinite
iterable, and `batch_size` counts **variate-rows, not windows**. So:

```
windows_per_step = batch_size * grad_accum / mean_F
num_steps(K epochs) = K * total_rows / (batch_size * grad_accum)
```

`estimate_steps.py` reads the real pkl and prints the exact figures + coverage
(`1 - e^-K`):

```bash
python estimate_steps.py --batch_size 160 --grad_accum 2 --epochs 3 5 10
```

> With the current data (N≈6.4k windows, mean_F≈23): 3 ep ≈ 1.4k steps (95%),
> 5 ep ≈ 2.3k steps (99.3%), 10 ep ≈ 4.6k steps (99.99%). We use **~2400 steps (≈5
> epochs)**. 10 epochs replays each rare anomaly window ~10× → overfitting risk, so
> prefer 5 with checkpointing unless the loss is still improving.

### 3. Fine-tune

```bash
NUM_STEPS=2400 ./run_finetune_total.sh
# with the validation probe (requires VAL_FRACTION>0 in step 1):
NUM_STEPS=2400 NO_VALIDATION=0 ./run_finetune_total.sh
```
Checkpoint → `chronos2-single-stage_TOTAL_v1/finetuned-ckpt`.

Key hyperparameters (margin/hinge loss `L = L_good + λ·max(0, τ − L_bad)`):

* **`MARGIN_TAU` (τ, default 8.0)** — the margin the anomaly loss is pushed toward.
  Must sit ABOVE the typical normal-window loss (~3–4 on the instance-normalized
  scale). Watch the logs: `anomaly_loss → ~τ` while `normal_loss` stays ~3–4. Too
  high → it degrades normal forecasting (shared weights).
* **`MARGIN_LAMBDA` (λ, default 1.0)** — weight on the anomaly push. Raise to ~1.5 if
  anomalies don't separate; lower to ~0.5 if normal forecasting degrades.
* **`ANOMALY_THRESHOLD`** must match the value used in data prep (both default 10).
* `BATCH_SIZE` is in **rows**; with mean_F≈23 you get only ~6 windows/micro-batch —
  raise it if GPU memory allows for a steadier hinge.

### 4. Evaluate (per dataset)

Reuses `../SMD_run/forward.py` unchanged, once per dataset. Writes
`results/<DATASET>_results.csv` and a concatenated `results/ALL_results.csv`.

```bash
# fine-tuned model (default): uses the [normal|context] prefix it was trained on
./run_forward_total.sh

# ZERO-SHOT baseline: context-only, NO normal prefix (base model can't read it)
CHECKPOINT="" ./run_forward_total.sh

# subset / ablations
DATASETS="SMD MSL" ./run_forward_total.sh
USE_NORMAL_PREFIX=0 ./run_forward_total.sh          # fine-tuned, force context-only (ablation)
```

The normal-prefix default is **mode-aware**: fine-tuned → prefix ON, zero-shot →
prefix OFF. Override with `USE_NORMAL_PREFIX=0/1` for the full 2×2.

---

## Files

| file | role |
|---|---|
| `prepare_total.py` | builds the combined train + per-dataset test sets (+ optional val probe) |
| `run_prepare_total.sh` | wrapper for `prepare_total.py` (all knobs as env vars) |
| `run_finetune_total.sh` | wrapper for `../finetune_anomaly_simple.py` on the combined train set |
| `run_forward_total.sh` | loops the per-dataset test sets through `../SMD_run/forward.py` |
| `estimate_steps.py` | prints `num_steps` for K epochs-equivalent from the real train pkl |

## Caveats

* `val_model_inputs.pkl` is a **subset copy of train** (no data removed) — it tracks
  **training loss on a fixed sample**, NOT generalization. The per-dataset test sets
  are the real held-out measure.
* The window's anomaly threshold used for **balancing** (data prep) and for
  **future_type** (training) must agree, or the balance won't reflect training labels.
