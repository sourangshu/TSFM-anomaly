# Unified_Normal_Total_run_HS — HS fine-tuning with a UNIFIED per-dataset normal prefix

Whole-mTSBench Chronos-2 anomaly fine-tuning that combines two previously separate
ideas:

1. **The HS arm** (`../TOTAL_RUN_maskloss_v2_HS/`): uncapped, thresholdless,
   per-dataset prep + a **hierarchical sampler** that fixes dataset imbalance
   (level 1: uniform dataset draw) and class imbalance (levels 2–3: kind draw +
   count-weighted window draw) at train time.
2. **The unified-normal idea** (`../SMD_run/Unified_Normal/`, previously SMD-only):
   instead of each series carrying its **own** 256-step normal-signal prefix, every
   series of a dataset shares **one global normal shape** — the dataset **medoid**
   — re-scaled into each series' own per-channel units.

This arm is `TOTAL_RUN_maskloss_v2_HS` with **exactly one variable changed: the
normal prefix in the data**. Trainer, sampler, loss, hyperparameters, evaluator and
aggregator are identical (md5-matched copies), so any score difference against the
HS arm is attributable to the unified prefix alone.

## The one change, precisely

Each training/test example is `[ normal_signal (256) | context (512) | future (64) ]`
with per-step `future_labels (64,)`.

| | `../TOTAL_RUN_maskloss_v2_HS` | **this dir** |
|---|---|---|
| normal prefix | carved from **that series' own** normal zones | the **dataset medoid's** z-normalized shape, re-scaled by that series' normal-zone mean/std |
| context / future / labels / metadata | — | **byte-identical** (same primitives, same seed-42 file split) |
| test-only datasets (7, single file) | per-series prefix | per-series prefix (**unchanged** — no training files → no leakage-free medoid) |

### How the unified prefix is built (per dataset)

Implemented in `build_unified_signal.py` (the same medoid machinery validated on
SMD in the earlier `SMD_run/Unified_Normal` experiment):

1. For every **training** file of the dataset, carve its `(F, 256)` normal signal
   from its normal zones and z-normalize per channel (that series' normal-zone
   mean/std) — compare **shape**, not scale.
2. Score all training pairs with a **phase-robust FFT similarity** (correlation of
   magnitude spectra, channel-averaged; `--metric pearson` available).
3. The **medoid** — the real training series most similar to all the others —
   becomes the dataset's `global_norm`, saved to
   `per_dataset/<DS>/global_normal_signal.npz`.
4. Every series (train **and** test) gets `global_norm * std + mean` prepended,
   using **its own** per-channel normal-zone stats — the same normal shape,
   expressed in each series' units.

**No leakage:** the medoid is selected from training files only. Test series never
influence the shared reference; they only receive it (re-scaled by their own
normal-zone stats — the same information the per-series prep already used).
A single-training-file dataset (room-occupancy) degenerates gracefully: the medoid
is that file.

**Why per dataset, not one for all of mTSBench:** channel counts differ (MITDB
F=2 … cicids F=72), so a single `(F, 256)` shape cannot exist. Within every mTSBench
dataset the channel count is uniform (verified), so one medoid per dataset works
everywhere.

**Why test pkls can't be symlinked from other arms** (the HS arm linked them from
v2): the 256-step prefix now differs for the 12 trained datasets, so this prep
re-carves them. Only the 7 test-only datasets (CalIt2, creditcard, GECCO, Genesis,
metro, PSM, swan) stay byte-identical, and `LINK_TEST_FROM` — if set — links just
those. Default is empty: everything is carved here, fully standalone.

## Standalone by construction

Deleting any sibling folder in `rajib_work_space` does not break this one:

- `prep_common.py` — the carving/windowing primitives, copied **verbatim** from
  `SMD_run/prepare_smd_split.py` (plus the two re-scaling helpers from
  `SMD_run/Unified_Normal/prepare_global_normal.py`). Verbatim means windows built
  with a per-series prefix are byte-identical to the other arms'.
- The only external dependencies are `../chronos` (the local package) and
  `../VUS_ROC_VUS_PR` (the metrics package `forward.py` imports) — both shared,
  unchanged, across every experiment arm and resolved via the `PYTHONPATH` the
  runners set, exactly as in every other arm.

## Files

| File | Role | Relation to `../TOTAL_RUN_maskloss_v2_HS/` |
|---|---|---|
| `prep_common.py` | standalone windowing/carving primitives | verbatim from `SMD_run/prepare_smd_split.py` |
| `build_unified_signal.py` | per-dataset medoid derivation | from `SMD_run/Unified_Normal`, imports made local |
| `prepare_total.py` / `run_prepare_total.sh` | uncapped per-dataset prep + **unified prefix** | prefix logic swapped; split/bookkeeping verbatim |
| `finetune_anomaly_simple.py` | HS sampler + per-step masked hinge trainer | **byte-identical** (md5 `65a9434140eb9efcd9541fff7a40ac84`) |
| `run_finetune_total.sh` | same knobs; checkpoint name differs | |
| `forward.py` | anomaly evaluator | **byte-identical** (md5 `a226d1f5e899d7ae332112e3f29d076f`) |
| `run_forward_total.sh` | loops `forward.py` over every per-dataset test set | only the default `CHECKPOINT` path differs |
| `aggregate_results.py` | per-series CSVs → one VUS-PR per dataset | **byte-identical** (md5 `9b591ef8fb3e67c02637875615263fd4`) |

## Data layout

```
prepared_total/
  manifest.json                       # adds unified_reference / unified_mean_similarity per dataset
  per_dataset/<DATASET>/
    global_normal_signal.npz          # the dataset's normalized medoid shape (trained DS only)
    train_model_inputs.pkl            # UNCAPPED train half, UNIFIED prefix (absent for test-only DS)
    train_n_anom.npy                  # int16 (N,) anomaly-step count per window
    test_model_inputs.pkl             # ordered + metadata; UNIFIED prefix (per-series for test-only DS)
    test_series_meta.pkl
```

Same seed (42) and test fraction (0.5) as all other arms → identical train/test
file assignment; results are directly comparable arm-to-arm.

## Environment

Same as the HS arm — one env for all three stages (server):

```bash
export PATH="/home/rajib/miniconda3/envs/debug_chronos/bin:$PATH"
```

(The runners set `PYTHONPATH` to `rajib_work_space` themselves so the local
`chronos` package is used.)

## How to run

### 1. Prepare (per-dataset pools + per-dataset medoid prefixes)

```bash
cd Unified_Normal_Total_run_HS
bash run_prepare_total.sh                          # everything carved here (standalone)
DATASETS="SMD MSL SMAP" bash run_prepare_total.sh  # subset
METRIC=pearson bash run_prepare_total.sh           # alternative medoid similarity
# on WSL:
DATA_ROOT="/mnt/c/Files/MTP Code Local Files/MTP_SEM_3_LOCAL_FILES/mTSBench" bash run_prepare_total.sh
```

Prep logs each dataset's chosen medoid and its mean similarity to the other
training files — inspect these before committing to a run. `auto` always returns
*a* medoid, but a low mean similarity means that dataset's normals are mutually
dissimilar, so one shared signal is a coarse fit there (SMD's medoid scored 0.436)
and per-series — or clustered — signals may serve it better.

Same hard/soft checks as the HS arm: fails loudly if a trained dataset has zero
anomaly windows; warns below `MIN_ANOM_WINDOWS=50`.

### 2. Fine-tune

```bash
bash run_finetune_total.sh                       # P_ANOM=1/3, MARGIN_M=5, NO_VALIDATION=1
DEBUG=1 bash run_finetune_total.sh               # 50 windows PER DATASET, smoke test
```

Checkpoint → `chronos2-single-stage_mtsbench_maskLossv2_HS_UN_v1/finetuned-ckpt`.
All training-side knobs and diagnostics are the HS arm's — see its README for the
sampler math, batch semantics, gradient-share caveat and NUM_STEPS discussion.

### 3. Evaluate

```bash
bash run_forward_total.sh                        # fine-tuned, all datasets -> results_FT/
CHECKPOINT="" bash run_forward_total.sh          # zero-shot baseline    -> results_ZS/
DATASETS="SMD MSL" bash run_forward_total.sh     # subset
```

Produces `results_<FT|ZS>/<DATASET>_results.csv`, the deliverable
`per_dataset_summary.csv`, and the printed **MACRO** VUS-PR. Compare directly
against `../TOTAL_RUN_maskloss_v2_HS/results_FT/per_dataset_summary.csv` — same
evaluator, same test windows up to the prefix, same file split.

## Config summary (differences from the HS arm only)

| Knob | Default | Where | Note |
|---|---|---|---|
| `METRIC` | `fft` | prep | per-dataset medoid similarity (`fft` phase-robust / `pearson`) |
| `LINK_TEST_FROM` | *(empty)* | prep | optional; links **test-only** datasets' pkls only |
| `OUTPUT_DIR` | `chronos2-single-stage_mtsbench_maskLossv2_HS_UN_v1` | finetune | new checkpoint name |
| everything else | — | — | identical to `../TOTAL_RUN_maskloss_v2_HS` |
