# TOTAL_RUN_v2 — whole-mTSBench fine-tuning, SMD strategy, **no data balancing**

This scales the **SMD_run** pipeline (which gave +4% VUS-PR) to the *entire*
mTSBench benchmark, with the data-balancing and per-file window caps of the first
`../TOTAL_RUN` attempt **removed**. The per-window recipe is byte-for-byte identical
to `SMD_run/prepare_smd_split.py`:

```
target = [ normal_signal (256) | context (512) | future (64) ]   + future_labels (64,)
```

The `normal_signal` prefix is **per-series**: every window of a `*test.csv` file is
prefixed with a reference normal signal carved from *that file's own* normal zones
(`extract_normal_signal`). This is the approach that worked for SMD — **not** the
global/unified normal signal.

## What changed vs `../TOTAL_RUN` (v1, unconvincing results)

| | v1 (`TOTAL_RUN`) | **v2 (this dir)** |
|---|---|---|
| Train windows | class-**balanced**: kept all rare anomaly windows, reservoir-subsampled normals to `normal_ratio×anom`, capped each dataset's anomaly count to the median | **every** train-half window kept — no balancing, no reservoir, no cap |
| Margin τ | 8.0 | **6.0** (SMD value) |
| Result | poor | replicates SMD recipe at scale |

Everything else (file-based 50/50 split, per-dataset test sets, margin/hinge LoRA
fine-tuning, `forward.py` evaluation) is the same.

## The 50/50 file split

For each dataset, its `*test.csv` files (anomalies live here) are split file-wise:

* **≥2 files** → seeded 50/50: half → combined train pool, half → that dataset's
  test set. No CSV appears in two splits.
* **1 file** (CalIt2, creditcard, GECCO, Genesis, metro, PSM, swan) → **test-only**:
  the single file goes entirely to test, contributes nothing to training.

Across mTSBench (351 `*test.csv`, 19 datasets) this yields ~172 train-half files →
~340K train windows (~7 GB `train_model_inputs.pkl`; ~15 GB peak RAM during prep).

## Run it

Activate an env with the local `chronos` + `VUS_ROC_VUS_PR` + `peft` (e.g.
`conda activate chronos_clean`), then from this directory:

```bash
# 1. Prepare — combined train pkl + per-dataset test sets  (no balancing)
./run_prepare_total.sh
#    -> prepared_total/train_model_inputs.pkl
#    -> prepared_total/per_dataset/<DATASET>/{test_model_inputs.pkl, test_series_meta.pkl}
#    -> prepared_total/manifest.json   (per-dataset split + window counts)

# 2. Fine-tune — margin/hinge LoRA on the combined train pkl (SMD hyperparameters)
./run_finetune_total.sh
#    -> chronos2-single-stage_TOTAL_v2/finetuned-ckpt

# 3. Evaluate — per-dataset VUS-PR/ROC etc. with the fine-tuned checkpoint
./run_forward_total.sh
#    -> results_finetuned/<DATASET>_results.csv, results_finetuned/ALL_results.csv
#    Prints mean VUS-PR/ROC/AUC across all test series.

# Zero-shot baseline for comparison:
CHECKPOINT="" RESULTS_DIR=./results_zeroshot ./run_forward_total.sh
```

Useful overrides (all env-var driven):

```bash
DATASETS="SMD MSL SMAP" ./run_prepare_total.sh    # subset (also works for step 3)
NUM_STEPS=8000 ./run_finetune_total.sh            # more steps for the larger set
FINETUNE_MODE=full ./run_finetune_total.sh        # full fine-tune instead of LoRA
```

## Geometry must stay consistent across all three steps

* `NORMAL_SIGNAL_LENGTH=256`, data `CONTEXT_LENGTH=512`, `PREDICTION_LENGTH=64`.
* Trainer `CONTEXT_LENGTH=768 = 256 + 512` (normal prefix + real context).
* `STRIDE=64` for **both** train and test — test windows **must** tile contiguously
  (`stride == prediction_length`) so `forward.py` can reassemble a per-timestamp
  anomaly score for VUS-PR.
* `ANOMALY_THRESHOLD=10` (a window is "anomalous" iff ≥10 of its 64 future steps are
  anomalous) must match between prep-stats logging and the trainer's `derive_future_type`.

## Files

| File | Role |
|---|---|
| `prepare_total.py` | data prep — per-series normal prefix, 50/50 file split, **no balancing** |
| `run_prepare_total.sh` | wrapper for step 1 |
| `run_finetune_total.sh` | wrapper for step 2 (reuses `../finetune_anomaly_simple.py`) |
| `run_forward_total.sh` | wrapper for step 3 (reuses `../SMD_run/forward.py`) |
