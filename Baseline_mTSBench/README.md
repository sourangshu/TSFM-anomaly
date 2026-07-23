# mTSBench baselines on the Chronos-2 held-out split

Runs mTSBench's own anomaly detectors on **the same held-out series, over the same
per-series index range, with the same metric code and the same aggregation** as our
Chronos-2 arm — so a baseline number and a Chronos number for the same dataset can go
side by side in a table without a caveat.

**Standalone.** The metric code, the aggregator, and the Chronos reference results are
vendored here (see [VENDORED.md](VENDORED.md)); every path in the code is relative to
the repo, so a clone works anywhere. The only external thing is the ~4.8 GB dataset.

---

## Quick start

```bash
# one-time
conda create -n mtsbench python=3.11 -y
~/miniconda3/envs/mtsbench/bin/pip install -r requirements.txt

# point at the mTSBench CSV tree (skip if it sits somewhere auto-detectable)
export MTSBENCH_DATA=/path/to/Datasets/mTSBench

# see what would run, without running it
LIST=1 ./run_baseline_total.sh

# run one pair
DATASETS="SMD" MODELS="PCA" ./run_baseline_total.sh
```

`covered_regions.json` is committed, so there is no prep step. (Regenerating it needs
the Chronos workspace — see [VENDORED.md](VENDORED.md) — but you never have to.)

### Finding the data

`run_baseline.py` resolves the dataset root in this order: `--data_root` → `$MTSBENCH_DATA`
→ a few locations near the repo. A candidate only counts if it holds dataset sub-directories
containing `*test.csv`.

That last check matters: **the upstream mTSBench clone ships an empty
`mTSBench/Datasets/mTSBench/`** with just a download script in it. Accepting it would
silently yield `MISSING file` on every series, so it is rejected. Every run prints the
`data_root` it settled on — check that line if results look wrong.

---

## Running other models and datasets

`run_baseline_total.sh` takes its job list from `config.py` (`MERGED_MODELS`) and filters
it with two environment variables. **Nothing runs unless it is in that mapping**, so you
cannot accidentally launch the full 67-job sweep — but you can launch it deliberately by
passing no filter.

```bash
# every (dataset, model) pair in config.py — 67 jobs, includes MITDB/SVDB
./run_baseline_total.sh

# all detectors config.py lists for these datasets
DATASETS="SMD MSL SMAP" ./run_baseline_total.sh

# these detectors, on every dataset where config.py lists them
MODELS="PCA IForest" ./run_baseline_total.sh

# exactly one pair
DATASETS="MITDB" MODELS="IForest" ./run_baseline_total.sh

# print the resolved job list and exit — always do this before a long run
LIST=1 DATASETS="SVDB" ./run_baseline_total.sh
```

Names are normalized, so `config.py`'s misspellings and your shorthand both work:
`OAF`→`OFA`, `OnmiAnomaly`/`OmniAnoomaly`→`OmniAnomaly`, `KmeanAD`/`KmeansAD`→`KMeansAD`,
`DONUT`→`Donut`, `IFOREST`→`IForest`, `calit2`→`CalIt2`, `Gecco`→`GECCO`, `Ghl`→`GHL`,
`ROOM-OCCU`→`room-occupancy`.

**For a pair not in `config.py`**, call the inner script directly — it never reads `config.py`:

```bash
~/miniconda3/envs/mtsbench/bin/python run_baseline.py --dataset GHL --model IForest
```

### Resuming

Every completed series is written to the output CSV immediately, and reruns skip series
already present. A killed job costs at most the series in flight.

```bash
DATASETS="MITDB" MODELS="CNN" ./run_baseline_total.sh              # picks up where it stopped
OVERWRITE=1 DATASETS="MITDB" MODELS="CNN" ./run_baseline_total.sh  # recompute from scratch
```

### Progress output

A tqdm bar tracks **series within a job** (`CNN/MITDB: 7/24`), showing the current series,
its length and dimensionality. It goes to **stderr**; per-series result lines go to
**stdout**. So `./run_baseline_total.sh > run.log` gives a clean log *and* a live bar.

The deep detectors (CNN, LSTMAD, USAD, OmniAnomaly, TranAD, Donut, OFA, ALLM4TS) also
print their own per-batch bars from inside mTSBench. The classical ones (PCA, IForest,
MCD, OCSVM, EIF, …) print nothing at all while fitting — on a 520K-step MITDB series
that is a long silence, which is exactly why the series-level bar exists.

### Other knobs

| variable | default | meaning |
|---|---|---|
| `PYTHON` | `~/miniconda3/envs/mtsbench/bin/python` | interpreter |
| `MTSBENCH_DATA` / `DATA_ROOT` | auto-detected | dataset root |
| `RESULTS_ROOT` | `./results_baselines` | output root |
| `CONFIG_PY` | `./config.py` | dataset→detector mapping |
| `LIST` | unset | print job list and exit |
| `OVERWRITE` | unset | recompute instead of resume |
| `SEED` | `2024` | same seed block as `Detectors/main.py` |
| `SLIDING_WINDOW_VUS` / `VUS_VERSION` / `VUS_THRE` | `100` / `opt` / `250` | **do not change** — these are what the Chronos run used |

---

## Reading the results

```
results_baselines/<MODEL>/<DATASET>_results.csv     one row per series, all 9 metrics
results_baselines/<MODEL>/<DATASET>_runlog.json     HP, per-series runtime, skips, failures
results_baselines/<MODEL>/per_dataset_summary.csv   THE DELIVERABLE: one row per dataset
```

`per_dataset_summary.csv` is produced by the vendored `aggregate_results.py`,
**unmodified** — per-dataset mean over series, then a MACRO mean over datasets.

### All 9 metrics

Every layer carries the full set — nothing is filtered to a headline number:

```
VUS-PR   VUS-ROC   AUC-PR   AUC-ROC
Standard-F1   PA-F1   Event-based-F1   R-based-F1   Affiliation-F
```

```bash
# one dataset x detector table per metric, plus an all-metric leaderboard
~/miniconda3/envs/mtsbench/bin/python summarize_baselines.py

# a subset, if you want a shorter read
~/miniconda3/envs/mtsbench/bin/python summarize_baselines.py --metrics VUS-PR VUS-ROC

# long-format CSV: one row per (dataset, source) with all 9 metric columns
~/miniconda3/envs/mtsbench/bin/python summarize_baselines.py --out baseline_vs_chronos.csv
```

---

## Does this need CUDA?

**Some of it uses the GPU; none of it requires one.** The torch detectors hardcode
`cuda=True` and fall back to CPU on their own via `Detectors/utils/torch_utility.get_gpu()`,
which prints the device it chose. There is no flag to enable it.

| GPU (auto-detected, used if present) | CPU only |
|---|---|
| `CNN`, `LSTMAD`, `USAD`, `AutoEncoder`, `OmniAnomaly`, `TranAD`, `Donut`, `OFA`, `ALLM4TS` | `PCA`, `IForest`, `KMeansAD`, `KNN`, `CBLOF`, `HBOS`, `EIF`, `RobustPCA`, `COPOD`, `MCD`, `OCSVM` |

- **No CUDA toolkit install needed.** The PyPI `torch==2.3.0` wheel bundles the CUDA 12.1
  runtime. Verified on a Quadro P5000 (16 GB).
- **To force CPU**, mask the device — the detectors have no CPU switch:
  ```bash
  CUDA_VISIBLE_DEVICES="" DATASETS="SMD" MODELS="CNN" ./run_baseline_total.sh
  ```
- **`OFA` needs network on its first run** — it builds `transformers.GPT2Model`, and
  `gpt2` may not be in the local HF cache.
- **One GPU is a serialization point.** Two deep-detector jobs at once on one 16 GB card
  will contend; CPU-only detectors run alongside them freely.

---

## Dividing the work between machines

The 67 jobs are wildly uneven — **MITDB and SVDB are ~69% of the total**, and the GPU
detectors are ~77% of it. Splitting by job count gives a 5× imbalance. Rough estimates
(anchored on one measurement — PCA/SMD at ~45 s/series — so treat as order-of-magnitude):

| | jobs | est. serial hours |
|---|---:|---:|
| GPU-detector jobs | 31 | ~84 h |
| CPU-only jobs | 36 | ~25 h |

Heaviest single jobs: MITDB `Donut` (~17 h), MITDB `ALLM4TS` (~17 h), SVDB `OmniAnomaly`
(~13 h), MITDB `CNN` (~12 h), SVDB `USAD` (~7 h).

If one machine has the GPU and the other is a big CPU box, split **by device class**
(verified to partition exactly: 60 + 7 = 67):

```bash
# ── CPU box ── 60 jobs. Launch several concurrently; output paths are disjoint.
DATASETS="CalIt2 cicids creditcard Daphnet Exathlon GECCO Genesis GutenTAG metro MSL OPPORTUNITY PSM room-occupancy SMAP SMD swan" ./run_baseline_total.sh
DATASETS="MITDB" MODELS="IForest" ./run_baseline_total.sh
DATASETS="SVDB"  MODELS="PCA IForest" ./run_baseline_total.sh

# ── GPU box ── 7 jobs, ~71 h serial. The critical path.
DATASETS="GHL"   ./run_baseline_total.sh
DATASETS="MITDB" MODELS="CNN Donut ALLM4TS" ./run_baseline_total.sh
DATASETS="SVDB"  MODELS="OmniAnomaly USAD" ./run_baseline_total.sh
```

Only MITDB and SVDB end up split; every other dataset finishes whole on one machine, so
you get complete table rows as you go. **Merging is safe for any split** — each
(dataset, model) owns a unique output path:

```bash
rsync -av <other>/results_baselines/ ./results_baselines/
./run_baseline_total.sh                      # re-aggregates; completed series are skipped
~/miniconda3/envs/mtsbench/bin/python summarize_baselines.py --out baseline_vs_chronos.csv
```

Caveat: **CPU and GPU can shift the torch detectors' numbers slightly** (different
kernels, despite the fixed seed). Each (dataset, model) cell is produced entirely on one
machine so nothing is internally inconsistent, but record which machine produced which
rows.

---

## What is held identical to the Chronos run

| | how |
|---|---|
| **Series** | The 178 held-out series frozen in `covered_regions.json`, derived from the Chronos arm's `prepared_total/manifest.json` (seed 42, 50/50 split over the real `*test.csv` files). Not re-derived — read from the manifest. |
| **Region** | Each series is scored on `[lo, hi)` — the exact slice Chronos forecast. The leading 512 context steps and the trailing `len % 64` remainder are excluded, matching `forward.py`. |
| **Metric code** | `get_metrics` from the vendored `VUS_ROC_VUS_PR`, byte-identical to the Chronos arm's copy and verified to agree to 1e-12. Called with `slidingWindow=100, version='opt', thre=250, pred=None` (oracle threshold). |
| **Skip rules** | A series with 0 positives, or 100% positives, inside `[lo, hi)` is skipped — same as `forward.py`. This is why `n_series` per dataset matches `results_FT`. |
| **Columns** | The same 9 metrics in the same order, so `aggregate_results.py` runs unchanged. |
| **Aggregation** | That same `aggregate_results.py`. |
| **Seed** | 2024, the same block as `Detectors/main.py`. |

### What legitimately differs

Semi-supervised detectors fit on each series' own `*_train.csv`, which is the mTSBench
protocol. That file is normal-only reference data and is independent of our 50/50 split
of the `*test.csv` files — **it is not leakage into the held-out half**. All 179 held-out
test files were verified to have a matching `_train.csv`.

---

## Files

| file | role |
|---|---|
| `config.py` | dataset → detector mapping (you own this) |
| `covered_regions.json` | per-series `(lo, hi)`, length, n_features, positives in region — committed |
| `dump_covered_regions.py` | regenerates the above from the Chronos prep (rarely needed) |
| `run_baseline.py` | one detector × one dataset; the actual work |
| `run_baseline_total.sh` | driver: job list from `config.py`, then aggregation |
| `summarize_baselines.py` | cross-model table vs Chronos, all 9 metrics |
| `aggregate_results.py` | vendored Chronos aggregator — see [VENDORED.md](VENDORED.md) |
| `VUS_ROC_VUS_PR/` | vendored metric implementation — see [VENDORED.md](VENDORED.md) |
| `chronos_reference/` | vendored Chronos-2 results used as the reference column |
| `mTSBench/` | upstream clone (`Detectors/`, unmodified) |

---

## Known issues

- **`ALLM4TS` has no entry in `Optimal_Multi_algo_HP_dict`** (the dict stops at `OFA`).
  Exathlon and MITDB both list it, so those runs use the wrapper defaults
  (`win_size=100, batch_size=64`). `run_baseline.py` prints a NOTE when this happens.
- **No `results_ZS`** — the zero-shot Chronos-2 run has not been done, so the comparison
  shows the FT column only. Produce it with `CHECKPOINT="" ./run_forward_total.sh` in the
  Chronos arm, then copy it to `chronos_reference/results_ZS/`; no code change needed.
- **`UndefinedMetricWarning` from sklearn** during scoring is noise from inside the shared
  `get_metrics` (precision with no predicted positives). The Chronos run emits it too.
- **MITDB and SVDB dominate runtime.** See "Dividing the work" above.

---

## Status

Validated end-to-end:

| detector / dataset | result |
|---|---|
| `PCA` / `SMD` | 9 series, `series_id` set identical to `results_FT/SMD_results.csv`; VUS-PR **0.5388** vs Chronos-2 FT **0.3716** |
| `COPOD` / `room-occupancy` | 1 series, VUS-PR **0.6750** |

The remaining 65 pairs in `config.py` have not been run.
