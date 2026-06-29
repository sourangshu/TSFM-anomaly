# Unified (Global) Normal-Signal Data Preparation

This folder prepares Chronos-2 anomaly fine-tuning data in which **every series in a
dataset shares one common "normal signal" reference**, instead of each series
carrying its own. The output `*.pkl` files are a drop-in replacement for the
per-series prep and can be pointed at directly by `SMD_run/run_finetune_smd.sh`.

Script: **`prepare_global_normal.py`**

---

## 1. What a prepared window looks like

Each training example is a fixed-length multivariate window with three parts
concatenated along the time axis:

```
[ normal_signal (N) | context (C) | future (P) ]      shape = (F, N + C + P)
```

- `F` — number of channels (sensors), e.g. 38 for SMD.
- `N` — `--normal_signal_length` (default 256): the normal-signal reference prefix.
- `C` — `--context_length` (default 512): real past steps.
- `P` — `--prediction_length` (default 64): real future steps to forecast.

Alongside each window we store `future_labels` — a length-`P` int array with one
label per future step (`0 = normal`, `1 = anomaly`).

The only difference from the per-series prep is **how the `normal_signal` prefix is
produced**: here it is a single global shape, re-scaled per series.

---

## 2. Data preparation steps

The script performs the following, in order.

### Step 1 — File-based train / test / val split
The split is **FILE-BASED** (mirrors `SMD_run/prepare_smd_split.py`): whole CSV files
are assigned to splits, so **no window from a given series ever appears in two
splits**.

1. Discover every `*test.csv` under `--data_dir` and shuffle deterministically with
   `--seed`.
2. Hold out `--test_fraction` of the files as the **test** set (`0` = no test split).
3. From the remaining **training pool**, optionally carve `--val_fraction` of the files
   as a file-based **validation** set (`0` = none); the rest are **training** files.

Using the **same `--seed` and `--test_fraction`** as `prepare_smd_split.py` yields the
**identical file partition**, so the global-normal and per-series experiments are
directly comparable.

### Step 2 — Build the global normal-signal *shape* (from training files only)
A single `(F, N)` reference shape (`global_norm`) is derived. By default
(`--reference auto`) it is found **automatically for whatever dataset you point at —
nothing is hardcoded** (`build_unified_signal.py`), and **only the training files are
considered so the test/val series cannot leak into the shared reference**:

1. For every **training** `*test.csv`, carve its `(F, N)` normal signal from its
   **normal zones** (contiguous stretches with no anomaly label) and **z-normalize it
   per channel** (using that series' normal-zone mean/std) to keep only the *shape*.
2. Score how representative each series is of all the others with a **phase-robust
   similarity** (`--metric fft`: correlation of magnitude spectra; or `pearson`).
3. Select the **medoid** — the real series most similar to all the rest — and use its
   normalized shape as `global_norm`.

The medoid (a real waveform) is used rather than a synthetic average, which would
collapse weakly-related normals into a near-flat, structureless signal.
`global_norm` is saved to `global_normal_signal.npz` for inspection/reuse.

> Overrides: pass `--reference <substring>` to force a specific **training** series
> (e.g. `machine-3-3`; naming a test/val file errors out, by design), or
> `--global_normal_npz <file>` to load a precomputed normalized `(F, N)` shape (key
> `signal`) and skip selection entirely. You can also derive and save the shape on its
> own with `build_unified_signal.py`.

### Step 3 — Build windows per split and attach the re-scaled global signal
For each series (skipping any shorter than `max(--min_length, C + P)` or whose channel
count `F` doesn't match the global shape):

1. Slide a window of stride `--stride` to create `[context | future]` pairs with
   per-step `future_labels` (windows without a full future are dropped).
2. Compute that series' own per-channel normal-zone **mean** and **std**.
3. **Re-scale the global shape into this series' units**:
   `normal_signal = global_norm * std + mean` (per channel) — the same normal shape
   expressed in each series' own scale, matching its context's units.
4. Prepend the re-scaled `normal_signal` to every window of the series.

### Step 4 — Finalize and save
Windows become fixed-length `(F, N + C + P)` targets and are written per split:

- **train / val** are **shuffled** (order irrelevant) and carry only
  `target` + `future_labels`.
- **test** is kept **unshuffled** — grouped by series, in temporal order — and each
  window additionally carries `series_id`, `future_start`, `future_end`,
  `series_length` so the 64-step predictions can be scattered back onto the series
  timeline for series-based metrics (VUS-PR etc.). A companion `test_series_meta.pkl`
  stores each test series' full per-timestamp ground truth.

---

## 3. How the global normal signal is picked (medoid strategy)

This expands Step 2: how `--reference auto` chooses *one* normal signal to represent
the dataset, considering the **training files only** (`M` = number of training series,
each with `F` channels; target prefix length `N`). Implemented in
`build_unified_signal.py`.

### 3.1 Extract one normal signal per (training) series
For each training `*test.csv`:

1. Load it as `(F, T)` features plus the per-step anomaly labels.
2. Find the **normal zones** — contiguous spans whose label is `0`.
3. Carve an `(F, N)` normal waveform from those zones (longest zone first; if a single
   zone is too short, concatenate zones until `N` steps are gathered; left-pad if the
   series still has fewer than `N` normal steps).

This yields `M` candidate signals, each `(F, N)`.

### 3.2 Put every signal on a common ruler (z-normalize per channel)
Different series live on wildly different scales (one channel may sit near 8,000,000,
another near 0.5). Comparing them raw would just rank by magnitude. So each channel of
each signal is **z-normalized using that series' normal-zone mean/std**:

```
signal_norm[f] = (signal[f] - mean[f]) / std[f]        # per channel f
```

with a degenerate (near-zero) `std` clamped to `1`, and any NaN (from padding) set to
`0`. Now every signal is centered at 0 with unit spread — we are comparing **shape**,
not scale.

### 3.3 Measure pairwise similarity (phase-robust)
We build an `M × M` similarity matrix. For each pair of series the similarity is
**averaged over the `F` channels**. Two metrics are available via `--metric`:

- **`fft` (default, recommended):** correlate the **magnitude spectra** of the two
  signals (`|rfft|`, dropping the DC bin). This compares *which rhythms/frequencies are
  present and how strong*, ignoring **where each cycle happens to start**. Two healthy
  series are essentially never phase-aligned, so this phase-robust view is the fair way
  to ask "is this the same kind of normal behaviour?"
- **`pearson`:** plain timestep-by-timestep correlation of the waveforms. Phase
  *sensitive* — two identical-but-shifted normals score ≈ 0 — so it usually
  under-reports real similarity. Provided mainly for comparison.

Each entry is in `[-1, 1]` (`1` = identical shape).

### 3.4 Select the medoid
For every series, take its **mean similarity to all the other series** (its own
diagonal excluded). The **medoid** is the series with the **highest** mean similarity —
i.e. the single real signal that is, on average, most like everyone else:

```
medoid = argmax_i  mean_j≠i  similarity[i, j]
```

Its z-normalized `(F, N)` waveform becomes `global_norm`. Its name and mean-similarity
score are logged (and stored in `global_normal_signal.npz`).

### 3.5 Why the medoid (a real signal) and not a synthetic average
A natural alternative is to **average** all the normalized signals (or build a
DTW barycenter) into one synthetic reference. In practice, when the series are only
weakly correlated, the parts where they disagree cancel out and the average collapses
to a **near-flat, structureless line that resembles no real series** — a poor reference.
The medoid is an actual waveform, so it preserves genuine periodic structure. (The
exploratory comparison that established this lives in `reasoning/`.)

### 3.6 Caveat — does one global signal even fit this dataset?
`auto` always returns *a* medoid, but that does not guarantee a single signal is a
*good* summary. If the dataset's normals are very dissimilar (low off-diagonal
similarity), one global signal is a coarse fit and per-series — or clustered — signals
may serve better. Inspect the spread before committing; the validation tooling in
`reasoning/` reports the off-diagonal similarity distribution for exactly this check.

---

## 4. Outputs

Written to `--output_dir`:

| File | Contents |
|---|---|
| `train_model_inputs.pkl` | training-pool windows, **shuffled**: `{"target": (F, N+C+P) float32, "future_labels": (P,) int32}` |
| `val_model_inputs.pkl`   | same format, shuffled (only when `--val_fraction > 0`) |
| `test_model_inputs.pkl`  | test windows, **ordered**, each also with `series_id`, `future_start`, `future_end`, `series_length` (only when `--test_fraction > 0`) |
| `test_series_meta.pkl`   | dict `series_id → {length, labels (full per-timestamp), n_features, context_length}` for the test files |
| `global_normal_signal.npz` | the normalized global shape `signal (F, N)`, plus `reference`, `normal_signal_length` |
| `log/prepare_global.log` | run log |

---

## 5. Usage

Preferred entry point is the wrapper `run_prepare_global.sh` (env-overridable); the
script can also be called directly:

The wrapper exposes the split via `TEST_FRACTION` and `CREATE_VAL` (like
`run_prepare_smd.sh`):

```bash
bash run_prepare_global.sh                       # 70/30 train/test, no val
TEST_FRACTION=0.6 bash run_prepare_global.sh     # 40/60 train/test
CREATE_VAL=1 VAL_FRACTION=0.1 bash run_prepare_global.sh   # also carve a val set
TEST_FRACTION=0 bash run_prepare_global.sh       # no test split (all training)
```

Or call the script directly (auto-derives the medoid from training files; nothing
hardcoded):

```bash
python prepare_global_normal.py \
    --data_dir /home/rajib/mTSBench/Datasets/mTSBench/SMD \
    --output_dir ./prepared_global \
    --reference auto \
    --test_fraction 0.3 --val_fraction 0.0 \
    --normal_signal_length 256 \
    --context_length 512 \
    --prediction_length 64 \
    --stride 64
```

### Arguments

| Argument | Default | Meaning |
|---|---|---|
| `--data_dir` | `.../mTSBench/SMD` | Directory holding the dataset's `*test.csv` files |
| `--output_dir` | `./prepared_global` | Where the pkls and global signal are written |
| `--reference` | `auto` | `auto` = derive the medoid from training files; or a substring forcing a specific training series |
| `--metric` | `fft` | Similarity for `auto` medoid selection: `fft` (phase-robust) or `pearson` |
| `--global_normal_npz` | `None` | Load a precomputed normalized `(F, N)` shape instead of deriving/selecting |
| `--normal_signal_length` | `256` | `N`, length of the normal-signal prefix |
| `--context_length` | `512` | `C`, context length |
| `--prediction_length` | `64` | `P`, future length |
| `--stride` | `64` | Sliding-window stride |
| `--min_length` | `50` | Discard series shorter than this |
| `--test_fraction` | `0.3` | Fraction of **files** held out for testing (file-based); `0` = no test split |
| `--val_fraction` | `0.0` | Fraction of **training-pool files** used for validation (file-based); `0` = none |
| `--seed` | `42` | RNG seed for the split and shuffle |

> **Comparable splits:** with the same `--seed` and `--test_fraction`, the file
> partition is identical to `SMD_run/prepare_smd_split.py`, so the global-normal and
> per-series experiments train/test on the same series.

> **Consistency with fine-tuning:** the fine-tuning script's `CONTEXT_LENGTH` must
> equal `N + C`, and its `NORMAL_SIGNAL_LENGTH` must equal `N`. With the defaults
> above that is `CONTEXT_LENGTH = 768` and `NORMAL_SIGNAL_LENGTH = 256`. Point
> `PREPARED_DIR` in `run_finetune_smd.sh` at this `--output_dir`.
