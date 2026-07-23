# Vendored files — provenance

This directory is standalone: it runs from a fresh clone anywhere, with no path
dependency on the Chronos workspace. That means three things were copied in rather
than referenced. They are **byte-identical copies**, not adaptations — the whole point
of using the Chronos arm's own metric code is that "same metric code" is a fact rather
than a claim, and a modified copy would destroy that.

Source workspace at time of copy:

```
Chronos_Finetuning/rajib_work_space/            (git 253461c, Tue Jul 21 2026)
```

Copied on **2026-07-23**.

---

## 1. `VUS_ROC_VUS_PR/` — the metric implementation

From `rajib_work_space/VUS_ROC_VUS_PR/`. This is what `forward.py` imports, so the
baselines and Chronos-2 compute their 9 metrics with the same code object.

| file | md5 |
|---|---|
| `metrics.py` | `9704ee7480d207dd90d4080ab7321380` |
| `basic_metrics.py` | `7e30c85fafb9c73a52f928233c2789ad` |
| `affiliation/__init__.py` | `81051bcc2cf1bedf378224b0a93e2877` |
| `affiliation/generics.py` | `d4da09a27ae3108743538036b2c374fb` |
| `affiliation/metrics.py` | `c84017f2134b563b9134edf84c09f9e7` |
| `affiliation/_affiliation_zone.py` | `c505cdfd1be5c55cd30c7457310764da` |
| `affiliation/_integral_interval.py` | `1371f332201072b4e85e68c5617dea5f` |
| `affiliation/_single_ground_truth_event.py` | `c7f8eec5c066daf9cecf302a6585f800` |

Verified after copying: on a fixed random input (4000 steps, 5% positives), the
vendored and original `get_metrics` agree to **1e-12 on all 9 metrics**.

### Relationship to mTSBench's own copy

`mTSBench/Detectors/evaluation/basic_metrics.py` is identical to
`VUS_ROC_VUS_PR/basic_metrics.py` once comments and whitespace are normalized, and
mTSBench's `get_metrics` only *appends* extra metrics after computing the same
`AUC_ROC` / `AUC_PR` / `generate_curve` core. Either would give the same 9 numbers;
we use this one so the provenance is unambiguous, and to avoid pulling in the extra
`range_metrics` dependency.

## 2. `aggregate_results.py` — the aggregator

From `rajib_work_space/TOTAL_RUN_maskloss_v2_HS/aggregate_results.py`,
md5 `9b591ef8fb3e67c02637875615263fd4`. Unmodified, so the baseline per-dataset
summaries are produced by the same code that produced the Chronos ones.

## 3. `chronos_reference/results_FT/` — the Chronos-2 numbers

From `rajib_work_space/TOTAL_RUN_maskloss_v2_HS/results_FT/` (20 CSVs: 19 per-dataset
result files + `per_dataset_summary.csv`). These are the fine-tuned Chronos-2 results
that `summarize_baselines.py` shows as the reference column.

**There is no `results_ZS`** — the zero-shot Chronos-2 run has not been done yet. When
it exists, copy it to `chronos_reference/results_ZS/` and `summarize_baselines.py`
picks it up with no code change (it already looks for that directory).

---

## What is deliberately NOT vendored

| | why |
|---|---|
| **The mTSBench dataset** (~4.8 GB) | Too large for a repo. Point at it with `$MTSBENCH_DATA`, or let `run_baseline.py` auto-detect it. Note the upstream clone ships an **empty** `mTSBench/Datasets/mTSBench/` holding only a download script — auto-detection explicitly rejects a directory that has no dataset sub-dirs with `*test.csv`, because silently picking it produced `MISSING file` on every series. |
| **`prepared_total/`** (20 GB, itself full of symlinks) | Only needed to *regenerate* `covered_regions.json`, which is committed. `covered_regions.json` is the complete extract: per-series `(lo, hi)`, length, n_features, positives in region. A clone never needs the Chronos workspace. To regenerate anyway: `export CHRONOS_PREPARED=/path/to/prepared_total && python dump_covered_regions.py`. |

---

## Re-verifying against the source

If the Chronos workspace is reachable and you want to confirm nothing drifted:

```bash
W=../Chronos_Finetuning/rajib_work_space          # adjust to your layout
diff -r VUS_ROC_VUS_PR "$W/VUS_ROC_VUS_PR" -x '__pycache__' && echo "VUS package: identical"
diff aggregate_results.py "$W/TOTAL_RUN_maskloss_v2_HS/aggregate_results.py" && echo "aggregator: identical"
```

If either reports a difference, the Chronos arm changed after 2026-07-23. Decide
deliberately whether to re-copy — re-copying changes what the baseline numbers are
being compared against, and any already-computed results were produced with the old
code.
