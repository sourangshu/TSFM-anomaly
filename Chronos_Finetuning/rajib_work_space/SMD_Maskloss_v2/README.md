# SMD MaskLoss **v2** — Anomaly-Aware Fine-Tuning for Chronos-2

Single-stage, anomaly-aware fine-tuning of Chronos-2 on SMD using a **per-step masked
margin (hinge) loss**. v2 folds in three team suggestions to boost per-timestep
detection (VUS is per-timestep), each reversible to the v1 behaviour via a flag.

```
L_total = L_good + λ · L_bad_term

  normal steps  (label 0) → L_good     : minimise (predict normal well)
  anomaly steps (label 1) → L_bad_term : push prediction error UP toward a margin,
                                          then saturate (self-limiting hinge)
```

At inference, high prediction error ⟹ high anomaly score.

---

## TL;DR — how to run

Everything runs from **this folder** (`SMD_Maskloss_v2/`). Use a Python env that has
`torch` + `chronos` (e.g. the `chronos_clean` conda env).

```bash
# 0) activate an env with torch + chronos
conda activate chronos_clean

# 1) prepare SMD data  → writes prepared_50_50/{train,val,test}_model_inputs.pkl
bash run_prepare_smd.sh

# 2) fine-tune (v2 defaults already encode all three suggestions)
bash run_finetune_smd.sh
#    checkpoint → chronos2-single-stage_SMD_maskLoss_v2/finetuned-ckpt

# 3) evaluate (VUS-ROC / VUS-PR) on the held-out test pkl
CHECKPOINT="$PWD/chronos2-single-stage_SMD_maskLoss_v2/finetuned-ckpt" \
    bash run_forward_smd.sh
```

> **Note:** `run_forward_smd.sh` still defaults `CHECKPOINT` to the **v1** checkpoint.
> For v2 results, override `CHECKPOINT` as shown above (point it at the v2 output dir).

---

## What changed in v2 (the three suggestions)

All logic lives in **`finetune_anomaly_simple.py`** (trainer + dataset); knobs are
plumbed through **`run_finetune_smd.sh`**. New defaults **are** the recommended config.

| # | Suggestion | Flag (env → CLI) | v2 default | v1 value |
|---|------------|------------------|------------|----------|
| 1 | Hinge **each** anomaly step, not the pooled mean | `HINGE_MODE` → `--hinge_mode` | `per_step` | `pooled` |
| 2 | **Relative** margin (× the window's own normal error), not absolute τ | `MARGIN_MODE` → `--margin_mode`, `MARGIN_M` → `--margin_m` | `relative`, `3` | `absolute`, τ=13 |
| 3 | **Count-weighted sampler** — fix class imbalance at train time (thresholdless) | `SAMPLING_TARGET` → `--sampling_target` | `0.4` | — (uniform) |

### 1. Hinge each anomaly step (not the pooled mean)

**Problem:** a handful of huge-error anomaly steps push the *mean* past the margin, so
other anomaly steps that stay well-predicted get **zero gradient** — and those are
exactly the per-timestep false negatives at inference.

**Fix:** hinge every anomaly step individually, then average.

```python
# finetune_anomaly_simple.py — Chronos2SingleStageTrainer.compute_loss
if self.hinge_mode == "per_step":
    # Suggestion 1: hinge EACH anomaly step, then average over anomaly steps.
    step_hinge = torch.clamp(margin_eff - per_step, min=0.0) * anomaly_step
    if self.agg_mode == "batch_global":
        L_bad_term = step_hinge.sum() / n_anom_steps.clamp(min=eps)
    else:
        hinge_w = step_hinge.sum(dim=1) / cnt_anom_w.clamp(min=eps)
        L_bad_term = (hinge_w * has_anom).sum() / has_anom.sum().clamp(min=eps)
else:  # pooled: hinge the MEAN anomaly loss (original v1 behaviour)
    ...
```

Verified behaviour — anomaly steps with errors `[big=10, small=0.5]`, margin=2.5:
per-step grads come out `[…, 0.0, −0.5]`. The saturated big-error step no longer
"covers" the well-predicted step, which keeps getting pushed. Under `pooled`, both
would be zeroed out.

### 2. Relative margin (self-scaling per window)

**Problem:** instance-norm doesn't equalize pinball-loss *scale* across SMD machines —
quiet series sit near loss ≈ 0.1, busy ones near 4+. A fixed τ over-pushes quiet
series and is already-satisfied (hinge inactive from step 0) on noisy ones.

**Fix:** margin = `margin_m × L_good_w.detach()` per window — "anomaly error must reach
`margin_m`× this window's **own** normal error."

```python
# Per-window NORMAL reference loss (detached) — the "own normal error".
L_good_w = (per_step * normal_step).sum(dim=1) / cnt_norm_w.clamp(min=eps)

# Effective per-step target margin, broadcastable to (rows, P):
if self.margin_mode == "relative":
    margin_eff = (self.margin_m * L_good_w.detach()).unsqueeze(1)   # (rows, 1)
else:
    margin_eff = per_step.new_full((), float(self.margin_tau))      # scalar
```

**Choosing `MARGIN_M`.** Default is **`3`** (SMD). With the count-weighted sampler (below)
now supplying a strong, steady anomaly gradient, `M` no longer needs to be cranked high to
compensate for sparse anomalies — `~2–4` is the useful range. For **mtsbench**, override
`MARGIN_M=5` to compare apples-to-apples against the earlier `M=5` (unbalanced) run. Tune by
watching `anomaly_active_frac` (want ~0.2–0.4, not collapsing to 0), the anomaly-step loss
(should climb), and the normal-step loss (should keep falling — if it stalls, `M` is too
high and is hurting `L_good`).

### 3. Count-weighted sampler (fix class imbalance at train time)

**Problem:** anomaly steps are **rare** — ~3.4% of SMD forecast steps (only ~6.9% of SMD
train windows are anomaly-bearing), ~14% on mtsbench. Under the stock **uniform** sampler
most batches carry few/no anomaly steps, so the bad term gets a **weak, bursty** gradient.
The optimizer then takes the easy path — it drives *normal* loss down and leaves *anomaly*
loss flat (the exact "normal↓, anomaly flat" symptom seen in earlier plots).

**Fix — sample windows by how much anomaly signal they carry.** The loss is per-**step**
but the sampler picks whole **windows**, so we tilt *which windows* get drawn: each window
is chosen with probability proportional to its anomaly-**step** count,

```
weight_i = n_anom_i + eps · H          (H = prediction_length = 64)
```

- **Thresholdless.** A window with 40 anomaly steps counts more than one with 2; there is
  **no `≥ T` cutoff** (the old window-classification threshold is gone).
- **`eps · H` floor** keeps *every* window — including pure-normal ones — reachable, so
  `L_good` still trains.
- **`eps` is auto-solved**, not hand-set. You give a **target anomaly-step fraction per
  batch** (`SAMPLING_TARGET`, default `0.4`) and the code binary-searches `eps` to hit it.
- **No balanced pkl, no data dropped.** The full pkl is kept; balancing happens on the fly.
- **Eval is never balanced** — validation/test use sequential batching, so eval metrics
  stay honest (real-world class ratio).

```python
# finetune_anomaly_simple.py — AnomalyChronos2Dataset
# __init__: solve eps from target, precompute the sampling distribution once
self._eps = self._solve_eps(self._n_anom, self._H, target)     # target = sampling_target
w = self._n_anom.astype(np.float64) + self._eps * self._H
self._cum = np.cumsum(w / w.sum())                             # cumulative dist over ALL windows

# _generate_train_batches: weighted RANDOM draw (not argmax) via inverse-CDF
input_idx = int(np.searchsorted(self._cum, np.random.random(), side="right"))
```

**This is weighted *random* sampling, not "pick the highest".** `searchsorted` on the
cumulative array is an inverse-CDF draw: a random `u ∈ [0,1)` maps to a window, with
high-weight windows *more likely* but never guaranteed. Every batch is still a fresh random
sample — only the class mix is tilted. Randomness and window diversity are preserved.

`sampling_target` is set on the class from `main()` before `pipeline.fit()` builds the
dataset. Verified on real data: mtsbench & SMD both hit **40.0%** realized anomaly-step
fraction for `target=0.4`.

**Unreachable targets self-report.** If a dataset's anomalies are *scattered* (few steps
per window), the target may be mathematically impossible — sampling can't create anomaly
steps that aren't there. The startup log prints `expected=` vs `target=`, and a
`WARNING` fires when the max achievable falls short, telling you to lower `SAMPLING_TARGET`.
(SMD is fine: its anomaly windows average ~31/64 anomaly steps, so 0.4 is reachable.)

### Bonus monitors

- **`anomaly_active_frac`** (in `trainer_state.json`, printed each eval): fraction of anomaly
  steps still **below** their margin, i.e. how much of the bad term still carries gradient.
  Near 0 ⟹ the margin is already satisfied everywhere. Watch it stay ~0.2–0.4 (not collapse
  to 0) as you tune `MARGIN_M`.
- **realized anomaly-step fraction** — the sampler logs the *actual* anomaly-step % it is
  delivering every 500 batches (`[sampler] realized anomaly-step fraction ...`), so you can
  confirm balancing is live and matches `SAMPLING_TARGET`.

---

## Configuration reference (`run_finetune_smd.sh`)

Override any of these as environment variables, e.g.
`HINGE_MODE=pooled MARGIN_MODE=absolute bash run_finetune_smd.sh`.

| Env var | Default | Meaning |
|---------|---------|---------|
| `HINGE_MODE` | `per_step` | `per_step` (Suggestion 1) or `pooled` (v1) |
| `MARGIN_MODE` | `relative` | `relative` (Suggestion 2) or `absolute` (v1) |
| `MARGIN_M` | `3` | relative-margin multiplier (used when `MARGIN_MODE=relative`); use `5` for mtsbench |
| `MARGIN_TAU` | `13.0` | absolute margin (used **only** when `MARGIN_MODE=absolute`) |
| `MARGIN_LAMBDA` | `1.0` | weight λ on the bad term |
| `SAMPLING_TARGET` | `0.4` | count-weighted sampler: desired anomaly-**step** fraction per batch (Suggestion 3); `eps` auto-solved. `≤0` or `≤ natural baseline` → uniform |
| `AGG_MODE` | `batch_global` | `batch_global` (each step equal) or `per_window` (each window equal) |
| `FINETUNE_MODE` | `lora` | `lora` or `full` |
| `NUM_STEPS` | `4000` | training steps |
| `BATCH_SIZE` / `GRAD_ACCUM` | `160` / `2` | rows per micro-batch / accumulation |
| `LR` | `1e-5` | learning rate |
| `ENABLE_SEP_TOKEN` | `1` | insert `[SEP]` between normal signal and context |
| `NORMAL_SIGNAL_LENGTH` | `256` | length of the normal-signal prefix (multiple of `INPUT_PATCH_SIZE`) |
| `CONTEXT_LENGTH` | `768` | = `NORMAL_SIGNAL_LENGTH` + actual context (256 + 512) |

### Run on mtsbench instead of SMD

```bash
PREPARED_DIR=/path/to/prepared_data_labeled \
TEST_DATA=/path/to/prepared_data_labeled/test_model_inputs.pkl \
MARGIN_M=5 \
    bash run_finetune_smd.sh
```

`MARGIN_M=5` keeps the relative margin comparable to the earlier (unbalanced) mtsbench
run so you can isolate the effect of the new count-weighted balancing.

### Reproduce (close to) v1 behaviour

```bash
HINGE_MODE=pooled MARGIN_MODE=absolute MARGIN_TAU=13 SAMPLING_TARGET=0 \
    bash run_finetune_smd.sh
```

`SAMPLING_TARGET=0` (≤ natural baseline) restores **uniform** sampling; combined with
`pooled` + `absolute` this recovers the old `L_good + λ·max(0, τ − mean_anomaly_loss)`
loss with uniform batches (the pre-balancing behaviour).

---

## How the loss is wired (data → trainer)

- **`AnomalyChronos2Dataset`** carries the per-timestep labels (`future_labels`,
  0=normal / 1=anomaly) through to the trainer, replicated per channel-row so they line
  up with `future_target`. It also (a) precomputes per-window anomaly counts + the
  count-weighted sampling distribution and (b) overrides `_generate_train_batches` to
  draw windows from it (weighted-random, thresholdless). `future_type` is still derived
  (presence: any anomaly step) for backward-compatible logging only.
- **`Chronos2SingleStageTrainer.compute_loss`** does a **single** forward pass, splits
  the model's `per_step_loss` (shape `(rows, horizon)`, summed over quantiles) by the
  step labels, and forms `L_good` + λ·`L_bad_term` per the `hinge_mode` / `margin_mode`
  / `agg_mode` knobs above.
- `pipeline.fit` builds its dataset internally, so `main()` swaps in the subclass and
  sets the sampling target just before fit:

  ```python
  chronos2_pipeline.Chronos2Dataset = AnomalyChronos2Dataset
  AnomalyChronos2Dataset.sampling_target = args.sampling_target   # count-weighted sampler
  ```

No files outside `SMD_Maskloss_v2/` are modified — the shared `chronos` package and the
data-prep pipeline are untouched. **The data prep (`prepare_smd_split.py`) keeps *all*
windows** (every sliding window, per-step `future_labels`); no balancing is done at prep
time — the class imbalance is handled entirely by the runtime sampler.

---

## Files

| File | Role |
|------|------|
| `run_prepare_smd.sh` → `prepare_smd_split.py` | build train/val/test pkls (file-based 50/50 split) |
| `run_finetune_smd.sh` → `finetune_anomaly_simple.py` | **v2 training** (all three suggestions live here) |
| `run_forward_smd.sh` → `forward.py` | inference + VUS-ROC / VUS-PR scoring |
| `plot_mse.ipynb` | training-curve / MSE plots from `trainer_state.json` |
| `chronos2-single-stage_SMD_maskLoss_v2/finetuned-ckpt` | output checkpoint |

---

## Verification done

- Python + bash syntax checks pass (`py_compile`, `bash -n`).
- **Loss** — numerical test over all 8 `hinge_mode × margin_mode × agg_mode`
  combinations: every combination is finite and produces gradients; the v1 combo
  reproduces the old loss exactly; per-step gradients confirm Suggestion 1 isolates and
  pushes the well-predicted anomaly steps that the pooled hinge would have starved.
- **Count-weighted sampler** — `_solve_eps` + Monte-Carlo draws tested on the **real**
  pkls: `SAMPLING_TARGET=0.4` yields a realized anomaly-step fraction of **40.0%**
  (mtsbench) and **40.2%** (SMD); all anomaly-bearing windows are reachable; edge cases
  (no anomalies / target ≤ baseline) fall back to uniform. Confirmed SMD's `0.4` target
  is reachable (anomaly windows average ~31/64 anomaly steps).

## Changelog vs the previous v2

- **Suggestion 3 replaced**: the presence-based window oversampler (`ANOMALY_FRAC`,
  which silently fell back to uniform because window labels were never derived) is gone.
  It's now a **thresholdless count-weighted sampler** (`SAMPLING_TARGET`, `eps`
  auto-solved) that actually engages.
- **Removed** the dead `--anomaly_threshold` / `ANOMALY_THRESHOLD` knob (count-weighting
  needs no threshold).
- **`MARGIN_M` default 5 → 3** (balancing now supplies the anomaly gradient).
- Startup log now reports **real** anomaly stats (previously mislogged "0 anomaly
  windows"); added the realized-fraction sampler monitor and an unreachable-target
  warning.
