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
| 2 | **Relative** margin (× the window's own normal error), not absolute τ | `MARGIN_MODE` → `--margin_mode`, `MARGIN_M` → `--margin_m` | `relative`, `2.5` | `absolute`, τ=13 |
| 3 | **Oversample** anomaly windows so every batch trains the bad term | `ANOMALY_FRAC` → `--anomaly_frac` | `0.5` | `0` (uniform) |

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

### 3. Oversample anomaly windows

**Problem:** only ~4.7% of SMD train windows contain anomalies, and a micro-batch is
only ~4 windows (160 rows ÷ 38 channels), so most optimizer steps never see an anomaly
step → the bad term rarely gets gradient.

**Fix:** a weighted train sampler that draws a target fraction of anomaly windows into
every batch. Implemented by overriding the dataset's batch generator (no pkl
duplication, no changes to the shared `chronos` package).

```python
# finetune_anomaly_simple.py — AnomalyChronos2Dataset._generate_train_batches
frac = float(type(self).anomaly_frac)
use_weighted = frac > 0.0 and len(self._anom_idx) > 0 and len(self._norm_idx) > 0
while True:
    current_batch_size, input_indices = 0, []
    while current_batch_size < self.batch_size:
        if use_weighted:
            pool = self._anom_idx if np.random.random() < frac else self._norm_idx
            input_idx = int(np.random.choice(pool))
        else:                                    # frac<=0 → original uniform sampler
            input_idx = np.random.randint(len(self.inputs))
        input_indices.append(input_idx)
        current_batch_size += self.inputs[input_idx]["context"].shape[0]
    yield self._build_batch(input_indices)
```

`anomaly_frac` is set on the class from `main()` before `pipeline.fit()` builds the
dataset. Validation is untouched (it uses sequential, not weighted, batching).

### Bonus: hinge-activity monitor

A new metric **`anomaly_active_frac`** (logged to `trainer_state.json` and printed at
each eval) reports the fraction of anomaly steps still **below** their margin, i.e. how
much of the bad term still carries gradient. Near 0 ⟹ the margin is already satisfied
everywhere (the "hinge inactive from step 0" symptom). Purely additive — existing
`plot_mse.ipynb` parsing is unaffected.

---

## Configuration reference (`run_finetune_smd.sh`)

Override any of these as environment variables, e.g.
`HINGE_MODE=pooled MARGIN_MODE=absolute bash run_finetune_smd.sh`.

| Env var | Default | Meaning |
|---------|---------|---------|
| `HINGE_MODE` | `per_step` | `per_step` (Suggestion 1) or `pooled` (v1) |
| `MARGIN_MODE` | `relative` | `relative` (Suggestion 2) or `absolute` (v1) |
| `MARGIN_M` | `2.5` | relative-margin multiplier (used when `MARGIN_MODE=relative`) |
| `MARGIN_TAU` | `13.0` | absolute margin (used **only** when `MARGIN_MODE=absolute`) |
| `MARGIN_LAMBDA` | `1.0` | weight λ on the bad term |
| `ANOMALY_FRAC` | `0.5` | target anomaly-window fraction per batch (Suggestion 3); `0` = uniform |
| `AGG_MODE` | `batch_global` | `batch_global` (each step equal) or `per_window` (each window equal) |
| `FINETUNE_MODE` | `lora` | `lora` or `full` |
| `NUM_STEPS` | `4000` | training steps |
| `BATCH_SIZE` / `GRAD_ACCUM` | `160` / `2` | rows per micro-batch / accumulation |
| `LR` | `1e-5` | learning rate |
| `ENABLE_SEP_TOKEN` | `1` | insert `[SEP]` between normal signal and context |
| `NORMAL_SIGNAL_LENGTH` | `256` | length of the normal-signal prefix (multiple of `INPUT_PATCH_SIZE`) |
| `CONTEXT_LENGTH` | `768` | = `NORMAL_SIGNAL_LENGTH` + actual context (256 + 512) |

### Reproduce exact v1 behaviour

```bash
HINGE_MODE=pooled MARGIN_MODE=absolute MARGIN_TAU=13 ANOMALY_FRAC=0 \
    bash run_finetune_smd.sh
```

This combination reproduces the old `L_good + λ·max(0, τ − mean_anomaly_loss)` loss
bit-for-bit (verified numerically).

---

## How the loss is wired (data → trainer)

- **`AnomalyChronos2Dataset`** carries the per-timestep labels (`future_labels`,
  0=normal / 1=anomaly) through to the trainer, replicated per channel-row so they line
  up with `future_target`. It also (a) overrides `_generate_train_batches` for anomaly
  oversampling and (b) builds `_anom_idx` / `_norm_idx` index pools.
- **`Chronos2SingleStageTrainer.compute_loss`** does a **single** forward pass, splits
  the model's `per_step_loss` (shape `(rows, horizon)`, summed over quantiles) by the
  step labels, and forms `L_good` + λ·`L_bad_term` per the `hinge_mode` / `margin_mode`
  / `agg_mode` knobs above.
- `pipeline.fit` builds its dataset internally, so `main()` swaps in the subclass and
  sets the oversampling fraction just before fit:

  ```python
  chronos2_pipeline.Chronos2Dataset = AnomalyChronos2Dataset
  AnomalyChronos2Dataset.anomaly_frac = args.anomaly_frac   # Suggestion 3
  ```

No files outside `SMD_Maskloss_v2/` are modified — the shared `chronos` package and the
data-prep pipeline are untouched.

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

- Python + bash syntax check pass.
- Numerical test over all 8 `hinge_mode × margin_mode × agg_mode` combinations: every
  combination is finite and produces gradients; the v1 combo reproduces the old loss
  exactly; per-step gradients confirm Suggestion 1 isolates and pushes the
  well-predicted anomaly steps that the pooled hinge would have starved.
