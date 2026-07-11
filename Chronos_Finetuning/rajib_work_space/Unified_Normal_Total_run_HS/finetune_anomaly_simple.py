"""
Single-stage anomaly-aware fine-tuning for Chronos-2  (maskloss v2 + HIERARCHICAL SAMPLING).

Uses a SINGLE combined dataset (normal + anomaly future pairs mixed together).
The loss is a PER-STEP masked margin (hinge) objective; per-timestep labels
(future_labels, 0=normal / 1=anomaly) mask the model's per-step pinball loss:

    L_total = L_good + lambda * L_bad_term

    normal steps  (label 0) →  L_good     : minimise (predict normal well)
    anomaly steps (label 1) →  L_bad_term : push UP toward a margin, then saturate

This file is TOTAL_RUN_maskloss_v2/finetune_anomaly_simple.py with exactly ONE
thing changed: the train-time sampler. The loss, the model, the LoRA config and
every hyperparameter are untouched, so HS-vs-2x is a one-variable ablation.

  1. hinge_mode=per_step  — hinge EACH anomaly step, max(0, margin - per_step_i),
     then average, instead of hinging only the pooled mean. Every anomalous
     timestep is pushed (VUS is per-timestep), so a few huge-error steps can't
     satisfy the margin on behalf of the rest.                          [unchanged]
  2. margin_mode=relative — margin = margin_m * L_good_w.detach() per window, so the
     bad target self-scales to each SMD machine's own normal error instead of a
     fixed tau that over-pushes quiet series and is pre-satisfied on busy ones.
                                                                        [unchanged]
  3. HIERARCHICAL SAMPLING (replaces v2's single global count-weighted sampler).
     See AnomalyChronos2Dataset below. Three levels per window draw:

         level 1   dataset  ~ Uniform(K)                         (dataset imbalance)
         level 2   kind     ~ Bernoulli(p_anom = 1/3)            (class imbalance)
         level 3   window   ~ count-weighted WITHIN that dataset by
                              n_anom      if kind == anomalous
                              64 - n_anom if kind == normal

     v2 used ONE global weight vector `n_anom_i + eps*H` for both jobs, which it
     cannot do: solving eps for a 40% anomaly-step target also drags MITDB/SVDB/
     cicids to 1.35x their pool share and GHL/OPPORTUNITY down to 0.68x, undoing
     the per-dataset cap that v2's prep had just applied. Splitting the draw into
     independent levels fixes both at once, and needs no eps, no target, and no
     window-level anomaly threshold (see level 3's self-gating below).

At inference: high prediction error ⟹ high anomaly score.

Usage:
    python finetune_anomaly_simple.py --data_dir ./prepared_total       # HS defaults
    python finetune_anomaly_simple.py --p_anom 0.5                      # 1:1 kind mix
    python finetune_anomaly_simple.py --hinge_mode pooled --margin_mode absolute \
        --margin_tau 13                                                 # ~v1 loss, HS sampler
"""

import argparse
import functools
import json
import logging
import os
import pickle

os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")

import torch
import numpy as np
from chronos import BaseChronosPipeline, Chronos2Pipeline
from chronos.chronos2.anomaly_trainer import Chronos2AnomalyTrainer
from chronos.chronos2.dataset import Chronos2Dataset
import chronos.chronos2.pipeline as chronos2_pipeline

log_path = os.path.join("./finetuning_log/log", "finetune_simple.log")
os.makedirs(os.path.dirname(log_path), exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[logging.StreamHandler(), logging.FileHandler(log_path)],
)
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
#  Dataset — carries future_type through to the trainer
# ─────────────────────────────────────────────────────────────────────────────

class AnomalyChronos2Dataset(Chronos2Dataset):
    """
    Chronos2Dataset that preserves a per-sample `future_type` (0=normal, 1=anomaly)
    and draws train windows with a HIERARCHICAL SAMPLER.

    The stock Chronos2Dataset validates inputs against {target, past_covariates,
    future_covariates} and drops everything else, so `future_type` would be lost.
    Here we strip our extra keys before the parent's validation/preparation, then
    re-attach them onto every batch the dataset yields — expanded to one entry per
    row so they line up with `context`/`future_target`/`group_ids` (each input series
    contributes `group_size` rows). The single-stage trainer pops them back off in
    compute_loss.

    Hierarchical sampling (dataset imbalance AND class imbalance)
    -------------------------------------------------------------
    Combined mTSBench has two independent imbalances: MITDB+SVDB are ~88% of all
    windows (DATASET), and anomaly steps are ~17% of forecast steps (CLASS). v2
    attacked the second with a single global weight vector `n_anom_i + eps*H` and
    the first with a per-dataset cap at prep time — but the global weight vector
    reaches across datasets and silently re-skews the dataset mix it had just fixed.

    HS draws each window in three independent levels:

        level 1   k ~ Uniform(K)                    K = number of train datasets
        level 2   kind ~ Bernoulli(p_anom)          p_anom = 1/3
        level 3   i ~ Categorical over dataset k's windows, with weight
                        n_anom_i          if kind == anomalous
                        H - n_anom_i      if kind == normal

    Level 1 gives every dataset exactly 1/K of the draws no matter how many windows
    it owns, so no cap and no discarding is needed at prep — the pool keeps every
    window. Level 2 fixes the normal:anomalous window ratio at (1-p):p = 2:1 in
    expectation, per dataset. Level 3 is thresholdless in BOTH branches:

      * anomalous branch is self-gating — a pure-normal window has n_anom = 0 and
        therefore weight 0, so it can never be drawn. No `>= T` cutoff is needed;
        this is the inconsistency that FT's (and hence 3x's) window-level threshold
        introduced into an otherwise per-step objective.
      * normal branch weights an anomaly-bearing window by its remaining normal
        steps H - n_anom, i.e. down in proportion to how anomalous it is, rather
        than excluding it. Nothing is thrown away.

    Because level 3's weights are renormalized WITHIN the chosen dataset, a dataset
    that is anomaly-sparse still yields anomalies (its own rare anomaly windows carry
    all the anomalous-branch mass) and a dataset that is anomaly-dense still yields
    normals. Empirically this maps natural anomaly-step rates spanning 1.7% (GHL) to
    18.9% (room-occupancy) onto a near-uniform ~30-33% per batch.

    Two consequences worth knowing:
      * `batch_size` counts CHANNEL ROWS, not windows (the parent fills a batch until
        sum of group_size >= batch_size). With mean F ~= 24.7 across the 12 train
        datasets a 160-row batch holds ~6-7 window draws from ~5 distinct datasets.
      * Level 1 balances DRAWS, not gradient. Under agg_mode=batch_global each drawn
        window contributes F_d rows of per-step loss, so a dataset's share of the
        gradient is F_d / sum(F). This is inherited from v2/3x, not introduced here;
        agg_mode=per_window would equalize it.

    Validation uses sequential batching (not this sampler), so eval keeps the natural,
    unbalanced distribution — eval metrics stay honest.
    """

    # Level-2 probability of drawing an ANOMALOUS-kind window. 1/3 gives the 2:1
    # normal:anomalous window ratio in expectation. Set on the class from main()
    # before pipeline.fit() builds the dataset.
    p_anom: float = 1.0 / 3.0

    @staticmethod
    def _cum_from_weights(w: np.ndarray):
        """Normalized cumulative distribution for O(log N) inverse-CDF draws.
        Returns None when the weights are degenerate (all zero)."""
        total = w.sum()
        if not np.isfinite(total) or total <= 0:
            return None
        cum = np.cumsum(w / total)
        cum[-1] = 1.0                                   # guard against fp drift
        return cum

    def __init__(self, inputs, *args, **kwargs):
        # Per-timestep labels (length = prediction_length), 0=normal / 1=anomaly.
        # These drive both the per-step masked loss AND level 3 of the sampler.
        future_labels = [np.asarray(d["future_labels"], dtype=np.int64) for d in inputs]
        # Window-level future_type is derived (presence: any anomaly step) purely for
        # backward-compatible logging/monitoring; the loss & sampler use future_labels.
        future_types = [int(fl.sum() >= 1) for fl in future_labels]
        # Level-1 grouping key, attached by load_train_pool() from the pkl's folder name.
        ds_names = [str(d.get("dataset", "_pool_")) for d in inputs]
        # Strip ALL THREE of our extra keys before the parent's validation:
        # `future_type` (window-level label), `future_labels` (per-timestep array),
        # and `dataset` (level-1 group id).
        cleaned = [
            {k: v for k, v in d.items()
             if k not in ("future_type", "future_labels", "dataset")}
            for d in inputs
        ]
        super().__init__(cleaned, *args, **kwargs)
        # Parent filters series shorter than min_past + prediction_length. Our fixed
        # 832-step targets are never filtered, so prepared inputs align 1:1 with
        # future_labels. Guard loudly in case lengths change and some get dropped.
        if len(self.inputs) != len(future_labels):
            raise RuntimeError(
                f"future_label alignment broke: {len(future_labels)} inputs given but "
                f"{len(self.inputs)} survived length filtering. Ensure every target is "
                "at least min_past + prediction_length steps long."
            )

        for prepared, ft, fl in zip(self.inputs, future_types, future_labels):
            prepared["future_type"] = ft
            prepared["future_labels"] = fl

        self._n_anom = np.asarray([int(fl.sum()) for fl in future_labels], dtype=np.int64)
        self._H = int(future_labels[0].shape[0]) if future_labels else 0
        self._p_anom = float(type(self).p_anom)
        if not 0.0 <= self._p_anom <= 1.0:
            raise ValueError(f"p_anom must be in [0, 1], got {self._p_anom}")

        # ── Level 1: group window indices by dataset ─────────────────────────────
        self._ds_names = sorted(set(ds_names))
        name_to_k = {name: k for k, name in enumerate(self._ds_names)}
        ds_of_window = np.asarray([name_to_k[n] for n in ds_names], dtype=np.int64)
        self._n_ds = len(self._ds_names)

        # ── Level 3: per-dataset cumulative distributions (normal / anomalous) ───
        # _groups[k] = (member_indices, cum_normal, cum_anomalous)
        self._groups = []
        exp_fracs = []
        for k, name in enumerate(self._ds_names):
            members = np.flatnonzero(ds_of_window == k)
            n_a = self._n_anom[members].astype(np.float64)
            n_n = float(self._H) - n_a
            cum_a = self._cum_from_weights(n_a)
            cum_n = self._cum_from_weights(n_n)

            if cum_a is None:
                # No anomaly steps anywhere in this dataset: the anomalous branch has
                # an all-zero weight vector. prepare_total.py refuses to write such a
                # dataset, so this only fires under --debug truncation. Degrade to the
                # normal branch rather than crash mid-run.
                logger.warning(
                    f"[hs] dataset '{name}' has {len(members)} windows and ZERO anomaly "
                    f"steps — its anomalous branch is undrawable. Falling back to the "
                    f"normal branch for this dataset; it contributes no bad-term gradient."
                )
                cum_a = cum_n
            if cum_n is None:
                logger.warning(
                    f"[hs] dataset '{name}' has no normal steps at all — falling back to "
                    f"the anomalous branch for its normal-kind draws."
                )
                cum_n = cum_a
            if cum_a is None:
                raise RuntimeError(f"[hs] dataset '{name}' has no drawable windows.")

            self._groups.append((members, cum_n, cum_a))

            # Expected anomaly steps per draw from this dataset, under the level-2/3
            # mixture:  p*E[n_anom | anom branch] + (1-p)*E[n_anom | normal branch].
            s_a, s_n = n_a.sum(), n_n.sum()
            e_a = float((n_a * n_a).sum() / s_a) if s_a > 0 else 0.0
            e_n = float((n_n * n_a).sum() / s_n) if s_n > 0 else 0.0
            exp_fracs.append((self._p_anom * e_a + (1.0 - self._p_anom) * e_n) / self._H)

        # Level 1 is uniform, so the pool-wide expectation is the mean of the per-dataset ones.
        self._exp_frac = float(np.mean(exp_fracs)) if exp_fracs else 0.0
        baseline = float(self._n_anom.mean() / self._H) if self._H else 0.0
        n_anom_win = int((self._n_anom >= 1).sum())

        logger.info(
            f"AnomalyChronos2Dataset: {len(self.inputs)} windows across {self._n_ds} datasets, "
            f"{n_anom_win} anomaly-bearing "
            f"({100.0 * n_anom_win / max(len(self.inputs), 1):.1f}%), "
            f"per-step anomaly baseline={100.0 * baseline:.1f}%."
        )
        logger.info(
            f"[hs] hierarchical sampler: level1 = uniform over {self._n_ds} datasets "
            f"({100.0 / max(self._n_ds, 1):.1f}% each), level2 = P(anomalous kind)="
            f"{self._p_anom:.3f}, level3 = count-weighted within dataset "
            f"(n_anom / {self._H}-n_anom). Expected anomaly-step fraction per draw: "
            f"{100.0 * self._exp_frac:.1f}% (natural {100.0 * baseline:.1f}%)."
        )
        for k, name in enumerate(self._ds_names):
            members = self._groups[k][0]
            n_a = self._n_anom[members]
            logger.info(
                f"[hs]   {name:<16s} {len(members):>7d} windows  "
                f"{int((n_a >= 1).sum()):>6d} anomaly-bearing  "
                f"natural {100.0 * n_a.mean() / self._H:>5.1f}%  ->  "
                f"HS {100.0 * exp_fracs[k]:>5.1f}%"
            )

        # Running realized-balance monitor (logged periodically from the sampler).
        self._seen_batches = 0
        self._seen_anom_steps = 0
        self._seen_total_steps = 0
        self._seen_ds_draws = np.zeros(self._n_ds, dtype=np.int64)

    def _draw_window(self) -> int:
        """One hierarchical draw: dataset -> kind -> window. Returns a global index."""
        k = np.random.randint(self._n_ds)                        # level 1
        members, cum_n, cum_a = self._groups[k]
        cum = cum_a if np.random.random() < self._p_anom else cum_n     # level 2
        j = int(np.searchsorted(cum, np.random.random(), side="right"))  # level 3
        if j >= len(members):                                    # fp edge guard
            j = len(members) - 1
        self._seen_ds_draws[k] += 1
        return int(members[j])

    def _generate_train_batches(self):
        """Hierarchically-sampled training batches.

        Draws windows one at a time via `_draw_window` until the parent's row budget
        (`batch_size` counts channel rows, not windows) is filled. Every draw
        re-samples the dataset, so a single batch mixes ~5 of the 12 datasets — which
        matters because agg_mode=batch_global pools the loss across the whole batch.

        Both levels are WEIGHTED RANDOM draws, not argmaxes, so every batch is a fresh
        random sample; only the dataset mix and the class mix are tilted.
        """
        while True:
            current_batch_size = 0
            input_indices = []

            while current_batch_size < self.batch_size:
                input_idx = self._draw_window()
                input_indices.append(input_idx)
                current_batch_size += self.inputs[input_idx]["context"].shape[0]

            # Realized-balance monitor: track the anomaly-STEP fraction and the dataset
            # mix the sampler is actually delivering, and log both periodically so the
            # achieved balance is visible in the run log.
            self._seen_batches += 1
            self._seen_anom_steps += int(self._n_anom[input_indices].sum())
            self._seen_total_steps += len(input_indices) * self._H
            if self._seen_batches % 500 == 0 and self._seen_total_steps > 0:
                tot_draws = max(int(self._seen_ds_draws.sum()), 1)
                mix = "  ".join(
                    f"{name}={100.0 * c / tot_draws:.1f}%"
                    for name, c in zip(self._ds_names, self._seen_ds_draws)
                )
                logger.info(
                    f"[hs] realized over {self._seen_batches} batches: anomaly-step "
                    f"fraction {100.0 * self._seen_anom_steps / self._seen_total_steps:.1f}% "
                    f"(expected {100.0 * self._exp_frac:.1f}%); dataset mix (uniform target "
                    f"{100.0 / max(self._n_ds, 1):.1f}%): {mix}"
                )

            yield self._build_batch(input_indices)

    def _build_batch(self, input_indices):
        batch = super()._build_batch(input_indices)

        row_future_types = []
        row_future_labels = []
        for input_idx in input_indices:
            group_size = self.inputs[input_idx]["context"].shape[0]
            # One window-level label / one (P,) per-step array, replicated across the
            # window's group_size channel rows so they line up with future_target.
            row_future_types.extend([self.inputs[input_idx]["future_type"]] * group_size)
            row_future_labels.extend([self.inputs[input_idx]["future_labels"]] * group_size)
        batch["future_type"] = torch.tensor(row_future_types, dtype=torch.long)
        batch["future_labels"] = torch.as_tensor(
            np.stack(row_future_labels), dtype=torch.long
        )  # (rows, prediction_length)

        return batch


# ─────────────────────────────────────────────────────────────────────────────
#  Single-Stage Trainer
# ─────────────────────────────────────────────────────────────────────────────

class Chronos2SingleStageTrainer(Chronos2AnomalyTrainer):
    """
    Single-stage trainer using a PER-STEP masked margin (hinge) objective.

        L = L_good + lambda * L_bad_term

    The per-timestep labels (`future_labels`, 0=normal / 1=anomaly) mask the model's
    per-step pinball loss (`Chronos2Output.per_step_loss`, shape (rows, horizon)):

      - L_good : mean per-step loss over NORMAL steps  -> minimised
      - L_bad_term : hinge pushing ANOMALY-step loss UP toward a margin
                     (self-saturating: no gradient once a step clears its margin).

    Two knobs shape the bad term, following the team's suggestions:

    hinge_mode (Suggestion 1 — hinge each step, not the pooled mean)
      - "per_step" (default): L_bad_term = mean over anomaly steps of
        max(0, margin - per_step_i). Every anomalous timestep is pushed
        individually, so a few huge-error steps can no longer satisfy the margin on
        behalf of other anomaly steps that stay well-predicted (the false negatives
        at inference, since VUS is per-timestep).
      - "pooled" (original): hinge the *mean* anomaly loss, max(0, margin - L_bad).

    margin_mode (Suggestion 2 — relative, not absolute, margin)
      - "relative" (default): margin = margin_m * L_good_w.detach(), i.e. per window
        the anomaly error must be margin_m x that window's OWN normal error. This
        self-scales across quiet (loss ~0.1) and busy (loss ~4+) SMD machines, which
        instance-norm does not equalize, so a fixed tau no longer over-pushes quiet
        series while being already-satisfied on noisy ones.
      - "absolute" (original): margin = margin_tau, a fixed constant.

    Aggregation (`agg_mode`) controls how the masked losses are pooled:
      - "batch_global" (default): pool all steps in the batch (each *step* weighted
        equally; longer/denser windows contribute more). Most stable across datasets.
      - "per_window": reduce within each window, then average over windows (each
        *window* weighted equally regardless of step count).

    A SINGLE forward pass over the whole batch is used; the normal/anomaly split is by
    step, not by window, so both terms come from the same predictions.

    Parameters
    ----------
    hinge_mode : str
        "per_step" (default) or "pooled".
    margin_mode : str
        "relative" (default) or "absolute".
    margin_m : float
        Relative-margin multiplier (used when margin_mode="relative"). ~2-3.
    margin_tau : float
        Absolute margin (used when margin_mode="absolute"). Per-step loss (summed over
        quantiles) sits ~3-4 for well-predicted normal steps, so tau ~10-15.
    margin_lambda : float
        Weight on the anomaly (bad) term. 1.0 is a sensible default.
    agg_mode : str
        "batch_global" (default) or "per_window".
    """

    def __init__(
        self,
        *args,
        margin_tau: float = 12.0,
        margin_lambda: float = 1.0,
        agg_mode: str = "batch_global",
        hinge_mode: str = "per_step",
        margin_mode: str = "relative",
        margin_m: float = 2.5,
        loss_ceiling: float | None = None,  # accepted for back-compat; unused by hinge
        **kwargs,
    ):
        super().__init__(*args, loss_ceiling=loss_ceiling, **kwargs)
        if agg_mode not in ("batch_global", "per_window"):
            raise ValueError(
                f"agg_mode must be 'batch_global' or 'per_window', got {agg_mode!r}"
            )
        if hinge_mode not in ("per_step", "pooled"):
            raise ValueError(
                f"hinge_mode must be 'per_step' or 'pooled', got {hinge_mode!r}"
            )
        if margin_mode not in ("relative", "absolute"):
            raise ValueError(
                f"margin_mode must be 'relative' or 'absolute', got {margin_mode!r}"
            )
        self.margin_tau = margin_tau
        self.margin_lambda = margin_lambda
        self.agg_mode = agg_mode
        self.hinge_mode = hinge_mode
        self.margin_mode = margin_mode
        self.margin_m = margin_m
        # Running sums of the RAW per-step metrics, split by step label, so we can log
        # normal/anomaly means separately for monitoring:
        #   *_sum     -> summed pinball loss   -> normal_loss / anomaly_loss
        #   *_mse_sum -> summed squared error  -> mse_normal_step / mse_anomaly_step
        #   *_cnt     -> number of steps (shared denominator for both)
        #   a_active_sum -> anomaly steps still BELOW their margin (hinge active); its
        #                   ratio to a_cnt shows how much of the bad term still has
        #                   gradient (near 0 => margin already satisfied everywhere).
        # Train and eval kept apart.
        def _fresh():
            return {"n_sum": 0.0, "n_mse_sum": 0.0, "n_cnt": 0,
                    "a_sum": 0.0, "a_mse_sum": 0.0, "a_cnt": 0,
                    "a_active_sum": 0.0}
        self._acc = {"train": _fresh(), "eval": _fresh()}
        self._fresh_acc = _fresh


    @staticmethod
    def _select(inputs: dict, mask: torch.Tensor) -> dict:
        """Return a subset of the batch using a boolean mask."""
        return {
            k: v[mask] if isinstance(v, torch.Tensor) and v.shape[0] == mask.shape[0] else v
            for k, v in inputs.items()
        }

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        # future_type is kept only for back-compat; the per-step objective uses
        # future_labels. Neither is passed to the model.
        inputs.pop("future_type", None)
        future_labels = inputs.pop("future_labels")        # (rows, P) long, 0/1

        outputs = model(**inputs)
        # (rows, T) per-step pinball loss (summed over quantiles), loss_mask applied.
        P = future_labels.shape[1]
        per_step = outputs.per_step_loss[:, :P]            # trim any horizon padding
        per_step_mse = outputs.per_step_mse[:, :P]         # per-step squared error
        labels = future_labels.to(per_step.device)

        normal_step = (labels == 0).float()                # (rows, P)
        anomaly_step = (labels == 1).float()

        eps = 1.0  # clamp denominators to avoid /0 on empty subsets

        # Per-window / batch step counts and presence masks.
        cnt_norm_w = normal_step.sum(dim=1)                # (rows,)
        cnt_anom_w = anomaly_step.sum(dim=1)
        has_norm = (cnt_norm_w > 0).float()
        has_anom = (cnt_anom_w > 0).float()
        n_norm_steps = normal_step.sum()
        n_anom_steps = anomaly_step.sum()

        # Per-window NORMAL reference loss (detached) — the "own normal error" that the
        # relative margin scales (Suggestion 2). Also reused as L_good_w below.
        L_good_w = (per_step * normal_step).sum(dim=1) / cnt_norm_w.clamp(min=eps)

        # Effective per-step target margin, broadcastable to (rows, P):
        #   relative -> margin_m * L_good_w.detach()  (per window)
        #   absolute -> margin_tau                    (scalar)
        if self.margin_mode == "relative":
            margin_eff = (self.margin_m * L_good_w.detach()).unsqueeze(1)   # (rows, 1)
        else:
            margin_eff = per_step.new_full((), float(self.margin_tau))      # scalar

        # ── Monitoring accumulation (pinball + MSE + active-hinge fraction) ─────────
        acc = self._acc["train" if model.training else "eval"]
        acc["n_sum"] += (per_step * normal_step).sum().detach().item()
        acc["n_mse_sum"] += (per_step_mse * normal_step).sum().detach().item()
        acc["n_cnt"] += int(n_norm_steps.item())
        acc["a_sum"] += (per_step * anomaly_step).sum().detach().item()
        acc["a_mse_sum"] += (per_step_mse * anomaly_step).sum().detach().item()
        acc["a_cnt"] += int(n_anom_steps.item())
        with torch.no_grad():
            # anomaly steps still below their margin (hinge > 0 => gradient flowing)
            step_active = ((margin_eff - per_step) > 0).float() * anomaly_step
            acc["a_active_sum"] += step_active.sum().item()

        # ── L_good : minimise normal-step loss ─────────────────────────────────────
        if self.agg_mode == "batch_global":
            L_good = (per_step * normal_step).sum() / n_norm_steps.clamp(min=eps)
        else:  # per_window: mean of per-window normal means over windows with normals
            L_good = (L_good_w * has_norm).sum() / has_norm.sum().clamp(min=eps)

        # ── L_bad_term : push anomaly-step loss up toward the margin ───────────────
        if self.hinge_mode == "per_step":
            # Suggestion 1: hinge EACH anomaly step, then average over anomaly steps.
            step_hinge = torch.clamp(margin_eff - per_step, min=0.0) * anomaly_step
            if self.agg_mode == "batch_global":
                L_bad_term = step_hinge.sum() / n_anom_steps.clamp(min=eps)
            else:
                hinge_w = step_hinge.sum(dim=1) / cnt_anom_w.clamp(min=eps)
                L_bad_term = (hinge_w * has_anom).sum() / has_anom.sum().clamp(min=eps)
        else:  # pooled: hinge the MEAN anomaly loss (original per-mean behaviour)
            if self.margin_mode == "relative":
                # Per-window margin => the mean must be taken per window.
                L_bad_w = (per_step * anomaly_step).sum(dim=1) / cnt_anom_w.clamp(min=eps)
                margin_w = self.margin_m * L_good_w.detach()
                hinge_w = torch.clamp(margin_w - L_bad_w, min=0.0) * has_anom
                if self.agg_mode == "batch_global":
                    # weight each window by its anomaly-step count (step-equal pooling)
                    L_bad_term = (hinge_w * cnt_anom_w).sum() / n_anom_steps.clamp(min=eps)
                else:
                    L_bad_term = hinge_w.sum() / has_anom.sum().clamp(min=eps)
            else:  # absolute margin
                if self.agg_mode == "batch_global":
                    L_bad = (per_step * anomaly_step).sum() / n_anom_steps.clamp(min=eps)
                    L_bad_term = torch.clamp(self.margin_tau - L_bad, min=0.0)
                else:
                    L_bad_w = (per_step * anomaly_step).sum(dim=1) / cnt_anom_w.clamp(min=eps)
                    hinge_w = torch.clamp(self.margin_tau - L_bad_w, min=0.0) * has_anom
                    L_bad_term = hinge_w.sum() / has_anom.sum().clamp(min=eps)

        total_loss = L_good + self.margin_lambda * L_bad_term
        return (total_loss, outputs) if return_outputs else total_loss

    def log(self, logs: dict, *args, **kwargs):
        """Inject separate normal/anomaly loss means into trainer_state.json log_history.

        With a dict eval_dataset, HF calls log() once per dataset, with the dataset
        name baked into the standard loss key: "eval_loss" (single set), or
        "eval_val_loss" / "eval_test_loss" (named sets). We mirror that prefix onto
        our custom normal/anomaly keys so each dataset is logged independently.
        A training log instead carries the bare "loss".
        """
        # print("--------------------LOG function()-----------------")
        # Find the standard eval loss key (excluding our own derived keys) to learn
        # both the phase and the per-dataset prefix.
        eval_loss_keys = [
            k for k in logs
            if k.startswith("eval") and k.endswith("loss")
            and not (k.endswith("normal_loss") or k.endswith("anomaly_loss"))
        ]
        if eval_loss_keys:
            phase = "eval"
            prefix = eval_loss_keys[0][: -len("loss")]  # "eval_" / "eval_val_" / "eval_test_"
        else:
            phase = "train"
            prefix = ""
        acc = self._acc[phase]

        if acc["n_cnt"] > 0:
            logs[f"{prefix}normal_loss"] = round(acc["n_sum"] / acc["n_cnt"], 4)
            logs[f"{prefix}mse_normal_step"] = round(acc["n_mse_sum"] / acc["n_cnt"], 4)
        if acc["a_cnt"] > 0:
            logs[f"{prefix}anomaly_loss"] = round(acc["a_sum"] / acc["a_cnt"], 4)
            logs[f"{prefix}mse_anomaly_step"] = round(acc["a_mse_sum"] / acc["a_cnt"], 4)
            # Fraction of anomaly steps still below their margin (hinge active). Near 0
            # means the margin is already satisfied and the bad term has little gradient.
            logs[f"{prefix}anomaly_active_frac"] = round(acc["a_active_sum"] / acc["a_cnt"], 4)

        if phase == "eval":
            tag = prefix.rstrip("_").upper()  # "EVAL" / "EVAL_VAL" / "EVAL_TEST"
            print(f"\n======================================================")
            print(f" [{tag}] Normal step loss: {logs.get(f'{prefix}normal_loss', 'N/A')} | Anomaly step loss: {logs.get(f'{prefix}anomaly_loss', 'N/A')}")
            print(f"              MSE Normal step: {logs.get(f'{prefix}mse_normal_step', 'N/A')} | MSE Anomaly step: {logs.get(f'{prefix}mse_anomaly_step', 'N/A')}")
            print(f"              Anomaly hinge-active frac: {logs.get(f'{prefix}anomaly_active_frac', 'N/A')}")
            print(f"======================================================\n", flush=True)

        self._acc[phase] = self._fresh_acc()
        out = super().log(logs, *args, **kwargs)
        # Persist the loss history every logging step. Without this, runs with
        # save_strategy="no" (i.e. --no_validation) never write trainer_state.json
        # and the loss curve is lost when training ends.
        try:
            os.makedirs(self.args.output_dir, exist_ok=True)
            with open(os.path.join(self.args.output_dir, "trainer_state.json"), "w") as fh:
                json.dump({"log_history": self.state.log_history}, fh, indent=2)
        except Exception as exc:  # never let logging kill training
            logger.warning(f"Could not write trainer_state.json: {exc}")
        return out


# ─────────────────────────────────────────────────────────────────────────────
#  Argument Parsing
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Single-stage anomaly-aware fine-tuning for Chronos-2"
    )

    # Model
    p.add_argument("--model_id", default="amazon/chronos-2",
                   help="Pretrained model ID or local path")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")

    # Data
    p.add_argument("--data_dir", default="./prepared_total",
                   help="Output dir from prepare_total.py. Must contain "
                        "per_dataset/<DS>/train_model_inputs.pkl for each train dataset "
                        "(the hierarchical sampler groups by that folder name).")
    p.add_argument("--no_validation", action="store_true")
    p.add_argument("--test_data", default=None,
                   help="Optional path to a third dataset pkl (e.g. test_model_inputs.pkl). "
                        "It is evaluated and logged every eval step exactly like the validation "
                        "set (forward + loss only, NO weight updates). Logged as eval_test_*.")
    p.add_argument("--debug", action="store_true",
                   help="Truncate train/val to the first 50 samples for a quick smoke test.")

    # Fine-tuning mode
    p.add_argument("--finetune_mode", default="lora", choices=["full", "lora"])
    p.add_argument("--lora_r",        type=int,   default=32)
    p.add_argument("--lora_alpha",    type=int,   default=16)
    p.add_argument("--lora_dropout",  type=float, default=0.0)

    # Training hyperparameters
    p.add_argument("--prediction_length", type=int,   default=64)
    p.add_argument("--context_length",    type=int,   default=768,
                   help="Must equal normal_signal_length + actual context length (256+512=768)")
    p.add_argument("--enable_sep_token",  action="store_true")
    p.add_argument("--normal_signal_length", type=int, default=256)
    p.add_argument("--input_patch_size",  type=int,   default=16)
    p.add_argument("--num_steps",         type=int,   default=5000)
    p.add_argument("--lr",                type=float, default=None,
                   help="Learning rate (default: 2e-5 for LoRA, 1e-6 for full)")
    p.add_argument("--batch_size",        type=int,   default=4)
    p.add_argument("--gradient_accumulation_steps", type=int, default=8)
    p.add_argument("--fp16",              action="store_true", default=True)
    p.add_argument("--no_fp16",           dest="fp16", action="store_false")
    p.add_argument("--logging_steps",     type=int,   default=100)
    p.add_argument("--eval_steps",        type=int,   default=100,
                   help="Run validation (and log eval_loss) every N steps. Ignored when "
                        "--no_validation. Must divide save_steps (100) for best-model selection.")
    p.add_argument("--warmup_ratio",      type=float, default=0.05)
    p.add_argument("--lr_scheduler_type", default="cosine",
                   choices=["linear", "cosine", "cosine_with_restarts", "constant"])

    # Margin (hinge) loss: L_good + lambda * L_bad_term
    p.add_argument("--hinge_mode", default="per_step",
                   choices=["per_step", "pooled"],
                   help="Suggestion 1. 'per_step' hinges EACH anomaly step then "
                        "averages (pushes every anomalous timestep, matching per-step "
                        "VUS); 'pooled' hinges the mean anomaly loss (original).")
    p.add_argument("--margin_mode", default="relative",
                   choices=["relative", "absolute"],
                   help="Suggestion 2. 'relative' uses margin = margin_m * "
                        "L_good_w.detach() per window (self-scales across quiet/busy "
                        "SMD machines); 'absolute' uses the fixed margin_tau.")
    p.add_argument("--margin_m", type=float, default=3,
                   help="Relative-margin multiplier (margin_mode=relative). Anomaly "
                        "error must reach margin_m x the window's own normal error. "
                        "~2-4 with count-weighted balancing on (the sampler now "
                        "supplies the anomaly gradient, so M need not be cranked high).")
    p.add_argument("--margin_tau", type=float, default=12.0,
                   help="Absolute margin (margin_mode=absolute). Must sit ABOVE the "
                        "normal-point loss (~3-4 here) to matter — use ~10-15, NOT 2.")
    p.add_argument("--margin_lambda", type=float, default=1.0,
                   help="Weight on the anomaly (bad) term. Default 1.0.")
    p.add_argument("--agg_mode", default="batch_global",
                   choices=["batch_global", "per_window"],
                   help="How to pool the per-step masked losses. 'batch_global' pools "
                        "all steps (each step weighted equally, most stable across "
                        "datasets); 'per_window' averages per-window means (each window "
                        "weighted equally).")
    p.add_argument("--p_anom", type=float, default=1.0 / 3.0,
                   help="Hierarchical sampler, level 2: probability that a draw asks for "
                        "an ANOMALOUS-kind window. 1/3 gives the 2:1 normal:anomalous "
                        "window ratio in expectation, per dataset. Level 1 (dataset) is "
                        "uniform and level 3 (window) is count-weighted within the chosen "
                        "dataset by n_anom / (H - n_anom) — thresholdless in both "
                        "branches. Eval is never balanced.")

    # Output
    p.add_argument("--output_dir", default="./chronos2-single-stage")

    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
#  Helpers
# ─────────────────────────────────────────────────────────────────────────────

def load_data(path: str, label: str) -> list[dict]:
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{label} data not found at {path}. Run prepare_total.py first."
        )
    logger.info(f"Loading {label} from {path}")
    with open(path, "rb") as f:
        data = pickle.load(f)
    logger.info(f"  {len(data)} samples loaded")
    return data


def load_train_pool(data_dir: str, debug_per_dataset: int = 0) -> list[dict]:
    """Load the per-dataset train pkls into one flat pool, tagged for level 1.

    Each window gets a `dataset` key = its folder name; AnomalyChronos2Dataset strips
    the key before Chronos2Dataset's input validation and uses it to build the
    level-1 groups. Datasets with only one *test.csv are test-only and have no
    train_model_inputs.pkl — they are skipped here and are invisible to the sampler.

    Order is irrelevant (the sampler never scans sequentially), so no shuffle is done.
    """
    per_ds_root = os.path.join(data_dir, "per_dataset")
    if not os.path.isdir(per_ds_root):
        raise FileNotFoundError(
            f"{per_ds_root} not found. The hierarchical sampler needs the per-dataset "
            f"train pool written by this arm's prepare_total.py. A flat "
            f"train_model_inputs.pkl (TOTAL_RUN / TOTAL_RUN_maskloss_v2 layout) cannot "
            f"be used: it carries no dataset identity, so level 1 has nothing to group by."
        )

    pool: list[dict] = []
    found = []
    for ds in sorted(os.listdir(per_ds_root)):
        path = os.path.join(per_ds_root, ds, "train_model_inputs.pkl")
        if not os.path.exists(path):
            continue                                    # test-only dataset
        with open(path, "rb") as f:
            items = pickle.load(f)
        if debug_per_dataset > 0:
            items = items[:debug_per_dataset]
        for d in items:
            d["dataset"] = ds
        pool.extend(items)
        found.append((ds, len(items)))
        logger.info(f"  {ds:<16s} {len(items):>7d} train windows")

    if not found:
        raise FileNotFoundError(f"No train_model_inputs.pkl found under {per_ds_root}")
    logger.info(f"Train pool: {len(pool)} windows across {len(found)} datasets "
                f"(uncapped, thresholdless)")
    return pool



def build_lora_config(args):
    try:
        from peft import LoraConfig
    except ImportError:
        raise ImportError("pip install peft")
    modules_to_save = ["shared"] if args.enable_sep_token else None
    return LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=[
            "self_attention.q", "self_attention.v",
            "self_attention.k", "self_attention.o",
            "output_patch_embedding.output_layer",
        ],
        modules_to_save=modules_to_save,
    )


# ─────────────────────────────────────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    if args.lr is None:
        args.lr = 5e-6 if args.finetune_mode == "lora" else 1e-6
    use_fp16 = args.fp16 and args.device != "cpu" and torch.cuda.is_available()

    # ── Load data ─────────────────────────────────────────────────────────────
    # Train comes from the PER-DATASET pkls, tagged with `dataset` for the sampler's
    # level 1. Under --debug we truncate WITHIN each dataset so every dataset (and so
    # every level-1 group) survives; truncating the flat pool would leave one dataset.
    val_path = os.path.join(args.data_dir, "val_model_inputs.pkl")
    if args.debug:
        logger.info("DEBUG mode: truncating to the first 50 windows PER DATASET, "
                    "and val/test to the first 50 samples.")
    train_data = load_train_pool(args.data_dir, debug_per_dataset=50 if args.debug else 0)

    val_data = load_data(val_path, "val") if not args.no_validation else None
    # Optional third dataset, evaluated exactly like val (no weight updates).
    test_data = (
        load_data(args.test_data, "test")
        if (args.test_data and not args.no_validation)
        else None
    )
    if args.debug:
        if val_data is not None:
            val_data = val_data[:50]
        if test_data is not None:
            test_data = test_data[:50]

    # ── Load model ────────────────────────────────────────────────────────────
    logger.info(f"Loading {args.model_id} on {args.device}")
    pipeline: Chronos2Pipeline = BaseChronosPipeline.from_pretrained(
        args.model_id, device_map=args.device
    )

    # ── LoRA config ───────────────────────────────────────────────────────────
    lora_config = build_lora_config(args) if args.finetune_mode == "lora" else None
    if lora_config:
        logger.info(f"LoRA: r={args.lora_r}, alpha={args.lora_alpha}")

    # ── Build fit kwargs ──────────────────────────────────────────────────────
    fit_kwargs = dict(
        inputs=train_data,
        prediction_length=args.prediction_length,
        min_past=args.context_length,
        finetune_mode=args.finetune_mode,
        lora_config=lora_config,
        learning_rate=args.lr,
        num_steps=args.num_steps,
        batch_size=args.batch_size,
        context_length=args.context_length,
        output_dir=args.output_dir,
        logging_steps=args.logging_steps,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        fp16=use_fp16,
        warmup_ratio=args.warmup_ratio,
        lr_scheduler_type=args.lr_scheduler_type,
        trainer_cls=functools.partial(
            Chronos2SingleStageTrainer,
            margin_tau=args.margin_tau,
            margin_lambda=args.margin_lambda,
            agg_mode=args.agg_mode,
            hinge_mode=args.hinge_mode,
            margin_mode=args.margin_mode,
            margin_m=args.margin_m,
        ),
    )
    if val_data is not None:
        # When a test set is supplied, hand HF a dict of eval datasets. It then
        # evaluates each one every eval step and logs eval_val_* / eval_test_*
        # separately. Best-model selection uses the first key ("val").
        if test_data is not None:
            fit_kwargs["validation_inputs"] = {"val": val_data, "test": test_data}
        else:
            fit_kwargs["validation_inputs"] = val_data
        # Override the eval cadence hardcoded inside pipeline.fit (default 100).
        # save_steps stays at 100; HF requires save_steps % eval_steps == 0 when
        # load_best_model_at_end=True, so keep eval_steps a divisor of 100.
        fit_kwargs["eval_steps"] = args.eval_steps
    if args.enable_sep_token:
        if args.normal_signal_length % args.input_patch_size != 0:
            raise ValueError(
                f"--normal_signal_length ({args.normal_signal_length}) must be "
                f"a multiple of --input_patch_size ({args.input_patch_size})"
            )
        fit_kwargs["enable_sep_token"] = True
        fit_kwargs["sep_patch_index"] = args.normal_signal_length // args.input_patch_size

    # ── Train ─────────────────────────────────────────────────────────────────
    # pipeline.fit builds its dataset internally; swap in our subclass so that
    # per-sample future_labels survive into the trainer's compute_loss.
    chronos2_pipeline.Chronos2Dataset = AnomalyChronos2Dataset
    # Level 2 of the hierarchical sampler. The dataset is instantiated inside
    # pipeline.fit, so pass it via the class attribute.
    AnomalyChronos2Dataset.p_anom = args.p_anom

    logger.info(
        f"Single-stage per-step masked training: lr={args.lr}, steps={args.num_steps}, "
        f"batch={args.batch_size}, agg_mode={args.agg_mode}, hinge_mode={args.hinge_mode}, "
        f"margin_mode={args.margin_mode}, margin_m={args.margin_m}, margin_tau={args.margin_tau}, "
        f"margin_lambda={args.margin_lambda}, p_anom={args.p_anom:.4f} (hierarchical "
        f"sampler), fp16={use_fp16}"
    )
    # print("----------------Calling pipeline_fit from main()---------------")
    pipeline.fit(**fit_kwargs)


    ckpt_path = os.path.join(args.output_dir, "finetuned-ckpt")
    logger.info(f"Done. Checkpoint saved to {ckpt_path}")
    logger.info(f"Load with: BaseChronosPipeline.from_pretrained('{ckpt_path}')")


if __name__ == "__main__":
    main()
