"""
Single-stage anomaly-aware fine-tuning for Chronos-2.

Uses a SINGLE combined dataset (normal + anomaly future pairs mixed together).
The loss is a margin (hinge) objective conditioned on the future type of each sample:

    L_total = L_good + lambda * max(0, tau - L_bad)

    future_type == 0  (normal)  →  L_good : minimise (predict normal well)
    future_type == 1  (anomaly) →  L_bad  : push UP toward margin tau, then stop

The hinge self-saturates: once L_bad >= tau it adds no gradient, so training can't
diverge (this replaces the old clamp+negate gradient-ascent ceiling).

At inference: high prediction error ⟹ high anomaly score.

Usage:
    python finetune_anomaly_simple.py
    python finetune_anomaly_simple.py --margin_tau 12 --margin_lambda 1.0
    python finetune_anomaly_simple.py --finetune_mode full --lr 1e-6
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
    Chronos2Dataset that preserves a per-sample `future_type` (0=normal, 1=anomaly).

    The stock Chronos2Dataset validates inputs against {target, past_covariates,
    future_covariates} and drops everything else, so `future_type` would be lost.
    Here we strip it before the parent's validation/preparation, then re-attach it
    onto every batch the dataset yields — expanded to one entry per row so it lines
    up with `context`/`future_target`/`group_ids` (each input series contributes
    `group_size` rows). The single-stage trainer pops it back off in compute_loss.
    """

    def __init__(self, inputs, *args, **kwargs):
        future_types = [int(d.get("future_type", 0)) for d in inputs]
        # Per-timestep labels (length = prediction_length), 0=normal / 1=anomaly.
        # These drive the per-step masked loss; the window-level future_type is kept
        # only for backward-compatible logging.
        future_labels = [np.asarray(d["future_labels"], dtype=np.int64) for d in inputs]
        # Strip BOTH our extra keys before the parent's validation: `future_type`
        # (the window-level label) and `future_labels` (the per-timestep array).
        cleaned = [
            {k: v for k, v in d.items() if k not in ("future_type", "future_labels")}
            for d in inputs
        ]
        super().__init__(cleaned, *args, **kwargs)
        # Parent filters series shorter than min_past + prediction_length. Our fixed
        # 832-step targets are never filtered, so prepared inputs align 1:1 with
        # future_types. Guard loudly in case lengths change and some get dropped.
        if len(self.inputs) != len(future_types):
            raise RuntimeError(
                f"future_type alignment broke: {len(future_types)} inputs given but "
                f"{len(self.inputs)} survived length filtering. Ensure every target is "
                "at least min_past + prediction_length steps long."
            )

        for prepared, ft, fl in zip(self.inputs, future_types, future_labels):
            prepared["future_type"] = ft
            prepared["future_labels"] = fl

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

        L = L_good + lambda * max(0, tau - L_bad)

    Instead of collapsing each window to one good/bad bit via a count threshold, the
    per-timestep labels (`future_labels`, 0=normal / 1=anomaly) mask the model's
    per-step pinball loss (`Chronos2Output.per_step_loss`, shape (rows, horizon)):

      - L_good : mean per-step loss over NORMAL steps  -> minimised
      - L_bad  : mean per-step loss over ANOMALY steps -> pushed UP toward margin tau
                 (self-saturating hinge: no gradient once L_bad >= tau).

    This removes both contaminations — no normal step is ever pushed up, no anomaly
    step is ever trained as normal — and makes the count threshold unnecessary.
    A SINGLE forward pass over the whole batch is used; the split is by step, not by
    window, so both terms come from the same predictions.

    Aggregation (`agg_mode`) controls how the masked means are pooled:
      - "batch_global": pool ALL normal steps in the batch for one L_good and ALL
        anomaly steps for one L_bad, with a single hinge. Each *step* weighted equally
        (longer/denser windows contribute more); most stable across datasets.
      - "per_window": compute L_good_w / L_bad_w within each window, form
        L_w = L_good_w + lambda*hinge(tau - L_bad_w) per window, then average over
        windows. Each *window* weighted equally regardless of step count.

    Parameters
    ----------
    margin_tau : float
        Margin the anomaly loss is pushed toward. Per-step loss (summed over 9
        quantiles) sits on the same scale as the old per-window loss (~3-4 for
        well-predicted normal steps), so tau ~10-15 as before.
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
        loss_ceiling: float | None = None,  # accepted for back-compat; unused by hinge
        **kwargs,
    ):
        super().__init__(*args, loss_ceiling=loss_ceiling, **kwargs)
        if agg_mode not in ("batch_global", "per_window"):
            raise ValueError(
                f"agg_mode must be 'batch_global' or 'per_window', got {agg_mode!r}"
            )
        self.margin_tau = margin_tau
        self.margin_lambda = margin_lambda
        self.agg_mode = agg_mode
        # Running sums of the RAW per-step metrics, split by step label, so we can log
        # normal/anomaly means separately for monitoring:
        #   *_sum     -> summed pinball loss   -> normal_loss / anomaly_loss
        #   *_mse_sum -> summed squared error  -> mse_normal_step / mse_anomaly_step
        #   *_cnt     -> number of steps (shared denominator for both)
        # Train and eval kept apart.
        def _fresh():
            return {"n_sum": 0.0, "n_mse_sum": 0.0, "n_cnt": 0,
                    "a_sum": 0.0, "a_mse_sum": 0.0, "a_cnt": 0}
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

        acc = self._acc["train" if model.training else "eval"]
        # Accumulate pooled per-step means for monitoring (pinball + MSE), split by
        # step label. Informative regardless of agg_mode.
        n_norm_steps = normal_step.sum()
        n_anom_steps = anomaly_step.sum()
        acc["n_sum"] += (per_step * normal_step).sum().detach().item()
        acc["n_mse_sum"] += (per_step_mse * normal_step).sum().detach().item()
        acc["n_cnt"] += int(n_norm_steps.item())
        acc["a_sum"] += (per_step * anomaly_step).sum().detach().item()
        acc["a_mse_sum"] += (per_step_mse * anomaly_step).sum().detach().item()
        acc["a_cnt"] += int(n_anom_steps.item())

        eps = 1.0  # clamp denominators to avoid /0 on empty subsets

        if self.agg_mode == "batch_global":
            # Pool every step in the batch: each step weighted equally.
            L_good = (per_step * normal_step).sum() / n_norm_steps.clamp(min=eps)
            total_loss = L_good
            if n_anom_steps > 0:
                L_bad = (per_step * anomaly_step).sum() / n_anom_steps.clamp(min=eps)
                hinge = torch.clamp(self.margin_tau - L_bad, min=0.0)
                total_loss = total_loss + self.margin_lambda * hinge
        else:  # per_window
            # Masked means within each window, then average over windows.
            cnt_norm_w = normal_step.sum(dim=1)                       # (rows,)
            cnt_anom_w = anomaly_step.sum(dim=1)
            L_good_w = (per_step * normal_step).sum(dim=1) / cnt_norm_w.clamp(min=eps)
            L_bad_w  = (per_step * anomaly_step).sum(dim=1) / cnt_anom_w.clamp(min=eps)
            has_norm = (cnt_norm_w > 0).float()
            has_anom = (cnt_anom_w > 0).float()
            hinge_w = torch.clamp(self.margin_tau - L_bad_w, min=0.0) * has_anom
            per_window = L_good_w * has_norm + self.margin_lambda * hinge_w
            total_loss = per_window.mean()

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

        if phase == "eval":
            tag = prefix.rstrip("_").upper()  # "EVAL" / "EVAL_VAL" / "EVAL_TEST"
            print(f"\n======================================================")
            print(f" [{tag}] Normal step loss: {logs.get(f'{prefix}normal_loss', 'N/A')} | Anomaly step loss: {logs.get(f'{prefix}anomaly_loss', 'N/A')}")
            print(f"              MSE Normal step: {logs.get(f'{prefix}mse_normal_step', 'N/A')} | MSE Anomaly step: {logs.get(f'{prefix}mse_anomaly_step', 'N/A')}")
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
    p.add_argument("--data_dir", default="./prepared_data_labeled",
                   help="Output dir from inst_data_prepare_labeled.py "
                        "(must contain train_model_inputs.pkl / val_model_inputs.pkl)")
    p.add_argument("--anomaly_threshold", type=int, default=10,
                   help="A future window is labeled anomalous (future_type=1) iff it "
                        "contains at least this many anomalous timesteps; else normal.")
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

    # Margin (hinge) loss: L_good + lambda * max(0, tau - L_bad)
    p.add_argument("--margin_tau", type=float, default=12.0,
                   help="Margin the anomaly loss is pushed toward. Must sit ABOVE the "
                        "normal-point loss (~3-4 here) to matter — use ~10-15, NOT 2.")
    p.add_argument("--margin_lambda", type=float, default=1.0,
                   help="Weight on the anomaly (bad) term. Default 1.0.")
    p.add_argument("--agg_mode", default="batch_global",
                   choices=["batch_global", "per_window"],
                   help="How to pool the per-step masked losses. 'batch_global' pools "
                        "all steps (each step weighted equally, most stable across "
                        "datasets); 'per_window' averages per-window means (each window "
                        "weighted equally).")

    # Output
    p.add_argument("--output_dir", default="./chronos2-single-stage")

    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
#  Helpers
# ─────────────────────────────────────────────────────────────────────────────

def load_data(path: str, label: str) -> list[dict]:
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{label} data not found at {path}. "
            "Run inst_data_prepare_labeled.py first."
        )
    logger.info(f"Loading {label} from {path}")
    with open(path, "rb") as f:
        data = pickle.load(f)
    logger.info(f"  {len(data)} samples loaded")
    return data


def derive_future_type(data: list[dict], threshold: int, label: str) -> list[dict]:
    """
    Collapse the per-timestep `future_labels` array (length = prediction_length) into a
    single window-level `future_type` via a count threshold:

        future_type = 1 (anomaly)  if  (#anomalous steps in the window) >= threshold
        future_type = 0 (normal)   otherwise

    Mutates each dict in place, adding `future_type`. Samples that already carry a
    `future_type` and have no `future_labels` (e.g. old-format data) are left as-is.
    """
    n_anom = 0
    for d in data:
        labels = d.get("future_labels")
        if labels is not None:
            d["future_type"] = int(int(np.sum(labels)) >= threshold)
        n_anom += int(d.get("future_type", 0))
    logger.info(
        f"  {label}: threshold={threshold} ones/window -> "
        f"anomaly={n_anom}, normal={len(data) - n_anom}"
    )
    return data


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
    train_path = os.path.join(args.data_dir, "train_model_inputs.pkl")
    val_path   = os.path.join(args.data_dir, "val_model_inputs.pkl")

    train_data = load_data(train_path, "train")
    # print(train_data[0]['target'].shape)
    # print(train_data[0]['future_labels'].shape)
    
    val_data   = load_data(val_path, "val") if not args.no_validation else None
    # Optional third dataset, evaluated exactly like val (no weight updates).
    test_data = (
        load_data(args.test_data, "test")
        if (args.test_data and not args.no_validation)
        else None
    )
    if args.debug:
        logger.info("DEBUG mode: truncating train/val/test to the first 50 samples.")
        train_data = train_data[:50]
        if val_data is not None:
            val_data = val_data[:50]
        if test_data is not None:
            test_data = test_data[:50]

    # Collapse per-timestep future_labels -> window-level future_type via the threshold.
    derive_future_type(train_data, args.anomaly_threshold, "train")
    if val_data is not None:
        derive_future_type(val_data, args.anomaly_threshold, "val")
    if test_data is not None:
        derive_future_type(test_data, args.anomaly_threshold, "test")

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
    # per-sample future_type survives into the trainer's compute_loss.
    chronos2_pipeline.Chronos2Dataset = AnomalyChronos2Dataset

    logger.info(
        f"Single-stage per-step masked training: lr={args.lr}, steps={args.num_steps}, "
        f"batch={args.batch_size}, agg_mode={args.agg_mode}, margin_tau={args.margin_tau}, "
        f"margin_lambda={args.margin_lambda}, fp16={use_fp16}"
    )
    # print("----------------Calling pipeline_fit from main()---------------")
    pipeline.fit(**fit_kwargs)


    ckpt_path = os.path.join(args.output_dir, "finetuned-ckpt")
    logger.info(f"Done. Checkpoint saved to {ckpt_path}")
    logger.info(f"Load with: BaseChronosPipeline.from_pretrained('{ckpt_path}')")


if __name__ == "__main__":
    main()
