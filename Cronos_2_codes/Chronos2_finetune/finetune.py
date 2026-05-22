"""
Single-stage anomaly-aware fine-tuning for Chronos-2.

Uses a SINGLE combined dataset (normal + anomaly future pairs mixed together).
The loss is conditioned on the future type of each sample within the batch:

    future_type == 0  (normal)  →  loss is MINIMISED  (gradient descent)
    future_type == 1  (anomaly) →  loss is MAXIMISED  (gradient ascent)

Effective batch loss:
    L = (1/B) * [ Σ_{normal} loss_i  -  Σ_{anomaly} clamp(loss_i, max=ceiling) ]

At inference: high prediction error ⟹ high anomaly score.

Usage:
    python finetune_anomaly_simple.py
    python finetune_anomaly_simple.py --finetune_mode lora --lora_r 16
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

log_path = os.path.join("./prepared_data_simple/log", "finetune_simple.log")
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
        cleaned = [{k: v for k, v in d.items() if k != "future_type"} for d in inputs]
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
        for prepared, ft in zip(self.inputs, future_types):
            prepared["future_type"] = ft

    def _build_batch(self, input_indices):
        batch = super()._build_batch(input_indices)
        row_future_types = []
        for input_idx in input_indices:
            group_size = self.inputs[input_idx]["context"].shape[0]
            row_future_types.extend([self.inputs[input_idx]["future_type"]] * group_size)
        batch["future_type"] = torch.tensor(row_future_types, dtype=torch.long)
        return batch


# ─────────────────────────────────────────────────────────────────────────────
#  Single-Stage Trainer
# ─────────────────────────────────────────────────────────────────────────────

class Chronos2SingleStageTrainer(Chronos2AnomalyTrainer):
    """
    Single-stage trainer that conditions the loss sign on future_type per sample.

    For each batch:
      - normal-future samples  (future_type==0): contribute +loss  (minimise)
      - anomaly-future samples (future_type==1): contribute -clamp(loss, ceiling)
                                                              (maximise, bounded)

    The two sub-batches are forwarded separately so their losses are independent.
    Final loss is the weighted mean (weighted by sub-batch size) so the gradient
    magnitude is comparable regardless of the normal/anomaly ratio in the batch.
    """

    def __init__(self, *args, loss_ceiling: float | None = 15.0, **kwargs):
        super().__init__(*args, loss_ceiling=loss_ceiling, **kwargs)
        self.loss_ceiling = loss_ceiling
        # Running (sample-weighted) sums of the RAW per-future-type losses, so we can
        # log normal_loss / anomaly_loss separately. Train and eval are kept apart.
        self._acc = {
            "train": {"n_sum": 0.0, "n_cnt": 0, "a_sum": 0.0, "a_cnt": 0},
            "eval":  {"n_sum": 0.0, "n_cnt": 0, "a_sum": 0.0, "a_cnt": 0},
        }

    @staticmethod
    def _select(inputs: dict, mask: torch.Tensor) -> dict:
        """Return a subset of the batch using a boolean mask."""
        return {
            k: v[mask] if isinstance(v, torch.Tensor) and v.shape[0] == mask.shape[0] else v
            for k, v in inputs.items()
        }

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        future_type = inputs.pop("future_type")          # (B,) — not passed to model
        normal_mask  = future_type == 0
        anomaly_mask = future_type == 1
        n_normal  = normal_mask.sum().item()
        n_anomaly = anomaly_mask.sum().item()
        n_total   = future_type.shape[0]

        total_loss = torch.zeros(1, device=future_type.device, requires_grad=True)
        outputs = None
        acc = self._acc["train" if model.training else "eval"]

        # ── Normal sub-batch: minimise loss ──────────────────────────────────
        if n_normal > 0:
            normal_out = model(**self._select(inputs, normal_mask))
            weight = n_normal / n_total
            total_loss = total_loss + weight * normal_out.loss
            outputs = normal_out
            acc["n_sum"] += normal_out.loss.detach().item() * n_normal
            acc["n_cnt"] += n_normal

        # ── Anomaly sub-batch: maximise loss (gradient ascent) ────────────────
        if n_anomaly > 0:
            anom_out  = model(**self._select(inputs, anomaly_mask))
            anom_loss = anom_out.loss
            # Log the RAW (unclamped, un-negated) prediction error on anomaly futures
            # so the curve shows the model's true error — which we want to go UP.
            acc["a_sum"] += anom_loss.detach().item() * n_anomaly
            acc["a_cnt"] += n_anomaly
            if self.loss_ceiling is not None:
                anom_loss = torch.clamp(anom_loss, max=self.loss_ceiling)
            weight = n_anomaly / n_total
            total_loss = total_loss - weight * anom_loss
            if outputs is None:
                outputs = anom_out

        return (total_loss, outputs) if return_outputs else total_loss

    def log(self, logs: dict, *args, **kwargs):
        """Inject separate normal/anomaly loss means into trainer_state.json log_history."""
        # An eval log carries "eval_loss"; a training log carries "loss".
        phase = "eval" if any(k.startswith("eval_") for k in logs) else "train"
        acc = self._acc[phase]
        prefix = "eval_" if phase == "eval" else ""
        if acc["n_cnt"] > 0:
            logs[f"{prefix}normal_loss"] = acc["n_sum"] / acc["n_cnt"]
        if acc["a_cnt"] > 0:
            logs[f"{prefix}anomaly_loss"] = acc["a_sum"] / acc["a_cnt"]
        self._acc[phase] = {"n_sum": 0.0, "n_cnt": 0, "a_sum": 0.0, "a_cnt": 0}
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
    p.add_argument("--data_dir", default="./prepared_data_simple",
                   help="Output dir from inst_data_prepare_simple.py "
                        "(must contain train_model_inputs.pkl / val_model_inputs.pkl)")
    p.add_argument("--no_validation", action="store_true")

    # Fine-tuning mode
    p.add_argument("--finetune_mode", default="lora", choices=["full", "lora"])
    p.add_argument("--lora_r",        type=int,   default=16)
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
    p.add_argument("--warmup_ratio",      type=float, default=0.05)
    p.add_argument("--lr_scheduler_type", default="cosine",
                   choices=["linear", "cosine", "cosine_with_restarts", "constant"])

    # Loss ceiling for anomaly gradient ascent
    p.add_argument("--loss_ceiling", type=float, default=15.0,
                   help="Cap anomaly loss before negating. Prevents divergence. "
                        "Set 0 to disable.")

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
            "Run inst_data_prepare_simple.py first."
        )
    logger.info(f"Loading {label} from {path}")
    with open(path, "rb") as f:
        data = pickle.load(f)
    n_anom = sum(d.get("future_type", 0) for d in data)
    logger.info(f"  {len(data)} samples — normal={len(data)-n_anom}, anomaly={n_anom}")
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
        args.lr = 1e-5 if args.finetune_mode == "lora" else 1e-6
    use_fp16 = args.fp16 and args.device != "cpu" and torch.cuda.is_available()
    loss_ceiling = args.loss_ceiling if args.loss_ceiling > 0 else None

    # ── Load data ─────────────────────────────────────────────────────────────
    train_path = os.path.join(args.data_dir, "train_model_inputs.pkl")
    val_path   = os.path.join(args.data_dir, "val_model_inputs.pkl")

    train_data = load_data(train_path, "train")
    val_data   = load_data(val_path, "val") if not args.no_validation else None

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
            Chronos2SingleStageTrainer, loss_ceiling=loss_ceiling
        ),
    )
    if val_data is not None:
        fit_kwargs["validation_inputs"] = val_data
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
        f"Single-stage training: lr={args.lr}, steps={args.num_steps}, "
        f"batch={args.batch_size}, loss_ceiling={loss_ceiling}, fp16={use_fp16}"
    )
    pipeline.fit(**fit_kwargs)

    ckpt_path = os.path.join(args.output_dir, "finetuned-ckpt")
    logger.info(f"Done. Checkpoint saved to {ckpt_path}")
    logger.info(f"Load with: BaseChronosPipeline.from_pretrained('{ckpt_path}')")


if __name__ == "__main__":
    main()
