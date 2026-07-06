"""
Estimate num_steps for a target number of 'epochs-equivalent' on the combined
train pkl, accounting for how Chronos-2 actually samples.

Chronos2Dataset (TRAIN mode) is an INFINITE IterableDataset that draws windows
uniformly at random WITH REPLACEMENT, and `batch_size` counts variate-ROWS, not
windows (it adds windows until the summed feature-count reaches batch_size). So:

    windows_per_step = batch_size * grad_accum / mean_F
    num_steps(K eps) = K * N * mean_F / (batch_size * grad_accum)
                     = K * total_rows / (batch_size * grad_accum)

Coverage (coupon-collector, sampling w/ replacement): after K epochs-equivalent the
fraction of windows never sampled is ~ exp(-K).

Usage:
    python estimate_steps.py                       # defaults: bs=160 ga=2 epochs=10
    python estimate_steps.py --batch_size 256 --grad_accum 2 --epochs 5
    python estimate_steps.py --pkl prepared_total/train_model_inputs.pkl
"""
import argparse, math, os, pickle
import numpy as np

p = argparse.ArgumentParser()
p.add_argument("--pkl", default=os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                             "prepared_total", "train_model_inputs.pkl"))
p.add_argument("--batch_size", type=int, default=160, help="ROWS per micro-batch (Chronos-2 sense)")
p.add_argument("--grad_accum", type=int, default=2)
p.add_argument("--epochs", type=float, nargs="*", default=[3, 5, 10])
args = p.parse_args()

with open(args.pkl, "rb") as f:
    data = pickle.load(f)

N = len(data)
Fs = np.array([d["target"].shape[0] for d in data])
R = int(Fs.sum())
mean_F = R / N
eff = args.batch_size * args.grad_accum                    # rows per optimizer step
windows_per_step = eff / mean_F

print(f"pkl                : {args.pkl}")
print(f"train windows  N   : {N:,}")
print(f"mean_F             : {mean_F:.2f}   (min={Fs.min()}, max={Fs.max()})")
print(f"total rows     R   : {R:,}")
print(f"rows/step (bs*ga)  : {eff}   -> windows/step ≈ {windows_per_step:.1f}")
print("-" * 56)
print(f"{'epochs':>8} | {'num_steps':>10} | {'window coverage':>16}")
for K in args.epochs:
    steps = math.ceil(K * R / eff)
    coverage = 1 - math.exp(-K)
    print(f"{K:>8} | {steps:>10,} | {coverage*100:>14.3f} %")
