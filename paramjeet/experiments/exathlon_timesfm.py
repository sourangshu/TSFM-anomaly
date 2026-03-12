# experiments/Exathlon_timesfm.py

# experiments/Exathlon_timesfm.py

import os
import sys
import argparse
import numpy as np
import pandas as pd
import torch
import pickle
from sklearn.metrics import roc_auc_score, average_precision_score, confusion_matrix
import timesfm

# ============================================================
# Command-line arguments
# ============================================================
parser = argparse.ArgumentParser()
parser.add_argument("--context", type=int, default=128,
                    help="Context length (default: 512)")
parser.add_argument("--horizon", type=int, default=20,
                    help="Horizon length (default: 20)")
args = parser.parse_args()

CONTEXT_LEN = args.context
HORIZON_LEN = args.horizon

print(f"\n{'='*50}")
print(f"Using CONTEXT_LEN = {CONTEXT_LEN}")
print(f"Using HORIZON_LEN = {HORIZON_LEN}")
print(f"{'='*50}")

# ============================================================
# Import project utilities
# ============================================================
sys.path.append("/home/paramjeet/times-fm/src")
from anomaly_timesfm import compute_timesfm_anomaly_scores

# ============================================================
# Config
# ============================================================
DATA_DIR = "/home/paramjeet/times-fm/datasets/Exathlon"
OUTPUT_DIR = "/home/paramjeet/times-fm/experiments"

CHECKPOINT_DIR = os.path.join(OUTPUT_DIR, "checkpoints")
VIZ_DIR = os.path.join(OUTPUT_DIR, "visualization")  # New directory for viz data

os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(VIZ_DIR, exist_ok=True)

TRAIN_FILE = "Exathlon_3_4_1000000_81_train.csv"
TEST_FILE  = "Exathlon_3_4_1000000_81_test.csv"

TRAIN_PATH = os.path.join(DATA_DIR, TRAIN_FILE)
TEST_PATH  = os.path.join(DATA_DIR, TEST_FILE)

ts_name = TRAIN_FILE.replace("_train.csv", "")

# ============================================================
# Step 1: Load raw data
# ============================================================
print("\n Loading data...")
train_df = pd.read_csv(TRAIN_PATH)
test_df  = pd.read_csv(TEST_PATH)

print("\n=== Raw data ===")
print(f"Train shape: {train_df.shape}")
print(f"Test shape : {test_df.shape}")

# ============================================================
# Step 2: Preprocessing (NO SCALING)
# ============================================================
from sklearn.impute import SimpleImputer

def preprocess_df(df, imputer=None, fit=False):
    df = df.copy()
    
    if "timestamp" in df.columns:
        df = df.drop(columns=["timestamp"])
    
    if "is_anomaly" in df.columns:
        y = df["is_anomaly"].values
        X = df.drop(columns=["is_anomaly"])
    else:
        y = None
        X = df
    
    X = X.replace(-1, np.nan)
    
    if fit:
        imputer = SimpleImputer(strategy="median")
        X = imputer.fit_transform(X)
        return imputer, X, y
    else:
        X = imputer.transform(X)
        return X, y

print("\nPreprocessing...")
imputer, X_train, y_train = preprocess_df(train_df, fit=True)
X_test, y_test = preprocess_df(test_df, imputer=imputer, fit=False)

print(f"\nAfter preprocessing:")
print(f"X_train: {X_train.shape}")
print(f"X_test : {X_test.shape}")
print(f"Anomaly ratio (test): {y_test.mean():.4f}")

# ============================================================
# Step 3: Load TimeFM model
# ============================================================
torch.set_float32_matmul_precision("high")
print("\nLoading TimeFM model...")

model = timesfm.TimesFM_2p5_200M_torch.from_pretrained(
    "google/timesfm-2.5-200m-pytorch"
)

model.compile(
    forecast_config=timesfm.ForecastConfig(
        max_context=CONTEXT_LEN,
        max_horizon=HORIZON_LEN,
        normalize_inputs=True,
        use_continuous_quantile_head=True,
        force_flip_invariance=True,
        infer_is_positive=True,
        fix_quantile_crossing=True,
    )
)

print(" Model loaded and compiled.")

# ============================================================
# Step 4: Compute anomaly scores WITH visualization data
# ============================================================
print("\n Computing anomaly scores...")

# Modified function call to also return visualization data
anomaly_scores, auroc, viz_data = compute_timesfm_anomaly_scores(
    model=model,
    X_train=X_train,
    X_test=X_test,
    y_test=y_test,
    context_len=CONTEXT_LEN,
    horizon=HORIZON_LEN,
    return_viz=True  # New flag to return visualization data
)

# ============================================================
# Step 5: Calculate all metrics
# ============================================================
print("\nCalculating metrics...")

# Use only valid indices (after context)
valid_start = CONTEXT_LEN
valid_end = len(y_test) - HORIZON_LEN
y_valid = y_test[valid_start:valid_end]
scores_valid = anomaly_scores[valid_start:valid_end]

# AUROC
auroc = roc_auc_score(y_valid, scores_valid)

# AUPRC (Precision-Recall AUC)
auprc = average_precision_score(y_valid, scores_valid)

# Find best threshold using Youden's J
from sklearn.metrics import roc_curve
fpr, tpr, thresholds = roc_curve(y_valid, scores_valid)
youden_j = tpr - fpr
best_idx = np.argmax(youden_j)
best_threshold = thresholds[best_idx]

# Confusion Matrix at best threshold
y_pred = (scores_valid > best_threshold).astype(int)
tn, fp, fn, tp = confusion_matrix(y_valid, y_pred).ravel()

# ============================================================
# Step 6: Print results
# ============================================================
print(f"\n{'='*60}")
print("FINAL RESULTS")
print(f"{'='*60}")
print(f"\n🔹 AUROC : {auroc:.4f}")
print(f"🔹 AUPRC : {auprc:.4f}")
print(f"\n🔹 Best Threshold (Youden's J): {best_threshold:.6f}")
print(f"\n🔹 Confusion Matrix at best threshold:")
print(f"   True Negatives : {tn:5d}")
print(f"   False Positives: {fp:5d}")
print(f"   False Negatives: {fn:5d}")
print(f"   True Positives : {tp:5d}")
print(f"\n🔹 Derived Metrics:")
print(f"   Precision: {tp/(tp+fp):.4f}" if (tp+fp)>0 else "   Precision: N/A")
print(f"   Recall   : {tp/(tp+fn):.4f}" if (tp+fn)>0 else "   Recall: N/A")
print(f"   F1-Score : {2*tp/(2*tp+fp+fn):.4f}" if (2*tp+fp+fn)>0 else "   F1-Score: N/A")
print(f"{'='*60}")

# ============================================================
# Step 7: Save everything
# ============================================================
print("\nSaving data...")

# 1. Save anomaly scores
score_path = os.path.join(
    CHECKPOINT_DIR,
    f"{ts_name}_ctx{CONTEXT_LEN}_hor{HORIZON_LEN}_scores.npy"
)
np.save(score_path, anomaly_scores)
print(f"Anomaly scores: {score_path}")

# 2. Save visualization data (context + forecast windows)
viz_path = os.path.join(
    VIZ_DIR,
    f"{ts_name}_ctx{CONTEXT_LEN}_hor{HORIZON_LEN}_viz.pkl"
)
with open(viz_path, 'wb') as f:
    pickle.dump({
        'viz_data': viz_data,
        'context_len': CONTEXT_LEN,
        'horizon': HORIZON_LEN,
        'feature_names': [f"F{i}" for i in range(X_test.shape[1])]
    }, f)
print(f" Visualization data: {viz_path}")

# 3. Save metrics as text
metrics_path = os.path.join(
    CHECKPOINT_DIR,
    f"{ts_name}_ctx{CONTEXT_LEN}_hor{HORIZON_LEN}_metrics.txt"
)
with open(metrics_path, 'w') as f:
    f.write(f"AUROC: {auroc:.6f}\n")
    f.write(f"AUPRC: {auprc:.6f}\n")
    f.write(f"Best Threshold: {best_threshold:.6f}\n")
    f.write(f"Confusion Matrix:\n")
    f.write(f"  TN: {tn}\n")
    f.write(f"  FP: {fp}\n")
    f.write(f"  FN: {fn}\n")
    f.write(f"  TP: {tp}\n")
print(f"Metrics: {metrics_path}")

# 4. Save summary as CSV
summary_path = os.path.join(
    CHECKPOINT_DIR,
    f"{ts_name}_ctx{CONTEXT_LEN}_hor{HORIZON_LEN}_summary.csv"
)
summary_df = pd.DataFrame([{
    'context_len': CONTEXT_LEN,
    'horizon': HORIZON_LEN,
    'auroc': auroc,
    'auprc': auprc,
    'best_threshold': best_threshold,
    'tn': tn,
    'fp': fp,
    'fn': fn,
    'tp': tp,
    'precision': tp/(tp+fp) if (tp+fp)>0 else None,
    'recall': tp/(tp+fn) if (tp+fn)>0 else None,
    'f1': 2*tp/(2*tp+fp+fn) if (2*tp+fp+fn)>0 else None
}])
summary_df.to_csv(summary_path, index=False)
print(f"Summary CSV: {summary_path}")


# print(f"\n Visualization data contains {len(viz_data)} windows")
# print(f"   Each window has:")
# print(f"   - timestamp: current time step")
# print(f"   - context_values: last {CONTEXT_LEN} points")
# print(f"   - forecast_values: next {HORIZON_LEN} predictions")
# print(f"   - actual_values: actual next {HORIZON_LEN} points")
# print(f"   - anomaly_score: score for this window")
# print(f"   - is_anomaly: true label")

# print(f"\n{'='*60}")
# print(f"Experiment complete!")
# print(f"{'='*60}")
# Usage:
# python exathlon_timesfm.py --context 512 --horizon 20