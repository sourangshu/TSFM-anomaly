
import os
import sys
import argparse
import numpy as np
import pandas as pd
import torch
import pickle
from datetime import datetime
from sklearn.metrics import roc_auc_score, average_precision_score, confusion_matrix
import timesfm

# ============================================================
# Command-line arguments
# ============================================================
parser = argparse.ArgumentParser()
parser.add_argument("--context", type=int, default=128,
                    help="Context length (default: 128)")
parser.add_argument("--horizon", type=int, default=20,
                    help="Horizon length (default: 20)")
parser.add_argument("--topk", type=int, default=3,
                    help="Top-K features for aggregation (default: 3)")
parser.add_argument("--dataset", type=str, default="3_4_1000000_81",
                    help="Dataset number (default: 3_4_1000000_81)")
args = parser.parse_args()

CONTEXT_LEN = args.context
HORIZON_LEN = args.horizon
TOP_K = args.topk
DATASET_ID = args.dataset

# Get current date for folder organization
CURRENT_DATE = datetime.now().strftime("%Y-%m-%d")  # Format: 2024-01-15

print(f"\n{'='*70}")
print(f" TimeFM Experiment Configuration")
print(f"{'='*70}")
print(f"Date           : {CURRENT_DATE}")
print(f"Context Length : {CONTEXT_LEN}")
print(f"Horizon Length : {HORIZON_LEN}")
print(f"Top-K Features : {TOP_K}")
print(f"Dataset ID     : {DATASET_ID}")
print(f"{'='*70}")

# ============================================================
# Import project utilities
# ============================================================
sys.path.append("/home/paramjeet/times-fm/src")
from anomaly_timesfm import compute_timesfm_anomaly_scores

# ============================================================
# Config with Date-wise Organization
# ============================================================
DATA_DIR = "/home/paramjeet/times-fm/datasets/Exathlon"
BASE_EXPERIMENT_DIR = "/home/paramjeet/times-fm/experiments"

# Create date folder
DATE_DIR = os.path.join(BASE_EXPERIMENT_DIR, CURRENT_DATE)
os.makedirs(DATE_DIR, exist_ok=True)

# Create dataset folder inside date folder
DATASET_DIR = os.path.join(DATE_DIR, f"Exathlon_{DATASET_ID}")
os.makedirs(DATASET_DIR, exist_ok=True)

# Create subfolders for different file types
CHECKPOINT_DIR = os.path.join(DATASET_DIR, "checkpoints")
VIZ_DIR = os.path.join(DATASET_DIR, "visualization")
METRICS_DIR = os.path.join(DATASET_DIR, "metrics")
SUMMARY_DIR = os.path.join(DATASET_DIR, "summary")

os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(VIZ_DIR, exist_ok=True)
os.makedirs(METRICS_DIR, exist_ok=True)
os.makedirs(SUMMARY_DIR, exist_ok=True)

print(f"\n Files will be saved in:")
print(f"   {DATASET_DIR}")

TRAIN_FILE = f"Exathlon_{DATASET_ID}_train.csv"
TEST_FILE = f"Exathlon_{DATASET_ID}_test.csv"

TRAIN_PATH = os.path.join(DATA_DIR, TRAIN_FILE)
TEST_PATH = os.path.join(DATA_DIR, TEST_FILE)

ts_name = TRAIN_FILE.replace("_train.csv", "")

# ============================================================
# Step 1: Load raw data
# ============================================================
print("\n Loading data...")
train_df = pd.read_csv(TRAIN_PATH)
test_df = pd.read_csv(TEST_PATH)

print(f"\nTrain shape: {train_df.shape}")
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

print("\n🛠️ Preprocessing...")
imputer, X_train, y_train = preprocess_df(train_df, fit=True)
X_test, y_test = preprocess_df(test_df, imputer=imputer, fit=False)

print(f"\n After preprocessing:")
print(f"X_train: {X_train.shape}")
print(f"X_test : {X_test.shape}")
print(f"Anomaly ratio (test): {y_test.mean():.4f}")

# ============================================================
# Step 3: Load TimeFM model
# ============================================================
torch.set_float32_matmul_precision("high")
print("\n Loading TimeFM model...")

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

print("✅ Model loaded and compiled.")

# ============================================================
# Step 4: Compute anomaly scores
# ============================================================
print("\n📊 Computing anomaly scores...")

results = compute_timesfm_anomaly_scores(
    model=model,
    X_train=X_train,
    X_test=X_test,
    y_test=y_test,
    context_len=CONTEXT_LEN,
    horizon=HORIZON_LEN,
    top_k=TOP_K,
    return_viz=True,
    viz_subsample=5  # Har 5th window store karo (memory efficient)
)

# ============================================================
# Step 5: Calculate metrics for each aggregation method
# ============================================================
print("\n📈 Calculating detailed metrics...")

valid_start = CONTEXT_LEN
valid_end = len(y_test) - HORIZON_LEN
y_valid = y_test[valid_start:valid_end]

all_metrics = {}

# for method in ["max", "topk", "l2", #"mse" --- IGNORE ---
for method in ["max", "topk", "l2", "mae"]:
    scores = results[method][valid_start:valid_end]
    
    # AUROC
    auroc = roc_auc_score(y_valid, scores)
    
    # AUPRC
    auprc = average_precision_score(y_valid, scores)
    
    # Best threshold (Youden's J)
    from sklearn.metrics import roc_curve
    fpr, tpr, thresholds = roc_curve(y_valid, scores)
    youden_j = tpr - fpr
    best_idx = np.argmax(youden_j)
    best_threshold = thresholds[best_idx]
    
    # Confusion Matrix
    y_pred = (scores > best_threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_valid, y_pred).ravel()
    
    # Precision, Recall, F1
    precision = tp/(tp+fp) if (tp+fp)>0 else 0
    recall = tp/(tp+fn) if (tp+fn)>0 else 0
    f1 = 2*tp/(2*tp+fp+fn) if (2*tp+fp+fn)>0 else 0
    
    all_metrics[method] = {
        'auroc': auroc,
        'auprc': auprc,
        'best_threshold': best_threshold,
        'tn': tn, 'fp': fp, 'fn': fn, 'tp': tp,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }

# ============================================================
# Step 6: Print results
# ============================================================
print(f"\n{'='*70}")
print("📊 FINAL RESULTS - All Aggregation Methods")
print(f"{'='*70}")

best_method = None
best_auroc = 0

for method, metrics in all_metrics.items():
    print(f"\n🔸 Method: {method.upper()}")
    print(f"   AUROC     : {metrics['auroc']:.4f}")
    print(f"   AUPRC     : {metrics['auprc']:.4f}")
    print(f"   F1-Score  : {metrics['f1']:.4f}")
    print(f"   Precision : {metrics['precision']:.4f}")
    print(f"   Recall    : {metrics['recall']:.4f}")
    
    if metrics['auroc'] > best_auroc:
        best_auroc = metrics['auroc']
        best_method = method

print(f"\n{'='*70}")
print(f"✅ BEST METHOD: {best_method.upper()} (AUROC={best_auroc:.4f})")
print(f"{'='*70}")

# ============================================================
# Step 7: Save everything with proper naming
# ============================================================
print("\n💾 Saving data...")

# Base filename without extension
base_filename = f"ctx{CONTEXT_LEN}_hor{HORIZON_LEN}_topk{TOP_K}"

# 1. Save all anomaly scores in checkpoints folder
# for method in ["max", "topk", "l2", "mse"]:
for method in ["max", "topk", "l2", "mae"]:
    score_path = os.path.join(CHECKPOINT_DIR, f"{base_filename}_{method}_scores.npy")
    np.save(score_path, results[method])
print(f" Anomaly scores saved in: {CHECKPOINT_DIR}")

# 2. Save visualization data in visualization folder
viz_path = os.path.join(VIZ_DIR, f"{base_filename}_viz.pkl")
with open(viz_path, 'wb') as f:
    pickle.dump({
        'viz_data': results['viz_data'],
        'context_len': CONTEXT_LEN,
        'horizon': HORIZON_LEN,
        'top_k': TOP_K,
        'date': CURRENT_DATE,
        'dataset': DATASET_ID,
        'feature_names': [f"F{i}" for i in range(X_test.shape[1])],
        'best_method': best_method,
        'all_metrics': all_metrics
    }, f)
print(f" Visualization data: {viz_path}")

# 3. Save all metrics in metrics folder
metrics_path = os.path.join(METRICS_DIR, f"{base_filename}_all_metrics.csv")
metrics_rows = []
for method, m in all_metrics.items():
    row = {'method': method, 'date': CURRENT_DATE, 'dataset': DATASET_ID}
    row.update(m)
    metrics_rows.append(row)

metrics_df = pd.DataFrame(metrics_rows)
metrics_df.to_csv(metrics_path, index=False)
print(f"Metrics CSV: {metrics_path}")

# 4. Save detailed summary in summary folder
summary_path = os.path.join(SUMMARY_DIR, f"{base_filename}_summary.txt")
with open(summary_path, 'w') as f:
    f.write("="*60 + "\n")
    f.write("TIMEFM EXPERIMENT SUMMARY\n")
    f.write("="*60 + "\n\n")
    f.write(f"Date    : {CURRENT_DATE}\n")
    f.write(f"Dataset : Exathlon_{DATASET_ID}\n")
    f.write(f"Context : {CONTEXT_LEN}\n")
    f.write(f"Horizon : {HORIZON_LEN}\n")
    f.write(f"Top-K   : {TOP_K}\n\n")
    
    f.write("="*60 + "\n")
    f.write("BEST METHOD\n")
    f.write("="*60 + "\n")
    f.write(f"Method  : {best_method.upper()}\n")
    f.write(f"AUROC   : {best_auroc:.4f}\n\n")
    
    f.write("="*60 + "\n")
    f.write("ALL METHODS\n")
    f.write("="*60 + "\n")
    for method, m in all_metrics.items():
        f.write(f"\n{method.upper()}:\n")
        f.write(f"  AUROC     : {m['auroc']:.4f}\n")
        f.write(f"  AUPRC     : {m['auprc']:.4f}\n")
        f.write(f"  F1-Score  : {m['f1']:.4f}\n")
        f.write(f"  Precision : {m['precision']:.4f}\n")
        f.write(f"  Recall    : {m['recall']:.4f}\n")
        f.write(f"  Threshold : {m['best_threshold']:.6f}\n")
        f.write(f"  TN: {m['tn']}, FP: {m['fp']}, FN: {m['fn']}, TP: {m['tp']}\n")
    
    f.write("\n" + "="*60 + "\n")
print(f" Summary: {summary_path}")

# 5. Save experiment config for reference
config_path = os.path.join(DATASET_DIR, "experiment_config.txt")
with open(config_path, 'w') as f:
    f.write(f"Experiment Run: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    f.write(f"Date Folder: {CURRENT_DATE}\n")
    f.write(f"Dataset: Exathlon_{DATASET_ID}\n")
    f.write(f"Context Length: {CONTEXT_LEN}\n")
    f.write(f"Horizon Length: {HORIZON_LEN}\n")
    f.write(f"Top-K: {TOP_K}\n")
    f.write(f"Best Method: {best_method}\n")
    f.write(f"Best AUROC: {best_auroc:.4f}\n")

print(f"\n{'='*70}")
print(f" Experiment Complete!")
print(f" All files saved in: {DATASET_DIR}")
print(f"{'='*70}")

# Usage examples:
# python Exathlon_timesfm.py --context 128 --horizon 20 --topk 3 --dataset 3_4_1000000_81
# python Exathlon_timesfm.py --context 256 --horizon 10 --topk 5 --dataset 1_2_100000_68
