import os
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    roc_curve,
    roc_auc_score,
    precision_recall_curve,
    average_precision_score,
    precision_score,
    recall_score,
    f1_score
)

# ==============================
# PATHS (CHANGE THESE)
# ==============================
DATA_DIR = "/home/paramjeet/times-fm/datasets/Exathlon"
TEST_FILE = "Exathlon_10_2_1000000_67_test.csv"

SCORES_PATH = "/home/paramjeet/times-fm/experiments/checkpoints/Exathlon_10_2_1000000_67_ctx256_hor10_scores.npy"
# SCORES_PATH = "/home/paramjeet/times-fm/experiments/exathlon_chronos_results/checkpoints/Exathlon_context512_h1_scores.npy"

CONTEXT_LEN = 256   # IMPORTANT: 

# ==============================
#  Reload test data
# ==============================
test_df = pd.read_csv(os.path.join(DATA_DIR, TEST_FILE))

y_test = test_df["is_anomaly"].values

# ==============================
#  Load anomaly scores
# ==============================
anomaly_scores = np.load(SCORES_PATH)

# Align (same as original script)
y_valid = y_test[CONTEXT_LEN:]
scores_valid = anomaly_scores[CONTEXT_LEN:]

# ==============================
#  Compute metrics
# ==============================
fpr, tpr, thresholds = roc_curve(y_valid, scores_valid)
auroc = roc_auc_score(y_valid, scores_valid)

precision, recall, pr_thresholds = precision_recall_curve(
    y_valid, scores_valid
)
auprc = average_precision_score(y_valid, scores_valid)

# ==============================
#  Best threshold
# ==============================
youden_j = tpr - fpr
best_idx = np.argmax(youden_j)
best_threshold = thresholds[best_idx]

y_pred = (scores_valid >= best_threshold).astype(int)

precision_best = precision_score(y_valid, y_pred)
recall_best = recall_score(y_valid, y_pred)
f1_best = f1_score(y_valid, y_pred)

# ==============================
# PRINT
# ==============================
print("\n===== RECOVERED RESULTS =====")
print(f"AUROC: {auroc:.4f}")
print(f"AUPRC: {auprc:.4f}")
print(f"Best Threshold: {best_threshold:.6f}")
print(f"Best TPR: {tpr[best_idx]:.4f}")
print(f"Best FPR: {fpr[best_idx]:.4f}")
print(f"Precision: {precision_best:.4f}")
print(f"Recall: {recall_best:.4f}")
print(f"F1-score: {f1_best:.4f}")
