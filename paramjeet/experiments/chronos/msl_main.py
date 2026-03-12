# ============================================================
# experiments/L_chronos_multits.py
# ============================================================

import os
import time
import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
import logging
import sys

# --------------------------------------------------
# Add src path if needed
# --------------------------------------------------
sys.path.append("/home/paramjeet/times-fm/src")
from anomaly_chronos import create_windows, compute_chronos2_multivariate_anomaly_scores
from preprocess import fit_preprocessor, transform_preprocessor  # 

# --------------------------------------------------
# Config
# --------------------------------------------------
DATA_DIR = "/home/paramjeet/times-fm/datasets/MSL"
OUTPUT_DIR = "/home/paramjeet/times-fm/experiments/chronos/MSL_chronos_31_ts"
os.makedirs(OUTPUT_DIR, exist_ok=True)

log_file = os.path.join(OUTPUT_DIR, "MSL_31_run.log")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger()

CONTEXT_LEN = 256
HORIZON_LEN = 10
BATCH_SIZE = 256

# --------------------------------------------------
# Load Chronos-2 model (single instance)
# --------------------------------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
logger.info(f"Using device: {device}")

from chronos import Chronos2Pipeline
model = Chronos2Pipeline.from_pretrained(
    "amazon/chronos-2",
    device_map=device,
    dtype=torch.bfloat16 if device == "cuda" else torch.float32,
)
logger.info("Chronos-2 model loaded.")

# --------------------------------------------------
# Get file list
# --------------------------------------------------
train_files = sorted([f for f in os.listdir(DATA_DIR) if "_train.csv" in f])
logger.info(f"Number of time series: {len(train_files)}")

results = []
all_aurocs = []
all_auprcs = []

start_all = time.time()

# --------------------------------------------------
# Loop over each time series
# --------------------------------------------------
for train_file in train_files:

    ts_name = train_file.replace("_train.csv", "")
    test_file = train_file.replace("_train.csv", "_test.csv")

    logger.info(f"\nProcessing: {ts_name}")

    train_df = pd.read_csv(os.path.join(DATA_DIR, train_file))
    test_df  = pd.read_csv(os.path.join(DATA_DIR, test_file))

    # ---------------- Preprocess ----------------
    imputer, scaler, X_train, y_train = fit_preprocessor(train_df)
    X_test, y_test = transform_preprocessor(test_df, imputer, scaler)

    # ---------------- Compute anomaly scores ----------------
    anomaly_scores = compute_chronos2_multivariate_anomaly_scores(
        model=model,
        data=X_test,
        context_len=CONTEXT_LEN,
        horizon=HORIZON_LEN,
        batch_size=BATCH_SIZE
    )

    # Remove initial context region
    y_valid = y_test[CONTEXT_LEN:]
    scores_valid = anomaly_scores[CONTEXT_LEN:]

    # ---------------- ROC ----------------
    from sklearn.metrics import roc_curve, roc_auc_score, precision_recall_curve, average_precision_score

    fpr, tpr, thresholds = roc_curve(y_valid, scores_valid)
    auroc = roc_auc_score(y_valid, scores_valid)

    precision, recall, _ = precision_recall_curve(y_valid, scores_valid)
    auprc = average_precision_score(y_valid, scores_valid)

    # ---------------- Best threshold (Youden’s J) ----------------
    youden_j = tpr - fpr
    best_idx = np.argmax(youden_j)

    best_threshold = thresholds[best_idx]
    best_tpr = tpr[best_idx]
    best_fpr = fpr[best_idx]

    logger.info(f"AUROC: {auroc:.4f}, AUPRC: {auprc:.4f}, Best Threshold: {best_threshold:.6f}")

    # ---------------- Save ROC curve ----------------
    roc_df = pd.DataFrame({"FPR": fpr, "TPR": tpr, "Threshold": thresholds})
    roc_df.to_csv(os.path.join(OUTPUT_DIR, f"{ts_name}_roc_curve.csv"), index=False)

    # ---------------- Save PR curve ----------------
    pr_df = pd.DataFrame({"Recall": recall, "Precision": precision})
    pr_df.to_csv(os.path.join(OUTPUT_DIR, f"{ts_name}_pr_curve.csv"), index=False)

    # ---------------- Plot ROC ----------------
    plt.figure()
    plt.plot(fpr, tpr, label=f"AUROC={auroc:.4f}")
    plt.plot([0,1],[0,1],'k--')
    plt.scatter(best_fpr,best_tpr,color="red",label="Best Threshold")
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.title(f"ROC - {ts_name}")
    plt.legend()
    plt.savefig(os.path.join(OUTPUT_DIR, f"{ts_name}_roc.png"))
    plt.close()

    # ---------------- Plot PR ----------------
    plt.figure()
    plt.plot(recall, precision, label=f"AUPRC={auprc:.4f}")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(f"PR - {ts_name}")
    plt.legend()
    plt.savefig(os.path.join(OUTPUT_DIR, f"{ts_name}_pr.png"))
    plt.close()

    # ---------------- Store results ----------------
    results.append([ts_name, auroc, auprc, best_threshold, best_tpr, best_fpr])
    all_aurocs.append(auroc)
    all_auprcs.append(auprc)

# --------------------------------------------------
# Save Summary
# --------------------------------------------------


summary_df = pd.DataFrame(
    results,
    columns=["time_series","AUROC","AUPRC","Best_Threshold","Best_TPR","Best_FPR"]
)

summary_csv = os.path.join(OUTPUT_DIR, "MSL_summary.csv")
summary_df.to_csv(summary_csv, index=False)
logger.info(f"Saved per-series summary to {summary_csv}")

# Compute mean metrics across all series
mean_auroc = np.nanmean([r[1] for r in results])  # AUROC column
mean_auprc = np.nanmean([r[2] for r in results])  # AUPRC column


# Save mean metrics separately
mean_metrics_df = pd.DataFrame({
    "Mean_AUROC": [mean_auroc],
    "Mean_AUPRC": [mean_auprc],
    # "Mean_Best_Threshold": [mean_best_threshold]
})

mean_csv = os.path.join(OUTPUT_DIR, "MSL_mean_metrics.csv")
mean_metrics_df.to_csv(mean_csv, index=False)
logger.info(f"Saved mean metrics to {mean_csv}")
