import os
import time
import pandas as pd
import numpy as np
import torch
import timesfm
import sys
import matplotlib.pyplot as plt
import logging
from sklearn.metrics import roc_curve, roc_auc_score

# --------------------------------------------------
# Add src path
# --------------------------------------------------
sys.path.append("/home/paramjeet/times-fm/src")

from preprocess import fit_preprocessor, transform_preprocessor
from anomaly_timesfm import compute_timesfm_anomaly_scores

# --------------------------------------------------
# Config
# --------------------------------------------------
DATA_DIR = "/home/paramjeet/times-fm/datasets/Exathlon"
OUTPUT_DIR = "/home/paramjeet/times-fm/experiments/logs_cntxt32_hr1"

# --------------------------------------------------
# Setup Logging
# --------------------------------------------------
log_file = os.path.join(OUTPUT_DIR, "run.log")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()  # logger.infos to terminal also
    ]
)

logger = logging.getLogger()


os.makedirs(OUTPUT_DIR, exist_ok=True)

CONTEXT_LEN = 32
HORIZON_LEN = 10

# --------------------------------------------------
# Load TimeFM model 
# --------------------------------------------------
logger.info("Loading TimeFM model...")

torch.set_float32_matmul_precision("high")

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

logger.info("Model ready.\n")

# --------------------------------------------------
# Get file list
# --------------------------------------------------
train_files = sorted([f for f in os.listdir(DATA_DIR) if "_train.csv" in f])

logger.info("Number of time series:%s", len(train_files))

results = []
all_aurocs = []

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
    anomaly_scores, _ = compute_timesfm_anomaly_scores(
        model=model,
        X_train=X_train,
        X_test=X_test,
        y_test=y_test,
        context_len=CONTEXT_LEN,
        horizon=HORIZON_LEN
    )

    # Remove initial context region
    y_valid = y_test[CONTEXT_LEN:]
    scores_valid = anomaly_scores[CONTEXT_LEN:]

    # ---------------- ROC ----------------
    fpr, tpr, thresholds = roc_curve(y_valid, scores_valid)
    auroc = roc_auc_score(y_valid, scores_valid)

    # ---------------- Best threshold (Youden’s J) ----------------
    j_scores = tpr - fpr
    best_idx = np.argmax(j_scores)

    best_threshold = thresholds[best_idx]
    best_tpr = tpr[best_idx]
    best_fpr = fpr[best_idx]

    logger.info(f"AUROC: {auroc:.4f}")
    logger.info(f"Best Threshold: {best_threshold:.6f}")
    logger.info(f"Best TPR: {best_tpr:.4f}")
    logger.info(f"Best FPR: {best_fpr:.4f}")

    # ---------------- Save ROC curve ----------------
    roc_df = pd.DataFrame({
        "FPR": fpr,
        "TPR": tpr,
        "Threshold": thresholds
    })

    roc_df.to_csv(
        os.path.join(OUTPUT_DIR, f"{ts_name}_roc_curve.csv"),
        index=False
    )

    # ---------------- Plot ROC ----------------
    plt.figure()
    plt.plot(fpr, tpr)
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC Curve - {ts_name} (AUROC={auroc:.4f})")

    plt.savefig(os.path.join(OUTPUT_DIR, f"{ts_name}_roc.png"))
    plt.close()

    # ---------------- Store results ----------------
    results.append([
        ts_name,
        auroc,
        best_threshold,
        best_tpr,
        best_fpr
    ])

    all_aurocs.append(auroc)

# --------------------------------------------------
# Save Summary
# --------------------------------------------------
summary_df = pd.DataFrame(
    results,
    columns=[
        "time_series",
        "AUROC",
        "Best_Threshold",
        "Best_TPR",
        "Best_FPR"
    ]
)

summary_path = os.path.join(OUTPUT_DIR, "auroc_summary.csv")
summary_df.to_csv(summary_path, index=False)

# ---------------- Mean AUROC ----------------
mean_auroc = np.mean(all_aurocs)

logger.info("\n====================================")
logger.info("MEAN AUROC: %.4f", mean_auroc)
logger.info("====================================")

logger.info("Total time: %.2f sec", time.time() - start_all)


# Save mean AUROC separately
with open(os.path.join(OUTPUT_DIR, "mean_auroc.txt"), "w") as f:
    f.write(f"Mean AUROC: {mean_auroc:.6f}\n")

logger.info("Total time: %.2f sec", time.time() - start_all)
