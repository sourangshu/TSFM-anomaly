# ============================================================
# experiments/Exathlon_chronos.py
# ============================================================

import os
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm 
from anomaly_chronos import create_windows, compute_chronos2_multivariate_anomaly_scores

from sklearn.metrics import (
    roc_curve,
    roc_auc_score,
    precision_recall_curve,
    average_precision_score,
    precision_score,
    recall_score,
    f1_score
)

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

from chronos import Chronos2Pipeline


# ============================================================
# Config
# ============================================================

DATA_DIR = "/home/paramjeet/times-fm/datasets/Exathlon"
OUTPUT_DIR = "/home/paramjeet/times-fm/experiments/exathlon_chronos_results"

CHECKPOINT_DIR = os.path.join(OUTPUT_DIR, "checkpoints")
PLOT_DIR = os.path.join(OUTPUT_DIR, "plots")

os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(PLOT_DIR, exist_ok=True)

TRAIN_FILE = "Exathlon_1_2_100000_68_train.csv"
TEST_FILE  = "Exathlon_1_2_100000_68_test.csv"

CONTEXT_LIST = [64]
HORIZON_LIST = [10]

BATCH_SIZE = 256  


# ============================================================
# Load data
# ============================================================

train_df = pd.read_csv(os.path.join(DATA_DIR, TRAIN_FILE))
test_df  = pd.read_csv(os.path.join(DATA_DIR, TEST_FILE))


def preprocess_df(df, imputer=None, scaler=None, fit=False):

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

        scaler = StandardScaler()
        X = scaler.fit_transform(X)
        return imputer, scaler, X, y
    else:
        X = imputer.transform(X)
        X = scaler.transform(X)
        return X, y


imputer, scaler, X_train, y_train = preprocess_df(train_df, fit=True)

X_test, y_test = preprocess_df(test_df, imputer=imputer, scaler=scaler)
# -------------------------------------------------------
#  (small subset test)
# -------------------------------------------------------
# DEBUG = True

# if DEBUG:
#     X_test = X_test[:600]     # Use only first 600 points for quick testing
#     y_test = y_test[:600]
#     print("Running in DEBUG mode with small dataset:", X_test.shape)


# ============================================================
# Load Chronos-2
# ============================================================

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

model = Chronos2Pipeline.from_pretrained(
    "amazon/chronos-2",
    device_map=device,
    dtype=torch.bfloat16 if device == "cuda" else torch.float32,
)

# ============================================================
# Main loop
# ============================================================

results = []

for context_len in CONTEXT_LIST:
    for horizon in HORIZON_LIST:

        print(f"\nRunning: context={context_len}, horizon={horizon}")

        anomaly_scores = compute_chronos2_multivariate_anomaly_scores(
            model=model,
            data=X_test,
            context_len=context_len,
            horizon=horizon,
            batch_size=BATCH_SIZE,
        )

        y_valid = y_test[context_len:]
        scores_valid = anomaly_scores[context_len:]

        # ROC
        fpr, tpr, thresholds = roc_curve(y_valid, scores_valid)
        auroc = roc_auc_score(y_valid, scores_valid)

        # PR
        precision, recall, _ = precision_recall_curve(y_valid, scores_valid)
        auprc = average_precision_score(y_valid, scores_valid)

        # Best threshold (Youden J)
        youden_j = tpr - fpr
        best_idx = np.argmax(youden_j)
        best_threshold = thresholds[best_idx]
        best_tpr = tpr[best_idx]
        best_fpr = fpr[best_idx]

        y_pred = (scores_valid >= best_threshold).astype(int)

        precision_best = precision_score(y_valid, y_pred, zero_division=0)
        recall_best = recall_score(y_valid, y_pred, zero_division=0)
        f1_best = f1_score(y_valid, y_pred, zero_division=0)

        print(f"AUROC: {auroc:.4f}")
        print(f"AUPRC: {auprc:.4f}")
        print(f"Best Threshold: {best_threshold:.6f}")
        print(f"TPR@Best: {best_tpr:.4f}, FPR@Best: {best_fpr:.4f}")
        print(f"Precision@Best: {precision_best:.4f}")
        print(f"Recall@Best: {recall_best:.4f}")
        print(f"F1@Best: {f1_best:.4f}")

        config_name = f"Exathlon_context{context_len}_h{horizon}"

        # Save scores
        np.save(os.path.join(CHECKPOINT_DIR, f"{config_name}_scores.npy"), anomaly_scores)

        # Save ROC
        pd.DataFrame({"FPR": fpr, "TPR": tpr, "Threshold": thresholds}).to_csv(
            os.path.join(CHECKPOINT_DIR, f"{config_name}_roc.csv"), index=False
        )

        # Save PR
        pd.DataFrame({"Recall": recall, "Precision": precision}).to_csv(
            os.path.join(CHECKPOINT_DIR, f"{config_name}_pr.csv"), index=False
        )

        # Save plots
        plt.figure()
        plt.plot(fpr, tpr, label=f"AUROC={auroc:.4f}")
        plt.plot([0, 1], [0, 1], "k--")
        plt.scatter(best_fpr, best_tpr, color="red", label="Best Threshold")
        plt.xlabel("FPR")
        plt.ylabel("TPR")
        plt.legend()
        plt.title(f"ROC - {config_name}")
        plt.savefig(os.path.join(PLOT_DIR, f"{config_name}_roc.png"))
        plt.close()

        plt.figure()
        plt.plot(recall, precision, label=f"AUPRC={auprc:.4f}")
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.legend()
        plt.title(f"PR - {config_name}")
        plt.savefig(os.path.join(PLOT_DIR, f"{config_name}_pr.png"))
        plt.close()

        results.append([
            context_len,
            horizon,
            auroc,
            auprc,
            best_threshold,
            best_tpr,
            best_fpr,
            precision_best,
            recall_best,
            f1_best
        ])

summary_df = pd.DataFrame(
    results,
    columns=[
        "Context",
        "Horizon",
        "AUROC",
        "AUPRC",
        "Best_Threshold",
        "TPR_at_Best",
        "FPR_at_Best",
        "Precision_at_Best",
        "Recall_at_Best",
        "F1_at_Best"
    ]
)

summary_df.to_csv(os.path.join(OUTPUT_DIR, "metrics_summary.csv"), index=False)

