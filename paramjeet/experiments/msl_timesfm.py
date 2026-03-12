# ============================================================
# experiments/MSL_timesfm.py
# ============================================================

import os
import sys
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

from sklearn.metrics import (
    roc_curve,
    roc_auc_score,
    precision_recall_curve,
    average_precision_score
)

# ============================================================
# Import project utilities
# ============================================================
sys.path.append("/home/paramjeet/times-fm/src")
from anomaly_timesfm import compute_timesfm_anomaly_scores

import timesfm

# ============================================================
# Config
# ============================================================
DATA_DIR = "/home/paramjeet/times-fm/datasets/MSL"
OUTPUT_DIR = "/home/paramjeet/times-fm/experiments/MSL_results"

CHECKPOINT_DIR = os.path.join(OUTPUT_DIR, "checkpoints")
PLOT_DIR = os.path.join(OUTPUT_DIR, "plots")

os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(PLOT_DIR, exist_ok=True)

TRAIN_FILE = "MSL_C-1_train.csv"
TEST_FILE  = "MSL_C-1_test.csv"

CONTEXT_LIST = [512]
HORIZON_LIST = [1]

# ============================================================
# Load data
# ============================================================
train_df = pd.read_csv(os.path.join(DATA_DIR, TRAIN_FILE))
test_df  = pd.read_csv(os.path.join(DATA_DIR, TEST_FILE))

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

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

# ============================================================
# Load TimeFM model once
# ============================================================
torch.set_float32_matmul_precision("high")

model = timesfm.TimesFM_2p5_200M_torch.from_pretrained(
    "google/timesfm-2.5-200m-pytorch"
)

# Must compile with max horizon multiple of 128
model.compile(
    timesfm.ForecastConfig(
        max_context=max(CONTEXT_LIST),
        max_horizon=128,
        normalize_inputs=True,
        use_continuous_quantile_head=True,
        force_flip_invariance=True,
        infer_is_positive=True,
        fix_quantile_crossing=True,
    )
)

# ============================================================
# Loop over configs
# ============================================================
results = []

for context_len in CONTEXT_LIST:
    for horizon in HORIZON_LIST:
        

        print(f"\nRunning: context={context_len}, horizon={horizon}")

        anomaly_scores, _ = compute_timesfm_anomaly_scores(
            model=model,
            X_train=X_train,
            X_test=X_test,
            y_test=y_test,
            context_len=context_len,
            horizon=horizon
        )

        y_valid = y_test[context_len:]
        scores_valid = anomaly_scores[context_len:]

        # ================= VALID DATA =================
        y_valid = y_test[context_len:]
        scores_valid = anomaly_scores[context_len:]

        # ================= ROC =================
        fpr, tpr, thresholds = roc_curve(y_valid, scores_valid)
        auroc = roc_auc_score(y_valid, scores_valid)

        # ================= PR =================
        precision, recall, pr_thresholds = precision_recall_curve(
            y_valid, scores_valid
        )
        auprc = average_precision_score(y_valid, scores_valid)

        # ================= BEST THRESHOLD (Youden) =================
        youden_j = tpr - fpr
        best_idx = np.argmax(youden_j)
        best_threshold = thresholds[best_idx]

        # Predictions at best threshold
        y_pred = (scores_valid >= best_threshold).astype(int)

        from sklearn.metrics import precision_score, recall_score, f1_score

        precision_best = precision_score(y_valid, y_pred)
        recall_best = recall_score(y_valid, y_pred)
        f1_best = f1_score(y_valid, y_pred)

        # ================= PRINT =================
        print(f"AUROC: {auroc:.4f}")
        print(f"AUPRC: {auprc:.4f}")
        print(f"Best Threshold: {best_threshold:.6f}")
        print(f"Best TPR: {tpr[best_idx]:.4f}")
        print(f"Best FPR: {fpr[best_idx]:.4f}")
        print(f"Precision@Best: {precision_best:.4f}")
        print(f"Recall@Best: {recall_best:.4f}")
        print(f"F1@Best: {f1_best:.4f}")

        config_name = f"MSL_context{context_len}_h{horizon}"

        # ================= SAVE SCORES =================
        np.save(
            os.path.join(CHECKPOINT_DIR, f"msl_{config_name}_scores.npy"),
            anomaly_scores
        )

        # ================= SAVE ROC VALUES =================
        roc_df = pd.DataFrame({
            "FPR": fpr,
            "TPR": tpr,
            "Threshold": thresholds
        })

        roc_df.to_csv(
            os.path.join(CHECKPOINT_DIR, f"msl_{config_name}_roc_values.csv"),
            index=False
        )

        # ================= SAVE PR VALUES =================
        pr_df = pd.DataFrame({
            "Recall": recall,
            "Precision": precision
        })

        pr_df.to_csv(
            os.path.join(CHECKPOINT_DIR, f"msl_{config_name}_pr_values.csv"),
            index=False
        )

        # ================= SAVE ROC PLOT =================
        plt.figure()
        plt.plot(fpr, tpr, label=f"AUROC={auroc:.4f}")
        plt.plot([0, 1], [0, 1], "k--")
        plt.scatter(fpr[best_idx], tpr[best_idx])
        plt.xlabel("FPR")
        plt.ylabel("TPR")
        plt.title(f"ROC - {config_name}")
        plt.legend()
        plt.savefig(os.path.join(PLOT_DIR, f"msl_{config_name}_roc.png"))
        plt.close()

        # ================= SAVE PR PLOT =================
        plt.figure()
        plt.plot(recall, precision, label=f"AUPRC={auprc:.4f}")
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.title(f"PR - {config_name}")
        plt.legend()
        plt.savefig(os.path.join(PLOT_DIR, f"msl_{config_name}_pr.png"))
        plt.close()

        # ================= STORE  RESULTS =================
        results.append([
            context_len,
            horizon,
            auroc,
            auprc,
            best_threshold,
            tpr[best_idx],
            fpr[best_idx],
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
        "Best_TPR",
        "Best_FPR",
        "Precision_at_Best",
        "Recall_at_Best",
        "F1_at_Best"
    ]
)

summary_df.to_csv(
    os.path.join(OUTPUT_DIR, "metrics_summary.csv"),
    index=False
)

print("\nFinished all configurations.")
print("Full metrics saved to metrics_summary.csv")


