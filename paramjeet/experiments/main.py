import os
import glob
import time
import numpy as np
import pandas as pd
import torch
import timesfm
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.preprocessing import MinMaxScaler

# ==== CONFIG ====
DATA_PATH = "/home/paramjeet/times-fm/datasets/Exathlon/*test.csv"

RUN_SINGLE = True
SINGLE_FILE = "Exathlon_10_2_1000000_67_test.csv"

WINDOW = 100
SPLIT_RATIO = 0.2
CONTEXT_LEN = 512
BATCH_SIZE = 128
TOP_K = 3

# ==== MODEL ====
model = timesfm.TimesFM_2p5_200M_torch.from_pretrained(
    "google/timesfm-2.5-200m-pytorch"
)
model.compile(timesfm.ForecastConfig(
    max_context=CONTEXT_LEN,
    max_horizon=WINDOW
))

# ==== PREP ====
def prepare_df(df):
    df = df.copy()
    df["timestamp"] = pd.date_range(start="2000-01-01", periods=len(df), freq="1s")
    return df

def split_data(df):
    split = int(len(df) * SPLIT_RATIO)
    return df.iloc[:split].reset_index(drop=True), df.iloc[split:].reset_index(drop=True)

# ==== FAST PREDICTION ====
def predict_all_features(df_train, df_test, features):
    n_features = len(features)
    all_preds = []

    num_windows = len(df_test) // WINDOW
    remainder = len(df_test) % WINDOW

    for i in range(num_windows):
        start = i * WINDOW

        if i == 0:
            train_window = df_train[features].values
        else:
            past_test = df_test[features].iloc[:start].values
            train_window = np.vstack([df_train[features].values, past_test])

        contexts = []
        for f in range(n_features):
            context = train_window[-CONTEXT_LEN:, f]
            contexts.append(context.astype(np.float32))

        preds_batch = []
        with torch.no_grad():
            for j in range(0, len(contexts), BATCH_SIZE):
                batch = contexts[j:j+BATCH_SIZE]
                forecast, _ = model.forecast(horizon=WINDOW, inputs=batch)
                preds_batch.append(np.array(forecast))

        preds_batch = np.concatenate(preds_batch, axis=0)
        preds_batch = preds_batch.reshape(n_features, WINDOW).T
        all_preds.append(preds_batch)

    # remainder
    if remainder > 0:
        start = num_windows * WINDOW
        past_test = df_test[features].iloc[:start].values
        train_window = np.vstack([df_train[features].values, past_test])

        contexts = []
        for f in range(n_features):
            context = train_window[-CONTEXT_LEN:, f]
            contexts.append(context.astype(np.float32))

        with torch.no_grad():
            forecast, _ = model.forecast(horizon=remainder, inputs=contexts)

        preds = np.array(forecast).reshape(n_features, remainder).T
        all_preds.append(preds)

    return np.vstack(all_preds)

# ==== FILE LIST ====
if RUN_SINGLE:
    base = DATA_PATH.replace("*test.csv", "")
    files = [os.path.join(base, SINGLE_FILE)]
else:
    files = glob.glob(DATA_PATH)

# ==== TIMER START ====
total_start = time.time()

# ==== MAIN ====
for f in files:
    file_start = time.time()

    print(f"\nProcessing: {os.path.basename(f)}")

    df = pd.read_csv(f)
    df = prepare_df(df)

    features = [c for c in df.columns if c not in ["timestamp", "is_anomaly"]]

    df_train, df_test = split_data(df)

    # ==== PREDICTION ====
    pred_values = predict_all_features(df_train, df_test, features)

    # ==== ERRORS ====
    mae = np.abs(pred_values - df_test[features].values)
    mse = (pred_values - df_test[features].values) ** 2

    # ==== SCALE ====
    scaler = MinMaxScaler()
    mse_scaled = scaler.fit_transform(mse)
    mae_scaled = scaler.fit_transform(mae)

    # ==== AGGREGATIONS ====
    scores_dict = {
        "mean": np.mean(mse_scaled, axis=1),
        "max": np.max(mse_scaled, axis=1),
        "l2": np.sqrt((mse_scaled ** 2).sum(axis=1)),
        "topk": np.mean(np.sort(mse_scaled, axis=1)[:, -TOP_K:], axis=1),
        "mae": np.mean(mae_scaled, axis=1),
    }

    y = df_test["is_anomaly"].values

    print("---- Metrics ----")
    for name, scores in scores_dict.items():
        if len(np.unique(y)) > 1:
            auroc = roc_auc_score(y, scores)
            auprc = average_precision_score(y, scores)
            print(f"{name:5} | AUROC: {auroc:.4f} | AUPRC: {auprc:.4f}")

    # ==== TIME ====
    file_time = time.time() - file_start
    print(f"Time for file: {file_time:.2f} sec")

# ==== TOTAL TIME ====
total_time = time.time() - total_start
print(f"\nTotal Execution Time: {total_time:.2f} sec ({total_time/60:.2f} min)")