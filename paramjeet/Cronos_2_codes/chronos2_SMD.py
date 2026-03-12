import os
import glob
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from collections import defaultdict

from chronos import BaseChronosPipeline, Chronos2Pipeline
from sklearn.metrics import roc_auc_score, average_precision_score

import warnings

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)


# -------------------- ARGUMENT PARSING --------------------
def parse_args():
    parser = argparse.ArgumentParser(description="Chronos SMD Anomaly Detection")
    parser.add_argument(
        "--split_ratio",
        type=float,
        default=0.2,
        help="Train/Test split ratio (e.g., 0.2 means 20%% train, 80%% test)"
    )
    parser.add_argument(
        "--window_length",
        type = int,
        default = 100,
        help = "Chronos will predict timestapms"
    )
    parser.add_argument(
        "--gpu",
        type=str,
        default="0",
        help="CUDA_VISIBLE_DEVICES"
    )
    return parser.parse_args()


# -------------------- GPU SETUP --------------------
args = parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu


# -------------------- LOAD CHRONOS --------------------
pipeline: Chronos2Pipeline = BaseChronosPipeline.from_pretrained(
    "amazon/chronos-2",
    device_map="cuda"
)


# -------------------- DATA PREPARATION --------------------
def prepare_df_test(df):
    df = df.copy()
    df["timestamp"] = pd.date_range(
        start="2000-02-01",
        periods=len(df),
        freq="1s"
    )
    df = df.sort_values("timestamp")

    ts = df["timestamp"]
    assert ts.is_monotonic_increasing
    assert ts.diff().dropna().nunique() == 1

    return df


def split_dataset(df, split_ratio):
    split_idx = int(len(df) * split_ratio)
    df_train = df.iloc[:split_idx].reset_index(drop=True)
    df_test = df.iloc[split_idx:].reset_index(drop=True)
    return df_train, df_test


# -------------------- PREDICTION --------------------
def generate_prediction(df_train, df_test, feature):
    window_length = args.window_length
    all_predictions = []
    target = feature
    id_column = "id"
    timestamp_column = "timestamp"
    df_test = df_test.copy()
    df_train = df_train.copy()

    num_windows = len(df_test) // window_length
    remainder   = len(df_test) % window_length

    for i in range(num_windows):
        start = i * window_length
        end   = start + window_length

        if i == 0:
            df_train_window = df_train.copy()
        else:
           
            df_past_test = df_test.iloc[:start].copy()
            df_train_window = pd.concat([df_train, df_past_test], ignore_index=True)

        df_future_window = df_test.iloc[start:end].copy()
        df_future_window = df_future_window.drop(columns=[target, "is_anomaly"])

        pred_df = pipeline.predict_df(
            df_train_window,
            future_df=df_future_window,
            prediction_length=window_length,
            quantile_levels=[0.1, 0.5, 0.9],
            id_column=id_column,
            timestamp_column=timestamp_column,
            target=target,
            validate_inputs=False
        )

        pred_df["window_id"] = i
        all_predictions.append(pred_df)

    if remainder > 0:
        start = num_windows * window_length

        df_past_test    = df_test.iloc[:start].copy()
        df_train_window = pd.concat([df_train, df_past_test], ignore_index=True)

        df_future_window = df_test.iloc[start:].copy()
        df_future_window = df_future_window.drop(columns=[target, "is_anomaly"])

        pred_df = pipeline.predict_df(
            df_train_window,
            future_df=df_future_window,
            prediction_length=remainder,
            quantile_levels=[0.1, 0.5, 0.9],
            id_column=id_column,
            timestamp_column=timestamp_column,
            target=target,
            validate_inputs=False
        )

        pred_df["window_id"] = num_windows
        all_predictions.append(pred_df)

    final_predictions = pd.concat(all_predictions, ignore_index=True)
    return final_predictions




# -------------------- PATHS --------------------
data_path = "/home/rajib/mTSBench/Datasets/mTSBench/SMD/*test.csv"
output_dir = "/home/rajib/mTSBench/results/chronos/SMD"
os.makedirs(output_dir, exist_ok=True)


results_file = os.path.join(output_dir, f"chronos_results_{int((args.split_ratio * 100))}%.csv")

if not os.path.exists(results_file):
    pd.DataFrame(columns=["file_name", "AUROC", "AUPRC"]).to_csv(
        results_file, index=False
    )


# -------------------- MAIN LOOP --------------------
file_list = glob.glob(data_path)

auroc_list = []
auprc_list = []
dic_for_each_file = defaultdict(list)


for f in tqdm(file_list, desc="Processing SMD files", unit="file"):
    file_name = os.path.basename(f).replace(".csv", "")
    print(f"\nProcessing: {file_name}")

    df_original = pd.read_csv(f)
    df_original = prepare_df_test(df_original)

    feature_list = [
        c for c in df_original.columns
        if c not in ["timestamp", "is_anomaly"]
    ]

    df_train, df_test = split_dataset(df_original, args.split_ratio)
    df_train['id'] = 'SMD'
    df_test['id'] = 'SMD'
    pred_dic = {"timestamp": df_test["timestamp"]}

    # -------- Predictions per feature --------
    for feature in feature_list:
        pred_df = generate_prediction(df_train, df_test, feature)
        pred_dic[feature] = pred_df["predictions"]
        print(f"  ✓ {feature}")

    prediction_df = pd.DataFrame(pred_dic)
    print(f'length of prediction df is {len(prediction_df)}')
    print(f'length of test df is {len(df_test)}')
    
    # -------- Anomaly score --------
    df_MSE = pd.DataFrame()
    for feature in feature_list:
        df_MSE[feature] = (prediction_df[feature] - df_test[feature]) ** 2

    df_MSE = (df_MSE - df_MSE.min()) / (df_MSE.max() - df_MSE.min())
    df_MSE = df_MSE.fillna(0)

    score_L2 = np.sqrt((df_MSE ** 2).sum(axis=1))

    y_true = df_test["is_anomaly"]
    y_score = score_L2

    auroc = roc_auc_score(y_true, y_score)
    auprc = average_precision_score(y_true, y_score)
    print(f"AUROC: {auroc:.4f} | AUPRC: {auprc:.4f}")

    dic_for_each_file['file_name'].append(file_name)
    dic_for_each_file['AUROC'].append(auroc)
    dic_for_each_file['AUPRC'].append(auprc)
    
    auroc_list.append(auroc)
    auprc_list.append(auprc)

    row = pd.DataFrame([{
    "file_name": file_name,
    "AUROC": auroc,
    "AUPRC": auprc
    }])

    row.to_csv(
        results_file,
        mode="a",
        header=not os.path.exists(results_file),
        index=False
    )

    # -------- Save results --------
df_results = pd.DataFrame(dic_for_each_file)
df_results.to_csv('alternate_results_file.csv', index=False)


print("\n Finished processing all SMD files")
print(f" Results saved to: {results_file}")

# -------------------- SAVE MEAN SCORES --------------------
mean_auroc = np.mean(auroc_list)
mean_auprc = np.mean(auprc_list)

score_file = os.path.join(output_dir, f"SMD_chronos_score_{int((args.split_ratio * 100))}.txt")

with open(score_file, "w") as f:
    f.write(f"Split ratio       : {args.split_ratio}\n")
    f.write(f"Mean AUROC        : {mean_auroc:.6f}\n")
    f.write(f"Mean AUPRC        : {mean_auprc:.6f}\n")

print("\n Mean scores saved")
print(f"Mean AUROC: {mean_auroc:.4f}")
print(f"Mean AUPRC: {mean_auprc:.4f}")
print(f" Saved at: {score_file}")