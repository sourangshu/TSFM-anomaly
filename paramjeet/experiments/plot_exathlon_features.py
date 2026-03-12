import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

# ============================================================
# WRITE FILE NAMES HERE
# ============================================================
TRAIN_FILE = "Exathlon_3_5_1000000_89_train.csv"
TEST_FILE  = "Exathlon_3_5_1000000_89_test.csv"

# ============================================================
# PATH SETUP
# ============================================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR   = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))

DATA_DIR = os.path.join(ROOT_DIR, "datasets", "Exathlon")

dataset_name = TRAIN_FILE.replace("_train.csv", "")

BASE_SAVE_DIR = os.path.join(
    ROOT_DIR,
    "experiments",
    "dataset_plots",
    "exathlon",
    dataset_name
)

LINE_DIR      = os.path.join(BASE_SAVE_DIR, "line_plots")
SCATTER_DIR   = os.path.join(BASE_SAVE_DIR, "scatter_plots")
RAW_DIST_DIR  = os.path.join(BASE_SAVE_DIR, "raw_distributions")

os.makedirs(LINE_DIR, exist_ok=True)
os.makedirs(SCATTER_DIR, exist_ok=True)
os.makedirs(RAW_DIST_DIR, exist_ok=True)

print("\nSaving plots to:")
print(BASE_SAVE_DIR)

# ============================================================
# LOAD DATA
# ============================================================
train_df = pd.read_csv(os.path.join(DATA_DIR, TRAIN_FILE))
test_df  = pd.read_csv(os.path.join(DATA_DIR, TEST_FILE))

if "timestamp" in train_df.columns:
    train_df = train_df.drop(columns=["timestamp"])
if "timestamp" in test_df.columns:
    test_df = test_df.drop(columns=["timestamp"])

if "is_anomaly" not in test_df.columns:
    raise ValueError("Expected 'is_anomaly' column in test file")

y_test = test_df["is_anomaly"].values

X_train_raw = train_df.drop(columns=["is_anomaly"], errors="ignore").values
X_test_raw  = test_df.drop(columns=["is_anomaly"], errors="ignore").values

print("Train shape:", X_train_raw.shape)
print("Test shape :", X_test_raw.shape)
print("Anomaly ratio:", y_test.mean())

# ============================================================
# RAW FEATURE ANALYSIS
# ============================================================
print("\n================ RAW FEATURE ANALYSIS ================\n")

variance_threshold = 1e-8
valid_features = []

for f in range(X_train_raw.shape[1]):

    train_std = np.std(X_train_raw[:, f])
    test_std  = np.std(X_test_raw[:, f])
    drift     = abs(np.mean(X_train_raw[:, f]) - np.mean(X_test_raw[:, f]))

    print(
        f"Feature {f} | "
        f"Train std: {train_std:.6f} | "
        f"Test std: {test_std:.6f} | "
        f"Mean drift: {drift:.6f}"
    )

    # Save RAW histogram
    plt.figure(figsize=(6,4))
    plt.hist(X_train_raw[:, f], bins=30, alpha=0.5, label="Train")
    plt.hist(X_test_raw[:, f], bins=30, alpha=0.5, label="Test")
    plt.title(f"RAW Distribution – Feature {f}")
    plt.legend()
    plt.grid(True)

    plt.savefig(
        os.path.join(RAW_DIST_DIR, f"raw_feature_{f}.png"),
        dpi=200,
        bbox_inches="tight"
    )
    plt.close()

    # Keep only informative features
    if train_std > variance_threshold:
        valid_features.append(f)

print("\nRemoved near-zero variance features:",
      set(range(X_train_raw.shape[1])) - set(valid_features))

# Filter features
X_train = X_train_raw[:, valid_features]
X_test  = X_test_raw[:, valid_features]

print("Remaining features:", len(valid_features))

# ============================================================
# MIN-MAX SCALING (TRAIN ONLY)
# ============================================================
scaler = MinMaxScaler(feature_range=(0, 1))
X_train = scaler.fit_transform(X_train)
X_test  = scaler.transform(X_test)

print("\nMin-Max scaling applied using TRAIN statistics.")

# ============================================================
# PLOTTING FUNCTION
# ============================================================
def plot_feature(feature_idx):

    train_series = X_train[:, feature_idx]
    test_series  = X_test[:, feature_idx]

    t_train = np.arange(len(train_series))
    t_test  = np.arange(len(test_series))

    # Force same y-scale
    global_min = min(train_series.min(), test_series.min())
    global_max = max(train_series.max(), test_series.max())

    # --------------------------------------------------------
    # LINE PLOTS
    # --------------------------------------------------------
    plt.figure(figsize=(14, 4))
    plt.plot(t_train, train_series)
    plt.ylim(global_min, global_max)
    plt.title(f"Train (Normalized) – Feature {feature_idx}")
    plt.grid(True)

    plt.savefig(os.path.join(LINE_DIR,
                f"train_feature_{feature_idx}.png"),
                dpi=200, bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(14, 4))
    plt.plot(t_test, test_series)
    plt.ylim(global_min, global_max)
    plt.title(f"Test (Normalized) – Feature {feature_idx}")
    plt.grid(True)

    plt.savefig(os.path.join(LINE_DIR,
                f"test_feature_{feature_idx}.png"),
                dpi=200, bbox_inches="tight")
    plt.close()

    # --------------------------------------------------------
    # SCATTER PLOTS
    # --------------------------------------------------------
    plt.figure(figsize=(6,4))
    plt.scatter(t_train, train_series, s=5)
    plt.ylim(global_min, global_max)
    plt.title(f"Train Scatter – Feature {feature_idx}")
    plt.grid(True)

    plt.savefig(os.path.join(SCATTER_DIR,
                f"train_feature_{feature_idx}.png"),
                dpi=200, bbox_inches="tight")
    plt.close()

    normal_idx  = y_test == 0
    anomaly_idx = y_test == 1

    plt.figure(figsize=(6,4))
    plt.scatter(t_test[normal_idx],
                test_series[normal_idx],
                s=5, label="Normal")
    plt.scatter(t_test[anomaly_idx],
                test_series[anomaly_idx],
                s=10, color="red", label="Anomaly")
    plt.ylim(global_min, global_max)
    plt.title(f"Test Scatter – Feature {feature_idx}")
    plt.legend()
    plt.grid(True)

    plt.savefig(os.path.join(SCATTER_DIR,
                f"test_feature_{feature_idx}.png"),
                dpi=200, bbox_inches="tight")
    plt.close()

    print(f"Feature {feature_idx} done.")

# ============================================================
# GENERATE PLOTS
# ============================================================
print("\nGenerating normalized plots...\n")

for i in range(X_train.shape[1]):
    plot_feature(i)

print("\n✅ All analysis completed.")
print("Saved to:", BASE_SAVE_DIR)