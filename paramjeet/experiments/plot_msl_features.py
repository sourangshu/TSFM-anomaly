import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ============================================================
# Config
# ============================================================
DATA_DIR = "datasets/MSL"

SAVE_DIR = "msl_plots"
os.makedirs(SAVE_DIR, exist_ok=True)

# ============================================================
# Load datasets
# ============================================================

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))

DATA_DIR = os.path.join(ROOT_DIR, "datasets", "MSL")

TRAIN_FILE = "MSL_C-1_train.csv"
TEST_FILE  = "MSL_C-1_test.csv"


train_df = pd.read_csv(os.path.join(DATA_DIR, TRAIN_FILE))
test_df  = pd.read_csv(os.path.join(DATA_DIR, TEST_FILE))

# ------------------------------------------------------------
# Drop timestamp if present
# ------------------------------------------------------------
if "timestamp" in train_df.columns:
    train_df = train_df.drop(columns=["timestamp"])
if "timestamp" in test_df.columns:
    test_df = test_df.drop(columns=["timestamp"])

# ------------------------------------------------------------
# Extract labels from TEST
# ------------------------------------------------------------
if "is_anomaly" not in test_df.columns:
    raise ValueError("Expected column 'is_anomaly' in test CSV")


y_test = test_df["is_anomaly"].values

# ------------------------------------------------------------
# Extract features
# ------------------------------------------------------------
X_train = train_df.drop(columns=["is_anomaly"], errors="ignore").values
X_test  = test_df.drop(columns=["is_anomaly"], errors="ignore").values

print("Train shape:", X_train.shape)
print("Test shape :", X_test.shape)
print("Anomaly ratio (test):", y_test.mean())

# ============================================================
# Plot function
# ============================================================
def plot_and_save_train_test(
    X_train,
    X_test,
    y_test,
    feature_idx=0,
    save_dir="plots",
):
    os.makedirs(save_dir, exist_ok=True)

    train_series = X_train[:, feature_idx]
    test_series  = X_test[:, feature_idx]

    t_train = np.arange(len(train_series))
    t_test  = np.arange(len(test_series))

    # ---------------------------
    # Train plot
    # ---------------------------
    plt.figure(figsize=(14, 4))
    plt.plot(t_train, train_series, linewidth=1)
    plt.title(f"Train Data – Feature {feature_idx}")
    plt.xlabel("Time")
    plt.ylabel("Value")
    plt.grid(True)

    train_path = os.path.join(save_dir, f"train_feature_{feature_idx}.png")
    plt.savefig(train_path, dpi=200, bbox_inches="tight")
    plt.close()

    # ---------------------------
    # Test plot (normal + anomaly)
    # ---------------------------
    normal_idx  = y_test == 0
    anomaly_idx = y_test == 1

    plt.figure(figsize=(14, 4))
    plt.plot(
        t_test[normal_idx],
        test_series[normal_idx],
        linewidth=1,
        label="Normal"
    )
    plt.scatter(
        t_test[anomaly_idx],
        test_series[anomaly_idx],
        color="red",
        marker="x",
        s=30,
        label="Anomaly"
    )

    plt.title(f"Test Data – Feature {feature_idx}")
    plt.xlabel("Time")
    plt.ylabel("Value")
    plt.legend()
    plt.grid(True)

    test_path = os.path.join(save_dir, f"test_feature_{feature_idx}.png")
    plt.savefig(test_path, dpi=200, bbox_inches="tight")
    plt.close()

    print(f"Saved feature {feature_idx}")

# ============================================================
# Plot ALL features
# ============================================================
num_features = X_train.shape[1]

for f in range(num_features):
    plot_and_save_train_test(
        X_train,
        X_test,
        y_test,
        feature_idx=f,
        save_dir=SAVE_DIR
    )

print("\nAll feature plots saved to:", SAVE_DIR)
