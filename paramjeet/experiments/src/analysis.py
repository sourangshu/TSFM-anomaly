import pandas as pd
import glob
from typing import List, Set, Tuple

# =========================
# Config
# =========================
DATA_DIR = "/home/paramjeet/times-fm/datasets/Exathlon"

# =========================
# Step 1: Load CSV files
# =========================
def load_csv_files(path_pattern: str) -> List[pd.DataFrame]:
    """Load all CSV files matching the given pattern."""
    files = glob.glob(path_pattern)
    print(f"Found {len(files)} files.")
    
    dfs = []
    for f in files:
        df = pd.read_csv(f)
        print(f"Loaded: {f}, Shape: {df.shape}")
        dfs.append(df)
    return dfs

# =========================
# Step 2: Extract features
# =========================
def extract_features(df: pd.DataFrame) -> List[str]:
    """Return all feature columns (exclude timestamp and is_anomaly)."""
    return [c for c in df.columns if c not in ["timestamp", "is_anomaly"]]

# =========================
# Step 3: Analyze datasets
# =========================
def analyze_datasets(dfs: List[pd.DataFrame]) -> Tuple[int, int, float, int, int, Set[str], Set[str]]:
    """Compute lengths, union, and intersection of features."""
    lengths = []
    all_features = []
    common_features = None
    
    for df in dfs:
        features = set(extract_features(df))
        all_features.append(features)
        
        if common_features is None:
            common_features = features
        else:
            common_features = common_features.intersection(features)
        
        lengths.append(len(df))
    
    union_features = set().union(*all_features)
    
    print("\n=== Dataset Analysis ===")
    print("Number of datasets:", len(dfs))
    print("Min length:", min(lengths))
    print("Max length:", max(lengths))
    print("Mean length:", sum(lengths)/len(lengths))
    print(f"Max features in a dataset: {max(len(f) for f in all_features)}")
    print(f"Min features in a dataset: {min(len(f) for f in all_features)}")
    print(f"Union of features: {len(union_features)}")
    print(f"Intersection (common) features: {len(common_features)}")
    print("Common features:", sorted(common_features))
    
    return min(lengths), max(lengths), sum(lengths)/len(lengths), \
           max(len(f) for f in all_features), min(len(f) for f in all_features), \
           union_features, common_features

# =========================
# Step 4: Combine datasets
# =========================
def combine_datasets(dfs: List[pd.DataFrame]) -> pd.DataFrame:
    """Simply concatenate all datasets."""
    combined = pd.concat(dfs, ignore_index=True)
    print("Combined shape:", combined.shape)
    return combined

# =========================
# Main workflow
# =========================
train_pattern = f"{DATA_DIR}/*_train.csv"
train_dfs = load_csv_files(train_pattern)
analyze_datasets(train_dfs)
# train_combined = combine_datasets(train_dfs)

print("\nFirst 5 rows, first 10 columns:\n", train_combined.iloc[:5, :10])
