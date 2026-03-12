import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler


def validate_input(df):
    if "is_anomaly" not in df.columns:
        print("[INFO] 'is_anomaly' column not found — assuming unsupervised dataset.")


def split_features_labels(df: pd.DataFrame):
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

    return X.values, y


# --------------------------------------------------
# TRAIN
# --------------------------------------------------
def fit_preprocessor(df: pd.DataFrame):
    # validate_input(df)

    X, y = split_features_labels(df)

    imputer = SimpleImputer(strategy="median")
    X = imputer.fit_transform(X)

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    return imputer, scaler, X, y


# --------------------------------------------------
# TEST
# --------------------------------------------------
def transform_preprocessor(df: pd.DataFrame, imputer, scaler):
    validate_input(df)

    X, y = split_features_labels(df)

    X = imputer.transform(X)
    X = scaler.transform(X)

    return X, y
