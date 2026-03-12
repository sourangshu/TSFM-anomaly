# ============================================================
# anomaly_chronos.py
# ============================================================

import numpy as np
from tqdm import tqdm

def create_windows(series, context_len, horizon):
    """
    series: (n_points, n_features)
    Returns:
        windows: (num_windows, context_len, n_features)
        future:  (num_windows, horizon, n_features)
    """
    n_points, n_features = series.shape
    total_len = context_len + horizon

    if n_points < total_len:
        return None, None

    # Sliding window over time axis
    X = np.lib.stride_tricks.sliding_window_view(series, window_shape=total_len, axis=0)

    # shape: (num_windows, n_features, total_len) → transpose to (num_windows, total_len, n_features)
    X = np.transpose(X, (0, 2, 1))

    windows = X[:, :context_len, :]
    future  = X[:, context_len:, :]

    return windows, future


def compute_chronos2_multivariate_anomaly_scores(model, data, context_len=128, horizon=1, batch_size=32):
    """
    Compute anomaly scores using Chronos-2 for multivariate data.
    Returns: anomaly_scores (n_points,)
    """
    n_points, n_features = data.shape

    # Create sliding windows
    windows, true_future = create_windows(data, context_len, horizon)
    if windows is None:
        return np.zeros(n_points)

    all_forecasts = []

    # Batch prediction
    for start in tqdm(range(0, len(windows), batch_size)):
        end = min(start + batch_size, len(windows))
        batch = windows[start:end]                  # (B, context_len, n_features)
        batch = np.transpose(batch, (0, 2, 1)).copy()  # (B, n_features, context_len)

        # Predict with Chronos
        _, mean = model.predict_quantiles(
            batch,
            prediction_length=horizon,
            quantile_levels=[0.5],
        )

        # Chronos returns a list of (features, horizon) → stack
        forecast = np.stack(mean, axis=0)  # (B, features, horizon)

        # TRANSPOSE to match true_future: (B, horizon, features)
        forecast = np.transpose(forecast, (0, 2, 1))

        all_forecasts.append(forecast)

    forecast_all = np.concatenate(all_forecasts, axis=0)  # (num_windows, horizon, features)

    # Multivariate MSE anomaly score
    error = np.mean((forecast_all - true_future) ** 2, axis=(1, 2))

    anomaly_scores = np.zeros(n_points)
    anomaly_scores[context_len:context_len + len(error)] = error

    return anomaly_scores
