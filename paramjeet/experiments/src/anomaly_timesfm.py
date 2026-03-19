"""
Improved TimeFM anomaly detection with multiple aggregation methods.
Location: /home/paramjeet/times-fm/src/anomaly_timesfm.py
"""

import numpy as np
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

def compute_timesfm_anomaly_scores(
    model,
    X_train,
    X_test,
    y_test,
    context_len,
    horizon,
    top_k=3,
    return_viz=False,
    viz_subsample=1  # Har nth window store karo (memory bachane ke liye)
):
    """
    Compute anomaly scores using TimeFM for multiple aggregation methods:
    max, top-k mean, L2 norm, MSE.
    
    Args:
        model: Loaded TimeFM model
        X_train: (T_train, F) training data
        X_test: (T_test, F) test data
        y_test: (T_test,) true labels
        context_len: context window size
        horizon: forecast horizon
        top_k: number of top features to average
        return_viz: whether to return visualization data
        viz_subsample: store every nth window for visualization
    
    Returns:
        Dictionary containing all scores and metrics
    """
    T_test, n_features = X_test.shape
    T_train, _ = X_train.shape

    print(f"\n📊 TimeFM Anomaly Detection:")
    print(f"   Features: {n_features}")
    print(f"   Context: {context_len}, Horizon: {horizon}")
    print(f"   Top-K: {top_k}")
    print(f"   Train samples: {T_train}, Test samples: {T_test}")

    # ============================================================
    # STEP 1: TRAINING CALIBRATION
    # ============================================================
    print("\n Calibrating on training data...")
    
    feature_means = np.zeros(n_features)
    feature_stds = np.zeros(n_features)
    train_errors = [[] for _ in range(n_features)]

    for t in tqdm(range(context_len, T_train - horizon), desc="Training windows"):
        # Get contexts for all features at once
        contexts = [X_train[t-context_len:t, f].tolist() for f in range(n_features)]
        forecast_result = model.forecast(horizon=horizon, inputs=contexts)

        # Extract mean and uncertainty
        if isinstance(forecast_result, tuple) and len(forecast_result) == 2:
            mean, quantiles = forecast_result
            # quantiles shape: (n_features, horizon, 3) where 3 = [q_low, q_median, q_high]
            q_low = quantiles[:, :, 0]
            q_high = quantiles[:, :, 2]
            uncertainty = (q_high - q_low) / (2 * 1.28)  # Approx std dev
        else:
            mean = forecast_result
            uncertainty = np.ones_like(mean)

        # Avoid division by zero
        uncertainty = np.clip(uncertainty, 1e-3, None)

        # Calculate error for each feature
        for f in range(n_features):
            actual = X_train[t:t+horizon, f]
            error = np.mean(np.abs(actual - mean[f]) / uncertainty[f])
            train_errors[f].append(error)

    # Calculate mean and std for each feature
    for f in range(n_features):
        if len(train_errors[f]) > 0:
            feature_means[f] = np.mean(train_errors[f])
            feature_stds[f] = np.std(train_errors[f]) + 1e-8
        else:
            feature_means[f] = 0
            feature_stds[f] = 1

    print(f" Calibration complete")

    # ============================================================
    # STEP 2: TEST SCORING
    # ============================================================
    print("\n🔍 Computing anomaly scores on test data...")
    
    feature_scores_all = np.zeros((T_test, n_features))
    viz_data = []

    for t in tqdm(range(context_len, T_test - horizon), desc="Test windows"):
        # Get contexts for all features
        contexts = [X_test[t-context_len:t, f].tolist() for f in range(n_features)]
        forecast_result = model.forecast(horizon=horizon, inputs=contexts)

        # Extract mean and uncertainty
        if isinstance(forecast_result, tuple) and len(forecast_result) == 2:
            mean, quantiles = forecast_result
            q_low = quantiles[:, :, 0]
            q_high = quantiles[:, :, 2]
            uncertainty = (q_high - q_low) / (2 * 1.28)
        else:
            mean = forecast_result
            uncertainty = np.ones_like(mean)

        uncertainty = np.clip(uncertainty, 1e-3, None)

        # Calculate calibrated scores for each feature
        feature_scores = []
        for f in range(n_features):
            actual = X_test[t:t+horizon, f]
            raw_error = np.mean(np.abs(actual - mean[f]) / uncertainty[f])
            calibrated = (raw_error - feature_means[f]) / feature_stds[f]
            feature_scores.append(calibrated)

        feature_scores_all[t] = feature_scores

        # Store visualization data (subsampled to save memory)
        if return_viz and (t - context_len) % viz_subsample == 0:
            viz_data.append({
                'timestamp': t,
                'context_values': [X_test[t-context_len:t, f].copy() for f in range(n_features)],
                'forecast_values': [mean[f].copy() for f in range(n_features)],
                'actual_values': [X_test[t:t+horizon, f].copy() for f in range(n_features)],
                'feature_scores': feature_scores.copy(),
                'anomaly_score_max': float(np.max(feature_scores)),
                'anomaly_score_topk': float(np.mean(np.sort(feature_scores)[-top_k:])),
                'anomaly_score_l2': float(np.sqrt(np.sum(np.array(feature_scores)**2))),
                'auroc_mae': roc_auc_score(y_valid, mae_scores[valid_start:valid_end]),
                # 'anomaly_score_mse': float(np.mean(np.array(feature_scores)**2)),
                'is_anomaly': int(y_test[t])
            })

    # ============================================================
    # STEP 3: AGGREGATION
    # ============================================================
    print("\n🔄 Computing aggregation methods...")
    
    # Different aggregation methods
    max_scores = np.max(feature_scores_all, axis=1)
    topk_scores = np.mean(np.sort(feature_scores_all, axis=1)[:, -top_k:], axis=1)
    l2_scores = np.sqrt(np.sum(feature_scores_all**2, axis=1))
    # mse_scores = np.mean(feature_scores_all**2, axis=1)
    mae_scores = np.mean(np.abs(feature_scores_all), axis=1)

    # ============================================================
    # STEP 4: AUROC CALCULATION
    # ============================================================
    valid_start = context_len
    valid_end = T_test - horizon
    y_valid = y_test[valid_start:valid_end]

    auroc_max = roc_auc_score(y_valid, max_scores[valid_start:valid_end])
    auroc_topk = roc_auc_score(y_valid, topk_scores[valid_start:valid_end])
    auroc_l2 = roc_auc_score(y_valid, l2_scores[valid_start:valid_end])
    # auroc_mse = roc_auc_score(y_valid, mse_scores[valid_start:valid_end])
    auroc_mae = roc_auc_score(y_valid, mae_scores[valid_start:valid_end])

    # ============================================================
    # STEP 5: RETURN RESULTS
    # ============================================================
    print(f"\nAUROC Results:")
    print(f"   Max aggregation : {auroc_max:.4f}")
    print(f"   Top-{top_k} mean   : {auroc_topk:.4f}")
    print(f"   L2 norm        : {auroc_l2:.4f}")
    # print(f"   MSE            : {auroc_mse:.4f}")
    print(f"   MAE            : {auroc_mae:.4f}")

    return {
        "feature_scores": feature_scores_all,  # Raw per-feature scores
        "max": max_scores,                      # Max aggregation
        "topk": topk_scores,                    # Top-K mean aggregation
        "l2": l2_scores,                        # L2 norm aggregation
        # "mse": mse_scores,                      # MSE aggregation
        "mae": mae_scores,                      # MAE aggregation
        "viz_data": viz_data if return_viz else None,
        "auroc": {
            "max": auroc_max,
            "topk": auroc_topk,
            "l2": auroc_l2,
            # "mse": auroc_mse
            "mae": auroc_mae
        },
        "metadata": {
            "context_len": context_len,
            "horizon": horizon,
            "top_k": top_k,
            "n_features": n_features
        }
    }
