import numpy as np
from sklearn.metrics import roc_auc_score

def compute_timesfm_anomaly_scores(
    model,
    X_train,
    X_test,
    y_test,
    context_len,
    horizon,
    return_viz=False  
):
    T_test, n_features = X_test.shape
    T_train = X_train.shape[0]
    anomaly_scores = np.zeros(T_test)
    
    print(f"\nTimeFM: {n_features} features, ctx={context_len}, hor={horizon}")
    
    # STEP 1: Training statistics (calibration)
    feature_means = []
    feature_stds = []
    
    for f in range(n_features):
        train_series = X_train[:, f]
        train_errors = []
        
        for t in range(context_len, T_train - horizon):
            context = train_series[t - context_len : t]
            
            forecast_result = model.forecast(
                horizon=horizon,
                inputs=[context.tolist()]
            )
            
            if isinstance(forecast_result, tuple) and len(forecast_result) == 2:
                mean, quantiles = forecast_result
                q_low = quantiles[0, :, 0]
                q_high = quantiles[0, :, 2]
                uncertainty = (q_high - q_low) / (2 * 1.28)
            else:
                mean = forecast_result
                uncertainty = np.ones(horizon)
            
            actual = train_series[t : t + horizon]
            error = np.mean(np.abs(actual - mean[0]) / (uncertainty + 1e-8))
            train_errors.append(error)
        
        if train_errors:
            feature_means.append(np.mean(train_errors))
            feature_stds.append(np.std(train_errors) + 1e-8)
        else:
            feature_means.append(0)
            feature_stds.append(1)
    
    # STEP 2: Test data
    viz_data = []  
    for f in range(n_features):
        test_series = X_test[:, f]
        f_mean = feature_means[f]
        f_std = feature_stds[f]
        
        for t in range(context_len, T_test - horizon):
            context = test_series[t - context_len : t]
            
            forecast_result = model.forecast(
                horizon=horizon,
                inputs=[context.tolist()]
            )
            
            if isinstance(forecast_result, tuple) and len(forecast_result) == 2:
                mean, quantiles = forecast_result
                q_low = quantiles[0, :, 0]
                q_high = quantiles[0, :, 2]
                uncertainty = (q_high - q_low) / (2 * 1.28)
            else:
                mean = forecast_result
                uncertainty = np.ones(horizon)
            
            actual = test_series[t : t + horizon]
            raw_error = np.mean(np.abs(actual - mean[0]) / (uncertainty + 1e-8))
            calibrated = (raw_error - f_mean) / f_std
            anomaly_scores[t] += calibrated
        
            #  Store visualization data (sirf first feature ke liye)
            if return_viz and f == 0:
                viz_data.append({
                    'timestamp': t,
                    'context_values': context.copy(),
                    'forecast_values': mean[0].copy(),
                    'actual_values': actual.copy(),
                    'anomaly_score': anomaly_scores[t],  # Note: ye cumulative score hai
                    'is_anomaly': y_test[t]
                })
    
    # STEP 3: AUROC on valid indices
    valid_start = context_len
    valid_end = T_test - horizon
    valid_scores = anomaly_scores[valid_start:valid_end]
    valid_labels = y_test[valid_start:valid_end]
    
    auroc = roc_auc_score(valid_labels, valid_scores)
    

    return anomaly_scores, auroc, viz_data
