import numpy as np
import pandas as pd
import pickle
from pathlib import Path
from sklearn.metrics import brier_score_loss
import config
import data_loader
import feature_engineering
import kalman_filter

def calculate_ece_mce(y_true, y_prob, n_bins=10):
    """
    Calculate Expected Calibration Error (ECE) and Maximum Calibration Error (MCE).
    """
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]
    
    ece = 0
    mce = 0
    
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        # Determine if points fall into this bin
        in_bin = (y_prob > bin_lower) & (y_prob <= bin_upper)
        prop_in_bin = np.mean(in_bin)
        
        if prop_in_bin > 0:
            accuracy_in_bin = np.mean(y_true[in_bin])
            avg_confidence_in_bin = np.mean(y_prob[in_bin])
            
            error = np.abs(avg_confidence_in_bin - accuracy_in_bin)
            ece += error * prop_in_bin
            mce = max(mce, error)
            
    return ece, mce

def run_calibration_supplement():
    print("--- [R1-Supp] Quantitative Calibration Analysis ---")
    
    # 1. Load Data
    meta, ball = data_loader.load_all_matches(config.DATA_DIR)
    # Filter for TEST seasons (2023-2025)
    test_meta = meta[meta['year'].isin(config.TEST_SEASONS)]
    test_ball = ball[ball['match_id'].isin(test_meta['match_id'])]
    
    # 2. Features
    venue_stats, _ = data_loader.compute_venue_stats(ball, meta)
    test_ball_eng = feature_engineering.engineer_features(test_ball, venue_stats)
    
    # 3. Load Models
    with open(config.MODELS_DIR / "kf_model.pkl", "rb") as f:
        kf_model = pickle.load(f)
    
    # 4. Prepare KF Data (need scalers - we'll re-extract from training for consistency)
    train_meta = meta[meta['year'].isin(config.TRAIN_SEASONS)]
    train_ball = ball[ball['match_id'].isin(train_meta['match_id'])]
    train_ball_eng = feature_engineering.engineer_features(train_ball, venue_stats)
    _, s_obs, s_ctrl = kalman_filter.prepare_kalman_data(train_ball_eng, fit_scalers=True)
    
    data_dict_te, _, _ = kalman_filter.prepare_kalman_data(
        test_ball_eng, scaler_obs=s_obs, scaler_ctrl=s_ctrl, fit_scalers=False
    )
    
    # 5. Predictions
    test_df_with_kf = kalman_filter.generate_kalman_features(kf_model, data_dict_te, test_ball_eng)
    # Attach labels
    import baseline_models
    test_df_with_kf = baseline_models.prepare_labels(test_df_with_kf, test_meta)
    
    y_true = test_df_with_kf[config.TARGET_COL].values
    y_prob_kf = kf_model.predict_win_probability(test_df_with_kf)
    
    # 6. Metrics
    ece, mce = calculate_ece_mce(y_true, y_prob_kf)
    
    results = {
        "Model": "Kalman Filter (Standard)",
        "Expected Calibration Error (ECE)": round(ece, 4),
        "Maximum Calibration Error (MCE)": round(mce, 4),
        "Brier Score (Overall)": round(brier_score_loss(y_true, y_prob_kf), 4)
    }
    
    res_df = pd.DataFrame([results])
    output_path = config.TABLES_DIR / "remark4_calibration.csv"
    res_df.to_csv(output_path, index=False)
    
    print(f"\nResults saved to {output_path}")
    print(res_df.to_string(index=False))

if __name__ == "__main__":
    run_calibration_supplement()
