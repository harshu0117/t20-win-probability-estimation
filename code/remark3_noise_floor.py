import numpy as np
import pandas as pd
import pickle
from pathlib import Path
from sklearn.metrics import brier_score_loss
import config
import data_loader
import feature_engineering
import kalman_filter
import baseline_models

def run_sensitivity_analysis():
    print("--- [S2-Supp] Noise Covariance (R) Sensitivity Analysis ---")
    
    # 1. Load Data
    meta, ball = data_loader.load_all_matches(config.DATA_DIR)
    # Use Validation set for sensitivity to keep Test set "clean"
    val_meta = meta[meta['year'].isin(config.VALIDATION_SEASONS)]
    val_ball = ball[ball['match_id'].isin(val_meta['match_id'])]
    
    # Training set for scalers
    train_meta = meta[meta['year'].isin(config.TRAIN_SEASONS)]
    train_ball = ball[ball['match_id'].isin(train_meta['match_id'])]
    
    venue_stats, _ = data_loader.compute_venue_stats(ball, meta)
    val_ball_eng = feature_engineering.engineer_features(val_ball, venue_stats)
    train_ball_eng = feature_engineering.engineer_features(train_ball, venue_stats)
    
    _, s_obs, s_ctrl = kalman_filter.prepare_kalman_data(train_ball_eng, fit_scalers=True)
    data_dict_val, _, _ = kalman_filter.prepare_kalman_data(
        val_ball_eng, scaler_obs=s_obs, scaler_ctrl=s_ctrl, fit_scalers=False
    )
    
    # 2. Load Base Model
    with open(config.MODELS_DIR / "kf_model.pkl", "rb") as f:
        base_kf = pickle.load(f)
    
    # 3. Vary R-floor
    # The note mentioned 1e-4 was used. We'll try a range.
    floors = [1e-6, 1e-4, 1e-3, 1e-2, 0.1, 0.5]
    results = []
    
    for floor in floors:
        print(f"  Testing R-floor = {floor}")
        # Clone kf and modify R
        kf = pickle.loads(pickle.dumps(base_kf))
        # Apply floor to the diagonal of R
        # R is diagonal already in this implementation
        R_diag = np.diag(kf.R)
        kf.R = np.diag(np.maximum(R_diag, floor))
        
        # Run filter
        val_df_temp = kalman_filter.generate_kalman_features(kf, data_dict_val, val_ball_eng)
        val_df_temp = baseline_models.prepare_labels(val_df_temp, val_meta)
        
        y_true = val_df_temp[config.TARGET_COL].values
        y_prob = kf.predict_win_probability(val_df_temp)
        
        brier = brier_score_loss(y_true, y_prob)
        
        # Calculate Volatility (delivery-to-delivery change in WP)
        volatility = np.mean(np.abs(np.diff(y_prob)))
        
        results.append({
            "Measurement Noise Floor (R_min)": floor,
            "Brier Score (Lower is better)": round(brier, 4),
            "Prediction Volatility": round(volatility, 4)
        })
    
    res_df = pd.DataFrame(results)
    output_path = config.TABLES_DIR / "remark3_noise_floor.csv"
    res_df.to_csv(output_path, index=False)
    
    print(f"\nResults saved to {output_path}")
    print(res_df.to_string(index=False))

if __name__ == "__main__":
    run_sensitivity_analysis()
