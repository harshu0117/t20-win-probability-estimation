import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import config
import data_loader
import feature_engineering
import kalman_filter

def analyze_wicket_impact():
    print("--- [K3-Supp] Latent State Diagnostics (Wicket Loading Analysis) ---")
    
    # 1. Load Model
    with open(config.MODELS_DIR / "kf_model.pkl", "rb") as f:
        kf = pickle.load(f)
    
    # 2. Extract H Matrix (Observation Matrix)
    # H maps latent states to observations
    # we want to see what observations inform which states
    report = kalman_filter.report_kf_parameters(kf)
    h_df = report["H (Observation)"]
    
    print("\nObservation Matrix (H) Snippet:")
    print(h_df.loc[h_df.index.str.contains("wickets")])
    
    # 3. Analyze Latent State Correlations in Test Set
    meta, ball = data_loader.load_all_matches(config.DATA_DIR)
    test_meta = meta[meta['year'].isin(config.TEST_SEASONS)]
    test_ball = ball[ball['match_id'].isin(test_meta['match_id'])]
    
    venue_stats, _ = data_loader.compute_venue_stats(ball, meta)
    test_ball_eng = feature_engineering.engineer_features(test_ball, venue_stats)
    
    # Need train for scalers
    train_meta = meta[meta['year'].isin(config.TRAIN_SEASONS)]
    train_ball = ball[ball['match_id'].isin(train_meta['match_id'])]
    train_ball_eng = feature_engineering.engineer_features(train_ball, venue_stats)
    _, s_obs, s_ctrl = kalman_filter.prepare_kalman_data(train_ball_eng, fit_scalers=True)
    
    data_dict_te, _, _ = kalman_filter.prepare_kalman_data(
        test_ball_eng, scaler_obs=s_obs, scaler_ctrl=s_ctrl, fit_scalers=False
    )
    
    test_df_kf = kalman_filter.generate_kalman_features(kf, data_dict_te, test_ball_eng)
    
    # Correlation between latent states
    latent_cols = ['kf_strength', 'kf_momentum', 'kf_pressure']
    corr_matrix = test_df_kf[latent_cols].corr().round(4)
    
    print("\nLatent State Correlation Matrix:")
    print(corr_matrix)
    
    # 4. Wicket Event Study
    # Find balls where a wicket fell and see the jump in latent states
    test_df_kf['is_wicket_next'] = test_df_kf['is_wicket'].shift(-1)
    wicket_balls = test_df_kf[test_df_kf['is_wicket'] == 1].copy()
    
    # ΔState = State(t) - State(t-1)
    for col in latent_cols:
        test_df_kf[f'delta_{col}'] = test_df_kf.groupby(['match_id', 'innings_number'])[col].diff()
    
    wicket_deltas = test_df_kf[test_df_kf['is_wicket'] == 1][[f'delta_{c}' for c in latent_cols]].mean().round(4)
    non_wicket_deltas = test_df_kf[test_df_kf['is_wicket'] == 0][[f'delta_{c}' for c in latent_cols]].mean().round(4)
    
    print("\nMean State Change (Δ) on Wicket vs Non-Wicket Balls:")
    comparison = pd.DataFrame({
        "Wicket Ball": wicket_deltas.values,
        "Non-Wicket Ball": non_wicket_deltas.values
    }, index=latent_cols)
    print(comparison)
    
    # Save diagnostics
    output_path = config.TABLES_DIR / "remark1_pressure_dynamics.csv"
    comparison.to_csv(output_path)
    print(f"\nDiagnostics saved to {output_path}")

if __name__ == "__main__":
    analyze_wicket_impact()
