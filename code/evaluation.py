"""
evaluation.py
=============
Comprehensive model evaluation (paper Sections 5.1–5.4, R1–R3, X1–X4, F5).

Tasks covered
-------------
R1  — Overall metrics table (Brier, Log-Loss, ROC-AUC)
R2  — 95% bootstrap confidence intervals (resampled over matches)
R3  — Pairwise Diebold-Mariano tests (Brier score differences)
X1  — Performance breakdown by innings phase × innings number
X2  — Close vs. non-close match performance
X3  — Season-by-season Brier + AUC
X4  — Leave-one-venue-out cross-venue evaluation
F5  — Prediction volatility (delivery-to-delivery WP change)

Public API
----------
    compute_overall_metrics(df)           -> metrics_df         (R1)
    compute_bootstrap_ci(df, n_resamples) -> ci_df              (R2)
    diebold_mariano_test(df)              -> dm_df               (R3)
    compute_phase_breakdown(df)           -> phase_df            (X1)
    compute_closeness_breakdown(df)       -> closeness_df        (X2)
    compute_season_breakdown(df)          -> season_df           (X3)
    compute_venue_breakdown(df, meta_df)  -> venue_df            (X4)
    compute_volatility(df)               -> (volatility_df, per_match_df) (F5)
    save_all_tables(results_dict)
"""

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from tqdm.auto import tqdm

from config import (
    BOOTSTRAP_N_RESAMPLES,
    BOOTSTRAP_CI_LEVEL,
    CALIBRATION_N_BINS,
    CLOSE_MATCH_MIN_CROSSES,
    VENUE_MIN_MATCHES,
    TABLES_DIR,
    TARGET_COL,
)

# Canonical model names and their win-probability column names
MODEL_COLS = {
    "Kalman Filter"       : "kf_win_prob",
    "Logistic Regression" : "lr_win_prob",
    "Random Forest"       : "rf_win_prob",
    "Gradient Boosting"   : "gb_win_prob",
}


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _brier(y_true, y_pred, weight=None) -> float:
    return float(np.average((y_true - y_pred) ** 2, weights=weight))


def _logloss(y_true, y_pred, weight=None) -> float:
    y_clip = np.clip(y_pred, 1e-10, 1 - 1e-10)
    terms  = y_true * np.log(y_clip) + (1 - y_true) * np.log(1 - y_clip)
    return float(-np.average(terms, weights=weight))


def _auc(y_true, y_pred, weight=None) -> float:
    try:
        return float(roc_auc_score(y_true, y_pred, sample_weight=weight))
    except ValueError:
        return float("nan")


def _available_models(df: pd.DataFrame) -> Dict[str, str]:
    return {name: col for name, col in MODEL_COLS.items() if col in df.columns}


# ---------------------------------------------------------------------------
# R1 — Overall metrics table
# ---------------------------------------------------------------------------

def compute_overall_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute Brier Score, Log-Loss, and ROC-AUC for all models
    on the supplied DataFrame (should be holdout test set).

    Metrics are temporally weighted using 'temporal_weight' if present.

    Returns
    -------
    pd.DataFrame with columns [Model, Brier_Score, Log_Loss, ROC_AUC]
    sorted by Brier_Score ascending.
    """
    y_true  = df[TARGET_COL].values
    weights = df["temporal_weight"].values if "temporal_weight" in df else None

    rows = []
    for name, col in _available_models(df).items():
        y_pred = df[col].values
        rows.append({
            "Model"      : name,
            "Brier_Score": round(_brier(y_true, y_pred, weights), 6),
            "Log_Loss"   : round(_logloss(y_true, y_pred, weights), 6),
            "ROC_AUC"    : round(_auc(y_true, y_pred, weights), 6),
        })

    return pd.DataFrame(rows).sort_values("Brier_Score").reset_index(drop=True)


# ---------------------------------------------------------------------------
# R2 — Bootstrap confidence intervals
# ---------------------------------------------------------------------------

def compute_bootstrap_ci(
    df: pd.DataFrame,
    n_resamples: int = BOOTSTRAP_N_RESAMPLES,
    ci_level: float = BOOTSTRAP_CI_LEVEL,
    seed: int = 42,
) -> pd.DataFrame:
    """
    95% bootstrap confidence intervals for Brier Score and AUC.
    FIXED: Massively optimized the resampling loop to prevent timeouts for KF.
    """
    rng      = np.random.default_rng(seed)
    alpha    = 1.0 - ci_level
    
    # Pre-group indices by match_id to avoid slow .isin() lookups in the loop
    match_indices = df.groupby("match_id").indices
    matches = list(match_indices.keys())
    
    # Extract true values once
    y_true_full = df[TARGET_COL].values
    rows = []

    for name, col in _available_models(df).items():
        y_pred_full = df[col].values
        
        # Point estimates
        point_brier = brier_score_loss(y_true_full, y_pred_full)
        point_auc   = roc_auc_score(y_true_full, y_pred_full)

        boot_brier = np.empty(n_resamples)
        boot_auc   = np.empty(n_resamples)

        # Optimized Bootstrap Loop
        for b in range(n_resamples):
            sampled_matches = rng.choice(matches, size=len(matches), replace=True)
            
            # Fast concatenation of pre-grouped indices
            idx = np.concatenate([match_indices[m] for m in sampled_matches])
            
            y_true_boot = y_true_full[idx]
            y_pred_boot = y_pred_full[idx]
            
            boot_brier[b] = brier_score_loss(y_true_boot, y_pred_boot)
            
            # Handle edge cases where a bootstrap sample might only have 1 class
            try:
                boot_auc[b] = roc_auc_score(y_true_boot, y_pred_boot)
            except ValueError:
                boot_auc[b] = np.nan

        # Calculate Percentiles
        rows.append({
            "Model"         : name,
            "Metric"        : "Brier_Score",
            "Point_Estimate": round(point_brier, 6),
            "CI_Lower"      : round(float(np.nanpercentile(boot_brier, 100 * alpha / 2)), 6),
            "CI_Upper"      : round(float(np.nanpercentile(boot_brier, 100 * (1 - alpha / 2))), 6),
        })
        
        rows.append({
            "Model"         : name,
            "Metric"        : "ROC_AUC",
            "Point_Estimate": round(point_auc, 6),
            "CI_Lower"      : round(float(np.nanpercentile(boot_auc, 100 * alpha / 2)), 6),
            "CI_Upper"      : round(float(np.nanpercentile(boot_auc, 100 * (1 - alpha / 2))), 6),
        })

    return pd.DataFrame(rows).sort_values(["Model", "Metric"]).reset_index(drop=True)


# ---------------------------------------------------------------------------
# R3 — Diebold-Mariano tests
# ---------------------------------------------------------------------------

def diebold_mariano_test(df: pd.DataFrame) -> pd.DataFrame:
    """
    Pairwise Diebold-Mariano tests on ball-level Brier score differences.
    FIXED: Increased lag to 6 (Newey-West HAC) to account for full-over autocorrelation.
    FIXED: Added missing baseline comparisons to complete the reporting table.
    """
    # Added the two missing baseline pairs to generate all p-values
    pairs = [
        ("Kalman Filter",       "Logistic Regression"),
        ("Kalman Filter",       "Random Forest"),
        ("Kalman Filter",       "Gradient Boosting"),
        ("Random Forest",       "Gradient Boosting"),
        ("Logistic Regression", "Random Forest"),      # NEW
        ("Logistic Regression", "Gradient Boosting"),  # NEW
    ]

    y_true = df[TARGET_COL].values
    avail  = _available_models(df)
    rows   = []
    
    # Statistical Rigor: Lag 6 for one full cricket over
    LAG = 6 

    for (a, b) in pairs:
        if a not in avail or b not in avail:
            continue
            
        loss_a = (y_true - df[avail[a]].values) ** 2
        loss_b = (y_true - df[avail[b]].values) ** 2
        d      = loss_a - loss_b
        n      = len(d)
        d_bar  = d.mean()
        
        # Newey-West HAC variance estimator with Bartlett kernel
        gamma0 = np.var(d, ddof=1)
        var_d_sum = gamma0
        
        for j in range(1, LAG + 1):
            if n > j:
                gamma_j = np.cov(d[:-j], d[j:])[0, 1]
                w_j = 1.0 - (j / (LAG + 1))  # Bartlett weights
                var_d_sum += 2 * w_j * gamma_j
                
        var_d  = var_d_sum / n
        dm_stat = d_bar / np.sqrt(max(var_d, 1e-12))
        p_val   = 2.0 * (1.0 - stats.norm.cdf(abs(dm_stat)))

        rows.append({
            "Model_A"     : a,
            "Model_B"     : b,
            "Mean_diff"   : round(d_bar, 6),
            "DM_Statistic": round(dm_stat, 4),
            "p_value"     : round(p_val, 6),
            "Significant" : p_val < 0.05,
        })

    return pd.DataFrame(rows).sort_values(["Model_A", "Model_B"]).reset_index(drop=True)


# ---------------------------------------------------------------------------
# X1 — Performance by innings phase
# ---------------------------------------------------------------------------

def compute_phase_breakdown(df: pd.DataFrame) -> pd.DataFrame:
    """
    Brier Score and AUC broken down by:
      Phase : Powerplay (overs 1–6), Middle (7–15), Death (16–20)
      Innings: 1st innings, 2nd innings (chase)

    Returns
    -------
    pd.DataFrame with columns [Innings, Phase, Model, Brier, AUC]
    """
    def _phase(row):
        if row["phase_powerplay"] == 1:
            return "Powerplay (1-6)"
        elif row["phase_middle"] == 1:
            return "Middle (7-15)"
        else:
            return "Death (16-20)"

    df = df.copy()
    df["Phase"] = df.apply(_phase, axis=1)
    y_true_all  = df[TARGET_COL].values
    avail       = _available_models(df)

    rows = []
    for innings in [1, 2]:
        for phase in ["Powerplay (1-6)", "Middle (7-15)", "Death (16-20)"]:
            mask  = (df["innings_number"] == innings) & (df["Phase"] == phase)
            sub   = df[mask]
            if len(sub) < 10:
                continue
            y_sub = sub[TARGET_COL].values
            for name, col in avail.items():
                y_pred = sub[col].values
                rows.append({
                    "Innings": innings,
                    "Phase"  : phase,
                    "Model"  : name,
                    "Brier"  : round(_brier(y_sub, y_pred), 6),
                    "AUC"    : round(_auc(y_sub, y_pred), 6),
                    "N_balls": int(len(sub)),
                })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# X2 — Close vs. non-close matches
# ---------------------------------------------------------------------------

def compute_closeness_breakdown(df: pd.DataFrame) -> pd.DataFrame:
    """
    Close match definition (X2):
        Win probability (kf_win_prob) crossed 50% at least CLOSE_MATCH_MIN_CROSSES
        times during the second innings.

    Returns
    -------
    pd.DataFrame with columns [Match_Type, Model, Brier, AUC, N_matches]
    """
    inn2 = df[df["innings_number"] == 2].copy()

    def _count_crosses(probs: pd.Series) -> int:
        arr    = probs.values
        above  = (arr > 0.5).astype(int)
        return int(np.sum(np.abs(np.diff(above))))

    kf_col = MODEL_COLS["Kalman Filter"]
    if kf_col not in inn2.columns:
        kf_col = list(_available_models(inn2).values())[0]

    cross_counts = (
        inn2.groupby("match_id")[kf_col]
        .apply(_count_crosses)
        .reset_index()
        .rename(columns={kf_col: "crosses"})
    )
    close_match_ids = set(
        cross_counts.loc[
            cross_counts["crosses"] >= CLOSE_MATCH_MIN_CROSSES, "match_id"
        ]
    )

    df2    = df.copy()
    df2["match_type"] = df2["match_id"].apply(
        lambda m: "Close" if m in close_match_ids else "Non-Close"
    )

    avail = _available_models(df2)
    rows  = []
    for mtype in ["Close", "Non-Close"]:
        sub = df2[df2["match_type"] == mtype]
        if len(sub) == 0:
            continue
        y_sub = sub[TARGET_COL].values
        n_matches = sub["match_id"].nunique()
        for name, col in avail.items():
            rows.append({
                "Match_Type": mtype,
                "Model"     : name,
                "Brier"     : round(_brier(y_sub, sub[col].values), 6),
                "AUC"       : round(_auc(y_sub, sub[col].values), 6),
                "N_matches" : n_matches,
            })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# X3 — Season-by-season performance
# ---------------------------------------------------------------------------

def compute_season_breakdown(df: pd.DataFrame) -> pd.DataFrame:
    """
    Brier Score and AUC for each model, for each IPL season individually.

    Returns
    -------
    pd.DataFrame with columns [Season, Model, Brier, AUC]
    """
    avail   = _available_models(df)
    seasons = sorted(df["year"].dropna().unique().astype(int))
    rows    = []

    for season in seasons:
        sub = df[df["year"] == season]
        if len(sub) < 20:
            continue
        y_sub = sub[TARGET_COL].values
        for name, col in avail.items():
            rows.append({
                "Season" : int(season),
                "Model"  : name,
                "Brier"  : round(_brier(y_sub, sub[col].values), 6),
                "AUC"    : round(_auc(y_sub, sub[col].values), 6),
                "N_balls": len(sub),
            })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# X4 — Cross-venue performance (leave-one-venue-out)
# ---------------------------------------------------------------------------

def compute_venue_breakdown(
    df: pd.DataFrame,
    metadata_df: pd.DataFrame,
    min_matches: int = VENUE_MIN_MATCHES,
) -> pd.DataFrame:
    """
    Leave-one-venue-out evaluation.

    For each major venue (≥ min_matches matches), report model Brier + AUC
    on balls from that venue using the globally trained model predictions
    (i.e., we do not retrain — we evaluate on held-out venue subsets).

    Note: Full retrain-per-venue is computationally prohibitive; this
    analysis uses existing predictions to measure venue-conditional accuracy.

    Returns
    -------
    pd.DataFrame with columns [Venue, Model, Brier, AUC, N_matches]
    """
    if "venue" not in df.columns:
        venue_map = metadata_df.set_index("match_id")["venue"].to_dict()
        df = df.copy()
        df["venue"] = df["match_id"].map(venue_map)

    venue_match_counts = df.groupby("venue")["match_id"].nunique()
    major_venues       = venue_match_counts[venue_match_counts >= min_matches].index

    avail = _available_models(df)
    rows  = []

    for venue in major_venues:
        sub = df[df["venue"] == venue]
        y_sub = sub[TARGET_COL].values
        n_matches = sub["match_id"].nunique()
        for name, col in avail.items():
            rows.append({
                "Venue"    : venue,
                "Model"    : name,
                "Brier"    : round(_brier(y_sub, sub[col].values), 6),
                "AUC"      : round(_auc(y_sub, sub[col].values), 6),
                "N_matches": int(n_matches),
            })

    return pd.DataFrame(rows).sort_values(["Venue", "Model"])


# ---------------------------------------------------------------------------
# F5 — Prediction volatility
# ---------------------------------------------------------------------------

def compute_volatility(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Mean absolute delivery-to-delivery change in win probability per model.

    |Δp_t| = |p̂_t − p̂_{t-1}|  (within each match-innings sequence)

    Returns
    -------
    summary_df    : overall + phase-level average volatility per model
    per_match_df  : per-match volatility (for box-plot, F5)
    """
    avail = _available_models(df)
    df_s  = df.sort_values(["match_id", "innings_number", "over", "ball"])

    phase_map = {
        "phase_powerplay": "Powerplay (1-6)",
        "phase_middle"   : "Middle (7-15)",
        "phase_death"    : "Death (16-20)",
    }

    summary_rows  = []
    permatch_rows = []

    for name, col in avail.items():
        # Compute per-ball deltas within each innings
        grp   = df_s.groupby(["match_id", "innings_number"])[col]
        delta = grp.transform(lambda x: x.diff().abs())
        df_s[f"_delta_{col}"] = delta

        # Per-match mean volatility
        per_match = (
            df_s.groupby("match_id")[f"_delta_{col}"]
            .mean()
            .reset_index()
            .rename(columns={f"_delta_{col}": "volatility"})
        )
        per_match["Model"] = name
        permatch_rows.append(per_match)

        # Overall
        summary_rows.append({
            "Model"  : name,
            "Phase"  : "Overall",
            "Mean_|Δp|": round(df_s[f"_delta_{col}"].mean(), 6),
        })

        # Phase-level
        for phase_col, phase_label in phase_map.items():
            if phase_col in df_s.columns:
                sub = df_s[df_s[phase_col] == 1]
                summary_rows.append({
                    "Model"   : name,
                    "Phase"   : phase_label,
                    "Mean_|Δp|": round(sub[f"_delta_{col}"].mean(), 6),
                })

    # Cleanup temp columns
    for name, col in avail.items():
        df_s.drop(columns=[f"_delta_{col}"], inplace=True, errors="ignore")

    summary_df   = pd.DataFrame(summary_rows)
    per_match_df = pd.concat(permatch_rows, ignore_index=True) if permatch_rows else pd.DataFrame()

    return summary_df, per_match_df





def compute_match_situations_table(df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes Table 9: Performance by match situation (Close, Moderate, One-sided, Death, Powerplay).
    Extracts margins dynamically from the final delivery of the match.
    """
    inn2 = df[df["innings_number"] == 2]
    
    # Identify the last ball of the 2nd innings for every match
    last_balls = inn2.sort_values(["match_id", "over", "ball"]).groupby("match_id").last().reset_index()
    
    close_mids, mod_mids, oneside_mids = set(), set(), set()
    
    for _, row in last_balls.iterrows():
        mid = row["match_id"]
        # If the chasing team won (Batting Team = Winner)
        if row[TARGET_COL] == 1:
            wickets_left = 10 - row["wickets_fallen"]
            if wickets_left <= 2:
                close_mids.add(mid)
            elif wickets_left <= 5:
                mod_mids.add(mid)
            else:
                oneside_mids.add(mid)
        # If the defending team won
        else:
            runs_short = row["runs_required"] if pd.notna(row["runs_required"]) else 0
            if runs_short <= 10:
                close_mids.add(mid)
            elif runs_short <= 30:
                mod_mids.add(mid)
            else:
                oneside_mids.add(mid)
                
    # Define the exact slices requested in the paper draft
    categories = {
        "Close matches (≤ 10 runs or ≤ 2 wickets)": df[df["match_id"].isin(close_mids)],
        "Moderate (11–30 runs or 3–5 wickets)": df[df["match_id"].isin(mod_mids)],
        "One-sided (> 30 runs or > 5 wickets)": df[df["match_id"].isin(oneside_mids)],
        "Death overs (16–20)": df[(df["over"] >= 15)], # 0-indexed over 15 is the 16th over
        "Powerplay (1–6)": df[(df["over"] <= 5)],
    }
    
    avail = _available_models(df)
    rows = []
    
    for cat_name, sub in categories.items():
        if len(sub) == 0:
            continue
            
        y_sub = sub[TARGET_COL].values
        briers = {}
        
        # Calculate Brier scores (Mean Squared Error between probability and outcome)
        for model_name, col in avail.items():
            pred = sub[col].values
            mse = np.mean((y_sub - pred) ** 2)
            briers[model_name] = round(mse, 4)
            
        kf_b = briers.get("Kalman Filter", np.nan)
        lr_b = briers.get("Logistic Regression", np.nan)
        
        # Calculate Improvement Percentage
        kf_imp = round(((lr_b - kf_b) / lr_b) * 100, 1) if not np.isnan(kf_b) and lr_b > 0 else np.nan
            
        rows.append({
            "Match Situation": cat_name,
            "KF Brier": kf_b,
            "LR Brier": lr_b,
            "KF Improvement": f"{kf_imp}%",
            "RF Brier": briers.get("Random Forest", np.nan),
            "GB Brier": briers.get("Gradient Boosting", np.nan),
            "N (Balls)": len(sub),
            "N (Matches)": sub["match_id"].nunique()
        })
        
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Save all tables
# ---------------------------------------------------------------------------

def save_all_tables(results: Dict[str, pd.DataFrame]) -> None:
    """
    Save every results DataFrame to TABLES_DIR as CSV.

    Parameters
    ----------
    results : dict mapping filename-stem to DataFrame, e.g.:
              {'R1_overall_metrics': df1, 'R2_bootstrap_ci': df2, ...}
    """
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    for stem, df in results.items():
        if df is None or (isinstance(df, pd.DataFrame) and df.empty):
            continue
        path = TABLES_DIR / f"{stem}.csv"
        df.to_csv(path, index=False)
        print(f"  Saved: {path}")



# ---------------------------------------------------------------------------
def compute_failure_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes model failure rates and Mean Absolute Error (MAE) stratified 
    by prediction confidence buckets (0-10%, 10-20%, etc.).
    """
    y_true = df[TARGET_COL].values
    avail = _available_models(df)
    
    # 1. Calculate overall failure rate (Brier > 0.5 means the model 
    # was > 70.7% confident in the completely wrong outcome).
    print("\n--- Model Failure Rates (Individual Brier > 0.5) ---")
    for name, col in avail.items():
        y_pred = df[col].values
        brier_per_ball = (y_true - y_pred) ** 2
        failure_rate = np.mean(brier_per_ball > 0.5) * 100
        print(f"  {name}: {failure_rate:.2f}% of predictions")
        
    # 2. Stratify MAE by prediction confidence buckets
    rows = []
    bins = np.linspace(0, 1, 11) # 10 decile bins from 0.0 to 1.0
    
    for name, col in avail.items():
        y_pred = df[col].values
        
        # Digitize places probabilities into bins 0-9
        bin_indices = np.digitize(y_pred, bins) - 1
        bin_indices = np.clip(bin_indices, 0, len(bins) - 2)
        
        for i in range(len(bins) - 1):
            mask = (bin_indices == i)
            if not np.any(mask):
                continue
            
            bin_y_true = y_true[mask]
            bin_y_pred = y_pred[mask]
            
            # Mean Absolute Error for this specific bucket
            mae = np.mean(np.abs(bin_y_true - bin_y_pred))
            
            rows.append({
                "Model": name,
                "Confidence Bucket": f"{bins[i]*100:.0f}% - {bins[i+1]*100:.0f}%",
                "MAE": round(mae, 4),
                "N (Balls)": np.sum(mask)
            })
            
    df_mae = pd.DataFrame(rows)
    return df_mae