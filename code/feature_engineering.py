"""
feature_engineering.py
======================
Feature construction pipeline for the IPL KF Win-Probability model.

All features are documented in paper Section 3 / D5.

Feature groups (D5)
-------------------
Group A — Instantaneous / ball-level
    current_score, wickets_fallen, balls_bowled, balls_remaining,
    current_run_rate, target, runs_required, required_run_rate,
    wickets_in_hand, run_rate_diff, pressure_index, innings_number

Group B — Rolling-window features  (windows: 6, 12, 24 balls)
    runs_last_{w}_balls, wickets_last_{w}_balls,
    dots_last_{w}_balls, boundaries_last_{w}_balls
    → 4 stats × 3 windows = 12 rolling features

Group C — Phase / match-context indicators
    phase_powerplay, phase_middle, phase_death
    (one-hot; mutually exclusive and exhaustive)

Group D — Venue & conditions  (D4)
    venue_avg_first_innings  — historical average 1st-innings total at venue
    par_deviation            — current_score minus on-pace par score

Total feature count: 13 + 12 + 3 + 2 = 30 features

Public API
----------
    engineer_features(ball_df, venue_stats_df) -> enriched_df
    get_obs_feature_matrix(df)  -> np.ndarray   (for KF observation vector)
    get_ctrl_feature_matrix(df) -> np.ndarray   (for KF control vector)
    get_ml_feature_matrix(df)   -> np.ndarray   (for baseline ML models)
    get_target_vector(df)       -> np.ndarray
"""

from typing import List

import numpy as np
import pandas as pd

from config import (
    ROLLING_WINDOWS,
    OBS_FEATURES,
    CTRL_FEATURES,
    ML_FEATURES,
    TARGET_COL,
    KF_VENUE_AVG_DEFAULT,
)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _rolling_per_group(
    df: pd.DataFrame,
    group_col: str,
    value_col: str,
    window: int,
    aggfunc: str = "sum",
) -> pd.Series:
    """
    Compute a look-back rolling aggregate (shifted by 1 to avoid leakage).

    Parameters
    ----------
    df        : DataFrame already sorted by time within group
    group_col : column identifying each independent sequence (e.g. 'match_innings')
    value_col : column to aggregate
    window    : number of past balls to include
    aggfunc   : 'sum' (default)

    Returns
    -------
    pd.Series aligned to df.index
    """
    return (
        df.groupby(group_col)[value_col]
        .transform(lambda x: x.shift(1).rolling(window, min_periods=1).agg(aggfunc))
    )


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def engineer_features(
    ball_df: pd.DataFrame,
    venue_stats_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Full feature engineering pipeline.  Must be called BEFORE train/val/test
    split to ensure rolling windows are computed correctly within each
    match-innings sequence, then the caller can subset by year.

    Parameters
    ----------
    ball_df       : raw delivery DataFrame from data_loader.load_all_matches()
    venue_stats_df: venue statistics from data_loader.compute_venue_stats()
                    must contain columns ['venue', 'avg_first_innings_total']

    Returns
    -------
    Enriched DataFrame with all 30 features added in-place.
    """
    df = (
        ball_df
        .copy()
        .sort_values(["match_id", "innings_number", "over", "ball"])
        .reset_index(drop=True)
    )

    # Unique sequence key (each innings of each match is independent)
    df["match_innings"] = df["match_id"].astype(str) + "_" + df["innings_number"].astype(str)

    # -----------------------------------------------------------------------
    # Group A — Derived instantaneous features
    # -----------------------------------------------------------------------
    df["wickets_in_hand"] = 10 - df["wickets_fallen"]
    df["run_rate_diff"]   = (df["required_run_rate"] - df["current_run_rate"]).fillna(0.0)

    df["pressure_index"] = 0.0
    chase_mask = df["target"].notna() & (df["balls_remaining"] > 0)
    df.loc[chase_mask, "pressure_index"] = (
        df.loc[chase_mask, "runs_required"] /
        df.loc[chase_mask, "balls_remaining"].clip(lower=1)
    )

    # -----------------------------------------------------------------------
    # Group B — Rolling-window features
    # -----------------------------------------------------------------------
    # Helper binary columns (temporary; dropped at end)
    df["_is_dot"]      = (df["total_runs"] == 0).astype(int)
    df["_is_boundary"] = df["batter_runs"].isin([4, 6]).astype(int)

    for w in ROLLING_WINDOWS:
        df[f"runs_last_{w}_balls"]       = _rolling_per_group(df, "match_innings", "total_runs",   w)
        df[f"wickets_last_{w}_balls"]    = _rolling_per_group(df, "match_innings", "is_wicket",    w)
        df[f"dots_last_{w}_balls"]       = _rolling_per_group(df, "match_innings", "_is_dot",      w)
        df[f"boundaries_last_{w}_balls"] = _rolling_per_group(df, "match_innings", "_is_boundary", w)

    df.drop(columns=["_is_dot", "_is_boundary"], inplace=True)

    # -----------------------------------------------------------------------
    # Group C — Phase indicators
    # -----------------------------------------------------------------------
    df["phase_powerplay"] = (df["over"] <= 5).astype(int)                          # overs 1–6
    df["phase_middle"]    = ((df["over"] >= 6) & (df["over"] <= 15)).astype(int)   # overs 7–15 (incl.)  (corrected: index 6..15)
    df["phase_death"]     = (df["over"] >= 16).astype(int)                         # overs 17–20

    # -----------------------------------------------------------------------
    # Group D — Venue fixed effect & par deviation  (D4)
    # -----------------------------------------------------------------------
    # Map venue average onto each ball
    venue_avg_map = (
        venue_stats_df.set_index("venue")["avg_first_innings_total"].to_dict()
    )
    df["venue_avg_first_innings"] = (
        df["match_id"]
        .map(_build_match_venue_map(df))
        .map(venue_avg_map)
        .fillna(KF_VENUE_AVG_DEFAULT)
    )

    # Par score for the current ball (linear interpolation of 1st-innings total)
    # par_deviation = current_score − (venue_avg × balls_bowled / 120)
    df["par_deviation"] = 0.0
    chase_mask2 = df["innings_number"] == 2
    df.loc[chase_mask2, "par_deviation"] = (
        df.loc[chase_mask2, "current_score"] -
        df.loc[chase_mask2, "venue_avg_first_innings"] *
        df.loc[chase_mask2, "balls_bowled"] / 120.0
    )

    # -----------------------------------------------------------------------
    # Fill any remaining NaNs / infinities with 0
    # (guards against edge cases at the very first ball of an innings)
    # -----------------------------------------------------------------------
    feature_cols = OBS_FEATURES + CTRL_FEATURES + [
        "wickets_in_hand", "par_deviation", "venue_avg_first_innings",
    ]
    for col in feature_cols:
        if col in df.columns:
            df[col] = (
                df[col]
                .replace([np.inf, -np.inf], np.nan)
                .fillna(0.0)
            )

    # Drop temporary grouping key
    df.drop(columns=["match_innings"], inplace=True)

    print(
        f"✅  Feature engineering complete.  "
        f"Shape: {df.shape}  |  "
        f"New feature cols: {len(feature_cols)}"
    )
    return df


def _build_match_venue_map(df: pd.DataFrame) -> dict:
    """Build a match_id → venue lookup from the ball DataFrame if available."""
    if "venue" in df.columns:
        return df.drop_duplicates("match_id").set_index("match_id")["venue"].to_dict()
    return {}


# ---------------------------------------------------------------------------
# Feature matrix extractors
# ---------------------------------------------------------------------------

def get_obs_feature_matrix(df: pd.DataFrame) -> np.ndarray:
    """
    Return the observation matrix Y fed into the Kalman Filter.

    Columns: OBS_FEATURES (defined in config.py)
    Shape  : (n_balls, n_obs_features)
    """
    _check_columns(df, OBS_FEATURES, "OBS_FEATURES")
    return df[OBS_FEATURES].values.astype(np.float64)


def get_ctrl_feature_matrix(df: pd.DataFrame) -> np.ndarray:
    """
    Return the control-input matrix U fed into the KF state equation.

    Columns: CTRL_FEATURES (defined in config.py)
    Shape  : (n_balls, n_ctrl_features)
    """
    _check_columns(df, CTRL_FEATURES, "CTRL_FEATURES")
    return df[CTRL_FEATURES].values.astype(np.float64)


def get_ml_feature_matrix(df: pd.DataFrame) -> np.ndarray:
    """
    Return the feature matrix used by LR / RF / GB baselines.

    Columns: ML_FEATURES (defined in config.py)
    Shape  : (n_balls, n_ml_features)
    """
    # 'venue_avg_first_innings' may not be in ML_FEATURES if venue is not used;
    # fall back gracefully.
    cols = [c for c in ML_FEATURES if c in df.columns]
    return df[cols].values.astype(np.float64)


def get_target_vector(df: pd.DataFrame) -> np.ndarray:
    """Return binary target vector (1 = batting team won, 0 = lost)."""
    if TARGET_COL not in df.columns:
        raise KeyError(
            f"Column '{TARGET_COL}' not found. "
            "Call prepare_labels() from baseline_models.py first."
        )
    return df[TARGET_COL].values.astype(np.float64)


def get_feature_names() -> dict:
    """
    Return a structured dict documenting all feature groups (D5 compliance).
    """
    return {
        "Group A — Instantaneous / ball-level": [
            "current_score", "wickets_fallen", "balls_bowled", "balls_remaining",
            "current_run_rate", "target", "runs_required", "required_run_rate",
            "wickets_in_hand", "run_rate_diff", "pressure_index", "innings_number",
        ],
        "Group B — Rolling-window (windows: 6, 12, 24 balls)": [
            f"{stat}_last_{w}_balls"
            for stat in ["runs", "wickets", "dots", "boundaries"]
            for w in ROLLING_WINDOWS
        ],
        "Group C — Phase / match-context indicators": [
            "phase_powerplay", "phase_middle", "phase_death",
        ],
        "Group D — Venue & conditions": [
            "venue_avg_first_innings",
            "par_deviation",
        ],
    }


def print_feature_summary() -> None:
    """Print a D5-compliant feature list to stdout."""
    groups = get_feature_names()
    total  = 0
    print("\n── Feature List (D5) ──────────────────────────────────────")
    for group, features in groups.items():
        print(f"\n  {group}  ({len(features)} features)")
        for f in features:
            print(f"    · {f}")
        total += len(features)
    print(f"\n  TOTAL: {total} features")
    print("────────────────────────────────────────────────────────────\n")


# ---------------------------------------------------------------------------
# Internal validation
# ---------------------------------------------------------------------------

def _check_columns(df: pd.DataFrame, cols: List[str], name: str) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise KeyError(
            f"{name}: missing columns {missing}. "
            "Make sure engineer_features() has been called first."
        )
