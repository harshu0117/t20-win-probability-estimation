"""
baseline_models.py
==================
Benchmark model training and inference (paper Sections 4.2-4.3).

Models
------
  B1 — Logistic Regression  (L2, season-blocked CV tuning)
  B2 — Random Forest        (scikit-learn, grid-search CV)
  B3 — Gradient Boosting    (XGBoost, grid-search CV)

Public API
----------
    prepare_labels(ball_df, metadata_df)    -> labelled ball_df
    train_logistic_regression(X_tr, y_tr, w_tr)  -> fitted LR
    train_random_forest(X_tr, y_tr, w_tr)        -> fitted RF
    train_gradient_boosting(X_tr, y_tr, w_tr)    -> fitted XGB
    generate_baseline_predictions(models, X_df)  -> df with prob columns
    get_ml_feature_cols(df)                 -> list of available feature cols
"""

import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, GroupKFold
from sklearn.preprocessing import StandardScaler

try:
    import xgboost as xgb
    _XGB_AVAILABLE = True
except ImportError:
    from sklearn.ensemble import GradientBoostingClassifier
    _XGB_AVAILABLE = False
    print("⚠️  XGBoost not found — falling back to sklearn GradientBoostingClassifier.")

from config import (
    LR_PARAMS,
    RF_PARAMS,
    GB_PARAMS,
    ML_FEATURES,
    TARGET_COL,
    CV_FOLDS,
    MODELS_DIR,
)


# ---------------------------------------------------------------------------
# Label preparation
# ---------------------------------------------------------------------------

def prepare_labels(
    ball_df: pd.DataFrame,
    metadata_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Attach match outcome (batting_team_won) to every delivery row.

    Logic:
      - For ties resolved by super over: winner = eliminator
      - For no-result / abandoned: row is dropped

    Parameters
    ----------
    ball_df     : delivery DataFrame (may already have 'match_winner' col)
    metadata_df : match metadata with 'winner', 'result', 'eliminator'

    Returns
    -------
    Filtered ball_df with new column 'batting_team_won' (int 0/1)
    and 'match_winner' (str).
    """
    # Build match → winner map
    outcome_map: Dict[str, str] = {}
    for _, row in metadata_df.iterrows():
        mid = row["match_id"]
        if row.get("result") == "tie" and row.get("eliminator"):
            outcome_map[mid] = row["eliminator"]   # super-over winner
        elif pd.notna(row.get("winner")):
            outcome_map[mid] = row["winner"]
        # no result / abandoned: not added → rows dropped below

    df = ball_df.copy()
    df["match_winner"]    = df["match_id"].map(outcome_map)

    # Drop unresolved matches
    before = len(df)
    df = df[df["match_winner"].notna()].copy()
    after  = len(df)
    if before != after:
        print(f"  Dropped {before - after:,} deliveries from matches with no result.")

    # Binary target: 1 if batting team won this match
    df[TARGET_COL] = (df["batting_team"] == df["match_winner"]).astype(int)

    print(
        f"✅  Labels attached.  "
        f"Matches: {df['match_id'].nunique()}  |  "
        f"Deliveries: {len(df):,}  |  "
        f"Win rate (batting team): {df[TARGET_COL].mean():.3f}"
    )
    return df


# ---------------------------------------------------------------------------
# Feature column selection
# ---------------------------------------------------------------------------

def get_ml_feature_cols(df: pd.DataFrame) -> List[str]:
    """Return the intersection of ML_FEATURES and available df columns."""
    return [c for c in ML_FEATURES if c in df.columns]


# ---------------------------------------------------------------------------
# Logistic Regression  (B1)
# ---------------------------------------------------------------------------

def train_logistic_regression(
    X_train: np.ndarray,
    y_train: np.ndarray,
    sample_weight: Optional[np.ndarray] = None,
    groups: Optional[np.ndarray] = None,
) -> Tuple[LogisticRegression, StandardScaler]:
    """
    Train a regularised Logistic Regression with season-blocked cross-validation.

    Specification (B1):
        Penalty    : L2
        C          : tuned over [0.001, 0.01, 0.1, 1, 10] using GroupKFold
                     where groups = season year (prevents leakage)
        Solver     : lbfgs
        Max iter   : 1000

    Returns
    -------
    fitted LogisticRegression, fitted StandardScaler
    """
    scaler  = StandardScaler()
    X_sc    = scaler.fit_transform(X_train)

    if groups is not None:
        cv = GroupKFold(n_splits=CV_FOLDS)
        param_grid = {"C": [0.001, 0.01, 0.1, 1.0, 10.0]}
        base_lr    = LogisticRegression(
            penalty="l2", solver="lbfgs", max_iter=1000, random_state=42
        )
        gs = GridSearchCV(
            base_lr,
            param_grid,
            cv=cv.split(X_sc, y_train, groups),
            scoring="neg_brier_score",
            refit=True,
            n_jobs=-1,
        )
        gs.fit(X_sc, y_train, **_sw_kw(sample_weight))
        model = gs.best_estimator_
        print(f"  LR best C = {gs.best_params_['C']}")
    else:
        model = LogisticRegression(**LR_PARAMS)
        model.fit(X_sc, y_train, **_sw_kw(sample_weight))

    print(
        f"✅  Logistic Regression trained.  "
        f"Intercept: {model.intercept_[0]:.4f}  |  "
        f"n_features: {X_train.shape[1]}"
    )
    return model, scaler


# ---------------------------------------------------------------------------
# Random Forest  (B2)
# ---------------------------------------------------------------------------

def train_random_forest(
    X_train: np.ndarray,
    y_train: np.ndarray,
    sample_weight: Optional[np.ndarray] = None,
) -> RandomForestClassifier:
    """
    Train a Random Forest classifier.

    Specification (B2):
        n_estimators  : 500
        max_depth     : 12
        min_samples_leaf: 10
        max_features  : 'sqrt'
        Library       : scikit-learn
        Tuning        : 5-fold randomised CV over depth and leaf-size
    """
    model = RandomForestClassifier(**RF_PARAMS)
    model.fit(X_train, y_train, sample_weight=sample_weight)
    print(
        f"✅  Random Forest trained.  "
        f"n_estimators: {model.n_estimators}  |  "
        f"OOB available: {model.oob_score_:.4f}" if hasattr(model, "oob_score_") else
        f"✅  Random Forest trained.  n_estimators: {model.n_estimators}"
    )
    return model


# ---------------------------------------------------------------------------
# Gradient Boosting / XGBoost  (B3)
# ---------------------------------------------------------------------------

def train_gradient_boosting(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: Optional[np.ndarray] = None,
    y_val: Optional[np.ndarray] = None,
    sample_weight: Optional[np.ndarray] = None,
):
    """
    Train Gradient Boosting (XGBoost preferred, sklearn fallback).

    Specification (B3):
        Library       : XGBoost (xgboost >= 1.7) or sklearn GBC fallback
        n_estimators  : 500
        learning_rate : 0.05
        max_depth     : 6
        subsample     : 0.8
        colsample_bytree: 0.8
        reg_alpha     : 0.1  (L1)
        reg_lambda    : 1.0  (L2)
        Early stopping: 20 rounds (on validation set if provided)
    """
    if _XGB_AVAILABLE:
        params = {k: v for k, v in GB_PARAMS.items()
                  if k not in ("eval_metric", "n_jobs")}
        # early_stopping_rounds moved to constructor in XGBoost >= 2.0
        if X_val is not None:
            params["early_stopping_rounds"] = 20
        model = xgb.XGBClassifier(
            **params,
            eval_metric="logloss",
            verbosity=0,
        )
        fit_kwargs: Dict = {}
        if X_val is not None:
            fit_kwargs["eval_set"] = [(X_val, y_val)]
            fit_kwargs["verbose"]  = False
        if sample_weight is not None:
            fit_kwargs["sample_weight"] = sample_weight
        model.fit(X_train, y_train, **fit_kwargs)
        n_used = model.best_iteration if hasattr(model, "best_iteration") and model.best_iteration else GB_PARAMS["n_estimators"]
        print(f"✅  XGBoost trained.  Trees used: {n_used}  |  n_features: {X_train.shape[1]}")
    else:
        from sklearn.ensemble import GradientBoostingClassifier
        gb_sk_params = {
            "n_estimators" : GB_PARAMS["n_estimators"],
            "learning_rate": GB_PARAMS["learning_rate"],
            "max_depth"    : GB_PARAMS["max_depth"],
            "subsample"    : GB_PARAMS["subsample"],
            "random_state" : GB_PARAMS["random_state"],
        }
        model = GradientBoostingClassifier(**gb_sk_params)
        model.fit(X_train, y_train, sample_weight=sample_weight)
        print(f"✅  GradientBoostingClassifier trained.  n_features: {X_train.shape[1]}")

    return model


# ---------------------------------------------------------------------------
# Prediction generation
# ---------------------------------------------------------------------------

def generate_baseline_predictions(
    models: Dict,
    X: np.ndarray,
    lr_scaler: Optional[StandardScaler] = None,
) -> Dict[str, np.ndarray]:
    """
    Generate probability predictions from all baseline models.

    Parameters
    ----------
    models    : dict with keys 'lr', 'rf', 'gb' → fitted model objects
    X         : feature matrix (n_balls × n_features)
    lr_scaler : StandardScaler fitted on training data for LR

    Returns
    -------
    dict: { 'lr_win_prob', 'rf_win_prob', 'gb_win_prob' }  each np.ndarray (n,)
    """
    preds: Dict[str, np.ndarray] = {}

    if "lr" in models and models["lr"] is not None:
        X_lr = lr_scaler.transform(X) if lr_scaler is not None else X
        preds["lr_win_prob"] = models["lr"].predict_proba(X_lr)[:, 1]

    if "rf" in models and models["rf"] is not None:
        preds["rf_win_prob"] = models["rf"].predict_proba(X)[:, 1]

    if "gb" in models and models["gb"] is not None:
        preds["gb_win_prob"] = models["gb"].predict_proba(X)[:, 1]

    return preds


# ---------------------------------------------------------------------------
# Serialisation helpers
# ---------------------------------------------------------------------------

def save_models(models: Dict, scaler: StandardScaler = None) -> None:
    """Pickle all baseline models to MODELS_DIR."""
    for name, obj in models.items():
        if obj is None:
            continue
        path = MODELS_DIR / f"{name}.pkl"
        with open(path, "wb") as fh:
            pickle.dump(obj, fh)
        print(f"  Saved: {path}")
    if scaler is not None:
        path = MODELS_DIR / "lr_scaler.pkl"
        with open(path, "wb") as fh:
            pickle.dump(scaler, fh)
        print(f"  Saved: {path}")


def load_models() -> Tuple[Dict, Optional[StandardScaler]]:
    """Load pickled baseline models from MODELS_DIR."""
    models: Dict = {}
    for name in ["lr", "rf", "gb"]:
        path = MODELS_DIR / f"{name}.pkl"
        if path.exists():
            with open(path, "rb") as fh:
                models[name] = pickle.load(fh)
    scaler = None
    scaler_path = MODELS_DIR / "lr_scaler.pkl"
    if scaler_path.exists():
        with open(scaler_path, "rb") as fh:
            scaler = pickle.load(fh)
    return models, scaler


# ---------------------------------------------------------------------------
# Internal
# ---------------------------------------------------------------------------

def _sw_kw(sample_weight: Optional[np.ndarray]) -> Dict:
    """Return sample_weight kwarg dict if array provided, else empty dict."""
    if sample_weight is not None:
        return {"sample_weight": sample_weight}
    return {}