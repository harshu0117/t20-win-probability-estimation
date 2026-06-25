"""
hybrid_models.py
================
Hybrid and extended Kalman Filter models (paper Sections 5.7–5.8, H1–H3).

H1 — Hybrid models: RF + KF states, GB + KF states
H2 — SHAP feature importance for hybrid RF
H3 — Extended Kalman Filter (EKF) and Unscented Kalman Filter (UKF)

The three KF filtered states [Ŝ_t, M̂_t, P̂_t] are appended to the original
ML feature matrix and used to train augmented RF and GB models.

Public API
----------
    build_hybrid_feature_matrix(X_base, kf_states_df) -> np.ndarray
    train_hybrid_rf(X_hybrid, y, w)  -> fitted RF
    train_hybrid_gb(X_hybrid, y, w)  -> fitted GB/XGB
    compute_shap_importance(model, X_hybrid, feature_names) -> shap_df  (H2)
    EKFWinProbability   — Extended Kalman Filter class       (H3)
    UKFWinProbability   — Unscented Kalman Filter class      (H3)
"""

from __future__ import annotations

import pickle
import warnings
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from tqdm.auto import tqdm

from config import RF_PARAMS, GB_PARAMS, KF_N_STATES, KF_Q_DIAG, MODELS_DIR, OBS_FEATURES, CTRL_FEATURES

try:
    import shap
    _SHAP_AVAILABLE = True
except ImportError:
    _SHAP_AVAILABLE = False
    warnings.warn("shap not installed — SHAP analysis will be skipped. `pip install shap`")

try:
    import xgboost as xgb
    _XGB_AVAILABLE = True
except ImportError:
    from sklearn.ensemble import GradientBoostingClassifier
    _XGB_AVAILABLE = False

warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------------
# H1 — Hybrid feature matrix
# ---------------------------------------------------------------------------

KF_STATE_COLS = ["kf_strength", "kf_momentum", "kf_pressure"]


def build_hybrid_feature_matrix(
    X_base: np.ndarray,
    df: pd.DataFrame,
) -> Tuple[np.ndarray, List[str]]:
    """
    Append filtered KF states to the baseline feature matrix.

    Parameters
    ----------
    X_base : (n, d) baseline feature matrix (from get_ml_feature_matrix)
    df     : DataFrame containing 'kf_strength', 'kf_momentum', 'kf_pressure'
             (aligned row-for-row with X_base)

    Returns
    -------
    X_hybrid       : (n, d+3) augmented feature matrix
    hybrid_names   : list of feature names (ML_FEATURES + KF state names)
    """
    kf_cols = [c for c in KF_STATE_COLS if c in df.columns]
    if not kf_cols:
        raise ValueError(
            "KF state columns not found in df. "
            "Run generate_kalman_features() before building hybrid matrix."
        )
    kf_states = df[kf_cols].values.astype(np.float64)
    kf_states = np.nan_to_num(kf_states, nan=0.0, posinf=0.0, neginf=0.0)

    X_hybrid = np.hstack([X_base, kf_states])

    # Feature names for interpretability
    ml_names = [c for c in OBS_FEATURES + CTRL_FEATURES if c in df.columns]
    hybrid_names = ml_names + kf_cols

    return X_hybrid, hybrid_names


# ---------------------------------------------------------------------------
# H1 — Hybrid RF
# ---------------------------------------------------------------------------

def train_hybrid_rf(
    X_hybrid: np.ndarray,
    y_train: np.ndarray,
    sample_weight: Optional[np.ndarray] = None,
) -> RandomForestClassifier:
    """
    Train Random Forest on the hybrid feature set (original + KF states).
    Same hyperparameters as baseline RF (B2) for fair comparison.
    """
    model = RandomForestClassifier(**RF_PARAMS)
    model.fit(X_hybrid, y_train, sample_weight=sample_weight)
    print(
        f"✅  Hybrid RF trained.  "
        f"n_features: {X_hybrid.shape[1]}  "
        f"(base + 3 KF states)"
    )
    return model


# ---------------------------------------------------------------------------
# H1 — Hybrid GB
# ---------------------------------------------------------------------------

def train_hybrid_gb(
    X_hybrid: np.ndarray,
    y_train: np.ndarray,
    X_val: Optional[np.ndarray] = None,
    y_val: Optional[np.ndarray] = None,
    sample_weight: Optional[np.ndarray] = None,
):
    """
    Train Gradient Boosting on the hybrid feature set.
    Same hyperparameters as baseline GB (B3) for fair comparison.
    """
    if _XGB_AVAILABLE:
        params = {k: v for k, v in GB_PARAMS.items()
                  if k not in ("eval_metric", "n_jobs")}
        model = xgb.XGBClassifier(
            **params,
            use_label_encoder=False,
            eval_metric="logloss",
            verbosity=0,
        )
        fit_kw: Dict = {}
        if X_val is not None:
            fit_kw["eval_set"]             = [(X_val, y_val)]
            fit_kw["early_stopping_rounds"] = 20
            fit_kw["verbose"]              = False
        if sample_weight is not None:
            fit_kw["sample_weight"] = sample_weight
        model.fit(X_hybrid, y_train, **fit_kw)
    else:
        from sklearn.ensemble import GradientBoostingClassifier
        params = {k: v for k, v in GB_PARAMS.items()
                  if k in ("n_estimators", "learning_rate", "max_depth",
                            "subsample", "random_state")}
        model = GradientBoostingClassifier(**params)
        model.fit(X_hybrid, y_train, sample_weight=sample_weight)

    print(
        f"✅  Hybrid GB trained.  "
        f"n_features: {X_hybrid.shape[1]}  "
        f"(base + 3 KF states)"
    )
    return model


# ---------------------------------------------------------------------------
# H2 — SHAP feature importance
# ---------------------------------------------------------------------------

def compute_shap_importance(
    model,
    X_hybrid: np.ndarray,
    feature_names: List[str],
    y_true: Optional[np.ndarray] = None,
    n_samples: int = 2000,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Compute feature importance for the hybrid RF model (H2).

    Primary  : SHAP TreeExplainer (exact, fast for tree ensembles).
               Install with: pip install shap
    Fallback : sklearn permutation_importance with neg-Brier scoring.

    Parameters
    ----------
    model        : fitted RandomForestClassifier or XGBClassifier
    X_hybrid     : (n, d+3) hybrid feature matrix (TEST set)
    feature_names: list of feature names aligned with X_hybrid columns
    y_true       : (n,) true binary labels — required for permutation fallback
    n_samples    : balls to subsample for SHAP computation (memory limit)
    seed         : random seed

    Returns
    -------
    shap_df : pd.DataFrame [Feature, Mean_|SHAP|, Rank, Is_KF]
              sorted by importance descending; KF states flagged with Is_KF=True
    """
    if not _SHAP_AVAILABLE:
        print("⚠️  SHAP not installed — falling back to permutation importance.")
        print("    To use SHAP: pip install shap")
        return _permutation_importance(
            model, X_hybrid, feature_names, y_true=y_true, seed=seed
        )

    rng = np.random.default_rng(seed)
    idx = rng.choice(len(X_hybrid), size=min(n_samples, len(X_hybrid)), replace=False)
    X_sub = X_hybrid[idx]

    explainer  = shap.TreeExplainer(model)
    shap_vals  = explainer.shap_values(X_sub)

# For binary classifiers, shap_values returns list of 2 arrays [class0, class1]
    if isinstance(shap_vals, list):
        shap_arr = shap_vals[1]   # class 1 = batting team wins
    else:
        shap_arr = shap_vals
        
    # NEW FIX: Handle newer SHAP versions that return a 3D array (samples, features, classes)
    if len(shap_arr.shape) == 3:
        shap_arr = shap_arr[:, :, 1] # Select class 1 (batting team wins)

    mean_abs   = np.abs(shap_arr).mean(axis=0)
    shap_df    = (
        pd.DataFrame({"Feature": feature_names, "Mean_|SHAP|": mean_abs})
        .sort_values("Mean_|SHAP|", ascending=False)
        .reset_index(drop=True)
    )
    shap_df["Rank"]   = shap_df.index + 1
    shap_df["Is_KF"]  = shap_df["Feature"].isin(KF_STATE_COLS)

    print("\n── Top 15 Features by SHAP Importance ─────────────────────")
    print(shap_df.head(15)[["Rank", "Feature", "Mean_|SHAP|", "Is_KF"]].to_string(index=False))
    print("────────────────────────────────────────────────────────────\n")

    # Rank of KF states specifically
    kf_ranks = shap_df[shap_df["Is_KF"]][["Feature", "Rank", "Mean_|SHAP|"]]
    print("KF state ranks:")
    print(kf_ranks.to_string(index=False))

    return shap_df


def _permutation_importance(
    model,
    X: np.ndarray,
    feature_names: List[str],
    y_true: Optional[np.ndarray] = None,
    n_repeats: int = 10,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Fallback permutation importance when SHAP is unavailable.

    Uses sklearn permutation_importance with neg_brier_score scoring.
    y_true must be provided; if None, the function raises a clear error.
    """
    from sklearn.inspection import permutation_importance as sk_pi
    from sklearn.metrics import brier_score_loss

    if y_true is None:
        raise ValueError(
            "y_true must be supplied to _permutation_importance when SHAP "
            "is unavailable. Pass y_true=test_y when calling compute_shap_importance()."
        )

    # Subclass wrapper so sklearn can call .predict_proba cleanly
    def _brier_scorer(estimator, X_s, y_s):
        proba = estimator.predict_proba(X_s)[:, 1]
        return -brier_score_loss(y_s, proba)   # negative because higher = better

    result = sk_pi(
        model, X, y_true,
        scoring=_brier_scorer,
        n_repeats=n_repeats,
        random_state=seed,
        n_jobs=-1,
    )

    df = pd.DataFrame({
        "Feature"    : feature_names,
        "Mean_|SHAP|": result.importances_mean,   # mean decrease in neg-brier = mean increase in brier loss
        "Std"        : result.importances_std,
        "Is_KF"      : [f in KF_STATE_COLS for f in feature_names],
    })
    df = df.sort_values("Mean_|SHAP|", ascending=False).reset_index(drop=True)
    df["Rank"] = df.index + 1
    return df


# ---------------------------------------------------------------------------
# H3 — Extended Kalman Filter (EKF)
# ---------------------------------------------------------------------------

class EKFWinProbability:
    """
    Extended Kalman Filter (EKF) with a nonlinear win-probability
    observation function.

    Nonlinear observation function (H3):
        h(x_t) = σ(H_lin · x_t)   where σ = element-wise sigmoid

    This captures the nonlinear saturation effect: once a team is very
    dominant (large state values), further improvement has diminishing
    effect on win probability.

    Jacobian (used in EKF linearisation):
        H̃_t = diag(σ'(H_lin · x_t)) · H_lin
        where σ'(z) = σ(z) · (1 − σ(z))

    Transition model remains linear (same as standard KF).
    """

    def __init__(self, n_states: int = KF_N_STATES) -> None:
        self.n_states = n_states
        self.A = None
        self.B = None
        self.H_lin = None     # linear part of the observation map
        self.Q = None
        self.R = None

    def initialize_parameters(
        self, n_controls: int, n_observations: int, kf_ref=None
    ) -> None:
        """
        Initialise from a trained standard KF (if provided) for consistency.
        """
        from config import KF_ALPHA_S, KF_ALPHA_M, KF_ALPHA_P, KF_Q_DIAG, KF_R_INIT_SCALE
        if kf_ref is not None:
            self.A     = kf_ref.A.copy()
            self.B     = kf_ref.B.copy()
            self.H_lin = kf_ref.H.copy()
            self.Q     = kf_ref.Q.copy()
            self.R     = kf_ref.R.copy()
        else:
            self.A     = np.diag([KF_ALPHA_S, KF_ALPHA_M, KF_ALPHA_P])
            self.B     = np.zeros((n_states, n_controls))
            self.H_lin = np.full((n_observations, n_states), 0.1)
            self.Q     = np.diag(KF_Q_DIAG)
            self.R     = np.eye(n_observations) * KF_R_INIT_SCALE

    @staticmethod
    def _sigmoid(z: np.ndarray) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-np.clip(z, -10, 10)))

    def _h(self, x: np.ndarray) -> np.ndarray:
        """Nonlinear observation function: h(x) = σ(H_lin · x)"""
        return self._sigmoid(self.H_lin @ x)

    def _H_jacobian(self, x: np.ndarray) -> np.ndarray:
        """Jacobian of h at x: H̃ = diag(σ'(H_lin·x)) · H_lin"""
        z      = self.H_lin @ x
        sigma  = self._sigmoid(z)
        d_sigma = sigma * (1.0 - sigma)              # σ'(z), shape (m,)
        return np.diag(d_sigma) @ self.H_lin          # (m, 3)

    def predict(self, x, P, u):
        x_pred = self.A @ x + self.B @ u
        P_pred = self.A @ P @ self.A.T + self.Q
        return x_pred, P_pred

    def update(self, x_pred, P_pred, y):
        H_jac     = self._H_jacobian(x_pred)
        innovation = y - self._h(x_pred)
        S          = H_jac @ P_pred @ H_jac.T + self.R
        S_inv      = np.linalg.inv(S + np.eye(S.shape[0]) * 1e-6)
        K          = P_pred @ H_jac.T @ S_inv
        x_updated  = x_pred + K @ innovation
        IKH        = np.eye(self.n_states) - K @ H_jac
        P_updated  = IKH @ P_pred @ IKH.T + K @ self.R @ K.T
        return x_updated, P_updated

    def filter_match(self, observations, controls, x0, P0):
        T = len(observations)
        states = np.zeros((T, self.n_states))
        x, P   = x0.copy(), P0.copy()
        for t in range(T):
            x, P = self.predict(x, P, controls[t])
            x, P = self.update(x, P, observations[t])
            if not np.isfinite(x).all():
                x = np.zeros(self.n_states)
            if not np.isfinite(P).all():
                P = np.eye(self.n_states)
            states[t] = x
        return states

    def initialize_state(self, innings_num, target=None, venue_avg=170.0):
        from config import KF_VENUE_AVG_DEFAULT
        venue_avg = venue_avg or KF_VENUE_AVG_DEFAULT
        if innings_num == 1:
            x0 = np.array([(venue_avg - 150.0) / 30.0, 0.0, 0.0])
        else:
            norm_target = ((target or venue_avg) - 150.0) / 30.0
            pressure    = (((target or venue_avg) / 20.0) - 8.5) / 2.0
            x0 = np.array([-norm_target * 0.5, 0.0, np.clip(pressure, -3, 3)])
        return x0, np.eye(self.n_states)


# ---------------------------------------------------------------------------
# H3 — Unscented Kalman Filter (UKF)
# ---------------------------------------------------------------------------

class UKFWinProbability:
    """
    Unscented Kalman Filter (UKF) using the Merwe scaled sigma-point method.

    The UKF propagates a set of 2n+1 deterministically chosen sigma points
    through the nonlinear h(·) without computing a Jacobian, yielding a
    more accurate covariance estimate than the EKF linearisation.

    Nonlinear observation function:
        h(x_t) = σ(H_lin · x_t)   (same as EKF)

    UKF parameters:
        α  = 1e-3  (spread of sigma points; small → near mean)
        β  = 2     (optimal for Gaussian distributions)
        κ  = 0     (secondary scaling parameter)
        λ  = α²(n + κ) − n
    """

    def __init__(self, n_states: int = KF_N_STATES) -> None:
        self.n_states = n_states
        self.A = None
        self.B = None
        self.H_lin = None
        self.Q = None
        self.R = None

        # UKF hyper-parameters (Merwe scaled sigma-points)
        self.alpha = 1e-3
        self.beta  = 2.0
        self.kappa = 0.0
        n          = n_states
        lam        = self.alpha ** 2 * (n + self.kappa) - n
        self._lam  = lam

        # Mean weights
        self.Wm = np.full(2 * n + 1, 1.0 / (2 * (n + lam)))
        self.Wm[0] = lam / (n + lam)

        # Covariance weights
        self.Wc = self.Wm.copy()
        self.Wc[0] += (1.0 - self.alpha ** 2 + self.beta)

    def initialize_parameters(self, n_controls, n_observations, kf_ref=None):
        from config import KF_ALPHA_S, KF_ALPHA_M, KF_ALPHA_P, KF_Q_DIAG, KF_R_INIT_SCALE
        if kf_ref is not None:
            self.A     = kf_ref.A.copy()
            self.B     = kf_ref.B.copy()
            self.H_lin = kf_ref.H.copy()
            self.Q     = kf_ref.Q.copy()
            self.R     = kf_ref.R.copy()
        else:
            self.A     = np.diag([KF_ALPHA_S, KF_ALPHA_M, KF_ALPHA_P])
            self.B     = np.zeros((self.n_states, n_controls))
            self.H_lin = np.full((n_observations, self.n_states), 0.1)
            self.Q     = np.diag(KF_Q_DIAG)
            self.R     = np.eye(n_observations) * KF_R_INIT_SCALE

    @staticmethod
    def _sigmoid(z):
        return 1.0 / (1.0 + np.exp(-np.clip(z, -10, 10)))

    def _h(self, x):
        return self._sigmoid(self.H_lin @ x)

    def _sigma_points(self, x, P):
        n   = self.n_states
        lam = self._lam
        try:
            S = np.linalg.cholesky((n + lam) * P)
        except np.linalg.LinAlgError:
            P_reg = P + np.eye(n) * 1e-6
            S     = np.linalg.cholesky((n + lam) * P_reg)

        sigma = np.zeros((2 * n + 1, n))
        sigma[0]     = x
        sigma[1:n+1] = x + S.T
        sigma[n+1:]  = x - S.T
        return sigma

    def filter_match(self, observations, controls, x0, P0):
        T = len(observations)
        states = np.zeros((T, self.n_states))
        x, P   = x0.copy(), P0.copy()
        n      = self.n_states

        for t in range(T):
            # Predict (linear transition — sigma points not needed)
            x_pred = self.A @ x + self.B @ controls[t]
            P_pred = self.A @ P @ self.A.T + self.Q

            # Update via sigma points through nonlinear h(·)
            sigma   = self._sigma_points(x_pred, P_pred)
            y_sigma = np.array([self._h(sp) for sp in sigma])  # (2n+1, m)

            y_pred  = (self.Wm[:, None] * y_sigma).sum(axis=0)

            Pyy = self.R.copy()
            Pxy = np.zeros((n, len(y_pred)))
            for i, (wc, sp, ys) in enumerate(zip(self.Wc, sigma, y_sigma)):
                dy   = ys - y_pred
                dx   = sp - x_pred
                Pyy += wc * np.outer(dy, dy)
                Pxy += wc * np.outer(dx, dy)

            Pyy_inv = np.linalg.inv(Pyy + np.eye(len(y_pred)) * 1e-6)
            K        = Pxy @ Pyy_inv

            y_obs    = observations[t]
            x        = x_pred + K @ (y_obs - y_pred)
            P        = P_pred - K @ Pyy @ K.T

            if not np.isfinite(x).all():
                x = np.zeros(n)
            if not np.isfinite(P).all():
                P = np.eye(n)

            states[t] = x

        return states

    def initialize_state(self, innings_num, target=None, venue_avg=170.0):
        from config import KF_VENUE_AVG_DEFAULT
        venue_avg = venue_avg or KF_VENUE_AVG_DEFAULT
        if innings_num == 1:
            x0 = np.array([(venue_avg - 150.0) / 30.0, 0.0, 0.0])
        else:
            norm_target = ((target or venue_avg) - 150.0) / 30.0
            pressure    = (((target or venue_avg) / 20.0) - 8.5) / 2.0
            x0 = np.array([-norm_target * 0.5, 0.0, np.clip(pressure, -3, 3)])
        return x0, np.eye(self.n_states)


# ---------------------------------------------------------------------------
# Serialisation
# ---------------------------------------------------------------------------

def save_hybrid_models(models: Dict) -> None:
    for name, obj in models.items():
        if obj is None:
            continue
        path = MODELS_DIR / f"hybrid_{name}.pkl"
        with open(path, "wb") as fh:
            pickle.dump(obj, fh)
        print(f"  Saved: {path}")
