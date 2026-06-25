"""
kalman_filter.py
================
Linear Kalman Filter with space-efficient EM parameter estimation and
second-stage logistic win-probability calibration.

State-space model (paper Section 4.1)
--------------------------------------
State vector:  x_t = [Ŝ_t, M̂_t, P̂_t]'   (Strength, Momentum, Pressure)

State equation:
    x_t = A · x_{t-1} + B · u_t + w_t,     w_t ~ N(0, Q)

Observation equation:
    y_t = H · x_t + v_t,                    v_t ~ N(0, R)

Win-probability (second-stage logistic, K6):
    z_t = [x_t', context_t']
    p_t = σ(β' · z_t)
    where β is fit by logistic regression on training outcomes.

Public API
----------
    KalmanFilterWinProbability          — main class
    prepare_kalman_data(df, scaler_obs, scaler_ctrl, fit_scalers)
    space_efficient_em(kf, data_dict, max_iters, tol) -> kf, ll_history
    report_kf_parameters(kf)           -> dict of DataFrames  (K1–K7)
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import linalg
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from tqdm.auto import tqdm

from config import (
    KF_N_STATES,
    KF_ALPHA_S,
    KF_ALPHA_M,
    KF_ALPHA_P,
    KF_Q_DIAG,
    KF_R_INIT_SCALE,
    KF_EM_MAX_ITERS,
    KF_EM_H_DAMPING,
    KF_EM_R_DAMPING,
    KF_EM_TOL,
    KF_EM_REG,
    KF_VENUE_AVG_DEFAULT,
    KF_CALIB_COLS,
    MODELS_DIR,
    OBS_FEATURES,
    CTRL_FEATURES,
)


# ---------------------------------------------------------------------------
# Core Kalman Filter class
# ---------------------------------------------------------------------------

class KalmanFilterWinProbability:
    """
    Linear Kalman Filter tracking three latent cricket states:
        Ŝ_t  — Batting Strength   (overall run-scoring ability)
        M̂_t  — Scoring Momentum  (recent run-rate trajectory)
        P̂_t  — Match Pressure    (run-requirement vs. resources)

    Parameters are:
        A  : (3×3) state-transition matrix         [K1]
        B  : (3×k) control-input matrix            [K2]
        H  : (m×3) observation/emission matrix     [K3]
        Q  : (3×3) process-noise covariance        [K4]
        R  : (m×m) measurement-noise covariance    [K4]

    Win probability is computed by a second-stage logistic regression
    trained on training-set outcomes (K6).
    """

    def __init__(self) -> None:
        self.n_states = KF_N_STATES   # 3

        # Matrices — populated by initialize_parameters()
        self.A: Optional[np.ndarray] = None
        self.B: Optional[np.ndarray] = None
        self.H: Optional[np.ndarray] = None
        self.Q: Optional[np.ndarray] = None
        self.R: Optional[np.ndarray] = None

        # Second-stage calibration model (K6)
        self.calibration_lr: Optional[LogisticRegression] = None
        self.calib_scaler: Optional[StandardScaler] = None

    # -----------------------------------------------------------------------
    # Initialisation  (K1–K4, K7)
    # -----------------------------------------------------------------------

    def initialize_parameters(
        self,
        n_controls: int,
        n_observations: int,
    ) -> None:
        """
        Set interpretable initial values for all KF matrices.

        A  — diagonal with persistence parameters (αS, αM, αP).
             Off-diagonal terms set to zero (tested and not significant, K1).
        B  — near-zero random init (domain physics priors are encoded
             during EM; we do not pre-impose hard sign constraints).
        H  — uniform small positive init (EM refines this; K3).
        Q  — diagonal with per-state variance from config (K4).
        R  — scaled identity; EM updates this (K4).
        """
        # K1: Transition matrix — diagonal persistence
        self.A = np.diag([KF_ALPHA_S, KF_ALPHA_M, KF_ALPHA_P])

        # K2: Control matrix — (3 × k), small random init to break symmetry
        rng = np.random.default_rng(seed=42)
        self.B = rng.normal(0, 0.01, size=(self.n_states, n_controls))

        # K3: Observation matrix — (m × 3), uniform small positive
        self.H = np.full((n_observations, self.n_states), 0.1)

        # K4: Process noise — diagonal
        self.Q = np.diag(KF_Q_DIAG)

        # K4: Measurement noise — scaled identity (refined by EM)
        self.R = np.eye(n_observations) * KF_R_INIT_SCALE

    # -----------------------------------------------------------------------
    # Kalman predict & update
    # -----------------------------------------------------------------------

    def predict(
        self,
        x: np.ndarray,
        P: np.ndarray,
        u: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Kalman prediction step.

        x_pred = A·x + B·u
        P_pred = A·P·A' + Q
        """
        x_pred = self.A @ x + self.B @ u
        P_pred = self.A @ P @ self.A.T + self.Q
        return x_pred, P_pred

    def update(
        self,
        x_pred: np.ndarray,
        P_pred: np.ndarray,
        y: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Kalman update step using the Joseph stabilised form.

        Innovation:  v   = y − H·x_pred
        Innovation cov:  S = H·P·H' + R
        Kalman gain:     K = P·H'·S^{-1}
        Updated state:   x = x_pred + K·ν
        Updated cov:     P = (I − KH)·P·(I − KH)' + K·R·K'   [Joseph form]

        The Joseph form guarantees positive-definiteness even with
        finite-precision arithmetic.
        """
        innovation = y - self.H @ x_pred
        S          = self.H @ P_pred @ self.H.T + self.R

        # Regularise S to prevent singular matrix
        S_reg = S + np.eye(S.shape[0]) * 1e-6
        try:
            S_inv = linalg.inv(S_reg)
        except linalg.LinAlgError:
            S_inv = np.linalg.pinv(S_reg)

        K = P_pred @ self.H.T @ S_inv
        x_updated = x_pred + K @ innovation

        # Joseph form
        I   = np.eye(self.n_states)
        IKH = I - K @ self.H
        P_updated = IKH @ P_pred @ IKH.T + K @ self.R @ K.T

        return x_updated, P_updated

    # -----------------------------------------------------------------------
    # Per-match filtering
    # -----------------------------------------------------------------------

    def filter_match(
        self,
        observations: np.ndarray,
        controls: np.ndarray,
        x0: np.ndarray,
        P0: np.ndarray,
    ) -> np.ndarray:
        """
        Run the Kalman filter forward pass for a single match innings.

        Parameters
        ----------
        observations : (T, m)  — scaled measurement vectors
        controls     : (T, k)  — scaled control vectors
        x0           : (3,)    — initial state mean
        P0           : (3, 3)  — initial state covariance

        Returns
        -------
        filtered_states : (T, 3)  — filtered state estimates x_{t|t}
        """
        T = len(observations)
        filtered_states = np.zeros((T, self.n_states))
        x, P = x0.copy(), P0.copy()

        for t in range(T):
            x_pred, P_pred = self.predict(x, P, controls[t])
            x, P           = self.update(x_pred, P_pred, observations[t])

            # Numerical safety guards
            if not np.isfinite(x).all():
                x = np.zeros(self.n_states)
            if not np.isfinite(P).all():
                P = np.eye(self.n_states)

            filtered_states[t] = x

        return filtered_states

    # -----------------------------------------------------------------------
    # State initialisation (K7)
    # -----------------------------------------------------------------------

    def initialize_state(
        self,
        innings_num: int,
        target: Optional[float] = None,
        venue_avg: float = KF_VENUE_AVG_DEFAULT,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Domain-informed state initialisation (K7).

        First innings (batting):
            Ŝ_0 = (venue_avg − 150) / 30    — relative to a neutral venue
            M̂_0 = 0  (no momentum yet)
            P̂_0 = 0  (no pressure yet)
            P0   = I  (high uncertainty)

        Second innings (chasing):
            Ŝ_0 = −norm_target × 0.5       — harder target → lower initial strength
            M̂_0 = 0
            P̂_0 = (target/20 − 8.5) / 2  — normalised required run-rate
            P0   = I
        """
        if innings_num == 1:
            x0 = np.array([
                (venue_avg - 150.0) / 30.0,
                0.0,
                0.0,
            ])
        else:
            # Chasing: Initial Pressure is proportional to Target vs Venue Average
            # Target > Venue Avg => High starting pressure (+ve P̂)
            norm_target = (target - venue_avg) / 20.0 if target else 0.0
            x0 = np.array([
                -0.5 * norm_target,  # Batting strength starts lower if target is huge
                0.0,
                np.clip(norm_target, -2.0, 4.0), # Initial Pressure
            ])

        P0 = np.eye(self.n_states)
        return x0, P0

    # -----------------------------------------------------------------------
    # Second-stage win-probability calibration (K6)
    # -----------------------------------------------------------------------

    def fit_calibration(
        self,
        filtered_states_df: pd.DataFrame,
        y_true: np.ndarray,
        sample_weight: Optional[np.ndarray] = None,
    ) -> None:
        """
        Fit a logistic regression mapping filtered states + context → win prob.

        Inputs to logistic regression (z_t):
            kf_strength, kf_momentum, kf_pressure,
            balls_remaining, wickets_in_hand, innings_number

        The scaler is fit on training data only; stored for inference.
        """
        cols = [c for c in KF_CALIB_COLS if c in filtered_states_df.columns]
        X    = filtered_states_df[cols].values.astype(np.float64)
        X    = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

        self.calib_scaler = StandardScaler()
        X_scaled = self.calib_scaler.fit_transform(X)

        self.calibration_lr = LogisticRegression(
            penalty="l2", C=1.0, solver="lbfgs", max_iter=1000, random_state=42
        )
        self.calibration_lr.fit(X_scaled, y_true, sample_weight=sample_weight)
        print(
            f"  Calibration LR fitted. "
            f"Intercept: {self.calibration_lr.intercept_[0]:.4f}  |  "
            f"Coefs: {self.calibration_lr.coef_[0].round(4)}"
        )

    def predict_win_probability(
        self,
        filtered_states_df: pd.DataFrame,
    ) -> np.ndarray:
        """
        Apply fitted second-stage logistic regression to get win probabilities.

        Returns
        -------
        np.ndarray of shape (n,) with values in [0, 1]
        """
        if self.calibration_lr is None or self.calib_scaler is None:
            raise RuntimeError(
                "Call fit_calibration() before predict_win_probability()."
            )
        cols   = [c for c in KF_CALIB_COLS if c in filtered_states_df.columns]
        X      = filtered_states_df[cols].values.astype(np.float64)
        X      = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        X_sc   = self.calib_scaler.transform(X)
        return self.calibration_lr.predict_proba(X_sc)[:, 1]

    # -----------------------------------------------------------------------
    # Serialisation
    # -----------------------------------------------------------------------

    def save(self, path: Path = MODELS_DIR / "kalman_filter.pkl") -> None:
        with open(path, "wb") as fh:
            pickle.dump(self, fh)
        print(f"  KF saved to {path}")

    @classmethod
    def load(cls, path: Path = MODELS_DIR / "kalman_filter.pkl") -> "KalmanFilterWinProbability":
        with open(path, "rb") as fh:
            return pickle.load(fh)


# ---------------------------------------------------------------------------
# Data preparation for KF  (scalers fitted on training set only)
# ---------------------------------------------------------------------------

def prepare_kalman_data(
    df: pd.DataFrame,
    scaler_obs: Optional[StandardScaler] = None,
    scaler_ctrl: Optional[StandardScaler] = None,
    fit_scalers: bool = True,
) -> Tuple[Dict, StandardScaler, StandardScaler]:
    """
    Prepare per-innings observation and control arrays for the KF.

    IMPORTANT: Pass fit_scalers=True only for TRAINING data.
    For validation/test, pass the fitted scalers and fit_scalers=False.

    Parameters
    ----------
    df           : feature-engineered ball DataFrame (one or more seasons)
    scaler_obs   : pre-fitted StandardScaler for observations (or None)
    scaler_ctrl  : pre-fitted StandardScaler for controls (or None)
    fit_scalers  : if True, fit new scalers on this data

    Returns
    -------
    data_dict  : { (match_id, innings_number) : {'observations', 'controls',
                                                  'innings', 'target',
                                                  'venue_avg', 'indices'} }
    scaler_obs : fitted StandardScaler for observation features
    scaler_ctrl: fitted StandardScaler for control features
    """
    obs_cols  = [c for c in OBS_FEATURES  if c in df.columns]
    ctrl_cols = [c for c in CTRL_FEATURES if c in df.columns]

    # Fill before scaling
    obs_vals  = df[obs_cols].fillna(0.0).replace([np.inf, -np.inf], 0.0).values.astype(np.float64)
    ctrl_vals = df[ctrl_cols].fillna(0.0).replace([np.inf, -np.inf], 0.0).values.astype(np.float64)

    if fit_scalers:
        scaler_obs  = StandardScaler().fit(obs_vals)
        scaler_ctrl = StandardScaler().fit(ctrl_vals)

    obs_scaled  = scaler_obs.transform(obs_vals)
    ctrl_scaled = scaler_ctrl.transform(ctrl_vals)

    # Sort df to match matrix row order
    df_sorted = df.sort_values(["match_id", "innings_number", "over", "ball"])

    data_dict: Dict = {}
    for (mid, inn), group in df_sorted.groupby(["match_id", "innings_number"]):
        idx       = group.index
        row_pos   = [df_sorted.index.get_loc(i) for i in idx]

        target    = group["target"].iloc[0] if "target" in group else None
        venue_avg = (
            group["venue_avg_first_innings"].iloc[0]
            if "venue_avg_first_innings" in group
            else KF_VENUE_AVG_DEFAULT
        )
        data_dict[(mid, inn)] = {
            "observations": obs_scaled[row_pos],
            "controls"    : ctrl_scaled[row_pos],
            "innings"     : inn,
            "target"      : target,
            "venue_avg"   : venue_avg,
            "indices"     : idx,           # original df indices for assignment
        }

    return data_dict, scaler_obs, scaler_ctrl


# ---------------------------------------------------------------------------
# EM algorithm  (K5)
# ---------------------------------------------------------------------------

def space_efficient_em(
    kf: KalmanFilterWinProbability,
    data_dict: Dict,
    max_iters: int = KF_EM_MAX_ITERS,
    tol: float = KF_EM_TOL,
) -> Tuple[KalmanFilterWinProbability, List[float]]:
    """
    Space-efficient EM to refine H (observation matrix) and R (measurement
    noise) of the Kalman Filter.

    We do NOT optimise A or B via EM — they remain domain-initialised.
    Optimising all matrices jointly with limited data leads to instability
    (tested; results in K5 discussion).

    E-step : run the Kalman filter forward pass, accumulate sufficient
             statistics (ΣYY, ΣYX, ΣXX, ΣRR) in a single pass per iteration.
    M-step : update H = ΣYX · (ΣXX + λI)^{-1}
                        (damped: kf.H ← (1-α)·kf.H + α·H_new)
                    R = diag(ΣRR / T)
                        (damped: kf.R ← (1-β)·kf.R + β·R_new)

    Convergence criterion: |ΔLL| / |LL| < tol
    (approximate Gaussian log-likelihood computed from residuals)

    Parameters
    ----------
    kf          : initialised KalmanFilterWinProbability
    data_dict   : output of prepare_kalman_data (TRAINING data only)
    max_iters   : maximum EM iterations
    tol         : relative log-likelihood change for convergence

    Returns
    -------
    kf            : updated KalmanFilterWinProbability
    ll_history    : list of approximate log-likelihood values per iteration
    """
    print(f"\n{'='*62}")
    print(f"  EM TRAINING  (max_iters={max_iters}, tol={tol:.0e})")
    print(f"{'='*62}")

    n_obs    = kf.H.shape[0]
    ll_history: List[float] = []
    prev_ll  = -np.inf

    for iteration in range(1, max_iters + 1):

        # Sufficient-statistics accumulators
        S_YY = np.zeros((n_obs, n_obs))           # Σ Y·Y'
        S_YX = np.zeros((n_obs, kf.n_states))        # Σ Y·X'
        S_XX = np.zeros((kf.n_states, kf.n_states))  # Σ X·X'
        S_RR = np.zeros((n_obs, n_obs))           # Σ residual·residual'
        
        # New accumulators for joint A and B optimization
        # x_t = [A|B] [x_{t-1}, u_t]' + ε
        n_ctrl = kf.B.shape[1]
        dim_z  = kf.n_states + n_ctrl
        S_XZ   = np.zeros((kf.n_states, dim_z))   # Σ x_t · [x_{t-1}, u_t]'
        S_ZZ   = np.zeros((dim_z, dim_z))         # Σ [x_{t-1}, u_t] · [x_{t-1}, u_t]'
        
        T_total = 0

        # Approximate log-likelihood accumulator (Gaussian residuals)
        ll_accum = 0.0

        for key, data in data_dict.items():
            obs  = data["observations"]   # (T, m)
            ctrl = data["controls"]       # (T, k)
            T    = len(obs)

            x0, P0 = kf.initialize_state(
                data["innings"], data["target"], data["venue_avg"]
            )
            states = kf.filter_match(obs, ctrl, x0, P0)  # (T, 3)

            # Accumulate for H and R
            S_YY += obs.T @ obs
            S_YX += obs.T @ states
            S_XX += states.T @ states

            # Accumulate for A and B (Jointly)
            if T > 1:
                x_curr = states[1:]      # (T-1, 3)
                x_prev = states[:-1]     # (T-1, 3)
                u_curr = ctrl[1:]        # (T-1, k)
                z_curr = np.hstack([x_prev, u_curr]) # (T-1, 3+k)
                
                S_XZ += x_curr.T @ z_curr
                S_ZZ += z_curr.T @ z_curr

            residuals = obs - (kf.H @ states.T).T    # (T, m)
            S_RR     += residuals.T @ residuals

            # Gaussian log-likelihood (diagonal R assumed)
            R_diag    = np.maximum(np.diag(kf.R), 1e-6)
            ll_accum += -0.5 * np.sum(residuals ** 2 / R_diag)
            ll_accum += -0.5 * T * np.sum(np.log(R_diag))
            T_total  += T

        # ------------------------------------------------------------------
        # M-step: update H
        # ------------------------------------------------------------------
        reg    = np.eye(kf.n_states) * KF_EM_REG
        H_new  = S_YX @ np.linalg.inv(S_XX + reg)
        kf.H   = (1.0 - KF_EM_H_DAMPING) * kf.H + KF_EM_H_DAMPING * H_new

        # ------------------------------------------------------------------
        # M-step: update A and B (Joint Optimization)
        # ------------------------------------------------------------------
        reg_z  = np.eye(dim_z) * KF_EM_REG
        AB_new = S_XZ @ np.linalg.inv(S_ZZ + reg_z)
        
        A_new  = AB_new[:, :kf.n_states]
        B_new  = AB_new[:, kf.n_states:]
        
        # Damping: persistence for A, standard damping for B
        kf.A   = (1.0 - 0.1) * kf.A + 0.1 * A_new
        kf.B   = (1.0 - KF_EM_H_DAMPING) * kf.B + KF_EM_H_DAMPING * B_new

        # ------------------------------------------------------------------
        # M-step: update R (diagonal only, for stability)
        # ------------------------------------------------------------------
        R_new  = np.diag(np.diag(S_RR / T_total))
        R_new  = np.maximum(R_new, np.eye(n_obs) * 1e-4)   # floor
        kf.R   = (1.0 - KF_EM_R_DAMPING) * kf.R + KF_EM_R_DAMPING * R_new

        # Normalise approximate LL
        ll_iter = ll_accum / T_total
        ll_history.append(ll_iter)

        delta_ll = abs(ll_iter - prev_ll) / (abs(prev_ll) + 1e-12)

        print(
            f"  Iter {iteration:3d}  |  "
            f"LL = {ll_iter:+.4f}  |  "
            f"ΔLL = {delta_ll:.2e}  |  "
            f"‖B‖ = {np.linalg.norm(kf.B):.4f}  |  "
            f"‖H‖ = {np.linalg.norm(kf.H):.4f}"
        )

        if iteration > 1 and delta_ll < tol:
            print(f"\n  ✅  Converged at iteration {iteration}.")
            break

        prev_ll = ll_iter

    else:
        print(f"\n  ⚠️  EM reached max_iters={max_iters} without convergence.")

    print(f"{'='*62}\n")
    return kf, ll_history


# ---------------------------------------------------------------------------
# Generate filtered state columns in DataFrame
# ---------------------------------------------------------------------------

def generate_kalman_features(
    kf: KalmanFilterWinProbability,
    data_dict: Dict,
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Run the trained KF over every match-innings in data_dict and assign the
    filtered states [Ŝ_t, M̂_t, P̂_t] back to the corresponding rows of df.

    Returns
    -------
    df : original DataFrame with three new columns:
         'kf_strength', 'kf_momentum', 'kf_pressure'
    """
    df = df.sort_values(["match_id", "innings_number", "over", "ball"]).copy()
    df["kf_strength"] = np.nan
    df["kf_momentum"] = np.nan
    df["kf_pressure"] = np.nan

    for (mid, inn), data in tqdm(data_dict.items(), desc="Filtering matches"):
        x0, P0 = kf.initialize_state(data["innings"], data["target"], data["venue_avg"])
        states  = kf.filter_match(data["observations"], data["controls"], x0, P0)

        idx    = data["indices"]
        # Guard: lengths must match
        n_rows = len(df.loc[df.index.isin(idx)])
        if n_rows != len(states):
            # Fallback: match by position within mask
            mask   = (df["match_id"] == mid) & (df["innings_number"] == inn)
            df.loc[mask, ["kf_strength", "kf_momentum", "kf_pressure"]] = states
        else:
            df.loc[idx, ["kf_strength", "kf_momentum", "kf_pressure"]] = states

    # Calculate non-linear interactions for the calibration layer
    df["kf_strength_wickets"] = df["kf_strength"] * df["wickets_in_hand"]
    df["kf_momentum_runs"]    = df["kf_momentum"] * df["runs_last_24_balls"]
    df["kf_pressure_rrr"]     = df["kf_pressure"] * df["required_run_rate"].fillna(0.0)

    print("✅  Kalman state columns and interaction terms assigned.")
    return df


# ---------------------------------------------------------------------------
# Parameter reporting  (K1–K7)
# ---------------------------------------------------------------------------

def report_kf_parameters(kf: KalmanFilterWinProbability) -> Dict[str, pd.DataFrame]:
    """
    Extract all KF matrices into labelled DataFrames for paper tables K1–K7.
    FIXED: Mapped exact config feature names to obs and ctrl arrays.
    """
    state_labels = ["Strength (Ŝ)", "Momentum (M̂)", "Pressure (P̂)"]
    
    # Use exact feature names defined in config.py
    obs_labels   = OBS_FEATURES[:kf.H.shape[0]]
    ctrl_labels  = CTRL_FEATURES[:kf.B.shape[1]]

    report = {}

    # K1: A matrix
    report["A (Transition)"] = pd.DataFrame(
        kf.A, index=state_labels, columns=state_labels
    ).round(6)

    # K2: B matrix
    report["B (Control)"] = pd.DataFrame(
        kf.B, index=state_labels, columns=ctrl_labels
    ).round(6)

    # K3: H matrix
    report["H (Observation)"] = pd.DataFrame(
        kf.H, index=obs_labels, columns=state_labels
    ).round(6)

    # K4: Q matrix
    report["Q (Process Noise)"] = pd.DataFrame(
        kf.Q, index=state_labels, columns=state_labels
    ).round(6)

    # K4: R matrix
    report["R (Measurement Noise)"] = pd.DataFrame(
        kf.R, index=obs_labels, columns=obs_labels
    ).round(6)

    # K6: Calibration logistic regression
    if kf.calibration_lr is not None:
        calib_cols = [c for c in KF_CALIB_COLS]
        coefs = kf.calibration_lr.coef_[0]
        report["Calibration (β)"] = pd.DataFrame({
            "Feature"    : calib_cols[:len(coefs)],
            "Coefficient": coefs.round(6),
        })

    return report