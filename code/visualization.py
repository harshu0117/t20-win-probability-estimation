"""
visualization.py
================
Publication-quality figure generation (paper Sections 5.2, 5.5–5.6, F1–F6).

All figures are saved at 300 DPI in both PDF (vector) and PNG (raster)
formats, using the IEEE/Springer journal style defined in config.py.

Tasks covered
-------------
F1 — Calibration curves (reliability diagrams) — all 4 models on one plot
F2 — Latent state trajectory plots — 3 case-study matches (4-panel each)
F3 — Average latent trajectories: winning vs. losing chases (3-panel)
F4 — KF vs RF prediction scatter with phase colour-coding + marginals
F5 — Prediction volatility box plots
F6 — Season-by-season Brier score line chart

Public API
----------
    plot_calibration_curves(df)                        -> fig  (F1)
    plot_case_study_match(df, match_id, description)   -> fig  (F2)
    select_case_study_matches(df)                      -> list of match_ids (F2)
    plot_average_latent_trajectories(df)               -> fig  (F3)
    plot_kf_vs_rf_scatter(df)                          -> fig  (F4)
    plot_volatility_boxplot(per_match_df)              -> fig  (F5)
    plot_season_brier(season_df)                       -> fig  (F6)
    plot_em_convergence(ll_history)                    -> fig  (EM diagnostic)
"""

from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
from scipy.stats import sem

from config import (
    MODEL_COLORS,
    PHASE_COLORS,
    FIG_WIDTH_DOUBLE,
    FIG_WIDTH_FULL,
    FIG_WIDTH_SINGLE,
    CALIBRATION_N_BINS,
    TARGET_COL,
    set_publication_style,
    save_figure,
)

# Apply style globally on import
set_publication_style()

# Canonical model display list (ordered for legend consistency)
_MODEL_ORDER = [
    "Kalman Filter",
    "Random Forest",
    "Gradient Boosting",
    "Logistic Regression",
]
_MODEL_COLS = {
    "Kalman Filter"       : "kf_win_prob",
    "Logistic Regression" : "lr_win_prob",
    "Random Forest"       : "rf_win_prob",
    "Gradient Boosting"   : "gb_win_prob",
}
_KF_STATE_COLS = ["kf_strength", "kf_momentum", "kf_pressure"]
_KF_STATE_LABELS = [
    r"Batting Strength ($\hat{S}_t$)",
    r"Scoring Momentum ($\hat{M}_t$)",
    r"Match Pressure ($\hat{P}_t$)",
]
_KF_STATE_COLORS = ["#2E86AB", "#FD7F20", "#DC3545"]


def _available_models(df: pd.DataFrame) -> List[Tuple[str, str]]:
    """Return list of (model_name, col_name) pairs available in df."""
    return [
        (n, c) for n in _MODEL_ORDER
        for c in [_MODEL_COLS.get(n, "")]
        if c and c in df.columns
    ]


# ---------------------------------------------------------------------------
# F1 — Calibration curves (reliability diagrams)
# ---------------------------------------------------------------------------

def plot_calibration_curves(df: pd.DataFrame) -> plt.Figure:
    """
    Reliability diagram: all four models on a single plot (F1).

    Each model's predicted probabilities are binned into CALIBRATION_N_BINS
    equal-width bins; within each bin the observed win frequency is plotted.

    Returns
    -------
    matplotlib Figure
    """
    from sklearn.calibration import calibration_curve

    fig, ax = plt.subplots(figsize=(FIG_WIDTH_DOUBLE, FIG_WIDTH_DOUBLE * 0.7))

    # Perfect calibration reference
    ax.plot([0, 1], [0, 1], "k--", linewidth=1.2, alpha=0.6, label="Perfect calibration", zorder=0)

    y_true = df[TARGET_COL].values

    for name, col in _available_models(df):
        y_pred = df[col].values
        prob_true, prob_pred = calibration_curve(
            y_true, y_pred, n_bins=CALIBRATION_N_BINS, strategy="uniform"
        )
        color = MODEL_COLORS[name]
        ax.plot(
            prob_pred, prob_true,
            "o-",
            color=color,
            linewidth=1.8,
            markersize=5,
            label=name,
            zorder=3,
        )

    ax.set_xlabel("Mean Predicted Probability", labelpad=6)
    ax.set_ylabel("Observed Win Frequency",     labelpad=6)
    ax.set_title("Reliability Diagram (Calibration Curves)", pad=10)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    ax.legend(loc="upper left", framealpha=0.9)
    ax.set_aspect("equal")

    # Shaded ±0.05 tolerance band around perfect calibration
    ax.fill_between([0, 1], [-0.05, 0.95], [0.05, 1.05],
                    color="gray", alpha=0.08, label="_nolegend_")

    fig.tight_layout()
    save_figure(fig, "F1_calibration_curves")
    return fig


# ---------------------------------------------------------------------------
# F2 — Case-study match trajectories
# ---------------------------------------------------------------------------

def select_case_study_matches(df: pd.DataFrame) -> List[Tuple[str, str]]:
    """
    Automatically select three representative second-innings matches (F2):
      1. Dramatic momentum swing  — highest std-dev of kf_win_prob
      2. Steady accumulation      — lowest std-dev of kf_win_prob (and correct prediction)
      3. Last-ball thriller       — win prob closest to 0.5 at the final delivery

    Returns
    -------
    List of (match_id, description) tuples
    """
    kf_col = _MODEL_COLS["Kalman Filter"]
    if kf_col not in df.columns:
        raise ValueError("kf_win_prob column not found. Run KF before selecting matches.")

    inn2 = df[(df["innings_number"] == 2) & (df[kf_col].notna())].copy()

    match_stats = (
        inn2.groupby("match_id")[kf_col]
        .agg(["std", "last"])
        .reset_index()
        .rename(columns={"std": "wp_std", "last": "wp_final"})
    )
    match_stats["final_uncertainty"] = (match_stats["wp_final"] - 0.5).abs()

    # 1. Dramatic swing
    m1 = match_stats.sort_values("wp_std", ascending=False).iloc[0]["match_id"]

    # 2. Steady / predictable  (low std + outcome already decided)
    predictable = match_stats[
        (match_stats["wp_final"] > 0.85) | (match_stats["wp_final"] < 0.15)
    ]
    m2 = (predictable.sort_values("wp_std").iloc[0]["match_id"]
          if len(predictable) > 0
          else match_stats.sort_values("wp_std").iloc[0]["match_id"])

    # 3. Last-ball thriller
    m3 = match_stats.sort_values("final_uncertainty").iloc[0]["match_id"]

    selected = [
        (m1, "Dramatic momentum swing (batting collapse during chase)"),
        (m2, "Steady accumulation — predictable outcome"),
        (m3, "Last-ball thriller — outcome decided in final over"),
    ]
    # Ensure distinct matches
    seen = set()
    unique = []
    for mid, desc in selected:
        if mid not in seen:
            unique.append((mid, desc))
            seen.add(mid)
    return unique[:3]


def plot_case_study_match(
    df: pd.DataFrame,
    match_id: str,
    description: str = "",
    fig_stem: Optional[str] = None,
) -> plt.Figure:
    """
    4-panel figure for a single case-study match (F2):
      Panel 1 : Win probability (KF + RF on same axes)
      Panel 2 : Batting Strength (Ŝ_t)
      Panel 3 : Scoring Momentum (M̂_t)
      Panel 4 : Match Pressure (P̂_t)

    Annotations: wickets (vertical dashed lines), sixes (star markers),
    powerplay and death-over shading.
    """
    match_df = (
        df[(df["match_id"] == match_id) & (df["innings_number"] == 2)]
        .sort_values(["over", "ball"])
        .reset_index(drop=True)
    )
    if len(match_df) == 0:
        # Fall back to any innings
        match_df = (
            df[df["match_id"] == match_id]
            .sort_values(["innings_number", "over", "ball"])
            .reset_index(drop=True)
        )

    if len(match_df) == 0:
        raise ValueError(f"No data found for match_id={match_id}")

    n_balls   = len(match_df)
    x         = np.arange(n_balls)
    ball_tick = np.arange(0, n_balls, 6)
    over_tick = [f"Ov {i//6+1}" for i in ball_tick]

    fig = plt.figure(figsize=(FIG_WIDTH_FULL, 10))
    gs  = gridspec.GridSpec(4, 1, figure=fig, hspace=0.08)
    axes = [fig.add_subplot(gs[i]) for i in range(4)]

    # ── Phase shading helper ──
    def _shade_phases(ax):
        pp = match_df.index[match_df["phase_powerplay"] == 1]
        dt = match_df.index[match_df["phase_death"]    == 1]
        if len(pp) > 0:
            ax.axvspan(pp[0], pp[-1], alpha=0.06, color=PHASE_COLORS["powerplay"], zorder=0)
        if len(dt) > 0:
            ax.axvspan(dt[0], dt[-1], alpha=0.06, color=PHASE_COLORS["death"],     zorder=0)

    # ── Wicket lines helper ──
    def _draw_wickets(ax, ymin=0, ymax=1):
        wkt_idx = match_df.index[match_df["is_wicket"] == True].tolist()
        for wi in wkt_idx:
            ax.axvline(wi, color="crimson", linestyle="--", linewidth=0.8, alpha=0.7, zorder=2)

    # ── Six markers on panel 1 ──
    six_idx = match_df.index[match_df["batter_runs"] == 6].tolist()

    # Panel 1: Win probability
    ax0 = axes[0]
    kf_col = _MODEL_COLS["Kalman Filter"]
    rf_col = _MODEL_COLS["Random Forest"]

    if kf_col in match_df.columns:
        ax0.plot(x, match_df[kf_col], color=MODEL_COLORS["Kalman Filter"],
                 linewidth=2.0, label="Kalman Filter", zorder=4)
    if rf_col in match_df.columns:
        ax0.plot(x, match_df[rf_col], color=MODEL_COLORS["Random Forest"],
                 linewidth=1.4, linestyle="--", alpha=0.75, label="Random Forest", zorder=3)

    if six_idx:
        yvals = match_df.loc[six_idx, kf_col] if kf_col in match_df.columns else [0.5] * len(six_idx)
        ax0.scatter(six_idx, yvals, marker="*", s=60,
                    color="gold", edgecolors="darkorange", linewidth=0.5,
                    label="Six", zorder=6)

    ax0.axhline(0.5, color="gray", linewidth=0.8, linestyle=":", alpha=0.7)
    _shade_phases(ax0)
    _draw_wickets(ax0)
    ax0.set_ylim([0, 1])
    ax0.set_ylabel("Win Prob.")
    ax0.legend(loc="upper right", ncol=3, fontsize=7)
    ax0.set_xticklabels([])

    # Panels 2–4: Latent states
    for i, (state_col, state_label, state_color) in enumerate(
        zip(_KF_STATE_COLS, _KF_STATE_LABELS, _KF_STATE_COLORS)
    ):
        ax = axes[i + 1]
        if state_col in match_df.columns:
            ax.plot(x, match_df[state_col], color=state_color, linewidth=1.8, zorder=4)
        ax.axhline(0, color="gray", linewidth=0.5, alpha=0.5)
        _shade_phases(ax)
        _draw_wickets(ax)
        ax.set_ylabel(state_label, fontsize=7)
        if i < 2:
            ax.set_xticklabels([])
        else:
            ax.set_xticks(ball_tick)
            ax.set_xticklabels(over_tick, fontsize=7)
            ax.set_xlabel("Ball Number (Over)")

    # Metadata in suptitle
    batting_team = match_df["batting_team"].iloc[0]
    bowling_team = match_df["bowling_team"].iloc[0]
    target       = match_df["target"].iloc[0]
    winner       = match_df.get("match_winner", pd.Series(["?"]))
    winner       = winner.iloc[0] if "match_winner" in match_df.columns else "?"

    fig.suptitle(
        f"{batting_team} vs {bowling_team}  |  Target: {int(target) if pd.notna(target) else '?'}"
        f"  |  Result: {winner} won\n{description}",
        fontsize=9, y=1.01, fontweight="bold",
    )

    stem = fig_stem or f"F2_case_study_{match_id}"
    save_figure(fig, stem)
    return fig


# ---------------------------------------------------------------------------
# F3 — Average latent trajectories: winning vs. losing chases
# ---------------------------------------------------------------------------

def plot_average_latent_trajectories(df: pd.DataFrame) -> plt.Figure:
    """
    Mean ± 1 SE of Ŝ_t, M̂_t, P̂_t at each ball number in 2nd innings,
    grouped by chase outcome (successful / failed).
    """
    inn2 = df[
        (df["innings_number"] == 2) &
        df["kf_strength"].notna()
    ].copy()

    if TARGET_COL not in inn2.columns:
        raise ValueError(f"Column '{TARGET_COL}' not in df.")

    inn2["ball_seq"] = inn2.groupby("match_id").cumcount()

    fig, axes = plt.subplots(3, 1, figsize=(FIG_WIDTH_DOUBLE, 8), sharex=True)
    fig.subplots_adjust(hspace=0.08)

    palette = {"Chase success": "#2E86AB", "Chase failure": "#DC3545"}

    for ax, col, label in zip(axes, _KF_STATE_COLS, _KF_STATE_LABELS):
        for outcome_val, outcome_label in [(1, "Chase success"), (0, "Chase failure")]:
            sub  = inn2[inn2[TARGET_COL] == outcome_val]
            grp  = sub.groupby("ball_seq")[col]
            mean = grp.mean()
            se   = grp.apply(sem)
            xs   = mean.index.values

            ax.plot(xs, mean.values, color=palette[outcome_label],
                    linewidth=1.8, label=outcome_label, zorder=4)
            ax.fill_between(xs, mean - se, mean + se,
                            color=palette[outcome_label], alpha=0.15, zorder=3)

        ax.axhline(0, color="gray", linewidth=0.5, alpha=0.5)
        ax.set_ylabel(label, fontsize=7)
        ax.legend(loc="upper right", fontsize=7)

    axes[-1].set_xlabel("Ball Number in 2nd Innings")
    axes[0].set_title(
        r"Mean Filtered Latent States: Successful vs. Failed Chases"
        "\n(Shaded region = ±1 SE across matches)",
        fontsize=9, pad=8,
    )

    fig.tight_layout()
    save_figure(fig, "F3_avg_latent_trajectories")
    return fig


# ---------------------------------------------------------------------------
# F4 — KF vs RF scatter with phase colour-coding
# ---------------------------------------------------------------------------

def plot_kf_vs_rf_scatter(df: pd.DataFrame) -> plt.Figure:
    """
    Scatter: KF predicted win prob (x) vs RF predicted win prob (y).
    Points coloured by innings phase. Marginal histograms on each axis.
    """
    from matplotlib.gridspec import GridSpec

    kf_col = _MODEL_COLS["Kalman Filter"]
    rf_col = _MODEL_COLS["Random Forest"]
    if kf_col not in df.columns or rf_col not in df.columns:
        raise ValueError("kf_win_prob or rf_win_prob not in df.")

    df = df.copy()

    def _phase_label(row):
        if row["phase_powerplay"] == 1:
            return "Powerplay"
        elif row["phase_middle"] == 1:
            return "Middle"
        else:
            return "Death"

    df["phase_label"] = df.apply(_phase_label, axis=1)

    fig = plt.figure(figsize=(FIG_WIDTH_DOUBLE, FIG_WIDTH_DOUBLE))
    gs  = GridSpec(2, 2, figure=fig,
                   width_ratios=[4, 1], height_ratios=[1, 4],
                   hspace=0.05, wspace=0.05)

    ax_main   = fig.add_subplot(gs[1, 0])
    ax_top    = fig.add_subplot(gs[0, 0], sharex=ax_main)
    ax_right  = fig.add_subplot(gs[1, 1], sharey=ax_main)

    # Sub-sample for speed (max 20,000 points)
    rng = np.random.default_rng(42)
    idx = rng.choice(len(df), size=min(20_000, len(df)), replace=False)
    sub = df.iloc[idx]

    for phase, color in [("Powerplay", PHASE_COLORS["powerplay"]),
                          ("Middle",    PHASE_COLORS["middle"]),
                          ("Death",     PHASE_COLORS["death"])]:
        mask = sub["phase_label"] == phase
        ax_main.scatter(
            sub.loc[mask, kf_col], sub.loc[mask, rf_col],
            s=3, alpha=0.3, color=color, label=phase, rasterized=True
        )

    # Perfect-agreement diagonal
    ax_main.plot([0, 1], [0, 1], "k--", linewidth=1.0, alpha=0.5, zorder=5)
    ax_main.set_xlabel("KF Predicted Win Probability")
    ax_main.set_ylabel("RF Predicted Win Probability")
    ax_main.set_xlim([0, 1])
    ax_main.set_ylim([0, 1])
    ax_main.legend(title="Phase", markerscale=4, fontsize=7)

    # Marginal histograms
    for phase, color in [("Powerplay", PHASE_COLORS["powerplay"]),
                          ("Middle",    PHASE_COLORS["middle"]),
                          ("Death",     PHASE_COLORS["death"])]:
        mask = sub["phase_label"] == phase
        ax_top.hist(sub.loc[mask, kf_col], bins=30, density=True,
                    color=color, alpha=0.5, histtype="stepfilled")
        ax_right.hist(sub.loc[mask, rf_col], bins=30, density=True,
                      color=color, alpha=0.5, histtype="stepfilled",
                      orientation="horizontal")

    ax_top.axis("off")
    ax_right.axis("off")
    fig.add_subplot(gs[0, 1]).axis("off")

    ax_main.set_title("KF vs RF Win Probability Predictions (Test Set)", pad=10)
    save_figure(fig, "F4_kf_vs_rf_scatter")
    return fig


# ---------------------------------------------------------------------------
# F5 — Volatility box plots
# ---------------------------------------------------------------------------

def plot_volatility_boxplot(per_match_df: pd.DataFrame) -> plt.Figure:
    """
    Box plot of per-match prediction volatility |Δp̂_t| distributions (F5).
    One box per model, sorted by median volatility.
    """
    avail_models = [m for m in _MODEL_ORDER if m in per_match_df["Model"].unique()]
    medians      = {m: per_match_df[per_match_df["Model"] == m]["volatility"].median()
                    for m in avail_models}
    ordered      = sorted(avail_models, key=lambda m: medians[m])

    data   = [per_match_df[per_match_df["Model"] == m]["volatility"].dropna().values
              for m in ordered]
    colors = [MODEL_COLORS[m] for m in ordered]

    fig, ax = plt.subplots(figsize=(FIG_WIDTH_DOUBLE, 4))
    bp = ax.boxplot(
        data,
        patch_artist=True,
        notch=False,
        showfliers=False,
        medianprops=dict(color="black", linewidth=1.5),
        whiskerprops=dict(linewidth=0.9),
        capprops=dict(linewidth=0.9),
    )
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    ax.set_xticks(range(1, len(ordered) + 1))
    ax.set_xticklabels(ordered)
    ax.set_ylabel(r"Mean $|\Delta\hat{p}_t|$ per Match")
    ax.set_title("Prediction Volatility Distribution by Model", pad=10)

    fig.tight_layout()
    save_figure(fig, "F5_volatility_boxplot")
    return fig


# ---------------------------------------------------------------------------
# F6 — Season-by-season Brier score line chart
# ---------------------------------------------------------------------------

def plot_season_brier(season_df: pd.DataFrame) -> plt.Figure:
    """
    Line chart: Brier score (y) vs. season (x) for all models (F6).
    """
    avail = [m for m in _MODEL_ORDER if m in season_df["Model"].unique()]
    seasons = sorted(season_df["Season"].unique())

    fig, ax = plt.subplots(figsize=(FIG_WIDTH_DOUBLE, 4))

    for name in avail:
        sub = season_df[season_df["Model"] == name].sort_values("Season")
        ax.plot(
            sub["Season"], sub["Brier"],
            "o-",
            color=MODEL_COLORS[name],
            linewidth=1.8,
            markersize=5,
            label=name,
        )

    ax.set_xlabel("IPL Season")
    ax.set_ylabel("Brier Score (lower = better)")
    ax.set_title("Season-by-Season Brier Score", pad=10)
    ax.set_xticks(seasons)
    ax.set_xticklabels([str(s) for s in seasons], rotation=45, ha="right", fontsize=7)
    ax.legend(loc="upper right", fontsize=7)
    ax.invert_yaxis()   # lower Brier = better → visually "up"

    fig.tight_layout()
    save_figure(fig, "F6_season_brier")
    return fig


# ---------------------------------------------------------------------------
# EM convergence diagnostic
# ---------------------------------------------------------------------------

import matplotlib.ticker as ticker

def plot_em_convergence(ll_history: list[float]) -> plt.Figure:
    """
    Line plot of approximate log-likelihood vs. EM iteration (K5 diagnostic).
    Fixed: Dynamic X-axis scaling to prevent unreadable, overlapping labels.
    """
    # Slightly wider figure for better readability
    fig, ax = plt.subplots(figsize=(FIG_WIDTH_SINGLE * 2.0, 3.5))
    
    iters = range(1, len(ll_history) + 1)
    ax.plot(iters, ll_history, "o-",
            color=MODEL_COLORS["Kalman Filter"], linewidth=1.8, markersize=5)
    
    ax.set_xlabel("EM Iteration")
    ax.set_ylabel("Approx. Log-Likelihood (normalised)")
    ax.set_title("EM Convergence", pad=8)
    
    # FIX: Use MaxNLocator to ensure only a readable number of integer ticks are shown
    ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True, nbins=10))
    
    # Optional: Add faint gridlines specifically for the x-axis to track iterations easily
    ax.grid(axis='x', linestyle='--', alpha=0.5)

    fig.tight_layout()
    # Save using the standard project wrapper
    save_figure(fig, "EM_convergence", subdir="")
    
    return fig


# ---------------------------------------------------------------------------
# SHAP summary plot  (H2)
# ---------------------------------------------------------------------------

def plot_shap_summary(
    shap_df: pd.DataFrame,
    n_top: int = 15,
) -> plt.Figure:
    """
    Horizontal bar chart of mean |SHAP| values for top n_top features.
    KF state features are highlighted with a distinct colour.
    """
    top    = shap_df.head(n_top).copy()
    colors = [MODEL_COLORS["Hybrid RF+KF"] if is_kf else "#AAAAAA"
              for is_kf in top["Is_KF"]]

    fig, ax = plt.subplots(figsize=(FIG_WIDTH_DOUBLE, n_top * 0.4 + 1.5))
    bars = ax.barh(top["Feature"][::-1], top["Mean_|SHAP|"][::-1],
                   color=colors[::-1], edgecolor="white", linewidth=0.5)

    ax.set_xlabel(r"Mean $|\text{SHAP value}|$")
    ax.set_title(f"Top {n_top} Feature Importances — Hybrid RF+KF Model", pad=10)

    # Legend
    kf_patch  = mpatches.Patch(color=MODEL_COLORS["Hybrid RF+KF"], label="KF latent state")
    ml_patch  = mpatches.Patch(color="#AAAAAA", label="Original feature")
    ax.legend(handles=[kf_patch, ml_patch], loc="lower right", fontsize=7)

    fig.tight_layout()
    save_figure(fig, "H2_shap_summary")
    return fig


# ---------------------------------------------------------------------------
# Sensitivity plots  (S1, S2)
# ---------------------------------------------------------------------------

def plot_window_sensitivity(sensitivity_df: pd.DataFrame) -> plt.Figure:
    """Bar chart of Brier scores for different rolling window configurations (S1)."""
    fig, ax = plt.subplots(figsize=(FIG_WIDTH_SINGLE * 2, 3))
    ax.bar(sensitivity_df["Config"], sensitivity_df["Brier"],
           color=MODEL_COLORS["Kalman Filter"], alpha=0.8, edgecolor="white")
    ax.set_ylabel("Brier Score")
    ax.set_title("Rolling-Window Length Sensitivity (S1)", pad=8)
    fig.tight_layout()
    save_figure(fig, "S1_window_sensitivity")
    return fig


def plot_noise_sensitivity(noise_df: pd.DataFrame) -> plt.Figure:
    """Line chart of Brier scores vs. Q scaling factor (S2)."""
    fig, ax = plt.subplots(figsize=(FIG_WIDTH_SINGLE * 2, 3))
    ax.plot(noise_df["Q_scale"], noise_df["Brier"],
            "o-", color=MODEL_COLORS["Kalman Filter"], linewidth=1.8, markersize=6)
    ax.set_xlabel("Q scale factor")
    ax.set_ylabel("Brier Score")
    ax.set_title("Process Noise Covariance Sensitivity (S2)", pad=8)
    ax.set_xticks(noise_df["Q_scale"])
    fig.tight_layout()
    save_figure(fig, "S2_noise_sensitivity")
    return fig
