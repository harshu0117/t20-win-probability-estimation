"""
config.py
=========
Central configuration module for the IPL Kalman Filter Win Probability paper.
All constants, paths, season splits, feature lists, model hyperparameters,
and publication figure styling are defined here.

DO NOT hard-code any of these values elsewhere — import from this module.
"""

from pathlib import Path
import matplotlib as mpl
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# 1. DIRECTORY PATHS
# ---------------------------------------------------------------------------

# Root of the project (directory containing this file)
PROJECT_ROOT = Path(__file__).parent.resolve()

# IPL JSON data: year-level sub-folders (2008/, 2009/, ..., 2024/)
DATA_DIR = PROJECT_ROOT / "ipl_json"

# All outputs (CSVs, figures, model artefacts)
OUTPUT_DIR = PROJECT_ROOT / "outputs"
FIGURES_DIR = OUTPUT_DIR / "figures"
TABLES_DIR  = OUTPUT_DIR / "tables"
MODELS_DIR  = OUTPUT_DIR / "models"

for _d in [OUTPUT_DIR, FIGURES_DIR, TABLES_DIR, MODELS_DIR]:
    _d.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# 2. TEMPORAL WEIGHTING
# ---------------------------------------------------------------------------

CURRENT_YEAR  = 2025
RECENT_YEARS  = 4          # last 4 seasons receive weight = 1.0
DECAY_RATE    = 0.05       # linear decay per year before the recent window
MIN_WEIGHT    = 0.50       # floor weight


def get_temporal_weight(year: int) -> float:
    """Return a sample weight in [MIN_WEIGHT, 1.0] for a given season year."""
    cutoff = CURRENT_YEAR - RECENT_YEARS + 1
    if year >= cutoff:
        return 1.0
    return max(MIN_WEIGHT, 1.0 - DECAY_RATE * (cutoff - year))


# ---------------------------------------------------------------------------
# 3. TRAIN / VALIDATION / HOLDOUT-TEST SEASON SPLITS
#    (D2 — CRITICAL: exact season assignments)
# ---------------------------------------------------------------------------

# IPL seasons available: 2008–2024
# Strategy: chronological split to avoid data leakage
#   Training  : 2008 – 2019  (12 seasons, ~780 matches)
#   Validation: 2020 – 2021  ( 2 seasons, ~120 matches — bio-bubble era)
#   Holdout   : 2022 – 2024  ( 3 seasons, ~210 matches)

TRAIN_SEASONS      = list(range(2008, 2021))   # 2008..2020 inclusive
VALIDATION_SEASONS = [2021, 2022]
TEST_SEASONS       = [2023, 2024, 2025]



# ---------------------------------------------------------------------------
# 4. FEATURE CONFIGURATION  (D5)
# ---------------------------------------------------------------------------

# Rolling window lengths (balls)
ROLLING_WINDOWS = [6, 12, 24]

# Observation features fed into the Kalman Filter measurement equation
OBS_FEATURES = [
    "runs_last_6_balls",
    "runs_last_12_balls",
    "runs_last_24_balls",
    "wickets_last_6_balls",
    "wickets_last_12_balls",
    "wickets_last_24_balls",
    "dots_last_6_balls",
    "dots_last_12_balls",
    "dots_last_24_balls",
    "boundaries_last_6_balls",
    "boundaries_last_12_balls",
    "boundaries_last_24_balls",
    "current_run_rate",
]

# Control / exogenous inputs fed into the B·u term
CTRL_FEATURES = [
    "balls_remaining",
    "wickets_in_hand",
    "run_rate_diff",
    "pressure_index",
    "phase_powerplay",
    "phase_middle",
    "phase_death",
    "required_run_rate",
]

# Features used by the ML baseline models (superset of observation features)
ML_FEATURES = OBS_FEATURES + CTRL_FEATURES + [
    "par_deviation",
    "venue_avg_first_innings",   # venue fixed effect (D4)
    "innings_number",
]

# Target column
TARGET_COL = "batting_team_won"

# Phase definitions (over numbers, 0-indexed)
PHASE_POWERPLAY_OVERS = (0, 5)    # overs 1–6
PHASE_MIDDLE_OVERS    = (6, 15)   # overs 7–15 (inclusive: over index 6..15)
PHASE_DEATH_OVERS     = (16, 19)  # overs 17–20


# ---------------------------------------------------------------------------
# 5. KALMAN FILTER PARAMETERS  (K1–K7)
# ---------------------------------------------------------------------------

KF_N_STATES = 3           # latent state: [Strength Ŝ, Momentum M̂, Pressure P̂]

# Diagonal persistence parameters for A (K1)
KF_ALPHA_S  = 0.85        # batting-strength persistence (Lowered for faster reaction)
KF_ALPHA_M  = 0.60        # momentum persistence (Lowered to track bursts)
KF_ALPHA_P  = 0.80        # pressure persistence

# Process noise (Q diagonal) — scaled per state (K4)
KF_Q_DIAG   = [0.05, 0.25, 0.10]   # Dramatically increased for higher state mobility

# Measurement noise initial scale (R = σ²·I); refined by EM  (K4)
KF_R_INIT_SCALE = 1.0     # Lowered to trust observations more initially

# EM algorithm settings (K5)
KF_EM_MAX_ITERS     = 150
KF_EM_H_DAMPING     = 0.40    # Faster learning
KF_EM_R_DAMPING     = 0.25
KF_EM_TOL           = 1e-6    
KF_EM_REG           = 0.05    # Minimal regularisation to prevent signal dampening

# Initialisation (K7)
KF_VENUE_AVG_DEFAULT = 175.0   

# Second-stage logistic calibration input columns (K6)
# Fully synchronized with the machine learning feature set + latent states
# Adding quadratic latent interactions for a non-linear probability surface
KF_CALIB_COLS = ['kf_strength', 'kf_momentum', 'kf_pressure'] + ML_FEATURES + [
    'kf_strength_wickets', 'kf_momentum_runs', 'kf_pressure_rrr'
]


# ---------------------------------------------------------------------------
# 6. BASELINE MODEL HYPERPARAMETERS  (B1–B3)
# ---------------------------------------------------------------------------

# Logistic Regression (B1)
LR_PARAMS = dict(
    penalty="l2",
    C=1.0,                  # tuned via 5-fold season-blocked CV
    solver="lbfgs",
    max_iter=1000,
    random_state=42,
)

# Random Forest (B2)
RF_PARAMS = dict(
    n_estimators=1000,      # Increased for stability
    max_depth=15,           # Deeper trees
    min_samples_leaf=5,
    max_features="sqrt",
    n_jobs=-1,
    random_state=42,
)

# Gradient Boosting / XGBoost (B3)
GB_PARAMS = dict(
    n_estimators=1000,
    learning_rate=0.03,     # Slower learning rate for better generalisation
    max_depth=7,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.1,
    reg_lambda=1.0,
    random_state=42,
    n_jobs=-1,
    eval_metric="logloss",
)

# Cross-validation folds for hyperparameter tuning
CV_FOLDS = 5


# ---------------------------------------------------------------------------
# 7. EVALUATION SETTINGS  (R1–R3, X1–X4)
# ---------------------------------------------------------------------------

BOOTSTRAP_N_RESAMPLES  = 1000     # bootstrap iterations for CIs (R2)
BOOTSTRAP_CI_LEVEL     = 0.95
CALIBRATION_N_BINS     = 10       # equal-width bins (F1)

# Close-match definition (X2) — using WP crossing criterion
CLOSE_MATCH_MIN_CROSSES = 3       # WP crossed 50% ≥ 3 times in 2nd innings

# Minimum matches per venue for leave-one-out eval (X4)
VENUE_MIN_MATCHES = 20


# ---------------------------------------------------------------------------
# 8. PUBLICATION FIGURE SETTINGS
# ---------------------------------------------------------------------------

# Consistent model colour palette across ALL figures
MODEL_COLORS = {
    "Kalman Filter"       : "#2E86AB",
    "Logistic Regression" : "#6C757D",
    "Random Forest"       : "#FD7F20",
    "Gradient Boosting"   : "#DC3545",
    "Hybrid RF+KF"        : "#8338EC",
    "Hybrid GB+KF"        : "#FB5607",
    "EKF"                 : "#3A86FF",
    "UKF"                 : "#06D6A0",
}

# Phase shading colours
PHASE_COLORS = {
    "powerplay" : "#4CAF50",
    "middle"    : "#FF9800",
    "death"     : "#F44336",
}

# Figure resolution and format
FIG_DPI        = 300
FIG_FORMAT     = ["pdf", "png"]    # save both vector and raster
FIG_FONT_FAMILY = "serif"

# IEEE double-column figure widths (inches)
FIG_WIDTH_SINGLE = 3.5
FIG_WIDTH_DOUBLE = 7.16
FIG_WIDTH_FULL   = 9.0

def set_publication_style() -> None:
    """
    Apply publication-quality matplotlib rcParams.
    Call once at the top of any plotting module or notebook.
    Uses a clean serif style suitable for IEEE/Springer journals.
    """
    mpl.rcParams.update({
        # Font
        "font.family"        : FIG_FONT_FAMILY,
        "font.size"          : 9,
        "axes.titlesize"     : 10,
        "axes.labelsize"     : 9,
        "xtick.labelsize"    : 8,
        "ytick.labelsize"    : 8,
        "legend.fontsize"    : 8,
        "figure.titlesize"   : 11,

        # Lines & markers
        "lines.linewidth"    : 1.5,
        "lines.markersize"   : 4,
        "patch.linewidth"    : 0.8,

        # Axes
        "axes.spines.top"    : False,
        "axes.spines.right"  : False,
        "axes.grid"          : True,
        "grid.alpha"         : 0.3,
        "grid.linewidth"     : 0.5,
        "axes.axisbelow"     : True,

        # Figure
        "figure.dpi"         : FIG_DPI,
        "savefig.dpi"        : FIG_DPI,
        "savefig.bbox"       : "tight",
        "savefig.pad_inches" : 0.05,

        # Legend
        "legend.framealpha"  : 0.9,
        "legend.edgecolor"   : "0.8",
        "legend.borderpad"   : 0.4,
    })


def save_figure(fig: mpl.figure.Figure, stem: str, subdir: str = "") -> None:
    """
    Save a figure in all formats defined in FIG_FORMAT.

    Parameters
    ----------
    fig   : matplotlib Figure object
    stem  : filename stem (no extension), e.g. 'calibration_curves'
    subdir: optional sub-folder inside FIGURES_DIR
    """
    out = FIGURES_DIR / subdir if subdir else FIGURES_DIR
    out.mkdir(parents=True, exist_ok=True)
    for ext in FIG_FORMAT:
        path = out / f"{stem}.{ext}"
        fig.savefig(path, format=ext)
        print(f"  Saved: {path}")

# ---------------------------------------------------------------------------
# VENUE STANDARDIZATION MAPPING
# ---------------------------------------------------------------------------
VENUE_MAPPING = {
    # Bengaluru
    "M Chinnaswamy Stadium": "M. Chinnaswamy Stadium",
    "M Chinnaswamy Stadium, Bengaluru": "M. Chinnaswamy Stadium",
    "M.Chinnaswamy Stadium": "M. Chinnaswamy Stadium",

    # Mumbai
    "Wankhede Stadium, Mumbai": "Wankhede Stadium",
    "Dr DY Patil Sports Academy, Mumbai": "Dr DY Patil Sports Academy",
    "Brabourne Stadium, Mumbai": "Brabourne Stadium",

    # Kolkata
    "Eden Gardens, Kolkata": "Eden Gardens",

    # Delhi
    "Feroz Shah Kotla": "Arun Jaitley Stadium",
    "Arun Jaitley Stadium, Delhi": "Arun Jaitley Stadium",

    # Hyderabad
    "Rajiv Gandhi International Stadium, Uppal": "Rajiv Gandhi International Stadium",
    "Rajiv Gandhi International Stadium, Uppal, Hyderabad": "Rajiv Gandhi International Stadium",

    # Chennai
    "MA Chidambaram Stadium, Chepauk": "MA Chidambaram Stadium",
    "MA Chidambaram Stadium, Chepauk, Chennai": "MA Chidambaram Stadium",

    # Jaipur
    "Sawai Mansingh Stadium, Jaipur": "Sawai Mansingh Stadium",

    # Mohali / Punjab
    "Punjab Cricket Association Stadium, Mohali": "Punjab Cricket Association IS Bindra Stadium",
    "Punjab Cricket Association IS Bindra Stadium, Mohali": "Punjab Cricket Association IS Bindra Stadium",
    "Punjab Cricket Association IS Bindra Stadium, Mohali, Chandigarh": "Punjab Cricket Association IS Bindra Stadium",
    "Maharaja Yadavindra Singh International Cricket Stadium, Mullanpur": "Maharaja Yadavindra Singh International Cricket Stadium",
    "Maharaja Yadavindra Singh International Cricket Stadium, New Chandigarh": "Maharaja Yadavindra Singh International Cricket Stadium",

    # Ahmedabad
    "Sardar Patel Stadium, Motera": "Narendra Modi Stadium",
    "Narendra Modi Stadium, Ahmedabad": "Narendra Modi Stadium",

    # Pune
    "Maharashtra Cricket Association Stadium, Pune": "Maharashtra Cricket Association Stadium",
    "Subrata Roy Sahara Stadium": "Maharashtra Cricket Association Stadium",

    # Visakhapatnam
    "Dr. Y.S. Rajasekhara Reddy ACA-VDCA Cricket Stadium, Visakhapatnam": "Dr. Y.S. Rajasekhara Reddy ACA-VDCA Cricket Stadium",

    # Dharamsala
    "Himachal Pradesh Cricket Association Stadium, Dharamsala": "Himachal Pradesh Cricket Association Stadium",

    # UAE Venues
    "Zayed Cricket Stadium, Abu Dhabi": "Sheikh Zayed Stadium",

    # Others
    "Barsapara Cricket Stadium, Guwahati": "Barsapara Cricket Stadium",
    "Vidarbha Cricket Association Stadium, Jamtha": "Vidarbha Cricket Association Stadium"
}