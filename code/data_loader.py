"""
data_loader.py
==============
Handles all data ingestion for the IPL Win-Probability paper.

Responsibilities:
  - Parse Cricsheet JSON files (organised as ipl_json/{year}/*.json)
  - Extract match-level metadata and ball-by-ball delivery records
  - Resolve season strings (e.g. "2007/08") to a canonical integer year
  - Apply DLS exclusion filter
  - Attach temporal sample weights
  - Perform strictly chronological train / validation / holdout splits

Public API
----------
    load_all_matches(data_dir)  -> (metadata_df, ball_df)
    split_by_season(ball_df, metadata_df)
        -> (train_df, val_df, test_df, train_meta, val_meta, test_meta)
    summary_statistics(ball_df, metadata_df) -> summary_df   # D3
"""

import json
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from config import (
    DATA_DIR,
    TRAIN_SEASONS,
    VALIDATION_SEASONS,
    TEST_SEASONS,
    get_temporal_weight,
    VENUE_MAPPING
)

warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _season_to_year(season: str) -> Optional[int]:
    if season is None:
        return None
    s = str(season).strip()
    try:
        return int(s[:4])
    except (ValueError, IndexError):
        return None


def _is_dls(info: Dict) -> bool:
    """Return True if the match outcome was decided by DLS method."""
    method = str(info.get("outcome", {}).get("method", "")).upper()
    return "DLS" in method


# ---------------------------------------------------------------------------
# Extraction functions
# ---------------------------------------------------------------------------

# Add this to your imports at the top of data_loader.py:
# from config import VENUE_MAPPING

def extract_match_metadata(match_data: Dict, match_id: str, folder_year: int) -> Dict:
    """
    Extract match-level information from a parsed Cricsheet JSON dict.
    Updated to standardize venue names.
    """
    info   = match_data["info"]
    teams  = info.get("teams", ["", ""])
    toss   = info.get("toss", {})
    outcome = info.get("outcome", {})

    result = outcome.get("result")
    is_abandoned = (result in ["no result", "abandoned"])
    is_dls = _is_dls(info)
    
    scheduled_overs = info.get("overs", 20)
    is_reduced = (scheduled_overs < 20)
    has_super_over = outcome.get("eliminator") is not None

    # FIX: Clean the venue string immediately upon extraction
    raw_venue = info.get("venue")
    clean_venue = VENUE_MAPPING.get(raw_venue, raw_venue) # fallback to raw if not in map

    return {
        "match_id"       : match_id,
        "match_date"     : info.get("dates", [None])[0],
        "season"         : info.get("season"),
        "year"           : folder_year,                  
        "city"           : info.get("city"),
        "venue"          : clean_venue,                  # Now using the standardized name!
        "team_a"         : teams[0] if len(teams) > 0 else None,
        "team_b"         : teams[1] if len(teams) > 1 else None,
        "toss_winner"    : toss.get("winner"),
        "toss_decision"  : toss.get("decision"),
        "winner"         : outcome.get("winner"),
        "result"         : result,
        "eliminator"     : outcome.get("eliminator"),
        "is_dls"         : is_dls,
        "is_abandoned"   : is_abandoned,
        "is_reduced"     : is_reduced,
        "has_super_over" : has_super_over,
        "temporal_weight": get_temporal_weight(folder_year) if folder_year else 1.0,
    }


def extract_ball_by_ball_data(match_data: Dict, match_id: str) -> pd.DataFrame:
    """
    Extract every delivery from both innings of a match into a flat DataFrame.

    Ball-level fields produced
    --------------------------
    match_id, innings_number, is_super_over,
    batting_team, bowling_team,
    over (0-indexed), ball (1-indexed within over),
    global_ball_number (1-indexed legal deliveries in innings),
    batter, non_striker, bowler,
    batter_runs, extra_runs, total_runs,
    is_wide, is_noball, is_valid_ball,
    is_wicket, wicket_kind,
    current_score, wickets_fallen,
    balls_bowled  (legal deliveries so far),
    balls_remaining,
    current_run_rate (runs per over),
    target, runs_required, required_run_rate
    """
    innings_list = match_data.get("innings", [])
    teams        = match_data["info"].get("teams", [])
    records: List[Dict] = []

    for inn_idx, inning in enumerate(innings_list):
        batting_team  = inning.get("team")
        bowling_team  = (teams[1] if batting_team == teams[0]
                         else teams[0]) if len(teams) == 2 else None
        is_super_over = bool(inning.get("super_over", False))

        # Only process the two main innings (skip additional super-overs)
        if inn_idx >= 2 and not is_super_over:
            continue

        total_balls_allowed = 6 if is_super_over else 120
        target_info = inning.get("target", {})
        target_runs = target_info.get("runs") if target_info else None

        current_score   = 0
        wickets_fallen  = 0
        balls_bowled    = 0     # legal deliveries only

        for over in inning.get("overs", []):
            over_num = over.get("over", 0)   # 0-indexed

            for ball_idx, delivery in enumerate(over.get("deliveries", [])):
                runs_info   = delivery.get("runs", {})
                extras_info = delivery.get("extras", {})

                batter_runs = runs_info.get("batter", 0)
                extra_runs  = runs_info.get("extras", 0)
                total_runs  = runs_info.get("total", 0)

                is_wide   = "wides"   in extras_info
                is_noball = "noballs" in extras_info
                is_valid  = not (is_wide or is_noball)

                # Wicket handling
                wicket_info = delivery.get("wickets", [])
                is_wicket   = len(wicket_info) > 0
                wicket_kind = wicket_info[0].get("kind") if is_wicket else None
                if is_wicket:
                    wickets_fallen += len(wicket_info)

                current_score += total_runs
                if is_valid:
                    balls_bowled += 1

                balls_remaining = max(0, total_balls_allowed - balls_bowled)
                crr = (current_score / balls_bowled * 6) if balls_bowled > 0 else 0.0
                runs_required = (target_runs - current_score) if target_runs else None
                rrr = (
                    (runs_required / balls_remaining * 6)
                    if (target_runs and balls_remaining > 0)
                    else None
                )

                records.append({
                    "match_id"           : match_id,
                    "innings_number"     : inn_idx + 1,
                    "is_super_over"      : is_super_over,
                    "batting_team"       : batting_team,
                    "bowling_team"       : bowling_team,
                    "over"               : over_num,
                    "ball"               : ball_idx + 1,
                    "global_ball_number" : balls_bowled,
                    "batter"             : delivery.get("batter"),
                    "non_striker"        : delivery.get("non_striker"),
                    "bowler"             : delivery.get("bowler"),
                    "batter_runs"        : batter_runs,
                    "extra_runs"         : extra_runs,
                    "total_runs"         : total_runs,
                    "is_wide"            : is_wide,
                    "is_noball"          : is_noball,
                    "is_valid_ball"      : is_valid,
                    "is_wicket"          : is_wicket,
                    "wicket_kind"        : wicket_kind,
                    "current_score"      : current_score,
                    "wickets_fallen"     : wickets_fallen,
                    "balls_bowled"       : balls_bowled,
                    "balls_remaining"    : balls_remaining,
                    "current_run_rate"   : crr,
                    "target"             : target_runs,
                    "runs_required"      : runs_required,
                    "required_run_rate"  : rrr,
                })

    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------

def load_all_matches(data_dir: Path = DATA_DIR) -> Tuple[pd.DataFrame, pd.DataFrame]:
    data_dir = Path(data_dir)
    json_files: List[Path] = sorted(data_dir.rglob("*.json"))
    
    all_meta: List[Dict] = []
    all_ball: List[pd.DataFrame] = []
    excluded_matches: List[Dict] = []  # To store details of skipped matches
    
    dls_count = 0
    abandoned_count = 0
    err_count = 0

    for json_file in tqdm(json_files, desc="Loading matches"):
        match_id = json_file.stem
        try:
            with open(json_file, "r", encoding="utf-8") as fh:
                match_data = json.load(fh)

            folder_year = _season_to_year(json_file.parent.name)
            meta = extract_match_metadata(match_data, match_id, folder_year)

            # --- EXCLUSION LOGIC ---
            if meta["is_dls"]:
                meta["exclusion_reason"] = "DLS Method Applied"
                excluded_matches.append(meta)
                dls_count += 1
                continue
            
            if meta["is_abandoned"] or meta["result"] in ["no result", "abandoned"]:
                meta["exclusion_reason"] = "Abandoned / No Result"
                excluded_matches.append(meta)
                abandoned_count += 1
                continue

            # --- VALID MATCHES ---
            all_meta.append(meta)
            ball_df = extract_ball_by_ball_data(match_data, match_id)
            
            if ball_df.empty:
                meta["exclusion_reason"] = "Empty ball-by-ball data"
                excluded_matches.append(meta)
                continue

            ball_df["season"] = meta["season"]
            ball_df["year"] = meta["year"]
            ball_df["temporal_weight"] = meta["temporal_weight"]
            all_ball.append(ball_df)

        except Exception as exc:
            err_count += 1
            excluded_matches.append({"match_id": match_id, "exclusion_reason": f"Error: {str(exc)}"})
            continue

    # Create DataFrames
    metadata_df = pd.DataFrame(all_meta)
    ball_df = pd.concat(all_ball, ignore_index=True) if all_ball else pd.DataFrame()
    
    # Save the Exclusions to CSV for your records
    if excluded_matches:
        excl_df = pd.DataFrame(excluded_matches)
        # Ensure the output directory exists
        import config
        excl_df.to_csv(config.TABLES_DIR / "D1_excluded_matches_audit.csv", index=False)
        print(f"--- [D1] Exclusion Audit saved to D1_excluded_matches_audit.csv ---")

    print(
        f"\n✅  Processed {len(metadata_df)} matches | "
        f"DLS excluded: {dls_count} | "
        f"Abandoned excluded: {abandoned_count} | "
        f"Errors: {err_count}"
    )
    
    return metadata_df, ball_df


# ---------------------------------------------------------------------------
# Train / Val / Test split   (D2 — CRITICAL)
# ---------------------------------------------------------------------------

def split_by_season(
    ball_df: pd.DataFrame,
    metadata_df: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame,
           pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Strictly chronological season-based split.

    Split assignments (from config.py):
        Training   : 2008 - 2019
        Validation : 2020 - 2021
        Holdout    : 2022 - 2024

    Parameters
    ----------
    ball_df     : full delivery DataFrame (must have 'year' column)
    metadata_df : full metadata DataFrame (must have 'year' column)

    Returns
    -------
    train_ball, val_ball, test_ball,
    train_meta, val_meta, test_meta
    """
    def _split(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        train = df[df["year"].isin(TRAIN_SEASONS)].copy()
        val   = df[df["year"].isin(VALIDATION_SEASONS)].copy()
        test  = df[df["year"].isin(TEST_SEASONS)].copy()
        return train, val, test

    train_ball, val_ball, test_ball = _split(ball_df)
    train_meta, val_meta, test_meta = _split(metadata_df)

    # Diagnostics — printed for paper table D2
    print("\n── Train / Validation / Holdout Split ────────────────────")
    for label, bdf, mdf in [
        ("Training  ", train_ball, train_meta),
        ("Validation", val_ball,   val_meta),
        ("Holdout   ", test_ball,  test_meta),
    ]:
        seasons = sorted(mdf["year"].dropna().unique().astype(int))
        print(
            f"  {label}: seasons {seasons[0]}–{seasons[-1]}  |  "
            f"{len(mdf):4d} matches  |  {len(bdf):>8,} deliveries"
        )
    print("───────────────────────────────────────────────────────────\n")

    return train_ball, val_ball, test_ball, train_meta, val_meta, test_meta


# ---------------------------------------------------------------------------
# Summary statistics table   (D3 — CRITICAL)
# ---------------------------------------------------------------------------

def summary_statistics(
    ball_df: pd.DataFrame,
    metadata_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Produce the D3 summary statistics table aggregated by season,
    plus a single totals row.

    Columns
    -------
    Season, Matches, Total_Balls, Avg_1st_Innings_Total,
    Avg_2nd_Innings_Total, Chase_Success_Rate_pct,
    Avg_Boundaries_per_Innings, Avg_Dot_Rate_pct

    Returns
    -------
    pd.DataFrame — one row per season + one 'All Seasons' row
    """
    # Attach match outcome (winner) to ball data
    winner_map = (
        metadata_df.set_index("match_id")["winner"].to_dict()
    )
    ball_df = ball_df.copy()
    ball_df["match_winner"] = ball_df["match_id"].map(winner_map)

    # First-innings totals per match
    inn1 = (
        ball_df[ball_df["innings_number"] == 1]
        .groupby(["year", "match_id"])["current_score"].max()
        .reset_index()
        .rename(columns={"current_score": "first_inn_score"})
    )
    # Second-innings totals per match
    inn2 = (
        ball_df[ball_df["innings_number"] == 2]
        .groupby(["year", "match_id"])
        .apply(lambda g: pd.Series({
            "second_inn_score": g["current_score"].max(),
            "batting_team"    : g["batting_team"].iloc[0],
            "match_winner"    : g["match_winner"].iloc[0],
        }))
        .reset_index()
    )
    inn2["chase_success"] = (inn2["batting_team"] == inn2["match_winner"]).astype(int)

    # Boundaries and dots
    ball_df["is_boundary"] = ball_df["batter_runs"].isin([4, 6]).astype(int)
    ball_df["is_dot"]      = (ball_df["total_runs"] == 0).astype(int)

    rows = []
    all_seasons = sorted(ball_df["year"].dropna().unique().astype(int))

    for season in all_seasons:
        sb  = ball_df[ball_df["year"] == season]
        si1 = inn1[inn1["year"] == season]
        si2 = inn2[inn2["year"] == season]

        n_matches = sb["match_id"].nunique()
        total_balls = sb["is_valid_ball"].sum() if "is_valid_ball" in sb else len(sb)
        avg_1st = si1["first_inn_score"].mean()
        avg_2nd = si2["second_inn_score"].mean()
        chase_pct = si2["chase_success"].mean() * 100 if len(si2) else np.nan

        # Boundaries per innings (both innings combined)
        bnd_per_inn = (
            sb[sb["is_valid_ball"] == True]["is_boundary"].sum() / (n_matches * 2)
            if n_matches else np.nan
        )
        dot_rate = (
            sb["is_dot"].sum() / len(sb) * 100 if len(sb) else np.nan
        )

        rows.append({
            "Season"                    : season,
            "Matches"                   : n_matches,
            "Total_Balls"               : int(total_balls),
            "Avg_1st_Innings_Total"     : round(avg_1st, 1),
            "Avg_2nd_Innings_Total"     : round(avg_2nd, 1),
            "Chase_Success_Rate_pct"    : round(chase_pct, 1),
            "Avg_Boundaries_per_Innings": round(bnd_per_inn, 1),
            "Avg_Dot_Rate_pct"          : round(dot_rate, 1),
        })

    summary = pd.DataFrame(rows)

    # Totals row
    totals = {
        "Season"                    : "All Seasons",
        "Matches"                   : summary["Matches"].sum(),
        "Total_Balls"               : summary["Total_Balls"].sum(),
        "Avg_1st_Innings_Total"     : round(summary["Avg_1st_Innings_Total"].mean(), 1),
        "Avg_2nd_Innings_Total"     : round(summary["Avg_2nd_Innings_Total"].mean(), 1),
        "Chase_Success_Rate_pct"    : round(summary["Chase_Success_Rate_pct"].mean(), 1),
        "Avg_Boundaries_per_Innings": round(summary["Avg_Boundaries_per_Innings"].mean(), 1),
        "Avg_Dot_Rate_pct"          : round(summary["Avg_Dot_Rate_pct"].mean(), 1),
    }
    summary = pd.concat([summary, pd.DataFrame([totals])], ignore_index=True)
    return summary


# ---------------------------------------------------------------------------
# Venue information   (D4 — IMPORTANT)
# ---------------------------------------------------------------------------

def compute_venue_stats(
    ball_df: pd.DataFrame,
    metadata_df: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Compute per-venue historical average first-innings total (D4).
    Returns both the overall stats and the season-by-season breakdown.
    """
    venue_map = metadata_df.set_index("match_id")["venue"].to_dict()
    ball_df = ball_df.copy()
    ball_df["venue"] = ball_df["match_id"].map(venue_map)

    # 1. Calculate max score for 1st innings per match
    first_inn_max = (
        ball_df[ball_df["innings_number"] == 1]
        .groupby(["year", "venue", "match_id"])["current_score"].max()
        .reset_index()
        .rename(columns={"current_score": "first_inn_score"})
    )

    # 2. Entire IPL Stats (D4 Overall)
    venue_stats_entire = (
        first_inn_max.groupby("venue")["first_inn_score"]
        .agg(n_matches="count", avg_first_innings_total="mean")
        .reset_index()
        .sort_values("n_matches", ascending=False)
    )
    venue_stats_entire["avg_first_innings_total"] = venue_stats_entire["avg_first_innings_total"].round(1)

    # 3. Yearly Venue Stats (D4 Yearly)
    venue_stats_yearly = (
        first_inn_max.groupby(["year", "venue"])["first_inn_score"]
        .agg(n_matches="count", avg_first_innings_total="mean")
        .reset_index()
    )
    venue_stats_yearly["avg_first_innings_total"] = venue_stats_yearly["avg_first_innings_total"].round(1)

    print(f"  Unique venues: {len(venue_stats_entire)}")
    return venue_stats_entire, venue_stats_yearly