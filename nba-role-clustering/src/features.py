from typing import List, Tuple

import numpy as np
import pandas as pd


def parse_height_to_inches(height_str: str) -> float:
    if not isinstance(height_str, str) or "-" not in height_str:
        return np.nan
    feet, inches = height_str.split("-")
    try:
        return int(feet) * 12 + int(inches)
    except ValueError:
        return np.nan


def build_feature_matrix(
    path: str,
    min_minutes: float = 600.0,
    min_games: int = 25,
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Build a player-season-level feature matrix aligned with the role framework.

    - Drops official position fields from the model features.
    - Keeps some meta fields (e.g., listed position) only for optional interpretation.
    """
    df = pd.read_parquet(path)

    # Filter out small-sample players based on total minutes and games played.
    # MIN and GP column names come from LeagueDashPlayerStats base measure.
    if "MIN" in df.columns and "GP" in df.columns:
        df["MIN_TOTAL"] = df["MIN"] * df["GP"]
        df = df[(df["MIN_TOTAL"] >= min_minutes) & (df["GP"] >= min_games)].copy()

    # Height / weight
    df["HEIGHT_IN"] = df["HEIGHT"].apply(parse_height_to_inches)
    df["WEIGHT_LB"] = pd.to_numeric(df["WEIGHT"], errors="coerce")

    # Core offensive load / playmaking (from Advanced & Usage tables)
    # Column names follow NBA stats conventions like USG_PCT, AST_PCT, TOV_PCT, TS_PCT, etc.
    # Some may not be present if endpoints change; we guard with get.
    for col in [
        "USG_PCT",
        "AST_PCT",
        "TOV_PCT",
        "TS_PCT",
        "POSS",
        "PACE",
        "OFF_RATING",
        "DEF_RATING",
        "NET_RATING",
    ]:
        if col not in df.columns:
            df[col] = np.nan

    # Assist-to-turnover ratio
    if "AST" in df.columns and "TOV" in df.columns:
        df["AST_TO_TOV"] = df["AST"] / df["TOV"].replace(0, np.nan)
    else:
        df["AST_TO_TOV"] = np.nan

    # Shot profile: 3PA rate, FTr, basic efficiency per minute
    for col in ["FGA", "FG3A", "FTA", "PTS", "MIN"]:
        if col not in df.columns:
            df[col] = np.nan

    df["THREE_RATE"] = df["FG3A"] / df["FGA"].replace(0, np.nan)
    df["FT_RATE"] = df["FTA"] / df["FGA"].replace(0, np.nan)
    df["SCORING_PER_MIN"] = df["PTS"] / df["MIN"].replace(0, np.nan)

    # Rebounding and defensive activity
    for col in ["OREB", "DREB", "REB", "STL", "BLK"]:
        if col not in df.columns:
            df[col] = np.nan

    df["OREB_PER_MIN"] = df["OREB"] / df["MIN"].replace(0, np.nan)
    df["DREB_PER_MIN"] = df["DREB"] / df["MIN"].replace(0, np.nan)
    df["REB_PER_MIN"] = df["REB"] / df["MIN"].replace(0, np.nan)
    df["STL_PER_MIN"] = df["STL"] / df["MIN"].replace(0, np.nan)
    df["BLK_PER_MIN"] = df["BLK"] / df["MIN"].replace(0, np.nan)

    # Clutch / impact proxy from clutch table (PLUS_MINUS, OFF_RATING, DEF_RATING, NET_RATING)
    for col in ["PLUS_MINUS", "OFF_RATING_CLUTCH", "DEF_RATING_CLUTCH", "NET_RATING_CLUTCH"]:
        if col not in df.columns:
            # Map raw clutch columns if they exist
            if col.startswith("OFF_RATING") and "OFF_RATING_CLUTCH" in df.columns:
                continue
            df[col] = np.nan

    # Select features for clustering (no positions)
    feature_cols: List[str] = [
        # Size
        "HEIGHT_IN",
        "WEIGHT_LB",
        # Usage / playmaking
        "USG_PCT",
        "AST_PCT",
        "TOV_PCT",
        "AST_TO_TOV",
        # Scoring / shooting profile
        "TS_PCT",
        "THREE_RATE",
        "FT_RATE",
        "SCORING_PER_MIN",
        # Rebounding / defense
        "OREB_PER_MIN",
        "DREB_PER_MIN",
        "REB_PER_MIN",
        "STL_PER_MIN",
        "BLK_PER_MIN",
        # Impact proxies
        "PLUS_MINUS",
        "OFF_RATING",
        "DEF_RATING",
        "NET_RATING",
    ]

    # Keep meta columns for later interpretation
    keep_cols = [
        "PLAYER_ID",
        "PLAYER_NAME",
        "TEAM_ID",
        "SEASON",
        "HEIGHT",
        "WEIGHT",
        "POSITION_LISTED",
    ]

    # Ensure all feature columns exist
    for col in feature_cols:
        if col not in df.columns:
            df[col] = np.nan

    feats = df[keep_cols + feature_cols].copy()
    feats = feats.dropna(subset=feature_cols, how="all")

    return feats, feature_cols

