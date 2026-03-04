import time
from typing import Iterable, List, Tuple

import pandas as pd
from nba_api.stats.endpoints import (
    leaguedashplayerstats,
    leaguedashplayerclutch,
    playergamelog,
)
from nba_api.stats.static import players
from nba_api.stats.endpoints import commonplayerinfo


def fetch_league_player_box(
    season: str,
    per_mode: str = "PerGame",
    measure_type: str = "Base",
) -> pd.DataFrame:
    """
    Fetch per-player per-season box/advanced stats from NBA stats.

    measure_type: "Base", "Advanced", "Scoring", "Usage", etc.
    """
    resp = leaguedashplayerstats.LeagueDashPlayerStats(
        season=season,
        per_mode_detailed=per_mode,
        measure_type_detailed_defense=measure_type,
    )
    df = resp.get_data_frames()[0]
    df["SEASON"] = season
    return df


def fetch_player_bio(active_only: bool = True) -> pd.DataFrame:
    """
    Fetch basic player bio info (height/weight, etc.).
    """
    if active_only:
        plist = players.get_active_players()
    else:
        plist = players.get_players()

    rows: List[dict] = []
    for p in plist:
        pid = p["id"]
        info = commonplayerinfo.CommonPlayerInfo(player_id=pid).get_data_frames()[0]
        row = info.iloc[0]
        rows.append(
            {
                "PLAYER_ID": pid,
                "PLAYER_NAME": row["DISPLAY_FIRST_LAST"],
                "HEIGHT": row["HEIGHT"],
                "WEIGHT": row["WEIGHT"],
                # Intentionally do NOT include listed position as modeling feature.
                "POSITION_LISTED": row.get("POSITION", None),
            }
        )
        time.sleep(0.6)  # avoid rate limits
    return pd.DataFrame(rows)


def fetch_clutch_on_off(season: str) -> pd.DataFrame:
    """
    Example of pulling clutch performance as a rough on/off-style proxy.
    This is optional and may be noisy, but can hint at impact in high-leverage minutes.
    """
    clutch = leaguedashplayerclutch.LeagueDashPlayerClutch(
        season=season,
        per_mode_detailed="Per100Possessions",
    ).get_data_frames()[0]
    clutch["SEASON"] = season
    return clutch[
        [
            "PLAYER_ID",
            "SEASON",
            "PLUS_MINUS",
            "OFF_RATING",
            "DEF_RATING",
            "NET_RATING",
            "GP",
        ]
    ]


def fetch_season_gamelogs(season: str) -> pd.DataFrame:
    """
    Pull game logs for all players for a season.
    Useful if you later want finer-grained distributions or on/off approximations.
    """
    plist = players.get_active_players()
    logs: List[pd.DataFrame] = []
    for p in plist:
        pid = p["id"]
        gl = playergamelog.PlayerGameLog(player_id=pid, season=season).get_data_frames()[
            0
        ]
        gl["PLAYER_ID"] = pid
        logs.append(gl)
        time.sleep(0.6)
    if not logs:
        return pd.DataFrame()
    out = pd.concat(logs, ignore_index=True)
    out["SEASON"] = season
    return out


def build_player_season_table(
    seasons: Iterable[str],
    active_only: bool = True,
    include_clutch: bool = True,
) -> pd.DataFrame:
    """
    Build a wide player-season table joining base, advanced, usage, scoring, and clutch data
    along with height/weight. Listed position is kept only for optional interpretation,
    but not used in modeling.
    """
    seasons = list(seasons)

    # Core box stats
    frames: List[pd.DataFrame] = []
    for s in seasons:
        base = fetch_league_player_box(season=s, per_mode="PerGame", measure_type="Base")
        adv = fetch_league_player_box(season=s, per_mode="PerGame", measure_type="Advanced")
        usage = fetch_league_player_box(season=s, per_mode="PerGame", measure_type="Usage")
        scoring = fetch_league_player_box(season=s, per_mode="PerGame", measure_type="Scoring")

        # Merge different measure types on player and season
        tmp = base.merge(
            adv,
            on=["PLAYER_ID", "PLAYER_NAME", "TEAM_ID", "SEASON"],
            suffixes=("", "_ADV"),
        )
        tmp = tmp.merge(
            usage,
            on=["PLAYER_ID", "PLAYER_NAME", "TEAM_ID", "SEASON"],
            suffixes=("", "_USG"),
        )
        tmp = tmp.merge(
            scoring,
            on=["PLAYER_ID", "PLAYER_NAME", "TEAM_ID", "SEASON"],
            suffixes=("", "_SCOR"),
        )

        if include_clutch:
            clutch = fetch_clutch_on_off(season=s)
            tmp = tmp.merge(
                clutch,
                on=["PLAYER_ID", "SEASON"],
                how="left",
                suffixes=("", "_CLUTCH"),
            )

        frames.append(tmp)

    stats = pd.concat(frames, ignore_index=True)

    bio = fetch_player_bio(active_only=active_only)

    merged = stats.merge(
        bio[["PLAYER_ID", "PLAYER_NAME", "HEIGHT", "WEIGHT", "POSITION_LISTED"]],
        on=["PLAYER_ID", "PLAYER_NAME"],
        how="left",
    )

    return merged


def cache_player_season_table(
    out_path: str,
    seasons: Tuple[str, ...] = ("2022-23", "2023-24"),
    active_only: bool = True,
) -> None:
    """
    Helper to fetch and cache a multi-season player table to disk.
    """
    df = build_player_season_table(seasons=seasons, active_only=active_only)
    out = pd.DataFrame(df)
    out.to_parquet(out_path, index=False)

