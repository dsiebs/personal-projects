from typing import Dict, Iterable, List, Tuple

import pandas as pd


# Canonical role schema provided by the user.
ROLE_SCHEMA: Dict[str, Dict[str, str]] = {
    "Offensive Hub / Do-It-All Engine": {
        "group": "Primary Offensive",
        "description": "High-usage offensive engine who drives both scoring and playmaking.",
    },
    "Pass-First Orchestrator": {
        "group": "Primary Offensive",
        "description": "Primary organizer with high assist share and moderate usage.",
    },
    "Secondary Creator / Combo Guard": {
        "group": "Primary Offensive",
        "description": "Can run offense in stretches while playing next to a star.",
    },
    "Isolation Scorer": {
        "group": "Primary Offensive",
        "description": "Shot-creation specialist with strong isolation scoring volume.",
    },
    "Three-Level Shot Creator": {
        "group": "Primary Offensive",
        "description": "Efficient at rim, midrange, and three with versatile scoring package.",
    },
    "Movement Shooter": {
        "group": "Off-Ball Scoring",
        "description": "High-movement shooter with significant off-screen / relocation activity.",
    },
    "Spot-Up Specialist": {
        "group": "Off-Ball Scoring",
        "description": "Low-usage spacer whose shots are mostly catch-and-shoot threes.",
    },
    "Slasher / Rim Pressure Wing": {
        "group": "Off-Ball Scoring",
        "description": "Attacks closeouts and pressures the rim with drives and free throws.",
    },
    "Transition Finisher": {
        "group": "Off-Ball Scoring",
        "description": "Scores efficiently in transition with limited half-court creation.",
    },
    "Microwave Scorer": {
        "group": "Off-Ball Scoring",
        "description": "Bench scorer with high usage per minute and volatile shooting.",
    },
    "Stretch Big": {
        "group": "Frontcourt Offensive",
        "description": "Big who spaces the floor from three and pick-and-pop actions.",
    },
    "Interior Scoring Big": {
        "group": "Frontcourt Offensive",
        "description": "Paint/post scorer with low three-point volume and strong rim efficiency.",
    },
    "Short-Roll Playmaker": {
        "group": "Frontcourt Offensive",
        "description": "Playmaking big in short-roll situations, creating for others.",
    },
    "Rim Runner / Vertical Spacer": {
        "group": "Frontcourt Offensive",
        "description": "Lob threat and screen-and-roll finisher with minimal self-creation.",
    },
    "Point-of-Attack Defender": {
        "group": "Defensive",
        "description": "Guarding primary ball handlers and applying on-ball pressure.",
    },
    "Switchable Wing Defender": {
        "group": "Defensive",
        "description": "Wing capable of guarding multiple perimeter positions.",
    },
    "Rim Protector": {
        "group": "Defensive",
        "description": "Anchors interior defense with blocks and rim deterrence.",
    },
    "Help-Side Roamer / Free Safety": {
        "group": "Defensive",
        "description": "Off-ball disruptor who jumps passing lanes and rotates aggressively.",
    },
    "Switch Big / Small-Ball 5": {
        "group": "Defensive",
        "description": "Mobile big who can switch across positions and anchor small units.",
    },
    "Connector": {
        "group": "Glue / Hybrid",
        "description": "Low-usage ball-mover who keeps offense flowing.",
    },
    "Energy Wing / Chaos Creator": {
        "group": "Glue / Hybrid",
        "description": "High-activity wing impacting possessions via hustle and disruption.",
    },
    "Rebounding Specialist": {
        "group": "Glue / Hybrid",
        "description": "Elite rebounder who influences possession margins.",
    },
}


def summarize_clusters(
    feats_clustered: pd.DataFrame,
    feature_cols: Iterable[str],
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Compute cluster-level means and sizes for quick inspection.
    """
    feature_cols = list(feature_cols)
    cluster_means = (
        feats_clustered.groupby("CLUSTER")[feature_cols].mean().sort_index()
    )
    cluster_sizes = feats_clustered.groupby("CLUSTER")["PLAYER_ID"].nunique().sort_index()
    return cluster_means, cluster_sizes


def assign_role_labels(
    feats_clustered: pd.DataFrame,
    cluster_to_role: Dict[int, str],
) -> pd.DataFrame:
    """
    Attach human-readable role labels (from ROLE_SCHEMA) to clustered players.
    """
    out = feats_clustered.copy()
    out["ROLE_NAME"] = out["CLUSTER"].map(cluster_to_role)
    out["ROLE_GROUP"] = out["ROLE_NAME"].map(
        lambda r: ROLE_SCHEMA.get(r, {}).get("group") if r is not None else None
    )
    return out


def role_summary_table(
    feats_with_roles: pd.DataFrame,
    feature_cols: Iterable[str],
) -> pd.DataFrame:
    """
    Summarize each role by feature means and player counts.
    """
    feature_cols = list(feature_cols)
    grouped = feats_with_roles.groupby("ROLE_NAME")
    means = grouped[feature_cols].mean()
    counts = grouped["PLAYER_ID"].nunique().rename("N_PLAYERS")
    summary = means.join(counts)
    return summary.sort_values("N_PLAYERS", ascending=False)


def example_players_by_role(
    feats_with_roles: pd.DataFrame,
    n: int = 10,
) -> Dict[str, pd.DataFrame]:
    """
    Return up to n example players per role (arbitrary ordering).
    """
    examples: Dict[str, pd.DataFrame] = {}
    for role in sorted(
        r for r in feats_with_roles["ROLE_NAME"].dropna().unique().tolist()
    ):
        sub = feats_with_roles[feats_with_roles["ROLE_NAME"] == role][
            ["PLAYER_NAME", "SEASON", "TEAM_ID", "HEIGHT", "WEIGHT", "POSITION_LISTED"]
        ].drop_duplicates()
        examples[role] = sub.head(n)
    return examples

