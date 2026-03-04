# NBA Role Clustering

This project clusters NBA players into modern, data-driven **playstyle roles** using unsupervised learning (K-Means) on play-by-play and box score statistics from recent seasons.

The core idea is to ignore a player's *listed position* and instead infer their role from how they actually play, along dimensions like usage, playmaking, shooting profile, rim pressure, defensive impact, and more.

The target role framework is the multi-part taxonomy you provided, including:

- **Primary offensive roles** (Offensive Hub / Do-It-All Engine, Pass-First Orchestrator, Secondary Creator / Combo Guard, Isolation Scorer, Three-Level Shot Creator)
- **Off-ball scoring roles** (Movement Shooter, Spot-Up Specialist, Slasher / Rim Pressure Wing, Transition Finisher, Microwave Scorer)
- **Frontcourt offensive roles** (Stretch Big, Interior Scoring Big, Short-Roll Playmaker, Rim Runner / Vertical Spacer)
- **Defensive roles** (Point-of-Attack Defender, Switchable Wing Defender, Rim Protector, Help-Side Roamer / Free Safety, Switch Big / Small-Ball 5)
- **Glue / hybrid roles** (Connector, Energy Wing / Chaos Creator, Rebounding Specialist)
- **Specialized micro-roles** (handled as optional tags rather than primary clusters)

The model does **not** use the official position labels (PG/SG/SF/PF/C) as input features, on purpose.

## Project Structure

- `requirements.txt` – Python dependencies.
- `src/`
  - `data_fetch.py` – Utilities to pull per-player season stats and advanced metrics from `nba_api`.
  - `features.py` – Feature engineering for usage, shooting profile, efficiency, play type frequencies, and defensive/impact stats.
  - `clustering.py` – Scaling, K-Means clustering, and model selection (e.g., silhouette score for K).
  - `roles.py` – Role schema, cluster summaries, and helper tools to inspect and label clusters.
- `notebooks/`
  - `nba_role_clustering.ipynb` – End-to-end workflow: fetch data, explore, cluster, summarize, and interpret clusters in terms of roles.

## Setup

From the project root:

```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

pip install -r requirements.txt
```

## Data Source

By default, the notebook uses the **official NBA.com stats** API via the `nba_api` package. It:

- Pulls **per-player per-season** box score and advanced stats for recent seasons (e.g., 2022–23 and 2023–24).
- Optionally pulls **play type** frequencies (PnR ball handler, PnR roll man, spot up, handoff, post up, transition, etc.) where available.
- Optionally integrates **on/off**-style impact metrics via external CSVs if you provide them.

Because `nba_api` depends on a live HTTP API, you should expect occasional rate limits or schema changes; the notebook includes small sleeps between calls, and the code is structured so you can cache intermediate CSV/Parquet files in the `data/` directory.

## Features (High-Level)

The feature set is designed to align with your role taxonomy. Examples include:

- **Offensive load & playmaking**
  - Usage % (USG%)
  - Assist % (AST%)
  - Turnover % (TOV%)
  - Assist-to-turnover ratio
  - Time of possession and touches (if available)
- **Shot profile & scoring**
  - True Shooting % (TS%)
  - 3PA rate, rim frequency, midrange frequency
  - Catch-and-shoot vs pull-up 3 frequency (if available)
  - Transition frequency and efficiency
  - Isolation frequency and efficiency (if available)
- **Big-man / play type context**
  - PnR ball-handler frequency & efficiency
  - PnR roll-man frequency & efficiency
  - Post-up frequency
  - Short-roll assists / secondary assists (approximate via AST%, touch location when available)
- **Defensive & rebounding**
  - Steal %, Block %
  - Offensive & defensive rebounding %
  - Rim protection proxies (blocks, contests, opponent rim FG% where available)
- **Impact**
  - On/off offensive and defensive rating (if you supply external impact metrics as CSV)
  - Net rating per 100 possessions for lineups that include the player (optional / advanced)

The notebook explicitly **drops listed position fields** before building the feature matrix used for clustering.

## Workflow

1. **Data download & caching**
   - Run the first section of `nba_role_clustering.ipynb` to fetch recent seasons of player stats via `nba_api`.
   - Data is cached to `data/` as Parquet/CSV so you can iterate without re-downloading.

2. **Feature engineering**
   - Build a player-season-level feature table with all engineered stats listed above.
   - Filter out extremely low-minute / low-game players to avoid noise.

3. **Exploratory data analysis**
   - Inspect distributions, correlations, and a few 2D projections (e.g., PCA) to get intuition for structure.

4. **Clustering**
   - Standardize features.
   - Run K-Means for a range of cluster counts (e.g., k = 8–24).
   - Choose K via silhouette score and a mix of **statistical fit** and **interpretability**.

5. **Role interpretation & naming**
   - For each cluster, examine:
     - Cluster-level means for all features.
     - Height/weight distributions (for interpretation only).
     - Example players closest to the cluster centroid.
   - Map each cluster to one (or a blend) of the roles in your framework (e.g., “Movement Shooter”, “Rim Runner / Vertical Spacer”, etc.).

6. **Outputs**
   - A table of players with assigned **cluster IDs and human-readable role label**.
   - Cluster-level summary statistics table.
   - Lists of example players for each role.

## Interpreting a Real Run

When you execute the notebook end-to-end on fresh data, you should:

- Inspect the **cluster summary table** and verify that feature patterns match your expectations for each role (e.g., high USG% + AST% + balanced shot chart ≈ Offensive Hub; high 3PA rate + low usage ≈ Spot-Up Specialist).
- Check **example players** from each cluster and see if they qualitatively align with your labels.
- Iterate on:
  - The **feature set** (add/remove features that muddy or sharpen distinctions).
  - The **number of clusters K** (too few = roles merged; too many = micro-roles / noise).
  - The **subset of players** (e.g., only players above a minute threshold, or separate guard/wing/big clustering using *height only* as a very weak positional prior).

This repo is meant to be an experimentation sandbox; feel free to fork the notebook, change feature definitions, and adjust the target number of roles to see how the taxonomy emerges from different perspectives.

