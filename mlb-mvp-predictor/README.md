# MLB MVP Predictor

A machine learning project that predicts MLB MVP award winners using historical batting statistics and an ensemble of regression models.

## Overview

This script fetches batting data from FanGraphs via [pybaseball](https://github.com/jldbc/pybaseball), engineers sabermetric features, trains multiple ML models, and generates MVP predictions for both the American League (AL) and National League (NL). It includes exploratory data analysis, model comparison, SHAP explainability, and historical validation against actual MVP winners.

## Features

- **Data source**: FanGraphs batting stats (2000–present) via pybaseball
- **Feature engineering**: WAR, wRC+, OPS, ISO, barrel rate, and 30+ derived metrics
- **Models**: Ridge, Lasso, ElasticNet, SVR, Random Forest, Extra Trees, Gradient Boosting, XGBoost
- **Outputs**: EDA visualizations, model comparison charts, feature importance, SHAP summary, and prediction rankings

## Python setup

**Option A: Virtual environment (recommended)**

```bash
cd mlb-mvp-predictor
python3 -m venv .venv
source .venv/bin/activate   # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

*(SHAP is optional; the script runs without it and skips the SHAP plot if the package isn’t installed.)*

**Option B: Global install**

```bash
pip install pybaseball pandas numpy scikit-learn matplotlib seaborn xgboost
# Optional: pip install shap
```

## Running the project

**Script (terminal):**

```bash
# From project folder, with venv activated:
python mlb_mvp_predictor.py
```

**Notebook (Jupyter / VS Code / Cursor):**

1. Open `mlb_mvp_predictor.ipynb`.
2. Select the kernel: **Python (mlb-mvp-predictor)** or choose the interpreter at `.venv/bin/python`.
3. Run cells (Run All or run section by section).

**Configuration** (edit at the top of the script or in the notebook):

- `START_YEAR`: First year of historical data (default: 2000)
- `PREDICT_YEAR`: Year to generate predictions for (default: 2024)
- `MIN_PA`: Minimum plate appearances to qualify (default: 350)

The first run fetches data from FanGraphs and may take 60–90 seconds. Results are cached for faster subsequent runs.

## Output

The script produces:

- **eda_overview.png** — WAR distribution, HR vs RBI, wRC+ trends, correlation heatmap, top MVP scores, WAR vs MVP share
- **model_comparison.png** — Cross-validated R² and MAE across models
- **feature_importance.png** — Top 20 features for the best model
- **shap_summary.png** — SHAP values for model interpretability
- **predictions.png** — Top 10 AL and NL MVP candidates with predicted vote share

## Historical Accuracy

The model is validated against known MVP winners from 2000–2023. Accuracy varies by season; the script reports match rate and highlights years where the model disagreed with the actual winner.

## License

MIT
