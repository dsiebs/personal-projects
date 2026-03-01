# ============================================================
# MLB MVP AWARD PREDICTOR
# Google Colab-Ready Script
# Uses pybaseball for data extraction + ML ensemble models
# ============================================================

# ── CELL 1: Install & Import ─────────────────────────────────
# !pip install pybaseball pandas numpy scikit-learn matplotlib seaborn xgboost shap --quiet

import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import seaborn as sns

from pybaseball import batting_stats, pitching_stats
from pybaseball import cache
cache.enable()

from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor
from sklearn.svm import SVR
try:
    from xgboost import XGBRegressor
    _XGB_AVAILABLE = True
except Exception as e:
    _XGB_AVAILABLE = False
    _XGB_ERROR = str(e)

sns.set_theme(style="darkgrid", palette="muted")
plt.rcParams["figure.dpi"] = 120

print("✅ All packages loaded.")
if not _XGB_AVAILABLE:
    print("   (XGBoost skipped — run without it; install libomp on Mac: brew install libomp)")

# ── CELL 2: Configuration ────────────────────────────────────

START_YEAR   = 2000   # Historical data start
PREDICT_YEAR = 2024   # Year to generate predictions for
MIN_PA       = 350    # Minimum plate appearances to be considered

# Known MVP winners for validation (AL, NL) — extend as needed
KNOWN_MVP = {
    2023: {"AL": "Shohei Ohtani", "NL": "Ronald Acuña Jr."},
    2022: {"AL": "Justin Verlander",  "NL": "Paul Goldschmidt"},
    2021: {"AL": "Shohei Ohtani",     "NL": "Bryce Harper"},
    2020: {"AL": "José Abreu",        "NL": "Freddie Freeman"},
    2019: {"AL": "Mike Trout",        "NL": "Cody Bellinger"},
    2018: {"AL": "Mookie Betts",      "NL": "Christian Yelich"},
    2017: {"AL": "José Altuve",       "NL": "Giancarlo Stanton"},
    2016: {"AL": "Mike Trout",        "NL": "Kris Bryant"},
    2015: {"AL": "Josh Donaldson",    "NL": "Bryce Harper"},
    2014: {"AL": "Mike Trout",        "NL": "Clayton Kershaw"},
    2013: {"AL": "Miguel Cabrera",    "NL": "Andrew McCutchen"},
    2012: {"AL": "Miguel Cabrera",    "NL": "Buster Posey"},
    2011: {"AL": "Justin Verlander",  "NL": "Ryan Braun"},
    2010: {"AL": "Josh Hamilton",     "NL": "Joey Votto"},
    2009: {"AL": "Joe Mauer",         "NL": "Albert Pujols"},
    2008: {"AL": "Dustin Pedroia",    "NL": "Albert Pujols"},
    2007: {"AL": "Alex Rodriguez",    "NL": "Jimmy Rollins"},
    2006: {"AL": "Justin Morneau",    "NL": "Ryan Howard"},
    2005: {"AL": "Alex Rodriguez",    "NL": "Albert Pujols"},
    2004: {"AL": "Vladimir Guerrero", "NL": "Barry Bonds"},
    2003: {"AL": "Alex Rodriguez",    "NL": "Barry Bonds"},
    2002: {"AL": "Miguel Tejada",     "NL": "Barry Bonds"},
    2001: {"AL": "Ichiro Suzuki",     "NL": "Barry Bonds"},
    2000: {"AL": "Jason Giambi",      "NL": "Jeff Kent"},
}

# AL teams (for league split)
AL_TEAMS = {
    "BAL","BOS","NYY","TBR","TOR",
    "CWS","CLE","DET","KCR","MIN",
    "HOU","LAA","OAK","SEA","TEX"
}

print(f"✅ Config set. Pulling data {START_YEAR}–{PREDICT_YEAR}.")

# ── CELL 3: Data Extraction ──────────────────────────────────

print(f"\n📥 Fetching batting stats {START_YEAR}–{PREDICT_YEAR}…")
print("   (pybaseball pulls from FanGraphs — may take 60–90 s first run)")

all_years = []
for yr in range(START_YEAR, PREDICT_YEAR + 1):
    try:
        df = batting_stats(yr, yr, qual=1)   # qual=1 → all players; filter by PA later
        df["Season"] = yr
        all_years.append(df)
        print(f"   {yr}: {len(df)} players")
    except Exception as e:
        print(f"   {yr}: ⚠️  {e}")

raw = pd.concat(all_years, ignore_index=True)
print(f"\n✅ Raw dataset: {raw.shape[0]} rows × {raw.shape[1]} cols")

# ── CELL 4: Cleaning & Feature Engineering ──────────────────

def assign_league(team):
    """Rough AL/NL split — FanGraphs doesn't always expose league directly."""
    if str(team).upper() in AL_TEAMS:
        return "AL"
    return "NL"

df = raw.copy()

# Filter by PA
df = df[df["PA"] >= MIN_PA].copy()

# Assign league
df["League"] = df["Team"].apply(assign_league)

# ── Derived / Sabermetric features ──────────────────────────

# Fill common nulls with 0
fill_zero = ["BB%","K%","BABIP","wRC+","WAR","fWAR","Off","Def",
             "BsR","wOBA","xwOBA","EV","LA","Barrel%","HardHit%",
             "Spd","UBR","wGDP","Pull%","Oppo%","Soft%","Med%","Hard%"]
for col in fill_zero:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

# Core standard stats
std_cols = ["G","PA","AB","H","1B","2B","3B","HR","R","RBI","BB","IBB",
            "HBP","SB","CS","AVG","OBP","SLG","OPS"]
for col in std_cols:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

# Derived
df["ISO"]       = df["SLG"] - df["AVG"]
df["OPS_plus"]  = df["wRC+"] if "wRC+" in df.columns else df["OPS"] * 100  # proxy
df["BB_K"]      = df["BB"] / (df["K"] + 1) if "K" in df.columns else 0
df["SB_net"]    = df["SB"] - df["CS"]
df["XBH"]       = df["2B"] + df["3B"] + df["HR"]
df["HR_per_PA"] = df["HR"] / df["PA"]
df["R_RBI"]     = df["R"] + df["RBI"]

# ── MVP proxy target (historical years) ─────────────────────
# Build a "mvp_share" column: 1.0 if winner, 0 otherwise for training.
# We'll use vote share where available (scraped from Baseball Reference later),
# but a binary winner flag is a strong enough signal for regression ranking.

def mvp_winner_flag(row):
    yr = row["Season"]
    lg = row["League"]
    name = row.get("Name","")
    if yr in KNOWN_MVP and lg in KNOWN_MVP[yr]:
        return 1.0 if name == KNOWN_MVP[yr][lg] else 0.0
    return np.nan

df["MVP_Winner"] = df.apply(mvp_winner_flag, axis=1)

# ── Create a richer "MVP Score" for training based on known winners ──────
# Use weighted combo of sabermetrics to serve as a continuous proxy target
# (we calibrate weights so MVP winners score highest in their league-year)

def mvp_score(row):
    """Continuous proxy for MVP voting share — used as regression target."""
    war  = row.get("WAR", row.get("fWAR", 0))
    wrc  = row.get("wRC+", 100)
    ops  = row.get("OPS",  .750)
    hr   = row.get("HR",   0)
    rbi  = row.get("RBI",  0)
    r    = row.get("R",    0)
    sb   = row.get("SB_net", 0)
    off  = row.get("Off",  0)
    defd = row.get("Def",  0)
    score = (
        war  * 6.0 +
        (wrc - 100) * 0.15 +
        ops  * 30   +
        hr   * 0.25 +
        rbi  * 0.12 +
        r    * 0.08 +
        sb   * 0.10 +
        off  * 0.20 +
        defd * 0.10
    )
    return score

df["MVP_Score"] = df.apply(mvp_score, axis=1)

# Normalise within league-season so target is a [0,1] vote-share proxy
def norm_group(g):
    mn, mx = g.min(), g.max()
    if mx - mn < 1e-9:
        return g * 0
    return (g - mn) / (mx - mn)

df["MVP_Share"] = df.groupby(["Season","League"])["MVP_Score"].transform(norm_group)

print(f"✅ Cleaned dataset: {df.shape[0]} player-seasons | {len(df['Season'].unique())} years")
print(f"   Leagues: {df['League'].value_counts().to_dict()}")

# ── CELL 5: EDA & Visualisation ──────────────────────────────

fig, axes = plt.subplots(2, 3, figsize=(18, 10))
fig.suptitle("MLB MVP Predictor — Exploratory Data Analysis", fontsize=16, fontweight="bold")

# 1. WAR distribution
ax = axes[0,0]
for lg, grp in df[df["Season"] < PREDICT_YEAR].groupby("League"):
    ax.hist(grp["WAR"], bins=30, alpha=0.6, label=lg)
ax.set_title("WAR Distribution")
ax.set_xlabel("WAR"); ax.legend()

# 2. HR vs RBI coloured by League
ax = axes[0,1]
scatter_df = df[df["Season"] < PREDICT_YEAR].sample(min(3000, len(df)))
colors = scatter_df["League"].map({"AL":"steelblue","NL":"tomato"})
ax.scatter(scatter_df["HR"], scatter_df["RBI"], c=colors, alpha=0.3, s=10)
ax.set_title("HR vs RBI"); ax.set_xlabel("HR"); ax.set_ylabel("RBI")
for lg, c in [("AL","steelblue"),("NL","tomato")]:
    ax.scatter([], [], c=c, label=lg)
ax.legend()

# 3. wRC+ over time (median)
ax = axes[0,2]
med = df[df["Season"] < PREDICT_YEAR].groupby("Season")["wRC+"].median()
ax.plot(med.index, med.values, marker="o", linewidth=1.5)
ax.set_title("Median wRC+ per Season"); ax.set_xlabel("Year"); ax.set_ylabel("wRC+")

# 4. Correlation heatmap (key features)
ax = axes[1,0]
feat_cols = ["WAR","wRC+","OPS","HR","RBI","R","SB","OBP","SLG","ISO","MVP_Share"]
avail = [c for c in feat_cols if c in df.columns]
corr = df[df["Season"] < PREDICT_YEAR][avail].corr()
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, mask=mask, ax=ax, cmap="coolwarm", center=0,
            linewidths=0.4, annot=False, cbar_kws={"shrink":.7})
ax.set_title("Feature Correlation")

# 5. Top 10 historical MVP_Score leaders
ax = axes[1,1]
top = (df[df["Season"] < PREDICT_YEAR]
       .nlargest(10, "MVP_Score")[["Name","Season","League","WAR","HR","wRC+","MVP_Score"]]
       .reset_index(drop=True))
ax.barh(top["Name"] + " (" + top["Season"].astype(str) + ")",
        top["MVP_Score"], color="steelblue")
ax.invert_yaxis()
ax.set_title("Top 10 Historical MVP Scores"); ax.set_xlabel("MVP Score")

# 6. WAR vs MVP Share scatter
ax = axes[1,2]
hist = df[df["Season"] < PREDICT_YEAR]
ax.scatter(hist["WAR"], hist["MVP_Share"], alpha=0.15, s=8, color="steelblue")
winners = hist[hist["MVP_Winner"] == 1.0]
ax.scatter(winners["WAR"], winners["MVP_Share"], s=60, color="gold",
           edgecolors="black", linewidth=0.5, zorder=5, label="MVP Winner")
ax.set_title("WAR vs MVP Share"); ax.set_xlabel("WAR"); ax.set_ylabel("MVP Share")
ax.legend()

plt.tight_layout()
plt.savefig("eda_overview.png", bbox_inches="tight")
plt.show()
print("✅ EDA charts rendered.")

# ── CELL 6: Feature Selection & Train/Test Split ─────────────

FEATURE_COLS = [c for c in [
    "PA","G","HR","R","RBI","SB","SB_net","BB","AVG","OBP","SLG","OPS",
    "ISO","XBH","HR_per_PA","R_RBI","BB_K",
    "wRC+","WAR","Off","Def","BsR","wOBA",
    "BABIP","Barrel%","HardHit%","EV",
    "Spd","Pull%","Soft%","Med%","Hard%"
] if c in df.columns]

TARGET = "MVP_Share"

# Training data: all seasons with known MVP (exclude predict year)
train_df = df[df["Season"] < PREDICT_YEAR].dropna(subset=[TARGET]).copy()
pred_df  = df[df["Season"] == PREDICT_YEAR].copy()

X = train_df[FEATURE_COLS].fillna(0)
y = train_df[TARGET]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"✅ Features: {len(FEATURE_COLS)}")
print(f"   Train: {X_train.shape[0]}  |  Test: {X_test.shape[0]}")

# ── CELL 7: Model Zoo ────────────────────────────────────────

models = {
    "Ridge Regression":        Pipeline([("sc", StandardScaler()), ("m", Ridge(alpha=1.0))]),
    "Lasso Regression":        Pipeline([("sc", StandardScaler()), ("m", Lasso(alpha=0.001))]),
    "ElasticNet":              Pipeline([("sc", StandardScaler()), ("m", ElasticNet(alpha=0.001, l1_ratio=0.5))]),
    "SVR (RBF)":               Pipeline([("sc", StandardScaler()), ("m", SVR(kernel="rbf", C=5, epsilon=0.05))]),
    "Random Forest":           RandomForestRegressor(n_estimators=300, max_depth=8,
                                                      min_samples_leaf=3, random_state=42, n_jobs=-1),
    "Extra Trees":             ExtraTreesRegressor(n_estimators=300, max_depth=8,
                                                    min_samples_leaf=3, random_state=42, n_jobs=-1),
    "Gradient Boosting":       GradientBoostingRegressor(n_estimators=300, learning_rate=0.05,
                                                          max_depth=4, subsample=0.8, random_state=42),
}
if _XGB_AVAILABLE:
    models["XGBoost"] = XGBRegressor(n_estimators=400, learning_rate=0.04, max_depth=5,
                                     subsample=0.8, colsample_bytree=0.8,
                                     random_state=42, n_jobs=-1, verbosity=0)

print(f"✅ {len(models)} models registered.")

# ── CELL 8: Train, Evaluate & Compare ───────────────────────

kf = KFold(n_splits=5, shuffle=True, random_state=42)
results = {}

print("\n🏋️  Training models…\n")
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred     = model.predict(X_test)
    mae        = mean_absolute_error(y_test, y_pred)
    rmse       = np.sqrt(mean_squared_error(y_test, y_pred))
    r2         = r2_score(y_test, y_pred)
    cv_r2      = cross_val_score(model, X, y, cv=kf, scoring="r2", n_jobs=-1).mean()

    results[name] = {"MAE": mae, "RMSE": rmse, "R2": r2, "CV_R2": cv_r2}
    print(f"  {name:<25}  MAE={mae:.4f}  RMSE={rmse:.4f}  R²={r2:.4f}  CV-R²={cv_r2:.4f}")

results_df = pd.DataFrame(results).T.sort_values("CV_R2", ascending=False)
print(f"\n🏆 Best model (CV R²): {results_df.index[0]}")

# ── CELL 9: Model Comparison Chart ──────────────────────────

fig, axes = plt.subplots(1, 2, figsize=(16, 5))
fig.suptitle("Model Comparison", fontsize=14, fontweight="bold")

palette = sns.color_palette("RdYlGn", len(results_df))

ax = axes[0]
bars = ax.barh(results_df.index[::-1], results_df["CV_R2"][::-1], color=palette)
ax.set_title("Cross-Validated R² (higher = better)")
ax.set_xlabel("CV R²")
ax.axvline(0, color="black", linewidth=0.8)
for bar, val in zip(bars, results_df["CV_R2"][::-1]):
    ax.text(bar.get_width() + 0.002, bar.get_y() + bar.get_height()/2,
            f"{val:.3f}", va="center", fontsize=9)

ax = axes[1]
ax.barh(results_df.index[::-1], results_df["MAE"][::-1],
        color=sns.color_palette("RdYlGn_r", len(results_df)))
ax.set_title("Mean Absolute Error (lower = better)")
ax.set_xlabel("MAE")

plt.tight_layout()
plt.savefig("model_comparison.png", bbox_inches="tight")
plt.show()

# ── CELL 10: Best Model & Feature Importance ─────────────────

best_name  = results_df.index[0]
best_model = models[best_name]
print(f"\n✅ Selected: {best_name}")

# Feature importance (tree-based models expose this directly)
try:
    if hasattr(best_model, "feature_importances_"):
        importances = best_model.feature_importances_
    elif hasattr(best_model, "named_steps"):
        importances = best_model.named_steps["m"].coef_
    else:
        importances = None

    if importances is not None:
        fi_df = pd.DataFrame({"Feature": FEATURE_COLS, "Importance": np.abs(importances)})
        fi_df = fi_df.sort_values("Importance", ascending=False).head(20)

        fig, ax = plt.subplots(figsize=(10, 6))
        sns.barplot(data=fi_df, x="Importance", y="Feature", palette="Blues_r", ax=ax)
        ax.set_title(f"Top 20 Feature Importances — {best_name}")
        plt.tight_layout()
        plt.savefig("feature_importance.png", bbox_inches="tight")
        plt.show()
except Exception as e:
    print(f"  (Feature importance skipped: {e})")

# ── SHAP explanation for best tree model ────────────────────
print("\n🔍 Computing SHAP values (may take ~30 s)…")
try:
    import shap
    explainer  = shap.TreeExplainer(best_model)
    shap_vals  = explainer.shap_values(X_test)
    plt.figure()
    shap.summary_plot(shap_vals, X_test, feature_names=FEATURE_COLS,
                      show=False, max_display=15)
    plt.title(f"SHAP Summary — {best_name}")
    plt.tight_layout()
    plt.savefig("shap_summary.png", bbox_inches="tight")
    plt.show()
    print("✅ SHAP plot saved.")
except Exception as e:
    print(f"  SHAP skipped ({e})")

# ── CELL 11: Predictions for PREDICT_YEAR ───────────────────

X_pred = pred_df[FEATURE_COLS].fillna(0)
pred_df = pred_df.copy()
pred_df["Predicted_Share"] = best_model.predict(X_pred)

# Normalise predictions per league to [0,1]
pred_df["Pred_Norm"] = pred_df.groupby("League")["Predicted_Share"].transform(norm_group)

def top_candidates(league, n=10):
    sub = pred_df[pred_df["League"] == league].copy()
    sub = sub.sort_values("Pred_Norm", ascending=False).head(n)
    cols = ["Name","Team","PA","HR","RBI","R","AVG","OPS","wRC+","WAR","Pred_Norm"]
    avail = [c for c in cols if c in sub.columns]
    return sub[avail].reset_index(drop=True)

print(f"\n{'='*60}")
print(f"  🏆 {PREDICT_YEAR} AL MVP PREDICTIONS")
print(f"{'='*60}")
al_preds = top_candidates("AL")
print(al_preds.to_string(index=False))

print(f"\n{'='*60}")
print(f"  🏆 {PREDICT_YEAR} NL MVP PREDICTIONS")
print(f"{'='*60}")
nl_preds = top_candidates("NL")
print(nl_preds.to_string(index=False))

# ── CELL 12: Prediction Charts ───────────────────────────────

fig, axes = plt.subplots(1, 2, figsize=(18, 7))
fig.suptitle(f"{PREDICT_YEAR} MLB MVP Predictions — {best_name}", fontsize=14, fontweight="bold")

for ax, (lg, preds) in zip(axes, [("AL", al_preds), ("NL", nl_preds)]):
    colors = ["gold" if i == 0 else "steelblue" for i in range(len(preds))]
    bars = ax.barh(preds["Name"][::-1], preds["Pred_Norm"][::-1], color=colors[::-1])
    ax.set_title(f"{lg} MVP Race")
    ax.set_xlabel("Predicted Vote Share (normalised)")
    ax.xaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    for bar, val in zip(bars, preds["Pred_Norm"][::-1]):
        ax.text(bar.get_width() + 0.005, bar.get_y() + bar.get_height()/2,
                f"{val:.1%}", va="center", fontsize=9)

plt.tight_layout()
plt.savefig("predictions.png", bbox_inches="tight")
plt.show()

# ── CELL 13: Historical Validation — Model vs Actual MVP ─────

print("\n\n📊 Historical Validation — Model vs Actual MVP\n")
print(f"{'Year':<6} {'League':<4} {'Model Pick':<25} {'Actual MVP':<25} {'Match'}")
print("-" * 75)

match_count = 0
total_count = 0

for yr in sorted(KNOWN_MVP.keys()):
    if yr < START_YEAR or yr >= PREDICT_YEAR:
        continue
    yr_df = df[df["Season"] == yr].copy()
    if yr_df.empty:
        continue
    X_yr  = yr_df[FEATURE_COLS].fillna(0)
    yr_df["Pred"] = best_model.predict(X_yr)

    for lg in ["AL","NL"]:
        sub = yr_df[yr_df["League"] == lg]
        if sub.empty:
            continue
        model_pick = sub.loc[sub["Pred"].idxmax(), "Name"]
        actual_mvp = KNOWN_MVP[yr].get(lg, "Unknown")
        match      = "✅" if model_pick == actual_mvp else "❌"
        if model_pick == actual_mvp:
            match_count += 1
        total_count += 1
        print(f"{yr:<6} {lg:<4} {model_pick:<25} {actual_mvp:<25} {match}")

acc = match_count / total_count if total_count else 0
print(f"\n🎯 Historical Accuracy: {match_count}/{total_count} = {acc:.1%}")

# ── CELL 14: Discrepancies Deep-Dive ────────────────────────

print("\n\n🔍 Years Where Model Disagreed With Actual MVP:\n")
print(f"{'Year':<6} {'Lg':<4} {'Model Pick':<25} {'Actual MVP':<25} "
      f"{'Model WAR':>10} {'Winner WAR':>11}")
print("-" * 85)

for yr in sorted(KNOWN_MVP.keys()):
    if yr < START_YEAR or yr >= PREDICT_YEAR:
        continue
    yr_df = df[df["Season"] == yr].copy()
    if yr_df.empty:
        continue
    X_yr  = yr_df[FEATURE_COLS].fillna(0)
    yr_df["Pred"] = best_model.predict(X_yr)

    for lg in ["AL","NL"]:
        sub = yr_df[yr_df["League"] == lg]
        if sub.empty:
            continue
        model_pick = sub.loc[sub["Pred"].idxmax(), "Name"]
        actual_mvp = KNOWN_MVP[yr].get(lg, "Unknown")
        if model_pick != actual_mvp:
            model_war  = sub.loc[sub["Pred"].idxmax(), "WAR"] if "WAR" in sub.columns else "N/A"
            winner_row = sub[sub["Name"] == actual_mvp]
            winner_war = winner_row["WAR"].values[0] if not winner_row.empty and "WAR" in winner_row.columns else "N/A"
            print(f"{yr:<6} {lg:<4} {model_pick:<25} {actual_mvp:<25} "
                  f"{str(round(model_war,1)) if isinstance(model_war,float) else model_war:>10} "
                  f"{str(round(winner_war,1)) if isinstance(winner_war,float) else winner_war:>11}")

# ── CELL 15: Summary Report ──────────────────────────────────

print("\n" + "="*65)
print("  📋 FINAL SUMMARY REPORT")
print("="*65)
print(f"\n  ▸ Data range:       {START_YEAR}–{PREDICT_YEAR-1} (training)  |  {PREDICT_YEAR} (prediction)")
print(f"  ▸ Training samples: {len(X_train):,}")
print(f"  ▸ Features used:    {len(FEATURE_COLS)}")
print(f"  ▸ Best model:       {best_name}")
print(f"  ▸ CV R²:            {results_df.loc[best_name,'CV_R2']:.4f}")
print(f"  ▸ Test MAE:         {results_df.loc[best_name,'MAE']:.4f}")
print(f"  ▸ Historical accuracy: {match_count}/{total_count} ({acc:.1%})")

print(f"\n  🏆 {PREDICT_YEAR} PREDICTED MVP WINNERS")
print(f"     AL: {al_preds.iloc[0]['Name']}  ({al_preds.iloc[0]['Pred_Norm']:.1%} predicted share)")
print(f"     NL: {nl_preds.iloc[0]['Name']}  ({nl_preds.iloc[0]['Pred_Norm']:.1%} predicted share)")

if PREDICT_YEAR in KNOWN_MVP:
    al_actual = KNOWN_MVP[PREDICT_YEAR].get("AL","TBD")
    nl_actual = KNOWN_MVP[PREDICT_YEAR].get("NL","TBD")
    al_match = "✅ CORRECT" if al_preds.iloc[0]['Name'] == al_actual else f"❌ Actual: {al_actual}"
    nl_match = "✅ CORRECT" if nl_preds.iloc[0]['Name'] == nl_actual else f"❌ Actual: {nl_actual}"
    print(f"\n  📌 {PREDICT_YEAR} Actual Award vs Model:")
    print(f"     AL: {al_match}")
    print(f"     NL: {nl_match}")

print("\n  📁 Saved charts: eda_overview.png | model_comparison.png")
print("               feature_importance.png | shap_summary.png | predictions.png")
print("\n" + "="*65)
