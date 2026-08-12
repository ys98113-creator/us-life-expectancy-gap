"""
Generates the static figures embedded in README.md.

Run after us_life_expectancy_gap_analysis.py logic changes, or after the
underlying CSV is rebuilt, to refresh figures/*.png.
"""

import json
import urllib.request

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from xgboost import XGBRegressor
import plotly.express as px

# ============================================================
# Shared style tokens (validated palette — see dataviz skill)
# ============================================================

SURFACE = "#fcfcfb"
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRIDLINE = "#e1e0d9"
BASELINE = "#c3c2b7"

BLUE = "#2a78d6"
RED = "#e34948"
GRAY = "#c3c2b7"

BLUE_RAMP = [
    "#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec",
    "#5598e7", "#3987e5", "#2a78d6", "#256abf", "#1c5cab",
    "#184f95", "#104281", "#0d366b",
]

plt.rcParams.update(
    {
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "text.color": INK_PRIMARY,
        "axes.edgecolor": BASELINE,
        "axes.labelcolor": INK_SECONDARY,
        "xtick.color": INK_MUTED,
        "ytick.color": INK_MUTED,
        "font.family": "sans-serif",
        "font.size": 12,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.spines.left": False,
    }
)

FIG_DIR = "figures"


# ============================================================
# Data + models (mirrors us_life_expectancy_gap_analysis.py)
# ============================================================

df = pd.read_csv("us_county_food_environment.csv")
df = df.replace([-9999, -8888], np.nan)

OUTCOME = "life_expectancy"
PREDICTORS = [
    "fast_food_density",
    "food_desert_pct",
    "snap_participation_pct",
    "farmers_market_density",
    "adult_obesity_pct",
    "diabetes_prevalence_pct",
    "physical_inactivity_pct",
    "uninsured_pct",
    "median_household_income",
    "income_inequality_ratio",
    "child_poverty_pct",
    "primary_care_physicians_per_1000",
]

model_df = df[["fips", "county_name", "state", OUTCOME] + PREDICTORS].dropna().copy()

X_train, X_test, y_train, y_test = train_test_split(
    model_df[PREDICTORS], model_df[OUTCOME], test_size=0.2, random_state=42
)

ols_model = sm.OLS(y_train, sm.add_constant(X_train)).fit()
ols_pred = ols_model.predict(sm.add_constant(X_test, has_constant="add"))

rf_model = RandomForestRegressor(n_estimators=500, random_state=42, n_jobs=-1)
rf_model.fit(X_train, y_train)
rf_pred = rf_model.predict(X_test)

xgb_model = XGBRegressor(n_estimators=500, max_depth=4, learning_rate=0.05, random_state=42)
xgb_model.fit(X_train, y_train)
xgb_pred = xgb_model.predict(X_test)

r2 = {
    "OLS": r2_score(y_test, ols_pred),
    "Random Forest": r2_score(y_test, rf_pred),
    "XGBoost": r2_score(y_test, xgb_pred),
}


# ============================================================
# Figure 1 — county choropleth of life expectancy
# ============================================================

GEOJSON_URL = "https://raw.githubusercontent.com/plotly/datasets/master/geojson-counties-fips.json"
GEOJSON_CACHE = "county_data_cache/geojson-counties-fips.json"

try:
    with open(GEOJSON_CACHE) as f:
        counties_geojson = json.load(f)
except FileNotFoundError:
    with urllib.request.urlopen(GEOJSON_URL) as resp:
        raw = resp.read()
    with open(GEOJSON_CACHE, "wb") as f:
        f.write(raw)
    counties_geojson = json.loads(raw)

map_df = df.dropna(subset=[OUTCOME]).copy()
map_df["fips"] = map_df["fips"].astype(int).astype(str).str.zfill(5)

blue_scale = [[i / (len(BLUE_RAMP) - 1), c] for i, c in enumerate(BLUE_RAMP)]

fig = px.choropleth(
    map_df,
    geojson=counties_geojson,
    locations="fips",
    color=OUTCOME,
    color_continuous_scale=blue_scale,
    scope="usa",
    labels={OUTCOME: "Life expectancy (yrs)"},
)
fig.update_traces(marker_line_width=0)
fig.update_layout(
    title=dict(
        text="Life Expectancy by U.S. County",
        font=dict(size=22, color=INK_PRIMARY, family="Arial, sans-serif"),
        x=0.5,
    ),
    paper_bgcolor=SURFACE,
    geo=dict(bgcolor=SURFACE, lakecolor=SURFACE),
    coloraxis_colorbar=dict(title="Years", tickfont=dict(color=INK_SECONDARY)),
    margin=dict(l=0, r=0, t=60, b=0),
    font=dict(family="Arial, sans-serif", color=INK_SECONDARY),
)
fig.write_image(f"{FIG_DIR}/county_life_expectancy_map.png", width=1400, height=900, scale=2)


# ============================================================
# Figure 2 — model comparison (OLS vs Random Forest vs XGBoost)
# ============================================================

models = ["OLS", "Random Forest", "XGBoost"]
scores = [r2[m] for m in models]
colors = [GRAY, BLUE, BLUE]

fig2, ax2 = plt.subplots(figsize=(7, 4.5))
bars = ax2.bar(models, scores, color=colors, width=0.55, zorder=3)

ax2.set_ylim(0, max(scores) * 1.25)
ax2.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))
ax2.grid(axis="y", color=GRIDLINE, linewidth=1, zorder=0)
ax2.set_axisbelow(True)
ax2.tick_params(axis="x", length=0)
ax2.tick_params(axis="y", length=0)
ax2.set_ylabel("Test R² (held-out counties)")
ax2.set_title("Tree-Based Models Predict Better Than OLS", color=INK_PRIMARY, fontsize=14, pad=14)

for bar, score in zip(bars, scores):
    ax2.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + max(scores) * 0.02,
        f"{score:.3f}",
        ha="center",
        va="bottom",
        color=INK_PRIMARY,
        fontsize=11,
    )

fig2.tight_layout()
fig2.savefig(f"{FIG_DIR}/model_comparison.png", dpi=150)
plt.close(fig2)


# ============================================================
# Figure 3 — correlation with life expectancy (diverging)
# ============================================================

correlations = (
    df[PREDICTORS + [OUTCOME]].corr()[OUTCOME].drop(OUTCOME).sort_values()
)

fig3, ax3 = plt.subplots(figsize=(9, 6))
bar_colors = [RED if v < 0 else BLUE for v in correlations.values]
bars3 = ax3.barh(correlations.index, correlations.values, color=bar_colors, height=0.6, zorder=3)

pad = 0.12
ax3.set_xlim(correlations.min() - pad, correlations.max() + pad)
ax3.axvline(0, color=BASELINE, linewidth=1, zorder=2)
ax3.grid(axis="x", color=GRIDLINE, linewidth=1, zorder=0)
ax3.set_axisbelow(True)
ax3.tick_params(axis="y", length=0)
ax3.tick_params(axis="x", length=0)
ax3.set_xlabel("Correlation with life expectancy")
ax3.set_title(
    "Correlation With Life Expectancy",
    color=INK_PRIMARY,
    fontsize=14,
    pad=14,
)

for bar, val in zip(bars3, correlations.values):
    offset = 0.01 if val >= 0 else -0.01
    ha = "left" if val >= 0 else "right"
    ax3.text(
        val + offset, bar.get_y() + bar.get_height() / 2, f"{val:+.2f}",
        va="center", ha=ha, color=INK_PRIMARY, fontsize=10,
    )

fig3.tight_layout()
fig3.savefig(f"{FIG_DIR}/correlation_with_life_expectancy.png", dpi=150)
plt.close(fig3)


# ============================================================
# Figure 4 — Random Forest feature importance (sequential)
# ============================================================

rf_importance = pd.Series(rf_model.feature_importances_, index=PREDICTORS).sort_values()

norm = rf_importance.values / rf_importance.values.max()
ramp_colors = [BLUE_RAMP[int(n * (len(BLUE_RAMP) - 1))] for n in norm]

fig4, ax4 = plt.subplots(figsize=(8, 6))
bars4 = ax4.barh(rf_importance.index, rf_importance.values, color=ramp_colors, height=0.6, zorder=3)

ax4.set_xlim(0, rf_importance.values.max() * 1.15)
ax4.grid(axis="x", color=GRIDLINE, linewidth=1, zorder=0)
ax4.set_axisbelow(True)
ax4.tick_params(axis="y", length=0)
ax4.tick_params(axis="x", length=0)
ax4.set_xlabel("Relative importance")
ax4.set_title("Random Forest Feature Importance", color=INK_PRIMARY, fontsize=14, pad=14)

for bar, val in zip(bars4, rf_importance.values):
    ax4.text(
        val + rf_importance.values.max() * 0.015,
        bar.get_y() + bar.get_height() / 2,
        f"{val:.2f}",
        va="center", ha="left", color=INK_PRIMARY, fontsize=10,
    )

fig4.tight_layout()
fig4.savefig(f"{FIG_DIR}/rf_feature_importance.png", dpi=150)
plt.close(fig4)

print("Figures written to figures/:")
print(" - county_life_expectancy_map.png")
print(" - model_comparison.png")
print(" - correlation_with_life_expectancy.png")
print(" - rf_feature_importance.png")
