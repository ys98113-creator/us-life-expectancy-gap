"""
U.S. County Life Expectancy Gap Analysis
=========================================

Research question
-----------------
What factors are associated with differences in life expectancy across
U.S. counties?

Project objective
-----------------
This project explores whether socioeconomic conditions, health behaviors,
health conditions, healthcare access, and the local food environment are
associated with county-level life expectancy.

The primary unit of analysis is the U.S. county. State-level comparisons
can be added later by aggregating county-level results.

Analysis workflow
-----------------
1. Load and inspect the data
2. Check and clean missing values
3. Explore variable distributions
4. Examine life expectancy across counties
5. Run correlation analysis
6. Estimate a multiple OLS regression
7. Check multicollinearity
8. Run regression diagnostics
9. Identify counties with higher/lower life expectancy than predicted
"""

# ============================================================
# 1. Import libraries
# ============================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm

from statsmodels.stats.outliers_influence import (
    variance_inflation_factor,
    OLSInfluence,
)
from statsmodels.stats.diagnostic import het_breuschpagan

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error
from xgboost import XGBRegressor


# ============================================================
# 2. Load the data
# ============================================================

# Update this path if your CSV is stored somewhere else.
DATA_PATH = (
    "/Users/iseunghwan/Desktop/us-life-expectancy-gap/"
    "us_county_food_environment.csv"
)

df = pd.read_csv(DATA_PATH)

print("Dataset shape:", df.shape)
print("\nDataset information:")
df.info()

print("\nDescriptive statistics:")
print(df.describe().T)


# ============================================================
# 3. Data quality and missing values
# ============================================================

# Identify missing observations.
missing = df.isna().sum().sort_values(ascending=False)

print("\nMissing values:")
print(missing[missing > 0])

# Some datasets use special numeric codes to represent missing
# or unavailable observations. These should not be treated as
# real measurements.
SPECIAL_MISSING_VALUES = [-9999, -8888]

df = df.replace(SPECIAL_MISSING_VALUES, np.nan)

print("\nMissing values after replacing special missing-value codes:")
print(df.isna().sum().sort_values(ascending=False).head(20))

# IMPORTANT:
# Do not automatically fill missing state/county names unless the
# correct values have been verified from the original data source.
#
# Example investigation:
# print(df[df["state"].isna()])
# print(df[df["county_name"].isna()])


# ============================================================
# 4. Define variables
# ============================================================

# Life expectancy is the dependent/outcome variable.
OUTCOME = "life_expectancy"

# Explanatory variables are grouped conceptually into:
# - Food environment
# - Health behaviors and conditions
# - Healthcare access
# - Economic conditions
# - Social assistance

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


# ============================================================
# 5. Exploratory data analysis
# ============================================================

print("\nLife expectancy summary:")
print(df[OUTCOME].describe())

# Distribution of life expectancy.
plt.figure(figsize=(7, 5))
sns.histplot(df[OUTCOME].dropna(), bins=30, kde=True)
plt.title("Distribution of Life Expectancy Across U.S. Counties")
plt.xlabel("Life Expectancy (years)")
plt.ylabel("Number of Counties")
plt.tight_layout()
plt.show()


# Examine distributions of numeric variables.
numeric_cols = df.select_dtypes(include="number").columns

for col in numeric_cols:
    print("=" * 60)
    print(col)
    print(df[col].describe())

    plt.figure(figsize=(6, 4))
    df[col].hist(bins=30)
    plt.title(f"Distribution of {col}")
    plt.xlabel(col)
    plt.ylabel("Number of Counties")
    plt.tight_layout()
    plt.show()


# ============================================================
# 6. Correlation analysis
# ============================================================

"""
Correlation analysis examines the pairwise linear relationship
between each explanatory variable and life expectancy.

Important:
Correlation describes association, not causation.
"""

correlations = (
    df[PREDICTORS + [OUTCOME]]
    .corr()[OUTCOME]
    .drop(OUTCOME)
    .sort_values()
)

print("\nCorrelations with life expectancy:")
print(correlations)

plt.figure(figsize=(8, 6))
correlations.plot(kind="barh")
plt.title("Correlation Between Explanatory Variables and Life Expectancy")
plt.xlabel("Correlation with Life Expectancy")
plt.axvline(0, linewidth=0.8)
plt.tight_layout()
plt.show()


# ============================================================
# 7. Example relationship: income and life expectancy
# ============================================================

plt.figure(figsize=(7, 5))
sns.scatterplot(
    data=df,
    x="median_household_income",
    y=OUTCOME,
    alpha=0.4,
    s=20,
)
plt.title("Median Household Income vs. Life Expectancy")
plt.xlabel("Median Household Income")
plt.ylabel("Life Expectancy (years)")
plt.tight_layout()
plt.show()

print("\nIncome/life expectancy correlation:")
print(df[["median_household_income", OUTCOME]].corr())


# ============================================================
# 8. Prepare data for OLS regression
# ============================================================

"""
The regression dataset retains county and state identifiers so that
model results can later be connected back to individual counties.
"""

MODEL_COLUMNS = [
    "county_name",
    "state",
    OUTCOME,
] + PREDICTORS

model_df = df[MODEL_COLUMNS].dropna().copy()

print("\nRegression dataset shape:", model_df.shape)

X = model_df[PREDICTORS]
X = sm.add_constant(X)

y = model_df[OUTCOME]


# ============================================================
# 9. Multiple OLS regression
# ============================================================

"""
OLS estimates the association between life expectancy and multiple
explanatory variables simultaneously.

Because this is observational county-level data, coefficients should
be interpreted as conditional associations rather than causal effects.
"""

model = sm.OLS(y, X).fit()

print("\nOLS Regression Results:")
print(model.summary())


# ============================================================
# 10. Multicollinearity: Variance Inflation Factor (VIF)
# ============================================================

"""
VIF evaluates whether explanatory variables are strongly related to
one another.

High VIF values can make individual regression coefficients unstable
and difficult to interpret.
"""

vif = pd.DataFrame()
vif["Variable"] = X.columns
vif["VIF"] = [
    variance_inflation_factor(X.values, i)
    for i in range(X.shape[1])
]

vif = vif.sort_values("VIF", ascending=False)

print("\nVariance Inflation Factors:")
print(vif)


# ============================================================
# 11. Regression diagnostics
# ============================================================

# Predicted values and residuals.
model_df["predicted_life_expectancy"] = model.predict(X)
model_df["residual"] = (
    model_df[OUTCOME] - model_df["predicted_life_expectancy"]
)

# Residuals vs. predicted values.
plt.figure(figsize=(7, 5))
plt.scatter(
    model_df["predicted_life_expectancy"],
    model_df["residual"],
    alpha=0.4,
)
plt.axhline(0, linewidth=0.8)
plt.xlabel("Predicted Life Expectancy")
plt.ylabel("Residual")
plt.title("Residuals vs. Predicted Life Expectancy")
plt.tight_layout()
plt.show()


# Breusch-Pagan test for heteroskedasticity.
bp_test = het_breuschpagan(model.resid, model.model.exog)

print("\nBreusch-Pagan test:")
print("LM statistic:", bp_test[0])
print("LM p-value:", bp_test[1])
print("F statistic:", bp_test[2])
print("F p-value:", bp_test[3])


# Q-Q plot for residual normality.
sm.qqplot(model.resid, line="45")
plt.title("Q-Q Plot of Regression Residuals")
plt.tight_layout()
plt.show()


# ============================================================
# 12. Influential observations: Cook's distance
# ============================================================

"""
Cook's distance helps identify observations that may have a large
influence on the estimated regression coefficients.

A high Cook's distance does not automatically mean that an observation
should be removed. It should be investigated to determine whether it
represents a data-quality problem, an unusual but legitimate county,
or an influential observation that meaningfully affects the model.
"""

influence = OLSInfluence(model)

model_df["cooks_d"] = influence.cooks_distance[0]

top_cooks = model_df.sort_values("cooks_d", ascending=False).head(10)

print("\nTop 10 observations by Cook's distance:")
print(
    top_cooks[
        ["county_name", "state", OUTCOME, "predicted_life_expectancy", "cooks_d"]
    ]
)


# ============================================================
# 13. Counties with higher/lower life expectancy than predicted
# ============================================================

"""
Residual = Observed Life Expectancy - Predicted Life Expectancy

Positive residual:
    The county has higher life expectancy than predicted by the model.

Negative residual:
    The county has lower life expectancy than predicted by the model.

This provides a useful second research question:
Which counties have unusually high or low life expectancy given
their observed socioeconomic, health, and healthcare characteristics?
"""

high_residual = (
    model_df
    .sort_values("residual", ascending=False)
    .head(10)
)

low_residual = (
    model_df
    .sort_values("residual")
    .head(10)
)

print("\nTop 10 counties with higher life expectancy than predicted:")
print(
    high_residual[
        [
            "county_name",
            "state",
            OUTCOME,
            "predicted_life_expectancy",
            "residual",
        ]
    ]
)

print("\nTop 10 counties with lower life expectancy than predicted:")
print(
    low_residual[
        [
            "county_name",
            "state",
            OUTCOME,
            "predicted_life_expectancy",
            "residual",
        ]
    ]
)


# ============================================================
# 14. Model summary
# ============================================================

print("\nModel performance:")
print(f"R-squared: {model.rsquared:.4f}")
print(f"Adjusted R-squared: {model.rsquared_adj:.4f}")
print(f"Number of observations: {int(model.nobs)}")


# ============================================================
# 15. Predictive comparison: OLS vs. Random Forest vs. XGBoost
# ============================================================

"""
Sections 6-14 use the full sample to describe which factors are
associated with life expectancy. This section asks a different
question: how well does each method predict life expectancy for
counties it did not see during training?

The same held-out test split is used for all three models so that
their predictive performance is directly comparable.
"""

X_train, X_test, y_train, y_test = train_test_split(
    model_df[PREDICTORS], model_df[OUTCOME], test_size=0.2, random_state=42
)

X_train_ols = sm.add_constant(X_train)
X_test_ols = sm.add_constant(X_test, has_constant="add")

ols_model = sm.OLS(y_train, X_train_ols).fit()
ols_pred = ols_model.predict(X_test_ols)

rf_model = RandomForestRegressor(n_estimators=500, random_state=42, n_jobs=-1)
rf_model.fit(X_train, y_train)
rf_pred = rf_model.predict(X_test)

xgb_model = XGBRegressor(
    n_estimators=500,
    max_depth=4,
    learning_rate=0.05,
    random_state=42,
)
xgb_model.fit(X_train, y_train)
xgb_pred = xgb_model.predict(X_test)


def rmse(y_true, y_pred):
    return mean_squared_error(y_true, y_pred) ** 0.5


comparison = pd.DataFrame(
    {
        "Model": ["OLS", "Random Forest", "XGBoost"],
        "Test R2": [
            r2_score(y_test, ols_pred),
            r2_score(y_test, rf_pred),
            r2_score(y_test, xgb_pred),
        ],
        "Test RMSE (years)": [
            rmse(y_test, ols_pred),
            rmse(y_test, rf_pred),
            rmse(y_test, xgb_pred),
        ],
    }
)

print("\nPredictive performance comparison (held-out test split):")
print(comparison.to_string(index=False))

# Random Forest feature importance.
rf_importance = pd.Series(
    rf_model.feature_importances_, index=PREDICTORS
).sort_values(ascending=False)

print("\nRandom Forest feature importance:")
print(rf_importance)

plt.figure(figsize=(8, 6))
rf_importance.sort_values().plot(kind="barh")
plt.title("Random Forest Feature Importance")
plt.xlabel("Importance")
plt.tight_layout()
plt.show()


# ============================================================
# 16. Next steps
# ============================================================

"""
Potential extensions:

1. Tune Random Forest / XGBoost hyperparameters (grid or random search)
   rather than using the current fixed settings.

2. Examine whether relationships differ by geographic region or
   urban/rural status.

3. Use SHAP values to explain individual Random Forest / XGBoost
   predictions, not just aggregate feature importance.

4. Continue to investigate counties with unusually positive or
   negative residuals as candidates for case studies.

5. Clearly distinguish prediction from causal/inferential conclusions
   in any write-up.

The central limitation of the current analysis is that it is based on
observational county-level data. The regression and tree-based models
identify associations and predictive power, not causal effects.
"""

print("\nAnalysis complete.")
