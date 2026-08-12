# What Explains the Gap in U.S. County Life Expectancy?

A data science project investigating why life expectancy varies so much across U.S. counties, and whether that variation can be predicted from a county's socioeconomic conditions, health behaviors, healthcare access, and food environment.

## Motivation

This project originally set out to compare U.S. life expectancy against other countries. That angle was dropped: cross-country life-expectancy and covariate data are too limited and inconsistent (differing survey years, definitions, and missingness across countries) to support a rigorous analysis. The project pivoted to a question with much richer data behind it: **within the United States, across all 50 states, why do some counties have life expectancies years apart from others, and how well can that gap be predicted?**

This is decomposition + predictive analysis, not causal identification — there's no natural experiment here, and that's stated explicitly rather than implied.

## Research question

What factors are associated with differences in life expectancy across U.S. counties, and how much of a county's life expectancy can be predicted from its socioeconomic conditions, health behaviors, healthcare access, and food environment?

A second, related question: **which counties have life expectancy far above or below what their profile predicts** — i.e., after accounting for income, health behaviors, insurance coverage, and food access, which counties are notable positive or negative outliers, and what might explain that?

## Data

Everything is pulled from public sources with no login/registration required, via the scripts in this repo.

| Dataset | Grain | Rows | Sources |
|---|---|---|---|
| `us_county_food_environment.csv` | County | 3,103 | USALEEP (CDC, life expectancy), USDA Food Environment Atlas, CDC PLACES, County Health Rankings |
| `us_county_panel.csv` | County-year | 28,377 | Same 4 sources as above, kept as separate real vintages instead of one cross-section |
| `us_tract_food_environment.csv` | Census tract | 65,162 | USALEEP (native resolution), USDA Food Access Research Atlas, CDC PLACES tract data |

Full source documentation — publisher, access method, exact variable codes, and known caveats for every column — lives in the docstring at the top of each `build_*.py` script.

**A real limitation, stated up front:** the county-level outcome variable (USALEEP life expectancy) is a 2010–2015 estimate that has never been updated, while most predictors are 2018–2023 vintage. Every free, no-login source for a more recent county life-expectancy series requires an IHME GHDx account; this project deliberately stayed login-free, so the vintage gap is real and disclosed rather than hidden.

## Repository structure

```
build_county_dataset.py              # county-level cross-section (the main dataset used for modeling)
build_tract_dataset.py               # census-tract-level version, ~20x more rows
build_county_panel.py                # county-year panel, real vintages kept separate rather than merged
us_life_expectancy_gap_analysis.py   # EDA, OLS regression + diagnostics, Random Forest / XGBoost comparison
us_county_food_environment.csv
us_county_panel.csv
us_tract_food_environment.csv
```

Each `build_*.py` is self-contained and re-runnable: `python build_county_dataset.py` re-downloads (or uses a local cache) and regenerates its output CSV from scratch.

## How to run

```
python us_life_expectancy_gap_analysis.py
```

This runs the full pipeline: load and clean the data, EDA, correlation analysis, an OLS regression with VIF/heteroskedasticity/Cook's-distance diagnostics, the top/bottom 10 counties by residual, and a held-out-test-split comparison of OLS vs. Random Forest vs. XGBoost.

## Method

`life_expectancy ~ food_environment + health_controls + socioeconomic_controls`

- **food_environment**: fast-food density, food desert %, SNAP participation, farmers' market density
- **health_controls**: adult obesity, diabetes prevalence, physical inactivity, uninsured rate
- **socioeconomic_controls**: median household income, income inequality ratio, child poverty rate, primary care physician density

Modeled three ways on the same held-out test split (20% of 2,416 counties with complete data), for a fair comparison:

| Model | Test R² | Test RMSE (years) |
|---|---|---|
| OLS (12 predictors) | 0.523 | 1.61 |
| Random Forest | 0.612 | 1.45 |
| XGBoost | **0.619** | **1.44** |

Both tree-based models outperform OLS by a meaningful margin — consistent with a Breusch-Pagan test rejecting homoscedasticity (p ≈ 1.1e-11) in the OLS fit, i.e. there's real evidence the true relationship isn't well captured by a single linear model.

**Child poverty rate is the dominant predictor**, both by raw correlation with life expectancy (-0.57, the strongest of any variable) and by Random Forest feature importance (about 40% of total importance, roughly 4x the next-largest feature). Food-environment variables — fast-food density in particular — are comparatively weak predictors on their own (raw correlation of only 0.03), suggesting that in this county-level data, economic hardship carries more of the story than food access specifically.

## Counties that outperform or underperform their profile

After fitting the OLS model, residuals (actual − predicted life expectancy) identify counties that are notable outliers relative to their socioeconomic/health/food-environment profile. The 10 largest positive and 10 largest negative residuals are printed by the analysis script — these are good candidates for case-study follow-up, since a large residual signals something the model's variables aren't capturing (a strong or weak local health system, a demographic skew, a data-quality issue, etc.).

## Known limitations

- **Vintage mismatch**: outcome (2010–2015) predates most predictors (2018–2023) — see Data section above.
- **Small-county noise**: `n_tracts` (the number of census tracts each county's life-expectancy estimate is built from) varies from 1 to over 2,000 — counties built from very few tracts have noisier life-expectancy estimates, and should be weighted or filtered before being treated as genuine outliers.
- **Definitional substitutions**: `income_inequality_ratio` (80th/20th percentile household income ratio) is related to but not the same statistic as a Gini coefficient; `child_poverty_pct` is child poverty specifically, not an all-ages county poverty rate — County Health Rankings doesn't publish one.
- **Tract-level boundary mismatch**: CDC PLACES tract data uses 2020 Census tract boundaries; USALEEP and USDA's Food Access Research Atlas use 2010 boundaries — about 23% of tracts don't join cleanly across this vintage change.
- **Observational data**: this is associational, not causal. Coefficients and feature importances describe conditional relationships in the data, not the effect of an intervention.

## Next steps

- Tune Random Forest / XGBoost hyperparameters rather than using fixed settings.
- Use SHAP values to explain individual predictions, not just aggregate feature importance.
- Examine whether relationships differ by geographic region or urban/rural status.
- Investigate the largest-residual counties as case studies.

## References

- Chetty, R. et al. (2016). "The Association Between Income and Life Expectancy in the United States, 2001-2014." *JAMA*.
- National Academies of Sciences, Engineering, and Medicine (2021). *High and Rising Mortality Rates Among Working-Age Adults.*
