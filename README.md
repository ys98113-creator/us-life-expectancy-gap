# Why Is U.S. Life Expectancy Lower Than Countries With Less Income?

A data science capstone investigating two related questions: (1) does the U.S. underperform its income level on life expectancy compared to other countries, and (2) within the U.S., why do some counties have much lower life expectancy than their socioeconomic and health profile would predict — and specifically, how much of that gap is explained by the local food environment?

## Motivation

In 2024, U.S. life expectancy (79.0 years) sat 3.7 years below the average of comparable high-income countries. That gap is well established in the literature (Chetty et al. 2016, *JAMA*; National Academies 2021, *High and Rising Mortality Rates Among Working-Age Adults*). This project doesn't try to re-prove that — instead it asks a narrower, less-answered question: **does the food environment (fast-food density, food access, SNAP participation) explain any of the county-to-county variation in life expectancy that isn't already explained by income, health behaviors, and access to care?**

This is decomposition + predictive analysis, not causal identification — there's no natural experiment here, and that's stated explicitly rather than implied.

## Data

Everything is pulled from public sources with no login/registration required, via the scripts in this repo. Four datasets, at two levels of geography:

| Dataset | Grain | Rows | Sources |
|---|---|---|---|
| `cross_country_life_expectancy.csv` | Country-year | 3,000 (100 countries × 1995–2024) | World Bank WDI (bulk CSV) |
| `us_county_food_environment.csv` | County | 3,103 | USALEEP (CDC, life expectancy), USDA Food Environment Atlas, CDC PLACES, County Health Rankings |
| `us_county_panel.csv` | County-year | 28,377 | Same 4 sources as above, kept as separate real vintages instead of one cross-section |
| `us_tract_food_environment.csv` | Census tract | 65,162 | USALEEP (native resolution), USDA Food Access Research Atlas, CDC PLACES tract data |

Full source documentation — publisher, access method, exact variable codes, and known caveats for every column — lives in the docstring at the top of each `build_*.py` script.

**A real limitation, stated up front:** the county-level outcome variable (USALEEP life expectancy) is a 2010–2015 estimate that has never been updated, while most predictors are 2018–2023 vintage. Every free, no-login source for a more recent county life-expectancy series requires an IHME GHDx account; this project deliberately stayed login-free, so the vintage gap is real and disclosed rather than hidden.

## Repository structure

```
build_county_dataset.py     # county-level cross-section (the main dataset used for modeling)
build_tract_dataset.py      # census-tract-level version, ~20x more rows
build_county_panel.py       # county-year panel, real vintages kept separate rather than merged
cross_country_life_expectancy.csv
us_county_food_environment.csv
us_county_panel.csv
us_tract_food_environment.csv
```

Each `build_*.py` is self-contained and re-runnable: `python build_county_dataset.py` re-downloads (or uses a local cache) and regenerates its output CSV from scratch.

## Method

`life_expectancy ~ food_environment + health_controls + socioeconomic_controls`

- **food_environment**: fast-food density, food desert %, SNAP participation, farmers' market density — the variable group the research question is actually about
- **health_controls**: adult obesity, diabetes prevalence, physical inactivity, uninsured rate — so a food-environment effect has to beat real health/access measures, not nothing
- **socioeconomic_controls**: median household income, income inequality ratio, child poverty rate, primary care physician density

Modeled three ways on the same held-out test split, for a fair comparison:

| Model | Test R² | Test RMSE (years) |
|---|---|---|
| OLS (12 predictors) | 0.523 | 1.61 |
| OLS (9 predictors, trimmed) | 0.530 | 1.65 |
| Random Forest | **0.651** | **1.42** |
| XGBoost | 0.642 | 1.44 |

Random Forest and XGBoost both outperform OLS by a meaningful margin — consistent with a Breusch-Pagan test rejecting homoscedasticity (p ≈ 2.6e-11) in the OLS fit, i.e. there's real evidence the true relationship isn't well captured by a single linear model.

## Known limitations

- **Vintage mismatch**: outcome (2010–2015) predates most predictors (2018–2023) — see Data section above.
- **Small-county noise**: `n_tracts` (county file) / `n_tracts` (life expectancy's underlying tract count) varies from 1 to 2,195 — counties built from very few tracts have noisier life-expectancy estimates, and should be weighted or filtered before being treated as genuine outliers.
- **Definitional substitutions**: `income_inequality_ratio` (80th/20th percentile household income ratio) is related to but not the same statistic as a Gini coefficient; `child_poverty_pct` is child poverty specifically, not an all-ages county poverty rate — County Health Rankings doesn't publish one.
- **Tract-level boundary mismatch**: CDC PLACES tract data uses 2020 Census tract boundaries; USALEEP and USDA's Food Access Research Atlas use 2010 boundaries — about 23% of tracts don't join cleanly across this vintage change.

## References

- Chetty, R. et al. (2016). "The Association Between Income and Life Expectancy in the United States, 2001-2014." *JAMA*.
- National Academies of Sciences, Engineering, and Medicine (2021). *High and Rising Mortality Rates Among Working-Age Adults.*
