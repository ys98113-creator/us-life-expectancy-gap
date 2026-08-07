"""
Build us_county_food_environment.csv
=====================================

Reproduces the county-level dataset from four public sources, no login/registration
required anywhere in the chain. Run this file top to bottom (`python build_county_dataset.py`)
and it regenerates us_county_food_environment.csv from scratch.

Grain: one row per US county (5-digit FIPS code). ~3,103 counties.

--------------------------------------------------------------------------------------
SOURCE 1 — USALEEP (U.S. Small-area Life Expectancy Estimates Project)
--------------------------------------------------------------------------------------
What it is:   The outcome variable (y). Life expectancy at birth, estimated by NCHS/CDC
              and NAPHSIS from actual death records, at the census-tract level.
Publisher:    CDC National Center for Health Statistics (NCHS)
Geography:    Census tract (~65,000 tracts) — aggregated here to county by simple mean
              of tract-level life expectancy. NOTE: a population-weighted mean would be
              more correct (bigger tracts should count more), but tract population isn't
              in this file — flagged rather than silently "fixed" for you.
Vintage:      2010-2015 — a single one-time release, never updated since. This is the
              oldest data in the merged file; every predictor below is newer.
Access:       Public CDC FTP, plain CSV per state, no login.
Columns used: e(0) (life expectancy at birth) -> life_expectancy

--------------------------------------------------------------------------------------
SOURCE 2 — USDA Food Environment Atlas
--------------------------------------------------------------------------------------
What it is:   The food_environment predictors — the variable group the research
              hypothesis is actually about.
Publisher:    USDA Economic Research Service (ERS)
Geography:    County
Vintage:      Mixed per indicator (2018-2022) — USDA doesn't refresh every variable on
              the same schedule; each column below carries its own vintage year in the
              underlying Variable_Code (e.g. FFRPTH20 = 2020).
Access:       Public direct-download CSV/Excel bundle on ers.usda.gov, no login.
Columns used: FFRPTH20             (fast-food restaurants per 1,000 pop, 2020)  -> fast_food_density
              PCT_LACCESS_POP19    (% pop w/ low access to a grocery store, 2019) -> food_desert_pct
              PCT_SNAP22           (SNAP participants, % of pop, 2022)          -> snap_participation_pct
              FMRKTPTH18           (farmers' markets per 1,000 pop, 2018)       -> farmers_market_density

--------------------------------------------------------------------------------------
SOURCE 3 — CDC PLACES (Local Data for Better Health)
--------------------------------------------------------------------------------------
What it is:   The health_controls predictors — so a food-environment effect has to beat
              actual health/access measures, not just "nothing."
Publisher:    CDC, modeled small-area estimates from the BRFSS survey
Geography:    County
Vintage:      2023 (all four measures used here share this release year)
Access:       Public Socrata SODA API (data.cdc.gov), queried directly as CSV, no login.
Columns used: OBESITY   -> adult_obesity_pct
              DIABETES  -> diabetes_prevalence_pct
              LPA       -> physical_inactivity_pct   (LPA = "leisure-time physical activity", inverted: this is the INACTIVE %)
              ACCESS2   -> uninsured_pct              (adults lacking health insurance)
              (value type used: crude prevalence, "CrdPrv" — not age-adjusted)

--------------------------------------------------------------------------------------
SOURCE 4 — County Health Rankings & Roadmaps
--------------------------------------------------------------------------------------
What it is:   The socioeconomic_controls predictors — income, inequality, poverty,
              primary-care access.
Publisher:    University of Wisconsin Population Health Institute + Robert Wood Johnson
              Foundation
Geography:    County
Vintage:      2025 release (most fields drawn from ACS ~2019-2023 and other recent years)
Access:       Public direct-download CSV on countyhealthrankings.org, no login.
Columns used: v063_rawvalue -> median_household_income
              v044_rawvalue -> income_inequality_ratio   (80th/20th percentile household
                                                            income ratio — NOT a Gini
                                                            coefficient, similar concept
                                                            only)
              v024_rawvalue -> child_poverty_pct          (children in poverty; CHR has
                                                            no all-ages county poverty
                                                            rate, so this is the closest
                                                            available measure — narrower
                                                            scope than "poverty_rate"
                                                            would imply)
              v004_rawvalue -> primary_care_physicians_per_1000 (raw value is physicians
                                                            per capita; rescaled x1000 for
                                                            readability)
              county, state -> county_name, state          (names, for readability/joins)

--------------------------------------------------------------------------------------
A note on vintage mismatch
--------------------------------------------------------------------------------------
The target (2010-2015) is older than every predictor (2018-2023). This is a real,
documented limitation, not an oversight — USALEEP has never been updated, and every
other candidate outcome source (e.g. IHME's county life-expectancy time series) requires
a GHDx account login, which this pipeline deliberately avoids. Report this mismatch
alongside any modeling results rather than treating the data as perfectly contemporaneous.
"""

import io
import zipfile
from pathlib import Path

import pandas as pd
import requests

PROJECT_ROOT = Path(__file__).parent
CACHE_DIR = PROJECT_ROOT / "county_data_cache"
CACHE_DIR.mkdir(exist_ok=True)

OUTPUT_PATH = PROJECT_ROOT / "us_county_food_environment.csv"

# Data dictionary for every column in the output file. Source detail (publisher, access
# method, definitional caveats) lives in the module docstring above — this is the quick
# per-column reference; print_data_dictionary() below renders it readably.
VARIABLE_DESCRIPTIONS = {
    "fips": "5-digit county FIPS code (state + county). The join key for every source.",
    "county_name": "County name, e.g. 'Autauga County'. From County Health Rankings.",
    "state": "2-letter state abbreviation. From County Health Rankings.",
    "life_expectancy": "Life expectancy at birth, in years. Outcome variable (y). Simple mean of tract-level USALEEP estimates within the county — NOT population-weighted (see n_tracts).",
    "n_tracts": "Number of census tracts averaged to build life_expectancy for this county. Low values (e.g. 1-3) mean a noisier estimate — check this before treating a county as a real outlier.",
    "fast_food_density": "Fast-food restaurants per 1,000 county residents (food_environment group).",
    "food_desert_pct": "% of county population with low access to a grocery store (>1 mile urban / >10 miles rural from a supermarket) (food_environment group).",
    "snap_participation_pct": "% of county population participating in SNAP (food stamps) (food_environment group).",
    "farmers_market_density": "Farmers' markets per 1,000 county residents (food_environment group).",
    "adult_obesity_pct": "% of adults with obesity (BMI >= 30), crude prevalence, modeled from BRFSS (health_controls group).",
    "diabetes_prevalence_pct": "% of adults with diagnosed diabetes, crude prevalence, modeled from BRFSS (health_controls group).",
    "physical_inactivity_pct": "% of adults reporting no leisure-time physical activity, crude prevalence (health_controls group).",
    "uninsured_pct": "% of adults aged 18-64 lacking health insurance, crude prevalence (health_controls group).",
    "median_household_income": "County median household income, in US dollars (socioeconomic_controls group).",
    "income_inequality_ratio": "Ratio of household income at the 80th percentile to the 20th percentile. Related to, but NOT the same statistic as, a Gini coefficient (socioeconomic_controls group).",
    "child_poverty_pct": "% of children (not all ages) living below the poverty line — County Health Rankings has no all-ages county poverty rate, so this is the closest available measure (socioeconomic_controls group).",
    "primary_care_physicians_per_1000": "Primary care physicians per 1,000 county residents (socioeconomic_controls group).",
}


def print_data_dictionary() -> None:
    print("\nVariable descriptions:")
    for col, desc in VARIABLE_DESCRIPTIONS.items():
        print(f"  {col:35s} {desc}")
    print("\n  Every variable above (except fips/county_name/state/n_tracts) also has a")
    print("  matching '<column>_year' field — the vintage that specific value came from.")
    print("  It's null wherever the value itself is null, so a missing value never carries")
    print("  a fake year label. life_expectancy_year is text ('2010-2015'); the rest are")
    print("  plain years.")


# ======================================================================================
# SOURCE 1 — USALEEP: life expectancy by census tract, aggregated to county
# ======================================================================================
def fetch_usaleep() -> pd.DataFrame:
    states = [
        "AL", "AK", "AZ", "AR", "CA", "CO", "CT", "DE", "DC", "FL", "GA", "HI", "ID",
        "IL", "IN", "IA", "KS", "KY", "LA", "ME", "MD", "MA", "MI", "MN", "MS", "MO",
        "MT", "NE", "NV", "NH", "NJ", "NM", "NY", "NC", "ND", "OH", "OK", "OR", "PA",
        "RI", "SC", "SD", "TN", "TX", "UT", "VT", "VA", "WA", "WV", "WI", "WY",
    ]
    base_url = "https://ftp.cdc.gov/pub/health_statistics/nchs/Datasets/NVSS/USALEEP/CSV/{}_A.CSV"

    frames = []
    for st in states:
        cache_file = CACHE_DIR / f"usaleep_{st}.csv"
        if cache_file.exists():
            frames.append(pd.read_csv(cache_file, dtype={"STATE2KX": str, "CNTY2KX": str}))
            continue
        resp = requests.get(base_url.format(st), timeout=30)
        resp.raise_for_status()
        df = pd.read_csv(io.StringIO(resp.text), dtype={"STATE2KX": str, "CNTY2KX": str})
        df.to_csv(cache_file, index=False)
        frames.append(df)

    tract = pd.concat(frames, ignore_index=True)
    tract["fips"] = tract["STATE2KX"].str.zfill(2) + tract["CNTY2KX"].str.zfill(3)

    county_le = tract.groupby("fips", as_index=False).agg(
        life_expectancy=("e(0)", "mean"),
        n_tracts=("e(0)", "size"),
    )
    return county_le


# ======================================================================================
# SOURCE 2 — USDA Food Environment Atlas: food-environment predictors
# ======================================================================================
def fetch_food_environment_atlas() -> pd.DataFrame:
    zip_path = CACHE_DIR / "food_environment_atlas.zip"
    if not zip_path.exists():
        url = "https://www.ers.usda.gov/media/5570/food-environment-atlas-csv-files.zip"
        resp = requests.get(url, timeout=30)
        resp.raise_for_status()
        zip_path.write_bytes(resp.content)

    with zipfile.ZipFile(zip_path) as zf:
        with zf.open("StateAndCountyData.csv") as f:
            fea_raw = pd.read_csv(f, dtype={"FIPS": str})

    fea_raw["FIPS"] = fea_raw["FIPS"].str.split(".").str[0].str.zfill(5)

    # USDA sentinel codes for missing data, per the Atlas's own ReadMe: -9999 = data not
    # available/not applicable/suppressed; -8888 = county didn't exist that year. Left as
    # literal numbers, these silently corrupt any regression (they did exactly that to
    # fast_food_density here) — must be NaN, not real values.
    fea_raw["Value"] = fea_raw["Value"].replace({-9999: pd.NA, -8888: pd.NA})

    variable_map = {
        "FFRPTH20":           "fast_food_density",
        "PCT_LACCESS_POP19":  "food_desert_pct",
        "PCT_SNAP22":         "snap_participation_pct",
        "FMRKTPTH18":         "farmers_market_density",
    }
    fea = (
        fea_raw[fea_raw["Variable_Code"].isin(variable_map)]
        .pivot_table(index="FIPS", columns="Variable_Code", values="Value", aggfunc="first")
        .rename(columns=variable_map)
        .reset_index()
        .rename(columns={"FIPS": "fips"})
    )
    return fea


# ======================================================================================
# SOURCE 3 — CDC PLACES: health-controls predictors
# ======================================================================================
def fetch_places() -> pd.DataFrame:
    cache_file = CACHE_DIR / "places_raw.csv"
    measure_map = {
        "OBESITY":  "adult_obesity_pct",
        "DIABETES": "diabetes_prevalence_pct",
        "LPA":      "physical_inactivity_pct",
        "ACCESS2":  "uninsured_pct",
    }

    if not cache_file.exists():
        # data.cdc.gov Socrata endpoint for "PLACES: Local Data for Better Health, County Data"
        base_url = "https://data.cdc.gov/resource/swc5-untb.csv"
        params = {
            "$select": "locationid,measureid,data_value,datavaluetypeid",
            "$where": "measureid in('OBESITY','DIABETES','LPA','ACCESS2') AND datavaluetypeid='CrdPrv'",
            "$limit": 50000,
        }
        resp = requests.get(base_url, params=params, timeout=60)
        resp.raise_for_status()
        cache_file.write_text(resp.text)

    places_raw = pd.read_csv(cache_file, dtype={"locationid": str})
    places_raw["locationid"] = places_raw["locationid"].str.zfill(5)

    places = (
        places_raw.pivot_table(index="locationid", columns="measureid", values="data_value", aggfunc="first")
        .rename(columns=measure_map)
        .reset_index()
        .rename(columns={"locationid": "fips"})
    )
    return places


# ======================================================================================
# SOURCE 4 — County Health Rankings: socioeconomic predictors + county/state names
# ======================================================================================
def fetch_county_health_rankings() -> pd.DataFrame:
    cache_file = CACHE_DIR / "chr_2025.csv"
    if not cache_file.exists():
        url = "https://www.countyhealthrankings.org/sites/default/files/media/document/analytic_data2025_v3.csv"
        resp = requests.get(url, timeout=60)
        resp.raise_for_status()
        cache_file.write_bytes(resp.content)

    # file has two header rows: human-readable labels, then machine-readable codes — use the second
    chr_raw = pd.read_csv(cache_file, low_memory=False, header=[0, 1])
    chr_raw.columns = [code for _label, code in chr_raw.columns]

    chr_small = chr_raw[[
        "fipscode", "county", "state",
        "v063_rawvalue", "v044_rawvalue", "v024_rawvalue", "v004_rawvalue",
    ]].copy()
    chr_small.columns = [
        "fips", "county_name", "state",
        "median_household_income", "income_inequality_ratio", "child_poverty_pct",
        "primary_care_physicians_per_1000",
    ]
    chr_small["fips"] = chr_small["fips"].astype(str).str.zfill(5)
    chr_small = chr_small[~chr_small["fips"].str.endswith("000")]  # drop state/US summary rows

    # CHR reports child poverty as a 0-1 fraction; rescale to 0-100 to match the other _pct columns
    chr_small["child_poverty_pct"] = chr_small["child_poverty_pct"] * 100
    # CHR reports physicians as a raw per-capita fraction; rescale to per-1,000 for readability
    chr_small["primary_care_physicians_per_1000"] = chr_small["primary_care_physicians_per_1000"] * 1000

    return chr_small


# ======================================================================================
# Merge everything onto the USALEEP base (county-level outcome variable)
# ======================================================================================
def build_dataset() -> pd.DataFrame:
    usaleep = fetch_usaleep()
    fea = fetch_food_environment_atlas()
    places = fetch_places()
    chr_data = fetch_county_health_rankings()

    merged = (
        chr_data[["fips", "county_name", "state"]]
        .merge(usaleep, on="fips", how="right")   # keep USALEEP's ~3,103 counties as the base
        .merge(fea, on="fips", how="left")
        .merge(places, on="fips", how="left")
        .merge(chr_data.drop(columns=["county_name", "state"]), on="fips", how="left")
    )

    # One vintage year per variable, not one year for the whole row — the four sources
    # don't share a calendar year (see the source docstrings above), so a single "year"
    # column would either be wrong or would hide real vintage spread. life_expectancy is
    # a 2010-2015 estimate, not a single year, so its companion column is left as text.
    variable_years = {
        "life_expectancy":                  "2010-2015",
        "fast_food_density":                2020,
        "food_desert_pct":                  2019,
        "snap_participation_pct":           2022,
        "farmers_market_density":           2018,
        "adult_obesity_pct":                2023,
        "diabetes_prevalence_pct":          2023,
        "physical_inactivity_pct":          2023,
        "uninsured_pct":                    2023,
        "median_household_income":          2025,
        "income_inequality_ratio":          2025,
        "child_poverty_pct":                2025,
        "primary_care_physicians_per_1000": 2025,
    }
    for col, year in variable_years.items():
        # null out the year wherever the value itself is missing — a year label next to a
        # missing value would wrongly imply a data point that doesn't actually exist
        merged[f"{col}_year"] = merged[col].notna().map({True: year, False: pd.NA})

    column_order = [
        "fips", "county_name", "state",
        "life_expectancy", "life_expectancy_year", "n_tracts",
        "fast_food_density", "fast_food_density_year",
        "food_desert_pct", "food_desert_pct_year",
        "snap_participation_pct", "snap_participation_pct_year",
        "farmers_market_density", "farmers_market_density_year",
        "adult_obesity_pct", "adult_obesity_pct_year",
        "diabetes_prevalence_pct", "diabetes_prevalence_pct_year",
        "physical_inactivity_pct", "physical_inactivity_pct_year",
        "uninsured_pct", "uninsured_pct_year",
        "median_household_income", "median_household_income_year",
        "income_inequality_ratio", "income_inequality_ratio_year",
        "child_poverty_pct", "child_poverty_pct_year",
        "primary_care_physicians_per_1000", "primary_care_physicians_per_1000_year",
    ]
    return merged[column_order]


if __name__ == "__main__":
    df = build_dataset()
    df.to_csv(OUTPUT_PATH, index=False)
    print(f"Wrote {len(df)} counties x {len(df.columns)} columns -> {OUTPUT_PATH}")
    print(df.isna().sum())
    print_data_dictionary()
