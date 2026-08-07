"""
Build us_county_panel.csv — a county x year panel (long format)
==================================================================

Six variables, requested as a "last 30 years" panel. Real availability turned out to be
far short of that — this file is honest about it instead of faking a rectangle. Run with
`python build_county_panel.py`.

Grain: one row per (county FIPS, year). ~3,103 counties x however many years a given
variable actually has data — most cells outside a variable's real vintages are NaN
BY DESIGN, not a bug. No single calendar year has all 6 variables populated; see the
coverage table this script prints at the end for what actually overlaps.

--------------------------------------------------------------------------------------
Real year coverage per variable (this is the finding, not an assumption):
--------------------------------------------------------------------------------------
  life_expectancy_2010_2015   constant, single 2010-2015 USALEEP estimate — does not
                               vary by year, attached to every row as-is (see note below)
  fast_food_density            2016, 2020                        (USDA FEA — only 2 vintages exist, ever)
  food_desert_pct               2015, 2019                        (USDA FEA — only 2 vintages exist, ever)
  adult_obesity_pct            2019, 2020, 2021, 2022, 2023       (CDC PLACES, 5 separate releases)
  diabetes_prevalence_pct      2019, 2020, 2021, 2022, 2023       (CDC PLACES, same 5 releases)
  uninsured_pct                2019, 2020, 2021, 2022, 2023       (CDC PLACES, same 5 releases)

  Note: a 2018 survey-year release also exists (dataset dv4u-3x3q) but its schema has no
  county FIPS column at all — only county name + state — unlike every other release. Rather
  than build a fragile name-matching join for one extra year, it's excluded here.
  median_household_income      2019, 2020, 2021, 2022, 2023, 2024, 2025 (County Health Rankings annual files)

Why not further back / a true 30-year span:
  - USALEEP (life expectancy) has never been re-run since its one 2010-2015 release —
    there is no year-by-year county life expectancy series available without registering
    for a login-gated source (IHME GHDx), which this pipeline deliberately avoids.
  - USDA only re-surveys food-environment indicators every several years, not annually —
    2015/2019 and 2016/2020 are literally the only two vintages that exist for these
    specific indicators.
  - CDC PLACES (the county-level successor to the "500 Cities" program) doesn't have
    estimates before ~2018.
  - County Health Rankings is the only source with a real annual run, and even that
    only reaches back to ~2019 in a consistently-downloadable file.

life_expectancy_2010_2015 is deliberately NOT named "life_expectancy" and is repeated
identically across every year for a given county — it is a constant baseline, not a
time-varying observation. Treat any year-over-year "trend" involving it as an artifact
of the other variables changing, not of life expectancy itself changing.
"""

from pathlib import Path

import pandas as pd
import requests

PROJECT_ROOT = Path(__file__).parent
CACHE_DIR = PROJECT_ROOT / "county_data_cache"
CACHE_DIR.mkdir(exist_ok=True)

OUTPUT_PATH = PROJECT_ROOT / "us_county_panel.csv"

# CDC PLACES: one Socrata dataset ID per release; each release's "year" field is the
# underlying BRFSS survey year, which lags the release year by ~2 years.
PLACES_RELEASES = {
    2019: "pqpp-u99h",  # 2021 release
    2020: "duw2-7jbt",  # 2022 release
    2021: "h3ej-a9ec",  # 2023 release
    2022: "fu4u-a9bh",  # 2024 release
    2023: "swc5-untb",  # 2025 release
}
PLACES_MEASURES = {
    "OBESITY":  "adult_obesity_pct",
    "DIABETES": "diabetes_prevalence_pct",
    "ACCESS2":  "uninsured_pct",
}

CHR_YEARS = [2019, 2020, 2021, 2022, 2023, 2024, 2025]


def fetch_usaleep_life_expectancy() -> pd.DataFrame:
    """Single constant value per county — see module docstring."""
    states = [
        "AL", "AK", "AZ", "AR", "CA", "CO", "CT", "DE", "DC", "FL", "GA", "HI", "ID",
        "IL", "IN", "IA", "KS", "KY", "LA", "ME", "MD", "MA", "MI", "MN", "MS", "MO",
        "MT", "NE", "NV", "NH", "NJ", "NM", "NY", "NC", "ND", "OH", "OK", "OR", "PA",
        "RI", "SC", "SD", "TN", "TX", "UT", "VT", "VA", "WA", "WV", "WI", "WY",
    ]
    base_url = "https://ftp.cdc.gov/pub/health_statistics/nchs/Datasets/NVSS/USALEEP/CSV/{}_A.CSV"
    cache_file = CACHE_DIR / "usaleep_all.csv"

    if cache_file.exists():
        tract = pd.read_csv(cache_file, dtype={"STATE2KX": str, "CNTY2KX": str})
    else:
        frames = []
        for st in states:
            resp = requests.get(base_url.format(st), timeout=30)
            resp.raise_for_status()
            frames.append(pd.read_csv(pd.io.common.StringIO(resp.text),
                                       dtype={"STATE2KX": str, "CNTY2KX": str}))
        tract = pd.concat(frames, ignore_index=True)
        tract.to_csv(cache_file, index=False)

    tract["fips"] = tract["STATE2KX"].str.zfill(2) + tract["CNTY2KX"].str.zfill(3)
    return tract.groupby("fips", as_index=False)["e(0)"].mean().rename(
        columns={"e(0)": "life_expectancy_2010_2015"}
    )


def fetch_fea_two_vintages() -> pd.DataFrame:
    """USDA FEA: fast_food_density (2016, 2020) and food_desert_pct (2015, 2019), long format."""
    zip_path = CACHE_DIR / "food_environment_atlas.zip"
    if not zip_path.exists():
        url = "https://www.ers.usda.gov/media/5570/food-environment-atlas-csv-files.zip"
        resp = requests.get(url, timeout=30)
        resp.raise_for_status()
        zip_path.write_bytes(resp.content)

    import zipfile
    with zipfile.ZipFile(zip_path) as zf:
        with zf.open("StateAndCountyData.csv") as f:
            fea_raw = pd.read_csv(f, dtype={"FIPS": str})
    fea_raw["FIPS"] = fea_raw["FIPS"].str.split(".").str[0].str.zfill(5)

    var_year_map = {
        "FFRPTH16":          ("fast_food_density", 2016),
        "FFRPTH20":          ("fast_food_density", 2020),
        "PCT_LACCESS_POP15": ("food_desert_pct", 2015),
        "PCT_LACCESS_POP19": ("food_desert_pct", 2019),
    }
    rows = []
    for code, (col, year) in var_year_map.items():
        sub = fea_raw[fea_raw["Variable_Code"] == code][["FIPS", "Value"]].copy()
        sub["year"] = year
        sub["variable"] = col
        sub = sub.rename(columns={"FIPS": "fips", "Value": "value"})
        rows.append(sub)
    long = pd.concat(rows, ignore_index=True)
    return long.pivot_table(index=["fips", "year"], columns="variable", values="value").reset_index()


def fetch_places_panel() -> pd.DataFrame:
    """CDC PLACES: obesity / diabetes / uninsured, one real survey year per release, 2018-2023."""
    frames = []
    for year, dataset_id in PLACES_RELEASES.items():
        cache_file = CACHE_DIR / f"places_{year}.csv"
        if not cache_file.exists():
            url = f"https://data.cdc.gov/resource/{dataset_id}.csv"
            params = {
                "$select": "locationid,measureid,data_value,datavaluetypeid",
                "$where": "measureid in('OBESITY','DIABETES','ACCESS2') AND datavaluetypeid='CrdPrv'",
                "$limit": 50000,
            }
            resp = requests.get(url, params=params, timeout=60)
            resp.raise_for_status()
            cache_file.write_text(resp.text)

        raw = pd.read_csv(cache_file, dtype={"locationid": str})
        raw["locationid"] = raw["locationid"].str.zfill(5)
        wide = raw.pivot_table(
            index="locationid", columns="measureid", values="data_value", aggfunc="first"
        ).rename(columns=PLACES_MEASURES).reset_index().rename(columns={"locationid": "fips"})
        wide["year"] = year
        frames.append(wide)

    return pd.concat(frames, ignore_index=True)


def fetch_chr_income_panel() -> pd.DataFrame:
    """County Health Rankings: median_household_income, one real file per year, 2019-2025."""
    frames = []
    for year in CHR_YEARS:
        cache_file = CACHE_DIR / f"chr_{year}.csv"
        if not cache_file.exists():
            url = f"https://www.countyhealthrankings.org/sites/default/files/media/document/analytic_data{year}.csv"
            resp = requests.get(url, timeout=60)
            resp.raise_for_status()
            cache_file.write_bytes(resp.content)

        raw = pd.read_csv(cache_file, low_memory=False, header=[0, 1])
        raw.columns = [code for _label, code in raw.columns]
        small = raw[["fipscode", "county", "state", "v063_rawvalue"]].copy()
        small.columns = ["fips", "county_name", "state", "median_household_income"]
        small["fips"] = small["fips"].astype(str).str.zfill(5)
        small = small[~small["fips"].str.endswith("000")]
        small["year"] = year
        frames.append(small)

    return pd.concat(frames, ignore_index=True)


def build_panel() -> pd.DataFrame:
    life_exp = fetch_usaleep_life_expectancy()
    fea = fetch_fea_two_vintages()
    places = fetch_places_panel()
    chr_panel = fetch_chr_income_panel()

    names = chr_panel[["fips", "county_name", "state"]].drop_duplicates("fips")

    all_years = sorted(set(fea["year"]) | set(places["year"]) | set(chr_panel["year"]))
    fips_list = names["fips"].unique()
    scaffold = pd.MultiIndex.from_product([fips_list, all_years], names=["fips", "year"]).to_frame(index=False)

    panel = (
        scaffold
        .merge(names, on="fips", how="left")
        .merge(life_exp, on="fips", how="left")
        .merge(fea, on=["fips", "year"], how="left")
        .merge(places, on=["fips", "year"], how="left")
        .merge(chr_panel[["fips", "year", "median_household_income"]], on=["fips", "year"], how="left")
    )

    col_order = [
        "fips", "county_name", "state", "year",
        "life_expectancy_2010_2015",
        "fast_food_density", "food_desert_pct",
        "adult_obesity_pct", "diabetes_prevalence_pct", "uninsured_pct",
        "median_household_income",
    ]
    return panel[col_order].sort_values(["fips", "year"]).reset_index(drop=True)


if __name__ == "__main__":
    df = build_panel()
    df.to_csv(OUTPUT_PATH, index=False)

    print(f"Wrote {len(df)} rows ({df['fips'].nunique()} counties x {df['year'].nunique()} years) -> {OUTPUT_PATH}")
    print("\nNon-null count per variable, by year (this IS the real coverage — no year has every column):")
    coverage = df.drop(columns=["fips", "county_name", "state"]).groupby("year").count()
    print(coverage.drop(columns=["life_expectancy_2010_2015"]).to_string())
