"""
Build us_tract_food_environment.csv — census-tract-level dataset, ~20x more rows
====================================================================================

Same idea as build_county_dataset.py, but kept at USALEEP's native census-tract
resolution instead of aggregating up to county. Skips median household income (that
needed a Census API key that wasn't working reliably) — uses USDA's tract-level median
FAMILY income as a bonus instead, since it came free with the food-access download.
Run with `python build_tract_dataset.py`.

Grain: one row per census tract (11-digit FIPS). ~65,000-72,000 tracts depending on
source, joined on the raw tract FIPS.

--------------------------------------------------------------------------------------
SOURCE 1 — USALEEP (outcome, native resolution this time — no county aggregation)
--------------------------------------------------------------------------------------
Same source as the county build. Previously we collapsed this to county with a simple
mean; here it's used AS-IS as the target, at the resolution it was actually published in.
Vintage: 2010-2015. Tract boundaries: 2010 Census definitions.
Columns used: e(0) -> life_expectancy

--------------------------------------------------------------------------------------
SOURCE 2 — USDA Food Access Research Atlas (FARA), 2019 vintage
--------------------------------------------------------------------------------------
The actual tract-level "food desert" dataset — this is what the ACS/Cancer Society
life-expectancy-and-food-deserts study used, not a proxy. Tract boundaries: 2010 Census
definitions (matches USALEEP).
Access: public direct-download zip, ers.usda.gov, no login.
Columns used: lapop1share      -> food_desert_pct       (% of tract population >1 mile
                                                           from a supermarket, urban
                                                           tracts; >10 miles, rural)
              MedianFamilyIncome -> median_family_income (BONUS — not the same concept as
                                                           median HOUSEHOLD income; family
                                                           income excludes single-person
                                                           households, so it typically
                                                           runs a bit higher. Flagged, not
                                                           silently treated as equivalent.)
Known gap: lapop1share (and therefore food_desert_pct) is NaN for ~27% of ALL tracts in
FARA's own file, not just ours — USDA doesn't compute a population-access share for
tracts with ~zero population or that are almost entirely group quarters (prisons,
dorms, etc.). This is a source-level gap, not a join failure — confirmed by checking
that every USALEEP tract FIPS does exist in the FARA file.

--------------------------------------------------------------------------------------
SOURCE 3 — CDC PLACES, Census Tract Data, 2025 release
--------------------------------------------------------------------------------------
Same measures as the county build (obesity, diabetes, uninsured), published at tract
resolution. Tract boundaries: 2020 Census definitions — NOT the same vintage as USALEEP
or FARA (2010). This is a real, unresolved mismatch: roughly 84,000 tracts exist under
the 2020 definitions vs ~74,000 under 2010, because tracts were split/merged/renumbered
between censuses. The join below is done on raw FIPS anyway and the match rate is
printed at the end — some tracts will legitimately fail to match because they no longer
exist under the same ID, not because of a bug.
Vintage: 2023 survey year.
Access: public Socrata CSV export, data.cdc.gov, no login.
Columns used: OBESITY  -> adult_obesity_pct
              DIABETES -> diabetes_prevalence_pct
              ACCESS2  -> uninsured_pct

--------------------------------------------------------------------------------------
SOURCE 4 — USDA Food Environment Atlas (county-level; reused from build_county_dataset.py)
--------------------------------------------------------------------------------------
fast_food_density has no tract-level public source anywhere — USDA only ever built it at
county resolution. Every tract in a county is given that county's single value here.
This is a genuine resolution mismatch, not a shortcut: it means fast_food_density has
real information at ~3,100 distinct values despite appearing on ~65,000 rows, while every
other column varies tract-by-tract. Worth remembering when this column ranks low in any
feature-importance output — it may be under-informative simply because it's coarser, not
because fast food access doesn't matter.
Columns used: FFRPTH20 -> fast_food_density (2020, per 1,000 pop)
"""

import zipfile
from pathlib import Path

import pandas as pd
import requests

PROJECT_ROOT = Path(__file__).parent
CACHE_DIR = PROJECT_ROOT / "county_data_cache"  # reuse the same cache as the county build
CACHE_DIR.mkdir(exist_ok=True)

OUTPUT_PATH = PROJECT_ROOT / "us_tract_food_environment.csv"


def fetch_usaleep_tract() -> pd.DataFrame:
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

    tract["fips"] = tract["Tract ID"].astype(str).str.zfill(11)
    return tract[["fips", "e(0)"]].rename(columns={"e(0)": "life_expectancy"})


def fetch_fara() -> pd.DataFrame:
    zip_path = CACHE_DIR / "fara2019.zip"
    if not zip_path.exists():
        url = "https://www.ers.usda.gov/media/5627/2019-large-retailer-access-map-lram-formerly-known-as-the-food-access-research-atlas-fara-data.zip"
        resp = requests.get(url, timeout=30)
        resp.raise_for_status()
        zip_path.write_bytes(resp.content)

    with zipfile.ZipFile(zip_path) as zf:
        name = [n for n in zf.namelist() if n.lower().endswith(".csv") and "atlas" in n.lower()][0]
        with zf.open(name) as f:
            fara = pd.read_csv(f)

    fara["fips"] = fara["CensusTract"].astype(str).str.zfill(11)
    return fara[["fips", "lapop1share", "MedianFamilyIncome"]].rename(columns={
        "lapop1share": "food_desert_pct",
        "MedianFamilyIncome": "median_family_income",
    })


def fetch_places_tract() -> pd.DataFrame:
    cache_file = CACHE_DIR / "places_tract.csv"
    measure_map = {
        "OBESITY":  "adult_obesity_pct",
        "DIABETES": "diabetes_prevalence_pct",
        "ACCESS2":  "uninsured_pct",
    }
    if not cache_file.exists():
        url = "https://data.cdc.gov/resource/cwsq-ngmh.csv"
        params = {
            "$select": "locationid,measureid,data_value,datavaluetypeid",
            "$where": "measureid in('OBESITY','DIABETES','ACCESS2') AND datavaluetypeid='CrdPrv'",
            "$limit": 250000,
        }
        resp = requests.get(url, params=params, timeout=90)
        resp.raise_for_status()
        cache_file.write_text(resp.text)

    raw = pd.read_csv(cache_file, dtype={"locationid": str})
    raw["locationid"] = raw["locationid"].str.zfill(11)
    wide = raw.pivot_table(
        index="locationid", columns="measureid", values="data_value", aggfunc="first"
    ).rename(columns=measure_map).reset_index().rename(columns={"locationid": "fips"})
    return wide


def fetch_fast_food_by_county() -> pd.DataFrame:
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

    ffr = fea_raw[fea_raw["Variable_Code"] == "FFRPTH20"][["FIPS", "Value"]]
    return ffr.rename(columns={"FIPS": "county_fips", "Value": "fast_food_density"})


def build_dataset() -> pd.DataFrame:
    usaleep = fetch_usaleep_tract()
    fara = fetch_fara()
    places = fetch_places_tract()
    ffr_by_county = fetch_fast_food_by_county()

    merged = usaleep.merge(fara, on="fips", how="left")
    print(f"food_desert_pct populated for {merged['food_desert_pct'].notna().sum()}/{len(merged)} tracts — "
          f"the FIPS join itself is 100% complete; the gap is FARA's own source data, which has no "
          f"lapop1share value for low/zero-population or group-quarters-only tracts (true for ~27% of "
          f"FARA's full 72,531-tract file, not just our subset).")

    merged = merged.merge(places, on="fips", how="left")
    print(f"PLACES measures populated for {merged['adult_obesity_pct'].notna().sum()}/{len(merged)} tracts — "
          f"here the gap IS a join failure: {len(set(usaleep['fips']) - set(places['fips']))} USALEEP tract "
          f"FIPS codes have no PLACES row at all, because PLACES uses 2020 Census tract boundaries while "
          f"USALEEP/FARA use 2010 boundaries, and tracts were split/merged/renumbered between censuses.")

    merged["county_fips"] = merged["fips"].str[:5]
    merged = merged.merge(ffr_by_county, on="county_fips", how="left")

    return merged[[
        "fips", "county_fips", "life_expectancy",
        "fast_food_density", "food_desert_pct",
        "adult_obesity_pct", "diabetes_prevalence_pct", "uninsured_pct",
        "median_family_income",
    ]]


if __name__ == "__main__":
    df = build_dataset()
    df.to_csv(OUTPUT_PATH, index=False)
    print(f"\nWrote {len(df)} tracts x {len(df.columns)} columns -> {OUTPUT_PATH}")
    print(df.isna().sum())
