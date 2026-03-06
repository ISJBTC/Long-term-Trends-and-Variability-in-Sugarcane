"""Build crop-year weather aggregates for Bellary and Dharwad from AGRIMET_SM daily files."""

import os
from datetime import datetime

import numpy as np
import pandas as pd


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPO_ROOT = os.path.dirname(BASE_DIR)

RAW_FILES = [
    {
        "district": "Bellary",
        "station_name": "Bellary",
        "station_abbr": "BLY",
        "path": os.path.join(REPO_ROOT, "data", "agromet_data", "AGRIMET_SM_Daily_Bellary.txt"),
    },
    {
        "district": "Dharwad",
        "station_name": "Dharwad (ET)",
        "station_abbr": "DHR",
        "path": os.path.join(REPO_ROOT, "data", "agromet_data", "AGRIMET_SM_Daily_Dharwad.txt"),
    },
]

WEATHER_AGG_PATH = os.path.join(REPO_ROOT, "data", "multidistrict_weather_aggregated.csv")


def parse_sm_daily(path: str, station_abbr: str, district: str) -> pd.DataFrame:
    rows = []
    header_seen = False

    with open(path, "r", encoding="latin-1") as f:
        for line in f:
            if not line.strip():
                continue
            # Skip explanation and separator lines
            if line.startswith("LIST OF") or line.startswith("---"):
                continue
            if line.lstrip().startswith("STATION"):
                # station line, skip
                continue
            if line.strip().startswith("INDEX"):
                header_seen = True
                continue
            if not header_seen:
                continue

            parts = line.split()
            if len(parts) < 8:
                continue
            # First four tokens: index, year, month, day
            try:
                index = int(parts[0])
                year = int(parts[1])
                month = int(parts[2])
                day = int(parts[3])
            except ValueError:
                continue

            weather_tokens = parts[4:]
            # Need at least RF, EVP, ET, RH1, RH2, MAX, MIN, MTMP
            if len(weather_tokens) < 8:
                continue

            def to_float(x):
                try:
                    return float(x)
                except Exception:
                    return np.nan

            # RF, EVP, ET are always first three tokens
            rf = to_float(weather_tokens[0])
            evp = to_float(weather_tokens[1])
            et = to_float(weather_tokens[2])

            # If SSH is reported, total weather_tokens after the first 4 tokens is 9.
            # Otherwise SSH is omitted and total length is 8.
            if len(weather_tokens) == 9:
                ssh = to_float(weather_tokens[3])
                offset = 1
            else:
                ssh = np.nan
                offset = 0

            rh1 = to_float(weather_tokens[3 + offset])
            rh2 = to_float(weather_tokens[4 + offset])
            tmax = to_float(weather_tokens[5 + offset])
            tmin = to_float(weather_tokens[6 + offset])
            tmean = to_float(weather_tokens[7 + offset])

            try:
                date = datetime(year, month, day)
            except ValueError:
                continue

            rows.append(
                {
                    "District": district,
                    "Station_Abbr": station_abbr,
                    "DATE": date,
                    "YEAR": year,
                    "RF": rf,
                    "EVP": evp,
                    "ET": et,
                    "SSH": ssh,
                    "RH1": rh1,
                    "RH2": rh2,
                    "MAX": tmax,
                    "MIN": tmin,
                    "MTMP": tmean,
                }
            )

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    return df


def aggregate_to_crop_year(df_daily: pd.DataFrame, district: str, station_name: str, station_abbr: str) -> pd.DataFrame:
    if df_daily.empty:
        return pd.DataFrame()

    # Determine year range from daily data
    years = sorted(df_daily["YEAR"].unique())
    if not years:
        return pd.DataFrame()

    start_year = min(years)
    end_year = max(years)

    aggregates = []

    for y in range(start_year, end_year):
        crop_start = datetime(y, 10, 1)
        crop_end = datetime(y + 1, 9, 30)

        mask = (df_daily["DATE"] >= crop_start) & (df_daily["DATE"] <= crop_end)
        sub = df_daily.loc[mask].copy()
        total_days = (crop_end - crop_start).days + 1
        available_days = len(sub)

        if available_days == 0:
            continue

        completeness = (available_days / total_days) * 100.0

        agg = {
            "District": district,
            "Station": station_name,
            "Station_Abbr": station_abbr,
            "Crop_Year": f"{y}-{y+1}",
            "Start_Year": y,
            "End_Year": y + 1,
            "Start_Date": crop_start.date().isoformat(),
            "End_Date": crop_end.date().isoformat(),
            "Total_Days": total_days,
            "Available_Days": available_days,
            "Completeness_Pct": completeness,
        }

        # Helper to aggregate totals/means/counts
        def add_total_mean_count(col_name, out_prefix, total=False):
            if col_name not in sub.columns:
                return
            series = pd.to_numeric(sub[col_name], errors="coerce")
            if total:
                agg[f"{out_prefix}_Total"] = float(series.sum(skipna=True))
            agg[f"{out_prefix}_Mean"] = float(series.mean(skipna=True))
            agg[f"{out_prefix}_Count"] = int(series.notna().sum())

        # Rainfall, evaporation, ET
        add_total_mean_count("RF", "RAINFALL_MM", total=True)
        add_total_mean_count("EVP", "EVAPORATION_MM", total=True)
        add_total_mean_count("ET", "EVAPOTRANSPIRATION_MM", total=True)

        # Sunshine hours (mean only)
        add_total_mean_count("SSH", "SUNSHINE_HOURS", total=False)

        # Temperatures
        add_total_mean_count("MAX", "MAX_TEMP_C", total=False)
        add_total_mean_count("MIN", "MIN_TEMP_C", total=False)
        add_total_mean_count("MTMP", "MEAN_TEMP_C", total=False)

        # Humidity
        add_total_mean_count("RH1", "RH_0700_PERCENT", total=False)
        add_total_mean_count("RH2", "RH_1400_PERCENT", total=False)

        # Core weather completeness: minimum of variable-specific availability across key variables
        key_vars = {
            "RF": "RAINFALL_MM",
            "EVP": "EVAPORATION_MM",
            "ET": "EVAPOTRANSPIRATION_MM",
            "MAX": "MAX_TEMP_C",
            "MIN": "MIN_TEMP_C",
            "MTMP": "MEAN_TEMP_C",
        }
        completeness_vals = []
        for col, prefix in key_vars.items():
            if f"{prefix}_Count" in agg:
                completeness_vals.append(agg[f"{prefix}_Count"] / total_days * 100.0)
        core_comp = min(completeness_vals) if completeness_vals else completeness
        agg["Core_Weather_Completeness_Pct"] = core_comp

        # Heat and rainfall event counts
        series_rf = pd.to_numeric(sub["RF"], errors="coerce")
        series_max = pd.to_numeric(sub["MAX"], errors="coerce")
        agg["Heat_Days_Tmax_ge_35C"] = int((series_max >= 35.0).sum(skipna=True))
        agg["Heat_Days_Tmax_ge_38C"] = int((series_max >= 38.0).sum(skipna=True))
        agg["Dry_Days_Rain_lt_1mm"] = int((series_rf < 1.0).sum(skipna=True))
        agg["Heavy_Rain_Days_Rain_ge_50mm"] = int((series_rf >= 50.0).sum(skipna=True))

        aggregates.append(agg)

    return pd.DataFrame(aggregates)


def main():
    print("=" * 80)
    print("KARNATAKA AGRIMET SM: BUILD WEATHER AGGREGATES FOR BELLARY AND DHARWAD")
    print("=" * 80)

    all_agg = []
    for meta in RAW_FILES:
        print(f"\nParsing daily SM file for {meta['district']} from {meta['path']} ...")
        df_daily = parse_sm_daily(meta["path"], meta["station_abbr"], meta["district"])
        print(f"  Parsed {len(df_daily)} daily rows.")
        if df_daily.empty:
            continue
        df_agg = aggregate_to_crop_year(
            df_daily, meta["district"], meta["station_name"], meta["station_abbr"]
        )
        print(f"  Aggregated to {len(df_agg)} crop years.")
        all_agg.append(df_agg)

    if not all_agg:
        print("No aggregates created; nothing to append.")
        return

    new_weather = pd.concat(all_agg, ignore_index=True)

    # Load existing weather aggregates and append, avoiding duplicates
    print(f"\nLoading existing weather aggregates from: {WEATHER_AGG_PATH}")
    existing = pd.read_csv(WEATHER_AGG_PATH)

    # Identify rows in new_weather that are not already present (by District + Crop_Year)
    key_cols = ["District", "Crop_Year"]
    existing_keys = set(zip(existing["District"], existing["Crop_Year"]))
    mask_new = ~new_weather.apply(lambda r: (r["District"], r["Crop_Year"]) in existing_keys, axis=1)
    to_add = new_weather.loc[mask_new].copy()

    print(f"Existing rows: {len(existing)}, new rows: {len(new_weather)}, to append: {len(to_add)}")
    combined = pd.concat([existing, to_add], ignore_index=True)

    # Ensure column order roughly matches existing file; if new columns appear, they will be appended.
    combined.to_csv(WEATHER_AGG_PATH, index=False)
    print(f"Updated weather aggregates written to: {WEATHER_AGG_PATH}")
    print("=" * 80)


if __name__ == "__main__":
    main()

