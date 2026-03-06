"""Build unified five-district production + weather panel."""

import pandas as pd
import numpy as np
import os

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
repo_root = os.path.dirname(base_dir)
prod_path = os.path.join(repo_root, 'data', 'multidistrict_production_area_series.csv')
weather_path = os.path.join(repo_root, 'data', 'multidistrict_weather_aggregated.csv')
output_dir = os.path.join(base_dir, 'multidistrict', 'data')
os.makedirs(output_dir, exist_ok=True)

print("="*80)
print("MULTIDISTRICT DATA PREPARATION")
print("="*80)

# Load production data
df_prod = pd.read_csv(prod_path)
df_prod['Crop_Year_Norm'] = df_prod['Crop_Year'].str.replace(' ', '').str.strip()
df_prod['Year_Start'] = pd.to_numeric(df_prod['Year_Start'], errors='coerce')
df_prod = df_prod.dropna(subset=['Year_Start'])
print(f"Production: {len(df_prod)} rows, districts: {df_prod['District'].unique().tolist()}")

# Load weather data
df_weather = pd.read_csv(weather_path)
df_weather['Crop_Year_Norm'] = df_weather['Crop_Year'].str.replace(' ', '').str.strip()
print(f"Weather: {len(df_weather)} rows")

# Merge by District and Crop_Year
df = pd.merge(
    df_prod,
    df_weather,
    on=['District', 'Crop_Year_Norm'],
    how='left',
    suffixes=('', '_w')
)
df = df[[c for c in df.columns if not c.endswith('_w')]]
print(f"Merged: {len(df)} rows, matched weather: {df['RAINFALL_MM_Total'].notna().sum()}")

# Derived variables
def assign_period(yr):
    if pd.isna(yr): return 'Other'
    yr = int(yr)
    if 1999 <= yr <= 2006: return 'Period 1 (1999-2006)'
    elif 2006 < yr <= 2013: return 'Period 2 (2006-2013)'
    elif 2013 < yr <= 2020: return 'Period 3 (2013-2020)'
    return 'Other'

df['Period'] = df['Year_Start'].apply(assign_period)

# YoY changes within district
df = df.sort_values(['District', 'Year_Start'])
for col in ['Area_hectares', 'Production_tonnes', 'Yield_tonnes_per_hectare']:
    df[f'{col}_Change_Pct'] = df.groupby('District')[col].pct_change() * 100

# 5-year moving average within district
df['Yield_MA5'] = df.groupby('District')['Yield_tonnes_per_hectare'].transform(
    lambda x: x.rolling(5, center=True, min_periods=1).mean()
)
df['Area_MA5'] = df.groupby('District')['Area_hectares'].transform(
    lambda x: x.rolling(5, center=True, min_periods=1).mean()
)
df['Production_MA5'] = df.groupby('District')['Production_tonnes'].transform(
    lambda x: x.rolling(5, center=True, min_periods=1).mean()
)

# Scale category
def scale_cat(a):
    if a < 50000: return 'Small (<50k ha)'
    elif a < 100000: return 'Medium (50-100k ha)'
    return 'Large (>=100k ha)'
df['Scale_Category'] = df['Area_hectares'].apply(scale_cat)

# Weather quality flag
df['Weather_High_Quality'] = (df['Core_Weather_Completeness_Pct'] >= 80) & (df['Completeness_Pct'] >= 80)

# Save
out_path = os.path.join(output_dir, 'multidistrict_panel.csv')
df.to_csv(out_path, index=False)
print(f"\nSaved: {out_path}")
print("="*80)
