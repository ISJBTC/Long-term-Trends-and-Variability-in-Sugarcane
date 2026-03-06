"""Load and prepare production and weather data."""

import pandas as pd
import numpy as np
from datetime import datetime
import os

base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
production_data_path = os.path.join(base_dir, 'data', 'sugarcane_production_data.csv')
weather_data_path = os.path.join(base_dir, 'data', 'rahuri_agromet_daily.csv')
output_dir = os.path.join(base_dir, 'data')

# Create output directory if not exists
os.makedirs(output_dir, exist_ok=True)

print("="*80)
print("PAPER 1: DATA PREPARATION")
print("="*80)

# ============================================================================
# 1. LOAD PRODUCTION DATA
# ============================================================================
print("\n1. Loading production data...")
df_prod = pd.read_csv(production_data_path)
print(f"   Loaded {len(df_prod)} years of production data")
print(f"   Years: {df_prod['Year'].min()} to {df_prod['Year'].max()}")

# Convert crop year to numeric year for sorting/analysis
def extract_start_year(year_str):
    return int(year_str.split('-')[0])

df_prod['Year_Start'] = df_prod['Year'].apply(extract_start_year)
df_prod = df_prod.sort_values('Year_Start').reset_index(drop=True)

# Verify data quality
print("\n   Data Quality Check:")
print(f"   - Missing values: {df_prod.isnull().sum().sum()}")
print(f"   - Area range: {df_prod['Area_hectares'].min():,.0f} to {df_prod['Area_hectares'].max():,.0f} ha")
print(f"   - Production range: {df_prod['Production_tonnes'].min():,.0f} to {df_prod['Production_tonnes'].max():,.0f} tonnes")
print(f"   - Yield range: {df_prod['Yield_tonnes_per_hectare'].min():.2f} to {df_prod['Yield_tonnes_per_hectare'].max():.2f} t/ha")

# Create derived variables
print("\n2. Creating derived variables...")

# Year-over-year changes
df_prod['Area_Change_Pct'] = df_prod['Area_hectares'].pct_change() * 100
df_prod['Production_Change_Pct'] = df_prod['Production_tonnes'].pct_change() * 100
df_prod['Yield_Change_Pct'] = df_prod['Yield_tonnes_per_hectare'].pct_change() * 100

# 5-year moving averages
df_prod['Area_MA5'] = df_prod['Area_hectares'].rolling(window=5, center=True, min_periods=1).mean()
df_prod['Production_MA5'] = df_prod['Production_tonnes'].rolling(window=5, center=True, min_periods=1).mean()
df_prod['Yield_MA5'] = df_prod['Yield_tonnes_per_hectare'].rolling(window=5, center=True, min_periods=1).mean()

# Period assignment
def assign_period(year_start):
    if 1999 <= year_start <= 2006:
        return 'Period 1 (1999-2006)'
    elif 2006 < year_start <= 2013:
        return 'Period 2 (2006-2013)'
    elif 2013 < year_start <= 2020:
        return 'Period 3 (2013-2020)'
    else:
        return 'Other'

df_prod['Period'] = df_prod['Year_Start'].apply(assign_period)

# Scale categories
def assign_scale(area):
    if area < 50000:
        return 'Small (<50k ha)'
    elif area < 100000:
        return 'Medium (50-100k ha)'
    else:
        return 'Large (≥100k ha)'

df_prod['Scale_Category'] = df_prod['Area_hectares'].apply(assign_scale)

# Time index (1 to 22)
df_prod['Time_Index'] = range(1, len(df_prod) + 1)

# Save prepared production data
output_file = os.path.join(output_dir, 'paper1_production_prepared.csv')
df_prod.to_csv(output_file, index=False)
print(f"\n   Saved prepared production data to: {output_file}")

# ============================================================================
# 3. LOAD WEATHER DATA
# ============================================================================
print("\n3. Loading weather data...")
df_weather = pd.read_csv(weather_data_path, low_memory=False)

# Convert date
df_weather['DATE_FULL'] = pd.to_datetime(df_weather['DATE_FULL'], errors='coerce')
df_weather = df_weather.dropna(subset=['DATE_FULL'])

print(f"   Loaded {len(df_weather)} daily weather records")
print(f"   Date range: {df_weather['DATE_FULL'].min()} to {df_weather['DATE_FULL'].max()}")

# ============================================================================
# 4. CROP-YEAR ALIGNMENT FOR WEATHER DATA
# ============================================================================
print("\n4. Aligning weather data to crop years...")

# High-quality years (>80% completeness)
high_quality_years = [
    (2002, 2003), (2003, 2004), (2004, 2005), (2005, 2006),
    (2006, 2007), (2007, 2008), (2009, 2010), (2015, 2016)
]

weather_aggregates = []

for start_year, end_year in high_quality_years:
    # Crop year: Oct-Dec of start_year + Jan-Sep of end_year
    start_date = datetime(start_year, 10, 1)
    end_date = datetime(end_year, 9, 30)
    
    # Filter weather data for this crop year
    mask = (df_weather['DATE_FULL'] >= start_date) & (df_weather['DATE_FULL'] <= end_date)
    crop_year_data = df_weather[mask].copy()
    
    # Calculate completeness
    total_days = (end_date - start_date).days + 1
    available_days = len(crop_year_data)
    completeness = (available_days / total_days) * 100
    
    if completeness < 80:
        continue
    
    # Aggregate weather variables
    agg_data = {
        'Crop_Year': f"{start_year}-{end_year}",
        'Start_Year': start_year,
        'End_Year': end_year,
        'Completeness_Pct': completeness,
        'Total_Days': total_days,
        'Available_Days': available_days,
    }
    
    # Weather aggregates
    weather_vars = ['RAINFALL_MM', 'EVAPORATION_MM', 'EVAPOTRANSPIRATION_MM',
                   'SUNSHINE_HOURS', 'MAX_TEMP_C', 'MIN_TEMP_C', 'MEAN_TEMP_C',
                   'RH_0700_PERCENT', 'RH_1400_PERCENT']
    
    for var in weather_vars:
        if var in crop_year_data.columns:
            # Convert to numeric, handling missing values
            numeric_data = pd.to_numeric(crop_year_data[var], errors='coerce')
            
            if 'TEMP' in var or 'RH' in var or 'SUNSHINE' in var:
                # Mean for temperature, humidity, sunshine
                agg_data[f'{var}_Mean'] = numeric_data.mean()
            else:
                # Sum for rainfall, evaporation
                agg_data[f'{var}_Total'] = numeric_data.sum()
            
            # Count non-null values
            agg_data[f'{var}_Count'] = numeric_data.notna().sum()
    
    weather_aggregates.append(agg_data)

df_weather_agg = pd.DataFrame(weather_aggregates)

# Save weather aggregates
output_file = os.path.join(output_dir, 'paper1_weather_aggregated.csv')
df_weather_agg.to_csv(output_file, index=False)
print(f"\n   Saved weather aggregates to: {output_file}")
print(f"   Aggregated {len(df_weather_agg)} crop years with >80% completeness")

# ============================================================================
# 5. MATCH WEATHER WITH PRODUCTION
# ============================================================================
print("\n5. Matching weather data with production data...")

# Match by crop year
df_weather_agg['Year'] = df_weather_agg['Crop_Year']
df_matched = pd.merge(df_prod, df_weather_agg, on='Year', how='left', suffixes=('', '_weather'))

matched_count = df_matched['Completeness_Pct'].notna().sum()
print(f"   Matched {matched_count} years with weather data")

# Save matched data
output_file = os.path.join(output_dir, 'paper1_production_weather_matched.csv')
df_matched.to_csv(output_file, index=False)
print(f"   Saved matched data to: {output_file}")

print("\n" + "="*80)
print("DATA PREPARATION COMPLETE")
print("="*80)
print(f"\nOutput files saved in: {output_dir}")



