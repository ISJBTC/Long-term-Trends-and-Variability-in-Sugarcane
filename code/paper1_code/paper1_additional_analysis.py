"""Breakpoint detection, climate indices, weather context."""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from statsmodels.tsa.stattools import acf
import os
import warnings
warnings.filterwarnings('ignore')

plt.style.use('seaborn-v0_8-darkgrid')
sns = __import__('seaborn')
sns.set_palette("husl")
plt.rcParams.update({
    'font.size': 12,
    'font.weight': 'bold',
    'axes.labelsize': 14,
    'axes.labelweight': 'bold',
    'axes.titlesize': 16,
    'axes.titleweight': 'bold',
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 12,
    'legend.title_fontsize': 14,
    'figure.titlesize': 18,
    'figure.titleweight': 'bold',
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.2,
    'lines.linewidth': 2,
    'axes.linewidth': 1.5,
    'grid.linewidth': 1,
})

base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
data_dir = os.path.join(base_dir, 'data')
figures_dir = os.path.join(base_dir, 'figures')
tables_dir = os.path.join(base_dir, 'tables')
results_dir = os.path.join(base_dir, 'results')

print("="*80)
print("PAPER 1: ADDITIONAL ANALYSIS")
print("="*80)

# ============================================================================
# LOAD DATA
# ============================================================================
print("\nLoading data...")
df = pd.read_csv(os.path.join(data_dir, 'paper1_production_prepared.csv'))
df_weather = pd.read_csv(os.path.join(data_dir, 'paper1_weather_aggregated.csv'))
df_matched = pd.read_csv(os.path.join(data_dir, 'paper1_production_weather_matched.csv'))
print(f"Loaded {len(df)} years of production data")
print(f"Loaded {len(df_weather)} years of weather data")

# ============================================================================
# 1. BREAKPOINT DETECTION
# ============================================================================
print("\n1. Breakpoint detection...")

def pettitt_test(y):
    n = len(y)
    u = np.zeros(n-1)
    for i in range(n-1):
        u[i] = np.sum(np.sign(y[i+1:] - y[i]))
    
    k = np.argmax(np.abs(u))
    k_stat = np.abs(u[k])
    
    # Critical value approximation
    p_value = 2 * np.exp(-6 * k_stat**2 / (n**3 + n**2))
    return k+1, k_stat, min(p_value, 1.0)

breakpoint_results = []

for var in ['Area_hectares', 'Production_tonnes', 'Yield_tonnes_per_hectare']:
    y = df[var].values
    k, stat, p = pettitt_test(y)
    breakpoint_year = df.iloc[k]['Year_Start']
    
    breakpoint_results.append({
        'Variable': var,
        'Breakpoint_Year': breakpoint_year,
        'Breakpoint_Index': k+1,
        'Pettitt_Statistic': stat,
        'P_Value': p,
        'Significant': 'Yes' if p < 0.05 else 'No'
    })

df_breakpoint = pd.DataFrame(breakpoint_results)
print(df_breakpoint.to_string(index=False))

df_breakpoint.to_csv(os.path.join(tables_dir, 'paper1_table8_breakpoint_detection.csv'), index=False)
print(f"\n   Saved: paper1_table8_breakpoint_detection.csv")

# ============================================================================
# 2. CLIMATE INDICES CALCULATION
# ============================================================================
print("\n2. Calculating climate indices...")

climate_indices = []

for idx, row in df_weather.iterrows():
    crop_year = row['Crop_Year']
    
    # Get weather variables
    rainfall = row.get('RAINFALL_MM_Total', np.nan)
    evaporation = row.get('EVAPORATION_MM_Total', np.nan)
    evapotranspiration = row.get('EVAPOTRANSPIRATION_MM_Total', np.nan)
    mean_temp = row.get('MEAN_TEMP_C_Mean', np.nan)
    max_temp = row.get('MAX_TEMP_C_Mean', np.nan)
    min_temp = row.get('MIN_TEMP_C_Mean', np.nan)
    
    # Drought Index
    if not pd.isna(rainfall) and not pd.isna(evaporation) and evaporation > 0:
        drought_index = (rainfall - evaporation) / evaporation
    else:
        drought_index = np.nan
    
    # Moisture Adequacy Index
    if not pd.isna(rainfall) and not pd.isna(evapotranspiration) and evapotranspiration > 0:
        mai = rainfall / evapotranspiration
    else:
        mai = np.nan
    
    # Water Balance
    if not pd.isna(rainfall) and not pd.isna(evaporation) and not pd.isna(evapotranspiration):
        water_balance = rainfall - evaporation - evapotranspiration
    else:
        water_balance = np.nan
    
    climate_indices.append({
        'Crop_Year': crop_year,
        'Rainfall_Total_mm': rainfall,
        'Evaporation_Total_mm': evaporation,
        'Evapotranspiration_Total_mm': evapotranspiration,
        'Mean_Temp_C': mean_temp,
        'Drought_Index': drought_index,
        'Moisture_Adequacy_Index': mai,
        'Water_Balance_mm': water_balance,
    })

df_climate = pd.DataFrame(climate_indices)
print(df_climate.to_string(index=False))

# Merge with production data
df_climate_merged = pd.merge(df, df_climate, left_on='Year', right_on='Crop_Year', how='left')
df_climate_merged.to_csv(os.path.join(data_dir, 'paper1_climate_indices.csv'), index=False)
print(f"\n   Saved: paper1_climate_indices.csv")

# Save table
df_climate.to_csv(os.path.join(tables_dir, 'paper1_table9_climate_indices.csv'), index=False)
print(f"   Saved: paper1_table9_climate_indices.csv")

# ============================================================================
# 3. WEATHER CONTEXT FOR PERIODS
# ============================================================================
print("\n3. Weather context for periods...")

# Calculate period-wise weather statistics
period_weather = []

for period in df['Period'].unique():
    period_data = df[df['Period'] == period]
    period_years = period_data['Year'].tolist()
    
    # Get weather data for this period
    period_weather_data = df_weather[df_weather['Crop_Year'].isin(period_years)]
    
    if len(period_weather_data) > 0:
        period_weather.append({
            'Period': period,
            'Years_Count': len(period_data),
            'Weather_Years_Count': len(period_weather_data),
            'Mean_Rainfall_mm': period_weather_data['RAINFALL_MM_Total'].mean() if 'RAINFALL_MM_Total' in period_weather_data.columns else np.nan,
            'Mean_Temp_C': period_weather_data['MEAN_TEMP_C_Mean'].mean() if 'MEAN_TEMP_C_Mean' in period_weather_data.columns else np.nan,
            'Mean_Yield_t_ha': period_data['Yield_tonnes_per_hectare'].mean(),
        })

df_period_weather = pd.DataFrame(period_weather)
print(df_period_weather.to_string(index=False))

df_period_weather.to_csv(os.path.join(tables_dir, 'paper1_table10_period_weather_context.csv'), index=False)
print(f"\n   Saved: paper1_table10_period_weather_context.csv")

# ============================================================================
# 4. EXTREME YEARS WEATHER CONTEXT
# ============================================================================
print("\n4. Extreme years weather context...")

# Top 3 and bottom 3 years
top3_years = df.nlargest(3, 'Yield_tonnes_per_hectare')['Year'].tolist()
bottom3_years = df.nsmallest(3, 'Yield_tonnes_per_hectare')['Year'].tolist()

extreme_weather = []

for year in top3_years + bottom3_years:
    prod_data = df[df['Year'] == year].iloc[0]
    weather_data = df_weather[df_weather['Crop_Year'] == year]
    
    if len(weather_data) > 0:
        weather_row = weather_data.iloc[0]
        extreme_weather.append({
            'Year': year,
            'Yield_t_ha': prod_data['Yield_tonnes_per_hectare'],
            'Category': 'Top 3' if year in top3_years else 'Bottom 3',
            'Rainfall_mm': weather_row.get('RAINFALL_MM_Total', np.nan),
            'Mean_Temp_C': weather_row.get('MEAN_TEMP_C_Mean', np.nan),
            'Drought_Index': df_climate[df_climate['Crop_Year'] == year]['Drought_Index'].values[0] if len(df_climate[df_climate['Crop_Year'] == year]) > 0 else np.nan,
        })

df_extreme_weather = pd.DataFrame(extreme_weather)
print(df_extreme_weather.to_string(index=False))

df_extreme_weather.to_csv(os.path.join(tables_dir, 'paper1_table11_extreme_years_weather.csv'), index=False)
print(f"\n   Saved: paper1_table11_extreme_years_weather.csv")

# ============================================================================
# 5. ADDITIONAL FIGURES
# ============================================================================
print("\n5. Creating additional figures...")

# Figure 5: Climate Indices vs Yield
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Climate Indices and Yield Relationships', fontsize=18, fontweight='bold', y=0.995)

df_plot = df_climate_merged.dropna(subset=['Yield_tonnes_per_hectare', 'Drought_Index'])

if len(df_plot) > 0:
    ax = axes[0, 0]
    ax.scatter(df_plot['Drought_Index'], df_plot['Yield_tonnes_per_hectare'], 
               s=100, alpha=0.7, edgecolors='black', linewidth=1.5, color='#2E86AB')
    # Trend line
    z = np.polyfit(df_plot['Drought_Index'], df_plot['Yield_tonnes_per_hectare'], 1)
    p = np.poly1d(z)
    ax.plot(df_plot['Drought_Index'], p(df_plot['Drought_Index']), 'r--', linewidth=2, alpha=0.8)
    ax.set_xlabel('Drought Index', fontsize=13, fontweight='bold')
    ax.set_ylabel('Yield (t/ha)', fontsize=13, fontweight='bold')
    ax.set_title('Drought Index vs Yield', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3, linestyle='--')

df_plot = df_climate_merged.dropna(subset=['Yield_tonnes_per_hectare', 'Moisture_Adequacy_Index'])
if len(df_plot) > 0:
    ax = axes[0, 1]
    ax.scatter(df_plot['Moisture_Adequacy_Index'], df_plot['Yield_tonnes_per_hectare'], 
               s=100, alpha=0.7, edgecolors='black', linewidth=1.5, color='#A23B72')
    z = np.polyfit(df_plot['Moisture_Adequacy_Index'], df_plot['Yield_tonnes_per_hectare'], 1)
    p = np.poly1d(z)
    ax.plot(df_plot['Moisture_Adequacy_Index'], p(df_plot['Moisture_Adequacy_Index']), 'r--', linewidth=2, alpha=0.8)
    ax.set_xlabel('Moisture Adequacy Index', fontsize=13, fontweight='bold')
    ax.set_ylabel('Yield (t/ha)', fontsize=13, fontweight='bold')
    ax.set_title('Moisture Adequacy vs Yield', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3, linestyle='--')

df_plot = df_climate_merged.dropna(subset=['Yield_tonnes_per_hectare', 'Rainfall_Total_mm'])
if len(df_plot) > 0:
    ax = axes[1, 0]
    ax.scatter(df_plot['Rainfall_Total_mm'], df_plot['Yield_tonnes_per_hectare'], 
               s=100, alpha=0.7, edgecolors='black', linewidth=1.5, color='#F18F01')
    z = np.polyfit(df_plot['Rainfall_Total_mm'], df_plot['Yield_tonnes_per_hectare'], 1)
    p = np.poly1d(z)
    ax.plot(df_plot['Rainfall_Total_mm'], p(df_plot['Rainfall_Total_mm']), 'r--', linewidth=2, alpha=0.8)
    ax.set_xlabel('Total Rainfall (mm)', fontsize=13, fontweight='bold')
    ax.set_ylabel('Yield (t/ha)', fontsize=13, fontweight='bold')
    ax.set_title('Rainfall vs Yield', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3, linestyle='--')

df_plot = df_climate_merged.dropna(subset=['Yield_tonnes_per_hectare', 'Mean_Temp_C'])
if len(df_plot) > 0:
    ax = axes[1, 1]
    ax.scatter(df_plot['Mean_Temp_C'], df_plot['Yield_tonnes_per_hectare'], 
               s=100, alpha=0.7, edgecolors='black', linewidth=1.5, color='#6A994E')
    z = np.polyfit(df_plot['Mean_Temp_C'], df_plot['Yield_tonnes_per_hectare'], 1)
    p = np.poly1d(z)
    ax.plot(df_plot['Mean_Temp_C'], p(df_plot['Mean_Temp_C']), 'r--', linewidth=2, alpha=0.8)
    ax.set_xlabel('Mean Temperature (°C)', fontsize=13, fontweight='bold')
    ax.set_ylabel('Yield (t/ha)', fontsize=13, fontweight='bold')
    ax.set_title('Temperature vs Yield', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3, linestyle='--')

plt.tight_layout(rect=[0, 0, 1, 0.98])
plt.savefig(os.path.join(figures_dir, 'paper1_fig5_climate_yield_relationships.png'), dpi=300, bbox_inches='tight')
print("   Created: paper1_fig5_climate_yield_relationships.png")
plt.close()

# Figure 6: Weather Context for Periods
if len(df_period_weather) > 0:
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle('Weather Context for Production Periods', fontsize=18, fontweight='bold', y=1.02)
    
    periods = df_period_weather['Period'].tolist()
    rainfall = df_period_weather['Mean_Rainfall_mm'].tolist()
    yield_vals = df_period_weather['Mean_Yield_t_ha'].tolist()
    
    ax = axes[0]
    ax.bar(periods, rainfall, alpha=0.7, edgecolor='black', linewidth=1.5, color='#4ECDC4')
    ax.set_ylabel('Mean Rainfall (mm)', fontsize=13, fontweight='bold')
    ax.set_title('Mean Rainfall by Period', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y', linestyle='--')
    ax.tick_params(axis='x', rotation=45)
    
    ax = axes[1]
    ax.bar(periods, yield_vals, alpha=0.7, edgecolor='black', linewidth=1.5, color='#FF6B6B')
    ax.set_ylabel('Mean Yield (t/ha)', fontsize=13, fontweight='bold')
    ax.set_title('Mean Yield by Period', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y', linestyle='--')
    ax.tick_params(axis='x', rotation=45)
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(os.path.join(figures_dir, 'paper1_fig6_period_weather_context.png'), dpi=300, bbox_inches='tight')
    print("   Created: paper1_fig6_period_weather_context.png")
    plt.close()

print("\n" + "="*80)
print("ADDITIONAL ANALYSIS COMPLETE")
print("="*80)



