"""Paper 1: descriptive statistics, trend analysis, period comparison, variability, extreme events."""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from statsmodels.tsa.seasonal import STL
from statsmodels.stats.anova import anova_lm
from statsmodels.formula.api import ols
import os
import warnings
warnings.filterwarnings('ignore')

plt.style.use('seaborn-v0_8-darkgrid')
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

for dir_path in [figures_dir, tables_dir, results_dir]:
    os.makedirs(dir_path, exist_ok=True)

print("="*80)
print("PAPER 1: COMPLETE ANALYSIS")
print("="*80)

# ============================================================================
# LOAD DATA
# ============================================================================
print("\nLoading data...")
df = pd.read_csv(os.path.join(data_dir, 'paper1_production_prepared.csv'))
df['Year_Start'] = pd.to_numeric(df['Year_Start'])
print(f"Loaded {len(df)} years of data")

# ============================================================================
# 1. DESCRIPTIVE STATISTICS
# ============================================================================
print("\n1. Generating descriptive statistics...")

variables = ['Area_hectares', 'Production_tonnes', 'Yield_tonnes_per_hectare']
var_names = ['Area (ha)', 'Production (tonnes)', 'Yield (t/ha)']

desc_stats = []
for var, name in zip(variables, var_names):
    stats_dict = {
        'Variable': name,
        'Mean': df[var].mean(),
        'Median': df[var].median(),
        'SD': df[var].std(),
        'Min': df[var].min(),
        'Max': df[var].max(),
        'CV_Pct': (df[var].std() / df[var].mean()) * 100,
        'Q1': df[var].quantile(0.25),
        'Q3': df[var].quantile(0.75),
        'IQR': df[var].quantile(0.75) - df[var].quantile(0.25),
    }
    desc_stats.append(stats_dict)

df_desc = pd.DataFrame(desc_stats)
print(df_desc.to_string(index=False))

# Save table
df_desc.to_csv(os.path.join(tables_dir, 'paper1_table1_descriptive_statistics.csv'), index=False)
print(f"\n   Saved: paper1_table1_descriptive_statistics.csv")

# Period-wise statistics
print("\n2. Period-wise descriptive statistics...")
period_stats = []
for period in df['Period'].unique():
    period_data = df[df['Period'] == period]
    for var, name in zip(variables, var_names):
        stats_dict = {
            'Period': period,
            'Variable': name,
            'Mean': period_data[var].mean(),
            'SD': period_data[var].std(),
            'CV_Pct': (period_data[var].std() / period_data[var].mean()) * 100,
            'Min': period_data[var].min(),
            'Max': period_data[var].max(),
        }
        period_stats.append(stats_dict)

df_period = pd.DataFrame(period_stats)
print(df_period.to_string(index=False))

# Save table
df_period.to_csv(os.path.join(tables_dir, 'paper1_table2_period_statistics.csv'), index=False)
print(f"\n   Saved: paper1_table2_period_statistics.csv")

# ============================================================================
# 3. TREND ANALYSIS
# ============================================================================
print("\n3. Trend analysis...")

trend_results = []

for var, name in zip(variables, var_names):
    y = df[var].values
    x = df['Time_Index'].values
    
    # Linear regression
    slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
    
    # Mann-Kendall test
    def mann_kendall_test(y):
        n = len(y)
        s = 0
        for i in range(n-1):
            for j in range(i+1, n):
                s += np.sign(y[j] - y[i])
        
        var_s = n * (n - 1) * (2*n + 5) / 18
        if s > 0:
            z = (s - 1) / np.sqrt(var_s)
        elif s < 0:
            z = (s + 1) / np.sqrt(var_s)
        else:
            z = 0
        
        p_value_mk = 2 * (1 - stats.norm.cdf(abs(z)))
        return z, p_value_mk
    
    z_mk, p_mk = mann_kendall_test(y)
    
    # Sen's slope
    def sens_slope(y):
        n = len(y)
        slopes = []
        for i in range(n):
            for j in range(i+1, n):
                if y[j] != y[i]:
                    slopes.append((y[j] - y[i]) / (j - i))
        return np.median(slopes) if slopes else 0
    
    sens_s = sens_slope(y)
    
    # Growth rate
    growth_rate = ((y[-1] / y[0]) ** (1/(len(y)-1)) - 1) * 100
    
    trend_results.append({
        'Variable': name,
        'Linear_Slope': slope,
        'Linear_R2': r_value**2,
        'Linear_P': p_value,
        'Mann_Kendall_Z': z_mk,
        'Mann_Kendall_P': p_mk,
        'Sens_Slope': sens_s,
        'Growth_Rate_Pct': growth_rate,
    })

df_trend = pd.DataFrame(trend_results)
print(df_trend.to_string(index=False))

# Save table
df_trend.to_csv(os.path.join(tables_dir, 'paper1_table3_trend_analysis.csv'), index=False)
print(f"\n   Saved: paper1_table3_trend_analysis.csv")

# ============================================================================
# 4. TIME SERIES DECOMPOSITION
# ============================================================================
print("\n4. Time series decomposition...")

# STL decomposition for yield
yield_series = df['Yield_tonnes_per_hectare'].values
time_index = df['Time_Index'].values

# Create time series
from statsmodels.tsa.seasonal import STL
ts = pd.Series(yield_series, index=pd.date_range('1999', periods=len(yield_series), freq='Y'))

# STL decomposition (adjust period if needed)
try:
    stl = STL(ts, period=min(5, len(ts)//2), robust=True)
    result = stl.fit()
    
    trend_comp = result.trend.values
    seasonal_comp = result.seasonal.values
    residual_comp = result.resid.values
except:
    # Fallback: simple trend extraction
    trend_comp = np.polyval(np.polyfit(time_index, yield_series, 2), time_index)
    seasonal_comp = np.zeros_like(yield_series)
    residual_comp = yield_series - trend_comp

# Save decomposition
df_decomp = pd.DataFrame({
    'Year': df['Year'],
    'Time_Index': time_index,
    'Observed': yield_series,
    'Trend': trend_comp,
    'Seasonal': seasonal_comp,
    'Residual': residual_comp,
})
df_decomp.to_csv(os.path.join(data_dir, 'paper1_yield_decomposition.csv'), index=False)
print("   Decomposition saved")

# ============================================================================
# 5. PERIOD COMPARISON (ANOVA)
# ============================================================================
print("\n5. Period comparison (ANOVA)...")

anova_results = []

for var, name in zip(variables, var_names):
    # ANOVA
    model = ols(f'{var} ~ C(Period)', data=df).fit()
    anova_table = anova_lm(model, typ=2)
    
    # Kruskal-Wallis (non-parametric)
    periods = df['Period'].unique()
    groups = [df[df['Period'] == p][var].values for p in periods]
    h_stat, p_kw = stats.kruskal(*groups)
    
    # Pairwise comparisons
    period_means = df.groupby('Period')[var].mean()
    
    anova_results.append({
        'Variable': name,
        'ANOVA_F': anova_table.loc['C(Period)', 'F'],
        'ANOVA_P': anova_table.loc['C(Period)', 'PR(>F)'],
        'Kruskal_Wallis_H': h_stat,
        'Kruskal_Wallis_P': p_kw,
        'Period1_Mean': period_means.iloc[0] if len(period_means) > 0 else np.nan,
        'Period2_Mean': period_means.iloc[1] if len(period_means) > 1 else np.nan,
        'Period3_Mean': period_means.iloc[2] if len(period_means) > 2 else np.nan,
    })

df_anova = pd.DataFrame(anova_results)
print(df_anova.to_string(index=False))

# Save table
df_anova.to_csv(os.path.join(tables_dir, 'paper1_table4_period_comparison.csv'), index=False)
print(f"\n   Saved: paper1_table4_period_comparison.csv")

# ============================================================================
# 6. VARIABILITY ANALYSIS
# ============================================================================
print("\n6. Variability analysis...")

variability_results = []

for var, name in zip(variables, var_names):
    y = df[var].values
    
    # Coefficient of variation
    cv = (np.std(y) / np.mean(y)) * 100
    
    # Volatility index (SD of year-over-year changes)
    yoy_changes = np.diff(y) / y[:-1] * 100
    volatility = np.std(yoy_changes)
    
    # Stability: years within ±10% of mean
    mean_val = np.mean(y)
    within_10pct = np.sum(np.abs(y - mean_val) / mean_val <= 0.10)
    stability_pct = (within_10pct / len(y)) * 100
    
    # Autocorrelation
    autocorr = np.corrcoef(y[:-1], y[1:])[0, 1]
    
    variability_results.append({
        'Variable': name,
        'CV_Pct': cv,
        'Volatility_Index': volatility,
        'Stability_Pct': stability_pct,
        'Autocorrelation': autocorr,
    })

df_var = pd.DataFrame(variability_results)
print(df_var.to_string(index=False))

# Save table
df_var.to_csv(os.path.join(tables_dir, 'paper1_table5_variability.csv'), index=False)
print(f"\n   Saved: paper1_table5_variability.csv")

# ============================================================================
# 7. EXTREME YEARS ANALYSIS
# ============================================================================
print("\n7. Extreme years analysis...")

# Top 3 and bottom 3 yield years
top3 = df.nlargest(3, 'Yield_tonnes_per_hectare')[['Year', 'Yield_tonnes_per_hectare', 'Area_hectares', 'Production_tonnes']]
bottom3 = df.nsmallest(3, 'Yield_tonnes_per_hectare')[['Year', 'Yield_tonnes_per_hectare', 'Area_hectares', 'Production_tonnes']]

print("\nTop 3 Yield Years:")
print(top3.to_string(index=False))
print("\nBottom 3 Yield Years:")
print(bottom3.to_string(index=False))

# Save tables
top3.to_csv(os.path.join(tables_dir, 'paper1_table6_top3_yields.csv'), index=False)
bottom3.to_csv(os.path.join(tables_dir, 'paper1_table7_bottom3_yields.csv'), index=False)
print(f"\n   Saved: paper1_table6_top3_yields.csv, paper1_table7_bottom3_yields.csv")

# ============================================================================
# 8. FIGURES
# ============================================================================
print("\n8. Creating publication-quality figures...")

# Figure 1: Temporal Trends
fig, axes = plt.subplots(3, 1, figsize=(12, 10))
fig.suptitle('Temporal Trends in Sugarcane Production (1999-2020)', fontsize=18, fontweight='bold', y=0.995)

for idx, (var, name) in enumerate(zip(variables, var_names)):
    ax = axes[idx]
    ax.plot(df['Year_Start'], df[var], 'o-', linewidth=2.5, markersize=8, label='Observed', color='#2E86AB')
    # Moving average column names
    ma_col = var.replace('_hectares', '_MA5').replace('_tonnes', '_MA5').replace('_tonnes_per_hectare', '_MA5')
    if ma_col in df.columns:
        ax.plot(df['Year_Start'], df[ma_col], '--', linewidth=2, label='5-year MA', color='#A23B72', alpha=0.7)
    
    # Add trend line
    slope, intercept, _, _, _ = stats.linregress(df['Time_Index'], df[var])
    trend_line = intercept + slope * df['Time_Index']
    ax.plot(df['Year_Start'], trend_line, '-', linewidth=2, label='Trend', color='#F18F01', alpha=0.8)
    
    ax.set_xlabel('Year', fontsize=14, fontweight='bold')
    ax.set_ylabel(name, fontsize=14, fontweight='bold')
    ax.set_title(name, fontsize=14, fontweight='bold')
    ax.legend(fontsize=11, frameon=True, fancybox=True, shadow=True)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_xlim(df['Year_Start'].min()-1, df['Year_Start'].max()+1)

plt.tight_layout(rect=[0, 0, 1, 0.98])
plt.savefig(os.path.join(figures_dir, 'paper1_fig1_ahmednagar_temporal_trends.png'), dpi=300, bbox_inches='tight')
print("   Created: paper1_fig1_temporal_trends.png")
plt.close()

# Figure 2: Period Comparison
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle('Period-wise Comparison of Sugarcane Production Variables', fontsize=18, fontweight='bold', y=1.02)

for idx, (var, name) in enumerate(zip(variables, var_names)):
    ax = axes[idx]
    period_data = [df[df['Period'] == p][var].values for p in sorted(df['Period'].unique())]
    bp = ax.boxplot(period_data, labels=sorted(df['Period'].unique()), patch_artist=True)
    
    # Color boxes
    colors = ['#FF6B6B', '#4ECDC4', '#95E1D3']
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    ax.set_ylabel(name, fontsize=13, fontweight='bold')
    ax.set_title(name, fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y', linestyle='--')
    ax.tick_params(axis='x', rotation=45)

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig(os.path.join(figures_dir, 'paper1_fig2_ahmednagar_period_comparison.png'), dpi=300, bbox_inches='tight')
print("   Created: paper1_fig2_period_comparison.png")
plt.close()

# Figure 3: Yield Decomposition
fig, axes = plt.subplots(4, 1, figsize=(12, 10))
fig.suptitle('Time Series Decomposition of Sugarcane Yield', fontsize=18, fontweight='bold', y=0.995)

axes[0].plot(df['Year_Start'], yield_series, 'o-', linewidth=2.5, markersize=6, color='#2E86AB')
axes[0].set_ylabel('Observed (t/ha)', fontsize=13, fontweight='bold')
axes[0].set_title('Observed Yield', fontsize=13, fontweight='bold')
axes[0].grid(True, alpha=0.3, linestyle='--')

axes[1].plot(df['Year_Start'], trend_comp, '-', linewidth=2.5, color='#A23B72')
axes[1].set_ylabel('Trend (t/ha)', fontsize=13, fontweight='bold')
axes[1].set_title('Trend Component', fontsize=13, fontweight='bold')
axes[1].grid(True, alpha=0.3, linestyle='--')

axes[2].plot(df['Year_Start'], seasonal_comp, '-', linewidth=2, color='#F18F01')
axes[2].axhline(y=0, color='k', linestyle='--', linewidth=1)
axes[2].set_ylabel('Seasonal (t/ha)', fontsize=13, fontweight='bold')
axes[2].set_title('Seasonal Component', fontsize=13, fontweight='bold')
axes[2].grid(True, alpha=0.3, linestyle='--')

axes[3].plot(df['Year_Start'], residual_comp, 'o-', linewidth=1.5, markersize=4, color='#6A994E')
axes[3].axhline(y=0, color='k', linestyle='--', linewidth=1)
axes[3].set_xlabel('Year', fontsize=13, fontweight='bold')
axes[3].set_ylabel('Residual (t/ha)', fontsize=13, fontweight='bold')
axes[3].set_title('Residual Component', fontsize=13, fontweight='bold')
axes[3].grid(True, alpha=0.3, linestyle='--')

for ax in axes:
    ax.set_xlim(df['Year_Start'].min()-1, df['Year_Start'].max()+1)

plt.tight_layout(rect=[0, 0, 1, 0.98])
plt.savefig(os.path.join(figures_dir, 'paper1_fig3_ahmednagar_yield_decomposition.png'), dpi=300, bbox_inches='tight')
print("   Created: paper1_fig3_yield_decomposition.png")
plt.close()

# Figure 4: Year-over-Year Changes
fig, axes = plt.subplots(3, 1, figsize=(12, 10))
fig.suptitle('Year-over-Year Percentage Changes', fontsize=18, fontweight='bold', y=0.995)

change_vars = ['Area_Change_Pct', 'Production_Change_Pct', 'Yield_Change_Pct']
change_names = ['Area Change (%)', 'Production Change (%)', 'Yield Change (%)']

for idx, (var, name) in enumerate(zip(change_vars, change_names)):
    ax = axes[idx]
    changes = df[var].dropna()
    years = df.loc[changes.index, 'Year_Start']
    
    colors = ['#FF6B6B' if x < 0 else '#4ECDC4' for x in changes]
    ax.bar(years, changes, color=colors, alpha=0.7, edgecolor='black', linewidth=1)
    ax.axhline(y=0, color='k', linestyle='-', linewidth=1.5)
    ax.set_ylabel(name, fontsize=13, fontweight='bold')
    ax.set_title(name, fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y', linestyle='--')

axes[-1].set_xlabel('Year', fontsize=13, fontweight='bold')

plt.tight_layout(rect=[0, 0, 1, 0.98])
plt.savefig(os.path.join(figures_dir, 'paper1_fig4_ahmednagar_year_over_year_changes.png'), dpi=300, bbox_inches='tight')
print("   Created: paper1_fig4_year_over_year_changes.png")
plt.close()

print("\n" + "="*80)
print("ANALYSIS COMPLETE")
print("="*80)
print(f"\nFigures saved in: {figures_dir}")
print(f"Tables saved in: {tables_dir}")
print(f"Results saved in: {results_dir}")

