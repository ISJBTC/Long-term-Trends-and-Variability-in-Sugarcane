"""Extended figures: comparative panels and district-specific profiles."""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from scipy import stats
import os
import warnings
warnings.filterwarnings('ignore')

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
repo_root = os.path.dirname(base_dir)
data_dir = os.path.join(base_dir, 'multidistrict', 'data')
fig_dir = os.path.join(repo_root, 'figures')
os.makedirs(fig_dir, exist_ok=True)
paper1_fig = paper2_fig = paper3_fig = fig_dir

plt.rcParams.update({
    'font.size': 12, 'font.weight': 'bold',
    'axes.labelsize': 14, 'axes.labelweight': 'bold',
    'axes.titlesize': 16, 'axes.titleweight': 'bold',
    'figure.dpi': 300, 'savefig.dpi': 300,
    'savefig.bbox': 'tight', 'savefig.pad_inches': 0.2,
})

DISTRICT_COLORS = {
    'Ahmednagar': '#2E86AB',
    'Solapur': '#A23B72',
    'Nashik': '#F18F01',
    'Bellary': '#4CAF50',
    'Dharwad': '#9B59B6',
}
DISTRICTS = ['Ahmednagar', 'Solapur', 'Nashik', 'Bellary', 'Dharwad']
PERIODS = ['Period 1 (1999-2006)', 'Period 2 (2006-2013)', 'Period 3 (2013-2020)']

df = pd.read_csv(os.path.join(data_dir, 'multidistrict_panel.csv'))
df['Year_Start'] = pd.to_numeric(df['Year_Start'], errors='coerce')
df = df.dropna(subset=['Year_Start'])
df_hq = df[df['Weather_High_Quality'] == True].copy()

print("="*80)
print("MULTIDISTRICT: CREATING EXTENDED FIGURES (16 + 19 + 19)")
print("="*80)

# =============================================================================
# PAPER 1: ADD 9 FIGURES (5 comparative + 3 Nashik + 3 Solapur) -> 16 total
# =============================================================================
print("\n--- PAPER 1 EXTENDED ---")

# P1-5: Comparative period boxplots (yield by period, by district)
fig, ax = plt.subplots(figsize=(14, 7))
data_by_dist = []
labels_list = []
colors_list = []
for dist in DISTRICTS:
    for p in PERIODS:
        sub = df[(df['District'] == dist) & (df['Period'] == p)]['Yield_tonnes_per_hectare'].dropna()
        if len(sub) > 0:
            data_by_dist.append(sub.values)
            labels_list.append(f"{dist[:3]}\n{p.split()[1]}")
            colors_list.append(DISTRICT_COLORS.get(dist, '#777777'))
if len(data_by_dist) > 0:
    bp = ax.boxplot(data_by_dist, labels=labels_list, patch_artist=True)
    for i, patch in enumerate(bp['boxes']):
        if i < len(colors_list):
            patch.set_facecolor(colors_list[i])
        else:
            patch.set_facecolor('#777777')
        patch.set_alpha(0.7)
    ax.set_ylabel('Yield (t/ha)', fontweight='bold')
    ax.set_title('Comparative: Yield Distribution by District and Period', fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    ax.tick_params(axis='x', rotation=30)
    plt.tight_layout()
    plt.savefig(os.path.join(paper1_fig, 'paper1_fig5_comparative_period_boxplots.png'), dpi=300, bbox_inches='tight')
    plt.close()
print("  P1-5: Comparative period boxplots")

# P1-6: District ranking by period (mean yield, panelled by period)
rank_data = []
for p in PERIODS:
    sub = df[df['Period'] == p]
    if len(sub) == 0:
        continue
    means = sub.groupby('District')['Yield_tonnes_per_hectare'].mean().sort_values(ascending=False)
    for r, (dist, my) in enumerate(means.items()):
        rank_data.append({'Period': p, 'District': dist, 'Mean_Yield': my, 'Rank': r+1})

if rank_data:
    df_rank = pd.DataFrame(rank_data)
    n_periods = len(PERIODS)
    fig, axes = plt.subplots(1, n_periods, figsize=(4 * n_periods + 2, 5), sharey=True)
    if n_periods == 1:
        axes = [axes]

    for idx, p in enumerate(PERIODS):
        ax = axes[idx]
        sub = df_rank[df_rank['Period'] == p].sort_values('Mean_Yield', ascending=False)
        if sub.empty:
            ax.axis('off')
            continue
        districts_p = sub['District'].tolist()
        x = np.arange(len(districts_p))
        vals = sub['Mean_Yield'].values
        bars = ax.bar(x, vals, color=[DISTRICT_COLORS.get(d, '#333333') for d in districts_p], alpha=0.85)
        # Annotate rank number on top of each bar
        for j, (b, rnk) in enumerate(zip(bars, sub['Rank'].values)):
            ax.text(b.get_x() + b.get_width()/2, b.get_height() + 1, f"{rnk}", ha='center', va='bottom', fontsize=9)
        ax.set_xticks(x)
        ax.set_xticklabels(districts_p, rotation=45, ha='right')
        ax.set_title(p.split('(')[1].rstrip(')'), fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
        if idx == 0:
            ax.set_ylabel('Mean Yield (t/ha)', fontweight='bold')

    # Global legend for colors
    uniq_dists = [d for d in DISTRICTS if d in df_rank['District'].unique()]
    handles = [plt.Rectangle((0, 0), 1, 1, color=DISTRICT_COLORS.get(d, '#333333'), label=d) for d in uniq_dists]
    fig.legend(handles=handles, loc='upper right', fontsize=9)
    fig.suptitle('Comparative: District Ranking by Period (Mean Yield)', fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    plt.savefig(os.path.join(paper1_fig, 'paper1_fig6_comparative_district_ranking.png'), dpi=300, bbox_inches='tight')
    plt.close()
print("  P1-6: Comparative district ranking (panelled)")

# P1-7: YoY change comparison across districts (separate panels for clarity)
n_dist_yoy = len(DISTRICTS)
fig, axes = plt.subplots(n_dist_yoy, 1, figsize=(12, 2.5 * n_dist_yoy + 2), sharex=True)
if n_dist_yoy == 1:
    axes = [axes]
for idx, dist in enumerate(DISTRICTS):
    ax = axes[idx]
    d = df[df['District'] == dist].sort_values('Year_Start')
    if 'Yield_tonnes_per_hectare_Change_Pct' in d.columns and len(d) > 1:
        d = d.dropna(subset=['Yield_tonnes_per_hectare_Change_Pct'])
        if len(d) > 0:
            ax.plot(d['Year_Start'], d['Yield_tonnes_per_hectare_Change_Pct'], 'o-', linewidth=2, markersize=4,
                    color=DISTRICT_COLORS.get(dist, '#333333'))
    ax.axhline(0, color='k', linestyle='--', linewidth=1)
    ax.set_ylabel('YoY change (%)', fontweight='bold')
    ax.set_title(dist, fontweight='bold', loc='left')
    ax.grid(True, alpha=0.3)
axes[-1].set_xlabel('Year', fontweight='bold')
fig.suptitle('Year-over-Year Yield Change by District', fontsize=16, fontweight='bold', y=0.995)
plt.tight_layout(rect=[0, 0, 1, 0.97])
plt.savefig(os.path.join(paper1_fig, 'paper1_fig7_comparative_yoy_yield.png'), dpi=300, bbox_inches='tight')
plt.close()
print("  P1-7: YoY yield by district (panels)")

# P1-8 to 10: District-specific profiles (trend, period stats, extreme years)
for dist in ['Nashik', 'Solapur', 'Bellary', 'Dharwad']:
    d = df[df['District'] == dist].sort_values('Year_Start')
    if len(d) < 3:
        continue
    # Trend profile
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(d['Year_Start'], d['Yield_tonnes_per_hectare'], 'o-', linewidth=2, markersize=6, color=DISTRICT_COLORS[dist])
    if 'Yield_MA5' in d.columns:
        ax.plot(d['Year_Start'], d['Yield_MA5'], '--', linewidth=2, alpha=0.8, color=DISTRICT_COLORS[dist])
    slope, intercept, _, _, _ = stats.linregress(range(len(d)), d['Yield_tonnes_per_hectare'].values)
    ax.plot(d['Year_Start'], intercept + slope * np.arange(len(d)), '-', linewidth=2, color='#333', alpha=0.6, label='Trend')
    ax.set_xlabel('Year', fontweight='bold')
    ax.set_ylabel('Yield (t/ha)', fontweight='bold')
    ax.set_title(f'{dist}: Yield Trend Profile', fontweight='bold')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(paper1_fig, f'paper1_fig_{dist.lower()}_trend_profile.png'), dpi=300, bbox_inches='tight')
    plt.close()
    # Period statistics
    fig, ax = plt.subplots(figsize=(10, 5))
    pmeans = [d[d['Period']==p]['Yield_tonnes_per_hectare'].mean() for p in PERIODS if len(d[d['Period']==p])>0]
    plabs = [p.split('(')[1].rstrip(')') for p in PERIODS if len(d[d['Period']==p])>0]
    if len(pmeans) > 0:
        ax.bar(plabs, pmeans, color=DISTRICT_COLORS[dist], alpha=0.85, edgecolor='black')
        ax.set_ylabel('Mean Yield (t/ha)', fontweight='bold')
        ax.set_title(f'{dist}: Period-wise Mean Yield', fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()
        plt.savefig(os.path.join(paper1_fig, f'paper1_fig_{dist.lower()}_period_statistics.png'), dpi=300, bbox_inches='tight')
        plt.close()
    # Extreme years
    top3 = d.nlargest(3, 'Yield_tonnes_per_hectare')
    bot3 = d.nsmallest(3, 'Yield_tonnes_per_hectare')
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.barh(range(3), top3['Yield_tonnes_per_hectare'].values, color='#4ECDC4', alpha=0.8, label='Top 3')
    ax.set_yticks(range(3))
    ax.set_yticklabels([str(int(y)) for y in top3['Year_Start'].values])
    ax.set_xlabel('Yield (t/ha)', fontweight='bold')
    ax.set_title(f'{dist}: Top 3 Yield Years', fontweight='bold')
    ax.grid(True, alpha=0.3, axis='x')
    plt.tight_layout()
    plt.savefig(os.path.join(paper1_fig, f'paper1_fig_{dist.lower()}_extreme_years.png'), dpi=300, bbox_inches='tight')
    plt.close()
print("  P1-8 to 10: Nashik + Solapur trend, period stats, extremes")

# =============================================================================
# PAPER 2: ADD 9 FIGURES (3 comparative + 3 Nashik + 3 Solapur) -> 19 total
# =============================================================================
print("\n--- PAPER 2 EXTENDED ---")

# P2-9: Comparative phase-wise weather (rainfall by month/quarter if available, else by district)
fig, ax = plt.subplots(figsize=(12, 6))
for dist in DISTRICTS:
    d = df_hq[df_hq['District'] == dist]
    if 'RAINFALL_MM_Total' in d.columns and d['RAINFALL_MM_Total'].notna().any() and len(d) > 2:
        d = d.sort_values('Year_Start')
        ax.plot(d['Year_Start'], d['RAINFALL_MM_Total'], 'o-', linewidth=2, markersize=5,
                label=dist, color=DISTRICT_COLORS[dist])
ax.set_xlabel('Crop Year', fontweight='bold')
ax.set_ylabel('Seasonal Rainfall (mm)', fontweight='bold')
ax.set_title('Comparative: Seasonal Rainfall Time Series by District', fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(paper2_fig, 'paper2_fig9_comparative_rainfall_timeseries.png'), dpi=300, bbox_inches='tight')
plt.close()
print("  P2-9: Comparative rainfall timeseries")

# P2-10: Weather-yield correlation heatmap by district (simplified)
corr_vars = ['Yield_tonnes_per_hectare', 'RAINFALL_MM_Total', 'Heat_Days_Tmax_ge_35C', 'MEAN_TEMP_C_Mean']
corr_vars = [v for v in corr_vars if v in df_hq.columns]
if len(corr_vars) >= 2:
    n_dist = len(DISTRICTS)
    fig, axes = plt.subplots(1, n_dist, figsize=(4 * n_dist, 5))
    if n_dist == 1:
        axes = [axes]
    im = None
    for idx, dist in enumerate(DISTRICTS):
        d = df_hq[df_hq['District'] == dist][corr_vars].apply(pd.to_numeric, errors='coerce').dropna()
        if len(d) >= 3:
            corr = d.corr()
            im = axes[idx].imshow(corr, cmap='coolwarm', aspect='auto', vmin=-1, vmax=1)
            axes[idx].set_xticks(range(len(corr_vars)))
            axes[idx].set_yticks(range(len(corr_vars)))
            axes[idx].set_xticklabels([v[:12] for v in corr_vars], rotation=45, ha='right')
            axes[idx].set_yticklabels([v[:12] for v in corr_vars])
            axes[idx].set_title(f'{dist}', fontweight='bold')
        else:
            axes[idx].axis('off')
    if im is not None:
        plt.suptitle('Comparative: Weather-Yield Correlation by District', fontweight='bold', y=1.02)
        plt.colorbar(im, ax=axes, shrink=0.6, label='Correlation')
    plt.tight_layout()
    plt.savefig(os.path.join(paper2_fig, 'paper2_fig10_comparative_correlation_by_district.png'), dpi=300, bbox_inches='tight')
    plt.close()
print("  P2-10: Comparative correlation by district")

# P2-11: Heat stress days comparison
if 'Heat_Days_Tmax_ge_35C' in df_hq.columns:
    fig, ax = plt.subplots(figsize=(12, 6))
    for dist in DISTRICTS:
        d = df_hq[df_hq['District'] == dist].sort_values('Year_Start')
        d = d.dropna(subset=['Heat_Days_Tmax_ge_35C'])
        if len(d) > 0:
            ax.plot(d['Year_Start'], d['Heat_Days_Tmax_ge_35C'], 'o-', linewidth=2, markersize=5,
                    label=dist, color=DISTRICT_COLORS[dist])
    ax.set_xlabel('Crop Year', fontweight='bold')
    ax.set_ylabel('Heat Stress Days (Tmax >= 35C)', fontweight='bold')
    ax.set_title('Comparative: Heat Stress Days by District', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(paper2_fig, 'paper2_fig11_comparative_heat_stress.png'), dpi=300, bbox_inches='tight')
    plt.close()
print("  P2-11: Comparative heat stress")

# P2-12 to 18: District-specific weather-yield and rainfall/heat series
for dist in ['Nashik', 'Solapur', 'Bellary', 'Dharwad']:
    d = df_hq[df_hq['District'] == dist]
    if len(d) < 5 or 'RAINFALL_MM_Total' not in d.columns:
        continue
    d = d.dropna(subset=['Yield_tonnes_per_hectare', 'RAINFALL_MM_Total'])
    if len(d) < 5:
        continue
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(d['RAINFALL_MM_Total'], d['Yield_tonnes_per_hectare'], s=100, alpha=0.8, 
               color=DISTRICT_COLORS[dist], edgecolors='black', linewidth=1)
    z = np.polyfit(d['RAINFALL_MM_Total'], d['Yield_tonnes_per_hectare'], 1)
    xl = np.linspace(d['RAINFALL_MM_Total'].min(), d['RAINFALL_MM_Total'].max(), 50)
    ax.plot(xl, np.poly1d(z)(xl), '--', linewidth=2, color='#333', alpha=0.8)
    ax.set_xlabel('Seasonal Rainfall (mm)', fontweight='bold')
    ax.set_ylabel('Yield (t/ha)', fontweight='bold')
    ax.set_title(f'{dist}: Yield vs Seasonal Rainfall', fontweight='bold')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(paper2_fig, f'paper2_fig_{dist.lower()}_yield_rainfall.png'), dpi=300, bbox_inches='tight')
    plt.close()
    # P2 extra: Seasonal rainfall timeseries
    d2 = df_hq[df_hq['District'] == dist].sort_values('Year_Start').dropna(subset=['RAINFALL_MM_Total'])
    if len(d2) >= 5:
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(d2['Year_Start'], d2['RAINFALL_MM_Total'], 'o-', linewidth=2, markersize=6, color=DISTRICT_COLORS[dist])
        ax.set_xlabel('Crop Year', fontweight='bold')
        ax.set_ylabel('Seasonal Rainfall (mm)', fontweight='bold')
        ax.set_title(f'{dist}: Seasonal Rainfall Time Series', fontweight='bold')
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(paper2_fig, f'paper2_fig_{dist.lower()}_rainfall_timeseries.png'), dpi=300, bbox_inches='tight')
        plt.close()
    # P2 extra: Heat stress timeseries
    if 'Heat_Days_Tmax_ge_35C' in df_hq.columns:
        d3 = df_hq[df_hq['District'] == dist].sort_values('Year_Start').dropna(subset=['Heat_Days_Tmax_ge_35C'])
        if len(d3) >= 5:
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.plot(d3['Year_Start'], d3['Heat_Days_Tmax_ge_35C'], 'o-', linewidth=2, markersize=6, color=DISTRICT_COLORS[dist])
            ax.set_xlabel('Crop Year', fontweight='bold')
            ax.set_ylabel('Heat Stress Days (Tmax >= 35C)', fontweight='bold')
            ax.set_title(f'{dist}: Heat Stress Days Over Time', fontweight='bold')
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join(paper2_fig, f'paper2_fig_{dist.lower()}_heat_stress.png'), dpi=300, bbox_inches='tight')
            plt.close()
print("  P2-12 to 21: Nashik + Solapur + Bellary + Dharwad yield-rainfall, rainfall ts, heat stress")

# =============================================================================
# PAPER 3: ADD 9 FIGURES (3 comparative + 3 Nashik + 3 Solapur) -> 19 total
# =============================================================================
print("\n--- PAPER 3 EXTENDED ---")

# P3-13: Comparative area-yield scatter by district
fig, ax = plt.subplots(figsize=(12, 7))
for dist in DISTRICTS:
    d = df[df['District'] == dist]
    if len(d) > 2:
        ax.scatter(d['Area_hectares'], d['Yield_tonnes_per_hectare'], s=80, alpha=0.8,
                  label=dist, color=DISTRICT_COLORS[dist], edgecolors='black', linewidth=1)
ax.set_xlabel('Area (ha)', fontweight='bold')
ax.set_ylabel('Yield (t/ha)', fontweight='bold')
ax.set_title('Comparative: Area-Yield Relationship by District', fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(paper3_fig, 'paper3_fig13_comparative_area_yield.png'), dpi=300, bbox_inches='tight')
plt.close()
print("  P3-13: Comparative area-yield")

# P3-14: Growth contribution comparison (area vs yield % contribution by district)
contrib_list = []
for dist in DISTRICTS:
    d = df[df['District'] == dist].sort_values('Year_Start')
    if len(d) < 2:
        continue
    p0, p1 = d.iloc[0], d.iloc[-1]
    d_area = (p1['Area_hectares'] - p0['Area_hectares']) * (p0['Yield_tonnes_per_hectare'] + p1['Yield_tonnes_per_hectare'])/2
    d_yield = (p1['Yield_tonnes_per_hectare'] - p0['Yield_tonnes_per_hectare']) * (p0['Area_hectares'] + p1['Area_hectares'])/2
    tot = abs(d_area) + abs(d_yield)
    if tot > 0:
        contrib_list.append({'District': dist, 'Area_Contrib': d_area/tot*100, 'Yield_Contrib': d_yield/tot*100})
if contrib_list:
    dfc = pd.DataFrame(contrib_list)
    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(dfc))
    ax.bar(x - 0.2, dfc['Area_Contrib'], 0.4, label='Area', color='#4ECDC4', alpha=0.85)
    ax.bar(x + 0.2, dfc['Yield_Contrib'], 0.4, label='Yield', color='#F18F01', alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(dfc['District'])
    ax.set_ylabel('Contribution (%)', fontweight='bold')
    ax.set_title('Comparative: Area vs Yield Contribution to Production Growth', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(os.path.join(paper3_fig, 'paper3_fig14_comparative_growth_contribution.png'), dpi=300, bbox_inches='tight')
    plt.close()
print("  P3-14: Comparative growth contribution")

# P3-15: Scale distribution by district (area categories)
if 'Scale_Category' in df.columns:
    fig, ax = plt.subplots(figsize=(10, 6))
    scale_counts = df.groupby(['District', 'Scale_Category']).size().unstack(fill_value=0)
    cols = scale_counts.columns.tolist()
    colors = ['#FF6B6B', '#4ECDC4', '#95E1D3', '#F18F01'][:len(cols)]
    scale_counts.plot(kind='bar', ax=ax, color=colors[:len(cols)], alpha=0.85)
    ax.set_ylabel('Number of Years', fontweight='bold')
    ax.set_title('Comparative: Scale Category Distribution by District', fontweight='bold')
    ax.legend(title='Scale')
    ax.tick_params(axis='x', rotation=30)
    plt.tight_layout()
    plt.savefig(os.path.join(paper3_fig, 'paper3_fig15_comparative_scale_distribution.png'), dpi=300, bbox_inches='tight')
    plt.close()
print("  P3-15: Comparative scale distribution")

# P3-16 to 21: District-specific area-yield and performance profile (including Karnataka)
for dist in ['Nashik', 'Solapur', 'Bellary', 'Dharwad']:
    d = df[df['District'] == dist]
    if len(d) < 3:
        continue
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(d['Area_hectares'], d['Yield_tonnes_per_hectare'], s=100, alpha=0.8,
               color=DISTRICT_COLORS[dist], edgecolors='black', linewidth=1)
    z = np.polyfit(d['Area_hectares'], d['Yield_tonnes_per_hectare'], 1)
    xl = np.linspace(d['Area_hectares'].min(), d['Area_hectares'].max(), 50)
    ax.plot(xl, np.poly1d(z)(xl), '--', linewidth=2, color='#333', alpha=0.8)
    ax.set_xlabel('Area (ha)', fontweight='bold')
    ax.set_ylabel('Yield (t/ha)', fontweight='bold')
    ax.set_title(f'{dist}: Area-Yield Relationship', fontweight='bold')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(paper3_fig, f'paper3_fig_{dist.lower()}_area_yield.png'), dpi=300, bbox_inches='tight')
    plt.close()
    # Performance profile (mean yield, CV, trend)
    my = d['Yield_tonnes_per_hectare'].mean()
    cv = d['Yield_tonnes_per_hectare'].std() / my * 100 if my > 0 else 0
    slope, _, _, _, _ = stats.linregress(range(len(d)), d['Yield_tonnes_per_hectare'].values)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(['Mean Yield\n(t/ha)', 'CV (%)', 'Trend Slope'], [my, cv, slope], color=DISTRICT_COLORS[dist], alpha=0.85)
    ax.set_ylabel('Value', fontweight='bold')
    ax.set_title(f'{dist}: Performance Profile', fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(os.path.join(paper3_fig, f'paper3_fig_{dist.lower()}_performance_profile.png'), dpi=300, bbox_inches='tight')
    plt.close()
    # P3 extra: Growth decomposition (area vs yield contribution)
    d_s = d.sort_values('Year_Start')
    if len(d_s) >= 2:
        p0, p1 = d_s.iloc[0], d_s.iloc[-1]
        d_area = (p1['Area_hectares'] - p0['Area_hectares']) * (p0['Yield_tonnes_per_hectare'] + p1['Yield_tonnes_per_hectare'])/2
        d_yield = (p1['Yield_tonnes_per_hectare'] - p0['Yield_tonnes_per_hectare']) * (p0['Area_hectares'] + p1['Area_hectares'])/2
        tot = abs(d_area) + abs(d_yield)
        if tot > 0:
            fig, ax = plt.subplots(figsize=(8, 5))
            ax.bar(['Area', 'Yield'], [d_area/tot*100, d_yield/tot*100], color=[DISTRICT_COLORS[dist], '#F18F01'], alpha=0.85)
            ax.set_ylabel('Contribution (%)', fontweight='bold')
            ax.set_title(f'{dist}: Area vs Yield Contribution to Production Growth', fontweight='bold')
            ax.grid(True, alpha=0.3, axis='y')
            plt.tight_layout()
            plt.savefig(os.path.join(paper3_fig, f'paper3_fig_{dist.lower()}_growth_decomposition.png'), dpi=300, bbox_inches='tight')
            plt.close()
print("  P3-16 to 21: Nashik + Solapur area-yield, performance, growth decomposition")

print("\n" + "="*80)
print("EXTENDED FIGURES CREATED")
print("="*80)
