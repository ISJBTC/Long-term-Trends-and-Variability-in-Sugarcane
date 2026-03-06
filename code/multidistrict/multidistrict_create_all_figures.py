"""Generate multi-district comparative figures for Paper 1-3."""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
import seaborn as sns
from scipy import stats
from sklearn.linear_model import LinearRegression
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
    'xtick.labelsize': 12, 'ytick.labelsize': 12,
    'legend.fontsize': 11, 'figure.dpi': 300, 'savefig.dpi': 300,
    'savefig.bbox': 'tight', 'savefig.pad_inches': 0.2,
    'lines.linewidth': 2, 'axes.linewidth': 1.5,
})

DISTRICT_COLORS = {
    'Ahmednagar': '#2E86AB',
    'Solapur': '#A23B72',
    'Nashik': '#F18F01',
    'Bellary': '#4CAF50',
    'Dharwad': '#9B59B6',
}
DISTRICTS = ['Ahmednagar', 'Solapur', 'Nashik', 'Bellary', 'Dharwad']


def radar_chart(ax, categories, values_list, labels, colors=None):
    n = len(categories)
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False).tolist()
    angles += angles[:1]

    if colors is None:
        colors = plt.cm.Set1(np.linspace(0, 1, len(values_list)))

    for vals, label, color in zip(values_list, labels, colors):
        v = list(vals) + [vals[0]]
        ax.plot(angles, v, 'o-', linewidth=2, label=label, color=color)
        ax.fill(angles, v, alpha=0.15, color=color)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, size=10)
    ax.set_ylim(0, 1.05)

    if labels:
        ax.legend(
            loc='upper left',
            bbox_to_anchor=(1.05, 1.05),
            borderaxespad=0.0,
            frameon=True
        )


print("="*80)
print("MULTIDISTRICT: CREATING ALL 12 FIGURES")
print("="*80)

# Load data
df = pd.read_csv(os.path.join(data_dir, 'multidistrict_panel.csv'))
df['Year_Start'] = pd.to_numeric(df['Year_Start'], errors='coerce')
df = df.dropna(subset=['Year_Start'])
df_hq = df[df['Weather_High_Quality'] == True].copy()

print(f"Panel: {len(df)} rows, high-quality weather: {len(df_hq)}")

# =============================================================================
# PAPER 1: 4 FIGURES
# =============================================================================
print("\n--- PAPER 1 ---")

# Fig 1: Long-term trends (Area, Production, Yield) - all districts
fig, axes = plt.subplots(3, 1, figsize=(14, 11))
fig.suptitle('Long-term Trends in Sugarcane Area, Production and Yield (All Districts)', fontsize=16, fontweight='bold', y=0.995)

for idx, (var, name) in enumerate([
    ('Area_hectares', 'Area (ha)'),
    ('Production_tonnes', 'Production (tonnes)'),
    ('Yield_tonnes_per_hectare', 'Yield (t/ha)')
]):
    ax = axes[idx]
    for dist in DISTRICTS:
        d = df[df['District'] == dist].sort_values('Year_Start')
        if len(d) > 0:
            ax.plot(d['Year_Start'], d[var], 'o-', linewidth=2, markersize=6,
                    label=dist, color=DISTRICT_COLORS[dist])
            ma_col = f'{var.replace("_hectares","").replace("_tonnes","").replace("_tonnes_per_hectare","")}_MA5'
            if ma_col in d.columns:
                ax.plot(d['Year_Start'], d[ma_col], '--', linewidth=1.5, alpha=0.7, color=DISTRICT_COLORS[dist])
    ax.set_ylabel(name, fontweight='bold')
    ax.set_title(name, fontweight='bold')
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_xlim(df['Year_Start'].min()-1, df['Year_Start'].max()+1)

axes[-1].set_xlabel('Year', fontweight='bold')
plt.tight_layout(rect=[0, 0, 1, 0.98])
plt.savefig(os.path.join(paper1_fig, 'paper1_fig1_multidistrict_trends.png'), dpi=300, bbox_inches='tight')
plt.close()
print("  Fig 1: Long-term trends")

periods = ['Period 1 (1999-2006)', 'Period 2 (2007-2013)', 'Period 3 (2014-2020)']
def assign_period(yr):
    if pd.isna(yr): return None
    yr = int(yr)
    if 1999 <= yr <= 2006: return 'Period 1 (1999-2006)'
    if 2007 <= yr <= 2013: return 'Period 2 (2007-2013)'
    if 2014 <= yr <= 2020: return 'Period 3 (2014-2020)'
    return None

df_fig = df.copy()
df_fig['Period_Fig'] = df_fig['Year_Start'].apply(assign_period)
contrib_data = []
for dist in DISTRICTS:
    d = df_fig[df_fig['District'] == dist].sort_values('Year_Start')
    if len(d) < 2:
        continue
    for p in periods:
        sub = d[d['Period_Fig'] == p]
        if len(sub) < 2:
            continue
        p0, p1 = sub.iloc[0], sub.iloc[-1]
        d_area = (p1['Area_hectares'] - p0['Area_hectares']) * (p0['Yield_tonnes_per_hectare'] + p1['Yield_tonnes_per_hectare']) / 2.0
        d_yield = (p1['Yield_tonnes_per_hectare'] - p0['Yield_tonnes_per_hectare']) * (p0['Area_hectares'] + p1['Area_hectares']) / 2.0
        tot = abs(d_area) + abs(d_yield)
        if tot == 0:
            continue
        contrib_area = (d_area / tot) * 100.0
        contrib_yield = (d_yield / tot) * 100.0
        contrib_data.append(
            {
                'District': dist,
                'Period': p,
                'Area_Contrib': contrib_area,
                'Yield_Contrib': contrib_yield,
            }
        )

df_contrib = pd.DataFrame(contrib_data)
if not df_contrib.empty:
    n_dist = len(DISTRICTS)
    n_rows = int(np.ceil(n_dist / 2))
    n_cols = 2 if n_dist > 1 else 1
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 3 * n_rows + 1), sharey=True)
    axes = np.array(axes).reshape(-1)

    x = np.arange(len(periods))
    width = 0.35
    for idx, dist in enumerate(DISTRICTS):
        ax = axes[idx]
        sub = df_contrib[df_contrib['District'] == dist].set_index('Period')
        if sub.empty:
            ax.axis('off')
            continue
        area_vals = sub.reindex(periods)['Area_Contrib'].values
        yield_vals = sub.reindex(periods)['Yield_Contrib'].values
        area_vals = np.where(np.isnan(area_vals), 0, area_vals)
        yield_vals = np.where(np.isnan(yield_vals), 0, yield_vals)
        bars_a = ax.bar(x - width / 2, area_vals, width, label='Area', color='#3498DB', alpha=0.8)
        bars_y = ax.bar(x + width / 2, yield_vals, width, label='Yield', color='#E67E22', alpha=0.8)
        for i, (va, vy) in enumerate(zip(area_vals, yield_vals)):
            if abs(va) >= 0.5:
                ypos = va + (2 if va >= 0 else -2)
                ax.text(x[i] - width/2, ypos, f'{va:.1f}%', fontsize=8, ha='center', va='bottom' if va >= 0 else 'top')
            if abs(vy) >= 0.5:
                ypos = vy + (2 if vy >= 0 else -2)
                ax.text(x[i] + width/2, ypos, f'{vy:.1f}%', fontsize=8, ha='center', va='bottom' if vy >= 0 else 'top')
        ax.axhline(0, color='k', linewidth=1)
        ax.set_xticks(x)
        ax.set_xticklabels([p.split('(')[1].rstrip(')') for p in periods], rotation=0)
        ax.set_title(dist, fontweight='bold', loc='left')
        ax.grid(True, alpha=0.3, axis='y')
        if idx % n_cols == 0:
            ax.set_ylabel('Contribution (%)', fontweight='bold')

    # Hide any unused subplots
    for j in range(len(DISTRICTS), len(axes)):
        axes[j].axis('off')

    handles = [
        plt.Rectangle((0, 0), 1, 1, color='#3498DB', label='Area contribution'),
        plt.Rectangle((0, 0), 1, 1, color='#E67E22', label='Yield contribution'),
    ]
    fig.legend(handles=handles, loc='upper right', fontsize=9)
    fig.suptitle('Period-wise Decomposition: Area vs Yield Contribution to Production Change', fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(os.path.join(paper1_fig, 'paper1_fig2_multidistrict_decomposition.png'), dpi=300, bbox_inches='tight')
    plt.close()
print("  Fig 2: Period decomposition (panelled)")

# Fig 3: District-wise yield over time (separate panels per district for clarity)
n_dist = len(DISTRICTS)
fig, axes = plt.subplots(n_dist, 1, figsize=(12, 2.5 * n_dist + 2), sharex=True)
if n_dist == 1:
    axes = [axes]
for idx, dist in enumerate(DISTRICTS):
    ax = axes[idx]
    d = df[df['District'] == dist].sort_values('Year_Start')
    if len(d) == 0:
        ax.axis('off')
        continue
    ax.plot(d['Year_Start'], d['Yield_tonnes_per_hectare'], 'o-', linewidth=2, markersize=5,
            color=DISTRICT_COLORS.get(dist, '#333333'))
    if 'Yield_MA5' in d.columns:
        ax.plot(d['Year_Start'], d['Yield_MA5'], '--', linewidth=1.5, alpha=0.7,
                color=DISTRICT_COLORS.get(dist, '#333333'))
    ax.set_ylabel('Yield (t/ha)', fontweight='bold')
    ax.set_title(dist, fontweight='bold', loc='left')
    ax.grid(True, alpha=0.3)
axes[-1].set_xlabel('Year', fontweight='bold')
fig.suptitle('District-wise Yield Over Time', fontsize=16, fontweight='bold', y=0.995)
plt.tight_layout(rect=[0, 0, 1, 0.97])
plt.savefig(os.path.join(paper1_fig, 'paper1_fig3_multidistrict_yield_variability.png'), dpi=300, bbox_inches='tight')
plt.close()
print("  Fig 3: Yield trajectories by district")

# Fig 4: Radar - structural profile (Mean yield, Mean area, CV yield, CV area, trend slope yield, trend slope area)
radar_cats = ['Mean Yield', 'Mean Area', 'CV Yield', 'CV Area', 'Trend Yield', 'Trend Area']
radar_vals = []
radar_labels = []
for dist in DISTRICTS:
    d = df[df['District'] == dist]
    if len(d) < 3:
        continue
    my = d['Yield_tonnes_per_hectare'].mean()
    ma = d['Area_hectares'].mean()
    cv_y = (d['Yield_tonnes_per_hectare'].std() / my * 100) if my > 0 else 0
    cv_a = (d['Area_hectares'].std() / ma * 100) if ma > 0 else 0
    slope_y, _, _, _, _ = stats.linregress(range(len(d)), d['Yield_tonnes_per_hectare'].values)
    slope_a, _, _, _, _ = stats.linregress(range(len(d)), d['Area_hectares'].values)
    radar_labels.append(dist)
    v1 = min(1, my / 110) if my > 0 else 0
    v2 = min(1, ma / 250000) if ma > 0 else 0
    v3 = min(1, max(0, 1 - cv_y/50))
    v4 = min(1, max(0, 1 - cv_a/80))
    v5 = min(1, max(0, (slope_y + 1) / 2))
    v6 = min(1, max(0, (slope_a + 2000) / 5000))
    radar_vals.append([v1, v2, v3, v4, v5, v6])

if radar_vals and radar_labels:
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(projection='polar'))
    colors = [DISTRICT_COLORS.get(l, '#333') for l in radar_labels]
    radar_chart(ax, radar_cats, radar_vals, radar_labels, colors)
    ax.set_title('Structural Profile: Production and Variability by District', fontweight='bold', pad=20)
    plt.tight_layout()
    plt.savefig(os.path.join(paper1_fig, 'paper1_fig4_multidistrict_radar_structural.png'), dpi=300, bbox_inches='tight')
    plt.close()
print("  Fig 4: Radar structural")

# =============================================================================
# PAPER 2: 4 FIGURES
# =============================================================================
print("\n--- PAPER 2 ---")

# Fig 5: Seasonal rainfall and soil moisture by district (bar chart of seasonal totals)
fig, ax = plt.subplots(figsize=(12, 6))
x_offset = 0
for dist in DISTRICTS:
    d = df[df['District'] == dist]
    if len(d) == 0:
        continue
    d = d.sort_values('Year_Start')
    if 'RAINFALL_MM_Total' in d.columns and d['RAINFALL_MM_Total'].notna().any():
        d = d.dropna(subset=['RAINFALL_MM_Total'])
        if len(d) > 0:
            ax.plot(d['Year_Start'], d['RAINFALL_MM_Total'], 'o-', linewidth=2, markersize=6,
                    label=dist, color=DISTRICT_COLORS[dist])
ax.set_xlabel('Crop Year Start', fontweight='bold')
ax.set_ylabel('Seasonal Rainfall (mm)', fontweight='bold')
ax.set_title('Seasonal Rainfall by District and Crop Year', fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3, axis='y')
ax.set_xlim(df['Year_Start'].min()-1, df['Year_Start'].max()+1)
plt.tight_layout()
plt.savefig(os.path.join(paper2_fig, 'paper2_fig5_multidistrict_rainfall.png'), dpi=300, bbox_inches='tight')
plt.close()
print("  Fig 5: Seasonal rainfall")

# Fig 6: Yield vs key weather indices (by district)
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
# Left: Yield vs Rainfall
ax = axes[0]
for dist in DISTRICTS:
    d = df_hq[df_hq['District'] == dist]
    if len(d) > 2 and d['RAINFALL_MM_Total'].notna().any():
        ax.scatter(d['RAINFALL_MM_Total'], d['Yield_tonnes_per_hectare'], s=80, alpha=0.8, 
                  label=dist, color=DISTRICT_COLORS[dist], edgecolors='black', linewidth=1)
        z = np.polyfit(d['RAINFALL_MM_Total'].dropna(), d.loc[d['RAINFALL_MM_Total'].notna(), 'Yield_tonnes_per_hectare'], 1)
        xl = np.linspace(d['RAINFALL_MM_Total'].min(), d['RAINFALL_MM_Total'].max(), 50)
        ax.plot(xl, np.poly1d(z)(xl), '--', color=DISTRICT_COLORS[dist], linewidth=2, alpha=0.8)
ax.set_xlabel('Seasonal Rainfall (mm)', fontweight='bold')
ax.set_ylabel('Yield (t/ha)', fontweight='bold')
ax.set_title('Yield vs Seasonal Rainfall', fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3)

# Right: Yield vs Heat stress days
ax = axes[1]
heat_col = 'Heat_Days_Tmax_ge_35C' if 'Heat_Days_Tmax_ge_35C' in df_hq.columns else 'Heat_Days_Tmax_ge_38C'
if heat_col in df_hq.columns:
    for dist in DISTRICTS:
        d = df_hq[df_hq['District'] == dist]
        d = d[d[heat_col].notna()]
        if len(d) > 2:
            ax.scatter(d[heat_col], d['Yield_tonnes_per_hectare'], s=80, alpha=0.8, 
                      label=dist, color=DISTRICT_COLORS[dist], edgecolors='black', linewidth=1)
    ax.set_xlabel('Heat Stress Days (Tmax >= 35C)', fontweight='bold')
    ax.set_ylabel('Yield (t/ha)', fontweight='bold')
    ax.set_title('Yield vs Heat Stress Days', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(paper2_fig, 'paper2_fig6_multidistrict_yield_weather.png'), dpi=300, bbox_inches='tight')
plt.close()
print("  Fig 6: Yield vs weather")

# Fig 7: Phase-wise weather anomalies (high vs low yield years)
# Approximate: top vs bottom quartile yield years
df_hq_valid = df_hq[df_hq['Yield_tonnes_per_hectare'].notna()].copy()
if len(df_hq_valid) >= 8:
    q75 = df_hq_valid['Yield_tonnes_per_hectare'].quantile(0.75)
    q25 = df_hq_valid['Yield_tonnes_per_hectare'].quantile(0.25)
    high = df_hq_valid[df_hq_valid['Yield_tonnes_per_hectare'] >= q75]
    low = df_hq_valid[df_hq_valid['Yield_tonnes_per_hectare'] <= q25]
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    vars_plot = [('RAINFALL_MM_Total', 'Rainfall (mm)'), ('Heat_Days_Tmax_ge_35C', 'Heat Days')]
    for ax, (v, lbl) in zip(axes, vars_plot):
        if v not in df_hq_valid.columns:
            continue
        data_high = high[v].dropna()
        data_low = low[v].dropna()
        if len(data_high) > 0 and len(data_low) > 0:
            bp = ax.boxplot([data_low, data_high], labels=['Low Yield Years', 'High Yield Years'], patch_artist=True)
            bp['boxes'][0].set_facecolor('#FF6B6B')
            bp['boxes'][1].set_facecolor('#4ECDC4')
            ax.set_ylabel(lbl, fontweight='bold')
            ax.set_title(lbl, fontweight='bold')
            ax.grid(True, alpha=0.3, axis='y')
    plt.suptitle('Weather Indices: High vs Low Yield Years (All Districts)', fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(paper2_fig, 'paper2_fig7_multidistrict_phase_anomalies.png'), dpi=300, bbox_inches='tight')
    plt.close()
print("  Fig 7: Phase anomalies")

# Fig 8: Radar - weather risk profile
wvars = ['RAINFALL_MM_Total', 'Heat_Days_Tmax_ge_35C', 'Dry_Days_Rain_lt_1mm', 'MEAN_TEMP_C_Mean']
wvars = [v for v in wvars if v in df_hq.columns]
if len(wvars) >= 3:
    radar_cats2 = [v.replace('_', ' ')[:15] for v in wvars]
    radar_vals2 = []
    for dist in DISTRICTS:
        d = df_hq[df_hq['District'] == dist]
        if len(d) < 2:
            continue
        vals = []
        for v in wvars:
            x = d[v].mean()
            if 'Rainfall' in v or 'RAINFALL' in v:
                vals.append(min(1, x / 1000))
            elif 'Heat' in v:
                vals.append(min(1, x / 150))
            elif 'Dry' in v:
                vals.append(min(1, x / 350))
            else:
                vals.append(min(1, (x - 20) / 15))
        radar_vals2.append(vals)
        radar_labels2 = [dist for dist in DISTRICTS if len(df_hq[df_hq['District'] == dist]) >= 2]
    
    if radar_vals2:
        fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(projection='polar'))
        colors2 = [DISTRICT_COLORS.get(l, '#333') for l in radar_labels2[:len(radar_vals2)]]
        radar_chart(ax, radar_cats2[:len(radar_vals2[0])], radar_vals2, radar_labels2[:len(radar_vals2)], colors2)
        ax.set_title('Weather-Climate Risk Profile by District', fontweight='bold', pad=20)
        plt.tight_layout()
        plt.savefig(os.path.join(paper2_fig, 'paper2_fig8_multidistrict_radar_weather.png'), dpi=300, bbox_inches='tight')
        plt.close()
print("  Fig 8: Radar weather")

# =============================================================================
# PAPER 3: 4 FIGURES
# =============================================================================
print("\n--- PAPER 3 ---")

# Fig 9: Production growth decomposition over time (each district)
n_dist = len(DISTRICTS)
fig, axes = plt.subplots(n_dist, 1, figsize=(12, 3 + 3 * n_dist))
if n_dist == 1:
    axes = [axes]
for idx, dist in enumerate(DISTRICTS):
    ax = axes[idx]
    d = df[df['District'] == dist].sort_values('Year_Start')
    if len(d) < 2:
        continue
    base = d['Production_tonnes'].iloc[0]
    if base <= 0:
        continue
    ax.plot(d['Year_Start'], d['Production_tonnes'] / base * 100, 'o-', linewidth=2, 
            label='Production Index', color=DISTRICT_COLORS[dist])
    ax.plot(d['Year_Start'], d['Area_hectares'] / d['Area_hectares'].iloc[0] * 100, '--', linewidth=1.5, 
            label='Area Index', color=DISTRICT_COLORS[dist], alpha=0.7)
    ax.plot(d['Year_Start'], d['Yield_tonnes_per_hectare'] / d['Yield_tonnes_per_hectare'].iloc[0] * 100, ':', linewidth=1.5, 
            label='Yield Index', color=DISTRICT_COLORS[dist], alpha=0.7)
    ax.set_ylabel('Index (base=100)', fontweight='bold')
    ax.set_title(f'{dist}: Production, Area and Yield Indices', fontweight='bold')
    ax.legend(loc='best', fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(df['Year_Start'].min()-1, df['Year_Start'].max()+1)
axes[-1].set_xlabel('Year', fontweight='bold')
plt.suptitle('Production Growth Decomposition by District', fontweight='bold', y=1.02)
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig(os.path.join(paper3_fig, 'paper3_fig9_multidistrict_growth_decomposition.png'), dpi=300, bbox_inches='tight')
plt.close()
print("  Fig 9: Growth decomposition")

# Fig 10: Stability vs level (mean yield vs CV of yield)
fig, ax = plt.subplots(figsize=(10, 7))
for dist in DISTRICTS:
    d = df[df['District'] == dist]
    if len(d) >= 3:
        my = d['Yield_tonnes_per_hectare'].mean()
        cv = d['Yield_tonnes_per_hectare'].std() / my * 100 if my > 0 else 0
        ax.scatter(my, cv, s=150, label=dist, color=DISTRICT_COLORS[dist], edgecolors='black', linewidth=2, zorder=5)
        ax.annotate(dist, (my, cv), xytext=(5, 5), textcoords='offset points', fontsize=11, fontweight='bold')
ax.set_xlabel('Mean Yield (t/ha)', fontweight='bold')
ax.set_ylabel('Coefficient of Variation of Yield (%)', fontweight='bold')
ax.set_title('Stability vs Level: Yield Variability by District', fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(paper3_fig, 'paper3_fig10_multidistrict_stability_level.png'), dpi=300, bbox_inches='tight')
plt.close()
print("  Fig 10: Stability vs level")

# Fig 11: Weather-normalised yield residuals (from simple regression)
resid_data = []
df_resid = pd.DataFrame()
for dist in DISTRICTS:
    d = df_hq[df_hq['District'] == dist]
    cols = ['Yield_tonnes_per_hectare', 'RAINFALL_MM_Total', 'Heat_Days_Tmax_ge_35C']
    if not all(c in d.columns for c in cols):
        continue
    d = d[cols].dropna()
    if len(d) < 5:
        continue
    X = d[['RAINFALL_MM_Total', 'Heat_Days_Tmax_ge_35C']].values
    y = d['Yield_tonnes_per_hectare'].values
    model = LinearRegression().fit(X, y)
    pred = model.predict(X)
    resid = y - pred
    for r in resid:
        resid_data.append({'District': dist, 'Residual': r})

if resid_data:
    df_resid = pd.DataFrame(resid_data)
    df_resid = pd.DataFrame(resid_data)
    fig, ax = plt.subplots(figsize=(10, 6))
    bp = ax.boxplot([df_resid[df_resid['District']==d]['Residual'].values for d in DISTRICTS if len(df_resid[df_resid['District']==d])>0],
                    labels=[d for d in DISTRICTS if len(df_resid[df_resid['District']==d])>0],
                    patch_artist=True)
    for i, d in enumerate([x for x in DISTRICTS if len(df_resid[df_resid['District']==x])>0]):
        bp['boxes'][i].set_facecolor(DISTRICT_COLORS[d])
        bp['boxes'][i].set_alpha(0.7)
    ax.axhline(0, color='k', linestyle='--', linewidth=1)
    ax.set_ylabel('Weather-Normalised Yield Residual (t/ha)', fontweight='bold')
    ax.set_title('Yield Residuals After Regressing on Rainfall and Heat Stress', fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(os.path.join(paper3_fig, 'paper3_fig11_multidistrict_residuals.png'), dpi=300, bbox_inches='tight')
    plt.close()
print("  Fig 11: Weather-normalised residuals")

# Fig 12: Radar - performance and resilience
perf_cats = ['Mean Yield', 'Trend', 'Stability', 'Residual', 'Resilience']
perf_vals = []
perf_labels = []
for dist in DISTRICTS:
    d = df[df['District'] == dist]
    if len(d) < 3:
        continue
    my = d['Yield_tonnes_per_hectare'].mean()
    cv = d['Yield_tonnes_per_hectare'].std() / my * 100 if my > 0 else 50
    slope, _, _, _, _ = stats.linregress(range(len(d)), d['Yield_tonnes_per_hectare'].values)
    sub_resid = df_resid[df_resid['District']==dist]['Residual'] if len(df_resid) > 0 else pd.Series([0])
    mean_resid = sub_resid.mean() if len(sub_resid) > 0 else 0
    perf_labels.append(dist)
    v1 = min(1, my / 110)
    v2 = min(1, max(0, (slope + 1) / 2))
    v3 = min(1, max(0, 1 - cv/50))
    v4 = min(1, max(0, (mean_resid + 20) / 40))
    v5 = (v1 + v3) / 2
    perf_vals.append([v1, v2, v3, v4, v5])

if perf_vals:
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(projection='polar'))
    colors3 = [DISTRICT_COLORS.get(l, '#333') for l in perf_labels]
    radar_chart(ax, perf_cats, perf_vals, perf_labels, colors3)
    ax.set_title('Performance and Resilience Profile by District', fontweight='bold', pad=20)
    plt.tight_layout()
    plt.savefig(os.path.join(paper3_fig, 'paper3_fig12_multidistrict_radar_performance.png'), dpi=300, bbox_inches='tight')
    plt.close()
print("  Fig 12: Radar performance")

print("\n" + "="*80)
print("="*80)
print(f"Figures saved: {fig_dir}")
