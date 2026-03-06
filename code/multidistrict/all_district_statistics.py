"""Compute statistics for all five districts; outputs CSVs for manuscript tables."""

import os
import pandas as pd
import numpy as np
from scipy import stats
from statsmodels.stats.anova import anova_lm
from statsmodels.formula.api import ols

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPO_ROOT = os.path.dirname(BASE_DIR)
PROD_PATH = os.path.join(REPO_ROOT, 'data', 'multidistrict_production_area_series.csv')
TABLES_DIR = os.path.join(REPO_ROOT, 'tables')
os.makedirs(TABLES_DIR, exist_ok=True)

# Common study window
YEAR_MIN, YEAR_MAX = 1999, 2020
DISTRICTS = ['Ahmednagar', 'Solapur', 'Nashik', 'Bellary', 'Dharwad']
VARIABLES = ['Area_hectares', 'Production_tonnes', 'Yield_tonnes_per_hectare']
VAR_NAMES = ['Area (ha)', 'Production (tonnes)', 'Yield (t/ha)']


def assign_period(yr):
    if pd.isna(yr):
        return None
    yr = int(yr)
    if 1999 <= yr <= 2006:
        return 'Period 1 (1999-2006)'
    if 2007 <= yr <= 2013:
        return 'Period 2 (2007-2013)'
    if 2014 <= yr <= 2020:
        return 'Period 3 (2014-2020)'
    return None


def mann_kendall_test(y):
    y = np.asarray(y)
    n = len(y)
    s = 0
    for i in range(n - 1):
        for j in range(i + 1, n):
            s += np.sign(y[j] - y[i])
    var_s = n * (n - 1) * (2 * n + 5) / 18.0
    if s > 0:
        z = (s - 1) / np.sqrt(var_s)
    elif s < 0:
        z = (s + 1) / np.sqrt(var_s)
    else:
        z = 0
    p = 2 * (1 - stats.norm.cdf(abs(z)))
    return z, p


def sens_slope(y):
    y = np.asarray(y)
    n = len(y)
    slopes = []
    for i in range(n):
        for j in range(i + 1, n):
            if j != i:
                slopes.append((y[j] - y[i]) / (j - i))
    return np.median(slopes) if slopes else 0.0


def main():
    print("=" * 60)
    print("ALL-DISTRICT STATISTICS FOR PAPER 1")
    print("=" * 60)

    df = pd.read_csv(PROD_PATH)
    df['Year_Start'] = pd.to_numeric(df['Year_Start'], errors='coerce')
    df = df.dropna(subset=['Year_Start', 'Area_hectares', 'Production_tonnes', 'Yield_tonnes_per_hectare'])
    df = df[(df['Year_Start'] >= YEAR_MIN) & (df['Year_Start'] <= YEAR_MAX)]
    df['Period'] = df['Year_Start'].apply(assign_period)
    df = df[df['Period'].notna()]

    # Time index per district (0, 1, 2, ...)
    df = df.sort_values(['District', 'Year_Start'])
    df['Time_Index'] = df.groupby('District').cumcount()

    # --- 1. DESCRIPTIVE STATISTICS (all districts) ---
    print("\n1. Descriptive statistics by district...")
    rows = []
    for dist in DISTRICTS:
        d = df[df['District'] == dist]
        if len(d) == 0:
            continue
        for var, name in zip(VARIABLES, VAR_NAMES):
            x = d[var].values
            rows.append({
                'District': dist,
                'Variable': name,
                'N': len(d),
                'Mean': x.mean(),
                'Median': np.median(x),
                'SD': x.std(),
                'Min': x.min(),
                'Max': x.max(),
                'CV_Pct': (x.std() / x.mean() * 100) if x.mean() != 0 else np.nan,
                'IQR': np.percentile(x, 75) - np.percentile(x, 25),
            })
    pd.DataFrame(rows).to_csv(os.path.join(TABLES_DIR, 'paper1_descriptive_all_districts.csv'), index=False)
    print(f"   Saved: paper1_descriptive_all_districts.csv")

    # --- 2. TREND ANALYSIS (all districts) ---
    print("\n2. Trend analysis by district...")
    rows = []
    for dist in DISTRICTS:
        d = df[df['District'] == dist].sort_values('Year_Start')
        if len(d) < 5:
            continue
        x = d['Time_Index'].values
        for var, name in zip(VARIABLES, VAR_NAMES):
            y = d[var].values
            slope, intercept, r, p_lin, _ = stats.linregress(x, y)
            z_mk, p_mk = mann_kendall_test(y)
            ss = sens_slope(y)
            rows.append({
                'District': dist,
                'Variable': name,
                'Linear_Slope': slope,
                'R2': r ** 2,
                'Linear_P': p_lin,
                'MK_Z': z_mk,
                'MK_P': p_mk,
                'Sens_Slope': ss,
            })
    pd.DataFrame(rows).to_csv(os.path.join(TABLES_DIR, 'paper1_trends_all_districts.csv'), index=False)
    print(f"   Saved: paper1_trends_all_districts.csv")

    # --- 3. PERIOD-WISE STATISTICS (all districts) ---
    print("\n3. Period-wise statistics by district...")
    rows = []
    for dist in DISTRICTS:
        d = df[df['District'] == dist]
        if len(d) == 0:
            continue
        for period in ['Period 1 (1999-2006)', 'Period 2 (2007-2013)', 'Period 3 (2014-2020)']:
            p = d[d['Period'] == period]
            if len(p) == 0:
                continue
            y = p['Yield_tonnes_per_hectare'].values
            rows.append({
                'District': dist,
                'Period': period,
                'N': len(p),
                'Area_Mean': p['Area_hectares'].mean(),
                'Production_Mean': p['Production_tonnes'].mean(),
                'Yield_Mean': y.mean(),
                'Yield_SD': y.std(),
                'Yield_CV_Pct': (y.std() / y.mean() * 100) if y.mean() != 0 else np.nan,
            })
    pd.DataFrame(rows).to_csv(os.path.join(TABLES_DIR, 'paper1_period_stats_all_districts.csv'), index=False)
    print(f"   Saved: paper1_period_stats_all_districts.csv")

    # --- 4. ANOVA / KRUSKAL-WALLIS (all districts) ---
    print("\n4. ANOVA and Kruskal-Wallis by district...")
    rows = []
    for dist in DISTRICTS:
        d = df[df['District'] == dist]
        if len(d) < 6 or d['Period'].nunique() < 2:
            continue
        for var, name in zip(VARIABLES, VAR_NAMES):
            try:
                model = ols(f'{var} ~ C(Period)', data=d).fit()
                atab = anova_lm(model, typ=2)
                f_val = atab.loc['C(Period)', 'F']
                p_anova = atab.loc['C(Period)', 'PR(>F)']
            except Exception:
                f_val = np.nan
                p_anova = np.nan
            grps = [d[d['Period'] == p][var].values for p in d['Period'].unique() if len(d[d['Period'] == p]) > 0]
            if len(grps) >= 2:
                try:
                    h_stat, p_kw = stats.kruskal(*grps)
                except Exception:
                    h_stat, p_kw = np.nan, np.nan
            else:
                h_stat, p_kw = np.nan, np.nan
            rows.append({
                'District': dist,
                'Variable': name,
                'ANOVA_F': f_val,
                'ANOVA_P': p_anova,
                'KW_H': h_stat,
                'KW_P': p_kw,
            })
    pd.DataFrame(rows).to_csv(os.path.join(TABLES_DIR, 'paper1_anova_all_districts.csv'), index=False)
    print(f"   Saved: paper1_anova_all_districts.csv")

    # --- 5. VARIABILITY (all districts) ---
    print("\n5. Variability by district...")
    rows = []
    for dist in DISTRICTS:
        d = df[df['District'] == dist].sort_values('Year_Start')
        if len(d) < 3:
            continue
        for var, name in zip(VARIABLES, VAR_NAMES):
            y = d[var].values
            cv = (np.std(y) / np.mean(y) * 100) if np.mean(y) != 0 else np.nan
            yoy = np.diff(y) / (y[:-1] + 1e-10) * 100
            vol = np.nanstd(yoy) if len(yoy) > 0 else np.nan
            mu = np.mean(y)
            within = np.sum(np.abs(y - mu) / (mu + 1e-10) <= 0.10)
            stability = (within / len(y)) * 100
            ac = np.corrcoef(y[:-1], y[1:])[0, 1] if len(y) > 1 else np.nan
            rows.append({
                'District': dist,
                'Variable': name,
                'CV_Pct': cv,
                'Volatility_Index': vol,
                'Stability_Pct': stability,
                'Autocorrelation': ac,
            })
    pd.DataFrame(rows).to_csv(os.path.join(TABLES_DIR, 'paper1_variability_all_districts.csv'), index=False)
    print(f"   Saved: paper1_variability_all_districts.csv")

    # --- 6. EXTREME YEARS (top 3 / bottom 3 yield per district) ---
    print("\n6. Extreme years by district...")
    rows = []
    for dist in DISTRICTS:
        d = df[df['District'] == dist].sort_values('Yield_tonnes_per_hectare', ascending=False)
        if len(d) == 0:
            continue
        top3 = d.head(3)
        for _, r in top3.iterrows():
            rows.append({
                'District': dist,
                'Category': 'Top 3',
                'Crop_Year': r['Crop_Year'],
                'Year_Start': int(r['Year_Start']),
                'Yield': r['Yield_tonnes_per_hectare'],
                'Area': r['Area_hectares'],
                'Production': r['Production_tonnes'],
            })
        bot3 = d.tail(3)
        for _, r in bot3.iterrows():
            rows.append({
                'District': dist,
                'Category': 'Bottom 3',
                'Crop_Year': r['Crop_Year'],
                'Year_Start': int(r['Year_Start']),
                'Yield': r['Yield_tonnes_per_hectare'],
                'Area': r['Area_hectares'],
                'Production': r['Production_tonnes'],
            })
    pd.DataFrame(rows).to_csv(os.path.join(TABLES_DIR, 'paper1_extreme_years_all_districts.csv'), index=False)
    print(f"   Saved: paper1_extreme_years_all_districts.csv")

    # --- 7. LaTeX snippet files for manuscript (optional) ---
    def fmt_num(x):
        if pd.isna(x) or abs(x) >= 1e6:
            return f"{x:,.0f}" if not pd.isna(x) else "---"
        if abs(x) >= 1:
            return f"{x:,.2f}"
        return f"{x:.3f}"

    # Descriptive: summary table (one row per district with key stats)
    with open(os.path.join(TABLES_DIR, 'paper1_latex_descriptive_all.txt'), 'w') as f:
        f.write("% Descriptive statistics by district (Yield) - use in Results\n")
        desc = pd.read_csv(os.path.join(TABLES_DIR, 'paper1_descriptive_all_districts.csv'))
        for dist in DISTRICTS:
            sub = desc[(desc['District'] == dist) & (desc['Variable'] == 'Yield (t/ha)')]
            if len(sub) == 0:
                continue
            r = sub.iloc[0]
            f.write(f"% {dist}: N={int(r['N'])}, Mean={r['Mean']:.2f}, SD={r['SD']:.2f}, CV={r['CV_Pct']:.2f}%\\n")

    print("\nDone. Tables written to tables/.")


if __name__ == '__main__':
    main()
