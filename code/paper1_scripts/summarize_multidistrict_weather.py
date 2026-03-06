import csv
import math
import os
import statistics
from collections import defaultdict


def main() -> None:
    base = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    path = os.path.join(base, "data", "multidistrict_yield_weather_matched.csv")

    stats: dict[str, dict[str, list[float]]] = {}

    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            comp_str = row.get("Core_Weather_Completeness_Pct") or row.get("Completeness_Pct")
            try:
                comp = float(comp_str) if comp_str is not None and comp_str != "" else math.nan
            except ValueError:
                comp = math.nan

            if math.isnan(comp) or comp < 80.0:
                continue

            dist = (row.get("District") or "").strip()
            if not dist:
                continue

            def f(field: str) -> float:
                v = row.get(field)
                if v is None or v == "":
                    return math.nan
                try:
                    return float(v)
                except ValueError:
                    return math.nan

            di = f("Drought_Index")
            mai = f("Moisture_Adequacy_Index")
            heat = f("Heat_Days_Tmax_ge_35C")
            yld = f("District_Mean_Yield_t_ha")

            d = stats.setdefault(dist, {"di": [], "mai": [], "heat": [], "yield": []})

            if not math.isnan(di):
                d["di"].append(di)
            if not math.isnan(mai):
                d["mai"].append(mai)
            if not math.isnan(heat):
                d["heat"].append(heat)
            if not math.isnan(yld):
                d["yield"].append(yld)

    print("District, HQ_years, mean_DI, mean_MAI, mean_heat_days, min_yield, max_yield")
    for dist in sorted(stats):
        d = stats[dist]
        if not d["yield"]:
            continue
        n = len(d["yield"])
        mean_di = statistics.mean(d["di"]) if d["di"] else math.nan
        mean_mai = statistics.mean(d["mai"]) if d["mai"] else math.nan
        mean_heat = statistics.mean(d["heat"]) if d["heat"] else math.nan
        min_y = min(d["yield"])
        max_y = max(d["yield"])
        def fmt(v: float, fmt_str: str) -> str:
            if math.isnan(v):
                return "nan"
            return format(v, fmt_str)

        print(
            f"{dist},{n},"
            f"{fmt(mean_di, '.2f')},"
            f"{fmt(mean_mai, '.2f')},"
            f"{fmt(mean_heat, '.0f')},"
            f"{min_y:.2f},{max_y:.2f}"
        )


if __name__ == "__main__":
    main()

