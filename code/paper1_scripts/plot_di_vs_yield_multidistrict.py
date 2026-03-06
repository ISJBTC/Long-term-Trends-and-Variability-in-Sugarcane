import csv
import math
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt


def main() -> None:
    base = Path(__file__).resolve().parents[2]
    data_path = base / "data" / "multidistrict_yield_weather_matched.csv"
    out_path = base / "figures" / "di_vs_yield_multidistrict.png"

    districts = ["Ahmednagar", "Nashik", "Solapur"]
    colours = {
        "Ahmednagar": "#1f77b4",
        "Nashik": "#2ca02c",
        "Solapur": "#d62728",
    }

    xs: dict[str, list[float]] = defaultdict(list)
    ys: dict[str, list[float]] = defaultdict(list)

    with data_path.open(newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            dist = (row.get("District") or "").strip()
            if dist not in districts:
                continue

            comp_str = row.get("Core_Weather_Completeness_Pct") or row.get("Completeness_Pct")
            try:
                comp = float(comp_str) if comp_str else math.nan
            except ValueError:
                comp = math.nan
            if math.isnan(comp) or comp < 80.0:
                continue

            try:
                di = float(row.get("Drought_Index") or "")
                yld = float(row.get("District_Mean_Yield_t_ha") or "")
            except ValueError:
                continue

            if math.isnan(di) or math.isnan(yld):
                continue

            xs[dist].append(di)
            ys[dist].append(yld)

    plt.figure(figsize=(5.5, 4.0), dpi=300)

    for dist in districts:
        if not xs[dist]:
            continue
        plt.scatter(xs[dist], ys[dist], label=dist, s=35, alpha=0.8, edgecolor="none", color=colours.get(dist))

    plt.axvline(0.0, color="grey", linewidth=0.5)
    plt.xlabel("Drought index (DI)")
    plt.ylabel("District mean yield (t/ha)")
    plt.title("Drought index vs. yield for high-quality years\nAhmednagar, Nashik, and Solapur")
    plt.legend(frameon=False)
    plt.tight_layout()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, bbox_inches="tight")


if __name__ == "__main__":
    main()

