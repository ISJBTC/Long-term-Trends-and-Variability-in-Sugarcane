# Long-term Trends and Variability in Sugarcane Production

Five-district comparative analysis of sugarcane area, production, and yield (Ahmednagar, Solapur, Nashik, Bellary, Dharwad), Maharashtra and Karnataka, India, 1999–2000 to 2020–2021.

## Structure

```
├── code/           Analysis and figure scripts
├── data/           Input and processed data (CSV)
├── tables/         Output tables
├── figures/        Output figures
├── results/        Logs and summaries
└── docs/           Documentation
```

## Requirements

Python 3.x with pandas, numpy, scipy, statsmodels, matplotlib, seaborn.

## Usage

From the repository root:

```bash
python code/multidistrict/run_multidistrict_pipeline.py
```

Runs data preparation and writes figures to `figures/`, tables to `tables/`.

Core analysis (single-district focus):

```bash
python code/paper1_code/paper1_analysis_complete.py
```

Expects prepared data in `data/` (run `code/paper1_code/paper1_data_preparation.py` first if regenerating from source).

## Data

- Production: Directorate of Economics and Statistics (DES), Maharashtra and Karnataka
- Weather: India Meteorological Department (IMD), Climate Research and Services, Pune

## Citation

Cite the associated manuscript if you use this repository.
