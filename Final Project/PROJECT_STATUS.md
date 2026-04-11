# Project Status

## Last Updated

2026-04-11

## 1. What Was Added or Changed

- Added `scripts/build_project_baseline.py` to parse the raw `Data.csv` file, detect the real monthly-return header, clean the portfolio columns, and save reproducible artifacts.
- Created processed datasets in `data/processed/`:
  - `monthly_portfolios_wide.csv`
  - `monthly_portfolios_tidy.csv`
- Generated baseline outputs in `outputs/`:
  - `tables/dataset_overview.csv`
  - `tables/summary_statistics.csv`
  - `tables/missingness_check.csv`
  - `tables/correlation_matrix.csv`
  - `figures/monthly_returns.png`
  - `figures/cumulative_growth.png`
- Generated a first report draft for the data section at `report/02_data_source_and_processing.md`.

## 2. What Ran Successfully

- Verified the required interpreter can import the packages used in this step:
  - `pandas 2.2.2`
  - `numpy 1.26.4`
  - `matplotlib 3.9.2`
- Ran:
  - `& 'd:\MG\anaconda3\python.exe' 'scripts\build_project_baseline.py'`
- Confirmed the cleaned monthly sample spans `1926-07-31` to `2026-01-31` with `1195` monthly observations for each of the `6` portfolios.
- Confirmed the extracted monthly block contains `0` missing values after converting the documented missing-value sentinels.

## 3. What Remains

- Perform univariate distribution diagnostics for each portfolio return series.
- Build and compare univariate time-series models for each portfolio, including a lightweight GARCH-style implementation if needed.
- Design exogenous-variable extensions for predictive modeling.
- Build multivariate analysis for cointegration and statistical arbitrage.
- Implement mean-variance optimization, efficient frontier plots, and backtests against an equally weighted benchmark.
- Expand the report beyond the data-processing section and integrate tables/figures into a final write-up.

## 4. Blockers or Assumptions

- Assumed the correct starting point is the monthly value-weighted return section in `Data.csv`, since that matches the assignment brief.
- No external data, literature sources, or downloaded factors have been added yet.
- If later steps require internet-based inputs and access is unavailable, those steps will need a documented offline fallback.
