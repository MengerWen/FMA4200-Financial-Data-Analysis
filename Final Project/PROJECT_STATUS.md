# Project Status

## Last Updated

2026-04-11

## 1. What Was Added or Changed

- Added the requested reproducible project structure:
  - `src/fma4200_project/`
  - `scripts/`
  - `data/processed/`
  - `output/figures/`
  - `output/tables/`
  - `output/models/`
  - `report/sections/`
  - `logs/`
- Added reusable source files:
  - `src/fma4200_project/config.py`
  - `src/fma4200_project/environment.py`
  - `src/fma4200_project/data_pipeline.py`
- Added runnable scripts:
  - `scripts/check_environment.py`
  - `scripts/clean_data.py`
  - `scripts/run_pipeline.py`
  - `scripts/build_project_baseline.py` as a backward-compatible wrapper around the cleaning step
- Added project-facing documentation:
  - `README.md`
  - `environment_used.md`
  - `report/sections/02_data_source_and_processing.md`
- Added cleaned data artifacts:
  - `data/processed/monthly_portfolio_returns_clean.csv`
  - `data/processed/monthly_portfolio_returns_decimal.csv`
  - `data/processed/data_dictionary.csv`
- Added saved sanity-check outputs:
  - `output/tables/date_coverage_check.csv`
  - `output/tables/duplicate_check.csv`
  - `output/tables/missingness_check.csv`
  - `output/tables/descriptive_statistics_pct.csv`
  - `output/tables/descriptive_statistics_decimal.csv`
  - `output/tables/sanity_checks_summary.csv`
- Added baseline figures and logs:
  - `output/figures/monthly_returns_overview.png`
  - `output/figures/cumulative_growth_of_1.png`
  - `logs/environment_check.txt`
  - `logs/pipeline_run.log`

## 2. What Ran Successfully

- Verified the required interpreter can import the packages actually used in this step:
  - `numpy 1.26.4`
  - `pandas 2.2.2`
  - `matplotlib 3.9.2`
- Ran successfully:
  - `& 'd:\MG\anaconda3\python.exe' 'scripts\check_environment.py'`
  - `& 'd:\MG\anaconda3\python.exe' 'scripts\clean_data.py'`
  - `& 'd:\MG\anaconda3\python.exe' 'scripts\run_pipeline.py'`
- Confirmed the cleaned monthly sample spans `1926-07-31` to `2026-01-31`.
- Confirmed the cleaned dataset contains `1195` monthly observations and `6` value-weighted portfolio return columns.
- Confirmed duplicate-date rows: `0`.
- Confirmed missing values after sentinel conversion: `0`.
- Saved date coverage, duplicate check, missingness check, and descriptive statistics tables under `output/tables/`.
- Saved a data dictionary documenting the percent and decimal return conventions.

## 3. What Remains

- Perform univariate distribution diagnostics for each portfolio return series.
- Build and compare univariate time-series models for each portfolio, including a lightweight GARCH-style implementation if needed.
- Design exogenous-variable extensions for predictive modeling.
- Build multivariate analysis for cointegration and statistical arbitrage.
- Implement mean-variance optimization, efficient frontier plots, and backtests against an equally weighted benchmark.
- Expand the report beyond the data-processing section and integrate tables/figures into a final write-up.

## 4. Blockers or Assumptions

- Assumed the correct starting point is the monthly value-weighted return section in `Data.csv`, since that matches the assignment brief.
- Assumed the canonical cleaned dataset should keep the six return columns in **percent** units and save a separate decimal companion dataset for analysis that requires decimal returns.
- No external data, literature sources, or downloaded factors have been added yet.
- If later steps require internet-based inputs and access is unavailable, those steps will need a documented offline fallback.
- An earlier `outputs/` folder from the initial baseline still exists, but the canonical pipeline created in this step writes to `output/`.
