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
- Added report-writing groundwork:
  - `report/outline.md`
  - `report/sections/01_introduction.md`
  - `report/sections/02_data_source_processing.md`
  - `report/references.md`
- Extended the reproducible pipeline with additional report-facing outputs:
  - `output/tables/portfolio_summary_snapshot.csv`
  - `output/tables/portfolio_correlation_matrix.csv`
  - `output/figures/portfolio_correlation_heatmap.png`
  - `report/sections/_02_data_snapshot_autogen.md`
- Completed a verified literature groundwork pass with 12 academic references plus the Kenneth French data-source entry recorded in `report/references.md`.
- Added the Section 3 univariate-modeling pipeline:
  - `src/fma4200_project/univariate_modeling.py`
  - `scripts/run_individual_modeling.py`
  - updated `scripts/run_pipeline.py` so the top-level pipeline now runs both cleaning and individual-return modeling
- Added Section 3 report outputs:
  - `report/sections/03_individual_returns_modeling.md`
  - `report/sections/appendix_individual_returns_modeling.md`
- Added combined modeling summary tables:
  - `output/tables/individual_returns/portfolio_statistical_test_summary.csv`
  - `output/tables/individual_returns/portfolio_model_comparison_summary.csv`
  - `output/tables/individual_returns/portfolio_garch_summary.csv`
- Added per-portfolio diagnostics, model tables, and text summaries under:
  - `output/figures/individual_returns/`
  - `output/tables/individual_returns/`
  - `output/models/individual_returns/`

## 2. What Ran Successfully

- Verified the required interpreter can import the packages actually used in this step:
  - `numpy 1.26.4`
  - `pandas 2.2.2`
  - `matplotlib 3.9.2`
  - `scipy 1.13.1`
  - `statsmodels 0.14.2`
- Ran successfully:
  - `& 'd:\MG\anaconda3\python.exe' 'scripts\check_environment.py'`
  - `& 'd:\MG\anaconda3\python.exe' 'scripts\clean_data.py'`
  - `& 'd:\MG\anaconda3\python.exe' 'scripts\run_individual_modeling.py'`
  - `& 'd:\MG\anaconda3\python.exe' 'scripts\run_pipeline.py'`
- Re-ran successfully after adding report-facing tables and figures:
  - `& 'd:\MG\anaconda3\python.exe' 'scripts\run_pipeline.py'`
- Confirmed the cleaned monthly sample spans `1926-07-31` to `2026-01-31`.
- Confirmed the cleaned dataset contains `1195` monthly observations and `6` value-weighted portfolio return columns.
- Confirmed duplicate-date rows: `0`.
- Confirmed missing values after sentinel conversion: `0`.
- Saved date coverage, duplicate check, missingness check, and descriptive statistics tables under `output/tables/`.
- Saved a data dictionary documenting the percent and decimal return conventions.
- Saved a report-ready summary snapshot and correlation matrix under `output/tables/`.
- Saved a report-ready correlation heatmap under `output/figures/`.
- Modeled all `6` portfolios individually and saved per-series time-series plots, histogram-density plots, QQ plots, ACF/PACF plots, residual diagnostics, volatility-clustering diagnostics, ARIMA candidate comparisons, selected-model parameter tables, and GARCH summaries.
- Confirmed the hand-rolled Gaussian GARCH(1,1) estimator converged for all six portfolios and substantially reduced ARCH effects in standardized residuals.

## 3. What Remains

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
- The manually written report sections in `report/sections/` are now separate from the auto-generated `_02_data_snapshot_autogen.md` file so future pipeline reruns do not overwrite the narrative draft.
- The custom GARCH(1,1) step is intentionally limited to Gaussian constant-mean volatility modeling because the `arch` package is unavailable. The saved results are suitable for Section 3 diagnostics and comparisons, but not a substitute for the broader inference and model families available in `arch`.
