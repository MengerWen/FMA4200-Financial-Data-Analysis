# Project Status

## Last Updated

2026-04-12

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
- Added the predictive-modeling extension:
  - `src/fma4200_project/predictive_modeling.py`
  - `scripts/run_predictive_modeling.py`
  - updated `scripts/run_pipeline.py` so the top-level pipeline now runs cleaning, univariate modeling, and predictive modeling
- Added authoritative exogenous data and derived predictor artifacts:
  - `data/processed/fama_french_3f_monthly.csv`
  - `data/processed/predictor_dataset_monthly.csv`
  - `data/processed/predictor_source_summary.csv`
- Added predictive-modeling outputs under:
  - `output/figures/predictive_individual_returns/`
  - `output/tables/predictive_individual_returns/`
  - `output/models/predictive_individual_returns/`
- Added combined predictive summary tables:
  - `output/tables/predictive_individual_returns/predictive_model_summary.csv`
  - `output/tables/predictive_individual_returns/predictive_forecast_metrics.csv`
  - `output/tables/predictive_individual_returns/predictive_forecasts.csv`
- Updated the Section 3 report files so they now include the predictive-modeling stage:
  - `report/sections/03_individual_returns_modeling.md`
  - `report/sections/appendix_individual_returns_modeling.md`

## 2. What Ran Successfully

- Verified the required interpreter can import the packages actually used in this step:
  - `numpy 1.26.4`
  - `pandas 2.2.2`
  - `matplotlib 3.9.2`
  - `scipy 1.13.1`
  - `statsmodels 0.14.2`
  - `pandas_datareader 0.10.0`
- Ran successfully:
  - `& 'd:\MG\anaconda3\python.exe' 'scripts\check_environment.py'`
  - `& 'd:\MG\anaconda3\python.exe' 'scripts\clean_data.py'`
  - `& 'd:\MG\anaconda3\python.exe' 'scripts\run_individual_modeling.py'`
  - `& 'd:\MG\anaconda3\python.exe' 'scripts\run_predictive_modeling.py'`
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
- Downloaded and cached the authoritative monthly Fama-French factors once, then verified later reruns use the cached local file.
- Built a reproducible monthly predictor panel with lagged Fama-French factors, size and value spreads, rolling volatility, rolling momentum, and a drawdown proxy.
- Estimated predictive ARIMAX and predictive-regression models for all `6` portfolios and saved per-portfolio parameter tables, forecast paths, forecast-comparison figures, and written interpretations.
- Verified the predictive-modeling stage on a 120-month expanding-window one-step-ahead forecast design.
- Confirmed the preferred predictive extension beats the benchmark RMSE in `2` of `6` portfolios in the current run, with the strongest gains in:
  - `big_hibm_vwret_pct`: `+5.00%` RMSE improvement versus the benchmark
  - `me2_bm2_vwret_pct`: `+2.90%` RMSE improvement versus the benchmark
- Confirmed the most common preferred predictive family in the current run is the predictive regression with lagged factors and internal signals.

## 3. What Remains

- Build multivariate analysis for cointegration and statistical arbitrage.
- Implement mean-variance optimization, efficient frontier plots, and backtests against an equally weighted benchmark.
- Expand the report beyond Section 3 and integrate the saved predictive tables and figures into a final write-up.

## 4. Blockers or Assumptions

- Assumed the correct starting point is the monthly value-weighted return section in `Data.csv`, since that matches the assignment brief.
- Assumed the canonical cleaned dataset should keep the six return columns in **percent** units and save a separate decimal companion dataset for analysis that requires decimal returns.
- The authoritative exogenous predictor source in the current run is the cached monthly Fama-French factor file created through `pandas_datareader`; future reruns can use that local cache even if live download is unavailable.
- If future steps require additional internet-based inputs beyond the cached factors and access is unavailable, those steps will need a documented offline fallback.
- An earlier `outputs/` folder from the initial baseline still exists, but the canonical pipeline created in this step writes to `output/`.
- The manually written report sections in `report/sections/` are now separate from the auto-generated `_02_data_snapshot_autogen.md` file so future pipeline reruns do not overwrite the narrative draft.
- The custom GARCH(1,1) step is intentionally limited to Gaussian constant-mean volatility modeling because the `arch` package is unavailable. The saved results are suitable for Section 3 diagnostics and comparisons, but not a substitute for the broader inference and model families available in `arch`.
- The predictive out-of-sample evaluation now uses monthly expanding-window refits for one-step-ahead forecasts. This is computationally heavier than a fixed-parameter walk-forward update, but it is stable with the available toolchain and avoids relying on fragile `statsmodels` state-append behavior.
