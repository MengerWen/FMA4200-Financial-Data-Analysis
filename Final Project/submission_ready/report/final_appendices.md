# Appendices

## Appendix A. Portfolio-by-Portfolio Modeling Notes

This appendix records concise portfolio-by-portfolio notes from both the univariate benchmark stage and the exogenous predictive-modeling extension.

## Small LoBM

- Distribution: mean = 0.974% per month, volatility = 7.432% per month, skewness = 0.566, kurtosis = 9.924.
- Benchmark ARIMA: (0, 0, 2) with residual Ljung-Box p-value at lag 12 = 0.3142.
- GARCH(1,1): persistence = 0.9791, standardized-residual ARCH-LM p-value = 0.9418.
- Predictive benchmark forecast metrics: RMSE = 6.6525, MAE = 4.8813, directional accuracy = 0.608.
- Preferred predictive model: Predictive regression with lagged factors and internal signals with RMSE = 6.7337, MAE = 4.9446, directional accuracy = 0.533, and RMSE improvement vs benchmark = -1.22%.
- Most informative predictive terms: market_vol_12m_lag1_pct (positive, p=0.067); value_spread_lag1_pct (positive, p=0.082); ff_hml_lag1_pct (negative, p=0.083)
- Saved predictive outputs: `output/figures/predictive_individual_returns/small_lobm/`, `output/tables/predictive_individual_returns/small_lobm/`, and `output/models/predictive_individual_returns/small_lobm/`.

## ME1 BM2

- Distribution: mean = 1.236% per month, volatility = 6.942% per month, skewness = 1.105, kurtosis = 16.406.
- Benchmark ARIMA: (2, 0, 2) with residual Ljung-Box p-value at lag 12 = 0.0053.
- GARCH(1,1): persistence = 0.9758, standardized-residual ARCH-LM p-value = 0.8947.
- Predictive benchmark forecast metrics: RMSE = 6.1671, MAE = 4.6652, directional accuracy = 0.575.
- Preferred predictive model: Predictive regression with lagged factors and internal signals with RMSE = 6.2379, MAE = 4.6976, directional accuracy = 0.583, and RMSE improvement vs benchmark = -1.15%.
- Most informative predictive terms: value_spread_lag1_pct (positive, p=0.056); ff_hml_lag1_pct (negative, p=0.057)
- Saved predictive outputs: `output/figures/predictive_individual_returns/me1_bm2/`, `output/tables/predictive_individual_returns/me1_bm2/`, and `output/models/predictive_individual_returns/me1_bm2/`.

## Small HiBM

- Distribution: mean = 1.420% per month, volatility = 8.080% per month, skewness = 1.960, kurtosis = 23.855.
- Benchmark ARIMA: (2, 0, 2) with residual Ljung-Box p-value at lag 12 = 0.0022.
- GARCH(1,1): persistence = 0.9803, standardized-residual ARCH-LM p-value = 0.9316.
- Predictive benchmark forecast metrics: RMSE = 6.8517, MAE = 5.1519, directional accuracy = 0.542.
- Preferred predictive model: Predictive regression with lagged factors and internal signals with RMSE = 6.9063, MAE = 5.2668, directional accuracy = 0.550, and RMSE improvement vs benchmark = -0.80%.
- Most informative predictive terms: value_spread_lag1_pct (positive, p=0.098); ff_hml_lag1_pct (negative, p=0.099)
- Saved predictive outputs: `output/figures/predictive_individual_returns/small_hibm/`, `output/tables/predictive_individual_returns/small_hibm/`, and `output/models/predictive_individual_returns/small_hibm/`.

## Big LoBM

- Distribution: mean = 0.958% per month, volatility = 5.266% per month, skewness = -0.128, kurtosis = 8.203.
- Benchmark ARIMA: (2, 0, 2) with residual Ljung-Box p-value at lag 12 = 0.9774.
- GARCH(1,1): persistence = 0.9800, standardized-residual ARCH-LM p-value = 0.7174.
- Predictive benchmark forecast metrics: RMSE = 4.7003, MAE = 3.6290, directional accuracy = 0.625.
- Preferred predictive model: ARIMAX with lagged Fama-French factors with RMSE = 4.7330, MAE = 3.6268, directional accuracy = 0.650, and RMSE improvement vs benchmark = -0.70%.
- Most informative predictive terms: ma.L2 (positive, p=0.000); ar.L2 (negative, p=0.000); ma.L1 (positive, p=0.000)
- Saved predictive outputs: `output/figures/predictive_individual_returns/big_lobm/`, `output/tables/predictive_individual_returns/big_lobm/`, and `output/models/predictive_individual_returns/big_lobm/`.

## ME2 BM2

- Distribution: mean = 0.962% per month, volatility = 5.602% per month, skewness = 1.160, kurtosis = 20.201.
- Benchmark ARIMA: (2, 0, 2) with residual Ljung-Box p-value at lag 12 = 0.0567.
- GARCH(1,1): persistence = 0.9798, standardized-residual ARCH-LM p-value = 0.5653.
- Predictive benchmark forecast metrics: RMSE = 4.8070, MAE = 3.4897, directional accuracy = 0.583.
- Preferred predictive model: ARIMAX with lagged Fama-French factors with RMSE = 4.6677, MAE = 3.3890, directional accuracy = 0.592, and RMSE improvement vs benchmark = 2.90%.
- Most informative predictive terms: ar.L2 (negative, p=0.000); ar.L1 (positive, p=0.000); ff_hml_lag1_pct (positive, p=0.000)
- Saved predictive outputs: `output/figures/predictive_individual_returns/me2_bm2/`, `output/tables/predictive_individual_returns/me2_bm2/`, and `output/models/predictive_individual_returns/me2_bm2/`.

## Big HiBM

- Distribution: mean = 1.213% per month, volatility = 7.081% per month, skewness = 1.443, kurtosis = 20.449.
- Benchmark ARIMA: (2, 0, 2) with residual Ljung-Box p-value at lag 12 = 0.0673.
- GARCH(1,1): persistence = 0.9678, standardized-residual ARCH-LM p-value = 0.9069.
- Predictive benchmark forecast metrics: RMSE = 6.4570, MAE = 4.8316, directional accuracy = 0.558.
- Preferred predictive model: Predictive regression with lagged factors and internal signals with RMSE = 6.1340, MAE = 4.5624, directional accuracy = 0.625, and RMSE improvement vs benchmark = 5.00%.
- Most informative predictive terms: No non-constant terms are significant at the 10% level.
- Saved predictive outputs: `output/figures/predictive_individual_returns/big_hibm/`, `output/tables/predictive_individual_returns/big_hibm/`, and `output/models/predictive_individual_returns/big_hibm/`.

## Appendix B. Supplementary Diagnostic Material

The main body cites the highest-signal tables and figures only. To keep the report body compact, the following diagnostics remain in the saved output folders rather than being reproduced inline:

- `output/figures/individual_returns/<portfolio>/`: time-series plots, histograms with density overlays, QQ plots, ACF/PACF, residual diagnostics, and volatility-clustering figures for each portfolio.
- `output/tables/individual_returns/<portfolio>/`: detailed descriptive statistics, Jarque-Bera output, ADF and KPSS tests, Ljung-Box diagnostics, ARCH-LM tests, and candidate-model comparison tables.
- `output/figures/predictive_individual_returns/<portfolio>/`: forecast-comparison figures for the exogenous predictive models.
- `output/tables/predictive_individual_returns/<portfolio>/`: fitted predictive-model summaries and forecast diagnostics.
- `output/figures/trading_strategies/var_irf_grid.png`: reduced-form impulse-response overview for the VAR.
- `output/figures/trading_strategies/var_fevd.png`: forecast-error variance decomposition summary for the VAR.
- `output/figures/trading_strategies/stat_arb_signal.png`: rolling spread z-score and trading signals for the cointegration strategy.

## Appendix C. Reproducibility Note

All tables, figures, cleaned datasets, and report outputs in this project are generated inside `Final Project/` with the fixed interpreter required by the assignment. The canonical end-to-end rerun command remains:

```powershell
& 'd:\MG\anaconda3\python.exe' 'scripts\run_pipeline.py'
```
