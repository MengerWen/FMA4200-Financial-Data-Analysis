# Appendices

## Appendix A. Portfolio-by-Portfolio Modeling Notes

This appendix records concise portfolio-by-portfolio interpretations generated from the saved diagnostics and model outputs.

## Small LoBM

- Mean and risk: 0.974% monthly mean, 7.432% monthly volatility.
- Shape: skewness = 0.566, kurtosis = 9.924, Jarque-Bera p-value = 0.0000, Shapiro-Wilk p-value = 0.0000, and best marginal fit = Student-t.
- Stationarity: ADF p-value = 0.0000, KPSS p-value = 0.1000.
- Selected ARIMA model: (0, 0, 2) with AIC = 8147.23, BIC = 8167.57, and residual Ljung-Box p-value at lag 12 = 0.3142.
- Selected volatility model: Student-t GARCH(1,1) with alpha = 0.1415, beta = 0.8370, persistence = 0.9785, standardized-squared-residual Ljung-Box p-value at lag 12 = 0.9581, and standardized-residual ARCH-LM p-value = 0.9575.
- Saved outputs: `output/figures/individual_returns/small_lobm/`, `output/tables/individual_returns/small_lobm/`, and `output/models/individual_returns/small_lobm/`.

## ME1 BM2

- Mean and risk: 1.236% monthly mean, 6.942% monthly volatility.
- Shape: skewness = 1.105, kurtosis = 16.406, Jarque-Bera p-value = 0.0000, Shapiro-Wilk p-value = 0.0000, and best marginal fit = Student-t.
- Stationarity: ADF p-value = 0.0000, KPSS p-value = 0.1000.
- Selected ARIMA model: (2, 0, 2) with AIC = 7970.82, BIC = 8001.32, and residual Ljung-Box p-value at lag 12 = 0.0053.
- Selected volatility model: Student-t GARCH(1,1) with alpha = 0.1282, beta = 0.8415, persistence = 0.9697, standardized-squared-residual Ljung-Box p-value at lag 12 = 0.8625, and standardized-residual ARCH-LM p-value = 0.8717.
- Saved outputs: `output/figures/individual_returns/me1_bm2/`, `output/tables/individual_returns/me1_bm2/`, and `output/models/individual_returns/me1_bm2/`.

## Small HiBM

- Mean and risk: 1.420% monthly mean, 8.080% monthly volatility.
- Shape: skewness = 1.960, kurtosis = 23.855, Jarque-Bera p-value = 0.0000, Shapiro-Wilk p-value = 0.0000, and best marginal fit = Student-t.
- Stationarity: ADF p-value = 0.0000, KPSS p-value = 0.1000.
- Selected ARIMA model: (2, 0, 2) with AIC = 8327.37, BIC = 8357.87, and residual Ljung-Box p-value at lag 12 = 0.0022.
- Selected volatility model: Student-t GARCH(1,1) with alpha = 0.1281, beta = 0.8502, persistence = 0.9783, standardized-squared-residual Ljung-Box p-value at lag 12 = 0.9682, and standardized-residual ARCH-LM p-value = 0.9680.
- Saved outputs: `output/figures/individual_returns/small_hibm/`, `output/tables/individual_returns/small_hibm/`, and `output/models/individual_returns/small_hibm/`.

## Big LoBM

- Mean and risk: 0.958% monthly mean, 5.266% monthly volatility.
- Shape: skewness = -0.128, kurtosis = 8.203, Jarque-Bera p-value = 0.0000, Shapiro-Wilk p-value = 0.0000, and best marginal fit = Student-t.
- Stationarity: ADF p-value = 0.0000, KPSS p-value = 0.1000.
- Selected ARIMA model: (2, 0, 2) with AIC = 7336.09, BIC = 7366.59, and residual Ljung-Box p-value at lag 12 = 0.9774.
- Selected volatility model: Student-t GARCH(1,1) with alpha = 0.1231, beta = 0.8437, persistence = 0.9668, standardized-squared-residual Ljung-Box p-value at lag 12 = 0.6281, and standardized-residual ARCH-LM p-value = 0.6321.
- Saved outputs: `output/figures/individual_returns/big_lobm/`, `output/tables/individual_returns/big_lobm/`, and `output/models/individual_returns/big_lobm/`.

## ME2 BM2

- Mean and risk: 0.962% monthly mean, 5.602% monthly volatility.
- Shape: skewness = 1.160, kurtosis = 20.201, Jarque-Bera p-value = 0.0000, Shapiro-Wilk p-value = 0.0000, and best marginal fit = Student-t.
- Stationarity: ADF p-value = 0.0000, KPSS p-value = 0.1000.
- Selected ARIMA model: (2, 0, 2) with AIC = 7457.81, BIC = 7488.31, and residual Ljung-Box p-value at lag 12 = 0.0567.
- Selected volatility model: Student-t GARCH(1,1) with alpha = 0.1320, beta = 0.8297, persistence = 0.9617, standardized-squared-residual Ljung-Box p-value at lag 12 = 0.3782, and standardized-residual ARCH-LM p-value = 0.4276.
- Saved outputs: `output/figures/individual_returns/me2_bm2/`, `output/tables/individual_returns/me2_bm2/`, and `output/models/individual_returns/me2_bm2/`.

## Big HiBM

- Mean and risk: 1.213% monthly mean, 7.081% monthly volatility.
- Shape: skewness = 1.443, kurtosis = 20.449, Jarque-Bera p-value = 0.0000, Shapiro-Wilk p-value = 0.0000, and best marginal fit = Student-t.
- Stationarity: ADF p-value = 0.0000, KPSS p-value = 0.1000.
- Selected ARIMA model: (2, 0, 2) with AIC = 8027.05, BIC = 8057.55, and residual Ljung-Box p-value at lag 12 = 0.0673.
- Selected volatility model: Student-t GARCH(1,1) with alpha = 0.1331, beta = 0.8289, persistence = 0.9620, standardized-squared-residual Ljung-Box p-value at lag 12 = 0.6114, and standardized-residual ARCH-LM p-value = 0.6704.
- Saved outputs: `output/figures/individual_returns/big_hibm/`, `output/tables/individual_returns/big_hibm/`, and `output/models/individual_returns/big_hibm/`.

## Appendix B. Supplementary Diagnostic Material

The main body cites the highest-signal tables and figures only. To keep the report body compact, the following diagnostics remain in the saved output folders rather than being reproduced inline:

- `output/figures/individual_returns/<portfolio>/`: time-series plots, histograms with density overlays, QQ plots, ACF/PACF, residual diagnostics, and volatility-clustering figures for each portfolio.
- `output/tables/individual_returns/<portfolio>/`: detailed descriptive statistics, Jarque-Bera and Shapiro-Wilk output, fitted normal-versus-Student-t comparisons, ADF and KPSS tests, Ljung-Box diagnostics, ARCH-LM tests, and candidate-model comparison tables.
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
