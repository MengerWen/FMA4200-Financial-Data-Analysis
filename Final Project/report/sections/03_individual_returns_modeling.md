# Modeling the Individual Portfolio Returns

## Univariate Benchmarks

Section 3 begins with the univariate benchmark workflow for each of the six cleaned monthly portfolio return series. Each series was examined with time-series plots, histogram-and-density plots, QQ plots, ACF/PACF, Jarque-Bera normality tests, ADF and KPSS stationarity checks, residual Ljung-Box diagnostics, and ARCH-LM tests. Those diagnostics support modeling the returns in levels while still showing clear non-normality and volatility clustering across the panel.

The strongest Jarque-Bera rejection still appears in **Small HiBM**, while the most persistent volatility process remains **Small HiBM** with GARCH persistence **0.980**. Low-order ARIMA models remain sensible mean benchmarks, but the clearest remaining residual autocorrelation still shows up in **ME1 BM2, Small HiBM**, which motivates adding exogenous predictive information rather than relying only on own-history dynamics.

## Exogenous Predictors and Predictive Design

The predictive extension now augments the benchmark models with lagged exogenous predictors. In the current environment, the project was able to use **cached authoritative Fama-French monthly factors** merged to the same sample and stored locally for reproducibility. The authoritative factor block contributes lagged market excess return, SMB, HML, and the risk-free rate. The project also constructs internal fallback and supplemental signals from the six available portfolios, including lagged size and value spreads, a 12-month rolling market-volatility proxy, a 12-month market-momentum proxy, and a drawdown proxy. This design keeps the workflow usable even if future reruns cannot reach the external data source, because both the cached factor file and the internal predictors live inside the project folder.

For each portfolio, three model classes are compared:

1. The selected univariate ARIMA benchmark from the earlier step.
2. An ARIMAX specification that keeps the same ARIMA order but adds lagged exogenous predictors.
3. A predictive regression with lagged returns plus lagged exogenous predictors.

Model comparison is explicit rather than informal. Full-sample fit is judged with AIC, BIC, residual Ljung-Box diagnostics, and the share of non-constant terms that are statistically significant at the 5% level. Forecast performance is judged with a 120-month expanding-window exercise using one-step-ahead monthly refits. The main out-of-sample metrics are RMSE, MAE, and directional accuracy.

## Predictive Results

The most common preferred predictive family across the six portfolios is **Predictive regression with lagged factors and internal signals**. Out-of-sample gains are modest rather than dramatic, which is consistent with the literature on return predictability at the monthly horizon. Even so, **2 of 6** portfolios show an RMSE improvement relative to the benchmark when the best predictive extension is used. The strongest improvement occurs for **Big HiBM**, where the preferred predictive model changes RMSE by **5.00%** relative to the benchmark.

These results should be interpreted cautiously. The predictive models add economic structure and sometimes improve forecast ranking or directional information, but the benchmark ARIMA models remain difficult to beat consistently. That pattern is informative in itself: the conditional mean of long-run monthly portfolio returns appears only weakly predictable, while the stronger and more stable evidence in the project still lies in distributional shape and conditional volatility.

## Saved Outputs

The predictive-modeling artifacts are saved under:

- `data/processed/predictor_dataset_monthly.csv`
- `data/processed/predictor_source_summary.csv`
- `output/figures/predictive_individual_returns/<portfolio>/`
- `output/tables/predictive_individual_returns/<portfolio>/`
- `output/models/predictive_individual_returns/<portfolio>/`
- `output/tables/predictive_individual_returns/predictive_model_summary.csv`
- `output/tables/predictive_individual_returns/predictive_forecast_metrics.csv`
- `output/tables/predictive_individual_returns/predictive_forecasts.csv`

The appendix extends the portfolio-by-portfolio notes with predictive-model selections, key terms, and benchmark-versus-predictive forecast comparisons.
