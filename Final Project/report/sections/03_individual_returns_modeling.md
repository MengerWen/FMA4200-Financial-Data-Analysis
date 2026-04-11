# Modeling the Individual Portfolio Returns

## Workflow

This section models each of the six value-weighted portfolio return series individually using the cleaned monthly dataset. For each portfolio, the pipeline saves a time-series plot, histogram-and-density plot, QQ plot, ACF/PACF, residual diagnostics, volatility-clustering diagnostics, ARIMA candidate comparison tables, a selected ARIMA summary, and a lightweight Gaussian GARCH(1,1) fit implemented with `scipy.optimize` because the `arch` package is unavailable in the required interpreter.

## Distributional and Diagnostic Evidence

The descriptive diagnostics indicate that return distributions are not well approximated by the Gaussian benchmark. The strongest Jarque-Bera rejection in this sample occurs for **Small HiBM**, and every portfolio exhibits nontrivial skewness or excess kurtosis. Stationarity tests support modeling returns in levels rather than prices: the series are monthly returns, ADF generally rejects a unit root, and KPSS results do not overturn the practical use of level ARIMA models.

The saved diagnostic artifacts for each portfolio are under:

- `output/figures/individual_returns/<portfolio>/`
- `output/tables/individual_returns/<portfolio>/`
- `output/models/individual_returns/<portfolio>/`

## Mean-Model Selection

AR, MA, and ARMA/ARIMA candidates were estimated with `statsmodels`. The selection rule was:

1. Use ADF and KPSS evidence to decide whether differenced candidates are even needed.
2. Fit a small, interpretable ARIMA grid with orders up to two autoregressive and two moving-average terms.
3. Compare candidates on AIC and BIC.
4. Prefer models whose residual Ljung-Box p-value at lag 12 is at least 0.05 and whose dynamic parameters are mostly statistically significant when such terms are present.
5. Break ties in favor of the simpler specification.

This rule is intentionally conservative because monthly equity returns often contain much weaker mean predictability than volatility predictability. In practice, the selected ARIMA models are low-order and in some cases close to white-noise benchmarks, which is consistent with the financial-return literature reviewed in the introduction.

Residual diagnostics also show that mean dynamics are not captured equally well across all portfolios. The clearest remaining residual autocorrelation at lag 12 appears in **ME1 BM2, Small HiBM**, so those selected ARIMA specifications should be read as reasonable low-order benchmarks rather than fully satisfactory final mean models.

The combined comparison table is saved at `output/tables/individual_returns/portfolio_model_comparison_summary.csv`.

## Volatility Modeling

A Gaussian GARCH(1,1) model with constant mean was estimated separately for each portfolio using a constrained parameterization and `scipy.optimize.minimize`. This hand-rolled approach is more limited than the unavailable `arch` package, especially for formal inference and richer specifications, but it is stable enough here to provide a reproducible volatility benchmark. The key use of the GARCH fits is to measure volatility persistence and to check whether standardized residuals show weaker ARCH effects than the raw series.

The most persistent volatility process in the current run is **Small HiBM**, with alpha + beta = **0.980**. That level of persistence is economically plausible for long samples of monthly equity data and supports the inclusion of volatility-focused models in the project.

The combined GARCH summary is saved at `output/tables/individual_returns/portfolio_garch_summary.csv`.

## Cross-Portfolio Interpretation

The portfolio with the highest average monthly return remains **Small HiBM**, at **1.420%** per month. Across the six portfolios, the mean-model evidence is modest, while the volatility-model evidence is stronger. That pattern supports an interpretation in which the conditional mean of monthly portfolio returns is only weakly predictable from its own history, but conditional risk is more persistent and structured.

Overall, the Section 3 evidence points to three conclusions. First, the return series are heavy tailed enough that normality-based intuition should be treated carefully. Second, low-order ARIMA models are adequate as mean benchmarks, but they do not reveal strong standalone predictability. Third, volatility clustering is real enough to justify GARCH-type modeling, even at the monthly frequency. These results set up the next project stages naturally: exogenous predictors for the conditional mean and multivariate dependence modeling for trading strategies.

Portfolio-by-portfolio interpretations and the denser output inventory are moved to the appendix file `report/sections/appendix_individual_returns_modeling.md`.
