# Modeling the Individual Portfolio Returns

## Workflow

This section models each of the six value-weighted portfolio return series individually using the cleaned monthly dataset. For each portfolio, the pipeline saves a time-series plot, a histogram-density plot with fitted Gaussian, Student-t, and NIG overlays, a normal-versus-recommended QQ comparison, a recommended-distribution CDF diagnostic, ACF/PACF, residual diagnostics, volatility-clustering diagnostics, ARIMA candidate comparison tables, a selected ARIMA summary, and an `arch`-based volatility-model comparison estimated on ARIMA residuals.

## Distributional and Diagnostic Evidence

The descriptive diagnostics confirm the stylized facts emphasized in the lecture notes: the monthly portfolio returns are stationary in levels, but they are not well described by a Gaussian law. Jarque-Bera and Shapiro-Wilk tests reject normality for essentially the entire panel, and heavy-tailed MLE fits dominate the Gaussian benchmark throughout the sample. The recommended marginal-fit counts are **NIG: 1, Student-t: 5**. The largest AIC improvement over the Gaussian fit appears in **Small HiBM**, where the recommended distribution is **Student-t**.

Stationarity tests support modeling returns in levels rather than prices: the series are monthly returns, ADF generally rejects a unit root, and KPSS does not overturn the practical use of level ARIMA models. At the same time, ARCH-LM tests reject homoskedasticity throughout the panel, with the strongest raw volatility clustering in **ME2 BM2**.

The saved diagnostic artifacts for each portfolio are under:

- `output/figures/individual_returns/<portfolio>/`
- `output/tables/individual_returns/<portfolio>/`
- `output/models/individual_returns/<portfolio>/`

## Mean-Model Selection

AR, MA, and ARMA/ARIMA candidates were estimated with `statsmodels`. The selection rule was:

1. Use ADF and KPSS evidence to decide whether differenced candidates are even needed.
2. Use the ACF/PACF patterns as a rough guide, then fit a small interpretable ARIMA grid with orders up to two autoregressive and two moving-average terms.
3. Compare candidates on AIC and BIC.
4. Prefer models whose residual Ljung-Box p-value at lag 12 is at least 0.05 and whose dynamic parameters are mostly statistically significant when such terms are present.
5. Break ties in favor of the simpler specification.

This rule is intentionally conservative because monthly equity returns often contain much weaker mean predictability than volatility predictability. In practice, the selected ARIMA models are low-order and in some cases close to white-noise benchmarks, which is consistent with the financial-return literature reviewed in the introduction.

Residual diagnostics also show that mean dynamics are not captured equally well across all portfolios. The clearest remaining residual autocorrelation at lag 12 appears in **ME1 BM2, Small HiBM**, so those selected ARIMA specifications should be read as reasonable low-order benchmarks rather than fully satisfactory final mean models.

The combined comparison table is saved at `output/tables/individual_returns/portfolio_model_comparison_summary.csv`.

## Volatility Modeling

Conditional volatility was modeled with the canonical `arch` package rather than with a custom optimizer. For each portfolio, the selected ARIMA residuals were passed to four GARCH(1,1) specifications: Gaussian, Student-t, skewed Student-t, and GED. Preferred-model selection does not rely on one metric alone. It combines AIC/BIC, Ljung-Box tests on standardized residuals and squared standardized residuals, residual ARCH-LM tests, innovation KS tests under the assumed distribution, and the significance share of the core volatility parameters.

The selected volatility-model counts are **Skewed Student-t: 2, Student-t: 4**. The most persistent selected volatility process in the current run is **Small LoBM**, with alpha + beta = **0.979**.

The combined volatility summary is saved at `output/tables/individual_returns/portfolio_garch_summary.csv`.

## Cross-Portfolio Interpretation

The portfolio with the highest average monthly return remains **Small HiBM**, at **1.420%** per month. Across the six portfolios, the mean-model evidence is modest, while the volatility-model evidence is stronger and more systematic. That pattern supports an interpretation in which the conditional mean of monthly portfolio returns is only weakly predictable from its own history, but conditional risk is persistent and benefits from explicitly heavy-tailed volatility specifications.

Overall, the Section 3 evidence points to three conclusions. First, the return series are heavy tailed enough that Gaussian diagnostics alone are too narrow, so the report now uses multiple normality checks, MLE-based Normal/Student-t/NIG comparisons, and KS goodness-of-fit tests. Second, low-order ARIMA models are adequate as mean benchmarks, but they do not reveal strong standalone predictability. Third, volatility clustering is real enough to justify GARCH-type modeling with `arch`, and non-Gaussian innovation assumptions often fit better than Gaussian innovations even at the monthly frequency.

Portfolio-by-portfolio interpretations and the denser output inventory are moved to the appendix file `report/sections/appendix_individual_returns_modeling.md`.
