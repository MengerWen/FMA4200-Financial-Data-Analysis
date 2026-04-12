# Small HiBM

- Distribution: mean = 1.420% per month, volatility = 8.080% per month, skewness = 1.960, kurtosis = 23.855.
- Normality: The Jarque-Bera test rejects normality.
- Stationarity: Both ADF and KPSS support treating the return series as stationary in levels.
- Selected mean model: ARIMA(2, 0, 2) chosen because lowest BIC within 2 AIC points because no candidate met the preferred residual-and-significance filter.
- Residual autocorrelation: Ljung-Box p-value at lag 12 = 0.002.
- Volatility model: Gaussian GARCH(1,1) estimates alpha = 0.141, beta = 0.840, and persistence = 0.980.
- Volatility clustering: raw ARCH-LM p-value = 0.0000; standardized-residual ARCH-LM p-value = 0.9316. ARCH effects are materially weaker after GARCH standardization.
