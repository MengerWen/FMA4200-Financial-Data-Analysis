# ME1 BM2

- Distribution: mean = 1.236% per month, volatility = 6.942% per month, skewness = 1.105, kurtosis = 16.406.
- Normality: The Jarque-Bera test rejects normality.
- Stationarity: Both ADF and KPSS support treating the return series as stationary in levels.
- Selected mean model: ARIMA(2, 0, 2) chosen because lowest BIC within 2 AIC points because no candidate met the preferred residual-and-significance filter.
- Residual autocorrelation: Ljung-Box p-value at lag 12 = 0.005.
- Volatility model: Gaussian GARCH(1,1) estimates alpha = 0.137, beta = 0.839, and persistence = 0.976.
- Volatility clustering: raw ARCH-LM p-value = 0.0000; standardized-residual ARCH-LM p-value = 0.8947. ARCH effects are materially weaker after GARCH standardization.
