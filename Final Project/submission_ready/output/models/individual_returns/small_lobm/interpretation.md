# Small LoBM

- Distribution: mean = 0.974% per month, volatility = 7.432% per month, skewness = 0.566, kurtosis = 9.924.
- Normality: The Jarque-Bera test rejects normality.
- Stationarity: Both ADF and KPSS support treating the return series as stationary in levels.
- Selected mean model: ARIMA(0, 0, 2) chosen because lowest BIC within 2 AIC points among models with acceptable residual autocorrelation and mostly significant dynamic terms.
- Residual autocorrelation: Ljung-Box p-value at lag 12 = 0.314.
- Volatility model: Gaussian GARCH(1,1) estimates alpha = 0.140, beta = 0.839, and persistence = 0.979.
- Volatility clustering: raw ARCH-LM p-value = 0.0000; standardized-residual ARCH-LM p-value = 0.9418. ARCH effects are materially weaker after GARCH standardization.
