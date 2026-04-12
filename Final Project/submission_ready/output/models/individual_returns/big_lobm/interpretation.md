# Big LoBM

- Distribution: mean = 0.958% per month, volatility = 5.266% per month, skewness = -0.128, kurtosis = 8.203.
- Normality: The Jarque-Bera test rejects normality.
- Stationarity: Both ADF and KPSS support treating the return series as stationary in levels.
- Selected mean model: ARIMA(2, 0, 2) chosen because lowest BIC within 2 AIC points among models with acceptable residual autocorrelation and mostly significant dynamic terms.
- Residual autocorrelation: Ljung-Box p-value at lag 12 = 0.977.
- Volatility model: Gaussian GARCH(1,1) estimates alpha = 0.134, beta = 0.847, and persistence = 0.980.
- Volatility clustering: raw ARCH-LM p-value = 0.0000; standardized-residual ARCH-LM p-value = 0.7174. ARCH effects are materially weaker after GARCH standardization.
