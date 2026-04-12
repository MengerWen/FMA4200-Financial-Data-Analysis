# Big HiBM

- Distribution: mean = 1.213% per month, volatility = 7.081% per month, skewness = 1.443, kurtosis = 20.449.
- Normality: The Jarque-Bera test rejects normality.
- Stationarity: Both ADF and KPSS support treating the return series as stationary in levels.
- Selected mean model: ARIMA(2, 0, 2) chosen because lowest BIC within 2 AIC points among models with acceptable residual autocorrelation and mostly significant dynamic terms.
- Residual autocorrelation: Ljung-Box p-value at lag 12 = 0.067.
- Volatility model: Gaussian GARCH(1,1) estimates alpha = 0.151, beta = 0.817, and persistence = 0.968.
- Volatility clustering: raw ARCH-LM p-value = 0.0000; standardized-residual ARCH-LM p-value = 0.9069. ARCH effects are materially weaker after GARCH standardization.
