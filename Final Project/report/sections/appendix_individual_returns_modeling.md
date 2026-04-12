# Appendix: Individual Portfolio Modeling Details

This appendix records concise portfolio-by-portfolio interpretations generated from the saved diagnostics and model outputs.

## Small LoBM

- Mean and risk: 0.974% monthly mean, 7.432% monthly volatility.
- Shape: skewness = 0.566, kurtosis = 9.924, Jarque-Bera p-value = 0.0000, Shapiro-Wilk p-value = 0.0000, and best marginal fit = Student-t (AIC gain vs Gaussian = 206.58).
- Stationarity: ADF p-value = 0.0000, KPSS p-value = 0.1000.
- Selected ARIMA model: (0, 0, 2) with AIC = 8147.23, BIC = 8167.57, and residual Ljung-Box p-value at lag 12 = 0.3142.
- Selected volatility model: Student-t GARCH(1,1) with alpha = 0.1415, beta = 0.8370, persistence = 0.9785, innovation KS p-value = 0.7917, standardized-squared-residual Ljung-Box p-value at lag 12 = 0.9581, and standardized-residual ARCH-LM p-value = 0.9575.
- Saved outputs: `output/figures/individual_returns/small_lobm/`, `output/tables/individual_returns/small_lobm/`, and `output/models/individual_returns/small_lobm/`.

## ME1 BM2

- Mean and risk: 1.236% monthly mean, 6.942% monthly volatility.
- Shape: skewness = 1.105, kurtosis = 16.406, Jarque-Bera p-value = 0.0000, Shapiro-Wilk p-value = 0.0000, and best marginal fit = Student-t (AIC gain vs Gaussian = 359.46).
- Stationarity: ADF p-value = 0.0000, KPSS p-value = 0.1000.
- Selected ARIMA model: (2, 0, 2) with AIC = 7970.82, BIC = 8001.32, and residual Ljung-Box p-value at lag 12 = 0.0053.
- Selected volatility model: Skewed Student-t GARCH(1,1) with alpha = 0.1275, beta = 0.8465, persistence = 0.9740, innovation KS p-value = 0.6206, standardized-squared-residual Ljung-Box p-value at lag 12 = 0.8881, and standardized-residual ARCH-LM p-value = 0.8958.
- Saved outputs: `output/figures/individual_returns/me1_bm2/`, `output/tables/individual_returns/me1_bm2/`, and `output/models/individual_returns/me1_bm2/`.

## Small HiBM

- Mean and risk: 1.420% monthly mean, 8.080% monthly volatility.
- Shape: skewness = 1.960, kurtosis = 23.855, Jarque-Bera p-value = 0.0000, Shapiro-Wilk p-value = 0.0000, and best marginal fit = Student-t (AIC gain vs Gaussian = 494.95).
- Stationarity: ADF p-value = 0.0000, KPSS p-value = 0.1000.
- Selected ARIMA model: (2, 0, 2) with AIC = 8327.37, BIC = 8357.87, and residual Ljung-Box p-value at lag 12 = 0.0022.
- Selected volatility model: Student-t GARCH(1,1) with alpha = 0.1281, beta = 0.8502, persistence = 0.9783, innovation KS p-value = 0.6205, standardized-squared-residual Ljung-Box p-value at lag 12 = 0.9682, and standardized-residual ARCH-LM p-value = 0.9680.
- Saved outputs: `output/figures/individual_returns/small_hibm/`, `output/tables/individual_returns/small_hibm/`, and `output/models/individual_returns/small_hibm/`.

## Big LoBM

- Mean and risk: 0.958% monthly mean, 5.266% monthly volatility.
- Shape: skewness = -0.128, kurtosis = 8.203, Jarque-Bera p-value = 0.0000, Shapiro-Wilk p-value = 0.0000, and best marginal fit = NIG (AIC gain vs Gaussian = 171.55).
- Stationarity: ADF p-value = 0.0000, KPSS p-value = 0.1000.
- Selected ARIMA model: (2, 0, 2) with AIC = 7336.09, BIC = 7366.59, and residual Ljung-Box p-value at lag 12 = 0.9774.
- Selected volatility model: Skewed Student-t GARCH(1,1) with alpha = 0.1270, beta = 0.8423, persistence = 0.9694, innovation KS p-value = 0.7767, standardized-squared-residual Ljung-Box p-value at lag 12 = 0.6356, and standardized-residual ARCH-LM p-value = 0.6408.
- Saved outputs: `output/figures/individual_returns/big_lobm/`, `output/tables/individual_returns/big_lobm/`, and `output/models/individual_returns/big_lobm/`.

## ME2 BM2

- Mean and risk: 0.962% monthly mean, 5.602% monthly volatility.
- Shape: skewness = 1.160, kurtosis = 20.201, Jarque-Bera p-value = 0.0000, Shapiro-Wilk p-value = 0.0000, and best marginal fit = Student-t (AIC gain vs Gaussian = 403.16).
- Stationarity: ADF p-value = 0.0000, KPSS p-value = 0.1000.
- Selected ARIMA model: (2, 0, 2) with AIC = 7457.81, BIC = 7488.31, and residual Ljung-Box p-value at lag 12 = 0.0567.
- Selected volatility model: Student-t GARCH(1,1) with alpha = 0.1320, beta = 0.8297, persistence = 0.9617, innovation KS p-value = 0.2626, standardized-squared-residual Ljung-Box p-value at lag 12 = 0.3782, and standardized-residual ARCH-LM p-value = 0.4276.
- Saved outputs: `output/figures/individual_returns/me2_bm2/`, `output/tables/individual_returns/me2_bm2/`, and `output/models/individual_returns/me2_bm2/`.

## Big HiBM

- Mean and risk: 1.213% monthly mean, 7.081% monthly volatility.
- Shape: skewness = 1.443, kurtosis = 20.449, Jarque-Bera p-value = 0.0000, Shapiro-Wilk p-value = 0.0000, and best marginal fit = Student-t (AIC gain vs Gaussian = 432.02).
- Stationarity: ADF p-value = 0.0000, KPSS p-value = 0.1000.
- Selected ARIMA model: (2, 0, 2) with AIC = 8027.05, BIC = 8057.55, and residual Ljung-Box p-value at lag 12 = 0.0673.
- Selected volatility model: Student-t GARCH(1,1) with alpha = 0.1331, beta = 0.8289, persistence = 0.9620, innovation KS p-value = 0.7416, standardized-squared-residual Ljung-Box p-value at lag 12 = 0.6114, and standardized-residual ARCH-LM p-value = 0.6704.
- Saved outputs: `output/figures/individual_returns/big_hibm/`, `output/tables/individual_returns/big_hibm/`, and `output/models/individual_returns/big_hibm/`.

