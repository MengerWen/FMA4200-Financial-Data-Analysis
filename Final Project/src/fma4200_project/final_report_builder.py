from __future__ import annotations

import os
import shutil
import subprocess
import textwrap
from pathlib import Path

import nbformat
import numpy as np
import pandas as pd
from nbconvert import HTMLExporter

try:
    from nbconvert.exporters.pdf import PDFExporter
except Exception:  # pragma: no cover - dependency varies by environment
    PDFExporter = None

try:
    from nbconvert.exporters.webpdf import WebPDFExporter
except Exception:  # pragma: no cover - dependency varies by environment
    WebPDFExporter = None

from .config import (
    APPENDIX_INDIVIDUAL_PATH,
    COINTEGRATION_ORDER_TESTS_PATH,
    COINTEGRATION_SUMMARY_PATH,
    CORRELATION_MATRIX_PATH,
    DESCRIPTIVE_STATS_PCT_PATH,
    EXPORT_NOTES_PATH,
    FINAL_APPENDICES_PATH,
    FINAL_REPORT_HTML_PATH,
    FINAL_REPORT_LOG_PATH,
    FINAL_REPORT_MD_PATH,
    FINAL_REPORT_NOTEBOOK_PATH,
    FINAL_REPORT_PDF_PATH,
    PORTFOLIO_GARCH_SUMMARY_PATH,
    PORTFOLIO_MODEL_COMPARISON_PATH,
    PORTFOLIO_TEST_SUMMARY_PATH,
    PREDICTIVE_MODEL_SUMMARY_PATH,
    PROJECT_STATUS_PATH,
    RAW_DATA_PATH,
    REFERENCES_PATH,
    REPORT_DIR,
    REPORT_SECTIONS_DIR,
    STAT_ARB_SIGNAL_PATH,
    STRATEGY_METRICS_PATH,
    SUMMARY_SNAPSHOT_PATH,
    VAR_DIAGNOSTICS_PATH,
    VAR_LAG_SELECTION_PATH,
)


PORTFOLIO_LABELS = {
    "small_lobm_vwret_pct": "Small LoBM",
    "me1_bm2_vwret_pct": "ME1 BM2",
    "small_hibm_vwret_pct": "Small HiBM",
    "big_lobm_vwret_pct": "Big LoBM",
    "me2_bm2_vwret_pct": "ME2 BM2",
    "big_hibm_vwret_pct": "Big HiBM",
}

SHORT_MODEL_LABELS = {
    "Benchmark ARIMA": "Benchmark ARIMA",
    "ARIMAX with lagged Fama-French factors": "ARIMAX",
    "Predictive regression with lagged factors and internal signals": "Predictive regression",
}


def _relative_to_report(path: Path) -> str:
    return Path(os.path.relpath(path, REPORT_DIR)).as_posix()


def _format_pct(value: float, digits: int = 2) -> str:
    return f"{value:.{digits}f}%"


def _format_float(value: float, digits: int = 3) -> str:
    return f"{value:.{digits}f}"


def _format_pvalue(value: float) -> str:
    if pd.isna(value):
        return "NA"
    if value < 0.001:
        return "<0.001"
    return f"{value:.3f}"


def _markdown_table(dataframe: pd.DataFrame) -> str:
    headers = list(dataframe.columns)
    rows = dataframe.astype(str).values.tolist()
    separator = ["---"] * len(headers)
    table_lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(separator) + " |",
    ]
    for row in rows:
        table_lines.append("| " + " | ".join(row) + " |")
    return "\n".join(table_lines)


def _load_report_inputs() -> dict[str, pd.DataFrame]:
    return {
        "summary": pd.read_csv(SUMMARY_SNAPSHOT_PATH),
        "stats": pd.read_csv(PORTFOLIO_TEST_SUMMARY_PATH),
        "models": pd.read_csv(PORTFOLIO_MODEL_COMPARISON_PATH),
        "garch": pd.read_csv(PORTFOLIO_GARCH_SUMMARY_PATH),
        "predictive": pd.read_csv(PREDICTIVE_MODEL_SUMMARY_PATH),
        "strategy_metrics": pd.read_csv(STRATEGY_METRICS_PATH),
        "var_lags": pd.read_csv(VAR_LAG_SELECTION_PATH),
        "var_diag": pd.read_csv(VAR_DIAGNOSTICS_PATH),
        "integration": pd.read_csv(COINTEGRATION_ORDER_TESTS_PATH),
        "cointegration": pd.read_csv(COINTEGRATION_SUMMARY_PATH),
        "stat_arb_signals": pd.read_csv(STAT_ARB_SIGNAL_PATH, parse_dates=["date"]),
        "correlation": pd.read_csv(CORRELATION_MATRIX_PATH, index_col=0),
        "descriptive": pd.read_csv(DESCRIPTIVE_STATS_PCT_PATH),
    }


def _build_variable_table() -> str:
    variable_rows = pd.DataFrame(
        [
            {"Variable": "date", "Definition": "Month-end date derived from the raw YYYYMM code", "Unit": "date"},
            {"Variable": "small_lobm_vwret_pct", "Definition": "Small size, low book-to-market portfolio return", "Unit": "percent"},
            {"Variable": "me1_bm2_vwret_pct", "Definition": "Small size, middle book-to-market portfolio return", "Unit": "percent"},
            {"Variable": "small_hibm_vwret_pct", "Definition": "Small size, high book-to-market portfolio return", "Unit": "percent"},
            {"Variable": "big_lobm_vwret_pct", "Definition": "Big size, low book-to-market portfolio return", "Unit": "percent"},
            {"Variable": "me2_bm2_vwret_pct", "Definition": "Big size, middle book-to-market portfolio return", "Unit": "percent"},
            {"Variable": "big_hibm_vwret_pct", "Definition": "Big size, high book-to-market portfolio return", "Unit": "percent"},
        ]
    )
    return _markdown_table(variable_rows)


def _build_summary_table(summary: pd.DataFrame) -> str:
    table = summary.copy()
    table["Portfolio"] = table["portfolio"].map(PORTFOLIO_LABELS)
    table["Mean (%)"] = table["mean_pct"].map(lambda value: f"{value:.2f}")
    table["Annualized mean (%)"] = table["annualized_mean_pct"].map(lambda value: f"{value:.2f}")
    table["Volatility (%)"] = table["std_pct"].map(lambda value: f"{value:.2f}")
    table["Annualized volatility (%)"] = table["annualized_vol_pct"].map(lambda value: f"{value:.2f}")
    return _markdown_table(
        table[
            [
                "Portfolio",
                "Mean (%)",
                "Annualized mean (%)",
                "Volatility (%)",
                "Annualized volatility (%)",
            ]
        ]
    )


def _build_distribution_table(stats: pd.DataFrame) -> str:
    table = stats.copy()
    table["Portfolio"] = table["portfolio"].map(PORTFOLIO_LABELS)
    table["Skewness"] = table["skewness"].map(lambda value: f"{value:.2f}")
    table["Excess kurtosis"] = table["excess_kurtosis"].map(lambda value: f"{value:.2f}")
    table["Jarque-Bera p"] = table["jarque_bera_pvalue"].map(_format_pvalue)
    table["Shapiro-Wilk p"] = table["shapiro_wilk_pvalue"].map(_format_pvalue)
    table["Best fit"] = table["best_marginal_fit"]
    table["Student-t df"] = table["student_t_df"].map(lambda value: f"{value:.2f}")
    table["ADF p"] = table["adf_pvalue"].map(_format_pvalue)
    table["ARCH-LM p"] = table["arch_lm_pvalue"].map(_format_pvalue)
    return _markdown_table(
        table[
            [
                "Portfolio",
                "Skewness",
                "Excess kurtosis",
                "Jarque-Bera p",
                "Shapiro-Wilk p",
                "Best fit",
                "Student-t df",
                "ADF p",
                "ARCH-LM p",
            ]
        ]
    )


def _build_model_table(models: pd.DataFrame, garch: pd.DataFrame) -> str:
    merged = models.merge(
        garch[
            [
                "portfolio",
                "model_label",
                "distribution",
                "persistence",
                "std_resid_sq_lb_pvalue_lag_12",
                "std_resid_arch_lm_pvalue",
            ]
        ],
        on="portfolio",
    )
    merged["Portfolio"] = merged["portfolio"].map(PORTFOLIO_LABELS)
    merged["Selected ARIMA"] = merged["selected_arima_order"]
    merged["Selected volatility"] = merged["model_label"]
    merged["Innovation dist."] = merged["distribution"]
    merged["Residual Ljung-Box p (12)"] = merged["residual_lb_pvalue_lag_12"].map(_format_pvalue)
    merged["GARCH persistence"] = merged["persistence"].map(lambda value: f"{value:.3f}")
    merged["Std. sq. resid. LB p (12)"] = merged["std_resid_sq_lb_pvalue_lag_12"].map(_format_pvalue)
    merged["Std. resid. ARCH-LM p"] = merged["std_resid_arch_lm_pvalue"].map(_format_pvalue)
    return _markdown_table(
        merged[
            [
                "Portfolio",
                "Selected ARIMA",
                "Selected volatility",
                "Innovation dist.",
                "Residual Ljung-Box p (12)",
                "GARCH persistence",
                "Std. sq. resid. LB p (12)",
                "Std. resid. ARCH-LM p",
            ]
        ]
    )


def _build_predictive_table(predictive: pd.DataFrame) -> str:
    benchmark = predictive.loc[predictive["is_benchmark"]].copy()
    preferred = predictive.loc[predictive["is_preferred_predictive"]].copy()
    merged = preferred.merge(
        benchmark[["portfolio", "rmse"]].rename(columns={"rmse": "benchmark_rmse"}),
        on="portfolio",
        how="left",
    )
    merged["Portfolio"] = merged["portfolio"].map(PORTFOLIO_LABELS)
    merged["Preferred model"] = merged["model_label"].map(lambda value: SHORT_MODEL_LABELS.get(value, value))
    merged["Benchmark RMSE"] = merged["benchmark_rmse"].map(lambda value: f"{value:.3f}")
    merged["Predictive RMSE"] = merged["rmse"].map(lambda value: f"{value:.3f}")
    merged["RMSE gain vs. benchmark (%)"] = (
        ((merged["benchmark_rmse"] - merged["rmse"]) / merged["benchmark_rmse"]) * 100.0
    ).map(lambda value: f"{value:.2f}")
    merged["Directional accuracy"] = merged["directional_accuracy"].map(lambda value: f"{value:.3f}")
    return _markdown_table(
        merged[
            [
                "Portfolio",
                "Preferred model",
                "Benchmark RMSE",
                "Predictive RMSE",
                "RMSE gain vs. benchmark (%)",
                "Directional accuracy",
            ]
        ]
    )


def _build_joint_diagnostics_table(
    var_lags: pd.DataFrame,
    var_diag: pd.DataFrame,
    integration: pd.DataFrame,
    cointegration: pd.DataFrame,
) -> str:
    selected_var = var_lags.loc[var_lags["selected_bic"]].iloc[0]
    raw_stationary_share = (
        integration.loc[integration["representation"] == "raw_return_pct", "conclusion"].eq("stationary").mean()
    )
    wealth_nonstationary_share = (
        integration.loc[integration["representation"] == "log_wealth_level", "conclusion"]
        .eq("nonstationary")
        .mean()
    )
    diff_stationary_share = (
        integration.loc[integration["representation"] == "diff_log_wealth", "conclusion"].eq("stationary").mean()
    )
    trace_row = cointegration.loc[cointegration["rank_null"] == 0].iloc[0]
    diagnostics = {
        row["test"]: row for _, row in var_diag.iterrows()
    }
    table = pd.DataFrame(
        [
            {"Metric": "Selected VAR lag (BIC)", "Result": str(int(selected_var["lag"]))},
            {"Metric": "VAR stability", "Result": "Stable" if bool(selected_var["is_stable"]) else "Unstable"},
            {"Metric": "Whiteness test p-value", "Result": _format_pvalue(float(diagnostics["portmanteau_whiteness"]["pvalue"]))},
            {"Metric": "Normality test p-value", "Result": _format_pvalue(float(diagnostics["jarque_bera_normality"]["pvalue"]))},
            {"Metric": "Raw returns classified stationary", "Result": _format_pct(raw_stationary_share * 100.0, 1)},
            {"Metric": "Log wealth classified nonstationary", "Result": _format_pct(wealth_nonstationary_share * 100.0, 1)},
            {"Metric": "Diff. log wealth classified stationary", "Result": _format_pct(diff_stationary_share * 100.0, 1)},
            {"Metric": "Johansen trace statistic (rank 0)", "Result": _format_float(float(trace_row["trace_stat"]), 3)},
            {"Metric": "Johansen 5% critical value", "Result": _format_float(float(trace_row["crit_95"]), 3)},
            {"Metric": "Selected cointegration rank", "Result": str(int(trace_row["selected_rank"]))},
        ]
    )
    return _markdown_table(table)


def _build_strategy_table(strategy_metrics: pd.DataFrame) -> str:
    table = strategy_metrics.copy()
    table["Strategy"] = table["strategy_label"]
    table["Annualized net return (%)"] = table["annual_return_net"].map(lambda value: f"{value * 100.0:.2f}")
    table["Annualized net volatility (%)"] = table["annual_vol_net"].map(lambda value: f"{value * 100.0:.2f}")
    table["Net Sharpe"] = table["sharpe_net"].map(lambda value: f"{value:.3f}")
    table["Max drawdown (%)"] = table["max_drawdown_net"].map(lambda value: f"{value * 100.0:.2f}")
    table["Average monthly turnover"] = table["average_turnover"].map(lambda value: f"{value:.4f}")
    return _markdown_table(
        table[
            [
                "Strategy",
                "Annualized net return (%)",
                "Annualized net volatility (%)",
                "Net Sharpe",
                "Max drawdown (%)",
                "Average monthly turnover",
            ]
        ]
    )


def _compose_conclusions(strategy_metrics: pd.DataFrame, predictive: pd.DataFrame) -> str:
    preferred = predictive.loc[predictive["is_preferred_predictive"]].copy()
    benchmark = predictive.loc[predictive["is_benchmark"], ["portfolio", "rmse"]].rename(
        columns={"rmse": "benchmark_rmse"}
    )
    preferred = preferred.merge(benchmark, on="portfolio", how="left")
    preferred["rmse_gain"] = (preferred["benchmark_rmse"] - preferred["rmse"]) / preferred["benchmark_rmse"]
    improvements = int((preferred["rmse_gain"] > 0).sum())

    metrics_indexed = strategy_metrics.set_index("strategy")
    plugin = metrics_indexed.loc["mv_plugin_sample"]
    improved = metrics_indexed.loc["mv_improved_shrinkage"]
    equal_weight = metrics_indexed.loc["equal_weight"]
    stat_arb = metrics_indexed.loc["stat_arb_cointegration"]

    turnover_ratio = plugin["average_turnover"] / improved["average_turnover"]
    conclusion_text = f"""# Conclusions

This project turns a century of monthly Kenneth French portfolio returns into an integrated modeling and strategy exercise. The descriptive results show economically meaningful differences across the six portfolios, especially the higher mean and higher volatility of the small-value portfolio. The univariate modeling stage shows that monthly returns are stationary in levels but strongly non-normal, with pronounced tail behavior and persistent volatility. Low-order ARIMA models remain useful mean benchmarks, yet the stronger regularity in the data lies in second moments rather than in large and stable mean dynamics.

Adding exogenous predictors improves the forecasting design more than it improves accuracy. The preferred predictive specifications beat the univariate benchmark RMSE in only {improvements} of the 6 portfolios, with the best gain coming from Big HiBM. That result is still useful because it aligns with the literature: predictable variation in monthly returns exists, but it is modest and difficult to exploit consistently once the benchmark already captures low-order mean dynamics.

The trading analysis yields an intentionally mixed set of conclusions. The VAR(1) is stable and confirms strong joint dependence, but the residual diagnostics still reject whiteness and normality. Cointegration is meaningful only after moving from stationary returns to nonstationary log-wealth indices, and even then the statistical-arbitrage evidence is weak out of sample. The cointegration strategy delivers {_format_pct(stat_arb["annual_return_net"] * 100.0)} annualized net return with a net Sharpe of {_format_float(stat_arb["sharpe_net"])} after transaction costs, so the report treats it as an honest negative result rather than forcing a profitable narrative.

The strongest practical result comes from portfolio allocation. Relative to equal weight, the sample plug-in mean-variance strategy earns a higher net Sharpe ({_format_float(plugin["sharpe_net"])} versus {_format_float(equal_weight["sharpe_net"])}), but the improved shrinkage-and-controls variant is the more implementable design because it preserves a competitive net Sharpe of {_format_float(improved["sharpe_net"])} while reducing average turnover by about {_format_float(turnover_ratio, 1)}x. The main contribution of the project is therefore not a single dominant forecasting model or arbitrage rule, but a transparent end-to-end comparison showing which textbook ideas remain useful after diagnostics, out-of-sample testing, and transaction costs are taken seriously.
    """
    return textwrap.dedent(conclusion_text).strip() + "\n"


def _compose_appendices() -> str:
    appendix_text = APPENDIX_INDIVIDUAL_PATH.read_text(encoding="utf-8").strip()
    extra_appendix = f"""
# Appendices

## Appendix A. Portfolio-by-Portfolio Modeling Notes

{appendix_text.replace("# Appendix: Individual Portfolio Modeling Details", "").strip()}

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
& 'd:\\MG\\anaconda3\\python.exe' 'scripts\\run_pipeline.py'
```
"""
    return textwrap.dedent(extra_appendix).strip() + "\n"


def _compose_final_report(inputs: dict[str, pd.DataFrame], conclusions: str) -> str:
    summary = inputs["summary"]
    stats = inputs["stats"]
    models = inputs["models"]
    garch = inputs["garch"]
    predictive = inputs["predictive"]
    strategy_metrics = inputs["strategy_metrics"]
    var_lags = inputs["var_lags"]
    var_diag = inputs["var_diag"]
    integration = inputs["integration"]
    cointegration = inputs["cointegration"]
    stat_arb_signals = inputs["stat_arb_signals"]
    correlation = inputs["correlation"]

    small_value = summary.loc[summary["portfolio"] == "small_hibm_vwret_pct"].iloc[0]
    big_growth = summary.loc[summary["portfolio"] == "big_lobm_vwret_pct"].iloc[0]
    off_diagonal = correlation.where(~np.eye(len(correlation), dtype=bool))
    lowest_corr = off_diagonal.stack().min()
    highest_corr = off_diagonal.stack().max()
    strongest_non_normal = stats.loc[stats["student_t_aic_gain_vs_normal"].idxmax()]
    strongest_arch = stats.loc[stats["arch_lm_stat"].idxmax()]
    student_t_marginal_count = int((stats["best_marginal_fit"] == "Student-t").sum())
    student_t_garch_count = int((garch["distribution"] == "Student-t").sum())
    stable_var = var_lags.loc[var_lags["selected_bic"]].iloc[0]
    preferred = predictive.loc[predictive["is_preferred_predictive"]].copy()
    benchmark = predictive.loc[predictive["is_benchmark"], ["portfolio", "rmse"]].rename(
        columns={"rmse": "benchmark_rmse"}
    )
    preferred = preferred.merge(benchmark, on="portfolio", how="left")
    preferred["rmse_gain"] = (preferred["benchmark_rmse"] - preferred["rmse"]) / preferred["benchmark_rmse"]
    best_predictive_gain = preferred.loc[preferred["rmse_gain"].idxmax()]
    best_predictive_count = int((preferred["rmse_gain"] > 0).sum())
    metrics_indexed = strategy_metrics.set_index("strategy")
    plugin = metrics_indexed.loc["mv_plugin_sample"]
    improved = metrics_indexed.loc["mv_improved_shrinkage"]
    equal_weight = metrics_indexed.loc["equal_weight"]
    stat_arb = metrics_indexed.loc["stat_arb_cointegration"]
    rolling_rank_share = float((stat_arb_signals["rank"] > 0).mean())

    final_text = f"""
# Final Report

## Modeling and Trading Strategies for Six Kenneth French Portfolios

FMA4200 Final Project  
Prepared from the reproducible pipeline in `Final Project/` on 2026-04-12.

## 1. Introduction

This project studies the monthly returns of six value-weighted portfolios sorted on size and book-to-market, using the century-long Kenneth French sample provided for the course. These portfolios are a useful laboratory because they compress two core asset-pricing dimensions into a small panel that is still rich enough for univariate modeling, multivariate dependence analysis, and portfolio construction. The assignment is therefore not only about describing returns; it is about connecting descriptive evidence to forecasting, relative-value trading, and mean-variance allocation decisions.

The research background combines empirical asset pricing with time-series econometrics. Markowitz (1952) provides the benchmark portfolio-choice framework, while Fama and French (1993) show why size and value portfolios are economically meaningful objects rather than arbitrary return series. On the econometric side, Hamilton (1994) motivates ARIMA-class mean models; Engle (1982) and Bollerslev (1986) explain why volatility clustering can remain highly predictable even when the conditional mean is weak. For multivariate trading, Engle and Granger (1987) and Gatev, Goetzmann, and Rouwenhorst (2006) motivate looking for long-run relations and mean-reverting spreads. For implementation, Jagannathan and Ma (2003), Ledoit and Wolf (2004), and DeMiguel, Garlappi, and Uppal (2009) caution that unconstrained plug-in optimization can look stronger in sample than out of sample.

That literature implies five questions for the current dataset. First, are the six monthly return series close to Gaussian, or do they show the heavy tails and volatility clustering familiar from financial data? Second, do low-order ARIMA models capture the conditional mean adequately, or is residual structure left over? Third, can lagged exogenous predictors such as Fama-French factors and internally constructed spread signals improve forecasting? Fourth, do the six portfolios contain enough long-run common structure to support cointegration-based statistical arbitrage once a nonstationary representation is used? Fifth, how do equal-weight, textbook mean-variance, and improved shrinkage-based allocation rules compare after transaction costs?

The report's main contribution is to answer those questions within one reproducible workflow. The evidence ultimately favors a balanced conclusion rather than a single triumphant model. The data show strong common movement, clear non-normality, and persistent conditional volatility. Predictive gains in the conditional mean are present but modest, cointegration is statistically meaningful only on wealth indices and not consistently stable through time, and the most robust practical results come from rolling portfolio allocation rather than from statistical arbitrage.

## 2. Data Source and Processing

The raw input is the course-provided [Data.csv]({_relative_to_report(RAW_DATA_PATH)}), which matches the Kenneth French six-portfolio file formed on size and book-to-market using the 202601 CRSP database. The local raw file contains descriptive text above the monthly return block and footer text below it, so the cleaning stage programmatically locates the true header row before parsing the monthly value-weighted panel. Sentinel values `-99.99` and `-999` are converted to missing values during import rather than treated as genuine returns.

The return-unit convention is explicit because the Kenneth French data are reported in percent units. A raw value of `1.0866` therefore means a return of `1.0866%`, not `1.0866` in decimal form. The canonical cleaned file keeps the six portfolio series in percent units with `_pct` suffixes, while a companion decimal file divides each return by `100.0` for steps that require decimal arithmetic.

**Table 1. Cleaned variables and units.**

{_build_variable_table()}

The cleaned sample spans July 1926 through January 2026 and contains 1,195 monthly observations with no duplicate dates and no missing values after sentinel conversion. Figure 1 shows that all six return series move with the broad U.S. equity market but differ in amplitude across crises and recoveries. Figure 2 translates those return differences into long-run wealth paths, where the small-value portfolio ultimately earns the strongest growth path but with visibly larger drawdowns. Figure 3 reinforces the same point in cross-sectional form: the portfolios are highly correlated, yet not so collinear that diversification and multivariate modeling become meaningless.

![Monthly return overview](../output/figures/monthly_returns_overview.png)

*Figure 1. Monthly portfolio returns from July 1926 to January 2026.*

![Growth of one dollar](../output/figures/cumulative_growth_of_1.png)

*Figure 2. Growth of $1 invested in each portfolio using monthly decimal returns.*

![Correlation heatmap](../output/figures/portfolio_correlation_heatmap.png)

*Figure 3. Cross-portfolio correlation heatmap for the six monthly return series.*

**Table 2. Portfolio summary statistics in percent units.**

{_build_summary_table(summary)}

Table 2 shows two patterns that drive the rest of the report. The highest average return belongs to {PORTFOLIO_LABELS[small_value["portfolio"]]} at {_format_pct(small_value["mean_pct"])} per month, but it is also the most volatile at {_format_pct(small_value["std_pct"])} per month. At the other end, {PORTFOLIO_LABELS[big_growth["portfolio"]]} is much less volatile at {_format_pct(big_growth["std_pct"])} per month. The pairwise correlation range, from about {_format_float(lowest_corr)} to {_format_float(highest_corr)}, is high enough to motivate joint modeling and efficient-frontier analysis while still leaving room for relative-value spreads and diversification.

## 3. Modeling the Individual Portfolio Returns

### 3.1 Distributional Properties and Stationarity

Each portfolio was analyzed with time-series plots, histogram-density plots that overlay fitted Gaussian and Student-t densities, two-panel QQ plots, ACF/PACF, Jarque-Bera tests, Shapiro-Wilk tests, fitted-distribution comparisons, ADF and KPSS tests, Ljung-Box diagnostics, and ARCH-LM tests. Detailed figures are kept in the appendix and the `output/figures/individual_returns/` folders so the main body can focus on the highest-signal results. The central finding is that the six series are stationary in levels but far from Gaussian, with visible tail risk and volatility clustering even at the monthly frequency.

**Table 3. Distribution, normality, stationarity, and ARCH diagnostics.**

{_build_distribution_table(stats)}

Table 3 makes the distributional result hard to miss. Jarque-Bera and Shapiro-Wilk p-values are effectively zero across the panel, and the fitted Student-t distribution dominates the fitted Gaussian benchmark in {student_t_marginal_count} of the 6 portfolios. The most extreme non-normality appears in {PORTFOLIO_LABELS[strongest_non_normal["portfolio"]]}, where the Student-t fit gains the most ground on AIC and the excess-kurtosis evidence is especially strong. At the same time, ADF p-values strongly reject a unit root for every raw return series, which justifies modeling returns in levels rather than differencing them. ARCH-LM tests also reject homoskedasticity throughout the panel, with the strongest raw-volatility clustering in {PORTFOLIO_LABELS[strongest_arch["portfolio"]]}.

### 3.2 ARIMA Benchmarks and GARCH-Type Volatility Models

Given the stationarity evidence, the mean-model search compared low-order AR, MA, and ARIMA specifications using AIC, BIC, residual Ljung-Box tests, and parameter significance. The volatility step now uses the canonical `arch` package rather than a custom optimizer. For each portfolio, the selected ARIMA residuals were passed to Gaussian and Student-t GARCH(1,1) specifications, and the preferred volatility model was selected using AIC, BIC, standardized-residual diagnostics, and core-parameter significance.

**Table 4. Selected benchmark mean and volatility models.**

{_build_model_table(models, garch)}

The mean dynamics are modest. Five of the six portfolios select ARIMA(2,0,2), while Small LoBM selects ARIMA(0,0,2). Residual diagnostics are acceptable for the larger, lower-volatility portfolios but remain weaker for ME1 BM2 and Small HiBM, which is why the report does not overstate the quality of purely univariate mean models. The variance dynamics are more striking. GARCH persistence remains high across the six series, and the selected arch-based volatility filters largely remove the leftover second-moment dependence. The Student-t volatility specification is selected in {student_t_garch_count} of the 6 portfolios, which matches the heavy-tail evidence already visible in the marginal distribution diagnostics. That pattern aligns with the finance literature: monthly returns are hard to forecast in mean, but their volatility remains persistent and often better captured with non-Gaussian innovations.

### 3.3 Predictive Models with Exogenous Variables

The predictive extension combines lagged authoritative Fama-French factors with internally constructed signals such as lagged size and value spreads, 12-month rolling market volatility, 12-month momentum, and drawdown-style indicators. Three classes are compared portfolio by portfolio: the selected benchmark ARIMA model, an ARIMAX specification with lagged factors, and a predictive regression with lagged returns plus the broader predictor set. Performance is evaluated both in sample and with a 120-month expanding one-step-ahead forecast exercise using RMSE, MAE, and directional accuracy.

**Table 5. Best predictive model versus the univariate benchmark.**

{_build_predictive_table(predictive)}

The out-of-sample gains are real but limited. Only {best_predictive_count} of the 6 portfolios improve on the benchmark RMSE, and the strongest gain belongs to {PORTFOLIO_LABELS[best_predictive_gain["portfolio"]]}, where the preferred {SHORT_MODEL_LABELS.get(best_predictive_gain["model_label"], best_predictive_gain["model_label"]).lower()} improves RMSE by {_format_pct(best_predictive_gain["rmse_gain"] * 100.0)} relative to the benchmark. This is an economically sensible result rather than a disappointment. Monthly portfolio returns should not be expected to yield large and stable conditional-mean predictability, and the evidence here agrees with that view: exogenous signals help selectively, but they do not overturn the basic conclusion that volatility structure is more reliable than mean structure.

## 4. Trading Strategies

### 4.1 Joint Multivariate Dynamics and Cointegration Logic

The multivariate stage begins with a VAR fitted to the six return series. Lag lengths from 1 to 12 were compared with standard information criteria, and the BIC-selected benchmark is VAR({_format_float(float(stable_var["lag"]), 0)}). Stability holds because all inverse roots remain outside the unit circle, but the residual diagnostics remain imperfect: the whiteness and multivariate normality tests are both decisively rejected. Those rejections are consistent with the heavy tails and volatility clustering already documented in the univariate analysis, so they weaken any claim of fully adequate Gaussian linear dynamics without invalidating the VAR as a descriptive benchmark.

Cointegration requires a nonstationary representation, so it would be incorrect to apply Johansen tests mechanically to raw returns. The report instead tests integration order first, then moves to cumulative log wealth. The raw returns are stationary for all six portfolios, while the log-wealth levels are nonstationary and their first differences are stationary. That is exactly the environment in which Johansen analysis becomes conceptually defensible.

**Table 6. Joint dynamics and cointegration diagnostics.**

{_build_joint_diagnostics_table(var_lags, var_diag, integration, cointegration)}

The full-sample Johansen trace test selects rank 1, which suggests one long-run equilibrium relation among the six wealth indices. That result is statistically interesting, but the economic interpretation must remain cautious. In the rolling estimation windows used for trading, positive cointegration rank appears in only {_format_pct(rolling_rank_share * 100.0, 1)} of the monthly windows, so the long-run relation is episodic rather than permanently stable.

### 4.2 Statistical Arbitrage Backtest

The statistical-arbitrage strategy uses a 240-month rolling estimation window and refits the first cointegration relation every 12 months. It trades the standardized spread when the z-score exits a +/-1.5 band and closes positions once the z-score returns inside +/-0.5. Turnover is tracked directly, and transaction costs are set to 10 basis points per one-way turnover unit. This design avoids look-ahead bias because signals and weights at month t are built only from information available through month t-1.

The performance is weak after costs. The strategy earns {_format_pct(stat_arb["annual_return_net"] * 100.0)} annualized net return with annualized net volatility of {_format_pct(stat_arb["annual_vol_net"] * 100.0)}, a net Sharpe of {_format_float(stat_arb["sharpe_net"])}, and a max drawdown of {_format_pct(stat_arb["max_drawdown_net"] * 100.0)}. The negative result is important because it shows that a statistically significant full-sample cointegration relation is not enough on its own to support a robust trading rule once parameter instability and transaction costs are taken seriously.

### 4.3 Mean-Variance Allocation and Improved Plug-In Strategy

The portfolio-allocation comparison is stronger than the stat-arb exercise. The report first traces the efficient frontier using rolling sample moments, then backtests three implementable strategies over the common out-of-sample period: equal weight, a sample plug-in mean-variance strategy with full investment and no short sales, and an improved plug-in strategy that replaces the sample covariance matrix with Ledoit-Wolf shrinkage and adds weight bounds plus a turnover penalty.

![Efficient frontier](../output/figures/trading_strategies/efficient_frontier.png)

*Figure 4. Efficient frontier from sample mean and covariance estimates.*

![Strategy cumulative wealth](../output/figures/trading_strategies/strategy_cumulative_wealth.png)

*Figure 5. Cumulative net wealth for the trading strategies over the common out-of-sample window.*

**Table 7. Out-of-sample strategy comparison after transaction costs.**

{_build_strategy_table(strategy_metrics)}

Figure 4 shows that the return-covariance structure does allow higher expected-return portfolios along the frontier, but Figure 5 makes the implementation lesson clearer: the allocation strategies dominate the cointegration strategy over the realized sample. Table 7 shows that the plug-in mean-variance strategy achieves the highest net Sharpe at {_format_float(plugin["sharpe_net"])}, outperforming equal weight on both return and risk-adjusted performance. The improved shrinkage strategy does not quite match the plug-in Sharpe, but it remains competitive while cutting average turnover from {_format_float(plugin["average_turnover"], 4)} to {_format_float(improved["average_turnover"], 4)}. That tradeoff is economically attractive because the lower-turnover strategy is much closer to what a practical allocator would want to implement repeatedly.

## 5. Conclusions

{conclusions.replace("# Conclusions", "").strip()}

## References

{REFERENCES_PATH.read_text(encoding="utf-8").replace("# References", "").strip()}
"""
    return textwrap.dedent(final_text).strip() + "\n"


def _candidate_browser_paths() -> list[Path]:
    candidates = [
        Path(r"C:\Program Files\Google\Chrome\Application\chrome.exe"),
        Path(r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe"),
        Path(r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"),
        Path(r"C:\Program Files\Microsoft\Edge\Application\msedge.exe"),
    ]
    existing: list[Path] = []
    for path in candidates:
        if path.exists() and path not in existing:
            existing.append(path)
    return existing


def _try_browser_pdf_export() -> tuple[bool, str]:
    crash_dir = REPORT_DIR.parent / "logs" / "browser_crashes"
    profile_dir = REPORT_DIR.parent / "logs" / "browser_headless_profile"
    crash_dir.mkdir(parents=True, exist_ok=True)
    profile_dir.mkdir(parents=True, exist_ok=True)

    html_uri = FINAL_REPORT_HTML_PATH.resolve().as_uri()
    for browser in _candidate_browser_paths():
        command = [
            str(browser),
            "--headless=new",
            "--disable-gpu",
            "--disable-breakpad",
            "--no-first-run",
            "--no-default-browser-check",
            f"--crash-dumps-dir={crash_dir}",
            f"--user-data-dir={profile_dir}",
            "--allow-file-access-from-files",
            f"--print-to-pdf={FINAL_REPORT_PDF_PATH}",
            html_uri,
        ]
        try:
            completed = subprocess.run(command, check=True, capture_output=True, text=True, encoding="utf-8")
        except Exception as exc:
            last_error = f"{browser}: {exc}"
            continue

        if FINAL_REPORT_PDF_PATH.exists() and FINAL_REPORT_PDF_PATH.stat().st_size > 0:
            output = "\n".join(part for part in [completed.stdout.strip(), completed.stderr.strip()] if part).strip()
            message = f"Generated PDF with {browser}"
            if output:
                message = f"{message}. Browser output: {output}"
            return True, message
        last_error = f"{browser}: completed without creating PDF"

    return False, last_error if 'last_error' in locals() else "No supported browser executable was found."


def _write_export_notes(pdf_result: tuple[bool, str]) -> dict[str, str | bool]:
    tools_checked = {
        "pandoc": shutil.which("pandoc"),
        "xelatex": shutil.which("xelatex"),
        "pdflatex": shutil.which("pdflatex"),
        "wkhtmltopdf": shutil.which("wkhtmltopdf"),
        "chromium": shutil.which("chromium"),
        "chrome": shutil.which("chrome"),
        "msedge": shutil.which("msedge"),
    }
    browser_candidates = [str(path) for path in _candidate_browser_paths()]
    pdf_supported = any((tools_checked["xelatex"], tools_checked["pdflatex"])) and PDFExporter is not None
    webpdf_supported = any((tools_checked["chromium"], tools_checked["chrome"], tools_checked["msedge"])) and WebPDFExporter is not None
    browser_pdf_generated, browser_pdf_message = pdf_result
    pdf_exists = FINAL_REPORT_PDF_PATH.exists() and FINAL_REPORT_PDF_PATH.stat().st_size > 0

    if browser_pdf_generated:
        status = "Automatic PDF export succeeded through a locally installed Chromium-based browser."
    elif pdf_exists:
        status = "A previously generated PDF is available, but this run did not refresh it automatically."
    elif pdf_supported:
        status = "Potentially supported via nbconvert PDF exporter."
    elif webpdf_supported:
        status = "Potentially supported via nbconvert WebPDF exporter."
    else:
        status = "Automatic PDF export is not supported in the current environment."

    export_text = f"""# Export Notes

## Status

{status}

## Generated Files

- `report/final_report.md`
- `report/final_report.ipynb`
- `report/final_report.html`
- `report/final_appendices.md`
- `report/final_report.pdf`

## PDF Export Status

The environment check for this report build found the following PDF-related tools on PATH:

- `pandoc`: {tools_checked["pandoc"]}
- `xelatex`: {tools_checked["xelatex"]}
- `pdflatex`: {tools_checked["pdflatex"]}
- `wkhtmltopdf`: {tools_checked["wkhtmltopdf"]}
- `chromium`: {tools_checked["chromium"]}
- `chrome`: {tools_checked["chrome"]}
- `msedge`: {tools_checked["msedge"]}

The report builder also checked these installed browser paths outside PATH:

{chr(10).join(f"- `{candidate}`" for candidate in browser_candidates) if browser_candidates else "- None found"}

Browser export result: {browser_pdf_message}
Current PDF file present: {pdf_exists}

## Final Export Step

The builder already attempts to create `report/final_report.pdf` automatically. If a future rerun cannot do that, open `report/final_report.html` in a browser and use **Print** > **Save as PDF**.

The HTML file remains the fallback print-ready source.
"""
    EXPORT_NOTES_PATH.write_text(textwrap.dedent(export_text).strip() + "\n", encoding="utf-8")
    return {"pdf_supported": pdf_exists or browser_pdf_generated or pdf_supported or webpdf_supported, "status": status}


def build_final_report() -> dict[str, str]:
    inputs = _load_report_inputs()
    conclusions = _compose_conclusions(inputs["strategy_metrics"], inputs["predictive"])
    report_text = _compose_final_report(inputs, conclusions)
    appendices_text = _compose_appendices()

    FINAL_REPORT_MD_PATH.write_text(report_text, encoding="utf-8")
    FINAL_APPENDICES_PATH.write_text(appendices_text, encoding="utf-8")
    (REPORT_SECTIONS_DIR / "05_conclusions.md").write_text(conclusions, encoding="utf-8")

    notebook = nbformat.v4.new_notebook()
    notebook.cells = [nbformat.v4.new_markdown_cell(report_text)]
    nbformat.write(notebook, FINAL_REPORT_NOTEBOOK_PATH)

    html_exporter = HTMLExporter()
    html_body, _ = html_exporter.from_notebook_node(notebook)
    FINAL_REPORT_HTML_PATH.write_text(html_body, encoding="utf-8")

    pdf_result = _try_browser_pdf_export()
    export_status = _write_export_notes(pdf_result)

    log_text = f"""Final report build completed.
Markdown: {FINAL_REPORT_MD_PATH}
Notebook: {FINAL_REPORT_NOTEBOOK_PATH}
HTML: {FINAL_REPORT_HTML_PATH}
PDF path reserved: {FINAL_REPORT_PDF_PATH}
Appendices: {FINAL_APPENDICES_PATH}
Project status file: {PROJECT_STATUS_PATH}
Export status: {export_status["status"]}
PDF result: {pdf_result[1]}
"""
    FINAL_REPORT_LOG_PATH.write_text(textwrap.dedent(log_text).strip() + "\n", encoding="utf-8")

    return {
        "final_report_md": str(FINAL_REPORT_MD_PATH),
        "final_report_ipynb": str(FINAL_REPORT_NOTEBOOK_PATH),
        "final_report_html": str(FINAL_REPORT_HTML_PATH),
        "final_report_pdf": str(FINAL_REPORT_PDF_PATH),
        "final_appendices": str(FINAL_APPENDICES_PATH),
        "export_notes": str(EXPORT_NOTES_PATH),
        "pdf_supported": str(export_status["pdf_supported"]),
    }
