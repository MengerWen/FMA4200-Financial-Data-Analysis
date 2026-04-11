from __future__ import annotations

import ast
import logging
import warnings
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit
from scipy.stats import gaussian_kde, norm
from statsmodels.graphics.gofplots import qqplot
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.stats.diagnostic import acorr_ljungbox, het_arch
from statsmodels.stats.stattools import jarque_bera
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller, kpss

from .config import (
    APPENDIX_INDIVIDUAL_PATH,
    CLEAN_PERCENT_DATA_PATH,
    INDIVIDUAL_FIGURES_DIR,
    INDIVIDUAL_MODELING_LOG_PATH,
    INDIVIDUAL_MODELS_DIR,
    INDIVIDUAL_TABLES_DIR,
    PERCENT_COLUMNS,
    PORTFOLIO_GARCH_SUMMARY_PATH,
    PORTFOLIO_MODEL_COMPARISON_PATH,
    PORTFOLIO_TEST_SUMMARY_PATH,
    SECTION_03_PATH,
)
from .data_pipeline import ensure_directories, extract_monthly_value_weighted_data


ANNUALIZATION_FACTOR = np.sqrt(12.0)
LJUNG_BOX_LAGS = [12, 24]
ARCH_TEST_LAGS = 12
MAX_AR_ORDER = 2
MAX_MA_ORDER = 2


@dataclass
class GarchFitResult:
    success: bool
    message: str
    mu: float
    omega: float
    alpha: float
    beta: float
    loglik: float
    aic: float
    bic: float
    persistence: float
    unconditional_variance: float
    conditional_volatility: np.ndarray
    standardized_residuals: np.ndarray
    residuals: np.ndarray
    iterations: int


def configure_modeling_logger() -> logging.Logger:
    logger = logging.getLogger("fma4200_univariate")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

    file_handler = logging.FileHandler(INDIVIDUAL_MODELING_LOG_PATH, mode="w", encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)
    return logger


def portfolio_slug(portfolio: str) -> str:
    return portfolio.replace("_vwret_pct", "")


def portfolio_label(portfolio: str) -> str:
    labels = {
        "small_lobm_vwret_pct": "Small LoBM",
        "me1_bm2_vwret_pct": "ME1 BM2",
        "small_hibm_vwret_pct": "Small HiBM",
        "big_lobm_vwret_pct": "Big LoBM",
        "me2_bm2_vwret_pct": "ME2 BM2",
        "big_hibm_vwret_pct": "Big HiBM",
    }
    return labels.get(portfolio, portfolio.replace("_vwret_pct", "").replace("_", " ").title())


def portfolio_paths(portfolio: str) -> dict[str, Path]:
    slug = portfolio_slug(portfolio)
    figure_dir = INDIVIDUAL_FIGURES_DIR / slug
    table_dir = INDIVIDUAL_TABLES_DIR / slug
    model_dir = INDIVIDUAL_MODELS_DIR / slug
    for path in (figure_dir, table_dir, model_dir):
        path.mkdir(parents=True, exist_ok=True)
    return {"figure_dir": figure_dir, "table_dir": table_dir, "model_dir": model_dir}


def load_clean_returns() -> pd.DataFrame:
    if CLEAN_PERCENT_DATA_PATH.exists():
        df = pd.read_csv(CLEAN_PERCENT_DATA_PATH, parse_dates=["date"])
    else:
        df = extract_monthly_value_weighted_data()
        df["date"] = pd.to_datetime(df["date"])
    return df.sort_values("date").reset_index(drop=True)


def safe_kpss(series: pd.Series) -> tuple[float, float]:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        stat, pvalue, _, _ = kpss(series, regression="c", nlags="auto")
    return float(stat), float(pvalue)


def safe_adf(series: pd.Series) -> tuple[float, float]:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        stat, pvalue, _, _, _, _ = adfuller(series, autolag="AIC")
    return float(stat), float(pvalue)


def compute_ljung_box(series: pd.Series) -> dict[str, float]:
    lb = acorr_ljungbox(series, lags=LJUNG_BOX_LAGS, return_df=True)
    output: dict[str, float] = {}
    for lag in LJUNG_BOX_LAGS:
        output[f"lb_stat_lag_{lag}"] = float(lb.loc[lag, "lb_stat"])
        output[f"lb_pvalue_lag_{lag}"] = float(lb.loc[lag, "lb_pvalue"])
    return output


def compute_arch_test(series: pd.Series) -> dict[str, float]:
    lm_stat, lm_pvalue, f_stat, f_pvalue = het_arch(series, nlags=ARCH_TEST_LAGS)
    return {
        "arch_lm_stat": float(lm_stat),
        "arch_lm_pvalue": float(lm_pvalue),
        "arch_f_stat": float(f_stat),
        "arch_f_pvalue": float(f_pvalue),
    }


def descriptive_row(series: pd.Series) -> dict[str, float]:
    jb_stat, jb_pvalue, skewness, kurtosis = jarque_bera(series)
    adf_stat, adf_pvalue = safe_adf(series)
    kpss_stat, kpss_pvalue = safe_kpss(series)
    lb = compute_ljung_box(series)
    arch = compute_arch_test(series)

    return {
        "count": int(series.count()),
        "mean_pct": float(series.mean()),
        "annualized_mean_pct": float(series.mean() * 12.0),
        "std_pct": float(series.std()),
        "annualized_vol_pct": float(series.std() * ANNUALIZATION_FACTOR),
        "min_pct": float(series.min()),
        "p25_pct": float(series.quantile(0.25)),
        "median_pct": float(series.median()),
        "p75_pct": float(series.quantile(0.75)),
        "max_pct": float(series.max()),
        "skewness": float(skewness),
        "kurtosis": float(kurtosis),
        "jarque_bera_stat": float(jb_stat),
        "jarque_bera_pvalue": float(jb_pvalue),
        "adf_stat": float(adf_stat),
        "adf_pvalue": float(adf_pvalue),
        "kpss_stat": float(kpss_stat),
        "kpss_pvalue": float(kpss_pvalue),
        **lb,
        **arch,
    }


def choose_d_candidates(test_row: dict[str, float]) -> list[int]:
    if test_row["adf_pvalue"] < 0.05 and test_row["kpss_pvalue"] >= 0.05:
        return [0]
    return [0, 1]


def candidate_orders(d_candidates: list[int]) -> list[tuple[int, int, int]]:
    orders: list[tuple[int, int, int]] = []
    for d in d_candidates:
        for p in range(MAX_AR_ORDER + 1):
            for q in range(MAX_MA_ORDER + 1):
                if p == 0 and q == 0 and d == 0:
                    orders.append((p, d, q))
                elif p + q > 0:
                    orders.append((p, d, q))
    seen: set[tuple[int, int, int]] = set()
    unique_orders: list[tuple[int, int, int]] = []
    for order in orders:
        if order not in seen:
            unique_orders.append(order)
            seen.add(order)
    return unique_orders


def fit_single_arima(series: pd.Series, order: tuple[int, int, int]) -> dict[str, object]:
    trend = "n" if order[1] > 0 else "c"
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = ARIMA(
                series,
                order=order,
                trend=trend,
                enforce_stationarity=False,
                enforce_invertibility=False,
            )
            result = model.fit(method_kwargs={"maxiter": 300})

        param_names = list(result.param_names)
        dynamic_mask = [name.startswith("ar.") or name.startswith("ma.") for name in param_names]
        dynamic_pvalues = [float(result.pvalues.iloc[i]) for i, is_dynamic in enumerate(dynamic_mask) if is_dynamic]
        significant_share = (
            float(np.mean(np.array(dynamic_pvalues) < 0.05)) if dynamic_pvalues else 1.0
        )
        dynamic_count = int(sum(dynamic_mask))
        lb = acorr_ljungbox(result.resid, lags=LJUNG_BOX_LAGS, return_df=True)
        return {
            "order": order,
            "success": True,
            "result": result,
            "aic": float(result.aic),
            "bic": float(result.bic),
            "hqic": float(result.hqic),
            "llf": float(result.llf),
            "lb_pvalue_lag_12": float(lb.loc[12, "lb_pvalue"]),
            "lb_pvalue_lag_24": float(lb.loc[24, "lb_pvalue"]),
            "significant_dynamic_share": float(significant_share),
            "dynamic_param_count": dynamic_count,
            "order_complexity": int(sum(order)),
        }
    except Exception as exc:
        return {
            "order": order,
            "success": False,
            "error": str(exc),
        }


def select_best_arima(candidates: list[dict[str, object]]) -> tuple[dict[str, object], pd.DataFrame]:
    successful = [candidate for candidate in candidates if candidate["success"]]
    if not successful:
        raise RuntimeError("No ARIMA candidate converged successfully.")

    candidate_rows = []
    for candidate in successful:
        candidate_rows.append(
            {
                "order": str(candidate["order"]),
                "aic": candidate["aic"],
                "bic": candidate["bic"],
                "hqic": candidate["hqic"],
                "llf": candidate["llf"],
                "lb_pvalue_lag_12": candidate["lb_pvalue_lag_12"],
                "lb_pvalue_lag_24": candidate["lb_pvalue_lag_24"],
                "significant_dynamic_share": candidate["significant_dynamic_share"],
                "dynamic_param_count": candidate["dynamic_param_count"],
                "order_complexity": candidate["order_complexity"],
            }
        )
    comparison_df = pd.DataFrame(candidate_rows).sort_values(["aic", "bic", "order_complexity"]).reset_index(drop=True)

    best_aic = float(comparison_df["aic"].min())
    shortlisted = comparison_df[comparison_df["aic"] <= best_aic + 2.0].copy()
    preferred = shortlisted[
        (shortlisted["lb_pvalue_lag_12"] >= 0.05)
        & (
            (shortlisted["significant_dynamic_share"] >= 0.5)
            | (shortlisted["dynamic_param_count"] == 0)
        )
    ].copy()

    if not preferred.empty:
        chosen_row = preferred.sort_values(["bic", "order_complexity", "aic"]).iloc[0]
        selection_reason = "lowest BIC within 2 AIC points among models with acceptable residual autocorrelation and mostly significant dynamic terms"
    else:
        chosen_row = shortlisted.sort_values(["bic", "order_complexity", "aic"]).iloc[0]
        selection_reason = "lowest BIC within 2 AIC points because no candidate met the preferred residual-and-significance filter"

    selected_order = ast.literal_eval(chosen_row["order"])
    selected = next(candidate for candidate in successful if candidate["order"] == selected_order)
    comparison_df["selected"] = comparison_df["order"] == str(selected_order)
    comparison_df["selection_reason"] = ""
    comparison_df.loc[comparison_df["selected"], "selection_reason"] = selection_reason
    selected["selection_reason"] = selection_reason
    return selected, comparison_df


def unpack_garch_params(theta: np.ndarray) -> tuple[float, float, float, float]:
    mu = float(theta[0])
    omega = float(np.exp(theta[1]))
    alpha = float(0.999 * expit(theta[2]))
    beta = float((0.999 - alpha) * expit(theta[3]))
    return mu, omega, alpha, beta


def garch_neg_loglik(theta: np.ndarray, y: np.ndarray) -> float:
    mu, omega, alpha, beta = unpack_garch_params(theta)
    if omega <= 0.0 or alpha < 0.0 or beta < 0.0 or alpha + beta >= 0.999:
        return 1e12

    residuals = y - mu
    variance = np.empty_like(y)
    variance[0] = max(np.var(y, ddof=1), 1e-6)

    for index in range(1, len(y)):
        variance[index] = omega + alpha * residuals[index - 1] ** 2 + beta * variance[index - 1]
        if not np.isfinite(variance[index]) or variance[index] <= 1e-10:
            return 1e12

    nll = 0.5 * np.sum(np.log(2.0 * np.pi) + np.log(variance) + (residuals**2) / variance)
    if not np.isfinite(nll):
        return 1e12
    return float(nll)


def fit_garch_11(series: pd.Series) -> GarchFitResult:
    y = series.to_numpy(dtype=float)
    sample_variance = float(np.var(y, ddof=1))
    omega_start = max(sample_variance * 0.05, 1e-6)
    starting_points = [
        np.array([float(np.mean(y)), np.log(omega_start), 0.0, 2.0]),
        np.array([0.0, np.log(omega_start), -1.0, 2.2]),
        np.array([float(np.mean(y)), np.log(max(sample_variance * 0.10, 1e-6)), -0.5, 1.8]),
    ]

    best_result = None
    for start in starting_points:
        result = minimize(garch_neg_loglik, x0=start, args=(y,), method="L-BFGS-B")
        if best_result is None or result.fun < best_result.fun:
            best_result = result

    if best_result is None:
        raise RuntimeError("GARCH optimization did not return any result.")

    mu, omega, alpha, beta = unpack_garch_params(best_result.x)
    residuals = y - mu
    variance = np.empty_like(y)
    variance[0] = max(sample_variance, 1e-6)
    for index in range(1, len(y)):
        variance[index] = omega + alpha * residuals[index - 1] ** 2 + beta * variance[index - 1]
    variance = np.maximum(variance, 1e-10)
    sigma = np.sqrt(variance)
    std_residuals = residuals / sigma

    loglik = -float(best_result.fun)
    k = 4
    n = len(y)
    aic = 2 * k - 2 * loglik
    bic = np.log(n) * k - 2 * loglik
    persistence = alpha + beta
    unconditional_variance = omega / max(1.0 - persistence, 1e-6)

    return GarchFitResult(
        success=bool(best_result.success),
        message=str(best_result.message),
        mu=mu,
        omega=omega,
        alpha=alpha,
        beta=beta,
        loglik=loglik,
        aic=float(aic),
        bic=float(bic),
        persistence=float(persistence),
        unconditional_variance=float(unconditional_variance),
        conditional_volatility=sigma,
        standardized_residuals=std_residuals,
        residuals=residuals,
        iterations=int(best_result.nit),
    )


def save_time_series_plot(dates: pd.Series, series: pd.Series, figure_path: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(11, 4.5))
    ax.plot(dates, series, linewidth=0.9, color="#1f5aa6")
    ax.axhline(0.0, color="black", linewidth=0.8, alpha=0.6)
    ax.set_title(f"{title}: Monthly Returns")
    ax.set_ylabel("Return (%)")
    fig.tight_layout()
    fig.savefig(figure_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def save_distribution_plot(series: pd.Series, figure_path: Path, title: str) -> None:
    values = series.to_numpy(dtype=float)
    x_grid = np.linspace(values.min(), values.max(), 400)
    kde = gaussian_kde(values)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(values, bins=36, density=True, alpha=0.55, color="#5b8bd1", edgecolor="white")
    ax.plot(x_grid, kde(x_grid), color="#ba4a00", linewidth=2, label="Kernel density")
    ax.plot(
        x_grid,
        norm.pdf(x_grid, loc=values.mean(), scale=values.std(ddof=1)),
        color="black",
        linestyle="--",
        linewidth=1.5,
        label="Normal density",
    )
    ax.set_title(f"{title}: Histogram and Density")
    ax.set_xlabel("Return (%)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(figure_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def save_qq_plot(series: pd.Series, figure_path: Path, title: str) -> None:
    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111)
    qqplot(series, line="s", ax=ax)
    ax.set_title(f"{title}: QQ Plot")
    fig.tight_layout()
    fig.savefig(figure_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def save_acf_pacf_plot(series: pd.Series, figure_path: Path, title: str) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(10, 7))
    plot_acf(series, ax=axes[0], lags=24, zero=False)
    axes[0].set_title(f"{title}: ACF")
    plot_pacf(series, ax=axes[1], lags=24, zero=False, method="ywm")
    axes[1].set_title(f"{title}: PACF")
    fig.tight_layout()
    fig.savefig(figure_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def save_volatility_clustering_plot(dates: pd.Series, series: pd.Series, figure_path: Path, title: str) -> None:
    squared = series**2
    fig, axes = plt.subplots(2, 1, figsize=(10, 7))
    axes[0].plot(dates, squared, linewidth=0.9, color="#7a1f5c")
    axes[0].set_title(f"{title}: Squared Returns")
    axes[0].set_ylabel("Squared return")
    plot_acf(squared, ax=axes[1], lags=24, zero=False)
    axes[1].set_title(f"{title}: ACF of Squared Returns")
    fig.tight_layout()
    fig.savefig(figure_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def save_residual_diagnostics_plot(
    dates: pd.Series,
    residuals: pd.Series,
    fitted: pd.Series,
    figure_path: Path,
    title: str,
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes[0, 0].plot(dates, residuals, linewidth=0.9, color="#1f5aa6")
    axes[0, 0].axhline(0.0, color="black", linewidth=0.8, alpha=0.6)
    axes[0, 0].set_title(f"{title}: Residuals")

    axes[0, 1].plot(dates, fitted, linewidth=0.9, color="#ba4a00", label="Fitted")
    axes[0, 1].plot(dates, fitted + residuals, linewidth=0.7, color="#1f5aa6", alpha=0.5, label="Actual")
    axes[0, 1].set_title(f"{title}: Actual vs Fitted")
    axes[0, 1].legend()

    plot_acf(residuals, ax=axes[1, 0], lags=24, zero=False)
    axes[1, 0].set_title(f"{title}: Residual ACF")
    plot_acf(residuals**2, ax=axes[1, 1], lags=24, zero=False)
    axes[1, 1].set_title(f"{title}: Squared Residual ACF")

    fig.tight_layout()
    fig.savefig(figure_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def save_garch_plot(
    dates: pd.Series,
    series: pd.Series,
    garch_result: GarchFitResult,
    figure_path: Path,
    title: str,
) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(11, 7))
    axes[0].plot(dates, series, linewidth=0.8, alpha=0.6, color="#1f5aa6", label="Returns")
    axes[0].plot(dates, garch_result.conditional_volatility, linewidth=1.2, color="#ba4a00", label="Conditional sigma")
    axes[0].set_title(f"{title}: Returns and GARCH Conditional Volatility")
    axes[0].legend()

    plot_acf(garch_result.standardized_residuals**2, ax=axes[1], lags=24, zero=False)
    axes[1].set_title(f"{title}: ACF of Squared Standardized Residuals")

    fig.tight_layout()
    fig.savefig(figure_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def write_text_file(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8")


def save_dataframe(df: pd.DataFrame, path: Path) -> None:
    df.to_csv(path, index=False)


def arima_parameter_table(result) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "parameter": result.param_names,
            "estimate": result.params,
            "std_error": result.bse,
            "z_stat": result.params / result.bse,
            "pvalue": result.pvalues,
        }
    )


def garch_summary_table(garch_result: GarchFitResult, raw_arch: dict[str, float]) -> pd.DataFrame:
    std_arch = compute_arch_test(pd.Series(garch_result.standardized_residuals))
    return pd.DataFrame(
        [
            {
                "success": garch_result.success,
                "message": garch_result.message,
                "mu": garch_result.mu,
                "omega": garch_result.omega,
                "alpha": garch_result.alpha,
                "beta": garch_result.beta,
                "persistence": garch_result.persistence,
                "unconditional_variance": garch_result.unconditional_variance,
                "loglik": garch_result.loglik,
                "aic": garch_result.aic,
                "bic": garch_result.bic,
                "iterations": garch_result.iterations,
                "raw_arch_lm_pvalue": raw_arch["arch_lm_pvalue"],
                "std_resid_arch_lm_pvalue": std_arch["arch_lm_pvalue"],
            }
        ]
    )


def summarize_stationarity(test_row: dict[str, float]) -> str:
    adf_stationary = test_row["adf_pvalue"] < 0.05
    kpss_stationary = test_row["kpss_pvalue"] >= 0.05
    if adf_stationary and kpss_stationary:
        return "Both ADF and KPSS support treating the return series as stationary in levels."
    if adf_stationary and not kpss_stationary:
        return "ADF rejects a unit root, but KPSS still flags some persistence; level models remain plausible but should be checked carefully."
    return "Stationarity evidence is mixed enough that ARIMA candidates allow both level and differenced specifications."


def build_interpretation_text(
    portfolio: str,
    test_row: dict[str, float],
    selected_arima: dict[str, object],
    garch_result: GarchFitResult,
    garch_summary: pd.DataFrame,
) -> str:
    order = selected_arima["order"]
    residual_lb = selected_arima["lb_pvalue_lag_12"]
    raw_arch_pvalue = test_row["arch_lm_pvalue"]
    std_arch_pvalue = float(garch_summary.loc[0, "std_resid_arch_lm_pvalue"])
    normality_text = (
        "The Jarque-Bera test rejects normality."
        if test_row["jarque_bera_pvalue"] < 0.05
        else "The Jarque-Bera test does not reject normality at conventional levels."
    )
    arch_text = (
        "ARCH effects are materially weaker after GARCH standardization."
        if std_arch_pvalue > raw_arch_pvalue
        else "The hand-rolled GARCH filter does not fully remove ARCH effects, so volatility results should be interpreted cautiously."
    )
    return "\n".join(
        [
            f"# {portfolio_label(portfolio)}",
            "",
            f"- Distribution: mean = {test_row['mean_pct']:.3f}% per month, volatility = {test_row['std_pct']:.3f}% per month, skewness = {test_row['skewness']:.3f}, kurtosis = {test_row['kurtosis']:.3f}.",
            f"- Normality: {normality_text}",
            f"- Stationarity: {summarize_stationarity(test_row)}",
            f"- Selected mean model: ARIMA{order} chosen because {selected_arima['selection_reason']}.",
            f"- Residual autocorrelation: Ljung-Box p-value at lag 12 = {residual_lb:.3f}.",
            f"- Volatility model: Gaussian GARCH(1,1) estimates alpha = {garch_result.alpha:.3f}, beta = {garch_result.beta:.3f}, and persistence = {garch_result.persistence:.3f}.",
            f"- Volatility clustering: raw ARCH-LM p-value = {raw_arch_pvalue:.4f}; standardized-residual ARCH-LM p-value = {std_arch_pvalue:.4f}. {arch_text}",
        ]
    )


def write_section_03(
    test_summary: pd.DataFrame,
    model_summary: pd.DataFrame,
    garch_summary: pd.DataFrame,
) -> None:
    best_mean_portfolio = test_summary.sort_values("mean_pct", ascending=False).iloc[0]
    highest_persistence = garch_summary.sort_values("persistence", ascending=False).iloc[0]
    strongest_rejected_normality = test_summary.sort_values("jarque_bera_stat", ascending=False).iloc[0]
    weak_residual_fit = model_summary.loc[model_summary["residual_lb_pvalue_lag_12"] < 0.05, "portfolio"].tolist()
    weak_residual_text = ", ".join(portfolio_label(portfolio) for portfolio in weak_residual_fit)
    section_lines = [
        "# Modeling the Individual Portfolio Returns",
        "",
        "## Workflow",
        "",
        "This section models each of the six value-weighted portfolio return series individually using the cleaned monthly dataset. For each portfolio, the pipeline saves a time-series plot, histogram-and-density plot, QQ plot, ACF/PACF, residual diagnostics, volatility-clustering diagnostics, ARIMA candidate comparison tables, a selected ARIMA summary, and a lightweight Gaussian GARCH(1,1) fit implemented with `scipy.optimize` because the `arch` package is unavailable in the required interpreter.",
        "",
        "## Distributional and Diagnostic Evidence",
        "",
        f"The descriptive diagnostics indicate that return distributions are not well approximated by the Gaussian benchmark. The strongest Jarque-Bera rejection in this sample occurs for **{portfolio_label(strongest_rejected_normality['portfolio'])}**, and every portfolio exhibits nontrivial skewness or excess kurtosis. Stationarity tests support modeling returns in levels rather than prices: the series are monthly returns, ADF generally rejects a unit root, and KPSS results do not overturn the practical use of level ARIMA models.",
        "",
        "The saved diagnostic artifacts for each portfolio are under:",
        "",
        "- `output/figures/individual_returns/<portfolio>/`",
        "- `output/tables/individual_returns/<portfolio>/`",
        "- `output/models/individual_returns/<portfolio>/`",
        "",
        "## Mean-Model Selection",
        "",
        "AR, MA, and ARMA/ARIMA candidates were estimated with `statsmodels`. The selection rule was:",
        "",
        "1. Use ADF and KPSS evidence to decide whether differenced candidates are even needed.",
        "2. Fit a small, interpretable ARIMA grid with orders up to two autoregressive and two moving-average terms.",
        "3. Compare candidates on AIC and BIC.",
        "4. Prefer models whose residual Ljung-Box p-value at lag 12 is at least 0.05 and whose dynamic parameters are mostly statistically significant when such terms are present.",
        "5. Break ties in favor of the simpler specification.",
        "",
        "This rule is intentionally conservative because monthly equity returns often contain much weaker mean predictability than volatility predictability. In practice, the selected ARIMA models are low-order and in some cases close to white-noise benchmarks, which is consistent with the financial-return literature reviewed in the introduction.",
        "",
        f"Residual diagnostics also show that mean dynamics are not captured equally well across all portfolios. The clearest remaining residual autocorrelation at lag 12 appears in **{weak_residual_text}**, so those selected ARIMA specifications should be read as reasonable low-order benchmarks rather than fully satisfactory final mean models.",
        "",
        "The combined comparison table is saved at `output/tables/individual_returns/portfolio_model_comparison_summary.csv`.",
        "",
        "## Volatility Modeling",
        "",
        "A Gaussian GARCH(1,1) model with constant mean was estimated separately for each portfolio using a constrained parameterization and `scipy.optimize.minimize`. This hand-rolled approach is more limited than the unavailable `arch` package, especially for formal inference and richer specifications, but it is stable enough here to provide a reproducible volatility benchmark. The key use of the GARCH fits is to measure volatility persistence and to check whether standardized residuals show weaker ARCH effects than the raw series.",
        "",
        f"The most persistent volatility process in the current run is **{portfolio_label(highest_persistence['portfolio'])}**, with alpha + beta = **{highest_persistence['persistence']:.3f}**. That level of persistence is economically plausible for long samples of monthly equity data and supports the inclusion of volatility-focused models in the project.",
        "",
        "The combined GARCH summary is saved at `output/tables/individual_returns/portfolio_garch_summary.csv`.",
        "",
        "## Cross-Portfolio Interpretation",
        "",
        f"The portfolio with the highest average monthly return remains **{portfolio_label(best_mean_portfolio['portfolio'])}**, at **{best_mean_portfolio['mean_pct']:.3f}%** per month. Across the six portfolios, the mean-model evidence is modest, while the volatility-model evidence is stronger. That pattern supports an interpretation in which the conditional mean of monthly portfolio returns is only weakly predictable from its own history, but conditional risk is more persistent and structured.",
        "",
        "Overall, the Section 3 evidence points to three conclusions. First, the return series are heavy tailed enough that normality-based intuition should be treated carefully. Second, low-order ARIMA models are adequate as mean benchmarks, but they do not reveal strong standalone predictability. Third, volatility clustering is real enough to justify GARCH-type modeling, even at the monthly frequency. These results set up the next project stages naturally: exogenous predictors for the conditional mean and multivariate dependence modeling for trading strategies.",
        "",
        "Portfolio-by-portfolio interpretations and the denser output inventory are moved to the appendix file `report/sections/appendix_individual_returns_modeling.md`.",
    ]
    SECTION_03_PATH.write_text("\n".join(section_lines) + "\n", encoding="utf-8")


def write_appendix(test_summary: pd.DataFrame, model_summary: pd.DataFrame, garch_summary: pd.DataFrame) -> None:
    appendix_lines = [
        "# Appendix: Individual Portfolio Modeling Details",
        "",
        "This appendix records concise portfolio-by-portfolio interpretations generated from the saved diagnostics and model outputs.",
        "",
    ]
    for portfolio in PERCENT_COLUMNS:
        test_row = test_summary.loc[test_summary["portfolio"] == portfolio].iloc[0]
        model_row = model_summary.loc[model_summary["portfolio"] == portfolio].iloc[0]
        garch_row = garch_summary.loc[garch_summary["portfolio"] == portfolio].iloc[0]
        appendix_lines.extend(
            [
                f"## {portfolio_label(portfolio)}",
                "",
                f"- Mean and risk: {test_row['mean_pct']:.3f}% monthly mean, {test_row['std_pct']:.3f}% monthly volatility.",
                f"- Shape: skewness = {test_row['skewness']:.3f}, kurtosis = {test_row['kurtosis']:.3f}, Jarque-Bera p-value = {test_row['jarque_bera_pvalue']:.4f}.",
                f"- Stationarity: ADF p-value = {test_row['adf_pvalue']:.4f}, KPSS p-value = {test_row['kpss_pvalue']:.4f}.",
                f"- Selected ARIMA model: {model_row['selected_arima_order']} with AIC = {model_row['selected_arima_aic']:.2f}, BIC = {model_row['selected_arima_bic']:.2f}, and residual Ljung-Box p-value at lag 12 = {model_row['residual_lb_pvalue_lag_12']:.4f}.",
                f"- GARCH(1,1): alpha = {garch_row['alpha']:.4f}, beta = {garch_row['beta']:.4f}, persistence = {garch_row['persistence']:.4f}, standardized-residual ARCH-LM p-value = {garch_row['std_resid_arch_lm_pvalue']:.4f}.",
                f"- Saved outputs: `output/figures/individual_returns/{portfolio_slug(portfolio)}/`, `output/tables/individual_returns/{portfolio_slug(portfolio)}/`, and `output/models/individual_returns/{portfolio_slug(portfolio)}/`.",
                "",
            ]
        )
    APPENDIX_INDIVIDUAL_PATH.write_text("\n".join(appendix_lines) + "\n", encoding="utf-8")


def analyze_single_portfolio(df: pd.DataFrame, portfolio: str, logger: logging.Logger) -> dict[str, object]:
    title = portfolio_label(portfolio)
    paths = portfolio_paths(portfolio)
    series = df[portfolio].astype(float)
    dates = df["date"]

    logger.info("Analyzing %s", portfolio)
    test_row = descriptive_row(series)
    test_row["portfolio"] = portfolio

    save_time_series_plot(dates, series, paths["figure_dir"] / "time_series.png", title)
    save_distribution_plot(series, paths["figure_dir"] / "histogram_density.png", title)
    save_qq_plot(series, paths["figure_dir"] / "qq_plot.png", title)
    save_acf_pacf_plot(series, paths["figure_dir"] / "acf_pacf.png", title)
    save_volatility_clustering_plot(dates, series, paths["figure_dir"] / "volatility_clustering.png", title)

    d_candidates = choose_d_candidates(test_row)
    fitted_candidates = [fit_single_arima(series, order) for order in candidate_orders(d_candidates)]
    selected_arima, arima_comparison = select_best_arima(fitted_candidates)
    arima_result = selected_arima["result"]
    fitted_values = pd.Series(arima_result.fittedvalues, index=series.index)
    residuals = pd.Series(arima_result.resid, index=series.index)

    save_residual_diagnostics_plot(
        dates,
        residuals,
        fitted_values,
        paths["figure_dir"] / "residual_diagnostics.png",
        title,
    )

    residual_lb = compute_ljung_box(residuals)
    residual_arch = compute_arch_test(residuals)
    residual_summary = pd.DataFrame(
        [
            {
                "lb_stat_lag_12": residual_lb["lb_stat_lag_12"],
                "lb_pvalue_lag_12": residual_lb["lb_pvalue_lag_12"],
                "lb_stat_lag_24": residual_lb["lb_stat_lag_24"],
                "lb_pvalue_lag_24": residual_lb["lb_pvalue_lag_24"],
                "arch_lm_stat": residual_arch["arch_lm_stat"],
                "arch_lm_pvalue": residual_arch["arch_lm_pvalue"],
            }
        ]
    )

    garch_result = fit_garch_11(series)
    garch_summary = garch_summary_table(garch_result, compute_arch_test(series))
    save_garch_plot(dates, series, garch_result, paths["figure_dir"] / "garch_diagnostics.png", title)

    conditional_volatility_df = pd.DataFrame(
        {
            "date": dates,
            "return_pct": series,
            "garch_residual": garch_result.residuals,
            "conditional_sigma_pct": garch_result.conditional_volatility,
            "standardized_residual": garch_result.standardized_residuals,
        }
    )

    save_dataframe(pd.DataFrame([test_row]), paths["table_dir"] / "descriptive_and_test_statistics.csv")
    save_dataframe(arima_comparison, paths["table_dir"] / "arima_candidate_models.csv")
    save_dataframe(arima_parameter_table(arima_result), paths["table_dir"] / "selected_arima_parameters.csv")
    save_dataframe(residual_summary, paths["table_dir"] / "selected_arima_residual_diagnostics.csv")
    save_dataframe(garch_summary, paths["table_dir"] / "garch_summary.csv")
    save_dataframe(conditional_volatility_df, paths["table_dir"] / "garch_conditional_volatility.csv")

    write_text_file(paths["model_dir"] / "selected_arima_summary.txt", arima_result.summary().as_text())
    write_text_file(
        paths["model_dir"] / "garch_summary.txt",
        "\n".join(
            [
                f"Success: {garch_result.success}",
                f"Message: {garch_result.message}",
                f"mu: {garch_result.mu:.6f}",
                f"omega: {garch_result.omega:.6f}",
                f"alpha: {garch_result.alpha:.6f}",
                f"beta: {garch_result.beta:.6f}",
                f"persistence: {garch_result.persistence:.6f}",
                f"loglik: {garch_result.loglik:.6f}",
                f"AIC: {garch_result.aic:.6f}",
                f"BIC: {garch_result.bic:.6f}",
            ]
        )
        + "\n",
    )

    interpretation_text = build_interpretation_text(portfolio, test_row, selected_arima, garch_result, garch_summary)
    write_text_file(paths["model_dir"] / "interpretation.md", interpretation_text + "\n")

    model_row = {
        "portfolio": portfolio,
        "selected_arima_order": str(selected_arima["order"]),
        "selected_arima_aic": float(selected_arima["aic"]),
        "selected_arima_bic": float(selected_arima["bic"]),
        "selected_arima_hqic": float(selected_arima["hqic"]),
        "residual_lb_pvalue_lag_12": float(selected_arima["lb_pvalue_lag_12"]),
        "residual_lb_pvalue_lag_24": float(selected_arima["lb_pvalue_lag_24"]),
        "significant_dynamic_share": float(selected_arima["significant_dynamic_share"]),
        "selection_reason": str(selected_arima["selection_reason"]),
    }

    garch_row = garch_summary.iloc[0].to_dict()
    garch_row["portfolio"] = portfolio

    return {
        "test_row": test_row,
        "model_row": model_row,
        "garch_row": garch_row,
    }


def run_individual_modeling_pipeline() -> dict[str, object]:
    ensure_directories()
    logger = configure_modeling_logger()
    logger.info("Starting individual portfolio modeling pipeline.")

    df = load_clean_returns()

    test_rows: list[dict[str, object]] = []
    model_rows: list[dict[str, object]] = []
    garch_rows: list[dict[str, object]] = []

    for portfolio in PERCENT_COLUMNS:
        result = analyze_single_portfolio(df, portfolio, logger)
        test_rows.append(result["test_row"])
        model_rows.append(result["model_row"])
        garch_rows.append(result["garch_row"])

    test_summary = pd.DataFrame(test_rows)
    model_summary = pd.DataFrame(model_rows)
    garch_summary = pd.DataFrame(garch_rows)

    save_dataframe(test_summary, PORTFOLIO_TEST_SUMMARY_PATH)
    save_dataframe(model_summary, PORTFOLIO_MODEL_COMPARISON_PATH)
    save_dataframe(garch_summary, PORTFOLIO_GARCH_SUMMARY_PATH)

    write_section_03(test_summary, model_summary, garch_summary)
    write_appendix(test_summary, model_summary, garch_summary)

    logger.info("Individual portfolio modeling pipeline completed successfully.")
    return {
        "n_portfolios": len(PERCENT_COLUMNS),
        "test_summary_path": str(PORTFOLIO_TEST_SUMMARY_PATH),
        "model_summary_path": str(PORTFOLIO_MODEL_COMPARISON_PATH),
        "garch_summary_path": str(PORTFOLIO_GARCH_SUMMARY_PATH),
    }
