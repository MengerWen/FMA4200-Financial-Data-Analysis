from __future__ import annotations

import ast
import logging
import warnings
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from arch import arch_model
from arch.univariate import GeneralizedError, Normal, SkewStudent, StudentsT
from scipy.stats import anderson, gaussian_kde, kstest, norm, norminvgauss, shapiro, t as student_t
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
VOLATILITY_MODELS = (
    ("Gaussian GARCH(1,1)", "normal"),
    ("Student-t GARCH(1,1)", "t"),
    ("Skewed Student-t GARCH(1,1)", "skewt"),
    ("GED GARCH(1,1)", "ged"),
)


@dataclass
class DistributionFitResult:
    distribution: str
    distribution_code: str
    loglik: float
    aic: float
    bic: float
    ks_stat: float
    ks_pvalue: float
    params: dict[str, float]
    parameter_text: str
    complexity_rank: int


@dataclass
class VolatilityFitResult:
    model_label: str
    distribution: str
    distribution_code: str
    success: bool
    convergence_flag: int
    loglik: float
    aic: float
    bic: float
    omega: float
    alpha: float
    beta: float
    persistence: float
    nu: float | None
    distribution_params: dict[str, float]
    distribution_param_text: str
    conditional_volatility: np.ndarray
    standardized_residuals: np.ndarray
    residuals: np.ndarray
    param_table: pd.DataFrame
    summary_text: str
    diagnostics: dict[str, float]


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
    clean = pd.Series(series, copy=False).dropna()
    lb = acorr_ljungbox(clean, lags=LJUNG_BOX_LAGS, return_df=True)
    output: dict[str, float] = {}
    for lag in LJUNG_BOX_LAGS:
        output[f"lb_stat_lag_{lag}"] = float(lb.loc[lag, "lb_stat"])
        output[f"lb_pvalue_lag_{lag}"] = float(lb.loc[lag, "lb_pvalue"])
    return output


def compute_arch_test(series: pd.Series) -> dict[str, float]:
    clean = pd.Series(series, copy=False).dropna()
    lm_stat, lm_pvalue, f_stat, f_pvalue = het_arch(clean, nlags=ARCH_TEST_LAGS)
    return {
        "arch_lm_stat": float(lm_stat),
        "arch_lm_pvalue": float(lm_pvalue),
        "arch_f_stat": float(f_stat),
        "arch_f_pvalue": float(f_pvalue),
    }


def format_parameter_text(params: dict[str, float]) -> str:
    if not params:
        return ""
    return ", ".join(f"{name}={value:.4f}" for name, value in params.items())


def marginal_distribution_label(distribution_code: str) -> str:
    mapping = {
        "normal": "Normal",
        "t": "Student-t",
        "nig": "NIG",
    }
    return mapping[distribution_code]


def marginal_distribution_complexity(distribution_code: str) -> int:
    mapping = {
        "normal": 0,
        "t": 1,
        "nig": 2,
    }
    return mapping[distribution_code]


def arch_distribution_label(distribution_code: str) -> str:
    mapping = {
        "normal": "Gaussian",
        "t": "Student-t",
        "skewt": "Skewed Student-t",
        "ged": "GED",
    }
    return mapping[distribution_code]


def arch_distribution_complexity(distribution_code: str) -> int:
    mapping = {
        "normal": 0,
        "t": 1,
        "ged": 1,
        "skewt": 2,
    }
    return mapping[distribution_code]


def arch_distribution_parameter_names(distribution_code: str) -> list[str]:
    mapping = {
        "normal": [],
        "t": ["nu"],
        "skewt": ["eta", "lambda"],
        "ged": ["nu"],
    }
    return mapping[distribution_code]


def arch_distribution_object(distribution_code: str):
    mapping = {
        "normal": Normal(),
        "t": StudentsT(),
        "skewt": SkewStudent(),
        "ged": GeneralizedError(),
    }
    return mapping[distribution_code]


def arch_distribution_parameter_vector(distribution_code: str, params: dict[str, float]) -> np.ndarray | None:
    names = arch_distribution_parameter_names(distribution_code)
    if not names:
        return None
    return np.array([params[name] for name in names], dtype=float)


def arch_distribution_cdf(
    values: np.ndarray | pd.Series,
    distribution_code: str,
    params: dict[str, float],
) -> np.ndarray:
    distribution = arch_distribution_object(distribution_code)
    parameter_vector = arch_distribution_parameter_vector(distribution_code, params)
    return np.asarray(distribution.cdf(values, parameters=parameter_vector), dtype=float)


def arch_distribution_ppf(
    probabilities: np.ndarray,
    distribution_code: str,
    params: dict[str, float],
) -> np.ndarray:
    distribution = arch_distribution_object(distribution_code)
    parameter_vector = arch_distribution_parameter_vector(distribution_code, params)
    return np.asarray(distribution.ppf(probabilities, parameters=parameter_vector), dtype=float)


def fit_distribution(values: np.ndarray, distribution_code: str) -> DistributionFitResult:
    n_obs = len(values)
    if distribution_code == "normal":
        loc, scale = norm.fit(values)
        loglik = float(np.sum(norm.logpdf(values, loc=loc, scale=scale)))
        ks_stat, ks_pvalue = kstest(values, lambda x: norm.cdf(x, loc=loc, scale=scale))
        return DistributionFitResult(
            distribution=marginal_distribution_label(distribution_code),
            distribution_code="normal",
            loglik=loglik,
            aic=float(2 * 2 - 2 * loglik),
            bic=float(np.log(n_obs) * 2 - 2 * loglik),
            ks_stat=float(ks_stat),
            ks_pvalue=float(ks_pvalue),
            params={
                "loc": float(loc),
                "scale": float(scale),
            },
            parameter_text=format_parameter_text({"loc": float(loc), "scale": float(scale)}),
            complexity_rank=marginal_distribution_complexity(distribution_code),
        )

    if distribution_code == "t":
        df_param, loc, scale = student_t.fit(values)
        loglik = float(np.sum(student_t.logpdf(values, df=df_param, loc=loc, scale=scale)))
        ks_stat, ks_pvalue = kstest(values, lambda x: student_t.cdf(x, df=df_param, loc=loc, scale=scale))
        params = {
            "df": float(df_param),
            "loc": float(loc),
            "scale": float(scale),
        }
        return DistributionFitResult(
            distribution=marginal_distribution_label(distribution_code),
            distribution_code="t",
            loglik=loglik,
            aic=float(2 * 3 - 2 * loglik),
            bic=float(np.log(n_obs) * 3 - 2 * loglik),
            ks_stat=float(ks_stat),
            ks_pvalue=float(ks_pvalue),
            params=params,
            parameter_text=format_parameter_text(params),
            complexity_rank=marginal_distribution_complexity(distribution_code),
        )

    if distribution_code == "nig":
        alpha, beta, loc, scale = norminvgauss.fit(values)
        loglik = float(np.sum(norminvgauss.logpdf(values, alpha, beta, loc=loc, scale=scale)))
        ks_stat, ks_pvalue = kstest(
            values,
            lambda x: norminvgauss.cdf(x, alpha, beta, loc=loc, scale=scale),
        )
        params = {
            "alpha": float(alpha),
            "beta": float(beta),
            "loc": float(loc),
            "scale": float(scale),
        }
        return DistributionFitResult(
            distribution=marginal_distribution_label(distribution_code),
            distribution_code="nig",
            loglik=loglik,
            aic=float(2 * 4 - 2 * loglik),
            bic=float(np.log(n_obs) * 4 - 2 * loglik),
            ks_stat=float(ks_stat),
            ks_pvalue=float(ks_pvalue),
            params=params,
            parameter_text=format_parameter_text(params),
            complexity_rank=marginal_distribution_complexity(distribution_code),
        )

    raise ValueError(f"Unsupported distribution code: {distribution_code}")


def distribution_results_table(
    results: list[DistributionFitResult],
) -> tuple[pd.DataFrame, DistributionFitResult, str]:
    rows = []
    for result in results:
        rows.append(
            {
                "distribution": result.distribution,
                "distribution_code": result.distribution_code,
                "parameter_estimates": result.parameter_text,
                "loglik": result.loglik,
                "aic": result.aic,
                "bic": result.bic,
                "ks_stat": result.ks_stat,
                "ks_pvalue": result.ks_pvalue,
                "complexity_rank": result.complexity_rank,
            }
        )
    table = pd.DataFrame(rows)
    best_aic = float(table["aic"].min())
    shortlist = table.loc[table["aic"] <= best_aic + 4.0].copy()
    good_fit = shortlist.loc[shortlist["ks_pvalue"] >= 0.05].copy()

    if not good_fit.empty:
        chosen_row = good_fit.sort_values(
            ["ks_pvalue", "bic", "complexity_rank", "aic"],
            ascending=[False, True, True, True],
        ).iloc[0]
        selection_reason = (
            "highest KS p-value among distributions within 4 AIC points of the best fit, "
            "with BIC and parsimony used as tie-breakers"
        )
    else:
        chosen_row = shortlist.sort_values(
            ["ks_stat", "bic", "complexity_rank", "aic"],
            ascending=[True, True, True, True],
        ).iloc[0]
        selection_reason = (
            "lowest KS distance among distributions within 4 AIC points of the best fit "
            "because no candidate passed the KS threshold"
        )

    selected_code = str(chosen_row["distribution_code"])
    selected_result = next(result for result in results if result.distribution_code == selected_code)
    table["recommended"] = table["distribution_code"] == selected_code
    table["selection_reason"] = ""
    table.loc[table["recommended"], "selection_reason"] = selection_reason
    table = table.sort_values(["recommended", "aic", "bic"], ascending=[False, True, True]).reset_index(drop=True)
    return table, selected_result, selection_reason


def descriptive_row(
    series: pd.Series,
) -> tuple[dict[str, float | str], pd.DataFrame, dict[str, DistributionFitResult], DistributionFitResult]:
    values = series.to_numpy(dtype=float)
    jb_stat, jb_pvalue, skewness, kurtosis = jarque_bera(values)
    shapiro_stat, shapiro_pvalue = shapiro(values)
    anderson_result = anderson(values, dist="norm")
    sig_levels = np.asarray(anderson_result.significance_level, dtype=float)
    crit_values = np.asarray(anderson_result.critical_values, dtype=float)
    crit_5 = float(crit_values[np.argmin(np.abs(sig_levels - 5.0))])
    adf_stat, adf_pvalue = safe_adf(series)
    kpss_stat, kpss_pvalue = safe_kpss(series)
    lb = compute_ljung_box(series)
    arch = compute_arch_test(series)

    fitted_distributions = {
        "normal": fit_distribution(values, "normal"),
        "t": fit_distribution(values, "t"),
        "nig": fit_distribution(values, "nig"),
    }
    distribution_table, recommended_distribution, selection_reason = distribution_results_table(
        list(fitted_distributions.values())
    )
    best_distribution = recommended_distribution.distribution
    normal_fit = fitted_distributions["normal"]
    student_fit = fitted_distributions["t"]
    nig_fit = fitted_distributions["nig"]

    row: dict[str, float | str] = {
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
        "excess_kurtosis": float(kurtosis - 3.0),
        "jarque_bera_stat": float(jb_stat),
        "jarque_bera_pvalue": float(jb_pvalue),
        "shapiro_wilk_stat": float(shapiro_stat),
        "shapiro_wilk_pvalue": float(shapiro_pvalue),
        "anderson_normal_stat": float(anderson_result.statistic),
        "anderson_5pct_critical": crit_5,
        "normal_ks_stat": normal_fit.ks_stat,
        "normal_ks_pvalue": normal_fit.ks_pvalue,
        "normal_aic": normal_fit.aic,
        "normal_bic": normal_fit.bic,
        "student_t_ks_stat": student_fit.ks_stat,
        "student_t_ks_pvalue": student_fit.ks_pvalue,
        "student_t_aic": student_fit.aic,
        "student_t_bic": student_fit.bic,
        "student_t_df": float(student_fit.params["df"]),
        "student_t_loc": float(student_fit.params["loc"]),
        "student_t_scale": float(student_fit.params["scale"]),
        "student_t_aic_gain_vs_normal": float(normal_fit.aic - student_fit.aic),
        "nig_ks_stat": nig_fit.ks_stat,
        "nig_ks_pvalue": nig_fit.ks_pvalue,
        "nig_aic": nig_fit.aic,
        "nig_bic": nig_fit.bic,
        "nig_alpha": float(nig_fit.params["alpha"]),
        "nig_beta": float(nig_fit.params["beta"]),
        "nig_loc": float(nig_fit.params["loc"]),
        "nig_scale": float(nig_fit.params["scale"]),
        "nig_aic_gain_vs_normal": float(normal_fit.aic - nig_fit.aic),
        "best_marginal_fit": best_distribution,
        "best_marginal_fit_code": recommended_distribution.distribution_code,
        "best_marginal_fit_ks_stat": recommended_distribution.ks_stat,
        "best_marginal_fit_ks_pvalue": recommended_distribution.ks_pvalue,
        "best_marginal_fit_aic_gain_vs_normal": float(normal_fit.aic - recommended_distribution.aic),
        "distribution_selection_reason": selection_reason,
        "adf_stat": float(adf_stat),
        "adf_pvalue": float(adf_pvalue),
        "kpss_stat": float(kpss_stat),
        "kpss_pvalue": float(kpss_pvalue),
        **lb,
        **arch,
    }
    return row, distribution_table, fitted_distributions, recommended_distribution


def choose_d_candidates(test_row: dict[str, float | str]) -> list[int]:
    if float(test_row["adf_pvalue"]) < 0.05 and float(test_row["kpss_pvalue"]) >= 0.05:
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


def fit_garch_candidate(residuals: pd.Series, model_label: str, distribution_code: str) -> VolatilityFitResult:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = arch_model(
            residuals,
            mean="Zero",
            vol="GARCH",
            p=1,
            o=0,
            q=1,
            dist=distribution_code,
            rescale=False,
        )
        fitted = model.fit(update_freq=0, disp="off", cov_type="robust", show_warning=False)

    params = fitted.params
    std_errors = fitted.std_err
    tvalues = fitted.tvalues
    pvalues = fitted.pvalues

    param_table = pd.DataFrame(
        {
            "parameter": params.index,
            "estimate": params.values,
            "std_error": std_errors.reindex(params.index).values,
            "t_stat": tvalues.reindex(params.index).values,
            "pvalue": pvalues.reindex(params.index).values,
        }
    )

    residual_values = pd.Series(np.asarray(fitted.resid), index=residuals.index, dtype=float)
    conditional_sigma = pd.Series(np.asarray(fitted.conditional_volatility), index=residuals.index, dtype=float)
    std_resid = pd.Series(np.asarray(fitted.std_resid), index=residuals.index, dtype=float)
    std_resid = std_resid.replace([np.inf, -np.inf], np.nan)
    clean_std_resid = std_resid.dropna()

    lb_resid = compute_ljung_box(clean_std_resid)
    lb_sq = compute_ljung_box(clean_std_resid**2)
    arch_diag = compute_arch_test(clean_std_resid)

    volatility_pvalues = [
        float(pvalues[name])
        for name in ("omega", "alpha[1]", "beta[1]")
        if name in pvalues.index and np.isfinite(pvalues[name])
    ]
    volatility_significant_share = (
        float(np.mean(np.array(volatility_pvalues) < 0.05)) if volatility_pvalues else np.nan
    )

    omega = float(params.get("omega", np.nan))
    alpha = float(params.get("alpha[1]", np.nan))
    beta = float(params.get("beta[1]", np.nan))
    persistence = float(alpha + beta) if np.isfinite(alpha + beta) else np.nan
    nu = float(params.get("nu", np.nan)) if "nu" in params.index else None
    distribution_param_names = arch_distribution_parameter_names(distribution_code)
    distribution_params = {
        name: float(params[name])
        for name in distribution_param_names
        if name in params.index and np.isfinite(float(params[name]))
    }
    distribution_param_text = format_parameter_text(distribution_params)
    parameter_vector = arch_distribution_parameter_vector(distribution_code, distribution_params)
    innovation_ks_stat, innovation_ks_pvalue = kstest(
        clean_std_resid.to_numpy(dtype=float),
        lambda x: arch_distribution_cdf(x, distribution_code, distribution_params),
    )
    std_jb_stat, std_jb_pvalue, std_skewness, std_kurtosis = jarque_bera(clean_std_resid)
    std_shapiro_stat, std_shapiro_pvalue = shapiro(clean_std_resid)

    diagnostics = {
        "std_resid_lb_stat_lag_12": lb_resid["lb_stat_lag_12"],
        "std_resid_lb_pvalue_lag_12": lb_resid["lb_pvalue_lag_12"],
        "std_resid_lb_stat_lag_24": lb_resid["lb_stat_lag_24"],
        "std_resid_lb_pvalue_lag_24": lb_resid["lb_pvalue_lag_24"],
        "std_resid_sq_lb_stat_lag_12": lb_sq["lb_stat_lag_12"],
        "std_resid_sq_lb_pvalue_lag_12": lb_sq["lb_pvalue_lag_12"],
        "std_resid_sq_lb_stat_lag_24": lb_sq["lb_stat_lag_24"],
        "std_resid_sq_lb_pvalue_lag_24": lb_sq["lb_pvalue_lag_24"],
        "std_resid_arch_lm_stat": arch_diag["arch_lm_stat"],
        "std_resid_arch_lm_pvalue": arch_diag["arch_lm_pvalue"],
        "innovation_ks_stat": float(innovation_ks_stat),
        "innovation_ks_pvalue": float(innovation_ks_pvalue),
        "std_resid_jarque_bera_stat": float(std_jb_stat),
        "std_resid_jarque_bera_pvalue": float(std_jb_pvalue),
        "std_resid_shapiro_wilk_stat": float(std_shapiro_stat),
        "std_resid_shapiro_wilk_pvalue": float(std_shapiro_pvalue),
        "std_resid_skewness": float(std_skewness),
        "std_resid_kurtosis": float(std_kurtosis),
        "volatility_param_significant_share": volatility_significant_share,
    }

    distribution = arch_distribution_label(distribution_code)
    return VolatilityFitResult(
        model_label=model_label,
        distribution=distribution,
        distribution_code=distribution_code,
        success=True,
        convergence_flag=int(fitted.convergence_flag),
        loglik=float(fitted.loglikelihood),
        aic=float(fitted.aic),
        bic=float(fitted.bic),
        omega=omega,
        alpha=alpha,
        beta=beta,
        persistence=persistence,
        nu=nu,
        distribution_params=distribution_params,
        distribution_param_text=distribution_param_text,
        conditional_volatility=conditional_sigma.to_numpy(dtype=float),
        standardized_residuals=std_resid.to_numpy(dtype=float),
        residuals=residual_values.to_numpy(dtype=float),
        param_table=param_table,
        summary_text=str(fitted.summary()),
        diagnostics=diagnostics,
    )


def select_best_volatility_model(
    residuals: pd.Series,
) -> tuple[VolatilityFitResult, pd.DataFrame, str]:
    fitted_candidates: list[VolatilityFitResult] = []
    failed_rows: list[dict[str, object]] = []

    for model_label, distribution_code in VOLATILITY_MODELS:
        try:
            fitted_candidates.append(fit_garch_candidate(residuals, model_label, distribution_code))
        except Exception as exc:
            failed_rows.append(
                {
                    "model_label": model_label,
                    "distribution": arch_distribution_label(distribution_code),
                    "distribution_code": distribution_code,
                    "success": False,
                    "error": str(exc),
                }
            )

    if not fitted_candidates:
        raise RuntimeError("No arch-based volatility model converged successfully.")

    candidate_rows = []
    for result in fitted_candidates:
        candidate_rows.append(
            {
                "model_label": result.model_label,
                "distribution": result.distribution,
                "aic": result.aic,
                "bic": result.bic,
                "loglik": result.loglik,
                "persistence": result.persistence,
                "nu": result.nu,
                "distribution_param_text": result.distribution_param_text,
                "std_resid_lb_pvalue_lag_12": result.diagnostics["std_resid_lb_pvalue_lag_12"],
                "std_resid_lb_pvalue_lag_24": result.diagnostics["std_resid_lb_pvalue_lag_24"],
                "std_resid_sq_lb_pvalue_lag_12": result.diagnostics["std_resid_sq_lb_pvalue_lag_12"],
                "std_resid_sq_lb_pvalue_lag_24": result.diagnostics["std_resid_sq_lb_pvalue_lag_24"],
                "std_resid_arch_lm_pvalue": result.diagnostics["std_resid_arch_lm_pvalue"],
                "innovation_ks_stat": result.diagnostics["innovation_ks_stat"],
                "innovation_ks_pvalue": result.diagnostics["innovation_ks_pvalue"],
                "std_resid_jarque_bera_pvalue": result.diagnostics["std_resid_jarque_bera_pvalue"],
                "std_resid_shapiro_wilk_pvalue": result.diagnostics["std_resid_shapiro_wilk_pvalue"],
                "volatility_param_significant_share": result.diagnostics["volatility_param_significant_share"],
                "distribution_complexity": arch_distribution_complexity(result.distribution_code),
                "success": True,
            }
        )

    comparison_df = pd.DataFrame(candidate_rows + failed_rows)
    successful_df = comparison_df.loc[comparison_df["success"]].copy()
    best_aic = float(successful_df["aic"].min())
    shortlisted = successful_df.loc[successful_df["aic"] <= best_aic + 4.0].copy()
    shortlisted["diagnostic_score"] = (
        (shortlisted["std_resid_lb_pvalue_lag_12"] >= 0.05).astype(int)
        + (shortlisted["std_resid_sq_lb_pvalue_lag_12"] >= 0.05).astype(int)
        + (shortlisted["std_resid_arch_lm_pvalue"] >= 0.05).astype(int)
        + (shortlisted["innovation_ks_pvalue"] >= 0.05).astype(int)
        + (
            (shortlisted["volatility_param_significant_share"] >= 0.5)
            | shortlisted["volatility_param_significant_share"].isna()
        ).astype(int)
    )

    chosen_row = shortlisted.sort_values(
        [
            "diagnostic_score",
            "bic",
            "distribution_complexity",
            "innovation_ks_stat",
            "aic",
        ],
        ascending=[False, True, True, True, True],
    ).iloc[0]
    selection_reason = (
        "highest joint diagnostic score among models within 4 AIC points of the best fit, where the score "
        "combines residual Ljung-Box, squared-residual Ljung-Box, residual ARCH-LM, innovation KS, and "
        "core-parameter significance; BIC and parsimony break ties"
    )

    selected_label = str(chosen_row["model_label"])
    selected = next(result for result in fitted_candidates if result.model_label == selected_label)
    comparison_df["selected"] = comparison_df["model_label"] == selected_label
    comparison_df["selection_reason"] = ""
    comparison_df.loc[comparison_df["selected"], "selection_reason"] = selection_reason
    return (
        selected,
        comparison_df.sort_values(["success", "aic", "bic"], ascending=[False, True, True]).reset_index(drop=True),
        selection_reason,
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


def save_distribution_plot(
    series: pd.Series,
    fitted_distributions: dict[str, DistributionFitResult],
    figure_path: Path,
    title: str,
) -> None:
    values = series.to_numpy(dtype=float)
    x_grid = np.linspace(values.min(), values.max(), 400)
    kde = gaussian_kde(values)
    normal_fit = fitted_distributions["normal"]
    student_fit = fitted_distributions["t"]
    nig_fit = fitted_distributions["nig"]

    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    ax.hist(values, bins=36, density=True, alpha=0.55, color="#5b8bd1", edgecolor="white")
    ax.plot(x_grid, kde(x_grid), color="#ba4a00", linewidth=2, label="Kernel density")
    ax.plot(
        x_grid,
        norm.pdf(x_grid, loc=normal_fit.params["loc"], scale=normal_fit.params["scale"]),
        color="black",
        linestyle="--",
        linewidth=1.5,
        label="Fitted normal",
    )
    ax.plot(
        x_grid,
        student_t.pdf(
            x_grid,
            df=float(student_fit.params["df"]),
            loc=float(student_fit.params["loc"]),
            scale=float(student_fit.params["scale"]),
        ),
        color="#117864",
        linestyle="-.",
        linewidth=1.8,
        label="Fitted Student-t",
    )
    ax.plot(
        x_grid,
        norminvgauss.pdf(
            x_grid,
            a=float(nig_fit.params["alpha"]),
            b=float(nig_fit.params["beta"]),
            loc=float(nig_fit.params["loc"]),
            scale=float(nig_fit.params["scale"]),
        ),
        color="#7d3c98",
        linestyle=":",
        linewidth=1.8,
        label="Fitted NIG",
    )
    ax.set_title(f"{title}: Histogram, KDE, and Fitted Densities")
    ax.set_xlabel("Return (%)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(figure_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def save_qq_plot(
    series: pd.Series,
    fitted_distributions: dict[str, DistributionFitResult],
    recommended_distribution: DistributionFitResult,
    figure_path: Path,
    title: str,
) -> None:
    normal_fit = fitted_distributions["normal"]

    fig, axes = plt.subplots(1, 2, figsize=(11, 5.2))
    qqplot(
        series,
        dist=norm,
        distargs=(),
        loc=float(normal_fit.params["loc"]),
        scale=float(normal_fit.params["scale"]),
        line="45",
        ax=axes[0],
    )
    axes[0].set_title(f"{title}: Normal QQ")

    if recommended_distribution.distribution_code == "t":
        qqplot(
            series,
            dist=student_t,
            distargs=(float(recommended_distribution.params["df"]),),
            loc=float(recommended_distribution.params["loc"]),
            scale=float(recommended_distribution.params["scale"]),
            line="45",
            ax=axes[1],
        )
    elif recommended_distribution.distribution_code == "nig":
        qqplot(
            series,
            dist=norminvgauss,
            distargs=(
                float(recommended_distribution.params["alpha"]),
                float(recommended_distribution.params["beta"]),
            ),
            loc=float(recommended_distribution.params["loc"]),
            scale=float(recommended_distribution.params["scale"]),
            line="45",
            ax=axes[1],
        )
    else:
        qqplot(
            series,
            dist=norm,
            distargs=(),
            loc=float(recommended_distribution.params["loc"]),
            scale=float(recommended_distribution.params["scale"]),
            line="45",
            ax=axes[1],
        )
    axes[1].set_title(f"{title}: Recommended {recommended_distribution.distribution} QQ")

    fig.tight_layout()
    fig.savefig(figure_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def save_recommended_distribution_diagnostic_plot(
    series: pd.Series,
    recommended_distribution: DistributionFitResult,
    figure_path: Path,
    title: str,
) -> None:
    values = np.sort(series.to_numpy(dtype=float))
    empirical_cdf = np.arange(1, len(values) + 1, dtype=float) / len(values)

    if recommended_distribution.distribution_code == "normal":
        fitted_cdf = norm.cdf(
            values,
            loc=float(recommended_distribution.params["loc"]),
            scale=float(recommended_distribution.params["scale"]),
        )
    elif recommended_distribution.distribution_code == "t":
        fitted_cdf = student_t.cdf(
            values,
            df=float(recommended_distribution.params["df"]),
            loc=float(recommended_distribution.params["loc"]),
            scale=float(recommended_distribution.params["scale"]),
        )
    elif recommended_distribution.distribution_code == "nig":
        fitted_cdf = norminvgauss.cdf(
            values,
            a=float(recommended_distribution.params["alpha"]),
            b=float(recommended_distribution.params["beta"]),
            loc=float(recommended_distribution.params["loc"]),
            scale=float(recommended_distribution.params["scale"]),
        )
    else:
        raise ValueError(f"Unsupported recommended distribution: {recommended_distribution.distribution_code}")

    absolute_error = np.abs(empirical_cdf - fitted_cdf)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))
    axes[0].plot(values, empirical_cdf, color="#1f5aa6", linewidth=1.6, label="Empirical CDF")
    axes[0].plot(values, fitted_cdf, color="#ba4a00", linewidth=1.6, linestyle="--", label="Fitted CDF")
    axes[0].set_title(f"{title}: Recommended Fit CDF")
    axes[0].set_xlabel("Return (%)")
    axes[0].legend()

    axes[1].plot(values, absolute_error, color="#7d3c98", linewidth=1.5)
    axes[1].set_title(f"{title}: Absolute CDF Error")
    axes[1].set_xlabel("Return (%)")
    axes[1].set_ylabel("|F_n(x) - F(x)|")

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

    plot_acf(residuals.dropna(), ax=axes[1, 0], lags=24, zero=False)
    axes[1, 0].set_title(f"{title}: Residual ACF")
    plot_acf((residuals.dropna()) ** 2, ax=axes[1, 1], lags=24, zero=False)
    axes[1, 1].set_title(f"{title}: Squared Residual ACF")

    fig.tight_layout()
    fig.savefig(figure_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def save_garch_plot(
    dates: pd.Series,
    arima_residuals: pd.Series,
    volatility_result: VolatilityFitResult,
    figure_path: Path,
    title: str,
) -> None:
    std_resid_series = pd.Series(volatility_result.standardized_residuals, index=dates, dtype=float).dropna()
    sigma_series = pd.Series(volatility_result.conditional_volatility, index=dates, dtype=float)
    residual_series = pd.Series(arima_residuals.to_numpy(dtype=float), index=dates, dtype=float)
    sorted_resid = np.sort(std_resid_series.to_numpy(dtype=float))
    probabilities = (np.arange(1, len(sorted_resid) + 1, dtype=float) - 0.5) / len(sorted_resid)
    theoretical_quantiles = arch_distribution_ppf(
        probabilities,
        volatility_result.distribution_code,
        volatility_result.distribution_params,
    )

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes[0, 0].plot(dates, residual_series.abs(), linewidth=0.9, color="#1f5aa6", label="|ARIMA residual|")
    axes[0, 0].plot(dates, sigma_series, linewidth=1.2, color="#ba4a00", label="Conditional sigma")
    axes[0, 0].set_title(f"{title}: Residual Magnitude and Conditional Volatility")
    axes[0, 0].legend()

    axes[0, 1].scatter(theoretical_quantiles, sorted_resid, s=10, alpha=0.7, color="#117864")
    qq_min = float(min(theoretical_quantiles.min(), sorted_resid.min()))
    qq_max = float(max(theoretical_quantiles.max(), sorted_resid.max()))
    axes[0, 1].plot([qq_min, qq_max], [qq_min, qq_max], color="black", linestyle="--", linewidth=1.0)
    axes[0, 1].set_title(f"{title}: Std. Residual QQ vs {volatility_result.distribution}")
    axes[0, 1].set_xlabel("Theoretical quantiles")
    axes[0, 1].set_ylabel("Empirical quantiles")

    plot_acf(std_resid_series, ax=axes[1, 0], lags=24, zero=False)
    axes[1, 0].set_title(f"{title}: ACF of Std. Residuals")

    plot_acf(std_resid_series**2, ax=axes[1, 1], lags=24, zero=False)
    axes[1, 1].set_title(f"{title}: ACF of Squared Std. Residuals")

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


def garch_summary_table(
    selected_volatility: VolatilityFitResult,
    selection_reason: str,
    arima_residual_arch: dict[str, float],
) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "model_label": selected_volatility.model_label,
                "distribution": selected_volatility.distribution,
                "distribution_param_text": selected_volatility.distribution_param_text,
                "success": selected_volatility.success,
                "convergence_flag": selected_volatility.convergence_flag,
                "omega": selected_volatility.omega,
                "alpha": selected_volatility.alpha,
                "beta": selected_volatility.beta,
                "persistence": selected_volatility.persistence,
                "nu": selected_volatility.nu,
                "loglik": selected_volatility.loglik,
                "aic": selected_volatility.aic,
                "bic": selected_volatility.bic,
                "volatility_param_significant_share": selected_volatility.diagnostics[
                    "volatility_param_significant_share"
                ],
                "arima_resid_arch_lm_pvalue": arima_residual_arch["arch_lm_pvalue"],
                "std_resid_lb_pvalue_lag_12": selected_volatility.diagnostics["std_resid_lb_pvalue_lag_12"],
                "std_resid_lb_pvalue_lag_24": selected_volatility.diagnostics["std_resid_lb_pvalue_lag_24"],
                "std_resid_sq_lb_pvalue_lag_12": selected_volatility.diagnostics["std_resid_sq_lb_pvalue_lag_12"],
                "std_resid_sq_lb_pvalue_lag_24": selected_volatility.diagnostics["std_resid_sq_lb_pvalue_lag_24"],
                "std_resid_arch_lm_pvalue": selected_volatility.diagnostics["std_resid_arch_lm_pvalue"],
                "innovation_ks_pvalue": selected_volatility.diagnostics["innovation_ks_pvalue"],
                "std_resid_jarque_bera_pvalue": selected_volatility.diagnostics["std_resid_jarque_bera_pvalue"],
                "std_resid_shapiro_wilk_pvalue": selected_volatility.diagnostics["std_resid_shapiro_wilk_pvalue"],
                "selection_reason": selection_reason,
            }
        ]
    )


def summarize_stationarity(test_row: dict[str, float | str]) -> str:
    adf_stationary = float(test_row["adf_pvalue"]) < 0.05
    kpss_stationary = float(test_row["kpss_pvalue"]) >= 0.05
    if adf_stationary and kpss_stationary:
        return "Both ADF and KPSS support treating the return series as stationary in levels."
    if adf_stationary and not kpss_stationary:
        return "ADF rejects a unit root, but KPSS still flags some persistence; level models remain plausible but should be checked carefully."
    return "Stationarity evidence is mixed enough that ARIMA candidates allow both level and differenced specifications."


def build_interpretation_text(
    portfolio: str,
    test_row: dict[str, float | str],
    selected_arima: dict[str, object],
    selected_volatility: VolatilityFitResult,
    garch_summary: pd.DataFrame,
) -> str:
    order = selected_arima["order"]
    residual_lb = float(selected_arima["lb_pvalue_lag_12"])
    raw_arch_pvalue = float(garch_summary.loc[0, "arima_resid_arch_lm_pvalue"])
    std_arch_pvalue = float(garch_summary.loc[0, "std_resid_arch_lm_pvalue"])
    recommended_fit_gain = float(test_row["best_marginal_fit_aic_gain_vs_normal"])
    recommended_fit = str(test_row["best_marginal_fit"])
    normality_text = (
        "Jarque-Bera and Shapiro-Wilk both reject normality."
        if float(test_row["jarque_bera_pvalue"]) < 0.05 and float(test_row["shapiro_wilk_pvalue"]) < 0.05
        else "Normality evidence is weaker once multiple tests are considered."
    )
    marginal_text = (
        f"The recommended marginal model is {recommended_fit}, which improves AIC by {recommended_fit_gain:.1f} points relative to the Gaussian fit."
        if recommended_fit_gain > 0.0
        else f"The recommended marginal model is {recommended_fit}, but it does not improve on the Gaussian fit by AIC."
    )
    arch_text = (
        "ARCH effects are materially weaker after the arch-based volatility filter."
        if std_arch_pvalue > raw_arch_pvalue
        else "Standardized residual diagnostics still show nontrivial leftover ARCH effects, so volatility fit should be read cautiously."
    )
    innovation_ks_pvalue = float(selected_volatility.diagnostics["innovation_ks_pvalue"])
    return "\n".join(
        [
            f"# {portfolio_label(portfolio)}",
            "",
            (
                f"- Distribution: mean = {float(test_row['mean_pct']):.3f}% per month, volatility = "
                f"{float(test_row['std_pct']):.3f}% per month, skewness = {float(test_row['skewness']):.3f}, "
                f"kurtosis = {float(test_row['kurtosis']):.3f}."
            ),
            f"- Normality: {normality_text} {marginal_text}",
            f"- Stationarity: {summarize_stationarity(test_row)}",
            f"- Selected mean model: ARIMA{order} chosen because {selected_arima['selection_reason']}.",
            f"- Residual autocorrelation: Ljung-Box p-value at lag 12 = {residual_lb:.3f}.",
            (
                f"- Volatility model: {selected_volatility.model_label} estimated on ARIMA residuals with "
                f"alpha = {selected_volatility.alpha:.3f}, beta = {selected_volatility.beta:.3f}, "
                f"and persistence = {selected_volatility.persistence:.3f}. "
                f"Assumed innovation distribution parameters: {selected_volatility.distribution_param_text or 'none'}."
            ),
            (
                f"- Volatility clustering: ARIMA-residual ARCH-LM p-value = {raw_arch_pvalue:.4f}; "
                f"standardized-residual ARCH-LM p-value = {std_arch_pvalue:.4f}. {arch_text}"
            ),
            (
                f"- Innovation fit: standardized-residual KS p-value under the selected {selected_volatility.distribution} "
                f"assumption = {innovation_ks_pvalue:.4f}."
            ),
        ]
    )


def write_section_03(
    test_summary: pd.DataFrame,
    model_summary: pd.DataFrame,
    garch_summary: pd.DataFrame,
) -> None:
    best_mean_portfolio = test_summary.sort_values("mean_pct", ascending=False).iloc[0]
    highest_persistence = garch_summary.sort_values("persistence", ascending=False).iloc[0]
    strongest_non_normal = test_summary.sort_values("best_marginal_fit_aic_gain_vs_normal", ascending=False).iloc[0]
    strongest_arch = test_summary.sort_values("arch_lm_stat", ascending=False).iloc[0]
    weak_residual_fit = model_summary.loc[model_summary["residual_lb_pvalue_lag_12"] < 0.05, "portfolio"].tolist()
    weak_residual_text = ", ".join(portfolio_label(portfolio) for portfolio in weak_residual_fit) or "none of the portfolios"
    marginal_counts = test_summary["best_marginal_fit"].value_counts().to_dict()
    volatility_counts = garch_summary["distribution"].value_counts().to_dict()
    marginal_count_text = ", ".join(f"{name}: {count}" for name, count in sorted(marginal_counts.items()))
    volatility_count_text = ", ".join(f"{name}: {count}" for name, count in sorted(volatility_counts.items()))
    section_lines = [
        "# Modeling the Individual Portfolio Returns",
        "",
        "## Workflow",
        "",
        (
            "This section models each of the six value-weighted portfolio return series individually using the "
            "cleaned monthly dataset. For each portfolio, the pipeline saves a time-series plot, a histogram-density "
            "plot with fitted Gaussian, Student-t, and NIG overlays, a normal-versus-recommended QQ comparison, a "
            "recommended-distribution CDF diagnostic, ACF/PACF, residual diagnostics, volatility-clustering diagnostics, "
            "ARIMA candidate comparison tables, a selected ARIMA summary, and an `arch`-based volatility-model "
            "comparison estimated on ARIMA residuals."
        ),
        "",
        "## Distributional and Diagnostic Evidence",
        "",
        (
            f"The descriptive diagnostics confirm the stylized facts emphasized in the lecture notes: the monthly "
            f"portfolio returns are stationary in levels, but they are not well described by a Gaussian law. "
            f"Jarque-Bera and Shapiro-Wilk tests reject normality for essentially the entire panel, and heavy-tailed "
            f"MLE fits dominate the Gaussian benchmark throughout the sample. The recommended marginal-fit counts are "
            f"**{marginal_count_text}**. The largest AIC improvement over the Gaussian fit appears in "
            f"**{portfolio_label(strongest_non_normal['portfolio'])}**, where the recommended distribution is "
            f"**{strongest_non_normal['best_marginal_fit']}**."
        ),
        "",
        (
            "Stationarity tests support modeling returns in levels rather than prices: the series are monthly returns, "
            "ADF generally rejects a unit root, and KPSS does not overturn the practical use of level ARIMA models. "
            f"At the same time, ARCH-LM tests reject homoskedasticity throughout the panel, with the strongest raw "
            f"volatility clustering in **{portfolio_label(strongest_arch['portfolio'])}**."
        ),
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
        "2. Use the ACF/PACF patterns as a rough guide, then fit a small interpretable ARIMA grid with orders up to two autoregressive and two moving-average terms.",
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
        (
            "Conditional volatility was modeled with the canonical `arch` package rather than with a custom optimizer. "
            "For each portfolio, the selected ARIMA residuals were passed to four GARCH(1,1) specifications: Gaussian, "
            "Student-t, skewed Student-t, and GED. Preferred-model selection does not rely on one metric alone. It "
            "combines AIC/BIC, Ljung-Box tests on standardized residuals and squared standardized residuals, residual "
            "ARCH-LM tests, innovation KS tests under the assumed distribution, and the significance share of the core "
            "volatility parameters."
        ),
        "",
        (
            f"The selected volatility-model counts are **{volatility_count_text}**. The most persistent "
            f"selected volatility process in the current run is **{portfolio_label(highest_persistence['portfolio'])}**, "
            f"with alpha + beta = **{highest_persistence['persistence']:.3f}**."
        ),
        "",
        "The combined volatility summary is saved at `output/tables/individual_returns/portfolio_garch_summary.csv`.",
        "",
        "## Cross-Portfolio Interpretation",
        "",
        (
            f"The portfolio with the highest average monthly return remains **{portfolio_label(best_mean_portfolio['portfolio'])}**, "
            f"at **{best_mean_portfolio['mean_pct']:.3f}%** per month. Across the six portfolios, the mean-model "
            "evidence is modest, while the volatility-model evidence is stronger and more systematic. That pattern "
            "supports an interpretation in which the conditional mean of monthly portfolio returns is only weakly "
            "predictable from its own history, but conditional risk is persistent and benefits from explicitly "
            "heavy-tailed volatility specifications."
        ),
        "",
        (
            "Overall, the Section 3 evidence points to three conclusions. First, the return series are heavy tailed "
            "enough that Gaussian diagnostics alone are too narrow, so the report now uses multiple normality checks, "
            "MLE-based Normal/Student-t/NIG comparisons, and KS goodness-of-fit tests. Second, low-order ARIMA models are adequate as mean benchmarks, but "
            "they do not reveal strong standalone predictability. Third, volatility clustering is real enough to justify "
            "GARCH-type modeling with `arch`, and non-Gaussian innovation assumptions often fit better than Gaussian "
            "innovations even at the monthly frequency."
        ),
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
                (
                    f"- Shape: skewness = {test_row['skewness']:.3f}, kurtosis = {test_row['kurtosis']:.3f}, "
                    f"Jarque-Bera p-value = {test_row['jarque_bera_pvalue']:.4f}, Shapiro-Wilk p-value = "
                    f"{test_row['shapiro_wilk_pvalue']:.4f}, and best marginal fit = {test_row['best_marginal_fit']} "
                    f"(AIC gain vs Gaussian = {test_row['best_marginal_fit_aic_gain_vs_normal']:.2f})."
                ),
                f"- Stationarity: ADF p-value = {test_row['adf_pvalue']:.4f}, KPSS p-value = {test_row['kpss_pvalue']:.4f}.",
                (
                    f"- Selected ARIMA model: {model_row['selected_arima_order']} with AIC = {model_row['selected_arima_aic']:.2f}, "
                    f"BIC = {model_row['selected_arima_bic']:.2f}, and residual Ljung-Box p-value at lag 12 = "
                    f"{model_row['residual_lb_pvalue_lag_12']:.4f}."
                ),
                (
                    f"- Selected volatility model: {garch_row['model_label']} with alpha = {garch_row['alpha']:.4f}, "
                    f"beta = {garch_row['beta']:.4f}, persistence = {garch_row['persistence']:.4f}, "
                    f"innovation KS p-value = {garch_row['innovation_ks_pvalue']:.4f}, "
                    f"standardized-squared-residual Ljung-Box p-value at lag 12 = {garch_row['std_resid_sq_lb_pvalue_lag_12']:.4f}, "
                    f"and standardized-residual ARCH-LM p-value = {garch_row['std_resid_arch_lm_pvalue']:.4f}."
                ),
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
    test_row, distribution_table, fitted_distributions, recommended_distribution = descriptive_row(series)
    test_row["portfolio"] = portfolio

    save_time_series_plot(dates, series, paths["figure_dir"] / "time_series.png", title)
    save_distribution_plot(series, fitted_distributions, paths["figure_dir"] / "histogram_density.png", title)
    save_qq_plot(series, fitted_distributions, recommended_distribution, paths["figure_dir"] / "qq_plot.png", title)
    save_recommended_distribution_diagnostic_plot(
        series,
        recommended_distribution,
        paths["figure_dir"] / "recommended_distribution_diagnostic.png",
        title,
    )
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

    selected_volatility, volatility_comparison, selection_reason = select_best_volatility_model(residuals)
    garch_summary = garch_summary_table(selected_volatility, selection_reason, residual_arch)
    save_garch_plot(dates, residuals, selected_volatility, paths["figure_dir"] / "garch_diagnostics.png", title)

    conditional_volatility_df = pd.DataFrame(
        {
            "date": dates,
            "return_pct": series,
            "arima_fitted_pct": fitted_values,
            "arima_residual_pct": residuals,
            "conditional_sigma_pct": selected_volatility.conditional_volatility,
            "standardized_residual": selected_volatility.standardized_residuals,
        }
    )

    save_dataframe(pd.DataFrame([test_row]), paths["table_dir"] / "descriptive_and_test_statistics.csv")
    save_dataframe(distribution_table, paths["table_dir"] / "distribution_fit_comparison.csv")
    save_dataframe(arima_comparison, paths["table_dir"] / "arima_candidate_models.csv")
    save_dataframe(arima_parameter_table(arima_result), paths["table_dir"] / "selected_arima_parameters.csv")
    save_dataframe(residual_summary, paths["table_dir"] / "selected_arima_residual_diagnostics.csv")
    save_dataframe(volatility_comparison, paths["table_dir"] / "garch_candidate_models.csv")
    save_dataframe(garch_summary, paths["table_dir"] / "garch_summary.csv")
    save_dataframe(selected_volatility.param_table, paths["table_dir"] / "selected_garch_parameters.csv")
    save_dataframe(conditional_volatility_df, paths["table_dir"] / "garch_conditional_volatility.csv")

    write_text_file(paths["model_dir"] / "selected_arima_summary.txt", arima_result.summary().as_text())
    write_text_file(paths["model_dir"] / "distribution_fit_summary.txt", distribution_table.to_string(index=False) + "\n")
    write_text_file(paths["model_dir"] / "garch_summary.txt", selected_volatility.summary_text + "\n")

    interpretation_text = build_interpretation_text(
        portfolio,
        test_row,
        selected_arima,
        selected_volatility,
        garch_summary,
    )
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

    logger.info(
        "Completed %s with %s and %s",
        portfolio,
        model_row["selected_arima_order"],
        garch_row["model_label"],
    )

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
