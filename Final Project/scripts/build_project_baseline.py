from __future__ import annotations

from io import StringIO
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RAW_PATH = ROOT / "Data.csv"
PROCESSED_DIR = ROOT / "data" / "processed"
TABLES_DIR = ROOT / "outputs" / "tables"
FIGURES_DIR = ROOT / "outputs" / "figures"
REPORT_PATH = ROOT / "report" / "02_data_source_and_processing.md"

RAW_TO_CLEAN = {
    "SMALL LoBM": "small_lobm",
    "ME1 BM2": "me1_bm2",
    "SMALL HiBM": "small_hibm",
    "BIG LoBM": "big_lobm",
    "ME2 BM2": "me2_bm2",
    "BIG HiBM": "big_hibm",
}


def ensure_directories() -> None:
    for path in (PROCESSED_DIR, TABLES_DIR, FIGURES_DIR, REPORT_PATH.parent):
        path.mkdir(parents=True, exist_ok=True)


def extract_monthly_block(raw_path: Path) -> pd.DataFrame:
    lines = raw_path.read_text(encoding="utf-8").splitlines()
    header_index = next(
        index for index, line in enumerate(lines) if line.strip().startswith(",SMALL LoBM")
    )

    block_lines = [lines[header_index]]
    for line in lines[header_index + 1 :]:
        if line.startswith("Copyright"):
            break
        if line.replace(",", "").strip() == "":
            break
        block_lines.append(line)

    monthly = pd.read_csv(StringIO("\n".join(block_lines)))
    monthly = monthly.rename(columns={monthly.columns[0]: "yyyymm", **RAW_TO_CLEAN})
    monthly["yyyymm"] = monthly["yyyymm"].astype(int)
    monthly["date"] = pd.to_datetime(monthly["yyyymm"].astype(str), format="%Y%m") + pd.offsets.MonthEnd(0)
    monthly = monthly.replace({-99.99: np.nan, -999.0: np.nan})

    ordered_columns = ["date", "yyyymm", *RAW_TO_CLEAN.values()]
    return monthly.loc[:, ordered_columns].sort_values("date").reset_index(drop=True)


def build_tidy_dataset(monthly_wide: pd.DataFrame) -> pd.DataFrame:
    monthly_tidy = monthly_wide.melt(
        id_vars=["date", "yyyymm"],
        var_name="portfolio",
        value_name="return_pct",
    )
    monthly_tidy["return_decimal"] = monthly_tidy["return_pct"] / 100.0
    return monthly_tidy.sort_values(["date", "portfolio"]).reset_index(drop=True)


def build_overview(monthly_wide: pd.DataFrame) -> pd.DataFrame:
    portfolio_columns = list(RAW_TO_CLEAN.values())
    overview = pd.DataFrame(
        {
            "metric": [
                "raw_file",
                "sample_start",
                "sample_end",
                "n_months",
                "n_portfolios",
                "n_missing_values",
                "return_units",
            ],
            "value": [
                RAW_PATH.name,
                monthly_wide["date"].min().date().isoformat(),
                monthly_wide["date"].max().date().isoformat(),
                len(monthly_wide),
                len(portfolio_columns),
                int(monthly_wide[portfolio_columns].isna().sum().sum()),
                "percent monthly returns",
            ],
        }
    )
    return overview


def build_summary(monthly_wide: pd.DataFrame) -> pd.DataFrame:
    portfolio_data = monthly_wide[list(RAW_TO_CLEAN.values())]
    summary = pd.DataFrame(
        {
            "n_obs": portfolio_data.count(),
            "missing_count": portfolio_data.isna().sum(),
            "mean_pct": portfolio_data.mean(),
            "std_pct": portfolio_data.std(),
            "annualized_mean_pct": portfolio_data.mean() * 12.0,
            "annualized_vol_pct": portfolio_data.std() * np.sqrt(12.0),
            "min_pct": portfolio_data.min(),
            "p25_pct": portfolio_data.quantile(0.25),
            "median_pct": portfolio_data.median(),
            "p75_pct": portfolio_data.quantile(0.75),
            "max_pct": portfolio_data.max(),
            "skewness": portfolio_data.skew(),
            "kurtosis": portfolio_data.kurt(),
            "autocorr_1": portfolio_data.apply(lambda series: series.autocorr(lag=1)),
        }
    )
    summary.index.name = "portfolio"
    return summary


def build_missingness(monthly_wide: pd.DataFrame) -> pd.DataFrame:
    portfolio_data = monthly_wide[list(RAW_TO_CLEAN.values())]
    missingness = pd.DataFrame(
        {
            "portfolio": portfolio_data.columns,
            "missing_count": portfolio_data.isna().sum().values,
            "missing_share": portfolio_data.isna().mean().round(6).values,
        }
    )
    return missingness


def build_correlations(monthly_wide: pd.DataFrame) -> pd.DataFrame:
    return monthly_wide[list(RAW_TO_CLEAN.values())].corr()


def plot_monthly_returns(monthly_wide: pd.DataFrame) -> None:
    try:
        plt.style.use("seaborn-v0_8-whitegrid")
    except OSError:
        pass

    portfolio_columns = list(RAW_TO_CLEAN.values())
    fig, axes = plt.subplots(3, 2, figsize=(14, 10), sharex=True)
    axes = axes.ravel()

    for axis, column in zip(axes, portfolio_columns):
        axis.plot(monthly_wide["date"], monthly_wide[column], linewidth=0.9)
        axis.axhline(0.0, color="black", linewidth=0.8, alpha=0.6)
        axis.set_title(column)
        axis.set_ylabel("Return (%)")

    fig.suptitle("Monthly Returns by Portfolio", fontsize=14)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "monthly_returns.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_cumulative_growth(monthly_wide: pd.DataFrame) -> None:
    try:
        plt.style.use("seaborn-v0_8-whitegrid")
    except OSError:
        pass

    portfolio_columns = list(RAW_TO_CLEAN.values())
    cumulative_growth = (1.0 + monthly_wide[portfolio_columns] / 100.0).cumprod()

    fig, ax = plt.subplots(figsize=(12, 6))
    for column in portfolio_columns:
        ax.plot(monthly_wide["date"], cumulative_growth[column], label=column, linewidth=1.1)

    ax.set_title("Growth of $1 Invested")
    ax.set_ylabel("Portfolio Value")
    ax.legend(ncol=3, fontsize=9)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "cumulative_growth.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def strongest_correlation_pair(correlations: pd.DataFrame) -> tuple[str, float]:
    upper_triangle = correlations.where(np.triu(np.ones(correlations.shape), k=1).astype(bool))
    stacked = upper_triangle.stack()
    pair = stacked.idxmax()
    return f"{pair[0]} vs {pair[1]}", float(stacked.max())


def weakest_correlation_pair(correlations: pd.DataFrame) -> tuple[str, float]:
    upper_triangle = correlations.where(np.triu(np.ones(correlations.shape), k=1).astype(bool))
    stacked = upper_triangle.stack()
    pair = stacked.idxmin()
    return f"{pair[0]} vs {pair[1]}", float(stacked.min())


def write_report(monthly_wide: pd.DataFrame, summary: pd.DataFrame, correlations: pd.DataFrame) -> None:
    highest_mean_portfolio = summary["mean_pct"].idxmax()
    highest_vol_portfolio = summary["std_pct"].idxmax()
    strongest_pair, strongest_value = strongest_correlation_pair(correlations)
    weakest_pair, weakest_value = weakest_correlation_pair(correlations)
    missing_total = int(summary["missing_count"].sum())

    report_text = f"""# Data Source and Processing

## Data Source

This project uses the local `Data.csv` file, which records monthly value-weighted returns for six Kenneth French portfolios sorted by market equity (ME) and book-to-market equity (BE/ME). The raw file states that the portfolios include utilities and financials and were created from the 202601 CRSP database.

## Processing Workflow

The raw CSV contains a descriptive metadata block, one monthly return section, and a footer. The ingestion script `scripts/build_project_baseline.py` automatically locates the true header row instead of relying on a fixed skip count. It then:

1. extracts the monthly return block only,
2. converts the `YYYYMM` code into a month-end calendar date,
3. renames the six portfolio columns to snake_case labels for downstream analysis, and
4. converts the documented missing-value sentinels (`-99.99` and `-999`) to `NaN`.

The cleaned datasets are saved in both wide and tidy forms under `data/processed/`.

## Dataset Snapshot

The cleaned sample runs from **{monthly_wide["date"].min():%B %Y}** to **{monthly_wide["date"].max():%B %Y}**, giving **{len(monthly_wide)} monthly observations** for each of the six portfolios. The extracted monthly block contains **{missing_total} missing values** after sentinel conversion.

From the baseline summary tables:

- `{highest_mean_portfolio}` has the highest average monthly return at **{summary.loc[highest_mean_portfolio, "mean_pct"]:.3f}%**.
- `{highest_vol_portfolio}` has the highest monthly volatility at **{summary.loc[highest_vol_portfolio, "std_pct"]:.3f}%**.
- The strongest pairwise correlation is **{strongest_pair} = {strongest_value:.3f}**.
- The weakest pairwise correlation is **{weakest_pair} = {weakest_value:.3f}**.

The main outputs for this section are:

- `outputs/tables/dataset_overview.csv`
- `outputs/tables/summary_statistics.csv`
- `outputs/tables/missingness_check.csv`
- `outputs/tables/correlation_matrix.csv`
- `outputs/figures/monthly_returns.png`
- `outputs/figures/cumulative_growth.png`

These files provide a clean starting point for the later univariate modeling, multivariate modeling, and portfolio strategy sections of the project.
"""

    REPORT_PATH.write_text(report_text, encoding="utf-8")


def main() -> None:
    ensure_directories()

    monthly_wide = extract_monthly_block(RAW_PATH)
    monthly_tidy = build_tidy_dataset(monthly_wide)
    overview = build_overview(monthly_wide)
    summary = build_summary(monthly_wide)
    missingness = build_missingness(monthly_wide)
    correlations = build_correlations(monthly_wide)

    monthly_wide.to_csv(PROCESSED_DIR / "monthly_portfolios_wide.csv", index=False)
    monthly_tidy.to_csv(PROCESSED_DIR / "monthly_portfolios_tidy.csv", index=False)
    overview.to_csv(TABLES_DIR / "dataset_overview.csv", index=False)
    summary.round(6).to_csv(TABLES_DIR / "summary_statistics.csv")
    missingness.to_csv(TABLES_DIR / "missingness_check.csv", index=False)
    correlations.round(6).to_csv(TABLES_DIR / "correlation_matrix.csv")

    plot_monthly_returns(monthly_wide)
    plot_cumulative_growth(monthly_wide)
    write_report(monthly_wide, summary, correlations)

    print("Baseline artifacts created successfully:")
    print(f"- {PROCESSED_DIR / 'monthly_portfolios_wide.csv'}")
    print(f"- {PROCESSED_DIR / 'monthly_portfolios_tidy.csv'}")
    print(f"- {TABLES_DIR / 'summary_statistics.csv'}")
    print(f"- {TABLES_DIR / 'correlation_matrix.csv'}")
    print(f"- {FIGURES_DIR / 'monthly_returns.png'}")
    print(f"- {FIGURES_DIR / 'cumulative_growth.png'}")
    print(f"- {REPORT_PATH}")


if __name__ == "__main__":
    main()
