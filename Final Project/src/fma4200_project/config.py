from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
RAW_DATA_PATH = ROOT / "Data.csv"
GUIDANCE_PATH = ROOT / "Guidance.md"

SRC_DIR = ROOT / "src"
SCRIPTS_DIR = ROOT / "scripts"
DATA_DIR = ROOT / "data"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
OUTPUT_DIR = ROOT / "output"
FIGURES_DIR = OUTPUT_DIR / "figures"
TABLES_DIR = OUTPUT_DIR / "tables"
MODELS_DIR = OUTPUT_DIR / "models"
REPORT_DIR = ROOT / "report"
REPORT_SECTIONS_DIR = REPORT_DIR / "sections"
LOGS_DIR = ROOT / "logs"

README_PATH = ROOT / "README.md"
PROJECT_STATUS_PATH = ROOT / "PROJECT_STATUS.md"
ENVIRONMENT_USED_PATH = ROOT / "environment_used.md"

AUTOGEN_DATA_SNAPSHOT_PATH = REPORT_SECTIONS_DIR / "_02_data_snapshot_autogen.md"
ENVIRONMENT_LOG_PATH = LOGS_DIR / "environment_check.txt"
PIPELINE_LOG_PATH = LOGS_DIR / "pipeline_run.log"

CLEAN_PERCENT_DATA_PATH = PROCESSED_DATA_DIR / "monthly_portfolio_returns_clean.csv"
CLEAN_DECIMAL_DATA_PATH = PROCESSED_DATA_DIR / "monthly_portfolio_returns_decimal.csv"
DATA_DICTIONARY_PATH = PROCESSED_DATA_DIR / "data_dictionary.csv"
FAMA_FRENCH_FACTORS_PATH = PROCESSED_DATA_DIR / "fama_french_3f_monthly.csv"
PREDICTOR_DATASET_PATH = PROCESSED_DATA_DIR / "predictor_dataset_monthly.csv"
PREDICTOR_SOURCE_SUMMARY_PATH = PROCESSED_DATA_DIR / "predictor_source_summary.csv"

DATE_COVERAGE_PATH = TABLES_DIR / "date_coverage_check.csv"
DUPLICATE_CHECK_PATH = TABLES_DIR / "duplicate_check.csv"
MISSINGNESS_PATH = TABLES_DIR / "missingness_check.csv"
DESCRIPTIVE_STATS_PCT_PATH = TABLES_DIR / "descriptive_statistics_pct.csv"
DESCRIPTIVE_STATS_DEC_PATH = TABLES_DIR / "descriptive_statistics_decimal.csv"
SUMMARY_SNAPSHOT_PATH = TABLES_DIR / "portfolio_summary_snapshot.csv"
CORRELATION_MATRIX_PATH = TABLES_DIR / "portfolio_correlation_matrix.csv"
SANITY_SUMMARY_PATH = TABLES_DIR / "sanity_checks_summary.csv"

RETURNS_FIGURE_PATH = FIGURES_DIR / "monthly_returns_overview.png"
GROWTH_FIGURE_PATH = FIGURES_DIR / "cumulative_growth_of_1.png"
CORRELATION_HEATMAP_PATH = FIGURES_DIR / "portfolio_correlation_heatmap.png"
MODELS_README_PATH = MODELS_DIR / "README.md"

INDIVIDUAL_FIGURES_DIR = FIGURES_DIR / "individual_returns"
INDIVIDUAL_TABLES_DIR = TABLES_DIR / "individual_returns"
INDIVIDUAL_MODELS_DIR = MODELS_DIR / "individual_returns"
PREDICTIVE_FIGURES_DIR = FIGURES_DIR / "predictive_individual_returns"
PREDICTIVE_TABLES_DIR = TABLES_DIR / "predictive_individual_returns"
PREDICTIVE_MODELS_DIR = MODELS_DIR / "predictive_individual_returns"

SECTION_03_PATH = REPORT_SECTIONS_DIR / "03_individual_returns_modeling.md"
APPENDIX_INDIVIDUAL_PATH = REPORT_SECTIONS_DIR / "appendix_individual_returns_modeling.md"
INDIVIDUAL_MODELING_LOG_PATH = LOGS_DIR / "individual_modeling.log"
PREDICTIVE_MODELING_LOG_PATH = LOGS_DIR / "predictive_modeling.log"

PORTFOLIO_MODEL_COMPARISON_PATH = INDIVIDUAL_TABLES_DIR / "portfolio_model_comparison_summary.csv"
PORTFOLIO_TEST_SUMMARY_PATH = INDIVIDUAL_TABLES_DIR / "portfolio_statistical_test_summary.csv"
PORTFOLIO_GARCH_SUMMARY_PATH = INDIVIDUAL_TABLES_DIR / "portfolio_garch_summary.csv"
PREDICTIVE_MODEL_SUMMARY_PATH = PREDICTIVE_TABLES_DIR / "predictive_model_summary.csv"
PREDICTIVE_FORECAST_METRICS_PATH = PREDICTIVE_TABLES_DIR / "predictive_forecast_metrics.csv"
PREDICTIVE_FORECASTS_PATH = PREDICTIVE_TABLES_DIR / "predictive_forecasts.csv"

REQUIRED_IMPORTS = ("numpy", "pandas", "matplotlib", "scipy", "statsmodels", "pandas_datareader")

RAW_TO_PERCENT_COLUMNS = {
    "SMALL LoBM": "small_lobm_vwret_pct",
    "ME1 BM2": "me1_bm2_vwret_pct",
    "SMALL HiBM": "small_hibm_vwret_pct",
    "BIG LoBM": "big_lobm_vwret_pct",
    "ME2 BM2": "me2_bm2_vwret_pct",
    "BIG HiBM": "big_hibm_vwret_pct",
}

PERCENT_COLUMNS = list(RAW_TO_PERCENT_COLUMNS.values())
DECIMAL_COLUMNS = [column.replace("_pct", "_dec") for column in PERCENT_COLUMNS]

PROJECT_DIRECTORIES = (
    SRC_DIR,
    SCRIPTS_DIR,
    PROCESSED_DATA_DIR,
    FIGURES_DIR,
    INDIVIDUAL_FIGURES_DIR,
    PREDICTIVE_FIGURES_DIR,
    TABLES_DIR,
    INDIVIDUAL_TABLES_DIR,
    PREDICTIVE_TABLES_DIR,
    MODELS_DIR,
    INDIVIDUAL_MODELS_DIR,
    PREDICTIVE_MODELS_DIR,
    REPORT_SECTIONS_DIR,
    LOGS_DIR,
)
