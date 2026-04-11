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

REPORT_SECTION_PATH = REPORT_SECTIONS_DIR / "02_data_source_and_processing.md"
ENVIRONMENT_LOG_PATH = LOGS_DIR / "environment_check.txt"
PIPELINE_LOG_PATH = LOGS_DIR / "pipeline_run.log"

CLEAN_PERCENT_DATA_PATH = PROCESSED_DATA_DIR / "monthly_portfolio_returns_clean.csv"
CLEAN_DECIMAL_DATA_PATH = PROCESSED_DATA_DIR / "monthly_portfolio_returns_decimal.csv"
DATA_DICTIONARY_PATH = PROCESSED_DATA_DIR / "data_dictionary.csv"

DATE_COVERAGE_PATH = TABLES_DIR / "date_coverage_check.csv"
DUPLICATE_CHECK_PATH = TABLES_DIR / "duplicate_check.csv"
MISSINGNESS_PATH = TABLES_DIR / "missingness_check.csv"
DESCRIPTIVE_STATS_PCT_PATH = TABLES_DIR / "descriptive_statistics_pct.csv"
DESCRIPTIVE_STATS_DEC_PATH = TABLES_DIR / "descriptive_statistics_decimal.csv"
SANITY_SUMMARY_PATH = TABLES_DIR / "sanity_checks_summary.csv"

RETURNS_FIGURE_PATH = FIGURES_DIR / "monthly_returns_overview.png"
GROWTH_FIGURE_PATH = FIGURES_DIR / "cumulative_growth_of_1.png"
MODELS_README_PATH = MODELS_DIR / "README.md"

REQUIRED_IMPORTS = ("numpy", "pandas", "matplotlib")

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
    TABLES_DIR,
    MODELS_DIR,
    REPORT_SECTIONS_DIR,
    LOGS_DIR,
)

