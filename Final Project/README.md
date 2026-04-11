# FMA4200 Final Project

This project analyzes six Kenneth French value-weighted portfolios using the course-provided `Data.csv` file. The reproducible setup in this folder focuses on data cleaning, environment verification, and baseline sanity checks so later modeling and trading-strategy work can build on a clean foundation.

## Canonical Structure

- `src/fma4200_project/`: reusable project code for environment checks and data cleaning
- `scripts/`: runnable scripts
- `data/processed/`: cleaned datasets and the data dictionary
- `output/figures/`: baseline figures
- `output/tables/`: sanity-check tables and descriptive statistics
- `output/models/`: reserved for later model outputs
- `report/sections/`: report section drafts
- `logs/`: environment and pipeline logs

## Return Units

The raw Kenneth French monthly value-weighted section reports returns in **percent** units. For example, `1.0866` means `1.0866%`, not `108.66%`.

- The canonical cleaned dataset is `data/processed/monthly_portfolio_returns_clean.csv`.
- Its six portfolio columns end with `_pct` and remain in percent units.
- The companion decimal-form dataset is `data/processed/monthly_portfolio_returns_decimal.csv`.
- Each `_dec` column is created by dividing the corresponding `_pct` column by `100.0`.

## Main Files

- Cleaned data: `data/processed/monthly_portfolio_returns_clean.csv`
- Decimal companion data: `data/processed/monthly_portfolio_returns_decimal.csv`
- Data dictionary: `data/processed/data_dictionary.csv`
- Environment summary: `environment_used.md`
- Pipeline log: `logs/pipeline_run.log`
- Data-processing report draft: `report/sections/02_data_source_and_processing.md`

## Reproducible Commands

Run the environment check:

```powershell
& 'd:\MG\anaconda3\python.exe' 'scripts\check_environment.py'
```

Run only the cleaning and sanity-check step:

```powershell
& 'd:\MG\anaconda3\python.exe' 'scripts\clean_data.py'
```

Run the full current pipeline:

```powershell
& 'd:\MG\anaconda3\python.exe' 'scripts\run_pipeline.py'
```

## Current Outputs

The current pipeline saves:

- cleaned percent and decimal datasets,
- a data dictionary,
- date coverage, duplicate, missingness, and descriptive-statistics tables,
- two baseline figures, and
- a short report section draft for data source and processing.

The canonical current output folder is `output/`.
