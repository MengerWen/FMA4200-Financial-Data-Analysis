# FMA4200 Final Project

This project analyzes six Kenneth French value-weighted portfolios using the course-provided `Data.csv` file. The reproducible setup in this folder now covers environment verification, data cleaning, baseline sanity checks, and the core univariate return-modeling stage for the six portfolios.

## Canonical Structure

- `src/fma4200_project/`: reusable project code for environment checks, data cleaning, and modeling
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
- Data-processing report draft: `report/sections/02_data_source_processing.md`
- Individual-modeling report draft: `report/sections/03_individual_returns_modeling.md`
- Modeling appendix: `report/sections/appendix_individual_returns_modeling.md`

## Reproducible Commands

Run the environment check:

```powershell
& 'd:\MG\anaconda3\python.exe' 'scripts\check_environment.py'
```

Run only the cleaning and sanity-check step:

```powershell
& 'd:\MG\anaconda3\python.exe' 'scripts\clean_data.py'
```

Run only the individual portfolio modeling step:

```powershell
& 'd:\MG\anaconda3\python.exe' 'scripts\run_individual_modeling.py'
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
- baseline portfolio figures,
- per-portfolio univariate diagnostics and model summaries,
- combined ARIMA and GARCH comparison tables, and
- report drafts for the data and individual-modeling sections.

The canonical current output folder is `output/`.
