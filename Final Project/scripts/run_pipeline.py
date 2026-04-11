from __future__ import annotations

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from fma4200_project.data_pipeline import run_cleaning_pipeline
from fma4200_project.environment import run_environment_check


def main() -> None:
    package_versions = run_environment_check()
    result = run_cleaning_pipeline()

    print("Full pipeline completed successfully.")
    print("Verified imports:")
    for name, version in package_versions:
        print(f"- {name}=={version}")
    print(f"Rows: {result['n_rows']}")
    print(f"Coverage: {result['start_date']:%Y-%m-%d} to {result['end_date']:%Y-%m-%d}")
    print(f"Duplicate date rows: {result['duplicate_date_rows']}")
    print(f"Total missing values: {result['total_missing_values']}")


if __name__ == "__main__":
    main()
