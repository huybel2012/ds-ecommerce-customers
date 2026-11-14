# Phase 2 — Reproducible Pipeline & Environment

**Base commit:** `36aa40a` on branch `Dung` (Phase 1). Python **3.11** is the team's tested interpreter. **Scope:** #22, #03, #04, #05, #24, #07.

## What was fixed

- `src/utils.py` supplies the previously missing `EXPECTED_COLUMNS`, `load_ecommerce_raw`, `validate_schema`, `unexpected_columns`, and `binary_domain_report`. It refuses to substitute processed data for missing raw data.
- Scripts `00`–`04` and the optional demo `02b` now resolve the repository root correctly. They can run from the root with `python -m ...` or through the runner.
- `requirements.txt` documents a compatible range of dependencies; `.python-version` records Python 3.11; `.gitignore` excludes raw downloads, intermediate CSVs, caches, and generated audit reports.
- `src/run_pipeline.py` orchestrates audit → data quality → missing analysis → outlier analysis → final cleaning, then checks source preservation, missing indicators and customer-time keys. A SHA-256 digest guards against silently modifying the source.
- Unit tests detect missing schemas, mutated source fields, imputed `Returns`, duplicate event keys and inconsistent missing indicators.

## First-time Windows setup

Open **PowerShell** in `C:\Code\KHDL\Project` (branch `Dung`). Do not delete your existing `venv` or Phase 1 files.

```powershell
cd C:\Code\KHDL\Project
git status
.\venv\Scripts\python.exe -m pip install -r requirements.txt
.\venv\Scripts\python.exe -m unittest discover -s tests -v
```

> This requirements file constrains main dependency versions but does **not** constitute a full transitive dependency lock. For an exact record of your working Windows installation use `python -m pip freeze > requirements-local.txt` and keep that machine-specific file private until reviewed.

## Required external dataset

The repository's ZIP intentionally **does not contain the raw CSV**. Obtain the original Kaggle dataset from:

https://www.kaggle.com/datasets/shriyashjagtap/e-commerce-customer-for-behavior-analysis

Place the original file **`ecommerce_customer_data_custom_ratios.csv`** at:

`C:\Code\KHDL\Project\data\raw\ecommerce_customer_data_custom_ratios.csv`

Do not rename the other dataset file to this filename, and do not synthesize raw data from the checked-in processed files. If using the Kaggle CLI, authenticate according to Kaggle's current instructions and download/unzip the dataset into `data/raw/`. The script will fail with a clear error if the raw source is missing or lacks one of the required 13 columns.

## Rebuild from raw

```powershell
.\venv\Scripts\python.exe -m src.run_pipeline --dry-run
.\venv\Scripts\python.exe -m src.run_pipeline
```

The runner executes sequentially:

1. `00_baseline_audit.py` (read-only raw audit)
2. `01_quality_assessment.py` (quality CSV)
3. `02_missing_analysis.py` (retains NaN; writes `Returns_was_missing`)
4. `03_outlier_analysis.py` (investigates candidate outliers, no automatic deletion)
5. `04_finalize_cleaning.py` (writes final cleaned CSV with integrity check)

**Key output:** `data/processed/ecommerce_customer_clean.csv` (generated, ignored by Git).

**Validation evidence:** `reports/phase2/run_manifest.json`, with raw/output SHA-256 and metrics; generated report folders `reports/missing/`, `reports/outliers/`, `reports/final/`.

Recheck an existing cleaned output without rerunning all stages:

```powershell
.\venv\Scripts\python.exe -m src.run_pipeline --validate-only
```

Optional exploratory figures:

```powershell
.\venv\Scripts\python.exe -m src.run_pipeline --stage eda
```

**Warning:** `--stage eda` runs the legacy `05_explanatory_visualization.py` and may overwrite tracked images in `reports/figures/`. Its titles and revenue semantics are not yet validated (Phase 3). Use the default `cleaning` stage when assessing Phase 2.

`02b_leakage_safe_imputation_demo.py` is an optional classroom demonstration, not a dependency of the raw → clean pipeline. Run separately from repo root if necessary:

```powershell
.\venv\Scripts\python.exe src\pipeline\02b_leakage_safe_imputation_demo.py
```

The W9 hypothesis test (`07`) and W10 regression (`06`) currently use the **legacy** checked-in file `ecommerce_customer_cleaned.csv`; the files are not included in the Phase 2 orchestrated build because their outcome definitions and handling of missing `Returns` require Phase 4 review. Do **not** claim that their reports were regenerated from the new raw → clean output in Phase 2.

## Acceptance checklist

- `unittest` runs Phase 1 and Phase 2 tests without errors.
- Actual 13-column Kaggle raw CSV exists; `--dry-run` says `Raw input available: True`.
- Default runner exits successfully; `run_manifest.json` records **250,000 rows**, **49,673 customers**, and **47,596 NULL Returns** if the exact same original source version was used.
- `Returns_was_missing.sum() == 47596`, no source values were overwritten and the raw input SHA-256 remains unchanged.
- A second `--validate-only` run passes.
- No generated intermediate CSVs or raw names are staged in Git.

**Local verification performed while preparing this patch:** 2,800-row **derived test fixture** (not the Kaggle raw dataset), with 530 artificially missing Returns, completed stages 00–04 and integrity gates. This verifies execution mechanics only. We cannot sign off #07 on the real 250,000-row raw data until the original file is available and the user runs the commands above.

## Problems belonging to future phases

- `Total Purchase Amount` vs `Price × Quantity`, and currency/license remain unresolved as stated in Phase 1.
- Dashboard currently prefers a newly generated clean CSV; this may change display results after a rebuild. Metric semantics and charts are Phase 3.
- Rebuilding W9/W10 model results on the corrected data is Phase 4; do not claim old results correspond to new data.
- The final README/academic report polish is Phase 7.
