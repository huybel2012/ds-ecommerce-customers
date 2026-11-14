"""Phase 2: reproducible entrypoint for the raw -> cleaned data pipeline.

Usage (from the repository root):
  python -m src.run_pipeline --dry-run
  python -m src.run_pipeline
  python -m src.run_pipeline --stage eda
  python -m src.run_pipeline --validate-only

The source CSV is a required, external Kaggle input; never reconstruct it from
processed data. W9/W10 model reports are deliberately not regenerated here:
those analyses depend on the unresolved metric contract (Phase 4).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from src.utils import RAW_DATA_PATH, REPO_ROOT, load_ecommerce_raw

CLEAN_PATH = REPO_ROOT / "data/processed/ecommerce_customer_clean.csv"
MANIFEST_PATH = REPO_ROOT / "reports/phase2/run_manifest.json"

STAGES = (
    "00_baseline_audit.py",
    "01_quality_assessment.py",
    "02_missing_analysis.py",
    "03_outlier_analysis.py",
    "04_finalize_cleaning.py",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_cleaned(source: pd.DataFrame, final: pd.DataFrame) -> dict:
    """Fail closed on row/field corruption; do not rewrite the raw records."""
    if len(source) != len(final) or not len(source):
        raise ValueError(f"Row count mismatch: raw={len(source)}, cleaned={len(final)}")
    required = ["Customer ID", "Purchase Date", "Product Price", "Quantity",
                "Total Purchase Amount", "Age", "Gender", "Churn", "Returns",
                "Returns_was_missing"]
    missing = set(required) - set(final.columns)
    if missing:
        raise ValueError(f"Final dataset is missing required columns: {sorted(missing)}")
    if "Customer Age" in final:
        raise ValueError("Redundant Customer Age should have been dropped")
    if final.duplicated(["Customer ID", "Purchase Date"]).any():
        raise ValueError("Final dataset has duplicated customer-timestamp keys")

    # The current cleaning stages retain event order. If that changes, this
    # guard intentionally rejects positional comparison rather than guessing.
    for column in ["Customer ID", "Product Price", "Quantity", "Total Purchase Amount", "Age", "Churn"]:
        if not np.array_equal(source[column].to_numpy(), final[column].to_numpy(), equal_nan=True):
            raise ValueError(f"Unexpected change in source field {column}")
    raw_dates = pd.to_datetime(source["Purchase Date"], errors="raise")
    clean_dates = pd.to_datetime(final["Purchase Date"], errors="raise")
    if not raw_dates.equals(clean_dates):
        raise ValueError("Purchase Date changed unexpectedly")
    if not final["Returns"].equals(source["Returns"]):
        # Comparing the values is more important than dtype (int/float due to NaN).
        left = pd.to_numeric(final["Returns"], errors="raise").to_numpy(dtype=float)
        right = pd.to_numeric(source["Returns"], errors="raise").to_numpy(dtype=float)
        if not np.array_equal(left, right, equal_nan=True):
            raise ValueError("Observed or missing Returns changed unexpectedly")
    expected_flag = source["Returns"].isna().astype("int8")
    if not final["Returns_was_missing"].reset_index(drop=True).eq(expected_flag.reset_index(drop=True)).all():
        raise ValueError("Returns_was_missing does not match raw nulls")
    if "Customer Age" in source and not source["Customer Age"].equals(source["Age"]):
        raise ValueError("Raw age columns disagree; deleting one would lose information")
    for column in ("Gender", "Product Category", "Payment Method"):
        same = final[column].astype("string").str.casefold().str.strip().eq(
            source[column].astype("string").str.casefold().str.strip()
        )
        if not same.fillna(False).all():
            raise ValueError(f"{column} changed unexpectedly")
    if (~final["Churn"].isin([0, 1])).any():
        raise ValueError("Churn contains invalid values")
    if (~(final["Returns"].isin([0, 1]) | final["Returns"].isna())).any():
        raise ValueError("Returns contains invalid values")
    return {
        "rows": int(len(final)),
        "customers": int(final["Customer ID"].nunique()),
        "returns_missing": int(final["Returns"].isna().sum()),
        "returns_observed": int(final["Returns"].notna().sum()),
        "return_rate_observed": float(final["Returns"].mean()),
        "amount_mismatches": int((final["Total Purchase Amount"] !=
                                 final["Product Price"] * final["Quantity"]).sum()),
        "columns": int(len(final.columns)),
    }


def check_requirements() -> None:
    import importlib.util
    missing = [m for m in ("numpy", "pandas", "scipy", "matplotlib", "sklearn")
               if importlib.util.find_spec(m) is None]
    if missing:
        raise RuntimeError(f"Missing dependencies {missing}; run pip install -r requirements.txt")


def execute(stage: str) -> None:
    name = f"src/pipeline/{stage}"
    print(f"\n>>> Running {name}", flush=True)
    env = os.environ.copy()
    env["MPLBACKEND"] = "Agg"
    env["PYTHONUNBUFFERED"] = "1"
    subprocess.run([sys.executable, name], cwd=REPO_ROOT, env=env, check=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("cleaning", "eda"), default="cleaning",
                        help="eda includes charts (05); do not use as proof of Phase 3 metric validity")
    parser.add_argument("--dry-run", action="store_true", help="Show stages without changing files")
    parser.add_argument("--validate-only", action="store_true", help="Check existing clean output")
    args = parser.parse_args(argv)

    steps = list(STAGES) + (["05_explanatory_visualization.py"] if args.stage == "eda" else [])
    print(f"Project: {REPO_ROOT}\nPython: {sys.version.split()[0]}\nSource: {RAW_DATA_PATH}")
    if args.dry_run:
        for step in steps:
            print(" -", step)
        print("Raw input available:", RAW_DATA_PATH.is_file())
        return 0
    if not RAW_DATA_PATH.is_file():
        raise FileNotFoundError(f"Missing raw CSV: {RAW_DATA_PATH}. See docs/phase2_runbook.md")
    check_requirements()
    raw_hash_before = sha256_file(RAW_DATA_PATH)
    raw = load_ecommerce_raw()
    if not args.validate_only:
        for stage in STAGES:
            execute(stage)
    if not CLEAN_PATH.exists():
        raise FileNotFoundError(f"Missing pipeline final output: {CLEAN_PATH}")
    cleaned = pd.read_csv(CLEAN_PATH)
    result = validate_cleaned(raw, cleaned)
    if sha256_file(RAW_DATA_PATH) != raw_hash_before:
        raise RuntimeError("The raw input changed during the pipeline run")
    if not args.validate_only and args.stage == "eda":
        print("Generating Phase 3 validated EDA charts (01–05) from cleaned CSV")
        execute("05_explanatory_visualization.py")
    manifest = {
        "phase": 2,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "python_version": sys.version.split()[0],
        "raw_path": str(RAW_DATA_PATH.relative_to(REPO_ROOT)),
        "raw_sha256": raw_hash_before,
        "clean_path": str(CLEAN_PATH.relative_to(REPO_ROOT)),
        "clean_sha256": sha256_file(CLEAN_PATH),
        "validated": result,
        "completed_stage": args.stage if not args.validate_only else "validate-only",
        "note": "Only a real Kaggle raw CSV can establish full end-to-end reproducibility."
    }
    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2, ensure_ascii=False)+"\n", encoding="utf-8")
    print("\nPASS: raw -> cleaned integrity gates:")
    print(json.dumps(result, indent=2, ensure_ascii=False))
    print("Manifest:", MANIFEST_PATH)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
