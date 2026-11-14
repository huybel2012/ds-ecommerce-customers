"""Phase 2 tests: no Kaggle download and no writes to the user's repository."""
from __future__ import annotations

from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd

from src.run_pipeline import validate_cleaned
from src.utils import (EXPECTED_COLUMNS, binary_domain_report,
                       load_ecommerce_raw, unexpected_columns, validate_schema)


def raw_fixture():
    # Same 13-column contract as Kaggle's custom_ratios CSV, intentionally tiny.
    return pd.DataFrame({
        "Customer ID": [1, 1, 2],
        "Purchase Date": ["2020-01-01 10:00:00", "2020-01-02 10:00:00", "2020-01-03 10:00:00"],
        "Product Category": ["Books", "Books", "Home"],
        "Product Price": [10, 30, 20],
        "Quantity": [2, 1, 3],
        "Total Purchase Amount": [100, 90, 200],
        "Payment Method": ["Cash", "Cash", "PayPal"],
        "Customer Age": [22, 22, 34],
        "Returns": [np.nan, 1.0, 0.0],
        "Customer Name": ["Synthetic One", "Synthetic One", "Synthetic Two"],
        "Age": [22, 22, 34],
        "Gender": ["Female", "Female", "Male"],
        "Churn": [1, 1, 0],
    })


def cleaned_fixture(raw):
    data = raw.drop(columns=["Customer Age"]).copy()
    data["Returns_was_missing"] = raw["Returns"].isna().astype("int8")
    return data


class RawContractTests(unittest.TestCase):
    def test_expected_schema_is_complete(self):
        frame = raw_fixture()
        self.assertEqual(len(EXPECTED_COLUMNS), 13)
        self.assertEqual(unexpected_columns(frame), [])
        report = validate_schema(frame)
        self.assertTrue(report["present"].all())

    def test_schema_reports_missing_and_unexpected_columns(self):
        frame = raw_fixture().drop(columns=["Customer Age"])
        frame["extra"] = [1, 2, 3]
        report = validate_schema(frame)
        self.assertFalse(report.loc[report.column == "Customer Age", "present"].item())
        self.assertEqual(unexpected_columns(frame), ["extra"])

    def test_loads_exact_raw_path(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "ecommerce_customer_data_custom_ratios.csv"
            raw_fixture().to_csv(path, index=False)
            loaded = load_ecommerce_raw(path)
            self.assertEqual(loaded.shape, (3, 13))
            self.assertEqual(int(loaded.Returns.isna().sum()), 1)

    def test_missing_raw_fails_instead_of_falling_back(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(FileNotFoundError, "Required raw dataset is missing"):
                load_ecommerce_raw(Path(tmp) / "does-not-exist.csv")

    def test_missing_raw_column_fails_fast(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "raw.csv"
            raw_fixture().drop(columns=["Customer Age"]).to_csv(path, index=False)
            with self.assertRaisesRegex(ValueError, "Raw schema is missing columns"):
                load_ecommerce_raw(path)

    def test_binary_domain_audit(self):
        frame = raw_fixture()
        frame.loc[2, "Returns"] = 9
        result = binary_domain_report(frame)
        self.assertEqual(result.loc[result.column == "Returns", "n_invalid"].item(), 1)
        self.assertEqual(result.loc[result.column == "Returns", "n_missing"].item(), 1)


class CleanedValidationTests(unittest.TestCase):
    def test_valid_missing_flag_and_source_preserved(self):
        source = raw_fixture()
        result = validate_cleaned(source, cleaned_fixture(source))
        self.assertEqual(result["rows"], 3)
        self.assertEqual(result["customers"], 2)
        self.assertEqual(result["returns_missing"], 1)
        self.assertAlmostEqual(result["return_rate_observed"], 0.5)

    def test_reject_mutated_amount(self):
        source = raw_fixture()
        final = cleaned_fixture(source)
        final.loc[0, "Total Purchase Amount"] = 500
        with self.assertRaisesRegex(ValueError, "Total Purchase Amount"):
            validate_cleaned(source, final)

    def test_reject_imputed_returns(self):
        source = raw_fixture()
        final = cleaned_fixture(source)
        final["Returns"] = final["Returns"].fillna(0)
        with self.assertRaisesRegex(ValueError, "Returns changed"):
            validate_cleaned(source, final)

    def test_reject_corrupted_missing_indicator(self):
        source = raw_fixture()
        final = cleaned_fixture(source)
        final.loc[0, "Returns_was_missing"] = 0
        with self.assertRaisesRegex(ValueError, "Returns_was_missing"):
            validate_cleaned(source, final)

    def test_reject_duplicate_business_key(self):
        source = raw_fixture()
        final = cleaned_fixture(source)
        final.loc[1, "Purchase Date"] = final.loc[0, "Purchase Date"]
        with self.assertRaisesRegex(ValueError, "duplicated customer-timestamp"):
            validate_cleaned(source, final)

    def test_dry_run_from_repo_root(self):
        with tempfile.TemporaryDirectory() as tmp:
            result = subprocess.run(
                [sys.executable, "-m", "src.run_pipeline", "--dry-run"],
                cwd=Path(__file__).resolve().parents[1],
                capture_output=True, text=True,
            )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("04_finalize_cleaning.py", result.stdout)


if __name__ == "__main__":
    unittest.main()
