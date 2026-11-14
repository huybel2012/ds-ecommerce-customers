"""Phase 3: tests use a synthetic fixture; no Kaggle raw, no Streamlit server."""
from __future__ import annotations

from pathlib import Path
import importlib
import subprocess
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd

from src.eda_metrics import (
    age_gender_median, category_totals, churn_segment_matrix, load_cleaned,
    monthly_amount, summary_metrics, validate_eda_data,
)

VIS = importlib.import_module("src.pipeline.05_explanatory_visualization")


def fixture() -> pd.DataFrame:
    # Customer 1: 3 events (Churn=1), Customer 2: 1 event (Churn=0).
    # Row-weighted churn would incorrectly be 75%; customer-weighted = 50%.
    rows = [
        (1, "2020-01-01 10:00:00", "Home", 10, 2, 200, "Cash", np.nan, 1, 28, "Female", 1),
        (1, "2020-01-02 10:00:00", "Home", 10, 2, 210, "Cash", 1., 0, 28, "Female", 1),
        (1, "2020-01-03 10:00:00", "Books", 5, 1, 100, "PayPal", 1., 0, 28, "Female", 1),
        (2, "2020-02-15 10:00:00", "Home", 10, 1, 120, "Cash", 0., 0, 47, "Male", 0),
    ]
    cols = ["Customer ID", "Purchase Date", "Product Category", "Product Price", "Quantity",
            "Total Purchase Amount", "Payment Method", "Returns", "Returns_was_missing",
            "Age", "Gender", "Churn"]
    return pd.DataFrame(rows, columns=cols)


class Phase3MetricContractTests(unittest.TestCase):
    def setUp(self):
        self.frame = fixture()
        self.df = validate_eda_data(self.frame)

    def test_requires_the_validated_clean_path_not_legacy(self):
        with tempfile.TemporaryDirectory() as temp:
            with self.assertRaisesRegex(FileNotFoundError, "Phase 2 validated CSV"):
                load_cleaned(Path(temp) / "ecommerce_customer_clean.csv")

    def test_requires_missing_indicator(self):
        with self.assertRaisesRegex(ValueError, "Returns_was_missing"):
            validate_eda_data(self.frame.drop(columns=["Returns_was_missing"]))

    def test_rejects_incorrect_missing_flag(self):
        broken = self.frame.copy()
        broken.loc[0, "Returns_was_missing"] = 0
        with self.assertRaisesRegex(ValueError, "Returns_was_missing"):
            validate_eda_data(broken)

    def test_churn_is_customer_level(self):
        metrics = summary_metrics(self.df)
        self.assertEqual(metrics["customers"], 2)
        self.assertEqual(metrics["transactions"], 4)
        self.assertAlmostEqual(metrics["churn_rate_customer"], .5)
        self.assertAlmostEqual(self.df["Churn"].mean(), .75)

    def test_returns_rate_ignores_nulls(self):
        metrics = summary_metrics(self.df)
        self.assertEqual(metrics["returns_missing"], 1)
        self.assertEqual(metrics["returns_observed"], 3)
        self.assertAlmostEqual(metrics["return_rate_observed"], 2/3)
        self.assertEqual(metrics["line_item_value_total"], 55)

    def test_segmentation_avoids_overweighting_repeated_purchases(self):
        matrix, counts = churn_segment_matrix(self.df)
        self.assertAlmostEqual(matrix.loc["Home", "Cash"], .5)
        self.assertEqual(counts.loc["Home", "Cash"], 2)

    def test_monthly_last_partial_not_marked_complete(self):
        monthly = monthly_amount(self.df)
        self.assertEqual(len(monthly), 2)
        self.assertTrue(bool(monthly.iloc[0]["complete_month"]))
        self.assertFalse(bool(monthly.iloc[1]["complete_month"]))
        self.assertEqual(monthly.iloc[0]["transactions"], 3)

    def test_dynamic_category_ranking(self):
        self.assertEqual(category_totals(self.df).index[0], "Home")
        self.assertEqual(category_totals(self.df).iloc[0], 530)

    def test_customer_churn_must_be_consistent(self):
        broken = self.frame.copy()
        broken.loc[1, "Churn"] = 0
        with self.assertRaisesRegex(ValueError, "constant per customer"):
            validate_eda_data(broken)

    def test_validation_does_not_mutate_source(self):
        original = self.frame.copy(deep=True)
        self.assertIsNot(original, self.df)
        pd.testing.assert_frame_equal(self.frame, original)

    def test_figures_have_zero_baseline_and_dynamic_title(self):
        fig = VIS.plot_category_revenue(self.df)
        try:
            self.assertIn("Home", fig.axes[0].get_title())
            self.assertNotIn("Electronics và Clothing", fig.axes[0].get_title())
            self.assertEqual(fig.axes[0].get_xlim()[0], 0)
            self.assertNotIn("USD", fig.axes[0].get_xlabel())
        finally:
            VIS.plt.close(fig)

    def test_exports_only_five_pngs_preserves_w9_w10(self):
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            marker = directory / "06_residual_diagnostics.png"
            marker.write_bytes(b"keep original W10 plot")
            names = VIS.export_figures(self.df, directory)
            self.assertEqual(len(names), 5)
            self.assertEqual(marker.read_bytes(), b"keep original W10 plot")
            self.assertTrue(all(name.exists() and name.stat().st_size > 0 for name in names))

    def test_partial_month_not_connected_as_full_month(self):
        fig = VIS.plot_monthly_revenue_trend(self.df)
        try:
            full_month_trace = fig.axes[0].get_lines()[0]
            self.assertEqual(len(full_month_trace.get_xdata()), 1)
        finally:
            VIS.plt.close(fig)

    def test_script_path_mode_supported_by_phase2_runner(self):
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            csv = directory / "clean.csv"
            self.frame.to_csv(csv, index=False)
            root = Path(__file__).resolve().parents[1]
            proc = subprocess.run(
                [sys.executable, str(root / "src/pipeline/05_explanatory_visualization.py"),
                 "--input", str(csv), "--output-dir", str(directory / "figs")],
                cwd=root, capture_output=True, text=True,
            )
            self.assertEqual(proc.returncode, 0, proc.stderr)
            self.assertTrue((directory / "figs/01_category_revenue.png").exists())

    def test_cli_export_and_manifest_on_fixture(self):
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            input_csv = directory / "clean.csv"
            self.frame.to_csv(input_csv, index=False)
            self.assertEqual(VIS.main(["--input", str(input_csv),
                                       "--output-dir", str(directory / "figs")]), 0)
            self.assertTrue((directory / "figs" / "phase3_visualization_manifest.json").exists())


if __name__ == "__main__":
    unittest.main()
