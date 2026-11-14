"""Fast, fully local tests for Phase 1 (no dataset download or pyarrow required)."""
import unittest

import numpy as np
import pandas as pd

from src.phase1_data_integrity import assess, restore_returns


def make_data():
    source = pd.DataFrame({
        "Customer ID": [101, 101, 202],
        "Purchase Date": ["2020-01-01 10:00:00", "2020-02-01 10:00:00", "2020-01-01 12:00:00"],
        "Product Category": ["Books", "Books", "Home"],
        "Product Price": [20, 50, 10],
        "Quantity": [2, 1, 1],
        "Total Purchase Amount": [150, 200, 40],
        "Payment Method": ["Cash", "Cash", "PayPal"],
        "Returns": [np.nan, 1., 0.],
        "Customer Name": ["Fake Person", "Fake Person", "Sample Name"],
        "Age": [22, 22, 31],
        "Gender": ["Male", "Male", "Female"],
        "Churn": [0, 0, 1],
        "Customer Age": [22, 22, 31],
    })
    legacy = source.drop(columns=["Customer Age"]).copy()
    legacy["Returns"] = legacy["Returns"].fillna(0).astype(int)
    return legacy, source


class Phase1IntegrityTest(unittest.TestCase):
    def test_restores_missing_without_using_row_order(self):
        legacy, source = make_data()
        shuffled = source.iloc[[2, 0, 1]].reset_index(drop=True)
        recovered = restore_returns(legacy, shuffled)
        self.assertEqual(len(recovered), 3)
        self.assertEqual(recovered["Returns_was_missing"].tolist(), [1, 0, 0])
        self.assertTrue(pd.isna(recovered.loc[0, "Returns"]))
        self.assertEqual(recovered["line_item_value_calc"].tolist(), [40, 50, 10])
        self.assertNotIn("Customer Name", recovered.columns)
        self.assertTrue(legacy["Returns"].eq([0, 1, 0]).all())

    def test_rejects_duplicate_event_key(self):
        legacy, source = make_data()
        source.loc[1, "Purchase Date"] = source.loc[0, "Purchase Date"]
        with self.assertRaisesRegex(ValueError, "duplicated business key"):
            restore_returns(legacy, source)

    def test_rejects_mutated_price(self):
        legacy, source = make_data()
        source.loc[1, "Product Price"] = 9999
        with self.assertRaisesRegex(ValueError, "Source mismatch"):
            restore_returns(legacy, source)

    def test_rejects_mutated_observed_return(self):
        legacy, source = make_data()
        legacy.loc[1, "Returns"] = 0
        with self.assertRaisesRegex(ValueError, "changed some observed"):
            restore_returns(legacy, source)

    def test_metrics_are_customer_level(self):
        legacy, source = make_data()
        fixed = restore_returns(legacy, source)
        report = assess(fixed)
        self.assertEqual(report["rows"], 3)
        self.assertEqual(report["unique_customers"], 2)
        self.assertEqual(report["returns_missing_count"], 1)
        self.assertEqual(report["returns_observed_count"], 2)
        self.assertEqual(report["return_rate_among_observed"], 0.5)
        self.assertEqual(report["churn_rate_customer_level"], 0.5)
        self.assertEqual(report["amount_equal_count"], 0)


if __name__ == "__main__":
    unittest.main()
