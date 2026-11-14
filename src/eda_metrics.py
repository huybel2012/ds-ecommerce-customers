"""Phase 3 metric contract: one validated clean source for EDA and dashboard.

Do not infer currency or real business revenue from synthetic Kaggle data.
Customer churn and segment churn use customers, never transaction-weighted rows.
"""
from __future__ import annotations

from calendar import monthrange
from pathlib import Path

import numpy as np
import pandas as pd

from src.utils import REPO_ROOT

CLEAN_PATH = REPO_ROOT / "data" / "processed" / "ecommerce_customer_clean.csv"
AMOUNT = "Total Purchase Amount"
REQUIRED_COLUMNS = (
    "Customer ID", "Purchase Date", "Product Category", "Product Price",
    "Quantity", AMOUNT, "Payment Method", "Returns", "Returns_was_missing",
    "Age", "Gender", "Churn",
)


def load_cleaned(path: Path = CLEAN_PATH) -> pd.DataFrame:
    """Fail closed if the Phase 2 output is absent; never silently use legacy CSV."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(
            f"Missing Phase 2 validated CSV: {path}. "
            "Run 'python -m src.run_pipeline' with the Kaggle raw CSV first."
        )
    return validate_eda_data(pd.read_csv(path))


def validate_eda_data(data: pd.DataFrame) -> pd.DataFrame:
    """Return a validated copy, not modifying caller's dataframe."""
    missing = set(REQUIRED_COLUMNS) - set(data.columns)
    if missing:
        raise ValueError(f"Phase 3 requires Phase 2 clean schema, missing: {sorted(missing)}")
    if data.empty:
        raise ValueError("Dataset is empty")
    df = data.copy()
    df["Purchase Date"] = pd.to_datetime(df["Purchase Date"], errors="raise")
    if df[list(REQUIRED_COLUMNS)].drop(columns=["Returns"]).isna().any().any():
        raise ValueError("Unexpected missing values outside Returns")
    if df.duplicated(subset=["Customer ID", "Purchase Date"]).any():
        raise ValueError("Duplicated customer-timestamp event key")
    returns = pd.to_numeric(df["Returns"], errors="raise")
    flags = pd.to_numeric(df["Returns_was_missing"], errors="raise")
    if not returns.dropna().isin([0, 1]).all():
        raise ValueError("Returns must be 0, 1 or missing")
    if not flags.isin([0, 1]).all() or not flags.eq(returns.isna().astype(int)).all():
        raise ValueError("Returns_was_missing does not agree with null Returns")
    if not df["Churn"].isin([0, 1]).all():
        raise ValueError("Invalid Churn label")
    if df.groupby("Customer ID")["Churn"].nunique(dropna=False).max() != 1:
        raise ValueError("Churn must be constant per customer in this dataset")
    if (df["Quantity"] <= 0).any() or (df["Product Price"] < 0).any() or (df[AMOUNT] < 0).any():
        raise ValueError("Invalid price, quantity or recorded amount")
    return df


def customer_table(df: pd.DataFrame) -> pd.DataFrame:
    return df.drop_duplicates(subset=["Customer ID"])[["Customer ID", "Churn", "Age", "Gender"]]


def summary_metrics(df: pd.DataFrame) -> dict:
    customers = customer_table(df)
    observed = df["Returns"].notna()
    return {
        "transactions": int(len(df)),
        "customers": int(len(customers)),
        "reported_amount_total": float(df[AMOUNT].sum()),
        "line_item_value_total": float((df["Product Price"] * df["Quantity"]).sum()),
        "amount_mismatch_count": int((df[AMOUNT] != df["Product Price"] * df["Quantity"]).sum()),
        "reported_amount_median": float(df[AMOUNT].median()),
        "churn_customers": int(customers["Churn"].sum()),
        "churn_rate_customer": float(customers["Churn"].mean()),
        "returns_observed": int(observed.sum()),
        "returns_missing": int((~observed).sum()),
        "return_rate_observed": float(df.loc[observed, "Returns"].mean()) if observed.any() else np.nan,
        "date_min": df["Purchase Date"].min(),
        "date_max": df["Purchase Date"].max(),
    }


def category_totals(df: pd.DataFrame) -> pd.Series:
    return df.groupby("Product Category")[AMOUNT].sum().sort_values(ascending=False)


def age_gender_median(df: pd.DataFrame) -> pd.DataFrame:
    bands = pd.cut(
        df["Age"], bins=[17, 25, 35, 45, 55, 65, np.inf],
        labels=["18–25", "26–35", "36–45", "46–55", "56–65", "66+"],
    )
    return (df.assign(age_band=bands).groupby(["age_band", "Gender"], observed=True)[AMOUNT]
            .median().unstack("Gender"))


def churn_segment_matrix(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Deduplicate each customer *within* each category-payment segment.

    Customers can appear in multiple segments; these cells are not independent.
    """
    subgroup = df[["Customer ID", "Product Category", "Payment Method", "Churn"]].drop_duplicates(
        ["Customer ID", "Product Category", "Payment Method"]
    )
    rates = subgroup.pivot_table(
        index="Product Category", columns="Payment Method", values="Churn",
        aggfunc="mean", observed=True,
    )
    counts = subgroup.pivot_table(
        index="Product Category", columns="Payment Method", values="Customer ID",
        aggfunc="count", observed=True,
    )
    return rates.sort_index().sort_index(axis=1), counts.sort_index().sort_index(axis=1)


def monthly_amount(df: pd.DataFrame) -> pd.DataFrame:
    """Month completeness based on available date range, not statistical guesses."""
    dates = df["Purchase Date"]
    monthly = (
        df.assign(Month=dates.dt.to_period("M").dt.to_timestamp())
        .groupby("Month", sort=True)[AMOUNT]
        .agg(["sum", "size"])
        .rename(columns={"sum": "amount_total", "size": "transactions"})
        .reset_index()
    )
    earliest, latest = dates.min(), dates.max()
    monthly["complete_month"] = True
    if earliest.day != 1:
        monthly.loc[monthly["Month"] == earliest.to_period("M").to_timestamp(), "complete_month"] = False
    if latest.day != monthrange(latest.year, latest.month)[1]:
        monthly.loc[monthly["Month"] == latest.to_period("M").to_timestamp(), "complete_month"] = False
    return monthly
