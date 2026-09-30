"""
Reusable utilities for ds-ecommerce-customers.

Bám sát Week 1-3:
- data/raw luôn bất biến.
- audit trước khi cleaning.
- phát hiện missing thật và missing giả dạng.
- chỉ xử lý missing khi có reasoning rõ ràng.
- ghi lại mọi bước cleaning vào reports/cleaning_log.csv.

Lưu ý:
- group_median_impute() phù hợp cho cleaning/EDA khi có bằng chứng hướng MAR.
- Khi train model, không impute trên toàn bộ dataset trước train/test split.
  Hãy dùng sklearn Pipeline để tránh data leakage.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Iterable

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

RAW_DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "ecommerce_customer_data_custom_ratios.csv"
)

PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
CLEANING_LOG_PATH = PROJECT_ROOT / "reports" / "cleaning_log.csv"

EXPECTED_COLUMNS = [
    "Customer ID",
    "Purchase Date",
    "Product Category",
    "Product Price",
    "Quantity",
    "Total Purchase Amount",
    "Payment Method",
    "Customer Age",
    "Returns",
    "Customer Name",
    "Age",
    "Gender",
    "Churn",
]

BINARY_COLUMNS = ["Returns", "Churn"]

def load_ecommerce_raw(
    path: str | Path = RAW_DATA_PATH,
    *,
    na_values: Iterable[object] | None = None,
    **read_csv_kwargs,
) -> pd.DataFrame:
    """
    Đọc raw dataset mà không sửa file gốc.

    Không tự gán sentinel values. Hãy inspect dữ liệu trước, sau đó
    truyền na_values nếu đã xác định được missing giả dạng.
    """
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Không tìm thấy dataset: {path}")

    return pd.read_csv(
        path,
        na_values=na_values,
        **read_csv_kwargs,
    )



def audit(df: pd.DataFrame) -> pd.DataFrame:
    """
    One-glance quality report for any DataFrame.

    Trả về:
    - dtype
    - số missing
    - % missing
    - số giá trị unique
    - một giá trị mẫu
    """
    if df.empty:
        sample = pd.Series(
            [pd.NA] * len(df.columns),
            index=df.columns,
        )
    else:
        sample = df.iloc[0]

    return pd.DataFrame({
        "dtype": df.dtypes,
        "n_missing": df.isna().sum(),
        "pct_missing": (df.isna().mean() * 100).round(1),
        "n_unique": df.nunique(),
        "sample": sample,
    })


def validate_schema(df: pd.DataFrame) -> pd.DataFrame:
    """
    Kiểm tra dataset có đủ các cột được kỳ vọng hay không.
    Không chỉnh sửa dữ liệu.
    """
    return pd.DataFrame({
        "column": EXPECTED_COLUMNS,
        "present": [
            column in df.columns
            for column in EXPECTED_COLUMNS
        ],
        "dtype": [
            str(df[column].dtype) if column in df.columns else None
            for column in EXPECTED_COLUMNS
        ],
    })


def unexpected_columns(df: pd.DataFrame) -> list[str]:
    """Liệt kê các cột có trong dữ liệu nhưng không nằm trong schema kỳ vọng."""
    expected = set(EXPECTED_COLUMNS)
    return [
        column
        for column in df.columns
        if column not in expected
    ]


def inspect_text_values(
    df: pd.DataFrame,
    top_n: int = 10,
    columns: Iterable[str] | None = None,
) -> None:
    """
    In các giá trị phổ biến trong cột text/categorical.

    Dùng để săn missing giả dạng như:
    "", " ", "-", "?", "N/A", "NULL", "Unknown", ...
    """
    if top_n <= 0:
        raise ValueError("top_n phải lớn hơn 0.")

    if columns is None:
        columns = df.select_dtypes(
            include=["object", "string", "category"]
        ).columns

    for column in columns:
        if column not in df.columns:
            print(f"[WARN] Không có cột: {column}")
            continue

        print(f"\n--- {column} ---")
        print(
            df[column]
            .value_counts(dropna=False)
            .head(top_n)
        )


def missing_by_group(
    df: pd.DataFrame,
    target: str,
    group: str,
) -> pd.DataFrame:
    """
    Tính tỷ lệ missing của target theo group.

    Hỗ trợ reasoning về MAR.
    Hàm này KHÔNG tự kết luận MCAR/MAR/MNAR.
    """
    if target not in df.columns:
        raise KeyError(f"Không có cột target: {target}")

    if group not in df.columns:
        raise KeyError(f"Không có cột group: {group}")

    temp = pd.DataFrame({
        group: df[group],
        "_is_missing": df[target].isna(),
    })

    result = (
        temp
        .groupby(group, dropna=False)["_is_missing"]
        .agg(
            n_rows="size",
            n_missing="sum",
            pct_missing="mean",
        )
        .reset_index()
    )

    result["pct_missing"] = (
        result["pct_missing"] * 100
    ).round(1)

    return result.sort_values(
        "pct_missing",
        ascending=False,
    ).reset_index(drop=True)


def binary_domain_report(
    df: pd.DataFrame,
    columns: Iterable[str] = BINARY_COLUMNS,
) -> pd.DataFrame:
    """
    Kiểm tra validity cho các cột nhị phân như Returns và Churn.

    Giá trị hợp lệ kỳ vọng: 0, 1 hoặc missing.
    Không chỉnh sửa dữ liệu.
    """
    rows = []

    for column in columns:
        if column not in df.columns:
            continue

        non_missing = df[column].dropna()
        invalid_mask = ~non_missing.isin([0, 1])

        rows.append({
            "column": column,
            "n_non_missing": int(len(non_missing)),
            "n_invalid": int(invalid_mask.sum()),
            "invalid_values": (
                non_missing[invalid_mask]
                .drop_duplicates()
                .tolist()
            ),
        })

    return pd.DataFrame(rows)



def add_missing_indicator(
    df: pd.DataFrame,
    column: str,
) -> pd.DataFrame:
    """
    Thêm cột <column>_was_missing trước khi impute.

    Hữu ích khi bản thân việc missing có thể mang thông tin.
    """
    if column not in df.columns:
        raise KeyError(f"Không có cột: {column}")

    result = df.copy()

    result[f"{column}_was_missing"] = (
        result[column]
        .isna()
        .astype("int8")
    )

    return result


def group_median_impute(
    df: pd.DataFrame,
    target: str,
    groups: str | list[str],
) -> pd.DataFrame:
    """
    Fill missing numeric target bằng median trong group.

    Chỉ dùng khi:
    - target là numeric.
    - có lý do tin rằng missingness liên quan tới biến group quan sát được.
    - mục đích là cleaning/EDA.

    KHÔNG dùng trực tiếp trên full dataset trước train/test split khi modeling.
    """
    if target not in df.columns:
        raise KeyError(f"Không có cột target: {target}")

    group_list = [groups] if isinstance(groups, str) else list(groups)

    if not group_list:
        raise ValueError("groups không được rỗng.")

    missing_groups = [
        column
        for column in group_list
        if column not in df.columns
    ]

    if missing_groups:
        raise KeyError(f"Không có cột group: {missing_groups}")

    if not pd.api.types.is_numeric_dtype(df[target]):
        raise TypeError(
            f"{target} không phải cột numeric, không thể dùng median."
        )

    result = df.copy()

    result[target] = (
        result
        .groupby(group_list, dropna=False)[target]
        .transform(
            lambda series:
            series.fillna(series.median())
        )
    )

    remaining = int(result[target].isna().sum())

    print(
        remaining,
        f"rows still missing in '{target}' after group median fill",
    )

    return result


def log_cleaning_step(
    step: str,
    reason: str,
    rows_before: int,
    rows_after: int,
    affected_rows: int,
    *,
    details: str = "",
    log_path: str | Path = CLEANING_LOG_PATH,
) -> None:
    """
    Ghi một bước xử lý vào reports/cleaning_log.csv.

    affected_rows bắt buộc truyền rõ ràng để tránh ghi sai.
    Ví dụ: impute 100 giá trị nhưng số dòng trước/sau không đổi.
    """
    if rows_before < 0 or rows_after < 0 or affected_rows < 0:
        raise ValueError(
            "rows_before, rows_after và affected_rows phải >= 0."
        )

    log_path = Path(log_path)

    log_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    row = pd.DataFrame([{
        "timestamp": datetime.now().isoformat(
            timespec="seconds"
        ),
        "step": step,
        "reason": reason,
        "rows_before": int(rows_before),
        "rows_after": int(rows_after),
        "affected_rows": int(affected_rows),
        "details": details,
    }])

    row.to_csv(
        log_path,
        mode="a",
        header=not log_path.exists(),
        index=False,
        encoding="utf-8-sig",
    )


__all__ = [
    "PROJECT_ROOT",
    "RAW_DATA_PATH",
    "PROCESSED_DIR",
    "CLEANING_LOG_PATH",
    "EXPECTED_COLUMNS",
    "BINARY_COLUMNS",
    "load_ecommerce_raw",
    "audit",
    "validate_schema",
    "unexpected_columns",
    "inspect_text_values",
    "missing_by_group",
    "binary_domain_report",
    "add_missing_indicator",
    "group_median_impute",
    "log_cleaning_step",
]
