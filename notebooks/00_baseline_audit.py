"""
Audit nền cho dữ liệu thương mại điện tử (nhiệm vụ #1-#15).

Mục tiêu của file này là đọc dữ liệu thô và kiểm tra cấu trúc, missing,
trùng lặp, miền giá trị và các dấu hiệu missing ẩn mà không sửa dữ liệu.
"""

from pathlib import Path
import sys

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils import (
    EXPECTED_COLUMNS,
    audit,
    binary_domain_report,
    load_ecommerce_raw,
    unexpected_columns,
    validate_schema,
)

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 180)


def section(title: str) -> None:
    print(f"\n=== {title} ===")


# Luôn audit trên dữ liệu thô để tránh kết quả bị ảnh hưởng bởi cleaning trước đó.
df = load_ecommerce_raw()

section("Tổng quan dữ liệu")
print(f"Kích thước: {df.shape[0]:,} dòng x {df.shape[1]} cột")
print("\n5 dòng đầu:")
print(df.head())

print("\nThông tin DataFrame:")
df.info()

print("\nKiểu dữ liệu:")
print(df.dtypes)


# Báo cáo audit cơ bản dùng lại cho missing và số lượng giá trị duy nhất.
audit_report = audit(df)

section("Giá trị thiếu")
missing_report = (
    audit_report[["n_missing", "pct_missing"]]
    .sort_values(["n_missing", "pct_missing"], ascending=False)
)
print(missing_report)

total_missing = int(df.isna().sum().sum())
total_cells = df.size
print(f"\nTổng số ô thiếu: {total_missing:,}")
print(f"Tỷ lệ thiếu toàn bộ dữ liệu: {total_missing / total_cells * 100:.4f}%")


section("Số giá trị duy nhất")
print(audit_report[["n_unique"]].sort_values("n_unique"))


section("Bản ghi trùng lặp hoàn toàn")
duplicate_count = int(df.duplicated().sum())
print(f"Số dòng trùng: {duplicate_count:,}")
print(f"Tỷ lệ trùng  : {duplicate_count / len(df) * 100:.4f}%")

if duplicate_count:
    print("\nVí dụ các dòng trùng:")
    print(df[df.duplicated(keep=False)].head(10))


section("Kiểm tra cấu trúc dữ liệu")
schema_report = validate_schema(df)
print(schema_report.to_string(index=False))

missing_columns = [c for c in EXPECTED_COLUMNS if c not in df.columns]
extra_columns = unexpected_columns(df)

print(f"\nSố cột kỳ vọng     : {len(EXPECTED_COLUMNS)}")
print(f"Số cột thực tế     : {len(df.columns)}")
print(f"Cột bị thiếu        : {missing_columns or 'Không có'}")
print(f"Cột ngoài dự kiến   : {extra_columns or 'Không có'}")


section("Phân bố các biến phân loại")
categorical_columns = [
    "Product Category",
    "Payment Method",
    "Gender",
    "Returns",
    "Churn",
]

for column in categorical_columns:
    if column not in df.columns:
        continue

    counts = df[column].value_counts(dropna=False)
    percentages = (
        df[column]
        .value_counts(dropna=False, normalize=True)
        .mul(100)
        .round(2)
    )

    print(f"\n{column}")
    print(pd.DataFrame({"count": counts, "pct": percentages}))


section("Kiểm tra hợp lệ biến nhị phân")
print(
    binary_domain_report(
        df,
        columns=["Returns", "Churn"],
    ).to_string(index=False)
)


section("Kiểm tra hợp lệ biến số")
# Các rule dưới đây là hard validity rules, tách biệt với outlier thống kê.
numeric_rules = {
    "Customer ID": ("Customer ID > 0", lambda s: s <= 0),
    "Product Price": ("Product Price >= 0", lambda s: s < 0),
    "Quantity": ("Quantity > 0", lambda s: s <= 0),
    "Total Purchase Amount": (
        "Total Purchase Amount >= 0",
        lambda s: s < 0,
    ),
    "Customer Age": (
        "0 <= Customer Age <= 120",
        lambda s: (s < 0) | (s > 120),
    ),
    "Age": (
        "0 <= Age <= 120",
        lambda s: (s < 0) | (s > 120),
    ),
}

validity_rows = []

for column, (rule_text, invalid_rule) in numeric_rules.items():
    if column not in df.columns:
        continue

    series = df[column]
    invalid_mask = invalid_rule(series) & series.notna()
    n_invalid = int(invalid_mask.sum())

    validity_rows.append(
        {
            "column": column,
            "rule": rule_text,
            "min": series.min(),
            "max": series.max(),
            "n_invalid": n_invalid,
            "pct_invalid": round(n_invalid / len(df) * 100, 4),
        }
    )

numeric_validity_report = pd.DataFrame(validity_rows)
print(numeric_validity_report.to_string(index=False))

for column, (_, invalid_rule) in numeric_rules.items():
    if column not in df.columns:
        continue

    invalid_mask = invalid_rule(df[column]) & df[column].notna()
    if invalid_mask.any():
        print(f"\nVí dụ giá trị không hợp lệ - {column}")
        print(df.loc[invalid_mask, [column]].head(10))


section("Đối chiếu Customer Age và Age")
if {"Customer Age", "Age"}.issubset(df.columns):
    same_mask = (
        df["Customer Age"].eq(df["Age"])
        | (df["Customer Age"].isna() & df["Age"].isna())
    )
    mismatch_mask = ~same_mask
    mismatch_count = int(mismatch_mask.sum())
    age_difference = df["Customer Age"] - df["Age"]

    print(f"Số dòng đối chiếu : {len(df):,}")
    print(f"Số dòng giống nhau: {int(same_mask.sum()):,}")
    print(f"Số dòng khác nhau : {mismatch_count:,}")
    print(f"Tỷ lệ khác nhau   : {mismatch_count / len(df) * 100:.4f}%")
    print(f"Chênh lệch nhỏ nhất: {age_difference.min()}")
    print(f"Chênh lệch lớn nhất: {age_difference.max()}")

    if mismatch_count:
        print("\nVí dụ các dòng không khớp:")
        print(
            df.loc[
                mismatch_mask,
                ["Customer ID", "Customer Name", "Customer Age", "Age"],
            ].head(10)
        )


section("Giá trị thiếu ẩn trong dữ liệu văn bản")
# Chuẩn hóa chuỗi trước khi dò các token thường dùng để biểu diễn missing.
hidden_missing_tokens = {
    "",
    "-",
    "--",
    "?",
    "n/a",
    "na",
    "null",
    "none",
    "unknown",
    "missing",
    "nan",
    "không rõ",
    "thỏa thuận",
    "negotiable",
}

hidden_missing_rows = []

for column in df.select_dtypes(include=["object", "string", "category"]).columns:
    normalized = df[column].astype("string").str.strip().str.casefold()
    hidden_mask = normalized.isin(hidden_missing_tokens) & df[column].notna()

    hidden_missing_rows.append(
        {
            "column": column,
            "n_hidden_missing": int(hidden_mask.sum()),
            "detected_values": (
                df.loc[hidden_mask, column].value_counts().to_dict()
                if hidden_mask.any()
                else {}
            ),
        }
    )

hidden_missing_report = pd.DataFrame(hidden_missing_rows)
print(hidden_missing_report.to_string(index=False))

print(
    f"\nTổng số ứng viên missing ẩn: "
    f"{int(hidden_missing_report['n_hidden_missing'].sum()):,}"
)
