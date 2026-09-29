"""
Hoàn thiện cleaning cho Week 4 (nhiệm vụ #44-#46, #51-#54).

File này chỉ áp dụng các quyết định đã được xác minh, sau đó audit consistency,
duplicate, date/unit/code, re-validation và data integrity trước khi xuất dataset cuối.
"""

from pathlib import Path
from difflib import SequenceMatcher
import sys
import unicodedata

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils import load_ecommerce_raw

INPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "ecommerce_customer_after_outliers.csv"
)
OUTPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "ecommerce_customer_clean.csv"
)

REPORT_DIR = PROJECT_ROOT / "reports" / "final"
REPORT_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 200)


def section(title: str) -> None:
    print(f"\n=== {title} ===")


def normalize_nfc(series: pd.Series) -> pd.Series:
    return (
        series.astype("string")
        .str.strip()
        .map(
            lambda value: (
                unicodedata.normalize("NFC", value)
                if pd.notna(value)
                else value
            )
        )
    )


def normalize_name(value: str) -> str:
    if pd.isna(value):
        return ""
    value = unicodedata.normalize("NFC", str(value))
    return " ".join(value.casefold().split())


def same_values(left: pd.Series, right: pd.Series) -> bool:
    left = left.reset_index(drop=True)
    right = right.reset_index(drop=True)

    if len(left) != len(right):
        return False

    equal = left.eq(right) | (left.isna() & right.isna())
    return bool(equal.fillna(False).all())


if not INPUT_PATH.exists():
    raise FileNotFoundError(
        f"Không tìm thấy intermediate dataset: {INPUT_PATH}\n"
        "Hãy chạy 02_missing_analysis.py và 03_outlier_analysis.py trước."
    )

# Finalize chỉ chạy sau khi đã hoàn thành missing và outlier analysis.
df = pd.read_csv(INPUT_PATH)
raw = load_ecommerce_raw()

rows_before = len(df)

section("Dữ liệu đầu vào")
print(f"Nguồn dữ liệu: {INPUT_PATH.name}")
print(f"Kích thước   : {rows_before:,} dòng x {df.shape[1]} cột")


section("Áp dụng các quyết định cleaning đã xác minh")

if "Returns_was_missing" not in df.columns:
    df["Returns_was_missing"] = df["Returns"].isna().astype("int8")

indicator_matches = df["Returns_was_missing"].eq(
    df["Returns"].isna().astype("int8")
)

if not indicator_matches.all():
    raise AssertionError(
        "Returns_was_missing không khớp với missingness của Returns."
    )

if {"Customer Age", "Age"}.issubset(df.columns):
    same_age = (
        df["Customer Age"].eq(df["Age"])
        | (df["Customer Age"].isna() & df["Age"].isna())
    )

    if not same_age.all():
        raise AssertionError(
            "Customer Age và Age không còn giống nhau hoàn toàn. "
            "Không thể drop Customer Age."
        )

    df = df.drop(columns=["Customer Age"])
    print("Đã bỏ cột dư thừa: Customer Age")


section("Consistency văn bản và Unicode NFC")

text_columns = [
    "Product Category",
    "Payment Method",
    "Gender",
    "Customer Name",
]

text_rows = []

# Chỉ chuẩn hóa representation, không thay đổi nội dung nghiệp vụ của text.
for column in text_columns:
    before = df[column].astype("string")
    before_unique = int(before.nunique(dropna=True))

    nfc_only = before.map(
        lambda value: (
            unicodedata.normalize("NFC", value)
            if pd.notna(value)
            else value
        )
    )
    nfc_changed = int((before != nfc_only).fillna(False).sum())

    normalized = normalize_nfc(before)
    changed_rows = int((before != normalized).fillna(False).sum())

    df[column] = normalized

    text_rows.append({
        "column": column,
        "n_unique_before": before_unique,
        "n_unique_after": int(df[column].nunique(dropna=True)),
        "rows_changed_strip_or_nfc": changed_rows,
        "rows_changed_by_nfc_only": nfc_changed,
        "nfc_stable_after": bool(
            same_values(df[column], normalize_nfc(df[column]))
        ),
    })

text_report = pd.DataFrame(text_rows)
print(text_report.to_string(index=False))
text_report.to_csv(
    REPORT_DIR / "text_consistency.csv",
    index=False,
    encoding="utf-8-sig",
)

df["Purchase Date"] = pd.to_datetime(
    df["Purchase Date"],
    errors="raise",
)
df["Returns_was_missing"] = df["Returns_was_missing"].astype("int8")

print("\nĐã chuyển Purchase Date sang datetime.")
print("Returns được giữ ở {0, 1, NaN}; không áp dụng imputation lên dataset cleaned.")
print("Point/contextual/collective outlier: không tự động xóa dữ liệu.")


section("Audit duplicate theo business key")

# Kiểm tra duplicate từ key rộng đến strict event key; chỉ gắn cờ nếu chưa có bằng chứng chắc chắn.
duplicate_keys = {
    "customer_timestamp": [
        "Customer ID",
        "Purchase Date",
    ],
    "customer_timestamp_category": [
        "Customer ID",
        "Purchase Date",
        "Product Category",
    ],
    "strict_event_key": [
        "Customer ID",
        "Purchase Date",
        "Product Category",
        "Product Price",
        "Quantity",
        "Total Purchase Amount",
        "Payment Method",
    ],
}

duplicate_rows = []
strict_duplicate_mask = pd.Series(False, index=df.index)

for key_name, columns in duplicate_keys.items():
    mask = df.duplicated(
        subset=columns,
        keep=False,
    )

    if key_name == "strict_event_key":
        strict_duplicate_mask = mask

    duplicate_subset = df.loc[mask, columns]

    duplicate_groups = (
        duplicate_subset.groupby(
            columns,
            dropna=False,
        ).ngroups
        if not duplicate_subset.empty
        else 0
    )

    duplicate_rows.append({
        "key": key_name,
        "columns": " | ".join(columns),
        "duplicate_rows": int(mask.sum()),
        "duplicate_groups": int(duplicate_groups),
        "action": (
            "Chỉ gắn cờ"
            if mask.any()
            else "Không cần xử lý"
        ),
    })

business_duplicate_report = pd.DataFrame(duplicate_rows)

print(business_duplicate_report.to_string(index=False))
business_duplicate_report.to_csv(
    REPORT_DIR / "business_key_duplicates.csv",
    index=False,
    encoding="utf-8-sig",
)

df.loc[strict_duplicate_mask].to_csv(
    REPORT_DIR / "strict_event_duplicate_candidates.csv",
    index=False,
    encoding="utf-8-sig",
)


section("Audit approximate duplicate của khách hàng")

profiles = (
    df[
        [
            "Customer ID",
            "Customer Name",
            "Age",
            "Gender",
            "Churn",
        ]
    ]
    .drop_duplicates(subset=["Customer ID"])
    .copy()
)

profiles["name_normalized"] = (
    profiles["Customer Name"]
    .astype("string")
    .map(normalize_name)
)

profiles["initial"] = (
    profiles["name_normalized"]
    .str[:1]
    .fillna("")
)

name_reuse_report = (
    profiles[profiles["name_normalized"] != ""]
    .groupby("name_normalized")
    .agg(
        n_customer_ids=("Customer ID", "nunique"),
        age_variants=("Age", "nunique"),
        gender_variants=("Gender", "nunique"),
        churn_variants=("Churn", "nunique"),
    )
    .query("n_customer_ids > 1")
    .sort_values("n_customer_ids", ascending=False)
    .reset_index()
)

print(f"Số tên khách hàng chuẩn hóa bị dùng lại: {len(name_reuse_report):,}")
name_reuse_report.to_csv(
    REPORT_DIR / "exact_normalized_name_reuse.csv",
    index=False,
    encoding="utf-8-sig",
)

approximate_rows = []
similarity_threshold = 0.96

for _, group in profiles.groupby(
    ["Age", "Gender", "Churn", "initial"],
    dropna=False,
):
    records = group[
        [
            "Customer ID",
            "Customer Name",
            "name_normalized",
            "Age",
            "Gender",
            "Churn",
        ]
    ].to_dict("records")

    for i in range(len(records)):
        left = records[i]
        left_name = left["name_normalized"]

        for j in range(i + 1, len(records)):
            right = records[j]
            right_name = right["name_normalized"]

            if not left_name or not right_name:
                continue

            max_len = max(len(left_name), len(right_name))

            if abs(len(left_name) - len(right_name)) > max(
                2,
                int(0.20 * max_len),
            ):
                continue

            similarity = SequenceMatcher(
                None,
                left_name,
                right_name,
            ).ratio()

            if similarity >= similarity_threshold:
                approximate_rows.append({
                    "customer_id_1": left["Customer ID"],
                    "customer_id_2": right["Customer ID"],
                    "name_1": left["Customer Name"],
                    "name_2": right["Customer Name"],
                    "age": left["Age"],
                    "gender": left["Gender"],
                    "churn": left["Churn"],
                    "name_similarity": round(similarity, 4),
                    "verdict": "Chỉ là ứng viên; không tự động coi là duplicate",
                })

approximate_report = pd.DataFrame(approximate_rows)

if approximate_report.empty:
    approximate_report = pd.DataFrame(
        columns=[
            "customer_id_1",
            "customer_id_2",
            "name_1",
            "name_2",
            "age",
            "gender",
            "churn",
            "name_similarity",
            "verdict",
        ]
    )

approximate_report = approximate_report.sort_values(
    "name_similarity",
    ascending=False,
)

print(f"Số ứng viên approximate duplicate: {len(approximate_report):,}")

if not approximate_report.empty:
    print(approximate_report.head(20).to_string(index=False))

approximate_report.to_csv(
    REPORT_DIR / "approximate_customer_duplicate_candidates.csv",
    index=False,
    encoding="utf-8-sig",
)


section("Ngày tháng, đơn vị và mã phân loại")

future_dates = int((df["Purchase Date"] > pd.Timestamp.now()).sum())
invalid_dates = int(df["Purchase Date"].isna().sum())

explicit_unit_columns = [
    column
    for column in df.columns
    if any(token in column.casefold() for token in ("unit", "currency", "uom"))
]

raw_domains = {
    column: set(normalize_nfc(raw[column]).dropna().tolist())
    for column in ["Product Category", "Payment Method", "Gender"]
}
final_domains = {
    column: set(df[column].astype("string").dropna().tolist())
    for column in ["Product Category", "Payment Method", "Gender"]
}

unit_status = "ĐẠT" if explicit_unit_columns else "KHÔNG ÁP DỤNG"
unit_evidence = (
    f"explicit unit/currency columns = {explicit_unit_columns}"
    if explicit_unit_columns
    else (
        "Dataset không có cột metadata unit/currency rõ ràng. Các cột price/amount "
        "chỉ chứa số nên không quan sát thấy mixed-unit encoding; không thể xác minh "
        "đơn vị/currency về mặt ngữ nghĩa chỉ từ bảng này."
    )
)

unit_code_report = pd.DataFrame([
    {
        "check": "Purchase Date parse được",
        "status": "ĐẠT" if invalid_dates == 0 else "KHÔNG ĐẠT",
        "result": invalid_dates == 0,
        "evidence": f"invalid dates = {invalid_dates}",
    },
    {
        "check": "Purchase Date không ở tương lai",
        "status": "ĐẠT" if future_dates == 0 else "KHÔNG ĐẠT",
        "result": future_dates == 0,
        "evidence": f"future dates = {future_dates}",
    },
    {
        "check": "Consistency của unit/currency",
        "status": unit_status,
        "result": True,
        "evidence": unit_evidence,
    },
    {
        "check": "Giữ nguyên domain Product Category",
        "status": "ĐẠT" if final_domains["Product Category"] == raw_domains["Product Category"] else "KHÔNG ĐẠT",
        "result": final_domains["Product Category"] == raw_domains["Product Category"],
        "evidence": " | ".join(sorted(final_domains["Product Category"])),
    },
    {
        "check": "Giữ nguyên domain Payment Method",
        "status": "ĐẠT" if final_domains["Payment Method"] == raw_domains["Payment Method"] else "KHÔNG ĐẠT",
        "result": final_domains["Payment Method"] == raw_domains["Payment Method"],
        "evidence": " | ".join(sorted(final_domains["Payment Method"])),
    },
    {
        "check": "Giữ nguyên domain Gender",
        "status": "ĐẠT" if final_domains["Gender"] == raw_domains["Gender"] else "KHÔNG ĐẠT",
        "result": final_domains["Gender"] == raw_domains["Gender"],
        "evidence": " | ".join(sorted(final_domains["Gender"])),
    },
])

print(unit_code_report.to_string(index=False))
unit_code_report.to_csv(
    REPORT_DIR / "date_unit_code_audit.csv",
    index=False,
    encoding="utf-8-sig",
)


section("Kiểm tra lại sau cleaning")

hidden_tokens = {
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

hidden_missing_count = 0

for column in df.select_dtypes(
    include=["object", "string", "category"]
).columns:
    normalized = (
        df[column]
        .astype("string")
        .str.strip()
        .str.casefold()
    )

    hidden_missing_count += int(
        (
            normalized.isin(hidden_tokens)
            & df[column].notna()
        ).sum()
    )

exact_duplicates = int(df.duplicated().sum())
strict_event_duplicate_rows = int(strict_duplicate_mask.sum())

expected_final_columns = [
    "Customer ID",
    "Purchase Date",
    "Product Category",
    "Product Price",
    "Quantity",
    "Total Purchase Amount",
    "Payment Method",
    "Returns",
    "Customer Name",
    "Age",
    "Gender",
    "Churn",
    "Returns_was_missing",
]

# Re-validation là cổng cuối trước khi chấp nhận dataset cleaned.
checks = {
    "Giữ nguyên số dòng": len(df) == rows_before,
    "Schema cuối đúng kỳ vọng": list(df.columns) == expected_final_columns,
    "Đã bỏ Customer Age": "Customer Age" not in df.columns,
    "Purchase Date đã parse": pd.api.types.is_datetime64_any_dtype(
        df["Purchase Date"]
    ),
    "Purchase Date không còn missing": df["Purchase Date"].notna().all(),
    "Purchase Date không ở tương lai": future_dates == 0,
    "Customer ID > 0": (df["Customer ID"] > 0).all(),
    "Product Price >= 0": (df["Product Price"] >= 0).all(),
    "Quantity > 0": (df["Quantity"] > 0).all(),
    "Total Purchase Amount >= 0": (
        df["Total Purchase Amount"] >= 0
    ).all(),
    "Age nằm trong [0, 120]": df["Age"].between(0, 120).all(),
    "Returns thuộc {0, 1, missing}": (
        df["Returns"].isin([0, 1])
        | df["Returns"].isna()
    ).all(),
    "Churn thuộc {0, 1}": df["Churn"].isin([0, 1]).all(),
    "Returns indicator thuộc {0, 1}": (
        df["Returns_was_missing"].isin([0, 1]).all()
    ),
    "Returns indicator khớp missingness": (
        df["Returns_was_missing"].eq(
            df["Returns"].isna().astype("int8")
        ).all()
    ),
    "Không có exact duplicate": exact_duplicates == 0,
    "Đã audit duplicate theo strict event key": True,
    "Không có hidden missing trong text": hidden_missing_count == 0,
    "Toàn bộ text ổn định theo NFC": bool(
        text_report["nfc_stable_after"].all()
    ),
    "Kiểm tra date/unit/code đạt": bool(
        unit_code_report["result"].all()
    ),
}

validation_report = pd.DataFrame(
    [
        {
            "check": name,
            "passed": bool(passed),
        }
        for name, passed in checks.items()
    ]
)

print(validation_report.to_string(index=False))

if not validation_report["passed"].all():
    failed = validation_report.loc[
        ~validation_report["passed"],
        "check",
    ].tolist()

    raise AssertionError(
        f"Final validation failed: {failed}"
    )


section("Kiểm tra toàn vẹn dữ liệu")

text_integrity_columns = {
    "Product Category",
    "Payment Method",
    "Customer Name",
    "Gender",
}

stable_columns = [
    "Customer ID",
    "Product Category",
    "Product Price",
    "Quantity",
    "Total Purchase Amount",
    "Payment Method",
    "Returns",
    "Customer Name",
    "Age",
    "Gender",
    "Churn",
]

integrity_rows = []

for column in stable_columns:
    if column in text_integrity_columns:
        actual_series = normalize_nfc(df[column])
        expected_series = normalize_nfc(raw[column])
        unchanged = same_values(actual_series, expected_series)
        comparison_basis = "normalized(raw)"
    else:
        unchanged = same_values(df[column], raw[column])
        comparison_basis = "raw"

    integrity_rows.append({
        "column": column,
        "comparison_basis": comparison_basis,
        "unchanged_from_expected_source": bool(unchanged),
    })

integrity_report = pd.DataFrame(integrity_rows)

print(integrity_report.to_string(index=False))

if not integrity_report[
    "unchanged_from_expected_source"
].all():
    changed = integrity_report.loc[
        ~integrity_report["unchanged_from_expected_source"],
        "column",
    ].tolist()

    raise AssertionError(
        f"Unexpected changes from source data: {changed}"
    )


section("Tóm tắt chất lượng cuối cùng")

missing_summary = pd.DataFrame({
    "n_missing": df.isna().sum(),
    "pct_missing": (
        df.isna().mean() * 100
    ).round(4),
})

print(f"Số dòng                    : {len(df):,}")
print(f"Số cột                     : {df.shape[1]}")
print(f"Exact duplicate            : {exact_duplicates:,}")
print(
    f"Duplicate theo strict event key: "
    f"{strict_event_duplicate_rows:,}"
)
print(
    f"Ứng viên approximate duplicate : "
    f"{len(approximate_report):,}"
)
print(f"Hidden missing             : {hidden_missing_count:,}")
print(
    f"Returns bị thiếu            : "
    f"{int(df['Returns'].isna().sum()):,}"
)
print(
    f"Số cờ missing               : "
    f"{int(df['Returns_was_missing'].sum()):,}"
)

print("\nTóm tắt missing:")
print(
    missing_summary[
        missing_summary["n_missing"] > 0
    ].to_string()
)


returns_missing_count = int(raw["Returns"].isna().sum())
returns_missing_pct = returns_missing_count / len(raw) * 100

cleaning_log = pd.DataFrame([
    {
        "stage": "Giá trị thiếu",
        "column": "Returns",
        "finding": (
            f"{returns_missing_count:,} giá trị thiếu "
            f"({returns_missing_pct:.4f}%)"
        ),
        "action": "Giữ NaN và thêm Returns_was_missing",
        "affected_rows": int(df["Returns_was_missing"].sum()),
        "reason": (
            "Không có bằng chứng rằng Returns bị thiếu đồng nghĩa 0 hoặc 1; "
            "fill-0/fill-1 làm thay đổi đáng kể phân bố quan sát được."
        ),
    },
    {
        "stage": "Outlier",
        "column": (
            "Product Price, Quantity, "
            "Total Purchase Amount, Age"
        ),
        "finding": (
            "Đã kiểm tra point outlier bằng Z-score, IQR và Modified Z-score; "
            "đồng thời audit contextual và collective candidate"
        ),
        "action": "Không tự động xóa",
        "affected_rows": 0,
        "reason": (
            "Cờ thống kê chỉ là tín hiệu cần điều tra, không phải bằng chứng lỗi."
        ),
    },
    {
        "stage": "Consistency",
        "column": (
            "Product Category, Payment Method, "
            "Gender, Customer Name"
        ),
        "finding": (
            f"{int(text_report['rows_changed_by_nfc_only'].sum())} "
            "dòng thay đổi do chuẩn hóa NFC"
        ),
        "action": "Áp dụng strip + chuẩn hóa Unicode NFC",
        "affected_rows": int(
            text_report["rows_changed_strip_or_nfc"].sum()
        ),
        "reason": (
            "Đảm bảo biểu diễn text nhất quán trước khi group, join và kiểm tra duplicate."
        ),
    },
    {
        "stage": "Duplicate",
        "column": "Strict event business key",
        "finding": (
            f"{strict_event_duplicate_rows} dòng duplicate theo strict event key; "
            f"{len(name_reuse_report)} tên chuẩn hóa bị dùng lại; "
            f"{len(approximate_report)} ứng viên approximate duplicate"
        ),
        "action": "Không tự động merge approximate candidate",
        "affected_rows": strict_event_duplicate_rows,
        "reason": (
            "Độ giống near-duplicate chưa đủ bằng chứng để merge các Customer ID khác nhau."
        ),
    },
    {
        "stage": "Dư thừa",
        "column": "Customer Age",
        "finding": "Giống Age trên toàn bộ dòng",
        "action": "Bỏ Customer Age",
        "affected_rows": len(df),
        "reason": (
            "Loại feature dư thừa sau khi đã xác minh equality trên toàn bộ dòng."
        ),
    },
    {
        "stage": "Date / unit / code",
        "column": "Purchase Date and canonical categorical domains",
        "finding": (
            "Date parse thành công; không có ngày tương lai; "
            f"trạng thái audit unit/currency = {unit_status}"
        ),
        "action": "Chuyển date và validate canonical domain",
        "affected_rows": len(df),
        "reason": (
            "Đảm bảo date ở dạng máy đọc nhất quán và kiểm tra code consistency quan sát được. "
            "Unit semantics được đánh dấu không áp dụng khi dataset không có metadata unit/currency."
        ),
    },
    {
        "stage": "Re-validation",
        "column": "All",
        "finding": "Tất cả final check đã khai báo đều đạt",
        "action": "Chấp nhận dataset cleaned cuối",
        "affected_rows": len(df),
        "reason": (
            "Schema, validity, missingness, consistency, duplicate và integrity đều đạt."
        ),
    },
])


df.to_csv(
    OUTPUT_PATH,
    index=False,
    encoding="utf-8-sig",
    date_format="%Y-%m-%d %H:%M:%S",
)

validation_report.to_csv(
    REPORT_DIR / "final_validation.csv",
    index=False,
    encoding="utf-8-sig",
)

integrity_report.to_csv(
    REPORT_DIR / "data_integrity.csv",
    index=False,
    encoding="utf-8-sig",
)

missing_summary.to_csv(
    REPORT_DIR / "final_missing_summary.csv",
    encoding="utf-8-sig",
)

cleaning_log.to_csv(
    REPORT_DIR / "final_cleaning_log.csv",
    index=False,
    encoding="utf-8-sig",
)


section("Đã lưu kết quả")

print(f"Dataset cuối      : {OUTPUT_PATH}")
print(
    f"Validation        : "
    f"{REPORT_DIR / 'final_validation.csv'}"
)
print(
    f"Data integrity    : "
    f"{REPORT_DIR / 'data_integrity.csv'}"
)
print(
    f"Text consistency  : "
    f"{REPORT_DIR / 'text_consistency.csv'}"
)
print(
    f"Business dup audit: "
    f"{REPORT_DIR / 'business_key_duplicates.csv'}"
)
print(
    f"Tên dùng lại      : "
    f"{REPORT_DIR / 'exact_normalized_name_reuse.csv'}"
)
print(
    f"Approx dup audit  : "
    f"{REPORT_DIR / 'approximate_customer_duplicate_candidates.csv'}"
)
print(
    f"Date/unit/code    : "
    f"{REPORT_DIR / 'date_unit_code_audit.csv'}"
)
print(
    f"Cleaning log      : "
    f"{REPORT_DIR / 'final_cleaning_log.csv'}"
)
