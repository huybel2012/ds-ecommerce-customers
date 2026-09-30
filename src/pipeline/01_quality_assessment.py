"""
File này chỉ đánh giá và ghi bằng chứng. Các ngưỡng chấm điểm 1-5 là rubric
nội bộ của project để so sánh nhất quán, không phải chuẩn phổ quát.
"""

from pathlib import Path
import sys

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils import load_ecommerce_raw

REPORT_PATH = PROJECT_ROOT / "reports" / "quality_assessment.csv"

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 180)

RUBRIC_NOTE = (
    "Điểm 1-5 dùng rubric nội bộ của project để so sánh nhất quán; "
    "đây không phải ngưỡng chất lượng dữ liệu phổ quát."
)


def section(title: str) -> None:
    print(f"\n=== {title} ===")


def score_completeness(max_missing_pct: float) -> int:
    if max_missing_pct == 0:
        return 5
    if max_missing_pct <= 5:
        return 4
    if max_missing_pct <= 20:
        return 3
    if max_missing_pct <= 40:
        return 2
    return 1


def score_rate(rate: float) -> int:
    if rate == 0:
        return 5
    if rate <= 0.1:
        return 4
    if rate <= 1:
        return 3
    if rate <= 5:
        return 2
    return 1


def score_timeliness(staleness_days: int) -> int:
    if staleness_days <= 90:
        return 5
    if staleness_days <= 180:
        return 4
    if staleness_days <= 365:
        return 3
    if staleness_days <= 730:
        return 2
    return 1


def main() -> None:
    # Đánh giá trực tiếp trên raw data, chưa áp dụng cleaning.
    df = load_ecommerce_raw()
    n_rows = len(df)

    if n_rows == 0:
        raise ValueError("Raw dataset is empty.")

    section("Rubric chấm điểm")
    print(RUBRIC_NOTE)

    section("Completeness - Mức độ đầy đủ")

    hidden_tokens = {
        "", "-", "--", "?", "n/a", "na", "null", "none",
        "unknown", "missing", "nan", "không rõ", "thỏa thuận", "negotiable",
    }

    missing_counts = df.isna().sum().copy()

    for column in df.select_dtypes(include=["object", "string", "category"]).columns:
        normalized = df[column].astype("string").str.strip().str.casefold()
        missing_counts[column] += int(
            (normalized.isin(hidden_tokens) & df[column].notna()).sum()
        )

    missing_pct = (missing_counts / n_rows * 100).round(4)
    completeness_report = pd.DataFrame({
        "n_missing": missing_counts,
        "pct_missing": missing_pct,
    }).sort_values("pct_missing", ascending=False)

    print(completeness_report)

    max_missing_column = completeness_report.index[0]
    max_missing_pct = float(completeness_report.iloc[0]["pct_missing"])
    overall_missing_pct = float(missing_counts.sum() / df.size * 100)
    completeness_score = score_completeness(max_missing_pct)

    section("Accuracy - Độ chính xác")

    customer_conflict_ids = set()
    customer_conflicts = {}

    for column in ["Customer Name", "Gender", "Churn"]:
        conflict_ids = set(
            df.groupby("Customer ID")[column]
            .nunique(dropna=False)
            .loc[lambda s: s > 1]
            .index
        )
        customer_conflicts[column] = len(conflict_ids)
        customer_conflict_ids.update(conflict_ids)

    age_mismatch_mask = ~(
        df["Customer Age"].eq(df["Age"])
        | (df["Customer Age"].isna() & df["Age"].isna())
    )
    age_mismatch = int(age_mismatch_mask.sum())

    print("Xung đột hồ sơ theo khách hàng:", customer_conflicts)
    print(f"Số khách hàng có xung đột hồ sơ: {len(customer_conflict_ids):,}")
    print(f"Số dòng Customer Age khác Age: {age_mismatch:,}")
    print("Ground truth bên ngoài: không có")

    total_accuracy_conflicts = len(customer_conflict_ids) + age_mismatch
    accuracy_score = 3 if total_accuracy_conflicts == 0 else 2

    section("Consistency - Tính nhất quán")

    consistency_mask = pd.Series(False, index=df.index)
    normalization_details = {}

    for column in ["Product Category", "Payment Method", "Gender"]:
        raw = df[column].astype("string")
        normalized = raw.str.strip().str.casefold()

        variants = (
            pd.DataFrame({"raw": raw, "normalized": normalized})
            .groupby("normalized", dropna=False)["raw"]
            .nunique(dropna=False)
        )

        inconsistent_keys = variants[variants > 1].index
        issue_mask = normalized.isin(inconsistent_keys).fillna(False)

        normalization_details[column] = int(issue_mask.sum())
        consistency_mask |= issue_mask

    if customer_conflict_ids:
        consistency_mask |= df["Customer ID"].isin(customer_conflict_ids)

    consistency_mask |= age_mismatch_mask

    purchase_date = pd.to_datetime(df["Purchase Date"], errors="coerce")
    invalid_date_mask = purchase_date.isna()
    consistency_mask |= invalid_date_mask

    consistency_issues = int(consistency_mask.sum())
    consistency_rate = consistency_issues / n_rows * 100

    print("Dòng có vấn đề chuẩn hóa văn bản:", normalization_details)
    print("Xung đột hồ sơ theo khách hàng:", customer_conflicts)
    print(f"Số dòng Age không khớp: {age_mismatch:,}")
    print(f"Ngày mua không parse được: {int(invalid_date_mask.sum()):,}")
    print(f"Số dòng duy nhất có vấn đề consistency: {consistency_issues:,}")
    print(f"Tỷ lệ dòng có vấn đề consistency: {consistency_rate:.4f}%")

    consistency_score = score_rate(consistency_rate)

    section("Validity - Tính hợp lệ")

    # Validity dùng hard rules; outlier thống kê được phân tích riêng ở Week 4.
    validity_checks = {
        "Customer ID > 0": df["Customer ID"] > 0,
        "Product Price >= 0": df["Product Price"] >= 0,
        "Quantity > 0": df["Quantity"] > 0,
        "Total Purchase Amount >= 0": df["Total Purchase Amount"] >= 0,
        "Customer Age in [0, 120]": df["Customer Age"].between(0, 120),
        "Age in [0, 120]": df["Age"].between(0, 120),
        "Returns in {0, 1, missing}": df["Returns"].isin([0, 1]) | df["Returns"].isna(),
        "Churn in {0, 1}": df["Churn"].isin([0, 1]),
        "Purchase Date parseable": purchase_date.notna(),
    }

    validity_rows = []
    invalid_total = 0

    for name, valid_mask in validity_checks.items():
        n_invalid = int((~valid_mask).sum())
        invalid_total += n_invalid
        validity_rows.append({
            "rule": name,
            "n_invalid": n_invalid,
            "pct_invalid": round(n_invalid / n_rows * 100, 4),
        })

    validity_report = pd.DataFrame(validity_rows)
    print(validity_report.to_string(index=False))

    validity_rate = invalid_total / (n_rows * len(validity_checks)) * 100
    validity_score = score_rate(validity_rate)

    section("Uniqueness - Tính duy nhất")

    exact_duplicates = int(df.duplicated().sum())

    event_key = [
        "Customer ID",
        "Purchase Date",
        "Product Category",
        "Product Price",
        "Quantity",
        "Total Purchase Amount",
        "Payment Method",
    ]

    possible_duplicate_event_rows = int(
        df.duplicated(subset=event_key, keep=False).sum()
    )

    print(f"Số dòng trùng hoàn toàn: {exact_duplicates:,}")
    print(f"Số dòng có thể trùng theo event key: {possible_duplicate_event_rows:,}")
    print("Cột Transaction ID: không có")

    if exact_duplicates > 0:
        uniqueness_score = score_rate(exact_duplicates / n_rows * 100)
    elif possible_duplicate_event_rows > 0:
        uniqueness_score = 3
    else:
        uniqueness_score = 4

    section("Timeliness - Tính cập nhật")

    valid_dates = purchase_date.dropna()
    if valid_dates.empty:
        raise ValueError("No parseable Purchase Date values are available.")

    earliest_date = valid_dates.min()
    latest_date = valid_dates.max()
    today = pd.Timestamp.now().normalize()

    future_dates = int((valid_dates > today).sum())
    staleness_days = max(0, int((today - latest_date.normalize()).days))

    print(f"Giao dịch sớm nhất: {earliest_date}")
    print(f"Giao dịch mới nhất: {latest_date}")
    print(f"Ngày ở tương lai  : {future_dates:,}")
    print(f"Độ cũ dữ liệu     : {staleness_days:,} ngày")

    timeliness_score = score_timeliness(staleness_days)
    if future_dates:
        timeliness_score = min(timeliness_score, 2)

    section("Tổng hợp đánh giá chất lượng")

    rows = [
        {
            "dimension": "Completeness (Đầy đủ)",
            "score": completeness_score,
            "evidence": (
                f"Tỷ lệ thiếu cao nhất: {max_missing_column} = {max_missing_pct:.2f}%; "
                f"tỷ lệ thiếu toàn dataset = {overall_missing_pct:.2f}%."
            ),
            "limitation": "Cơ chế missing được phân tích riêng trong Missing Analysis.",
        },
        {
            "dimension": "Accuracy (Chính xác)",
            "score": accuracy_score,
            "evidence": (
                f"Khách hàng có xung đột hồ sơ = {len(customer_conflict_ids):,}; "
                f"số dòng Age không khớp = {age_mismatch:,}."
            ),
            "limitation": (
                "Không thể xác minh đầy đủ Accuracy so với thực tế khi không có "
                "nguồn ground truth bên ngoài."
            ),
        },
        {
            "dimension": "Consistency (Nhất quán)",
            "score": consistency_score,
            "evidence": (
                f"Số dòng duy nhất có vấn đề consistency = {consistency_issues:,}; "
                f"tỷ lệ = {consistency_rate:.4f}%."
            ),
            "limitation": "Consistency sâu hơn cho text/date/unit được kiểm tra ở Week 4.",
        },
        {
            "dimension": "Validity (Hợp lệ)",
            "score": validity_score,
            "evidence": (
                f"Số vi phạm validity rule = {invalid_total:,}; "
                f"tỷ lệ vi phạm theo rule-cell = {validity_rate:.4f}%."
            ),
            "limitation": "Outlier thống kê không mặc định được xem là giá trị không hợp lệ.",
        },
        {
            "dimension": "Uniqueness (Duy nhất)",
            "score": uniqueness_score,
            "evidence": (
                f"Exact duplicate = {exact_duplicates:,}; "
                f"dòng có thể trùng theo event key = {possible_duplicate_event_rows:,}."
            ),
            "limitation": (
                "Không có Transaction/Order ID nên không thể chứng minh tuyệt đối "
                "Uniqueness ở cấp giao dịch."
            ),
        },
        {
            "dimension": "Timeliness (Cập nhật)",
            "score": timeliness_score,
            "evidence": (
                f"Khoảng thời gian {earliest_date} đến {latest_date}; "
                f"bản ghi mới nhất cách ngày chạy {staleness_days:,} ngày; "
                f"ngày tương lai = {future_dates:,}."
            ),
            "limitation": (
                "Timeliness được chấm tương đối theo ngày chạy script và mục tiêu phân tích hiện tại."
            ),
        },
    ]

    quality_report = pd.DataFrame(rows)
    quality_report.insert(2, "rubric", "project-defined (nội bộ project)")

    print(quality_report.to_string(index=False))

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    quality_report.to_csv(REPORT_PATH, index=False, encoding="utf-8-sig")
    print(f"\nĐã lưu báo cáo: {REPORT_PATH}")


if __name__ == "__main__":
    main()
