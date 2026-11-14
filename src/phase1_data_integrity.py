"""Phase 1: evidence-first data integrity audit and Returns restoration.

Run on Windows from repository root:
    .\\venv\\Scripts\\python.exe -m src.phase1_data_integrity

Requires pandas and pyarrow (both already installed by the team).
Never overwrites the legacy cleaned CSV or the source parquet file.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CSV = ROOT / "data/processed/ecommerce_customer_cleaned.csv"
DEFAULT_PARQUET = ROOT / "data/processed/ecommerce_clean.parquet"
DEFAULT_OUTPUT = ROOT / "data/processed/ecommerce_customer_integrity.csv"
DEFAULT_REPORT_DIR = ROOT / "reports/phase1"

KEY = ["Customer ID", "Purchase Date"]
EXPECTED = [
    "Customer ID", "Purchase Date", "Product Category", "Product Price",
    "Quantity", "Total Purchase Amount", "Payment Method", "Returns",
    "Customer Name", "Age", "Gender", "Churn",
]
IDENTITY_COLUMNS = [
    "Product Category", "Product Price", "Quantity", "Total Purchase Amount",
    "Payment Method", "Age", "Gender", "Churn",
]
MISSING_TOKENS = {"", "na", "n/a", "null", "none", "missing", "unknown", "?", "-", "--"}


def load_legacy(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    absent = sorted(set(EXPECTED) - set(frame.columns))
    if absent:
        raise ValueError(f"Legacy CSV missing columns: {absent}")
    return frame


def _normalize_dates(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    out["Purchase Date"] = pd.to_datetime(out["Purchase Date"], errors="raise")
    if out["Purchase Date"].isna().any():
        raise ValueError("Purchase Date contains NaT")
    return out


def restore_returns(legacy: pd.DataFrame, rawlike: pd.DataFrame) -> pd.DataFrame:
    """Reconcile on a verified unique event key, NEVER on row position.

    Rejects any difference in stable business fields or in observed Returns.
    Makes a new sanitized dataframe; leaves both inputs untouched.
    """
    legacy = _normalize_dates(legacy)
    rawlike = _normalize_dates(rawlike)
    for col in KEY + IDENTITY_COLUMNS + ["Returns"]:
        if col not in rawlike:
            raise ValueError(f"Source parquet is missing {col!r}")
    if len(legacy) != len(rawlike):
        raise ValueError(f"Row counts disagree: CSV={len(legacy)}, Parquet={len(rawlike)}")
    for label, data in [("CSV", legacy), ("Parquet", rawlike)]:
        if data.duplicated(KEY).any():
            raise ValueError(f"{label} has duplicated business key {KEY}; cannot match uniquely")
    if "Customer Age" in rawlike and not rawlike["Customer Age"].eq(rawlike["Age"]).all():
        raise ValueError("Customer Age does not equal Age in Parquet; do not discard it")
    reference = rawlike[KEY + IDENTITY_COLUMNS + ["Returns"]].rename(
        columns={name: f"_source_{name}" for name in IDENTITY_COLUMNS + ["Returns"]}
    )
    joined = legacy.merge(reference, on=KEY, how="left", indicator=True,
                          sort=False, validate="one_to_one")
    if not joined["_merge"].eq("both").all():
        raise ValueError("At least one CSV event is absent from the source Parquet")
    for col in IDENTITY_COLUMNS:
        left, right = joined[col], joined[f"_source_{col}"]
        if col in {"Product Category", "Payment Method", "Gender"}:
            same = left.astype("string").str.strip().str.casefold().eq(
                right.astype("string").str.strip().str.casefold()
            )
        else:
            same = left.eq(right)
        if not bool(same.fillna(False).all()):
            raise ValueError(f"Source mismatch in column {col!r}; refuse to attach Returns")
    raw_returns = pd.to_numeric(joined["_source_Returns"], errors="raise")
    if not raw_returns.dropna().isin([0, 1]).all():
        raise ValueError("Parquet Returns values must be 0, 1 or NaN")
    recorded = pd.to_numeric(joined["Returns"], errors="raise")
    observed = raw_returns.notna()
    if not recorded.loc[observed].eq(raw_returns.loc[observed]).all():
        raise ValueError("The old CSV changed some observed 0/1 Returns values")
    if not recorded.loc[~observed].eq(0).all():
        raise ValueError("The old CSV did not fill the missing Returns with 0 as expected")

    # Keep the original column order, preserving every other measured field.
    out = legacy.copy()
    out["Returns"] = raw_returns.to_numpy()
    out["Returns_was_missing"] = raw_returns.isna().astype("int8").to_numpy()
    out["line_item_value_calc"] = out["Product Price"] * out["Quantity"]
    # Faker-generated names add no analytic value; suppress them in the new public-facing output.
    out = out.drop(columns=["Customer Name"], errors="ignore")
    if not out["Returns_was_missing"].eq(out["Returns"].isna().astype("int8")).all():
        raise AssertionError("Returns indicator does not match missingness")
    return out


def assess(data: pd.DataFrame) -> dict:
    frame = _normalize_dates(data)
    result = {}
    result["rows"] = int(len(frame))
    result["columns"] = int(len(frame.columns))
    result["unique_customers"] = int(frame["Customer ID"].nunique())
    result["min_purchase_date"] = str(frame["Purchase Date"].min())
    result["max_purchase_date"] = str(frame["Purchase Date"].max())
    result["nulls_by_column"] = {k: int(v) for k, v in frame.isna().sum().items()}
    result["duplicate_rows"] = int(frame.duplicated().sum())
    result["duplicate_customer_timestamp"] = int(frame.duplicated(KEY).sum())
    result["age_outside_18_70"] = int((~frame["Age"].between(18, 70)).sum())
    result["age_outside_0_120"] = int((~frame["Age"].between(0, 120)).sum())
    result["nonpositive_price"] = int((frame["Product Price"] <= 0).sum())
    result["nonpositive_quantity"] = int((frame["Quantity"] <= 0).sum())
    result["negative_reported_amount"] = int((frame["Total Purchase Amount"] < 0).sum())
    result["invalid_churn"] = int((~frame["Churn"].isin([0, 1])).sum())
    result["invalid_returns"] = int((~(frame["Returns"].isin([0, 1]) | frame["Returns"].isna())).sum())
    result["churn_inconsistent_customer_ids"] = int((frame.groupby("Customer ID")["Churn"].nunique() > 1).sum())
    result["name_inconsistent_customer_ids"] = (int((frame.groupby("Customer ID")["Customer Name"].nunique() > 1).sum())
                                                 if "Customer Name" in frame.columns else None)
    result["age_inconsistent_customer_ids"] = int((frame.groupby("Customer ID")["Age"].nunique() > 1).sum())
    result["gender_inconsistent_customer_ids"] = int((frame.groupby("Customer ID")["Gender"].nunique() > 1).sum())
    result["hidden_missing_tokens"] = {}
    for col in frame.select_dtypes(include=["object", "string"]).columns:
        n = int(frame[col].astype("string").str.strip().str.casefold().isin(MISSING_TOKENS).sum())
        if n:
            result["hidden_missing_tokens"][col] = n
    calc = frame["Product Price"] * frame["Quantity"]
    reported = frame["Total Purchase Amount"]
    result["amount_equal_count"] = int(reported.eq(calc).sum())
    result["amount_mismatch_count"] = int(reported.ne(calc).sum())
    result["amount_correlation"] = float(reported.corr(calc))
    result["reported_amount_sum"] = int(reported.sum())
    result["line_item_value_calc_sum"] = int(calc.sum())
    result["reported_amount_mean"] = float(reported.mean())
    result["churn_rate_customer_level"] = float(frame.drop_duplicates("Customer ID")["Churn"].mean())
    result["churn_rate_row_level"] = float(frame["Churn"].mean())
    result["returns_missing_count"] = int(frame["Returns"].isna().sum())
    result["returns_observed_count"] = int(frame["Returns"].notna().sum())
    result["return_rate_among_observed"] = float(frame["Returns"].mean())
    result["return_rate_if_missing_zero"] = float(frame["Returns"].fillna(0).mean())
    if "Returns_was_missing" in frame:
        if not frame["Returns_was_missing"].eq(frame["Returns"].isna().astype("int8")).all():
            raise ValueError("Returns_was_missing indicator is inconsistent")
    result["returns_indicator_available"] = "Returns_was_missing" in frame.columns
    result["customer_names_present"] = "Customer Name" in frame.columns
    result["synthetic_data"] = True  # provenance, NOT statistical inference from this frame
    return result


def assert_quality_gates(m: dict) -> None:
    """Hard-fail only on well-supported structural/validity constraints.

    Unknown currency and inconsistent Total vs Price*Quantity are documented as
    unresolved semantics; neither is "fixed" by deleting or changing records.
    """
    blocking = {
        "empty_dataset": m["rows"] == 0,
        "exact_duplicates": m["duplicate_rows"] > 0,
        "non_unique_event_key": m["duplicate_customer_timestamp"] > 0,
        "age_outside_plausible_range": m["age_outside_0_120"] > 0,
        "nonpositive_product_price": m["nonpositive_price"] > 0,
        "nonpositive_quantity": m["nonpositive_quantity"] > 0,
        "negative_reported_amount": m["negative_reported_amount"] > 0,
        "invalid_churn_domain": m["invalid_churn"] > 0,
        "invalid_returns_domain": m["invalid_returns"] > 0,
        "inconsistent_customer_churn": m["churn_inconsistent_customer_ids"] > 0,
        "inconsistent_customer_age": m["age_inconsistent_customer_ids"] > 0,
        "inconsistent_customer_gender": m["gender_inconsistent_customer_ids"] > 0,
        "hidden_missing_text_tokens": bool(m["hidden_missing_tokens"]),
    }
    issues = [name for name, failed in blocking.items() if failed]
    if issues:
        raise ValueError("Data quality gates failed: " + ", ".join(issues))


def produce_report(before: dict, after: dict | None, source_error: str | None = None) -> str:
    a = after or before
    missing_rate = a["returns_missing_count"] / a["rows"] * 100
    correct = a["amount_equal_count"] / a["rows"] * 100
    status = "Đã khôi phục từ Parquet và đối chiếu từng giao dịch" if after else "Audit legacy CSV CHỈ: chưa khôi phục missing Returns"
    legacy_note = (
        "**Cảnh báo:** Legacy CSV đã điền missing Returns thành 0; bảng dưới chỉ phản ánh CSV đã biến đổi. "
        "Số missing thực trước imputation KHÔNG thể suy ra từ CSV này. Để có các tỷ lệ trả hàng đúng, "
        "cần chạy lệnh không có `--audit-only` để đọc Parquet.\n\n"
        if after is None else ""
    )
    return f"""# Phase 1 — Data Integrity & Validation

**Input:** `data/processed/ecommerce_customer_cleaned.csv` và `data/processed/ecommerce_clean.parquet`.
**Trạng thái Returns:** {status}.

{legacy_note}## Số liệu đã kiểm toán

| Phép đo | Giá trị |
|---|---:|
| Bản ghi | {a['rows']:,} |
| Khách hàng duy nhất | {a['unique_customers']:,} |
| Khoảng ngày mua | {a['min_purchase_date']} tới {a['max_purchase_date']} |
| Trùng toàn dòng | {a['duplicate_rows']:,} |
| Trùng khóa `(Customer ID, Purchase Date)` | {a['duplicate_customer_timestamp']:,} |
| Tổng `Total Purchase Amount` | {a['reported_amount_sum']:,} (đơn vị nguồn chưa xác minh) |
| Tổng `Product Price × Quantity` | {a['line_item_value_calc_sum']:,} (đơn vị nguồn chưa xác minh) |
| Số dòng hai giá trị bằng nhau | {a['amount_equal_count']:,} / {a['rows']:,} ({correct:.4f}%) |
| Tương quan giữa hai giá trị | {a['amount_correlation']:.6f} |
| Tỷ lệ Churn theo khách hàng | {a['churn_rate_customer_level']:.4%} |
| Tỷ lệ Churn theo dòng giao dịch | {a['churn_rate_row_level']:.4%} |
| Returns còn thiếu trong file đang đo | {a['returns_missing_count']:,} ({missing_rate:.2f}%) |
| Tỷ lệ trả hàng trên dòng không thiếu trong file đang đo | {a['return_rate_among_observed']:.4%}{" (KHÔNG phải tỷ lệ trên quan sát gốc)" if not after else ""} |
| Tỷ lệ sau khi giả định missing = 0 | {a['return_rate_if_missing_zero']:.4%} |

## Sáu khía cạnh chất lượng dữ liệu

1. **Completeness:** Returns thiếu {a['returns_missing_count']:,} giá trị ({missing_rate:.2f}%); báo cáo tỷ lệ trả hàng chỉ trên bản ghi không thiếu.
2. **Accuracy:** `Total Purchase Amount` khác `Product Price × Quantity` ở {a['amount_mismatch_count']:,} dòng. Không tự sửa giá trị gốc; chưa có quy tắc đối soát đáng tin cậy.
3. **Consistency:** Churn không nhất quán giữa các giao dịch của một khách: {a['churn_inconsistent_customer_ids']:,} mã; Age: {a['age_inconsistent_customer_ids']:,}; Gender: {a['gender_inconsistent_customer_ids']:,}.
4. **Validity:** Age ngoài [18,70]: {a['age_outside_18_70']:,}; Price <= 0: {a['nonpositive_price']:,}; Quantity <= 0: {a['nonpositive_quantity']:,}; Churn sai miền: {a['invalid_churn']:,}; Returns sai miền: {a['invalid_returns']:,}.
5. **Uniqueness:** Trùng toàn dòng: {a['duplicate_rows']:,}; trùng khóa sự kiện `(Customer ID, Purchase Date)`: {a['duplicate_customer_timestamp']:,}. Khóa này chỉ là khóa kiểm toán, không được tác giả xác nhận là order ID.
6. **Timeliness:** Dataset mô phỏng có ngày mua tới {a['max_purchase_date']}; không được suy rộng thành hành vi hiện tại của người tiêu dùng.

## Quy tắc diễn giải đã chốt cho Phase 1

- **`Total Purchase Amount`:** 'Giá trị được ghi nhận trong trường Total Purchase Amount', không tự động gọi là doanh thu thực tế hoặc đã được đối soát.
- **`line_item_value_calc`:** `Product Price × Quantity`, giá trị tính toán từ hai cột đầu vào, không chứng minh là số tiền thực thanh toán.
- **`Churn`:** tỷ lệ khách hàng duy nhất mang nhãn `Churn=1`.
- **`Returns`:** 1 là trả hàng quan sát được, 0 là không trả hàng quan sát được, NULL là không có dữ liệu. Tuyệt đối không gán NULL=0 khi tính tỷ lệ mô tả.
- **Đơn vị tiền:** chưa xác minh qua metadata gốc, nên không tự thêm ký hiệu `$` hoặc khẳng định USD.
- **Phạm vi:** dữ liệu tổng hợp bằng Faker, không được suy luận kết quả là tình hình thương mại điện tử thực tế.
- **License:** chưa xác minh từ metadata tác giả Kaggle; phải kiểm tra trước khi tái phân phối dataset.
- **Tên khách hàng:** chỉ là trường định danh do Faker tạo, mặc định bỏ khỏi output kiểm toán để giảm phơi lộ dữ liệu không cần thiết.

## Cổng nghiệm thu

- [x] Nguồn và tính chất mô phỏng được xác định.
- [x] Sai lệch hai chỉ tiêu tiền tệ được tính chính xác và cách diễn giải thận trọng được ghi lại.
- [x] Sáu khía cạnh chất lượng dữ liệu được rà soát.
- [{'x' if after else ' '}] Returns missing được khôi phục từ Parquet với đối chiếu khóa sự kiện 1:1.
- [ ] License và đơn vị tiền được xác nhận từ metadata của tác giả (chưa có bằng chứng trực tiếp).
- [ ] Phase 3 sẽ cập nhật dashboard và các mô hình để sử dụng metric được chốt ở trên.
""" + (f"\n**Lỗi khi đọc Parquet:** `{source_error}`\n" if source_error else "")


def run(csv: Path = DEFAULT_CSV, parquet: Path = DEFAULT_PARQUET,
        output: Path = DEFAULT_OUTPUT, report_dir: Path = DEFAULT_REPORT_DIR,
        audit_only: bool = False) -> tuple[dict, dict | None]:
    legacy = load_legacy(csv)
    before = assess(legacy)
    assert_quality_gates(before)
    repaired = None
    after = None
    source_error = None
    if not audit_only:
        if not parquet.exists():
            raise FileNotFoundError(f"Source Parquet not found: {parquet}")
        # Fail closed: if pyarrow is absent or the source does not match, do not emit a 'fixed' file.
        source = pd.read_parquet(parquet)
        repaired = restore_returns(legacy, source)
        after = assess(repaired)
        assert_quality_gates(after)
        if after["rows"] != before["rows"]:
            raise AssertionError("Row count changed during recovery")
        if output.resolve() in {csv.resolve(), parquet.resolve()}:
            raise ValueError("Refusing to overwrite source datasets")
        output.parent.mkdir(parents=True, exist_ok=True)
        repaired.to_csv(output, index=False, encoding="utf-8-sig", date_format="%Y-%m-%d %H:%M:%S")
    else:
        source_error = "audit-only: skip Parquet recovery; original Returns missingness is not recovered"

    report_dir.mkdir(parents=True, exist_ok=True)
    with (report_dir / "data_integrity_metrics.json").open("w", encoding="utf-8") as f:
        json.dump({"legacy_csv": before, "validated": after}, f, ensure_ascii=False, indent=2, allow_nan=False)
    (report_dir / "data_integrity_report.md").write_text(produce_report(before, after, source_error), encoding="utf-8")
    print(f"Rows={before['rows']:,}; customers={before['unique_customers']:,}; "
          f"Amount mismatches={before['amount_mismatch_count']:,}")
    if after is not None:
        print(f"Recovered NULL Returns={after['returns_missing_count']:,}; "
              f"Observed return rate={after['return_rate_among_observed']:.2%}")
        print(f"Validated output: {output}")
    else:
        print("Audit-only: Returns missingness cannot be measured from legacy CSV")
    print(f"Audit report: {report_dir / 'data_integrity_report.md'}")
    return before, after


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--parquet", type=Path, default=DEFAULT_PARQUET)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--report-dir", type=Path, default=DEFAULT_REPORT_DIR)
    parser.add_argument("--audit-only", action="store_true", help="Audit CSV without recovering Returns")
    args = parser.parse_args()
    run(args.csv, args.parquet, args.output, args.report_dir, args.audit_only)


if __name__ == "__main__":
    main()
