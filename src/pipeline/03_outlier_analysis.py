"""
Bao gồm point outlier, contextual outlier, collective outlier và noise audit.
Các detector chỉ dùng để gắn cờ điều tra, không tự động xóa hay cap dữ liệu.
"""

from pathlib import Path
import sys

import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils import load_ecommerce_raw

INPUT_PATH = PROJECT_ROOT / "data" / "processed" / "ecommerce_customer_after_missing.csv"
OUTPUT_PATH = PROJECT_ROOT / "data" / "processed" / "ecommerce_customer_after_outliers.csv"

REPORT_DIR = PROJECT_ROOT / "reports" / "outliers"
FIGURE_DIR = REPORT_DIR / "figures"

REPORT_DIR.mkdir(parents=True, exist_ok=True)
FIGURE_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 200)


COLUMN_LABELS = {
    "Product Price": "Giá sản phẩm",
    "Quantity": "Số lượng",
    "Total Purchase Amount": "Tổng giá trị mua hàng",
    "Age": "Tuổi",
}


def section(title: str) -> None:
    print(f"\n=== {title} ===")


def iqr_mask(series: pd.Series, multiplier: float = 1.5):
    clean = series.dropna()
    q1, q3 = clean.quantile([0.25, 0.75])
    iqr = q3 - q1
    lower = q1 - multiplier * iqr
    upper = q3 + multiplier * iqr
    mask = (series < lower) | (series > upper)
    return mask.fillna(False), q1, q3, iqr, lower, upper


def modified_z_mask(series: pd.Series, threshold: float = 3.5):
    clean = series.dropna()
    median = clean.median()
    mad = (clean - median).abs().median()

    if mad == 0 or pd.isna(mad):
        score = pd.Series(0.0, index=series.index)
        return pd.Series(False, index=series.index), score, median, mad

    score = 0.6745 * (series - median) / mad
    mask = score.abs() > threshold
    return mask.fillna(False), score, median, mad


def z_score_mask(series: pd.Series, threshold: float = 3.0):
    clean = series.dropna()
    mean = clean.mean()
    std = clean.std()

    if std == 0 or pd.isna(std):
        score = pd.Series(0.0, index=series.index)
        return pd.Series(False, index=series.index), score, mean, std

    score = (series - mean) / std
    mask = score.abs() > threshold
    return mask.fillna(False), score, mean, std


def save_distribution_plots(data: pd.DataFrame, column: str) -> None:
    """Tạo 4 biểu đồ nhẹ, dễ đọc cho một biến số."""
    series = data[column].dropna()
    name = column.lower().replace(" ", "_")
    label = COLUMN_LABELS.get(column, column)

    # Histogram: nhìn nhanh hình dạng phân bố và vị trí trung vị.
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(series, bins=40)
    median = float(series.median())
    ax.axvline(median, linestyle="--", linewidth=1.2, label=f"Trung vị = {median:,.1f}")
    ax.set_title(f"Phân phối {label}")
    ax.set_xlabel(label)
    ax.set_ylabel("Tần suất")
    ax.grid(axis="y", alpha=0.2)
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / f"{name}_histogram.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # Boxplot: hỗ trợ đọc nhanh median, IQR và các điểm nằm ngoài whisker.
    fig, ax = plt.subplots(figsize=(8, 3.5))
    ax.boxplot(series, orientation="horizontal")
    ax.set_title(f"Boxplot của {label}")
    ax.set_xlabel(label)
    ax.grid(axis="x", alpha=0.2)
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / f"{name}_boxplot.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # Scatter theo index chỉ lấy tối đa 30.000 điểm để giữ hiệu suất.
    scatter_data = data[[column]].dropna()
    if len(scatter_data) > 30000:
        scatter_data = scatter_data.sample(30000, random_state=42).sort_index()

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.scatter(scatter_data.index, scatter_data[column], s=6, alpha=0.35)
    ax.set_title(f"{label} theo thứ tự dòng (mẫu tối đa 30.000 điểm)")
    ax.set_xlabel("Chỉ số dòng")
    ax.set_ylabel(label)
    ax.grid(alpha=0.15)
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / f"{name}_scatter_index.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # ECDF giúp nhìn phần đuôi phân bố mà không phụ thuộc số lượng bins.
    sorted_values = np.sort(series.to_numpy())
    ecdf = np.arange(1, len(sorted_values) + 1) / len(sorted_values)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(sorted_values, ecdf)
    ax.set_title(f"ECDF của {label}")
    ax.set_xlabel(label)
    ax.set_ylabel("Tỷ lệ tích lũy")
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=1, decimals=0))
    ax.set_ylim(0, 1.02)
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / f"{name}_ecdf.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


if not INPUT_PATH.exists():
    raise FileNotFoundError(
        f"Không tìm thấy dataset trung gian: {INPUT_PATH}\n"
        "Hãy chạy 02_missing_analysis.py trước 03_outlier_analysis.py."
    )

# Pipeline phải đi qua bước missing trước khi phân tích outlier.
df = pd.read_csv(INPUT_PATH)
source_name = INPUT_PATH.name
raw_df = load_ecommerce_raw()

if len(df) != len(raw_df):
    raise AssertionError("Số dòng của dataset trung gian không khớp raw data.")

if not df["Customer ID"].reset_index(drop=True).equals(
    raw_df["Customer ID"].reset_index(drop=True)
):
    raise AssertionError("Thứ tự dòng giữa dataset trung gian và raw data không được bảo toàn.")

# Chỉ phân tích các biến số có ý nghĩa định lượng; bỏ ID và biến nhị phân.
numeric_columns = [
    "Product Price",
    "Quantity",
    "Total Purchase Amount",
    "Age",
]

section("Dataset đầu vào")
print(f"Nguồn dữ liệu: {source_name}")
print(f"Kích thước   : {df.shape[0]:,} dòng x {df.shape[1]} cột")
print(f"Các cột số được phân tích: {numeric_columns}")


section("Kiểm tra trực quan")

distribution_rows = []

for column in numeric_columns:
    save_distribution_plots(df, column)

    series = df[column]
    distribution_rows.append({
        "column": column,
        "min": series.min(),
        "q1": series.quantile(0.25),
        "median": series.median(),
        "q3": series.quantile(0.75),
        "max": series.max(),
        "mean": round(float(series.mean()), 4),
        "std": round(float(series.std()), 4),
        "skew": round(float(series.skew()), 4),
    })

distribution_report = pd.DataFrame(distribution_rows)
print(distribution_report.to_string(index=False))
distribution_report.to_csv(
    REPORT_DIR / "distribution_summary.csv",
    index=False,
    encoding="utf-8-sig",
)


section("Point outlier: Z-score, IQR và Modified Z-score")

method_rows = []
flag_frames = []
candidate_scores = {}

for column in numeric_columns:
    z_outlier, z_score, z_mean, z_std = z_score_mask(df[column])
    iqr_outlier, q1, q3, iqr, lower, upper = iqr_mask(df[column])
    mz_outlier, mz_score, median, mad = modified_z_mask(df[column])

    union = z_outlier | iqr_outlier | mz_outlier
    all_three = z_outlier & iqr_outlier & mz_outlier

    method_rows.append({
        "column": column,
        "z_mean": z_mean,
        "z_std": z_std,
        "z_flagged": int(z_outlier.sum()),
        "iqr_q1": q1,
        "iqr_q3": q3,
        "iqr": iqr,
        "iqr_lower": lower,
        "iqr_upper": upper,
        "iqr_flagged": int(iqr_outlier.sum()),
        "modified_z_median": median,
        "mad": mad,
        "modified_z_flagged": int(mz_outlier.sum()),
        "all_three_flagged": int(all_three.sum()),
        "union_flagged": int(union.sum()),
        "z_only": int((z_outlier & ~iqr_outlier & ~mz_outlier).sum()),
        "iqr_only": int((iqr_outlier & ~z_outlier & ~mz_outlier).sum()),
        "modified_z_only": int((mz_outlier & ~z_outlier & ~iqr_outlier).sum()),
    })

    candidate_scores[column] = pd.concat(
        [
            z_score.abs().rename("z"),
            mz_score.abs().rename("mz"),
        ],
        axis=1,
    ).max(axis=1)

    flagged = pd.DataFrame({
        "row_index": df.index,
        "column": column,
        "value": df[column],
        "z_flag": z_outlier,
        "iqr_flag": iqr_outlier,
        "modified_z_flag": mz_outlier,
        "abs_z": z_score.abs(),
        "abs_modified_z": mz_score.abs(),
    })

    flagged = flagged[
        flagged["z_flag"]
        | flagged["iqr_flag"]
        | flagged["modified_z_flag"]
    ]

    if not flagged.empty:
        flag_frames.append(flagged)

method_comparison = pd.DataFrame(method_rows)
print(method_comparison.to_string(index=False))
method_comparison.to_csv(
    REPORT_DIR / "method_comparison.csv",
    index=False,
    encoding="utf-8-sig",
)

if flag_frames:
    all_flagged = pd.concat(flag_frames, ignore_index=True)
else:
    all_flagged = pd.DataFrame(
        columns=[
            "row_index",
            "column",
            "value",
            "z_flag",
            "iqr_flag",
            "modified_z_flag",
            "abs_z",
            "abs_modified_z",
        ]
    )

all_flagged.to_csv(
    REPORT_DIR / "flagged_outliers.csv",
    index=False,
    encoding="utf-8-sig",
)


section("Các ứng viên point outlier nổi bật")

if not all_flagged.empty:
    ranked = all_flagged.copy()
    ranked["candidate_score"] = ranked[
        ["abs_z", "abs_modified_z"]
    ].max(axis=1)

    top_candidates = (
        ranked
        .sort_values(["candidate_score", "value"], ascending=[False, False])
        .drop_duplicates(subset=["row_index", "column"])
        .head(5)
        .copy()
    )
else:
    candidate_rows = []

    for column in numeric_columns:
        scores = candidate_scores[column].fillna(0)

        for row_index in scores.nlargest(5).index:
            candidate_rows.append({
                "row_index": row_index,
                "column": column,
                "value": df.at[row_index, column],
                "z_flag": False,
                "iqr_flag": False,
                "modified_z_flag": False,
                "abs_z": 0.0,
                "abs_modified_z": 0.0,
                "candidate_score": float(scores.at[row_index]),
            })

    top_candidates = (
        pd.DataFrame(candidate_rows)
        .sort_values(["candidate_score", "value"], ascending=[False, False])
        .drop_duplicates(subset=["row_index", "column"])
        .head(5)
        .copy()
    )

investigation_rows = []

for _, candidate in top_candidates.iterrows():
    row_index = int(candidate["row_index"])
    column = candidate["column"]
    value = candidate["value"]

    raw_value = raw_df.at[row_index, column]
    source_matches = (
        (pd.isna(value) and pd.isna(raw_value))
        or value == raw_value
    )

    impossible = False
    reason = ""

    if column == "Product Price":
        impossible = value < 0
        reason = "giá âm" if impossible else "nằm trong miền giá hợp lệ"
    elif column == "Quantity":
        impossible = value <= 0
        reason = "số lượng không dương" if impossible else "nằm trong miền số lượng hợp lệ"
    elif column == "Total Purchase Amount":
        impossible = value < 0
        reason = "tổng tiền âm" if impossible else "nằm trong miền tổng tiền hợp lệ"
    elif column == "Age":
        impossible = value < 0 or value > 120
        reason = "tuổi ngoài [0, 120]" if impossible else "nằm trong miền tuổi hợp lệ"

    is_error = bool(impossible)
    verdict = "Lỗi dữ liệu" if is_error else "Chưa có bằng chứng là lỗi"
    decision = (
        "Đối chiếu nguồn trước khi sửa/xóa"
        if is_error
        else "Giữ nguyên"
    )

    investigation_rows.append({
        "row_index": row_index,
        "column": column,
        "value": value,
        "z_flag": bool(candidate.get("z_flag", False)),
        "iqr_flag": bool(candidate.get("iqr_flag", False)),
        "modified_z_flag": bool(candidate.get("modified_z_flag", False)),
        "candidate_score": round(float(candidate["candidate_score"]), 4),
        "same_value_in_raw": bool(source_matches),
        "validity_check": reason,
        "is_error": is_error,
        "verdict": verdict,
        "decision": decision,
    })

investigation_report = pd.DataFrame(investigation_rows)
print(investigation_report.to_string(index=False))
investigation_report.to_csv(
    REPORT_DIR / "top5_investigation.csv",
    index=False,
    encoding="utf-8-sig",
)


section("Contextual outlier")

contextual_rows = []
contextual_summary_rows = []

for column in ["Product Price", "Total Purchase Amount"]:
    for category, group in df.groupby("Product Category", dropna=False):
        if len(group) < 20:
            continue

        mask, q1, q3, iqr, lower, upper = iqr_mask(group[column])
        flagged = group.loc[mask, [column]]

        contextual_summary_rows.append({
            "context": "Product Category",
            "group": category,
            "column": column,
            "n_rows": len(group),
            "q1": q1,
            "q3": q3,
            "iqr": iqr,
            "lower": lower,
            "upper": upper,
            "n_flagged": len(flagged),
        })

        for row_index, row in flagged.iterrows():
            contextual_rows.append({
                "row_index": row_index,
                "context": "Product Category",
                "group": category,
                "column": column,
                "value": row[column],
                "lower": lower,
                "upper": upper,
                "decision": "Chỉ điều tra; không tự động xóa",
            })

contextual_summary = pd.DataFrame(contextual_summary_rows)
contextual_outliers = pd.DataFrame(contextual_rows)

print(
    contextual_summary[
        ["context", "group", "column", "n_rows", "n_flagged"]
    ].to_string(index=False)
)

contextual_summary.to_csv(
    REPORT_DIR / "contextual_outlier_summary.csv",
    index=False,
    encoding="utf-8-sig",
)
contextual_outliers.to_csv(
    REPORT_DIR / "contextual_outliers.csv",
    index=False,
    encoding="utf-8-sig",
)


section("Collective outlier ở cấp khách hàng")

purchase_date = pd.to_datetime(df["Purchase Date"], errors="coerce")

# Collective outlier được đánh giá trên hành vi tổng hợp theo Customer ID.
customer_behavior = (
    df.assign(_purchase_date=purchase_date)
    .groupby("Customer ID")
    .agg(
        transaction_count=("Customer ID", "size"),
        total_spend=("Total Purchase Amount", "sum"),
        avg_purchase_amount=("Total Purchase Amount", "mean"),
        total_quantity=("Quantity", "sum"),
        avg_product_price=("Product Price", "mean"),
        category_count=("Product Category", "nunique"),
        first_purchase=("_purchase_date", "min"),
        last_purchase=("_purchase_date", "max"),
    )
    .reset_index()
)

customer_behavior["purchase_span_days"] = (
    customer_behavior["last_purchase"]
    - customer_behavior["first_purchase"]
).dt.days

collective_metrics = [
    "transaction_count",
    "total_spend",
    "avg_purchase_amount",
    "total_quantity",
    "avg_product_price",
    "category_count",
    "purchase_span_days",
]

collective_masks = pd.DataFrame(
    False,
    index=customer_behavior.index,
    columns=collective_metrics,
)

collective_method_rows = []

for column in collective_metrics:
    iqr_flag, _, _, _, _, _ = iqr_mask(customer_behavior[column])
    mz_flag, _, _, _ = modified_z_mask(customer_behavior[column])

    union = iqr_flag | mz_flag
    collective_masks[column] = union

    collective_method_rows.append({
        "metric": column,
        "iqr_flagged": int(iqr_flag.sum()),
        "modified_z_flagged": int(mz_flag.sum()),
        "union_flagged": int(union.sum()),
    })

customer_behavior["n_collective_flags"] = collective_masks.sum(axis=1)
collective_candidates = customer_behavior[
    customer_behavior["n_collective_flags"] > 0
].sort_values(
    ["n_collective_flags", "total_spend", "transaction_count"],
    ascending=False,
)

collective_summary = pd.DataFrame(collective_method_rows)

print(collective_summary.to_string(index=False))
print(f"Số khách hàng là ứng viên collective outlier: {len(collective_candidates):,}")

collective_summary.to_csv(
    REPORT_DIR / "collective_outlier_summary.csv",
    index=False,
    encoding="utf-8-sig",
)
collective_candidates.to_csv(
    REPORT_DIR / "collective_customer_candidates.csv",
    index=False,
    encoding="utf-8-sig",
)


section("Đánh giá noise")

profile_conflicts = {}

for column in ["Customer Name", "Age", "Gender", "Churn"]:
    profile_conflicts[column] = int(
        (df.groupby("Customer ID")[column].nunique(dropna=False) > 1).sum()
    )

noise_rows = []

for column in numeric_columns:
    series = df[column].dropna()
    unique_values = np.sort(series.unique())
    positive_steps = np.diff(unique_values)
    positive_steps = positive_steps[positive_steps > 0]

    min_step = (
        float(positive_steps.min())
        if len(positive_steps)
        else np.nan
    )

    integer_valued = bool(
        np.allclose(series.to_numpy(), np.round(series.to_numpy()))
    )

    noise_rows.append({
        "column": column,
        "integer_valued": integer_valued,
        "n_unique": int(series.nunique()),
        "min_observed_step": min_step,
        "smoothing_applied": False,
        "assessment": (
            "Không thấy bằng chứng về measurement noise cần smoothing; điều này không chứng minh dữ liệu hoàn toàn không có noise."
        ),
    })

noise_report = pd.DataFrame(noise_rows)

print(noise_report.to_string(index=False))
print("Xung đột hồ sơ khách hàng:", profile_conflicts)

noise_report.to_csv(
    REPORT_DIR / "noise_assessment.csv",
    index=False,
    encoding="utf-8-sig",
)

pd.DataFrame(
    [
        {
            "field": key,
            "customer_ids_with_conflict": value,
        }
        for key, value in profile_conflicts.items()
    ]
).to_csv(
    REPORT_DIR / "profile_stability.csv",
    index=False,
    encoding="utf-8-sig",
)


section("Quyết định xử lý outlier")

n_flagged_cells = len(all_flagged)
n_flagged_rows = (
    int(all_flagged["row_index"].nunique())
    if not all_flagged.empty
    else 0
)
verified_errors = int(investigation_report["is_error"].sum())
all_point_flags_investigated = n_flagged_cells <= len(investigation_report)

if n_flagged_cells == 0:
    treatment = "Giữ nguyên"
    treatment_reason = (
        "Không có point outlier nào bị gắn cờ bởi Z-score, IQR hoặc Modified Z-score. "
        "Các cờ contextual và collective chỉ được giữ ở mức ứng viên điều tra."
    )
elif verified_errors > 0:
    treatment = "Đối chiếu nguồn trước khi sửa"
    treatment_reason = (
        "Có ít nhất một point candidate vi phạm hard validity rule. "
        "Chỉ sửa hoặc xóa khi đã xác minh là lỗi nguồn/nhập liệu."
    )
elif all_point_flags_investigated:
    treatment = "Giữ các point flag đã điều tra"
    treatment_reason = (
        "Tất cả point flag đã được điều tra và không vi phạm hard validity rule. "
        "Cờ thống kê đơn thuần không được xem là lỗi dữ liệu."
    )
else:
    treatment = "Tiếp tục điều tra các point flag còn lại"
    treatment_reason = (
        "Mới chỉ điều tra thủ công các point flag xếp hạng cao nhất. "
        "Không suy rộng verdict của top candidate sang các flag chưa được review."
    )

treatment_report = pd.DataFrame([{
    "point_flagged_cells_union": n_flagged_cells,
    "point_flagged_rows_union": n_flagged_rows,
    "contextual_candidates": len(contextual_outliers),
    "collective_customer_candidates": len(collective_candidates),
    "top_point_candidates_investigated": len(investigation_report),
    "verified_errors_in_top5": verified_errors,
    "all_point_flags_investigated": all_point_flags_investigated,
    "chosen_treatment": treatment,
    "reason": treatment_reason,
    "modeling_note": (
        "Mọi capping, clipping hoặc fitted transform mang tính thống kê phải được học "
        "chỉ từ training data."
    ),
}])

print(treatment_report.to_string(index=False))
treatment_report.to_csv(
    REPORT_DIR / "treatment_decision.csv",
    index=False,
    encoding="utf-8-sig",
)


section("So sánh trước và sau xử lý outlier")

# Không tự động xóa/cap outlier; treatment hiện tại giữ nguyên dữ liệu.
treated = df.copy()
before_after_rows = []

for column in numeric_columns:
    before_after_rows.append({
        "column": column,
        "rows_before": len(df),
        "rows_after": len(treated),
        "min_before": df[column].min(),
        "min_after": treated[column].min(),
        "max_before": df[column].max(),
        "max_after": treated[column].max(),
        "mean_before": round(float(df[column].mean()), 6),
        "mean_after": round(float(treated[column].mean()), 6),
        "median_before": round(float(df[column].median()), 6),
        "median_after": round(float(treated[column].median()), 6),
        "std_before": round(float(df[column].std()), 6),
        "std_after": round(float(treated[column].std()), 6),
        "changed_values": int((df[column] != treated[column]).sum()),
    })

before_after = pd.DataFrame(before_after_rows)
print(before_after.to_string(index=False))
before_after.to_csv(
    REPORT_DIR / "before_after.csv",
    index=False,
    encoding="utf-8-sig",
)

original_unchanged = df.equals(treated)
print(f"\nCác giá trị gốc không thay đổi: {original_unchanged}")

if not original_unchanged:
    raise AssertionError("Phát hiện thay đổi dữ liệu ngoài dự kiến trong bước phân tích outlier.")

treated.to_csv(
    OUTPUT_PATH,
    index=False,
    encoding="utf-8-sig",
)

decision_log = investigation_report[
    ["row_index", "column", "value", "verdict", "decision"]
].copy()

decision_log["stage"] = "Phân tích outlier"
decision_log["reason"] = (
    "Đã điều tra bằng classical Z-score, IQR và Modified Z-score; "
    "cờ thống kê đơn thuần không được xem là lỗi."
)

decision_log.to_csv(
    REPORT_DIR / "outlier_decisions.csv",
    index=False,
    encoding="utf-8-sig",
)

print(f"\nĐã lưu dataset trung gian: {OUTPUT_PATH}")
print(f"Đã lưu các báo cáo       : {REPORT_DIR}")
