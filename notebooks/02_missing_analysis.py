"""
Phân tích missing cho Week 3 (nhiệm vụ #23-#34).

File này tìm missing thường và missing ẩn, khảo sát cơ chế MCAR/MAR/MNAR,
so sánh treatment cho Returns và tạo dataset trung gian sau bước missing.
Các biểu đồ được giữ nhẹ để chạy tốt với 250.000 dòng dữ liệu.
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

REPORT_DIR = PROJECT_ROOT / "reports" / "missing"
FIGURE_DIR = REPORT_DIR / "figures"
OUTPUT_PATH = PROJECT_ROOT / "data" / "processed" / "ecommerce_customer_after_missing.csv"

REPORT_DIR.mkdir(parents=True, exist_ok=True)
FIGURE_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 180)

DISPLAY_LABELS = {
    "Customer ID": "Mã khách hàng",
    "Purchase Date": "Ngày mua",
    "Product Category": "Danh mục sản phẩm",
    "Product Price": "Giá sản phẩm",
    "Quantity": "Số lượng",
    "Total Purchase Amount": "Tổng giá trị mua",
    "Payment Method": "Phương thức thanh toán",
    "Customer Age": "Tuổi khách hàng",
    "Returns": "Trả hàng",
    "Customer Name": "Tên khách hàng",
    "Age": "Tuổi",
    "Gender": "Giới tính",
    "Churn": "Rời bỏ",
}


def section(title: str) -> None:
    print(f"\n=== {title} ===")


def missing_by_group(data: pd.DataFrame, target: str, group: str) -> pd.DataFrame:
    result = (
        data.assign(_missing=data[target].isna())
        .groupby(group, dropna=False, observed=False)["_missing"]
        .agg(n_rows="size", n_missing="sum", pct_missing="mean")
        .reset_index()
    )
    result["pct_missing"] = (result["pct_missing"] * 100).round(4)
    return result.sort_values("pct_missing", ascending=False).reset_index(drop=True)


def treatment_stats(name: str, series: pd.Series) -> dict:
    return {
        "scenario": name,
        "n": int(series.notna().sum()),
        "n_missing": int(series.isna().sum()),
        "mean": round(float(series.mean()), 6),
        "std": round(float(series.std()), 6),
        "count_0": int((series == 0).sum()),
        "count_1": int((series == 1).sum()),
    }


def save_figure(fig, filename: str) -> None:
    """Lưu figure nhất quán, đủ nét nhưng không tăng đáng kể thời gian chạy."""
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / filename, dpi=150, bbox_inches="tight")
    plt.close(fig)


# Phân tích missing luôn bắt đầu từ raw data.
df = load_ecommerce_raw()
target = "Returns"

section("Các cột có giá trị thiếu")

missing_summary = pd.DataFrame({
    "n_missing": df.isna().sum(),
    "pct_missing": (df.isna().mean() * 100).round(4),
})
missing_summary = missing_summary[missing_summary["n_missing"] > 0].sort_values(
    "pct_missing", ascending=False
)
print(missing_summary)
missing_summary.to_csv(REPORT_DIR / "missing_summary.csv", encoding="utf-8-sig")


section("Giá trị thiếu ẩn")

hidden_tokens = {
    "", "-", "--", "?", "n/a", "na", "null", "none",
    "unknown", "missing", "nan", "không rõ", "thỏa thuận", "negotiable",
}

hidden_rows = []
detected_na_values = set()

for column in df.select_dtypes(include=["object", "string", "category"]).columns:
    normalized = df[column].astype("string").str.strip().str.casefold()
    mask = normalized.isin(hidden_tokens) & df[column].notna()
    detected = df.loc[mask, column].value_counts().to_dict()

    hidden_rows.append({
        "column": column,
        "n_hidden_missing": int(mask.sum()),
        "detected_values": detected,
    })
    detected_na_values.update(map(str, detected.keys()))

date_values = pd.to_datetime(df["Purchase Date"], errors="coerce")
sentinel_dates = date_values.dt.normalize().isin(
    [pd.Timestamp("1900-01-01"), pd.Timestamp("1970-01-01")]
)

hidden_rows.append({
    "column": "Purchase Date (sentinel)",
    "n_hidden_missing": int(sentinel_dates.sum()),
    "detected_values": (
        df.loc[sentinel_dates, "Purchase Date"].value_counts().to_dict()
        if sentinel_dates.any()
        else {}
    ),
})

hidden_report = pd.DataFrame(hidden_rows)
print(hidden_report.to_string(index=False))
hidden_report.to_csv(
    REPORT_DIR / "hidden_missing_report.csv",
    index=False,
    encoding="utf-8-sig",
)

if detected_na_values:
    print(f"\nna_values gợi ý: {sorted(detected_na_values)}")
else:
    print("\nna_values gợi ý: Không có")


section("Ứng viên sentinel dạng số")

sentinel_rules = {
    "Customer ID": {-9999, -999, -99, -1, 0},
    "Product Price": {-9999, -999, -99, -1},
    "Quantity": {-9999, -999, -99, -1, 0, 999, 9999},
    "Total Purchase Amount": {-9999, -999, -99, -1},
    "Customer Age": {-9999, -999, -99, -1, 999, 9999},
    "Age": {-9999, -999, -99, -1, 999, 9999},
}

sentinel_rows = []
for column, candidates in sentinel_rules.items():
    present = {
        value: int((df[column] == value).sum())
        for value in sorted(candidates)
        if (df[column] == value).any()
    }
    sentinel_rows.append({
        "column": column,
        "candidate_values_found": present,
        "n_candidate_cells": sum(present.values()),
        "classification": (
            "chỉ là ứng viên; cần bằng chứng domain/source trước khi coi là missing"
            if present
            else "không phát hiện"
        ),
    })

numeric_sentinel_report = pd.DataFrame(sentinel_rows)
print(numeric_sentinel_report.to_string(index=False))
numeric_sentinel_report.to_csv(
    REPORT_DIR / "numeric_sentinel_candidates.csv",
    index=False,
    encoding="utf-8-sig",
)


section("Mẫu hình missing")

# Biểu đồ 1: mức độ đầy đủ theo từng cột.
completeness = (1 - df.isna().mean()).sort_values() * 100
fig, ax = plt.subplots(figsize=(10, 6.5))
labels = [DISPLAY_LABELS.get(column, column) for column in completeness.index]
bars = ax.barh(labels, completeness.values)
ax.set_title("Mức độ đầy đủ của từng cột")
ax.set_xlabel("Tỷ lệ đầy đủ (%)")
ax.set_ylabel("Cột dữ liệu")
ax.set_xlim(0, 105)
ax.xaxis.set_major_formatter(PercentFormatter(xmax=100, decimals=0))
ax.grid(axis="x", alpha=0.25)
for bar, value in zip(bars, completeness.values):
    ax.text(min(value + 0.6, 104), bar.get_y() + bar.get_height() / 2,
            f"{value:.1f}%", va="center", fontsize=8)
save_figure(fig, "missing_completeness_bar.png")

# Biểu đồ 2: ma trận missing trên một mẫu ngẫu nhiên để nhìn pattern nhanh.
sample = df.sample(min(5000, len(df)), random_state=42)
matrix = sample.isna().astype(int).T
fig, ax = plt.subplots(figsize=(12, 6.5))
image = ax.imshow(matrix, aspect="auto", interpolation="nearest")
ax.set_title("Ma trận giá trị thiếu (mẫu tối đa 5.000 dòng)")
ax.set_xlabel("Dòng trong mẫu")
ax.set_ylabel("Cột dữ liệu")
ax.set_yticks(range(len(matrix.index)))
ax.set_yticklabels([DISPLAY_LABELS.get(column, column) for column in matrix.index])
colorbar = fig.colorbar(image, ax=ax, pad=0.01)
colorbar.set_ticks([0, 1])
colorbar.set_ticklabels(["Có dữ liệu", "Thiếu"])
save_figure(fig, "missing_matrix.png")

# Heatmap chỉ có ý nghĩa khi có ít nhất hai cột bị missing.
missing_columns = df.columns[df.isna().any()].tolist()
if len(missing_columns) < 2:
    print("Bỏ qua heatmap tương quan missingness vì chỉ có một cột bị thiếu.")
else:
    corr = df[missing_columns].isna().corr()
    fig, ax = plt.subplots(figsize=(7, 6))
    image = ax.imshow(corr, vmin=-1, vmax=1)
    ax.set_title("Tương quan missingness giữa các cột")
    ax.set_xticks(range(len(corr.columns)))
    ax.set_xticklabels([DISPLAY_LABELS.get(column, column) for column in corr.columns], rotation=45, ha="right")
    ax.set_yticks(range(len(corr.index)))
    ax.set_yticklabels([DISPLAY_LABELS.get(column, column) for column in corr.index])
    fig.colorbar(image, ax=ax, label="Hệ số tương quan")
    save_figure(fig, "missing_heatmap.png")


section("Điều tra missing của Returns")

analysis_df = df.copy()
analysis_df["Purchase Year"] = date_values.dt.year
analysis_df["Age Band"] = pd.cut(
    analysis_df["Age"],
    bins=[0, 24, 34, 44, 54, 64, np.inf],
    labels=["<=24", "25-34", "35-44", "45-54", "55-64", "65+"],
)

group_columns = [
    "Product Category",
    "Payment Method",
    "Gender",
    "Churn",
    "Quantity",
    "Purchase Year",
    "Age Band",
]

group_reports = []
group_effect_rows = []

for group in group_columns:
    report = missing_by_group(analysis_df, target, group)
    group_effect_rows.append({
        "variable": group,
        "min_missing_pct": float(report["pct_missing"].min()),
        "max_missing_pct": float(report["pct_missing"].max()),
        "spread_pp": round(
            float(report["pct_missing"].max() - report["pct_missing"].min()), 4
        ),
    })

    report.insert(0, "group_variable", group)
    report = report.rename(columns={group: "group_value"})
    group_reports.append(report)

missing_group_report = pd.concat(group_reports, ignore_index=True)
group_effect_report = pd.DataFrame(group_effect_rows).sort_values(
    "spread_pp", ascending=False
)

print("\nTỷ lệ missing theo các nhóm quan sát được:")
print(group_effect_report.to_string(index=False))

missing_group_report.to_csv(
    REPORT_DIR / "returns_missing_by_group.csv",
    index=False,
    encoding="utf-8-sig",
)
group_effect_report.to_csv(
    REPORT_DIR / "returns_group_effects.csv",
    index=False,
    encoding="utf-8-sig",
)

numeric_columns = [
    "Product Price",
    "Quantity",
    "Total Purchase Amount",
    "Age",
]

missing_mask = df[target].isna()
numeric_effect_rows = []

for column in numeric_columns:
    observed = df.loc[~missing_mask, column]
    missing = df.loc[missing_mask, column]
    scale = df[column].std()
    smd = 0.0 if scale == 0 else (missing.mean() - observed.mean()) / scale

    numeric_effect_rows.append({
        "variable": column,
        "observed_mean": round(float(observed.mean()), 4),
        "missing_mean": round(float(missing.mean()), 4),
        "observed_median": round(float(observed.median()), 4),
        "missing_median": round(float(missing.median()), 4),
        "standardized_mean_difference": round(float(smd), 4),
    })

numeric_effect_report = pd.DataFrame(numeric_effect_rows)
print("\nKhác biệt numeric giữa nhóm Returns quan sát được và nhóm bị thiếu:")
print(numeric_effect_report.to_string(index=False))
numeric_effect_report.to_csv(
    REPORT_DIR / "returns_numeric_effects.csv",
    index=False,
    encoding="utf-8-sig",
)

customer_missing = (
    df.assign(_missing=missing_mask)
    .groupby("Customer ID")["_missing"]
    .agg(n_transactions="size", n_missing="sum", pct_missing="mean")
)
customer_missing["pct_missing"] *= 100

print(
    "\nKhách hàng có toàn bộ Returns bị thiếu:",
    f"{int((customer_missing['pct_missing'] == 100).sum()):,}",
)
print(
    "Khách hàng không thiếu Returns      :",
    f"{int((customer_missing['pct_missing'] == 0).sum()):,}",
)

customer_missing.reset_index().to_csv(
    REPORT_DIR / "returns_missing_by_customer.csv",
    index=False,
    encoding="utf-8-sig",
)

year_report = missing_by_group(analysis_df, target, "Purchase Year").sort_values("Purchase Year")
fig, ax = plt.subplots(figsize=(8, 5))
bars = ax.bar(year_report["Purchase Year"].astype(str), year_report["pct_missing"])
ax.set_title("Tỷ lệ thiếu Returns theo năm mua hàng")
ax.set_xlabel("Năm mua hàng")
ax.set_ylabel("Tỷ lệ thiếu Returns (%)")
ax.yaxis.set_major_formatter(PercentFormatter(xmax=100, decimals=0))
ax.grid(axis="y", alpha=0.25)
ax.set_ylim(0, max(25, float(year_report["pct_missing"].max()) + 3))
for bar, value in zip(bars, year_report["pct_missing"]):
    ax.text(bar.get_x() + bar.get_width() / 2, value + 0.25,
            f"{value:.2f}%", ha="center", va="bottom", fontsize=9)
save_figure(fig, "returns_missing_by_year.png")


section("Đánh giá cơ chế missing")

# Ngưỡng heuristic chỉ dùng để định hướng giả thuyết MCAR/MAR.
MAR_GROUP_SPREAD_THRESHOLD_PP = 5.0
MAR_SMD_THRESHOLD = 0.10

max_group_spread = float(group_effect_report["spread_pp"].max())
max_abs_smd = float(
    numeric_effect_report["standardized_mean_difference"].abs().max()
)

print(
    "Ngưỡng heuristic: "
    f"group spread >= {MAR_GROUP_SPREAD_THRESHOLD_PP:.1f} pp or "
    f"|SMD| >= {MAR_SMD_THRESHOLD:.2f}."
)
print("Các ngưỡng này chỉ hỗ trợ giả thuyết làm việc, không phải bằng chứng thống kê tuyệt đối.")

if (
    max_group_spread >= MAR_GROUP_SPREAD_THRESHOLD_PP
    or max_abs_smd >= MAR_SMD_THRESHOLD
):
    mechanism = "MAR (giả thuyết làm việc)"
    mechanism_reason = (
        "Missingness cho thấy quan hệ đáng kể với các biến quan sát được. "
        "Điều này ủng hộ MAR như giả thuyết làm việc chính."
    )
else:
    mechanism = "MCAR (giả thuyết làm việc, chưa chứng minh tuyệt đối)"
    mechanism_reason = (
        "Không phát hiện quan hệ mạnh với các biến quan sát được. "
        "MCAR chỉ là giả thuyết làm việc, không phải kết luận chứng minh tuyệt đối."
    )

print(f"Cơ chế missing của Returns: {mechanism}")
print(f"Độ chênh missing lớn nhất giữa nhóm: {max_group_spread:.4f} điểm phần trăm")
print(f"|SMD| lớn nhất                    : {max_abs_smd:.4f}")
print(mechanism_reason)
print(
    "Không thể loại trừ MNAR chỉ từ dataset này vì chính các giá trị Returns bị thiếu "
    "không thể quan sát được."
)

mechanism_report = pd.DataFrame([{
    "column": target,
    "mechanism": mechanism,
    "max_group_spread_pp": round(max_group_spread, 4),
    "max_abs_smd": round(max_abs_smd, 4),
    "reason": mechanism_reason,
    "limitation": (
        "Không thể chứng minh tuyệt đối MCAR/MAR/MNAR chỉ từ bảng dữ liệu; "
        "cần metadata về quá trình sinh dữ liệu để kết luận chắc chắn."
    ),
}])
mechanism_report.to_csv(
    REPORT_DIR / "missing_mechanism.csv",
    index=False,
    encoding="utf-8-sig",
)


section("So sánh các phương án xử lý missing")

observed_only = df[target].dropna()
fill_zero = df[target].fillna(0)
fill_one = df[target].fillna(1)

comparison = pd.DataFrame([
    treatment_stats("observed_only", observed_only),
    treatment_stats("fill_0", fill_zero),
    treatment_stats("fill_1", fill_one),
])

print(comparison.to_string(index=False))
comparison.to_csv(
    REPORT_DIR / "treatment_comparison.csv",
    index=False,
    encoding="utf-8-sig",
)

distribution = pd.DataFrame({
    "observed_only": observed_only.value_counts(normalize=True).reindex([0, 1], fill_value=0),
    "fill_0": fill_zero.value_counts(normalize=True).reindex([0, 1], fill_value=0),
    "fill_1": fill_one.value_counts(normalize=True).reindex([0, 1], fill_value=0),
})

x = np.arange(2)
width = 0.25
legend_labels = {
    "observed_only": "Chỉ dữ liệu quan sát được",
    "fill_0": "Điền missing bằng 0",
    "fill_1": "Điền missing bằng 1",
}

fig, ax = plt.subplots(figsize=(9, 5.5))
for i, column in enumerate(distribution.columns):
    values = distribution[column].values * 100
    bars = ax.bar(x + (i - 1) * width, values, width, label=legend_labels[column])
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 0.8,
                f"{value:.1f}%", ha="center", va="bottom", fontsize=8)

ax.set_title("Ảnh hưởng của các phương án điền missing lên phân bố Returns")
ax.set_xticks(x)
ax.set_xticklabels(["Returns = 0", "Returns = 1"])
ax.set_xlabel("Giá trị Returns")
ax.set_ylabel("Tỷ lệ (%)")
ax.yaxis.set_major_formatter(PercentFormatter(xmax=100, decimals=0))
ax.set_ylim(0, min(100, max(65, float((distribution * 100).to_numpy().max()) + 8)))
ax.grid(axis="y", alpha=0.25)
ax.legend(title="Kịch bản")
save_figure(fig, "returns_treatment_comparison.png")


section("Treatment được chọn")

# Treatment đã chọn: giữ NaN và bổ sung cờ missing, không bịa giá trị 0/1.
treated = df.copy()
treated["Returns_was_missing"] = treated[target].isna().astype("int8")

decision = pd.DataFrame([{
    "column": target,
    "working_mechanism": mechanism,
    "chosen_treatment": "Thêm missing indicator và giữ nguyên NaN",
    "affected_rows": int(treated["Returns_was_missing"].sum()),
    "reason": (
        "Returns là biến nhị phân. Không có bằng chứng nguồn rằng NaN đồng nghĩa 'không trả hàng', "
        "nên điền toàn bộ missing bằng 0 hoặc 1 sẽ tạo ra outcome không có căn cứ. "
        "Missing indicator giữ lại thông tin missing trong khi vẫn bảo toàn giá trị gốc."
    ),
    "modeling_note": (
        "Nếu Returns được dùng làm feature ở bước modeling, mọi imputation phải được fit "
        "bên trong training pipeline sau khi chia train/test."
    ),
}])

print(decision.to_string(index=False))
decision.to_csv(
    REPORT_DIR / "treatment_decision.csv",
    index=False,
    encoding="utf-8-sig",
)


section("So sánh trước và sau xử lý")

before_after = pd.DataFrame([
    {
        "metric": "số dòng",
        "before": len(df),
        "after": len(treated),
    },
    {
        "metric": "số cột",
        "before": df.shape[1],
        "after": treated.shape[1],
    },
    {
        "metric": "Returns bị thiếu",
        "before": int(df[target].isna().sum()),
        "after": int(treated[target].isna().sum()),
    },
    {
        "metric": "Mean Returns quan sát được",
        "before": round(float(df[target].mean()), 6),
        "after": round(float(treated[target].mean()), 6),
    },
    {
        "metric": "Std Returns quan sát được",
        "before": round(float(df[target].std()), 6),
        "after": round(float(treated[target].std()), 6),
    },
    {
        "metric": "Returns_was_missing = 1",
        "before": 0,
        "after": int(treated["Returns_was_missing"].sum()),
    },
])

print(before_after.to_string(index=False))

returns_unchanged = df[target].equals(treated[target])
other_columns_unchanged = df.equals(treated[df.columns])

print(f"\nReturns không thay đổi     : {returns_unchanged}")
print(f"Dữ liệu gốc không thay đổi: {other_columns_unchanged}")

if not returns_unchanged or not other_columns_unchanged:
    raise AssertionError("Phát hiện thay đổi dữ liệu ngoài dự kiến trong bước xử lý missing.")

before_after.to_csv(
    REPORT_DIR / "before_after.csv",
    index=False,
    encoding="utf-8-sig",
)

treated.to_csv(OUTPUT_PATH, index=False, encoding="utf-8-sig")
print(f"\nĐã lưu dataset trung gian: {OUTPUT_PATH}")
print(f"Đã lưu các báo cáo       : {REPORT_DIR}")
