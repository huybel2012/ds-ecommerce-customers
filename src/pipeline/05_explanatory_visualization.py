"""Trực quan hóa dữ liệu
1. Doanh thu lũy kế theo Ngành hàng 
2. Phân phối Giá trị Đơn hàng
3. Ma trận Tỷ lệ Churn
4. Mức chi tiêu theo Độ tuổi & Giới tính
5. Chuỗi thời gian Doanh thu theo Tháng
"""

from pathlib import Path
import sys

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

CLEAN_CANDIDATES = [
    PROJECT_ROOT / "data" / "processed" / "ecommerce_customer_clean.csv",
    PROJECT_ROOT / "data" / "processed" / "ecommerce_customer_cleaned.csv",
]

INPUT_PATH = next((p for p in CLEAN_CANDIDATES if p.exists()), None)
if not INPUT_PATH:
    raise FileNotFoundError("Không tìm thấy file dữ liệu sạch trong data/processed/.")

FIGURE_DIR = PROJECT_ROOT / "reports" / "figures"
FIGURE_DIR.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(INPUT_PATH)
df["Purchase Date"] = pd.to_datetime(df["Purchase Date"], errors="coerce")


plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Segoe UI", "Arial", "DejaVu Sans"],
        "figure.facecolor": "#ffffff",
        "axes.facecolor": "#ffffff",
        "axes.edgecolor": "#cbd5e1",
        "axes.linewidth": 0.8,
    }
)



# Doanh thu theo Ngành hàng

def plot_category_revenue(data: pd.DataFrame) -> None:
    cat_rev = (
        data.groupby("Product Category")["Total Purchase Amount"]
        .sum()
        .sort_values(ascending=True)
        .div(1e6)
    )

    fig, ax = plt.subplots(figsize=(9, 4.5), dpi=300)
    max_val = cat_rev.max()
    colors = [
        "#1d4ed8" if val == max_val else "#94a3b8" for val in cat_rev.values
    ]
    bars = ax.barh(
        cat_rev.index, cat_rev.values, color=colors, height=0.55, zorder=3
    )

    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.spines["bottom"].set_color("#cbd5e1")
    ax.grid(axis="x", linestyle="--", alpha=0.5, color="#e2e8f0", zorder=0)
    ax.grid(axis="y", visible=False)
    ax.set_xticks([])
    ax.set_xlim(0, max_val * 1.18)

    for bar in bars:
        w = bar.get_width()
        is_lead = w == max_val
        ax.text(
            w + (max_val * 0.015),
            bar.get_y() + bar.get_height() / 2,
            f"${w:,.2f}M",
            va="center",
            ha="left",
            fontsize=9.5,
            fontweight="bold" if is_lead else "normal",
            color="#0f172a" if is_lead else "#64748b",
        )

    ax.text(
        0,
        1.18,
        "Electronics và Clothing dẫn đầu tổng doanh thu bán lẻ",
        transform=ax.transAxes,
        fontsize=12.5,
        fontweight="bold",
        color="#0f172a",
    )
    ax.text(
        0,
        1.08,
        "Tổng giá trị đơn hàng tích lũy phân theo ngành hàng (Triệu USD)",
        transform=ax.transAxes,
        fontsize=9.5,
        color="#64748b",
    )

    ax.tick_params(axis="y", labelsize=10, length=0, colors="#1e293b")
    fig.text(
        0.02,
        -0.04,
        "Nguồn: E-commerce Customer Dataset (250.000 giao dịch) | Module 05",
        fontsize=8,
        color="#94a3b8",
    )
    fig.savefig(
        FIGURE_DIR / "01_category_revenue.png", dpi=300, bbox_inches="tight"
    )
    plt.close(fig)
    print("✓ Đã xuất: 01_category_revenue.png")



#Phân phối Giá trị Đơn hàng

def plot_purchase_distribution(data: pd.DataFrame) -> None:
    amounts = data["Total Purchase Amount"].dropna()

    fig, ax = plt.subplots(figsize=(9, 4.8), dpi=300)
    n, bins, patches = ax.hist(
        amounts,
        bins=30,
        color="#bae6fd",
        edgecolor="#ffffff",
        linewidth=0.8,
        zorder=3,
    )

    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="x", visible=False)
    ax.grid(axis="y", linestyle="--", alpha=0.5, color="#e2e8f0", zorder=0)

    for i in range(len(patches)):
        if bins[i] >= 500 and bins[i + 1] <= 5100:
            patches[i].set_facecolor("#38bdf8")

    ax.annotate(
        "Dải chi tiêu cốt lõi ($500 – $5,000):\nSố lượng đơn hàng dàn trải đồng đều,\nkhông có khoảng giá nào áp đảo",
        xy=(2750, 7500),
        xytext=(1500, 8500),
        arrowprops=dict(
            arrowstyle="->",
            color="#0284c7",
            lw=1.2,
            connectionstyle="arc3,rad=.15",
        ),
        bbox=dict(
            boxstyle="round,pad=0.6",
            facecolor="#f0f9ff",
            edgecolor="#bae6fd",
            linewidth=0.8,
        ),
        fontsize=9,
        color="#0369a1",
        fontweight="medium",
    )

    ax.text(
        0,
        1.18,
        "Quy mô đơn hàng phân bổ đều từ $500 đến $5,000 (Phân phối đều - Uniform)",
        transform=ax.transAxes,
        fontsize=12.5,
        fontweight="bold",
        color="#0f172a",
    )
    ax.text(
        0,
        1.08,
        "Số lượng giao dịch theo từng mức giá; tỷ lệ đơn hàng phát sinh tương đương nhau trên toàn dải giá",
        transform=ax.transAxes,
        fontsize=9.5,
        color="#64748b",
    )

    ax.set_xlabel("Tổng giá trị đơn hàng (USD)", fontsize=9.5, color="#475569")
    ax.set_ylabel("Số lượng giao dịch", fontsize=9.5, color="#475569")
    ax.set_xlim(0, 5500)
    ax.set_ylim(0, 9800)

    fig.text(
        0.02,
        -0.04,
        "Nguồn: E-commerce Customer Dataset (250.000 giao dịch) | Module 05",
        fontsize=8,
        color="#94a3b8",
    )
    fig.savefig(
        FIGURE_DIR / "02_purchase_amount_distribution.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)
    print("✓ Đã xuất: 02_purchase_amount_distribution.png")



#Ma trận Tỷ lệ Churn

def plot_churn_cross_matrix(data: pd.DataFrame) -> None:
    churn_matrix = (
        pd.crosstab(
            data["Product Category"],
            data["Payment Method"],
            values=data["Churn"],
            aggfunc="mean",
        )
        * 100
    )

    fig, ax = plt.subplots(figsize=(8, 4.8), dpi=300)
    ax.grid(False)

    sns.heatmap(
        churn_matrix,
        annot=True,
        fmt=".1f",
        cmap="Blues",
        vmin=0,
        vmax=40,
        linewidths=2.0,
        linecolor="#ffffff",
        cbar_kws={
            "label": "Tỷ lệ rời bỏ - Churn (%)",
            "shrink": 0.85,
            "ticks": [0, 10, 20, 30, 40],
        },
        annot_kws={"size": 10.5, "color": "#0f172a", "weight": "bold"},
        ax=ax,
    )

    ax.text(
        0,
        1.18,
        "Tỷ lệ khách hàng rời bỏ đồng đều (~20%) trên mọi ngành hàng và cổng thanh toán",
        transform=ax.transAxes,
        fontsize=12,
        fontweight="bold",
        color="#0f172a",
    )
    ax.text(
        0,
        1.08,
        "Tỷ lệ Churn Rate (%) thực tế chênh lệch không đáng kể (< 2%) giữa các kênh giao dịch",
        transform=ax.transAxes,
        fontsize=9.5,
        color="#64748b",
    )

    ax.set_xlabel("Phương thức thanh toán", fontsize=9.5, color="#475569")
    ax.set_ylabel("Danh mục sản phẩm", fontsize=9.5, color="#475569")
    ax.tick_params(axis="both", labelsize=9.5, length=0)

    fig.text(
        0.02,
        -0.04,
        "Nguồn: E-commerce Customer Dataset (250.000 giao dịch) | Module 05",
        fontsize=8,
        color="#94a3b8",
    )
    fig.savefig(
        FIGURE_DIR / "03_churn_rate_heatmap.png", dpi=300, bbox_inches="tight"
    )
    plt.close(fig)
    print("✓ Đã xuất: 03_churn_rate_heatmap.png")



#Mức chi tiêu theo Độ tuổi & Giới tính

def plot_spending_by_age_gender(data: pd.DataFrame) -> None:
    plot_df = data.copy()
    plot_df["Age Band"] = pd.cut(
        plot_df["Age"],
        bins=[17, 25, 35, 45, 55, 65, 100],
        labels=["18-25", "26-35", "36-45", "46-55", "56-65", "Trên 65"],
    )

    summary = (
        plot_df.groupby(["Age Band", "Gender"], observed=False)[
            "Total Purchase Amount"
        ]
        .median()
        .reset_index()
    )

    fig, ax = plt.subplots(figsize=(9, 4.8), dpi=300)
    palette = {"Male": "#2563eb", "Female": "#ec4899"}

    sns.barplot(
        data=summary,
        x="Age Band",
        y="Total Purchase Amount",
        hue="Gender",
        palette=palette,
        ax=ax,
        width=0.6,
        edgecolor="#ffffff",
        linewidth=0.8,
        zorder=3,
    )

    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="x", visible=False)
    ax.grid(axis="y", linestyle="--", alpha=0.5, color="#e2e8f0", zorder=0)

    max_y = summary["Total Purchase Amount"].max()
    ax.set_ylim(0, max_y * 1.22)

    for p in ax.patches:
        h = p.get_height()
        if h > 0:
            ax.annotate(
                f"${h:,.0f}",
                (p.get_x() + p.get_width() / 2.0, h),
                ha="center",
                va="bottom",
                fontsize=8.5,
                color="#475569",
                xytext=(0, 3),
                textcoords="offset points",
            )

    ax.text(
        0,
        1.18,
        "Mức chi tiêu trung vị duy trì cân bằng giữa các thế hệ và giới tính",
        transform=ax.transAxes,
        fontsize=12,
        fontweight="bold",
        color="#0f172a",
    )
    ax.text(
        0,
        1.08,
        "So sánh giá trị đơn hàng trung vị (USD) phân nhóm theo độ tuổi và giới tính",
        transform=ax.transAxes,
        fontsize=9.5,
        color="#64748b",
    )

    ax.set_xlabel("Nhóm độ tuổi", fontsize=9.5, color="#475569")
    ax.set_ylabel("Chi tiêu trung vị (USD)", fontsize=9.5, color="#475569")
    ax.legend(title="", frameon=False, loc="upper right")

    fig.text(
        0.02,
        -0.04,
        "Nguồn: E-commerce Customer Dataset (250.000 giao dịch) | Module 05",
        fontsize=8,
        color="#94a3b8",
    )
    fig.savefig(
        FIGURE_DIR / "04_spending_by_age_gender.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)
    print("✓ Đã xuất: 04_spending_by_age_gender.png")



#Chuỗi thời gian Doanh thu

def plot_monthly_revenue_trend(data: pd.DataFrame) -> None:
    monthly = (
        data.dropna(subset=["Purchase Date"])
        .assign(
            Month=lambda d: d["Purchase Date"]
            .dt.to_period("M")
            .dt.to_timestamp()
        )
        .groupby("Month")["Total Purchase Amount"]
        .sum()
        .div(1e6)
        .reset_index()
    )

    fig, ax = plt.subplots(figsize=(10.5, 5), dpi=300)

    
    ax.grid(
        axis="x",
        linestyle="--",
        alpha=0.6,
        color="#cbd5e1",
        linewidth=0.9,
        zorder=0,
    )
    ax.grid(
        axis="y",
        linestyle="--",
        alpha=0.4,
        color="#e2e8f0",
        linewidth=0.8,
        zorder=0,
    )

    
    ax.plot(
        monthly["Month"],
        monthly["Total Purchase Amount"],
        color="#1d4ed8",
        linewidth=2.2,
        zorder=3,
    )
    ax.fill_between(
        monthly["Month"],
        monthly["Total Purchase Amount"],
        color="#38bdf8",
        alpha=0.12,
        zorder=2,
    )

   
    max_idx = monthly["Total Purchase Amount"].idxmax()
    max_date = monthly.loc[max_idx, "Month"]
    max_rev = monthly.loc[max_idx, "Total Purchase Amount"]

    
    ax.vlines(
        x=max_date,
        ymin=0,
        ymax=max_rev,
        color="#dc2626",
        linestyle=":",
        linewidth=1.6,
        zorder=3,
    )
    ax.scatter([max_date], [max_rev], color="#dc2626", s=55, zorder=4)

    ax.annotate(
        f"Đỉnh doanh thu: ${max_rev:.1f}M\n(Tháng 12/2020)",
        xy=(max_date, max_rev),
        xytext=(max_date - pd.Timedelta(days=160), max_rev + 0.6),
        arrowprops=dict(
            arrowstyle="->",
            color="#dc2626",
            lw=1.2,
            connectionstyle="arc3,rad=-.1",
        ),
        fontsize=9,
        fontweight="bold",
        color="#991b1b",
    )

   
    last_idx = monthly.index[-1]
    last_date = monthly.loc[last_idx, "Month"]
    last_rev = monthly.loc[last_idx, "Total Purchase Amount"]

    ax.scatter([last_date], [last_rev], color="#ea580c", s=45, zorder=4)
    ax.annotate(
        "Kỳ thu thập kết thúc\n(Tháng chưa trọn vẹn)",
        xy=(last_date, last_rev),
        xytext=(last_date - pd.Timedelta(days=200), last_rev + 1.8),
        arrowprops=dict(
            arrowstyle="->",
            color="#ea580c",
            lw=1.2,
            connectionstyle="arc3,rad=.15",
        ),
        fontsize=8.5,
        fontweight="bold",
        color="#c2410c",
    )

    
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%m/%Y"))

    ax.spines[["top", "right"]].set_visible(False)
    ax.set_ylim(0, 18.5)

    ax.text(
        0,
        1.18,
        "Doanh thu duy trì biên độ ổn định (~$15M/tháng) xuyên suốt giai đoạn 2020 - 2023",
        transform=ax.transAxes,
        fontsize=12.5,
        fontweight="bold",
        color="#0f172a",
    )
    ax.text(
        0,
        1.08,
        "Đường lưới dọc nét đứt hỗ trợ đối chiếu chính xác các mốc thời gian (Đơn vị: Triệu USD)",
        transform=ax.transAxes,
        fontsize=9.5,
        color="#64748b",
    )

    ax.set_xlabel("Mốc thời gian (Tháng/Năm)", fontsize=9.5, color="#475569")
    ax.set_ylabel("Doanh thu (Triệu USD)", fontsize=9.5, color="#475569")
    ax.tick_params(axis="x", labelsize=9, rotation=0, colors="#1e293b")

    fig.text(
        0.02,
        -0.05,
        "Nguồn: E-commerce Customer Dataset (250.000 giao dịch) | Module 05",
        fontsize=8,
        color="#94a3b8",
    )
    fig.savefig(
        FIGURE_DIR / "05_monthly_revenue_trend.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)
    print("✓ Đã xuất: 05_monthly_revenue_trend.png")


def main() -> None:
    print(f"Đang đọc dữ liệu từ: {INPUT_PATH.name}")

    
    for old_file in FIGURE_DIR.glob("*.*"):
        old_file.unlink()

    print("\n=== ĐANG XUẤT 5 BIỂU ĐỒ BÁO CÁO CỐT LÕI ===")
    plot_category_revenue(df)
    plot_purchase_distribution(df)
    plot_churn_cross_matrix(df)
    plot_spending_by_age_gender(df)
    plot_monthly_revenue_trend(df)

    print(
        f"\n✓ Hoàn tất! Danh sách 5 biểu đồ sạch sẽ đã lưu tại: {FIGURE_DIR}\n"
    )


if __name__ == "__main__":
    main()