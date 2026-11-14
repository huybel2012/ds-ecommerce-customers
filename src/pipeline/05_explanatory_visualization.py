"""Phase 3 EDA visualizations from *only* the Phase 2 validated clean CSV.

Usage:
  python -m src.pipeline.05_explanatory_visualization
  python -m src.pipeline.05_explanatory_visualization --output-dir tmp/figures

Generates exactly figures 01–05. NEVER deletes W9/W10 figures 06–09.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # Windows terminal / headless CI
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Keep compatibility with Phase 2 runner, which invokes this file by path.
if __package__ in (None, ""):
    root = Path(__file__).resolve().parents[2]
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

from src.eda_metrics import (
    AMOUNT, CLEAN_PATH, age_gender_median, category_totals, churn_segment_matrix,
    load_cleaned, monthly_amount, summary_metrics, validate_eda_data,
)
from src.utils import REPO_ROOT

FIGURES_DIR = REPO_ROOT / "reports" / "figures"
REPORT_DIR = REPO_ROOT / "reports" / "phase3"

FIGURE_NAMES = {
    "category": "01_category_revenue.png",  # historical name kept for dashboard/report references
    "distribution": "02_purchase_amount_distribution.png",
    "churn": "03_churn_rate_heatmap.png",
    "age": "04_spending_by_age_gender.png",
    "monthly": "05_monthly_revenue_trend.png",
}

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans", "Arial", "Segoe UI"],
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.titlesize": 12,
    "axes.titleweight": "bold",
    "savefig.facecolor": "white",
})


def _finish(fig, ax, subtitle: str = ""):
    if subtitle:
        ax.set_title(ax.get_title() + "\n" + subtitle, fontsize=11, pad=15)
    fig.tight_layout(pad=2)
    return fig


def plot_category_revenue(df: pd.DataFrame):
    amounts = category_totals(df) / 1e6
    asc = amounts.sort_values()
    fig, ax = plt.subplots(figsize=(9, 4.5))
    color = ["#2563eb" if c == amounts.index[0] else "#9ca3af" for c in asc.index]
    ax.barh(asc.index, asc.values, color=color, height=.65)
    ax.set_xlim(0, float(asc.max()) * 1.22)
    for i, v in enumerate(asc.values):
        ax.text(v + asc.max() * .015, i, f"{v:,.2f}M", va="center", fontsize=10)
    leaders = " & ".join(amounts.index[:2])
    ax.set_title(f"{leaders}: hai danh mục có tổng giá trị ghi nhận cao nhất")
    ax.set_xlabel("Tổng giá trị ghi nhận (triệu; đơn vị tiền chưa xác minh)")
    ax.grid(axis="x", alpha=.16)
    ax.set_axisbelow(True)
    return _finish(fig, ax)


def plot_purchase_distribution(df: pd.DataFrame):
    values = df[AMOUNT].dropna()
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.hist(values, bins=30, color="#93c5fd", edgecolor="white")
    median = values.median()
    ax.axvline(median, color="#1d4ed8", linestyle="--", linewidth=1.5,
               label=f"Trung vị: {median:,.0f}")
    ax.set_title("Phân phối Total Purchase Amount theo giao dịch")
    ax.set_xlabel("Giá trị ghi nhận (đơn vị tiền chưa xác minh)")
    ax.set_ylabel("Số giao dịch")
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=.16)
    ax.set_axisbelow(True)
    return _finish(fig, ax)


def plot_churn_cross_matrix(df: pd.DataFrame):
    rates, counts = churn_segment_matrix(df)
    fig, ax = plt.subplots(figsize=(9, 4.8))
    img = ax.imshow(rates * 100, vmin=0, vmax=100, cmap="Blues", aspect="auto")
    ax.set_xticks(range(len(rates.columns)), labels=rates.columns, rotation=0)
    ax.set_yticks(range(len(rates.index)), labels=rates.index)
    for i in range(len(rates.index)):
        for j in range(len(rates.columns)):
            rate = rates.iloc[i, j]
            n = counts.iloc[i, j]
            label = f"{rate * 100:.1f}%\n(n={n:,.0f})" if pd.notna(rate) else "N/A"
            ax.text(j, i, label, ha="center", va="center", fontsize=9,
                    color="white" if pd.notna(rate) and rate > .55 else "#0f172a")
    fig.colorbar(img, ax=ax, fraction=.04, pad=.04, label="Churn theo KH trong nhóm (%) · thang 0–100")
    ax.set_title("Tỷ lệ Churn của khách hàng trong từng tổ hợp danh mục × thanh toán")
    ax.set_xlabel("Phương thức thanh toán")
    ax.set_ylabel("Danh mục sản phẩm")
    fig.text(.02, -.01, "Một khách hàng có thể xuất hiện ở nhiều nhóm; không suy ra quan hệ nhân quả.",
             fontsize=8, color="#475569")
    return _finish(fig, ax)


def plot_spending_by_age_gender(df: pd.DataFrame):
    summary = age_gender_median(df)
    fig, ax = plt.subplots(figsize=(9, 4.8))
    idx = np.arange(len(summary))
    genders = list(summary.columns)
    width = .78 / max(len(genders), 1)
    colors = {"Female": "#ec4899", "Male": "#2563eb"}
    for j, gender in enumerate(genders):
        xx = idx - .39 + width * (j + .5)
        vals = summary[gender].to_numpy(dtype=float)
        bars = ax.bar(xx, vals, width=width * .95, label=gender,
                      color=colors.get(gender, "#94a3b8"))
        for b, v in zip(bars, vals):
            if np.isfinite(v):
                ax.text(b.get_x() + b.get_width()/2, v + np.nanmax(summary.to_numpy())*.012,
                        f"{v:,.0f}", ha="center", fontsize=8)
    ax.set_xticks(idx, labels=summary.index.astype(str))
    ax.set_title("Trung vị Total Purchase Amount theo nhóm tuổi và giới tính")
    ax.set_xlabel("Nhóm tuổi")
    ax.set_ylabel("Trung vị (đơn vị tiền chưa xác minh)")
    ax.set_ylim(0, float(np.nanmax(summary.to_numpy())) * 1.23)
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=.16)
    ax.set_axisbelow(True)
    return _finish(fig, ax)


def plot_monthly_revenue_trend(df: pd.DataFrame):
    monthly = monthly_amount(df)
    fig, ax = plt.subplots(figsize=(10, 4.8))
    x = monthly["Month"]
    y = monthly["amount_total"] / 1e6
    full = monthly["complete_month"]
    # Do not connect a partial boundary month as if it were a full-month collapse.
    ax.plot(x[full], y[full], color="#2563eb", lw=2, marker="o", markersize=2.5)
    partial = monthly.loc[~full]
    if not partial.empty:
        ax.scatter(partial["Month"], partial["amount_total"] / 1e6,
                   color="#ea580c", zorder=5, s=52, label="Tháng chưa đầy đủ")
    complete = monthly.loc[full]
    if not complete.empty:
        top = complete.loc[complete["amount_total"].idxmax()]
        ax.scatter(top["Month"], top["amount_total"] / 1e6, color="#dc2626", zorder=4)
        ax.annotate(f"Đỉnh tháng đủ dữ liệu: {top['Month']:%m/%Y}",
                    xy=(top["Month"], top["amount_total"] / 1e6),
                    xytext=(0, 12), textcoords="offset points", ha="center", fontsize=9)
    ax.set_title("Giá trị ghi nhận theo tháng (không chuẩn hóa theo số ngày)")
    ax.set_xlabel("Tháng")
    ax.set_ylabel("Giá trị ghi nhận (triệu; đơn vị tiền chưa xác minh)")
    # Headroom keeps the dynamic peak annotation inside the plotting area.
    ax.set_ylim(0, float(y.max()) * 1.22 if len(y) else 1)
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%m/%Y"))
    ax.grid(alpha=.16)
    if not partial.empty:
        ax.legend(frameon=False, loc="lower left")
    fig.autofmt_xdate(rotation=0)
    return _finish(fig, ax)


CHARTS = {
    "category": plot_category_revenue,
    "distribution": plot_purchase_distribution,
    "churn": plot_churn_cross_matrix,
    "age": plot_spending_by_age_gender,
    "monthly": plot_monthly_revenue_trend,
}


def export_figures(df: pd.DataFrame, output_dir: Path = FIGURES_DIR) -> list[Path]:
    """Overwrite only 01–05. Preserve figures 06–09 and any other reports."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    produced = []
    for key, fn in CHARTS.items():
        fig = fn(df)
        target = output_dir / FIGURE_NAMES[key]
        try:
            fig.savefig(target, dpi=150, bbox_inches="tight")
        finally:
            plt.close(fig)
        produced.append(target)
    return produced


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=CLEAN_PATH)
    parser.add_argument("--output-dir", type=Path, default=FIGURES_DIR)
    args = parser.parse_args(argv)
    df = load_cleaned(args.input)
    figures = export_figures(df, args.output_dir)
    results = summary_metrics(df)
    manifest = {
        "phase": 3, "generated_utc": datetime.now(timezone.utc).isoformat(),
        "source": str(args.input.resolve()), "source_sha256": _sha256(args.input),
        "rows": results["transactions"], "customers": results["customers"],
        "churn_rate_customer": results["churn_rate_customer"],
        "returns_observed": results["returns_observed"],
        "returns_missing": results["returns_missing"],
        "charts": [p.name for p in figures],
        "note": "Figures show synthetic recorded amounts, NOT verified business USD revenue.",
    }
    # Save manifest next to generated figures, so custom output-dir never mutates repo.
    target = Path(args.output_dir) / "phase3_visualization_manifest.json"
    target.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("Phase 3: generated figures from", args.input)
    for f in figures:
        print(" -", f)
    print("Manifest:", target)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
