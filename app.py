"""Phase 3 Streamlit dashboard, strictly reading the validated Phase 2 CSV.

Run: python -m streamlit run app.py
No legacy CSV fallback, no stale pre-generated PNGs and no real-world revenue claims.
"""
from __future__ import annotations

from io import BytesIO
from pathlib import Path
import importlib

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from src.eda_metrics import (
    CLEAN_PATH, AMOUNT, category_totals, churn_segment_matrix, load_cleaned,
    monthly_amount, summary_metrics,
)

st.set_page_config(page_title="E-Commerce · Data Science Dashboard", page_icon="📊", layout="wide")

st.markdown("""
<style>
.block-container { max-width: 1500px; padding-top: 2rem; }
h1, h2, h3 { letter-spacing: -.02em; }
[data-testid="stMetricValue"] { font-size: 1.65rem; }
[data-testid="stMetricLabel"] { font-weight: 600; }
</style>
""", unsafe_allow_html=True)


@st.cache_data(show_spinner=False, max_entries=3)
def read_data(path: str, stat_ns: int) -> pd.DataFrame:
    # stat_ns invalidates the cache when Phase 2 output is regenerated.
    return load_cleaned(Path(path))


@st.cache_data(show_spinner=False, max_entries=3)
def render_charts(path: str, stat_ns: int) -> dict[str, bytes]:
    # Generate all charts from the SAME validated dataset; never use legacy PNGs.
    df = load_cleaned(Path(path))
    vis = importlib.import_module("src.pipeline.05_explanatory_visualization")
    output = {}
    for key, draw in vis.CHARTS.items():
        figure = draw(df)
        buffer = BytesIO()
        try:
            figure.savefig(buffer, format="png", dpi=130, bbox_inches="tight")
            output[key] = buffer.getvalue()
        finally:
            plt.close(figure)
    return output


st.title("Phân tích hành vi khách hàng thương mại điện tử")
st.caption("INFO3020 · Dữ liệu Kaggle mô phỏng · Phase 3: EDA, Visualization & Dashboard")
st.warning(
    "**Phạm vi diễn giải:** Dữ liệu tổng hợp (synthetic). `Total Purchase Amount` không đối soát "
    "với `Product Price × Quantity` ở phần lớn giao dịch; đơn vị tiền chưa xác minh. "
    "Các tổng số dưới đây không được coi là doanh thu USD thực tế.",
    icon="⚠️",
)

if not CLEAN_PATH.is_file():
    st.error(
        "Chưa tìm thấy `data/processed/ecommerce_customer_clean.csv`. "
        "Hãy đặt CSV Kaggle vào `data/raw/` rồi chạy "
        "`python -m src.run_pipeline`. Không tự động dùng CSV legacy.",
        icon="🛑",
    )
    st.stop()

try:
    version = CLEAN_PATH.stat().st_mtime_ns
    df = read_data(str(CLEAN_PATH), version)
    summary = summary_metrics(df)
except (OSError, ValueError, TypeError) as err:
    st.error(f"Dữ liệu không vượt qua kiểm tra Phase 3: {err}")
    st.stop()

st.caption(f"Nguồn đang dùng: `{CLEAN_PATH.name}` · {summary['transactions']:,} giao dịch · "
           f"{summary['customers']:,} khách hàng · "
           f"{summary['date_min']:%d/%m/%Y} đến {summary['date_max']:%d/%m/%Y}")

row1 = st.columns(4)
row1[0].metric("Tổng giá trị ghi nhận", f"{summary['reported_amount_total']/1e6:,.2f}M",
               help="Tổng Total Purchase Amount; đơn vị tiền chưa xác minh, không gọi là doanh thu thật.")
row1[1].metric("Giao dịch (dòng)", f"{summary['transactions']:,}",
               help="Không có Order ID chuẩn tắc nên chưa khẳng định mỗi dòng tương ứng một đơn hàng độc lập.")
row1[2].metric("Khách hàng duy nhất", f"{summary['customers']:,}")
row1[3].metric("Churn cấp khách hàng", f"{summary['churn_rate_customer']:.2%}",
               help="Một khách hàng được tính một lần; không phải trung bình Churn trên giao dịch.")

row2 = st.columns(4)
row2[0].metric("Return rate (đã quan sát)",
               f"{summary['return_rate_observed']:.2%}" if pd.notna(summary['return_rate_observed']) else "N/A",
               help=f"Chỉ {summary['returns_observed']:,} giao dịch có Returns=0/1.")
row2[1].metric("Returns thiếu", f"{summary['returns_missing']:,}",
               help=f"{summary['returns_missing']/summary['transactions']:.2%} giao dịch có Returns=NULL.")
row2[2].metric("Trung vị giá trị ghi nhận", f"{summary['reported_amount_median']:,.0f}")
row2[3].metric("Tổng Price × Quantity", f"{summary['line_item_value_total']/1e6:,.2f}M",
               help="Giá trị tính toán, không phải tiền thực thanh toán.")

st.caption(
    f"**Đối soát tiền tệ:** {summary['amount_mismatch_count']:,}/{summary['transactions']:,} "
    "bản ghi có `Total Purchase Amount` khác `Product Price × Quantity`. "
    f"**Returns:** {summary['returns_missing']:,} NULL không được gán bằng 0."
)

try:
    images = render_charts(str(CLEAN_PATH), version)
except Exception as err:
    st.error(f"Không tạo được biểu đồ từ dữ liệu sạch: {err}")
    st.stop()

revenue_tab, customers_tab, churn_tab, data_tab = st.tabs([
    "Giá trị ghi nhận & Thời gian", "Phân phối & Nhân khẩu học",
    "Churn khách hàng", "Kiểm toán & Tra cứu",
])

with revenue_tab:
    st.subheader("Giá trị ghi nhận theo danh mục và thời gian")
    col_a, col_b = st.columns(2)
    with col_a:
        st.image(images["category"], use_container_width=True)
        top = category_totals(df).head(2)
        st.caption("Đứng đầu trong mẫu: " + " và ".join(top.index.tolist())
                   + ". Xếp hạng theo tổng trường `Total Purchase Amount`.")
    with col_b:
        st.image(images["monthly"], use_container_width=True)
        months = monthly_amount(df)
        incomplete = months.loc[~months["complete_month"], "Month"]
        if not incomplete.empty:
            st.caption("Tháng biên chưa đầy đủ: " + ", ".join(incomplete.dt.strftime("%m/%Y"))
                       + ". Không so sánh trực tiếp với tháng trọn vẹn.")
    st.info("Tổng theo tháng là tổng giá trị ghi nhận trong dữ liệu mô phỏng, "
            "không phải doanh thu thực tế hay xu hướng kinh doanh được xác minh.")

with customers_tab:
    st.subheader("Phân phối giá trị và nhóm nhân khẩu học")
    col_a, col_b = st.columns(2)
    with col_a:
        st.image(images["distribution"], use_container_width=True)
    with col_b:
        st.image(images["age"], use_container_width=True)
    st.caption("Trung vị theo giao dịch; mỗi khách có thể đóng góp nhiều giao dịch. "
               "Đơn vị tiền và mức đại diện cho dân số thực chưa được xác minh.")

with churn_tab:
    st.subheader("Churn cấp khách hàng theo tổ hợp phân khúc")
    col_a, col_b = st.columns([3, 2])
    with col_a:
        st.image(images["churn"], use_container_width=True)
    with col_b:
        rate, sample_size = churn_segment_matrix(df)
        st.markdown("**Cách đọc bảng:** Mỗi khách được tính tối đa một lần **trong từng ô**. "
                    "Một khách có thể thuộc nhiều ô nếu đã giao dịch qua nhiều danh mục/phương thức.")
        st.metric("Churn trên toàn bộ khách hàng", f"{summary['churn_rate_customer']:.2%}")
        st.caption(f"Khoảng tỷ lệ nhóm: {rate.min().min():.2%} đến {rate.max().max():.2%}. "
                   "Đây là mô tả trên dữ liệu tổng hợp, không chứng minh tác động nhân quả.")
        st.dataframe((rate * 100).round(2).rename_axis("Danh mục"), use_container_width=True)

with data_tab:
    st.subheader("Tra cứu dữ liệu đã xác minh")
    st.caption("Bộ lọc chỉ áp dụng cho bảng bên dưới; các KPI và biểu đồ phía trên "
               "luôn tính trên toàn bộ dataset.")
    col1, col2, col3, col4 = st.columns(4)
    categories = sorted(df["Product Category"].dropna().unique().tolist())
    payments = sorted(df["Payment Method"].dropna().unique().tolist())
    genders = sorted(df["Gender"].dropna().unique().tolist())
    with col1:
        selected_categories = st.multiselect("Danh mục", categories, default=categories)
    with col2:
        selected_payments = st.multiselect("Thanh toán", payments, default=payments)
    with col3:
        selected_genders = st.multiselect("Giới tính", genders, default=genders)
    with col4:
        selected_returns = st.multiselect("Returns", ["0 · Không", "1 · Có", "NULL · Thiếu"],
                                          default=["0 · Không", "1 · Có", "NULL · Thiếu"])
    mask = (df["Product Category"].isin(selected_categories)
            & df["Payment Method"].isin(selected_payments)
            & df["Gender"].isin(selected_genders))
    keep_returns = (df["Returns"].eq(0) & ("0 · Không" in selected_returns)
                    | df["Returns"].eq(1) & ("1 · Có" in selected_returns)
                    | df["Returns"].isna() & ("NULL · Thiếu" in selected_returns))
    filtered = df.loc[mask & keep_returns]
    # Do not expose synthetic names in the public-facing table; ID remains a pseudonymous key.
    safe_columns = [c for c in ["Customer ID", "Purchase Date", "Product Category", "Product Price",
                                "Quantity", AMOUNT, "Payment Method", "Returns",
                                "Returns_was_missing", "Age", "Gender", "Churn"] if c in filtered]
    st.dataframe(filtered[safe_columns].head(100), use_container_width=True, hide_index=True)
    st.caption(f"Hiển thị tối đa 100/{len(filtered):,} giao dịch khớp bộ lọc. "
               "`Customer Name` được ẩn, dữ liệu gốc không bị sửa.")
    with st.expander("Quy tắc tính và giới hạn dữ liệu"):
        st.markdown("""
- **Tổng giá trị ghi nhận:** tổng `Total Purchase Amount`, **không** gọi là doanh thu thực.
- **Giá trị tính theo dòng:** tổng `Product Price × Quantity`, không ghi đè dữ liệu gốc.
- **Churn:** số khách Churn=1 / số Customer ID duy nhất.
- **Return rate:** số dòng Returns=1 / số dòng Returns không thiếu; không điền NULL=0.
- **Thời gian:** tháng đầu/cuối có thể thiếu ngày, không thể dùng làm bằng chứng xu hướng ổn định.
- **Suy rộng:** dataset mô phỏng bằng Faker; chưa xác minh đơn vị tiền tệ và giấy phép.
        """)
        st.code("python -m src.run_pipeline --stage eda", language="powershell")
