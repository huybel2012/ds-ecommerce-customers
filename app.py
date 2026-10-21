from pathlib import Path
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="E-Commerce Executive Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed",
)

PROJECT_ROOT = Path(__file__).resolve().parent
CLEAN_CANDIDATES = [
    PROJECT_ROOT / "data" / "processed" / "ecommerce_customer_clean.csv",
    PROJECT_ROOT / "data" / "processed" / "ecommerce_customer_cleaned.csv",
]
DATA_PATH = next((p for p in CLEAN_CANDIDATES if p.exists()), None)
FIGURES_DIR = PROJECT_ROOT / "reports" / "figures"

@st.cache_data
def load_data():
    if not DATA_PATH:
        return None
    data = pd.read_csv(DATA_PATH)
    data["Purchase Date"] = pd.to_datetime(
        data["Purchase Date"], errors="coerce"
    )
    return data

df = load_data()

if df is None:
    st.error("Không tìm thấy file dữ liệu sạch trong thư mục data/processed/.")
    st.stop()

# ==============================================================================
# CSS TÙY BIẾN GIAO DIỆN & ÉP CĂN GIỮA NÚT TAB THÀNH THẺ NỔI
# ==============================================================================
st.markdown(
    """
    <style>
    /* 1. Mặc định ẩn nút Hamburger & Nút mở Sidebar góc trên */
    [data-testid="stSidebarCollapsedControl"] {
        display: none !important;
    }
    
    /* 2. Tiêu đề căn chính giữa */
    .dashboard-header {
        text-align: center;
        margin-bottom: 1.5rem;
        width: 100%;
    }
    .dashboard-title {
        font-size: 1.85rem;
        font-weight: 700;
        color: #f8fafc;
        margin-bottom: 0.3rem;
        letter-spacing: -0.02em;
    }
    .dashboard-subtitle {
        color: #94a3b8;
        font-size: 0.95rem;
    }

    /* 3. CĂN GIỮA HÀNG 4 NÚT TAB VÀ TẠO KHOẢNG CÁCH DỄ NHÌN */
    div[data-testid="stTabs"] > div:first-child,
    div[data-testid="stTabs"] div[role="tablist"],
    div[data-testid="stTabs"] div[data-baseweb="tab-list"] {
        display: flex !important;
        justify-content: center !important;
        align-items: center !important;
        gap: 18px !important;
        border-bottom: none !important;
        padding: 10px 0 25px 0 !important;
        width: 100% !important;
    }

    /* 4. TRIỆT TIÊU TOÀN BỘ VẠCH GẠCH ĐỎ MẶC ĐỊNH */
    div[data-testid="stTabs"] div[data-baseweb="tab-highlight"],
    div[data-testid="stTabs"] div[data-baseweb="tab-border"],
    div[data-baseweb="tab-highlight"],
    div[data-baseweb="tab-border"] {
        display: none !important;
        height: 0px !important;
        background: transparent !important;
    }

    /* 5. ĐỊNH HÌNH NÚT TAB THÀNH THẺ NỔI (CARDS) BO GÓC */
    div[data-testid="stTabs"] button[role="tab"],
    div[data-testid="stTabs"] button[data-baseweb="tab"] {
        background-color: #1e293b !important;
        color: #94a3b8 !important;
        border: 1px solid #334155 !important;
        border-radius: 10px !important;
        padding: 10px 24px !important;
        margin: 0 !important;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.3) !important;
        transition: all 0.25s cubic-bezier(0.4, 0, 0.2, 1) !important;
        cursor: pointer !important;
    }

    div[data-testid="stTabs"] button[role="tab"] p,
    div[data-testid="stTabs"] button[data-baseweb="tab"] p {
        font-size: 0.92rem !important;
        font-weight: 600 !important;
        color: inherit !important;
        margin: 0 !important;
    }

    /* 6. HIỆU ỨNG HOVER RÊ CHUỘT VÀO NÚT */
    div[data-testid="stTabs"] button[role="tab"]:hover,
    div[data-testid="stTabs"] button[data-baseweb="tab"]:hover {
        background-color: #334155 !important;
        color: #ffffff !important;
        border-color: #60a5fa !important;
        transform: translateY(-3px) !important;
        box-shadow: 0 8px 18px rgba(96, 165, 250, 0.25) !important;
    }

    /* 7. NÚT ĐANG ĐƯỢC CHỌN (ACTIVE) */
    div[data-testid="stTabs"] button[role="tab"][aria-selected="true"],
    div[data-testid="stTabs"] button[data-baseweb="tab"][aria-selected="true"] {
        background-color: #2563eb !important;
        color: #ffffff !important;
        border-color: #3b82f6 !important;
        transform: translateY(-1px) !important;
        box-shadow: 0 6px 16px rgba(37, 99, 235, 0.45) !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# JavaScript ép đóng Sidebar trên trình duyệt
st.components.v1.html(
    """
    <script>
    function forceCloseSidebar() {
        const parentDoc = window.parent.document;
        const sidebar = parentDoc.querySelector('section[data-testid="stSidebar"]');
        const collapseBtn = parentDoc.querySelector('button[kind="header"]');
        if (sidebar && sidebar.getAttribute('aria-expanded') === 'true' && collapseBtn) {
            collapseBtn.click();
        }
    }
    setTimeout(forceCloseSidebar, 100);
    setTimeout(forceCloseSidebar, 500);
    </script>
    """,
    height=0,
)

st.markdown(
    """
    <div class="dashboard-header">
        <div class="dashboard-title">Báo Cáo Phân Tích Hành Vi Khách Hàng E-Commerce</div>
        <div class="dashboard-subtitle">Hệ thống kiểm toán chất lượng dữ liệu và trực quan hóa phân tích kinh doanh điều hành.</div>
    </div>
    """,
    unsafe_allow_html=True,
)

col1, col2, col3, col4 = st.columns(4)

total_rev = df["Total Purchase Amount"].sum() / 1e6
total_orders = len(df)
median_spend = df["Total Purchase Amount"].median()
churn_rate = df["Churn"].mean() * 100

with col1:
    st.metric(
        "Tổng Doanh Thu",
        f"${total_rev:,.2f}M",
        help="Tổng giá trị đơn hàng tích lũy",
    )
with col2:
    st.metric(
        "Tổng Số Đơn Hàng",
        f"{total_orders:,}",
        help="Tổng số giao dịch được ghi nhận",
    )
with col3:
    st.metric(
        "Chi Tiêu Trung Vị",
        f"${median_spend:,.0f}",
        help="Mức chi tiêu đơn hàng trung vị",
    )
with col4:
    st.metric(
        "Tỷ Lệ Rời Bỏ (Churn)",
        f"{churn_rate:.1f}%",
        delta=f"{churn_rate - 20.0:.2f}% vs trung bình",
        delta_color="inverse",
    )

st.divider()

tab_revenue, tab_customer, tab_churn, tab_data = st.tabs([
    "Doanh Thu & Thời Gian",
    "Khách Hàng & Phân Phối",
    "Tỷ Lệ Rời Bỏ (Churn Rate)",
    "Tra Cứu Dữ Liệu Sạch",
])

with tab_revenue:
    st.subheader("1. Tổng quan Doanh thu theo Ngành hàng & Mốc Thời gian")
    col_a, col_b = st.columns(2)
    with col_a:
        img_01 = FIGURES_DIR / "01_category_revenue.png"
        if img_01.exists():
            st.image(str(img_01), use_container_width=True)
        else:
            st.warning("Chưa tìm thấy ảnh 01_category_revenue.png")
    with col_b:
        img_05 = FIGURES_DIR / "05_monthly_revenue_trend.png"
        if img_05.exists():
            st.image(str(img_05), use_container_width=True)
        else:
            st.warning("Chưa tìm thấy ảnh 05_monthly_revenue_trend.png")

with tab_customer:
    st.subheader("2. Phân phối Chi tiêu và So sánh Nhân khẩu học")
    col_c, col_d = st.columns(2)
    with col_c:
        img_02 = FIGURES_DIR / "02_purchase_amount_distribution.png"
        if img_02.exists():
            st.image(str(img_02), use_container_width=True)
        else:
            st.warning("Chưa tìm thấy ảnh 02_purchase_amount_distribution.png")
    with col_d:
        img_04 = FIGURES_DIR / "04_spending_by_age_gender.png"
        if img_04.exists():
            st.image(str(img_04), use_container_width=True)
        else:
            st.warning("Chưa tìm thấy ảnh 04_spending_by_age_gender.png")

with tab_churn:
    st.subheader("3. Phân tích Tỷ lệ Khách hàng Rời bỏ (Churn Rate)")
    col_e, col_f = st.columns([1.5, 1])
    with col_e:
        img_03 = FIGURES_DIR / "03_churn_rate_heatmap.png"
        if img_03.exists():
            st.image(str(img_03), use_container_width=True)
        else:
            st.warning("Chưa tìm thấy ảnh 03_churn_rate_heatmap.png")
    with col_f:
        st.markdown(
            """
            ### Đánh giá hành vi khách hàng:
            * **Tỷ lệ rời bỏ ổn định (~20%):** Churn rate duy trì đồng đều quanh mốc 20% trên mọi phương thức thanh toán và danh mục ngành hàng.
            * **Đặc trưng dữ liệu:** Mức độ biến thiên giữa các kênh giao dịch dưới 2%, cho thấy nguy cơ rời bỏ không bắt nguồn từ hạn chế của bất kỳ cổng thanh toán hay dòng sản phẩm cụ thể nào.
            * **Ý nghĩa thực tiễn:** Doanh nghiệp không cần đổi cổng thanh toán mà nên tập trung chiến dịch chăm sóc lại khách hàng định kỳ sau mua sắm.
            """
        )

with tab_data:
    st.subheader("4. Bảng tra cứu dữ liệu giao dịch sau làm sạch")
    
    with st.expander("🔍 **Bộ Lọc Phân Tích Dữ Liệu (Bấm để mở/ẩn bộ lọc)**", expanded=True):
        f_col1, f_col2, f_col3 = st.columns(3)
        with f_col1:
            sel_cat = st.multiselect(
                "Danh mục sản phẩm:",
                options=sorted(df["Product Category"].dropna().unique().tolist()),
                default=sorted(df["Product Category"].dropna().unique().tolist()),
            )
        with f_col2:
            sel_pay = st.multiselect(
                "Phương thức thanh toán:",
                options=sorted(df["Payment Method"].dropna().unique().tolist()),
                default=sorted(df["Payment Method"].dropna().unique().tolist()),
            )
        with f_col3:
            sel_gen = st.multiselect(
                "Giới tính khách hàng:",
                options=sorted(df["Gender"].dropna().unique().tolist()),
                default=sorted(df["Gender"].dropna().unique().tolist()),
            )

    filtered_df = df[
        (df["Product Category"].isin(sel_cat))
        & (df["Payment Method"].isin(sel_pay))
        & (df["Gender"].isin(sel_gen))
    ]

    st.dataframe(filtered_df.head(100), use_container_width=True)
    st.caption(
        f"Hiển thị 100 bản ghi đầu tiên trong tổng số {len(filtered_df):,} giao dịch khớp với bộ lọc."
    )