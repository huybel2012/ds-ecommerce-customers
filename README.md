# Phân tích hành vi mua sắm khách hàng thương mại điện tử

## Mô tả đề tài
Phân tích hành vi mua sắm của khách hàng trên nền tảng thương mại điện tử — tìm hiểu các yếu tố liên quan đến giá trị đơn hàng, danh mục sản phẩm, tỷ lệ trả hàng... Từ đó đưa ra insight/khuyến nghị kinh doanh phù hợp.

## Dataset
- **Nguồn:** [E-commerce Customer Data For Behavior Analysis](https://www.kaggle.com/datasets/shriyashjagtap/e-commerce-customer-for-behavior-analysis) (Kaggle)
- **File sử dụng:** `ecommerce_customer_data_custom_ratios.csv`
- **Lưu ý:** dataset gốc có 2 file, chỉ dùng file trên (theo chỉ định của tác giả dataset), không dùng file còn lại

### Các cột chính

| Cột | Ý nghĩa |
|---|---|
| `Customer ID` | Mã định danh khách hàng |
| `Purchase Date` | Ngày mua hàng |
| `Product Category` | Danh mục sản phẩm |
| `Product Price` | Giá sản phẩm |
| `Quantity` | Số lượng mua |
| `Total Purchase Amount` | Tổng giá trị đơn hàng |
| `Payment Method` | Phương thức thanh toán |
| `Customer Age` / `Age` | Tuổi của khách hàng (Có 2 cột tuổi) |
| `Returns` | Trạng thái trả hàng |
| `Customer Name` | Tên khách hàng |
| `Gender` | Giới tính |
| `Churn` | Khách hàng đã rời bỏ hay chưa |

## Cấu trúc thư mục
```text
ds-ecommerce-customers/
├── data/
│   ├── raw/            # data gốc tải từ Kaggle (không commit lên Git)
│   └── processed/      # data đã xử lý
├── notebooks/
│   ├── 00_baseline_audit.py          # Kiểm tra tổng quan dữ liệu
│   ├── 01_quality_assessment.py      # Đánh giá chất lượng dữ liệu
│   ├── 02_missing_analysis.py        # Phân tích dữ liệu thiếu (missing values)
│   ├── 02b_leakage_safe_imputation_demo.py # Xử lý dữ liệu thiếu chống rò rỉ
│   ├── 03_outlier_analysis.py        # Phân tích dữ liệu ngoại lai (outliers)
│   ├── 04_finalize_cleaning.py       # Hoàn thiện bước làm sạch dữ liệu
│   ├── 01_eda.py                     # Script EDA
│   └── 02_eda.ipynb                  # Notebook EDA
├── src/                  
│   └── utils.py          # Các hàm hỗ trợ cho quá trình xử lý
├── reports/              # biểu đồ, kết quả, nhật ký làm sạch dữ liệu
├── requirements.txt
├── .gitignore
└── README.md
```

## Setup môi trường
```bash
python -m venv venv
venv\Scripts\activate        # Windows
source venv/bin/activate     # Mac/Linux

pip install -r requirements.txt
```

## Tải dataset
```bash
python -m kaggle datasets download -d shriyashjagtap/e-commerce-customer-for-behavior-analysis -p data/raw --unzip
```
*(Cần cấu hình Kaggle API token trước — xem [hướng dẫn Kaggle API](https://github.com/Kaggle/kaggle-api))*

## Cách chạy
Chạy tuần tự các file script trong thư mục `notebooks/` để kiểm tra, làm sạch và phân tích dữ liệu:
```bash
python notebooks/00_baseline_audit.py
python notebooks/01_quality_assessment.py
python notebooks/02_missing_analysis.py
python notebooks/03_outlier_analysis.py
python notebooks/04_finalize_cleaning.py
python notebooks/01_eda.py
```

## Nhật ký làm sạch dữ liệu
Mọi bước xử lý dữ liệu bất thường (giá trị thiếu, trùng lặp, sai định dạng...) được ghi lại tại thư mục `reports/` (ví dụ `cleaning_log.csv`), gồm: bước xử lý, lý do, số dòng trước/sau, số dòng bị ảnh hưởng.

## Nhóm thực hiện
Cập nhật danh sách thành viên và phân công công việc tại đây.