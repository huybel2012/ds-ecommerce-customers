# Phân tích hành vi mua sắm khách hàng thương mại điện tử

## Mô tả đề tài
Phân tích hành vi mua sắm của khách hàng trên nền tảng thương mại điện tử — tìm hiểu các yếu tố liên quan đến giá trị đơn hàng, danh mục sản phẩm, tỷ lệ trả hàng... Từ đó đưa ra insight/khuyến nghị kinh doanh phù hợp.

## Dataset
- **Nguồn:** [E-commerce Customer Data For Behavior Analysis](https://www.kaggle.com/datasets/shriyashjagtap/e-commerce-customer-for-behavior-analysis) (Kaggle)
- **File sử dụng:** `ecommerce_customer_data_custom_ratios.csv`
- **Lưu ý:** dataset gốc có 2 file, chỉ dùng file trên (theo chỉ định của tác giả dataset), không dùng file còn lại

### Các cột chính
> Cập nhật bảng này sau khi chạy `df.columns` / `df.info()` để ghi đúng tên và ý nghĩa từng cột.

| Cột | Ý nghĩa |
|---|---|
| ... | ... |

## Cấu trúc thư mục
```
ds-ecommerce-customers/
├── data/
│   ├── raw/            # data gốc tải từ Kaggle (không commit lên Git)
│   └── processed/      # data đã xử lý
├── notebooks/
│   └── 01_eda.py        # EDA — chạy bằng: python notebooks/01_eda.py
├── src/                  # code tái sử dụng (khi pipeline ổn định)
├── models/               # model đã train (nếu có)
├── reports/              # biểu đồ, kết quả, nhật ký làm sạch dữ liệu
│   └── cleaning_log.csv  # nhật ký xử lý dữ liệu — bắt buộc khi nộp
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
(Cần cấu hình Kaggle API token trước — xem [hướng dẫn Kaggle API](https://github.com/Kaggle/kaggle-api))

## Cách chạy
```bash
python notebooks/01_eda.py
```

## Nhật ký làm sạch dữ liệu
Mọi bước xử lý dữ liệu bất thường (giá trị thiếu, trùng lặp, sai định dạng...) được ghi lại tại `reports/cleaning_log.csv`, gồm: bước xử lý, lý do, số dòng trước/sau, số dòng bị ảnh hưởng.

## Nhóm thực hiện
Cập nhật danh sách thành viên và phân công công việc tại đây.