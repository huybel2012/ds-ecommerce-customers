# Phân tích sự hài lòng khách hàng trong dịch vụ CSKH thương mại điện tử

## Mô tả đề tài
Phân tích các yếu tố ảnh hưởng đến điểm hài lòng khách hàng (CSAT Score) trong hoạt động chăm sóc khách hàng (CSKH) của một nền tảng thương mại điện tử. Từ đó đưa ra khuyến nghị công ty nên cải thiện khâu nào (kênh hỗ trợ, loại vấn đề, ca làm việc của agent...) để nâng cao trải nghiệm khách hàng.

## Dataset
- **Nguồn:** [Ecommerce Customer Service Satisfaction](https://www.kaggle.com/datasets/ddosad/ecommerce-customer-service-satisfaction) (Kaggle)
- **Kích thước:** 85,907 dòng, 20 cột
- **Đơn vị quan sát:** mỗi dòng là 1 ticket/tương tác hỗ trợ khách hàng
- **Biến mục tiêu:** `CSAT Score` (thang điểm 1–5, không thiếu dữ liệu)

### Các cột chính
| Cột | Ý nghĩa |
|---|---|
| `channel_name` | Kênh hỗ trợ (Inbound, Outcall...) |
| `category` / `Sub-category` | Loại vấn đề khách hàng gặp phải |
| `Tenure Bucket` | Thâm niên của agent xử lý |
| `Agent Shift` | Ca làm việc của agent |
| `CSAT Score` | Điểm hài lòng khách hàng (target) |

Một số cột thiếu dữ liệu nhiều (>65%) như `connected_handling_time`, `order_date_time`, `Customer_City`, `Product_category`, `Item_price`, `Customer Remarks` — cân nhắc loại bỏ hoặc xử lý riêng khi phân tích.

## Cấu trúc thư mục
```
ds-ecommerce-customers/
├── data/
│   ├── raw/            # data gốc tải từ Kaggle (không commit lên Git)
│   └── processed/      # data đã xử lý
├── notebooks/           # code khám phá/phân tích dữ liệu
│   └── 01_eda.py        # EDA ban đầu — chạy bằng: python notebooks/01_eda.py
├── src/                  # code tái sử dụng (khi pipeline ổn định)
├── models/               # model đã train (nếu có)
├── reports/              # biểu đồ, kết quả xuất ra
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
kaggle datasets download -d ddosad/ecommerce-customer-service-satisfaction -p data/raw --unzip
```
(Cần cấu hình Kaggle API token trước — xem [hướng dẫn Kaggle API](https://github.com/Kaggle/kaggle-api))

## Cách chạy
```bash
python notebooks/01_eda.py
```

## Nhóm thực hiện
Cập nhật danh sách thành viên và phân công công việc tại đây.