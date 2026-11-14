# Phân tích hành vi mua sắm khách hàng thương mại điện tử

**Repo:** `huybel2012/ds-ecommerce-customers` · **Nhánh làm việc:** `Dung`

Phân tích chất lượng dữ liệu, hành vi giao dịch mô phỏng, kiểm định giả thuyết, hồi quy và dashboard Streamlit. **Dataset sinh bằng Faker**, vì vậy không được suy rộng các insight thành kết luận về thị trường hoặc khách hàng thực.

## Dữ liệu

Nguồn: [E-commerce Customer Data for Behavior Analysis (Kaggle)](https://www.kaggle.com/datasets/shriyashjagtap/e-commerce-customer-for-behavior-analysis).

Chỉ dùng file `ecommerce_customer_data_custom_ratios.csv`. File **raw không được commit**; tải từ Kaggle rồi đặt tại `data/raw/ecommerce_customer_data_custom_ratios.csv`.

**Lưu ý:** `Total Purchase Amount` không khớp `Product Price × Quantity` tại hầu hết các dòng; `Returns` thiếu khoảng 19%. Xem [Data dictionary](docs/data_dictionary.md), [Phase 1](docs/phase1_runbook.md). Tránh tự coi số tiền này là doanh thu thực hay gán đơn vị USD khi nguồn chưa xác minh.

## Quickstart (Windows, Python 3.11)

```powershell
cd C:\Code\KHDL\Project
py -3.11 -m venv venv
.\venv\Scripts\python.exe -m pip install -r requirements.txt
.\venv\Scripts\python.exe -m unittest discover -s tests -v
```

Sau khi tải file raw từ Kaggle:

```powershell
.\venv\Scripts\python.exe -m src.run_pipeline --dry-run
.\venv\Scripts\python.exe -m src.run_pipeline
.\venv\Scripts\python.exe -m src.run_pipeline --validate-only
```

Lệnh chạy core Phase 2: audit → data quality → missing → outlier → cleaned CSV; kết quả cuối ở `data/processed/ecommerce_customer_clean.csv` và `reports/phase2/run_manifest.json` (các file phát sinh không commit). Script **không dùng processed CSV thay thế** nếu raw vắng mặt.

Chạy dashboard đang có (đã kiểm tra thủ công trên Windows):

```powershell
.\venv\Scripts\python.exe -m streamlit run app.py
```

Dashboard và các kết quả W9–W10 là **legacy**: định nghĩa KPI và phương pháp mô hình phải được rà soát trong Phase 3–4; không suy ra chúng đã được cập nhật tự động bằng Phase 2.

## Repository

```text
app.py                         # Dashboard Streamlit
src/utils.py                   # Utilities, raw schema, loader
src/run_pipeline.py            # Entrypoint Phase 2 (00–04)
src/phase1_data_integrity.py   # Phase 1: data integrity
src/pipeline/                  # 00–07: script phân tích gốc
notebooks/01_data_cleaning.ipynb
notebooks/01_eda.py
reports/                       # Báo cáo, figures
reports/phase1/                # Kiểm toán Phase 1
reports/phase2/                # Run manifest tự sinh (không commit)
docs/data_dictionary.md
docs/phase1_runbook.md
docs/phase2_runbook.md
tests/                         # Unit tests không cần Kaggle raw
requirements.txt
```

**Hướng dẫn chi tiết về dữ liệu gốc, triển khai và nghiệm thu:** [Phase 2 runbook](docs/phase2_runbook.md).

## Nhóm và kết quả cuối kỳ

Phân công thành viên, tổng hợp insight và tài liệu bảo vệ sẽ hoàn thiện trong Phase 7. Đừng dùng các báo cáo mẫu hiện có thay cho kết luận cuối khi chưa qua validation.
