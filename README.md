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

### Phase 3: EDA & Dashboard

Sau Phase 2, phải có `data/processed/ecommerce_customer_clean.csv` (bản đã **giữ nguyên `Returns=NULL`**).

```powershell
# Tạo lại đúng 5 hình EDA, không xóa hình W9/W10 (06–09):
.\venv\Scripts\python.exe -m src.pipeline.05_explanatory_visualization

# Dashboard chạy trực tiếp từ cùng file clean, hình được render tự động từ CSV:
.\venv\Scripts\python.exe -m streamlit run app.py
```

Hoặc làm lại cả cleaning + EDA từ Kaggle raw: `python -m src.run_pipeline --stage eda`.

**Metric contract:** Dashboard dùng `Total Purchase Amount` như một **giá trị ghi nhận** (đơn vị tiền chưa xác minh), **không** coi là doanh thu USD thực. Churn tính mỗi `Customer ID` **một lần**; tỷ lệ Returns chỉ trên dòng đã quan sát (`NULL` không phải 0). Tháng cuối không đầy đủ được đánh dấu, bảng không hiển thị `Customer Name`. Không dùng `ecommerce_customer_cleaned.csv` legacy làm fallback. Xem [Phase 3 runbook](docs/phase3_runbook.md), [Visualization Audit](reports/phase3/visualization_audit.md) và [Data Dictionary](docs/data_dictionary.md).

Các báo cáo W9–W10 và hình 06–09 chưa được tính lại theo metric contract và sẽ được đánh giá trong Phase 4.

## Repository

```text
app.py                         # Dashboard Streamlit
src/utils.py                   # Utilities, raw schema, loader
src/eda_metrics.py             # Phase 3: metric definitions, customer-level churn
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
docs/phase3_runbook.md
tests/                         # Unit tests không cần Kaggle raw
requirements.txt
```

**Hướng dẫn chi tiết về dữ liệu gốc, triển khai và nghiệm thu:** [Phase 2 runbook](docs/phase2_runbook.md).

## Nhóm và kết quả cuối kỳ

Phân công thành viên, tổng hợp insight và tài liệu bảo vệ sẽ hoàn thiện trong Phase 7. Đừng dùng các báo cáo mẫu hiện có thay cho kết luận cuối khi chưa qua validation.

## Phase 4: Statistical Inference & Regression (W9–W10)

Run after Phase 2 creates `data/processed/ecommerce_customer_clean.csv`:

```bash
python -m src.run_phase4 --dry-run
python -m src.run_phase4
```

Results: `reports/hypothesis_tests.md`, `reports/regression_results.md`, figures `06`–`09` and machine-readable metrics under `reports/phase4/`. See `docs/phase4_runbook.md` and the **proposal-only** `docs/ab_test_design.md`. The dataset is **synthetic**, currency/recorded-amount semantics are unresolved and this analysis is not evidence about real-world treatment effects. These scripts reject the old processed CSV with filled-in `Returns`.
