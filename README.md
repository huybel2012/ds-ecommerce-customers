# Phân tích hành vi mua sắm khách hàng thương mại điện tử

**INFO3020 · Nhập môn Khoa học dữ liệu · Đồ án nhóm**
**Repository:** `huybel2012/ds-ecommerce-customers` · **Nhánh bàn giao:** `Dung` · **Python:** 3.11

Đồ án thực hiện quy trình từ đánh giá chất lượng dữ liệu, xử lý thiếu, EDA/dashboard đến kiểm định W9, hồi quy W10, phân loại W11 và kiểm toán đạo đức/bias W12. Dữ liệu **synthetic tạo bằng Faker**. Các kết quả là nghiên cứu phương pháp trên dữ liệu mô phỏng, **không chứng minh hành vi của khách hàng thực, không đưa vào vận hành**.

> **Trạng thái bàn giao:** mã nguồn và tài liệu Phase 1–7 đã được chuẩn bị trên `Dung`; **không merge sang `main` trong Phase 7**. Cần xác nhận giấy phép Kaggle, quyết định xử lý các tệp dữ liệu legacy đã commit và ký duyệt nội dung báo cáo trước khi phát hành rộng rãi.

## 1. Kết quả cốt lõi

| Nội dung | Kết quả có căn cứ | Cách diễn giải đúng |
|---|---|---|
| Dữ liệu | 250.000 giao dịch, 49.673 khách (2020–09/2023) | Dữ liệu mô phỏng, không đại diện thị trường |
| Chất lượng | `Returns` thiếu 47.596 dòng (19,04%); 249.952 dòng có `Total Purchase Amount ≠ Price × Quantity` | Giữ `Returns=NULL`; tiền tệ/ngữ nghĩa amount chưa được xác minh |
| EDA | Tổng `Total Purchase Amount` ghi nhận: 681.342.683; churn theo khách: 20,0149% | Không gọi amount là doanh thu thực hoặc USD/VND |
| W9 | H1/H2/H3 có Holm p ≈ 0,2400 / 0,6238 / 0,1674 | Không bác bỏ H0 trong ba kiểm định chính; không suy ra các nhóm bằng nhau |
| W10 | OLS test R² ≈ 0,0032; MAE cải thiện ≈ 1,292 đơn vị so với baseline | Cải thiện rất nhỏ, dự đoán yếu, không phải mô hình doanh thu thực |
| W11 | Random Forest test ROC-AUC ≈ 0,4949; F1 ≈ 0,3293; baseline all-positive F1 ≈ 0,3293 | Gần ngẫu nhiên, gần như đoán tất cả khách là Churn |
| Phase 5.1 | Mở rộng 19 → 28 features, repeated CV, shuffled labels: không qua sàng lọc | Giữ mô hình W11; không claim hiệu năng cải thiện |
| W12 | Fairness theo giới tính, tuổi và giao nhau; model card | Khoảng cách selection-rate nhỏ nhưng **không chứng minh công bằng** |

Chi tiết có nguồn truy vết: **[Báo cáo cuối kỳ](reports/final_project_report.md)** và [mục lục bằng chứng](docs/delivery_checklist.md).

## 2. Nguồn dữ liệu và nguồn sự thật

Nguồn: [Kaggle, E-commerce Customer Data for Behavior Analysis](https://www.kaggle.com/datasets/shriyashjagtap/e-commerce-customer-for-behavior-analysis). Chỉ lấy tệp **`ecommerce_customer_data_custom_ratios.csv`** theo hướng dẫn của dự án, đặt tại:

```text
data/raw/ecommerce_customer_data_custom_ratios.csv
```

**Không commit raw CSV hoặc CSV sạch mới.** `data/processed/ecommerce_customer_clean.csv`, sinh từ Phase 2, là **nguồn duy nhất** cho EDA, W9–W12. Hai tệp được commit từ lịch sử cũ (`data/processed/ecommerce_customer_cleaned.csv` và `.parquet`) là **legacy**, không được dùng thay thế và vẫn đang chờ quyết định công bố. Chi tiết tại [Datasheet](docs/dataset_datasheet.md) và [Data Dictionary](docs/data_dictionary.md).

## 3. Chạy lại từ đầu (Windows PowerShell)

Từ thư mục clone dự án, ví dụ `C:\Code\KHDL\Project`:

```powershell
py -3.11 -m venv venv
.\venv\Scripts\python.exe -m pip install -r requirements.txt
.\venv\Scripts\python.exe -m unittest discover -s tests -v
```

Sau khi đặt CSV Kaggle tại `data/raw/`:

```powershell
.\venv\Scripts\python.exe -m src.run_pipeline
.\venv\Scripts\python.exe -m src.run_pipeline --validate-only
.\venv\Scripts\python.exe -m src.pipeline.05_explanatory_visualization
.\venv\Scripts\python.exe -m src.run_phase4
.\venv\Scripts\python.exe -m src.run_phase5
.\venv\Scripts\python.exe -m src.run_phase51
.\venv\Scripts\python.exe -m src.run_phase6
.\venv\Scripts\python.exe -m src.run_phase7
```

Dashboard chạy riêng:

```powershell
.\venv\Scripts\python.exe -m streamlit run app.py
```

Xem **[Hướng dẫn tái lập và kiểm tra đầu ra](docs/reproducibility_guide.md)** để phân biệt chạy `--dry-run`, báo cáo được commit, kết quả sinh cục bộ, và trường hợp kết quả không khớp. Các phiên bản package có khoảng tương thích, chưa có lockfile byte-exact.

## 4. Mục lục tài liệu và biểu đồ

| Hạng mục | Tài liệu/bằng chứng |
|---|---|
| Dữ liệu, quality gates | [Phase 1](docs/phase1_runbook.md), [Phase 2](docs/phase2_runbook.md), [Data Dictionary](docs/data_dictionary.md) |
| EDA và dashboard (W7) | [Phase 3](docs/phase3_runbook.md), [audit visualization](reports/phase3/visualization_audit.md), hình `01–05` |
| Statistical inference (W9) | [Hypothesis tests](reports/hypothesis_tests.md), [AB proposal](docs/ab_test_design.md), hình `09` |
| Regression (W10) | [Regression report](reports/regression_results.md), hình `06–08` |
| Classification (W11) | [Classifier report](reports/classification_results.md), [label validity](docs/churn_label_assessment.md), hình `10–12` |
| Improvement study | [Phase 5.1](reports/phase5/phase51_improvement_study.md), hình `13–14` |
| Ethics & bias (W12) | [Fairness report](reports/phase6/ethics_and_bias.md), [Model Card](reports/phase6/model_card.md), [ethics checklist](docs/ethics_checklist.md) |
| Final delivery | [Báo cáo cuối kỳ](reports/final_project_report.md), [Defence guide](docs/defense_guide.md), [Release audit](reports/phase7/release_audit.md), [Delivery checklist](docs/delivery_checklist.md) |

`reports/figures/` có **14 biểu đồ**. W8 (giữa kỳ) và các bài EX không nằm trong phạm vi tổng kết được yêu cầu.

## 5. Cấu trúc dự án

```text
app.py                         Streamlit dashboard
src/run_pipeline.py            Raw → clean, integrity check
src/run_phase4.py              W9 statistical tests + W10 regression
src/run_phase5.py              W11 frozen classifier & metrics
src/run_phase51.py             Exploratory train-only improvement study
src/run_phase6.py              W12 fairness replay on frozen classifier
src/run_phase7.py              Read-only final evidence/release audit
data/raw/                      Local Kaggle download (gitignored)
data/processed/                Canonical generated clean CSV + legacy tracked files
docs/                          Runbooks, datasheet, reproducibility, ethics, defense
reports/                       JSON metrics, detailed reports, final report, 14 figures
tests/                         Python unittest suite (no Kaggle raw required)
.github/workflows/             Python 3.11 CI
```

## 6. Hạn chế bắt buộc khi trình bày

Nhãn `Churn` không có thời điểm phát sinh, nên không đủ căn cứ để dự đoán churn trong tương lai. A/B testing trong báo cáo chỉ là **đề xuất thiết kế**, không phải một thử nghiệm ngẫu nhiên đã thực hiện. Không đưa kết quả synthetic vào quyết định với con người. Cần xác minh **license** tại trang dataset gốc trước khi tái phân phối, kiểm tra quyền công bố hai tệp legacy chứa tên/ID giả lập, và không tuyên bố k-anonymity hay permission chưa được đo/xác nhận.

## 7. AI assistance & học thuật

Nhóm có sử dụng **ChatGPT (OpenAI)** để hỗ trợ rà soát code, phương pháp, test và biên tập một số tài liệu. Kết quả được nhóm thực thi lại và cần kiểm chứng, giải thích, chịu trách nhiệm về giả định và kết luận. Xem [AI usage disclosure](docs/ai_usage_disclosure.md); nhóm phải xác nhận và bổ sung sự tham gia thực tế trước khi nộp. Không xem AI là nguồn chứng thực giấy phép, dữ liệu hay đánh giá độc lập.

**Bàn giao:** Các đầu mục chờ người phụ trách ký duyệt tại [Delivery Checklist](docs/delivery_checklist.md). **Merge deferred**: không có thao tác tạo PR/merge vào `main` trong Phase 7.
