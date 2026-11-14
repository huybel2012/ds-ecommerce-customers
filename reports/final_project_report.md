# BÁO CÁO CUỐI KỲ: PHÂN TÍCH HÀNH VI MUA SẮM KHÁCH HÀNG THƯƠNG MẠI ĐIỆN TỬ

**Học phần:** INFO3020 — Nhập môn Khoa học dữ liệu, Trường Đại học CMC
**Nhóm thực hiện, lớp, MSSV:** *[Nhóm bổ sung thông tin đã được xác nhận trước khi nộp]*
**Nguồn mã và dữ liệu phân tích:** `huybel2012/ds-ecommerce-customers`, nhánh `Dung`
**Phiên bản báo cáo:** Phase 7 (dựa trên các kết quả đã chạy/commit ở Phase 1–6)
**Phạm vi:** W1–W7 và W9–W12; không bao gồm báo cáo W8 hoặc các bài EX riêng lẻ.

> **Cảnh báo học thuật:** Đây là phân tích *synthetic* nhằm thực hành phương pháp, không phải bằng chứng về khách hàng thật. Nhãn `Churn` không có thời điểm ghi nhận; số tiền ghi nhận không khớp với đơn giá nhân số lượng và chưa có chứng cứ về đơn vị tiền tệ. Báo cáo **không** đề xuất triển khai mô hình thực tế.

## Tóm tắt

Nhóm thực hiện chu trình khoa học dữ liệu trên 250.000 giao dịch mô phỏng thuộc 49.673 khách hàng, bao gồm kiểm toán nguồn và chất lượng, chuẩn hóa pipeline, phân tích khám phá, dashboard, ba kiểm định suy luận thống kê, hồi quy liên tục, phân loại Churn và đánh giá thiên kiến theo nhóm. `Returns` thiếu 47.596 dòng (19,04%) và được giữ nguyên tình trạng thiếu; 249.952 dòng có `Total Purchase Amount` khác `Product Price × Quantity`. Sau điều chỉnh Holm cho ba kiểm định W9, không bác bỏ giả thuyết không ở ngưỡng 5%. Hồi quy OLS đạt R² test ≈ 0,0032 và chỉ cải thiện MAE ≈ 1,292 đơn vị so với mô hình trung bình. Phân loại Random Forest có ROC-AUC test ≈ 0,4949; nghiên cứu bổ sung 28 đặc trưng không cải thiện khả năng phân biệt. Đánh giá W12 nhận thấy tỷ lệ chọn Churn theo nhóm gần bằng nhau chủ yếu vì mô hình dự đoán gần như tất cả khách thuộc lớp dương. Đóng góp chính là một quy trình **tái lập, kiểm tra được, minh bạch giới hạn**, không phải một thuật toán dự đoán thương mại hiệu quả.

**Từ khóa:** Data Quality, EDA, Statistical Inference, Regression, Classification, Data Leakage, Fairness, Synthetic Data.

## 1. Bài toán và câu hỏi nghiên cứu (W1)

Dữ liệu giao dịch thương mại điện tử cho phép mô tả sản phẩm, cách thanh toán, giá trị ghi nhận, thời gian mua và đặc điểm khách. Tuy nhiên, mỗi khách xuất hiện nhiều lần và có các trường thiếu, nên một thống kê trực tiếp theo số dòng có thể tạo ra kết luận sai lệch. Nhóm đặt ra các câu hỏi:

1. Những vấn đề nào khiến dữ liệu chưa thể dùng làm dữ liệu kinh doanh đã đối soát?
2. Phân bố sản phẩm, số tiền ghi nhận, thời gian mua và tỷ lệ Churn như thế nào khi chọn đúng đơn vị phân tích?
3. Các mối liên hệ giữa Churn, giới tính, giá trị ghi nhận và danh mục có đạt ý nghĩa thống kê khi kiểm soát so sánh nhiều lần không?
4. Hồi quy có vượt được baseline trên khách hàng chưa thấy không? Các classifier có phân biệt Churn tốt hơn mức ngẫu nhiên không?
5. Kết quả có thiên kiến/giới hạn nào và điều kiện đạo đức nào phải đáp ứng trước khi công bố hoặc sử dụng?

**Đối tượng:** 250.000 giao dịch mô phỏng trong khoảng 2020-01-01 đến 2023-09-15; 49.673 `Customer ID` độc lập về khóa. **Không khẳng định mẫu đại diện** cho khách hàng thật. Các biến mục tiêu thống kê khác nhau theo câu hỏi, không áp đặt một mô hình lên tất cả các bước.

## 2. Nguồn dữ liệu và phương pháp (W2–W5)

### 2.1. Nguồn và quản trị

Dữ liệu từ [E-commerce Customer Data for Behavior Analysis, Kaggle](https://www.kaggle.com/datasets/shriyashjagtap/e-commerce-customer-for-behavior-analysis), tệp `ecommerce_customer_data_custom_ratios.csv`, được mô tả là sinh bằng Faker. Nhóm không thực hiện web scraping hay thử nghiệm A/B thật. Không sử dụng license của một notebook Kaggle bên thứ ba để xác nhận quyền tái phân phối. Xem [datasheet](../docs/dataset_datasheet.md) và [data dictionary](../docs/data_dictionary.md).

Pipeline Phase 2 coi raw Kaggle là bắt buộc, giữ bản raw bất biến; `data/processed/ecommerce_customer_clean.csv` là output chuẩn (không commit). Hai bản `ecommerce_customer_cleaned.csv` và `ecommerce_clean.parquet` từ lịch sử cũ không phải nguồn sự thật vì đã làm sai lệch missingness của `Returns` hoặc chưa có validation chuẩn.

### 2.2. Kiểm toán chất lượng

| Chỉ số | Kết quả | Cách xử lý |
|---|---:|---|
| Giao dịch / khách | 250.000 / 49.673 | Phân biệt đơn vị giao dịch và khách hàng |
| `Returns` thiếu | 47.596 dòng (19,0384%) | **Không** gán NA = 0; có indicator `Returns_was_missing` |
| Tỷ lệ Returns trên quan sát | 100.769 / 202.404 = **49,7861%** | Không chia trên toàn bộ 250.000 khi giải thích observed returns |
| Dòng amount khác `Price × Quantity` | **249.952 / 250.000** | Không sửa ép để khớp; không tuyên bố doanh thu đã đối soát |
| Tổng amount ghi nhận | **681.342.683** | Đơn vị tiền chưa xác minh |
| Tổng line value tính toán | **190.908.939** | Đại lượng khác với trường amount ghi nhận |
| Trùng toàn dòng và trùng (`Customer ID`,`Purchase Date`) | **0 / 0** | Kiểm tra khóa sự kiện tạm thời |
| Khoảng ngày | 2020-01-01 → 2023-09-15 | Tháng 09/2023 chưa đủ, không giải thích như sự sụt giảm tháng hoàn chỉnh |

Quy tắc cleaning trong `src/run_pipeline.py` kiểm tra số dòng, giá trị gốc, khóa sự kiện, cờ missing, schema và hash nguồn trước/sau. Các bước xử lý outlier và missing có audit; không loại bỏ dữ liệu hoặc bù missing chỉ để làm mô hình tốt hơn. [Báo cáo Phase 1](phase1/data_integrity_report.md) và [Data Dictionary](../docs/data_dictionary.md) ghi đầy đủ hợp đồng chỉ số.

## 3. EDA và trực quan hóa (W6–W7)

Dashboard Streamlit và 5 biểu đồ đều dùng **cùng một bản CSV sạch đã qua validation**. Phần lớn thời lượng quan sát dùng biến `Total Purchase Amount` như **recorded amount**, không gọi là USD/VND revenue.

- **Danh mục:** tổng recorded amount của Books = 204.939.601, Clothing = 204.532.405, Electronics = 136.599.467, Home = 135.271.210. Chỉ là tổng trên dữ liệu mô phỏng; không chứng minh nhu cầu thị trường.
- **Tỷ lệ Churn:** 9.942 / 49.673 = **20,0149%** trên khách hàng duy nhất. Nếu lấy trung bình trên giao dịch sẽ ra **19,9496%** và làm những khách nhiều giao dịch có trọng số lớn hơn.
- **Returns:** 49,7861% trong các dòng có quan sát, trong khi nếu lấp NA bằng 0 tỷ lệ bị giảm giả tạo còn 40,3076%.
- **Thời gian:** đồ thị phân biệt tháng hoàn chỉnh và tháng 09/2023 chỉ có một phần dữ liệu.

Minh họa gốc có thể dẫn vào bản trình bày: [01 Danh mục](figures/01_category_revenue.png), [02 Phân bố amount](figures/02_purchase_amount_distribution.png), [03 Churn heatmap](figures/03_churn_rate_heatmap.png), [04 Theo tuổi/giới](figures/04_spending_by_age_gender.png), [05 Theo tháng](figures/05_monthly_revenue_trend.png).

## 4. Kiểm định suy luận thống kê và đề xuất A/B (W9)

Đơn vị khách cho H1/H2 được khử trùng lặp theo `Customer ID`; H3 xử lý giao dịch lặp của khách qua standard errors cluster. Ba giả thuyết chính sử dụng kiểm định hai phía với α=0,05, điều chỉnh **Holm** cho family ba kiểm định. Các giả thuyết đã xem xét trên dữ liệu, không gọi là nghiên cứu pre-registered.

| Giả thuyết H0 | Phương pháp | p gốc | p Holm (3 test) | Kết luận |
|---|---|---:|---:|---|
| H1: tỷ lệ Churn của Male = Female | Two-proportion z, cấp khách | 0,1200 | **0,2400** | Không bác bỏ H0 |
| H2: giá trị amount trung bình mỗi khách bằng nhau theo Churn | Welch t, cấp khách | 0,6238 | **0,6238** | Không bác bỏ H0 |
| H3: amount trung bình của 4 danh mục bằng nhau | Wald omnibus, SE cluster khách | 0,05579 | **0,1674** | Không bác bỏ H0 |

Không bác bỏ H0 **không chứng minh hai nhóm giống nhau**. Các phân tích pairwise danh mục được đánh dấu *exploratory* (có một cặp đạt Holm p nhỏ hơn 0,05 trong family pairwise riêng), nhưng **không đảo ngược kết luận family ba kiểm định chính** và không chứng minh hiệu ứng kinh tế đáng kể. Xem [W9 hypothesis tests](hypothesis_tests.md), [hình 09](figures/09_hypothesis_tests.png).

**A/B testing:** [Bản thiết kế A/B](../docs/ab_test_design.md) trình bày giả thuyết, metric, ước lượng cỡ mẫu và phân nhóm cho một thí nghiệm *tương lai*. Dataset hiện tại không có random assignment hoặc các nhóm can thiệp/đối chứng; nhóm **chưa chạy A/B test** và không tuyên bố tác động nhân quả.

## 5. Hồi quy liên tục (W10)

**Target:** `Total Purchase Amount` (đơn vị chưa xác minh). Feature gồm Product Price, Quantity, Age, Category, Payment Method, Gender; loại Returns/Churn/ID và biến không phù hợp theo thời điểm. Chia train/test theo **nhóm Customer ID** (80/20) tránh cùng khách hiện ở hai tập; 3-fold Group CV trong train. So sánh baseline dự đoán trung bình với OLS/Ridge/Lasso.

| Mô hình | MAE test | RMSE test | R² test |
|---|---:|---:|---:|
| Baseline (mean) | 1.251,4015 | 1.445,0418 | xấp xỉ 0 |
| **OLS** | **1.250,1095** | **1.442,7104** | **0,0032** |
| Ridge | 1.250,1130 | 1.442,7135 | 0,0032 |
| Lasso | 1.250,1897 | 1.442,8167 | 0,0031 |

OLS cải thiện MAE ≈ **1,292 đơn vị** (~0,103% MAE baseline), bootstrap CI95% theo khách [0,635; 1,947]. Dù có chênh lệch số học, **R² ~0,0032 cho thấy giải thích biến thiên mục tiêu rất yếu**, còn target chưa được xác minh là tiền doanh thu thật. Có hệ số Age thống kê khác 0 trong mô hình mô phỏng, nhưng không được diễn giải thành quan hệ nhân quả hoặc lợi ích kinh doanh. [Báo cáo W10](regression_results.md), hình [06](figures/06_residual_diagnostics.png), [07](figures/07_coefficients_ci.png), [08](figures/08_actual_vs_predicted.png).

## 6. Phân loại Churn và Feature Engineering (W11)

Dùng mốc quan sát **2023-01-01 (exclusive)** để chỉ tạo feature từ giao dịch quá khứ: 202.313 giao dịch, **49.128 khách hợp lệ**; loại 545 khách không có lịch sử trước cutoff. Nhãn Churn lấy từ thuộc tính khách, cố định nhưng **không có effective timestamp**, do vậy chỉ có thể gọi là phân loại hồi cứu nhãn synthetic, không phải forecast churn 30/60/90 ngày.

19 feature hành vi (tần suất, recency, quantity, giá trị, diversity, tỷ lệ theo category/payment) được tổng hợp mỗi khách một dòng; không dùng Customer ID/Name, Returns, Age, Gender hoặc nhãn làm predictors. Chia train=39.302 và holdout test=9.826 khách riêng biệt; threshold/model được chọn trên train out-of-fold, không dùng test labels.

| Mô hình | F1 holdout | ROC-AUC | PR-AUC (AP) |
|---|---:|---:|---:|
| Prior baseline | 0,0000 | 0,5000 | 0,1971 |
| All-positive baseline | **0,3293** | 0,5000 | 0,1971 |
| Logistic Regression | 0,3293 | 0,4930 | 0,1912 |
| **Random Forest** (train-only selected) | **0,3293** | **0,4949** | **0,1965** |

Random Forest threshold=**0,375** có confusion matrix **[TN=3, FP=7.886, FN=1, TP=1.936]**. Mô hình gần như dự đoán *mọi khách đều Churn*, nên recall ≈1 không đồng nghĩa chất lượng cao; F1 không vượt all-positive baseline. 95% bootstrap CI của ROC-AUC ~[0,4802; 0,5101] bao gồm 0,5. Không triển khai. [W11](classification_results.md), [label assessment](../docs/churn_label_assessment.md), [hình ROC/PR](figures/10_classification_roc_pr.png) và [confusion matrices](figures/11_classification_confusion_matrices.png).

### 6.1. Nghiên cứu bổ sung Phase 5.1

Thử thêm chín feature thời gian (tổng 28), Logistic Regression và HistGradientBoosting, repeated customer-disjoint CV 3-fold × 2 lần trên **train**, cùng đối chứng nhãn xáo trộn. ROC-AUC tốt nhất ≈0,4993 (19 features, Logistic Regression), trong khi đối chứng nhãn xáo trộn ≈0,5026. Mẫu mở rộng 28 features không vượt ngưỡng sàng lọc đặt trước; **không thay thế mô hình W11**, không dùng lại holdout để tuning. Xem [Phase 5.1](phase5/phase51_improvement_study.md), [hình 13](figures/13_phase51_univariate_signal.png) và [14](figures/14_phase51_repeated_cv.png).

## 7. Dữ liệu lớn, đạo đức và thiên kiến (W12)

### 7.1. Có cần Big Data framework không?

250.000 giao dịch và ~50.000 khách hàng đủ quy mô cho Pandas/NumPy/scikit-learn trên một máy khi tối ưu đúng. Chưa có benchmark RAM và thời gian đo chuẩn để suy ra khả năng mở rộng sản xuất. Chỉ xem xét partitioned Parquet, query engine hoặc Spark nếu số liệu thực nghiệm về bộ nhớ/I/O/latency cho thấy cần; không đưa Spark vào đồ án để tạo vẻ phức tạp. [Scalability assessment](../docs/scalability_assessment.md).

### 7.2. Fairness audit

Phase 6 **replay mô hình/threshold Phase 5** và so khớp confusion matrix với frozen metrics; đánh giá giới tính, sáu dải tuổi và 12 nhóm giao nhau. Trên holdout 9.826 khách, chênh lệch *selection rate* (max−min) theo Gender ≈**0,0405 điểm %**, Age ≈**0,0657 điểm %**, Gender×Age ≈**0,1377 điểm %**. Nhưng cả hai giới đều bị dự đoán Churn gần 100%, kéo false positive rate lên rất cao: đây là **parity vô nghĩa về vận hành**, không phải chứng nhận công bằng. ROC-AUC tổng chỉ xấp xỉ 0,495 và dữ liệu chỉ có Gender nhị phân/tuổi 18–70; không ngoại suy công bằng sang người thật. [Báo cáo fairness](phase6/ethics_and_bias.md), [metrics](phase6/fairness_metrics.json), [model card](phase6/model_card.md).

### 7.3. Privacy, giấy phép, AI disclosure

Dữ liệu có `Customer Name` synthetic và `Customer ID` trong các file legacy đã commit. Không thể coi việc sinh dữ liệu giả là bằng chứng đã ẩn danh hay đạt k-anonymity; **chưa đo k**. Giấy phép Kaggle **chưa có bằng chứng xác minh trực tiếp** để bảo đảm quyền tái phân phối. Các kết quả aggregate không chứa danh sách dự đoán từng khách, nhưng quyết định xử lý legacy file/lịch sử Git cần được nhóm chấp thuận. [Datasheet](../docs/dataset_datasheet.md), [ethics checklist](../docs/ethics_checklist.md), [release audit](phase7/release_audit.md).

**Khai báo AI:** ChatGPT (OpenAI) đã hỗ trợ gợi ý phương pháp, review mã, draft code/test và chỉnh sửa tài liệu. Nhóm chịu trách nhiệm kiểm tra nguồn, chạy và giải thích mã, kết quả, giấy phép và phần trình bày. Câu khai báo cần được nhóm xác nhận theo thực tế sử dụng. Xem [AI usage disclosure](../docs/ai_usage_disclosure.md).

## 8. Kết luận, giới hạn và khuyến nghị

**Những gì đồ án chứng minh được:** quy trình kiểm toán và khôi phục missingness, metric đúng cấp khách, báo cáo EDA có chú thích tháng không đầy đủ, kiểm định có kiểm soát so sánh nhiều lần, đánh giá holdout không leakage khách, baseline trung thực, negative controls và công khai rủi ro fairness. Các kết quả được lưu bằng JSON/Markdown và tái lập bằng các entrypoint Python; kiểm thử CI chạy không cần raw Kaggle.

**Những gì đồ án *không* chứng minh:** doanh thu thật/đơn vị tiền; tác động nhân quả hay hiệu quả A/B; khả năng dự đoán Churn tương lai; tính đại diện của dữ liệu synthetic; sự công bằng của model cho con người; quyền tái phân phối dataset; hay tính khả dụng của mô hình ngoài lớp học.

**Khuyến nghị:** không cố tuning để tăng ROC-AUC trên nhãn gần ngẫu nhiên; nếu muốn một dự báo có ích, cần nhãn có event timestamp/horizon và dữ liệu thật có quyền sử dụng. Giữ nguyên báo cáo kết quả âm tính; tiếp tục review pháp lý và nguồn gốc dữ liệu, xác nhận đóng góp thành viên, demo dashboard bằng dữ liệu hợp lệ và thực hiện checklist bàn giao. Không merge vào `main` trong lần bàn giao này.

## 9. Khả năng tái lập và bằng chứng

- **Mã chạy chuẩn:** xem [Reproducibility Guide](../docs/reproducibility_guide.md), `requirements.txt`, CI `.github/workflows/phase2-tests.yml`.
- **Metrics có cấu trúc:** `reports/phase1/data_integrity_metrics.json`, `phase4/hypothesis_results.json`, `phase4/regression_metrics.json`, `phase5/classification_metrics.json`, `phase5/phase51_metrics.json`, `phase6/fairness_metrics.json`.
- **Báo cáo kiểm toán cuối:** `python -m src.run_phase7` tạo `reports/phase7/release_audit.md` và `.json` (đọc frozen evidence, không giả lập chạy lại raw dữ liệu).
- **Kiểm tra cuối:** `python -m unittest discover -s tests -v` và xác minh trên máy nhóm; báo cáo không tự biến một lần CI thành kết quả nghiên cứu trên dữ liệu raw.

## Tài liệu tham khảo và truy nguyên

1. Trường Đại học CMC. *INFO3020 — Nhập môn Khoa học dữ liệu, Đề cương chi tiết (2026).* Tài liệu học phần do giảng viên cung cấp.
2. CMC University. *INFO3020 lecture slides W1–W7, W9–W12*, đặc biệt W9 Hypothesis Tests, W10 Regression, W11 Classification/Feature Engineering, W12 Big Data/Ethics/Bias. Tài liệu lớp học.
3. Shriyash Jagtap. [*E-commerce Customer Data for Behavior Analysis*](https://www.kaggle.com/datasets/shriyashjagtap/e-commerce-customer-for-behavior-analysis), Kaggle. **Trạng thái license chưa được xác minh trực tiếp; tham chiếu này không cấp quyền phát hành dữ liệu.**
4. Báo cáo và kết quả máy đọc của chính repository `huybel2012/ds-ecommerce-customers`, nhánh `Dung`: đường dẫn chi tiết theo từng phần trong báo cáo. Các số được dẫn từ kết quả Phase 1–6, không phải từ các bài viết/blog bên ngoài.

---

*Bản dự thảo giao nộp: nhóm cần bổ sung thông tin thành viên, phân công, nguồn license và duyệt lại AI disclosure trước khi xác nhận bản chính thức. Chưa merge `main`.*
