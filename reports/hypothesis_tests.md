# Week 9 — EX9.1: kiểm định giả thuyết

**Dữ liệu:** `ecommerce_customer_cleaned.csv` — 250,000 dòng giao dịch, 49,673 khách hàng (trung bình ~5 đơn/khách).
**Mức ý nghĩa:** α = 0.05 (quy ước, không phải quy luật tự nhiên). Ba giả thuyết là toàn bộ các test đưa vào báo cáo; trước đó mình có chạy thử sơ bộ trên cùng dữ liệu nên đây là phân tích có xem trước, cần nói rõ khi nộp.
**Đơn vị phân tích:** `Churn` và `Gender` là thuộc tính của khách, một khách có nhiều dòng nên H1, H2 gộp về mức khách hàng để các quan sát độc lập.
**Phạm vi suy rộng:** chỉ là mẫu giao dịch của dataset này (Kaggle), không tự suy ra thị trường thương mại điện tử nói chung.

---

## H1 — Churn theo giới tính
- **H0:** tỉ lệ churn nam = nữ. **H1:** khác nhau (hai phía).
- **Test:** hai tỉ lệ z-test (tương đương chi-square 2×2 không hiệu chỉnh). Điều kiện: expected count nhỏ nhất = 4,947 (≫ 5), khách độc lập (mức khách hàng).
- **Kết quả:** churn nam 20.30% (n = 24,715), nữ 19.74% (n = 24,958).
- **Effect size:** chênh lệch **+0.56 điểm %** (nam − nữ), 95% CI [-0.15; +1.26] điểm %, Cohen's h = 0.014, z = 1.55, **p = 0.120**.
- **Kết luận:** không bác bỏ H0 ở α = 0.05. Khoảng tin cậy chứa 0 và hẹp: nếu có chênh lệch giữa hai giới thì nhiều khả năng không quá ~1.3 điểm %. Không chứng minh được hai giới giống nhau, chỉ là chưa có đủ bằng chứng để nói khác.

## H2 — Chi tiêu của khách churn so với khách còn lại
- **H0:** chi tiêu trung bình mỗi đơn (trung bình theo khách) của nhóm churn = nhóm còn lại. **H1:** khác nhau (hai phía).
- **Test:** Welch t-test (mặc định theo slide). Kiểm tra hình dạng: skew = -0.05 (churn), 0.01 (còn lại), gần đối xứng, n lớn nên Welch phù hợp. Kiểm tra chéo bằng Mann-Whitney: p = 0.359.
- **Kết quả:** churn 2,727.9 (n = 9,942), còn lại 2,723.8 (n = 39,731).
- **Effect size:** chênh lệch **+4.1** (churn − còn lại), 95% CI [-12.2; +20.3], Cohen's d = 0.006, t = 0.49, **p = 0.624**.
- **Kết luận:** không bác bỏ H0. Chênh lệch ước tính chỉ ~0.15% so với mức chi tiêu trung bình và khoảng tin cậy chứa 0, nên không có bằng chứng rằng chi tiêu liên quan tới việc rời bỏ.

## H3 — Giá trị đơn hàng theo danh mục
- **H0:** trung bình Total Purchase Amount bằng nhau ở 4 danh mục. **H1:** ít nhất một danh mục khác.
- **Test:** one-way ANOVA (4 nhóm số). Kiểm chéo bằng Kruskal-Wallis: p = 0.057. Tukey HSD làm hậu kiểm: p nhỏ nhất giữa các cặp = 0.039.
- **Kết quả:** trung bình theo danh mục: Books 2,735.7, Clothing 2,725.2, Electronics 2,721.9, Home 2,713.5 (n = 250,000 dòng).
- **Effect size:** khoảng cách lớn nhất (Books − Home) = **+22.2**, 95% CI cluster-bootstrap theo khách [+6.8; +38.8]; η² = 0.00003. F = 2.51, **p = 0.057**.
- **Kết luận:** không bác bỏ H0 ở α = 0.05 (ANOVA p = 0.057, Kruskal-Wallis p = 0.057), tức bằng chứng ở mức biên. Hậu kiểm Tukey có một cặp p = 0.039, nhưng vì omnibus không đạt ngưỡng nên chỉ coi là gợi ý thăm dò. Điều quan trọng hơn p: độ lớn rất nhỏ, khoảng cách lớn nhất chỉ 0.82% mức trung bình và η² ≈ 0.003% phương sai, nên dù có đạt ý nghĩa thống kê cũng không có ý nghĩa kinh doanh.
- **Lưu ý 1:** ANOVA ở mức dòng giả định các dòng độc lập, trong khi một khách có nhiều dòng, nên p có thể hơi lạc quan. CI cluster-bootstrap theo khách ở trên đã tính đến điều này.
- **Lưu ý 2:** cặp (Books, Home) được chọn vì nó là cặp chênh nhiều nhất sau khi nhìn dữ liệu, nên CI [+6.8; +38.8] của khoảng cách này bị lệch lạc quan (winner's curse). Không nên đọc nó như bằng chứng riêng của cặp đó.

---

## Nhìn chung
- Cả ba giả thuyết đều **không bác bỏ được H0** ở α = 0.05 (H3 ở mức biên). Với 3 test, ngưỡng Bonferroni là 0.05/3 ≈ 0.0167 và kết luận không đổi.
- H1 và H2 có CI hẹp quanh 0 trên mẫu lớn (hàng chục nghìn khách), nên nếu có hiệu ứng thì rất nhỏ. H3 có hiệu ứng ước tính rất nhỏ ở mọi cách đọc.
- Nhất quán với Week 10: R² của mô hình hồi quy chỉ ~0.003, tức các biến hành vi giải thích rất ít biến thiên của giá trị đơn hàng.
- Giới hạn: Returns thiếu 19% và đã được fill 0 trong file đã clean, nên không đưa vào giả thuyết ở đây. Muốn test Returns thì chỉ dùng các dòng `Returns_was_missing == 0` (cột này được tạo ở bước 3.2 của notebook và không có trong file CSV; mình chưa đọc được file parquet để xác nhận nó có ở đó).
- Kết luận chỉ áp dụng cho mẫu của dataset này, không suy ra toàn bộ thị trường.
