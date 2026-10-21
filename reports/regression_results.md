# Regression results — Week 10

**Biến mục tiêu:** `Total Purchase Amount` · **Biến giải thích:** Product Price, Quantity, Age, Product Category, Payment Method, Gender, Returns
**Split:** 80/20 theo `Customer ID` (GroupShuffleSplit, seed 42) · train = 199,771, test = 50,229

## 1. Đánh giá trên TEST (EX10.1)
|  | MAE | RMSE | R2 |
|---|---|---|---|
| Baseline (mean) | 1,251.401 | 1,445.042 | -0.000 |
| OLS | 1,250.124 | 1,442.725 | 0.003 |
| Ridge (alpha=1000) | 1,250.127 | 1,442.727 | 0.003 |
| Lasso (alpha=1) | 1,250.145 | 1,442.735 | 0.003 |

R² trên train = 0.0026, trên test = 0.0032 → không overfit, nhưng cũng không có tín hiệu.

## 2. Chẩn đoán (EX10.2)
- Adj. R² (train, OLS) = 0.0025; F = 47.26, p = 0.000
- Breusch-Pagan: LM = 15.50, p = 0.161
- Durbin-Watson (sắp theo thời gian) = 2.004 (≈2 là độc lập)
- Cook's D > 4/n: 1,212 điểm (0.6%)
- VIF lớn nhất = 1.40 (ngưỡng 5)

Hình: `reports/figures/06_residual_diagnostics.png`

## 3. Hệ số OLS, sai số chuẩn HC3 (EX10.3)
|  | coef | se_HC3 | ci_low | ci_high | p |
|---|---|---|---|---|---|
| const | 2,528.1792 | 15.2403 | 2,498.3086 | 2,558.0498 | 0.0000 |
| Product Price | -0.0264 | 0.0228 | -0.0710 | 0.0182 | 0.2462 |
| Quantity | 0.6845 | 2.2783 | -3.7809 | 5.1500 | 0.7638 |
| Age | 4.7411 | 0.2095 | 4.3304 | 5.1517 | 0.0000 |
| Product Category_Books | 5.7429 | 8.3317 | -10.5869 | 22.0728 | 0.4906 |
| Product Category_Electronics | -2.9963 | 9.2902 | -21.2050 | 15.2124 | 0.7471 |
| Product Category_Home | -17.8438 | 9.3102 | -36.0916 | 0.4041 | 0.0553 |
| Payment Method_Cash | -3.0421 | 8.7998 | -20.2895 | 14.2053 | 0.7296 |
| Payment Method_Crypto | -7.3357 | 11.4448 | -29.7672 | 15.0959 | 0.5215 |
| Payment Method_PayPal | -1.5889 | 7.7910 | -16.8591 | 13.6813 | 0.8384 |
| Gender_Male | -1.9754 | 6.4465 | -14.6104 | 10.6596 | 0.7593 |
| Returns_1 | -3.9783 | 6.5647 | -16.8450 | 8.8884 | 0.5445 |

Hình: `reports/figures/07_coefficients_ci.png`

### Diễn giải từng hệ số
- **Product Price**: mỗi 1 đơn vị giá sản phẩm tăng thêm có liên quan tới tổng giá trị đơn hàng thay đổi -0.0 (95% CI [-0.1; 0.0]) đơn vị tiền, giữ nguyên các biến còn lại trong mô hình; chỉ đúng trong khoảng quan sát 10–500. (khoảng tin cậy cắt 0 → chưa đủ bằng chứng)
- **Quantity**: mỗi 1 sản phẩm mua thêm có liên quan tới tổng giá trị đơn hàng thay đổi +0.7 (95% CI [-3.8; 5.2]) đơn vị tiền, giữ nguyên các biến còn lại trong mô hình; chỉ đúng trong khoảng quan sát 1–5. (khoảng tin cậy cắt 0 → chưa đủ bằng chứng)
- **Age**: mỗi 1 tuổi tăng thêm có liên quan tới tổng giá trị đơn hàng thay đổi +4.7 (95% CI [4.3; 5.2]) đơn vị tiền, giữ nguyên các biến còn lại trong mô hình; chỉ đúng trong khoảng quan sát 18–70. (khoảng tin cậy không chứa 0)
- **Product Category_Books**: so với nhóm tham chiếu 'Clothing', Product Category = 'Books' có liên quan tới tổng giá trị đơn hàng thay đổi +5.7 (95% CI [-10.6; 22.1]) đơn vị tiền, giữ nguyên các biến còn lại trong mô hình; chỉ đúng với các nhóm đã quan sát. (khoảng tin cậy cắt 0 → chưa đủ bằng chứng)
- **Product Category_Electronics**: so với nhóm tham chiếu 'Clothing', Product Category = 'Electronics' có liên quan tới tổng giá trị đơn hàng thay đổi -3.0 (95% CI [-21.2; 15.2]) đơn vị tiền, giữ nguyên các biến còn lại trong mô hình; chỉ đúng với các nhóm đã quan sát. (khoảng tin cậy cắt 0 → chưa đủ bằng chứng)
- **Product Category_Home**: so với nhóm tham chiếu 'Clothing', Product Category = 'Home' có liên quan tới tổng giá trị đơn hàng thay đổi -17.8 (95% CI [-36.1; 0.4]) đơn vị tiền, giữ nguyên các biến còn lại trong mô hình; chỉ đúng với các nhóm đã quan sát. (khoảng tin cậy cắt 0 → chưa đủ bằng chứng)
- **Payment Method_Cash**: so với nhóm tham chiếu 'Credit Card', Payment Method = 'Cash' có liên quan tới tổng giá trị đơn hàng thay đổi -3.0 (95% CI [-20.3; 14.2]) đơn vị tiền, giữ nguyên các biến còn lại trong mô hình; chỉ đúng với các nhóm đã quan sát. (khoảng tin cậy cắt 0 → chưa đủ bằng chứng)
- **Payment Method_Crypto**: so với nhóm tham chiếu 'Credit Card', Payment Method = 'Crypto' có liên quan tới tổng giá trị đơn hàng thay đổi -7.3 (95% CI [-29.8; 15.1]) đơn vị tiền, giữ nguyên các biến còn lại trong mô hình; chỉ đúng với các nhóm đã quan sát. (khoảng tin cậy cắt 0 → chưa đủ bằng chứng)
- **Payment Method_PayPal**: so với nhóm tham chiếu 'Credit Card', Payment Method = 'PayPal' có liên quan tới tổng giá trị đơn hàng thay đổi -1.6 (95% CI [-16.9; 13.7]) đơn vị tiền, giữ nguyên các biến còn lại trong mô hình; chỉ đúng với các nhóm đã quan sát. (khoảng tin cậy cắt 0 → chưa đủ bằng chứng)
- **Gender_Male**: so với nhóm tham chiếu 'Female', Gender = 'Male' có liên quan tới tổng giá trị đơn hàng thay đổi -2.0 (95% CI [-14.6; 10.7]) đơn vị tiền, giữ nguyên các biến còn lại trong mô hình; chỉ đúng với các nhóm đã quan sát. (khoảng tin cậy cắt 0 → chưa đủ bằng chứng)
- **Returns_1**: so với nhóm tham chiếu '0', Returns = '1' có liên quan tới tổng giá trị đơn hàng thay đổi -4.0 (95% CI [-16.8; 8.9]) đơn vị tiền, giữ nguyên các biến còn lại trong mô hình; chỉ đúng với các nhóm đã quan sát. (khoảng tin cậy cắt 0 → chưa đủ bằng chứng)

## 4. OLS vs Ridge vs Lasso (EX10.4) — hệ số trên dữ liệu đã scale
|  | OLS | Ridge | Lasso |
|---|---|---|---|
| num__Product Price | -3.73 | -3.71 | -2.72 |
| num__Quantity | 0.97 | 0.96 | 0.00 |
| num__Age | 72.84 | 72.48 | 71.85 |
| cat__Product Category_Clothing | -5.74 | -5.07 | -0.00 |
| cat__Product Category_Electronics | -8.74 | -7.95 | -0.00 |
| cat__Product Category_Home | -23.59 | -22.43 | -12.97 |
| cat__Payment Method_Credit Card | 3.04 | 2.99 | 0.00 |
| cat__Payment Method_Crypto | -4.29 | -4.09 | -0.00 |
| cat__Payment Method_PayPal | 1.45 | 1.42 | 0.00 |
| cat__Gender_Male | -1.98 | -1.94 | -0.00 |
| cat__Returns_1 | -3.98 | -3.90 | -0.00 |
