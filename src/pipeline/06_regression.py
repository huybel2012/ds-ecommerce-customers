"""
Week 10 — Regression analysis trên dữ liệu ecommerce (đã clean ở Week 3-5).

Mục tiêu hồi quy: dự đoán `Total Purchase Amount` (giá trị đơn hàng).
Quy trình theo slide "The regression workflow":
  1. Split first (GroupShuffleSplit theo Customer ID)
  2. Build the pipeline (impute/scale/encode nằm trong Pipeline)
  3. Fit on train only
  4. Diagnose (LINE, VIF, Cook's distance)
  5. Evaluate on the test set (MAE, RMSE, R2 + baseline trung bình)
  6. Interpret (một câu cho mỗi hệ số, kèm khoảng tin cậy 95%)

Chạy từ thư mục gốc của project:
    python src/pipeline/06_regression.py
(statsmodels không bắt buộc: OLS + CI + HC3 + Breusch-Pagan được tính bằng numpy/scipy.)
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupShuffleSplit, GroupKFold, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

DATA = "data/processed/ecommerce_customer_cleaned.csv"
FIG_DIR = "reports/figures"
OUT_MD = "reports/regression_results.md"
os.makedirs(FIG_DIR, exist_ok=True)

TARGET = "Total Purchase Amount"
NUM = ["Product Price", "Quantity", "Age"]
CAT = ["Product Category", "Payment Method", "Gender", "Returns"]
# Reference level = nhóm phổ biến nhất / dễ so sánh nhất (theo slide 55)
REF = {"Product Category": "Clothing", "Payment Method": "Credit Card",
       "Gender": "Female", "Returns": 0}

df = pd.read_csv(DATA, parse_dates=["Purchase Date"])
y = df[TARGET]
X = df[NUM + CAT]
groups = df["Customer ID"]

# ---------------------------------------------------------------- 1. SPLIT
gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
tr_idx, te_idx = next(gss.split(X, y, groups))
X_tr, X_te, y_tr, y_te = X.iloc[tr_idx], X.iloc[te_idx], y.iloc[tr_idx], y.iloc[te_idx]
g_tr = groups.iloc[tr_idx]
assert set(g_tr).isdisjoint(set(groups.iloc[te_idx])), "một khách nằm ở cả train và test"
print(f"train={len(X_tr):,}  test={len(X_te):,}  (tách theo Customer ID, không rò rỉ)")

# ------------------------------------------------------------ 2. PIPELINE
prep = ColumnTransformer([
    ("num", StandardScaler(), NUM),
    ("cat", OneHotEncoder(drop="first", sparse_output=False), CAT),
])


def metrics(model, Xa, ya):
    p = model.predict(Xa)
    return dict(MAE=mean_absolute_error(ya, p),
                RMSE=np.sqrt(mean_squared_error(ya, p)),
                R2=r2_score(ya, p))


# ------------------------------------------------- 3 & 5. FIT + EVALUATE
results = {}
baseline = DummyRegressor(strategy="mean").fit(X_tr, y_tr)
results["Baseline (mean)"] = metrics(baseline, X_te, y_te)

ols = Pipeline([("prep", prep), ("model", LinearRegression())]).fit(X_tr, y_tr)
results["OLS"] = metrics(ols, X_te, y_te)

cv = GroupKFold(n_splits=5)
grid_params = {"model__alpha": [0.1, 1, 10, 100, 1000]}
ridge = GridSearchCV(Pipeline([("prep", prep), ("model", Ridge())]), grid_params,
                     cv=cv, scoring="neg_mean_absolute_error")
ridge.fit(X_tr, y_tr, groups=g_tr)
results[f"Ridge (alpha={ridge.best_params_['model__alpha']})"] = metrics(ridge, X_te, y_te)

lasso = GridSearchCV(Pipeline([("prep", prep), ("model", Lasso(max_iter=10000))]),
                     {"model__alpha": [0.01, 0.1, 1, 10, 100]},
                     cv=cv, scoring="neg_mean_absolute_error")
lasso.fit(X_tr, y_tr, groups=g_tr)
results[f"Lasso (alpha={lasso.best_params_['model__alpha']})"] = metrics(lasso, X_te, y_te)

res_df = pd.DataFrame(results).T.round(4)
print("\n== Kết quả trên TEST ==\n", res_df)

# Train vs test để thấy overfitting (không có: mô hình tuyến tính, ít biến)
train_r2 = r2_score(y_tr, ols.predict(X_tr))
test_r2 = r2_score(y_te, ols.predict(X_te))

# ------------------------- OLS có suy diễn (thay cho statsmodels) trên TRAIN
def design(Xa):
    D = pd.get_dummies(Xa, columns=CAT, drop_first=False).astype(float)
    for c, r in REF.items():
        D = D.drop(columns=[f"{c}_{r}"])
    D.insert(0, "const", 1.0)
    return D


D = design(X_tr)
A = D.values
yv = y_tr.values.astype(float)
n, k = A.shape
XtX_inv = np.linalg.inv(A.T @ A)
beta = XtX_inv @ A.T @ yv
fitted = A @ beta
resid = yv - fitted
dof = n - k
sigma2 = resid @ resid / dof
se = np.sqrt(np.diag(XtX_inv) * sigma2)

# HC3 robust SE (slide 53, Fix 2)
h = np.einsum("ij,jk,ik->i", A, XtX_inv, A)  # leverage
meat = (A * (resid / (1 - h))[:, None]).T @ (A * (resid / (1 - h))[:, None])
se_hc3 = np.sqrt(np.diag(XtX_inv @ meat @ XtX_inv))

tcrit = stats.t.ppf(0.975, dof)
coef = pd.DataFrame({
    "coef": beta, "se_HC3": se_hc3,
    "ci_low": beta - tcrit * se_hc3, "ci_high": beta + tcrit * se_hc3,
    "p": 2 * stats.t.sf(np.abs(beta / se_hc3), dof),
}, index=D.columns)

ss_res, ss_tot = resid @ resid, ((yv - yv.mean()) ** 2).sum()
r2_in = 1 - ss_res / ss_tot
adj_r2 = 1 - (1 - r2_in) * (n - 1) / dof
F = ((ss_tot - ss_res) / (k - 1)) / sigma2
F_p = stats.f.sf(F, k - 1, dof)

# ------------------------------------------------------------ 4. DIAGNOSE
# VIF (slide 52)
feat = D.drop(columns="const")
vif = {}
for c in feat.columns:
    others = np.column_stack([np.ones(n), feat.drop(columns=c).values])
    b = np.linalg.lstsq(others, feat[c].values, rcond=None)[0]
    r = feat[c].values - others @ b
    vif[c] = 1 / (1 - (1 - (r @ r) / ((feat[c] - feat[c].mean()) ** 2).sum()))
vif = pd.Series(vif).sort_values(ascending=False)

# Breusch-Pagan (LM) — slide 53
e2 = resid ** 2
b_bp = np.linalg.lstsq(A, e2, rcond=None)[0]
r_bp = e2 - A @ b_bp
r2_bp = 1 - (r_bp @ r_bp) / ((e2 - e2.mean()) ** 2).sum()
bp_stat = n * r2_bp
bp_p = stats.chi2.sf(bp_stat, k - 1)

# Cook's distance — slide 54 (ngưỡng 4/n)
cooks = (resid ** 2 / (k * sigma2)) * (h / (1 - h) ** 2)
n_infl = int((cooks > 4 / n).sum())

# Durbin-Watson theo thứ tự thời gian (độc lập — slide 50)
order = np.argsort(df.loc[X_tr.index, "Purchase Date"].values)
r_sorted = resid[order]
dw = np.sum(np.diff(r_sorted) ** 2) / np.sum(r_sorted ** 2)

# Plots
rng = np.random.default_rng(0)
s = rng.choice(n, size=min(8000, n), replace=False)
fig, ax = plt.subplots(1, 3, figsize=(16, 4.6))
ax[0].scatter(fitted[s], resid[s], s=4, alpha=.35)
ax[0].axhline(0, color="r", lw=1)
ax[0].set(title="Residuals vs Fitted", xlabel="Fitted values", ylabel="Residuals")
(osm, osr), (sl, ic, _) = stats.probplot(resid[s], dist="norm")
ax[1].scatter(osm, osr, s=4, alpha=.35)
ax[1].plot(osm, sl * osm + ic, "r", lw=1)
ax[1].set(title="QQ-plot of residuals", xlabel="Theoretical quantiles", ylabel="Sample quantiles")
ax[2].vlines(np.arange(len(s)), 0, cooks[s], lw=.5)
ax[2].axhline(4 / n, color="r", lw=1, ls="--", label="4/n")
ax[2].set(title="Cook's distance (mẫu 8,000 điểm)", xlabel="Observation", ylabel="Cook's D")
ax[2].legend()
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/06_residual_diagnostics.png", dpi=130)
plt.close()

# Coefficient chart with 95% CI (slide 65)
cc = coef.drop(index="const").sort_values("coef")
fig, ax = plt.subplots(figsize=(8, 5.5))
colors = ["#2c7fb8" if (lo > 0 or hi < 0) else "#bbbbbb"
          for lo, hi in zip(cc.ci_low, cc.ci_high)]
ax.barh(cc.index, cc.coef, color=colors)
ax.errorbar(cc.coef, cc.index, xerr=[cc.coef - cc.ci_low, cc.ci_high - cc.coef],
            fmt="none", ecolor="k", capsize=3, lw=1)
ax.axvline(0, color="k", lw=.8)
ax.set(title="Hệ số OLS (VND-đơn vị của y) và khoảng tin cậy 95% (HC3)\nxám = khoảng cắt 0 → không có bằng chứng",
       xlabel="Hệ số")
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/07_coefficients_ci.png", dpi=130)
plt.close()

# Actual vs predicted on test
pred_te = ols.predict(X_te)
fig, ax = plt.subplots(figsize=(5.5, 5))
idx = rng.choice(len(y_te), size=6000, replace=False)
ax.scatter(y_te.values[idx], pred_te[idx], s=4, alpha=.3)
ax.plot([100, 5350], [100, 5350], "r", lw=1, label="dự đoán hoàn hảo")
ax.set(xlabel="Thực tế", ylabel="Dự đoán", title="Test set: thực tế vs dự đoán (OLS)")
ax.legend()
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/08_actual_vs_predicted.png", dpi=130)
plt.close()

# ------------------------------------------- OLS vs Ridge vs Lasso (EX10.4)
names = list(prep.fit(X_tr).get_feature_names_out())
cmp = pd.DataFrame({
    "OLS": ols.named_steps["model"].coef_,
    "Ridge": ridge.best_estimator_.named_steps["model"].coef_,
    "Lasso": lasso.best_estimator_.named_steps["model"].coef_,
}, index=names).round(2)
print("\n== Hệ số (đã scale) OLS vs Ridge vs Lasso ==\n", cmp)

# --------------------------------------------------- 6. INTERPRET (EX10.3)
rng_txt = {c: (int(X_tr[c].min()), int(X_tr[c].max())) for c in NUM}
unit = {"Product Price": "mỗi 1 đơn vị giá sản phẩm tăng thêm",
        "Quantity": "mỗi 1 sản phẩm mua thêm", "Age": "mỗi 1 tuổi tăng thêm"}


def sentence(name, row):
    others = [c for c in coef.index if c not in ("const", name)]
    hold = "các biến còn lại trong mô hình"
    ci = f"[{row.ci_low:,.1f}; {row.ci_high:,.1f}]"
    sig = "khoảng tin cậy cắt 0 → chưa đủ bằng chứng" if row.ci_low < 0 < row.ci_high else "khoảng tin cậy không chứa 0"
    if name in NUM:
        lo, hi = rng_txt[name]
        return (f"- **{name}**: {unit[name]} có liên quan tới tổng giá trị đơn hàng thay đổi "
                f"{row.coef:+,.1f} (95% CI {ci}) đơn vị tiền, giữ nguyên {hold}; "
                f"chỉ đúng trong khoảng quan sát {lo}–{hi}. ({sig})")
    col, lvl = name.rsplit("_", 1)[0], name.split("_", 1)[1] if False else None
    for c in CAT:
        if name.startswith(c + "_"):
            lvl = name[len(c) + 1:]
            return (f"- **{name}**: so với nhóm tham chiếu '{REF[c]}', {c} = '{lvl}' có liên quan tới "
                    f"tổng giá trị đơn hàng thay đổi {row.coef:+,.1f} (95% CI {ci}) đơn vị tiền, "
                    f"giữ nguyên {hold}; chỉ đúng với các nhóm đã quan sát. ({sig})")


sentences = [sentence(nm, coef.loc[nm]) for nm in coef.index if nm != "const"]

# ------------------------------------------------------------- REPORT
def md_table(d, fmt="{:,.3f}"):
    d = d.copy()
    head = "| " + " | ".join([d.index.name or ""] + list(d.columns)) + " |"
    sep = "|" + "---|" * (len(d.columns) + 1)
    rows = ["| " + " | ".join([str(i)] + [fmt.format(v) for v in r]) + " |" for i, r in zip(d.index, d.values)]
    return "\n".join([head, sep] + rows)


md = f"""# Regression results — Week 10

**Biến mục tiêu:** `{TARGET}` · **Biến giải thích:** {', '.join(NUM + CAT)}
**Split:** 80/20 theo `Customer ID` (GroupShuffleSplit, seed 42) · train = {len(X_tr):,}, test = {len(X_te):,}

## 1. Đánh giá trên TEST (EX10.1)
{md_table(res_df)}

R² trên train = {train_r2:.4f}, trên test = {test_r2:.4f} → không overfit, nhưng cũng không có tín hiệu.

## 2. Chẩn đoán (EX10.2)
- Adj. R² (train, OLS) = {adj_r2:.4f}; F = {F:.2f}, p = {F_p:.3f}
- Breusch-Pagan: LM = {bp_stat:.2f}, p = {bp_p:.3f}
- Durbin-Watson (sắp theo thời gian) = {dw:.3f} (≈2 là độc lập)
- Cook's D > 4/n: {n_infl:,} điểm ({n_infl / n:.1%})
- VIF lớn nhất = {vif.max():.2f} (ngưỡng 5)

Hình: `reports/figures/06_residual_diagnostics.png`

## 3. Hệ số OLS, sai số chuẩn HC3 (EX10.3)
{md_table(coef.round(4), "{:,.4f}")}

Hình: `reports/figures/07_coefficients_ci.png`

### Diễn giải từng hệ số
{chr(10).join(sentences)}

## 4. OLS vs Ridge vs Lasso (EX10.4) — hệ số trên dữ liệu đã scale
{md_table(cmp, "{:,.2f}")}
"""
with open(OUT_MD, "w", encoding="utf-8") as f:
    f.write(md)

print("\n== Chẩn đoán ==")
print(f"adjR2={adj_r2:.4f} F={F:.2f} p={F_p:.3f} | BP p={bp_p:.3f} | DW={dw:.3f} | "
      f"Cook>4/n: {n_infl} | max VIF={vif.max():.2f}")
print(f"train R2={train_r2:.4f} test R2={test_r2:.4f}")
print(coef.round(3))
print(f"\nĐã ghi {OUT_MD} và 3 hình trong {FIG_DIR}/")
