"""Week 9 — EX9.1: ba giả thuyết trên dataset e-commerce customers.

Chạy từ thư mục gốc repo:  python src/pipeline/07_hypothesis_tests.py
Chỉ cần pandas, numpy, scipy, matplotlib (không cần statsmodels).

Quy ước: giả thuyết và alpha được viết TRƯỚC khi nhìn kết quả (xem HYPOTHESES).
Đơn vị phân tích: Churn/Gender là thuộc tính của KHÁCH HÀNG (mỗi khách ~5 dòng),
nên H1, H2 chạy ở mức khách hàng để các quan sát độc lập.
"""
from pathlib import Path
import numpy as np, pandas as pd
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2] if "__file__" in globals() else Path(".")
DATA = ROOT / "data/processed/ecommerce_customer_cleaned.csv"
REPORT = ROOT / "reports/hypothesis_tests.md"
FIG = ROOT / "reports/figures/09_hypothesis_tests.png"
ALPHA, SEED, B = 0.05, 42, 5000
rng = np.random.default_rng(SEED)

HYPOTHESES = {
 "H1": "Tỉ lệ churn của khách nam khác khách nữ.  H0: p_nam = p_nữ.",
 "H2": "Chi tiêu trung bình mỗi đơn của khách đã churn khác khách còn lại.  H0: μ_churn = μ_giữ chân.",
 "H3": "Giá trị đơn hàng (Total Purchase Amount) khác nhau giữa 4 danh mục sản phẩm.  H0: μ_Books = μ_Clothing = μ_Electronics = μ_Home.",
}

df = pd.read_csv(DATA)
amt = "Total Purchase Amount"
cust = df.groupby("Customer ID").agg(Gender=("Gender", "first"), Churn=("Churn", "first"),
                                     mean_amt=(amt, "mean"), n=(amt, "size")).reset_index()
N_CUST = len(cust)
out = {}

# ---------- H1: hai tỉ lệ (customer level) ----------
m = cust[cust.Gender == "Male"]; f = cust[cust.Gender == "Female"]
x1, n1, x0, n0 = m.Churn.sum(), len(m), f.Churn.sum(), len(f)
p1, p0 = x1 / n1, x0 / n0
diff = p1 - p0
pp = (x1 + x0) / (n1 + n0)
z = diff / np.sqrt(pp * (1 - pp) * (1 / n1 + 1 / n0))
p_h1 = 2 * (1 - stats.norm.cdf(abs(z)))
se = np.sqrt(p1 * (1 - p1) / n1 + p0 * (1 - p0) / n0)
ci_h1 = (diff - 1.96 * se, diff + 1.96 * se)
cohen_h = 2 * np.arcsin(np.sqrt(p1)) - 2 * np.arcsin(np.sqrt(p0))
exp_min = stats.chi2_contingency(pd.crosstab(cust.Gender, cust.Churn), correction=False)[3].min()
out["H1"] = dict(p1=p1, p0=p0, diff=diff, ci=ci_h1, z=z, p=p_h1, h=cohen_h, n1=n1, n0=n0, exp_min=exp_min)

# ---------- H2: hai nhóm số (customer level) ----------
a = cust.loc[cust.Churn == 1, "mean_amt"]; b = cust.loc[cust.Churn == 0, "mean_amt"]
t, p_t = stats.ttest_ind(a, b, equal_var=False)
u, p_u = stats.mannwhitneyu(a, b, alternative="two-sided")
d_ = a.mean() - b.mean()
se2 = np.sqrt(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b))
dfw = se2**4 / ((a.var(ddof=1) / len(a))**2 / (len(a) - 1) + (b.var(ddof=1) / len(b))**2 / (len(b) - 1))
tcrit = stats.t.ppf(0.975, dfw)
ci_h2 = (d_ - tcrit * se2, d_ + tcrit * se2)
sp = np.sqrt(((len(a) - 1) * a.var(ddof=1) + (len(b) - 1) * b.var(ddof=1)) / (len(a) + len(b) - 2))
out["H2"] = dict(ma=a.mean(), mb=b.mean(), diff=d_, ci=ci_h2, t=t, p=p_t, p_mw=p_u, d=d_ / sp,
                 skew_a=a.skew(), skew_b=b.skew(), na=len(a), nb=len(b))

# ---------- H3: 4 nhóm (row level) + kiểm tra phụ thuộc theo khách ----------
cats = sorted(df["Product Category"].unique())
groups = [df.loc[df["Product Category"] == c, amt].values for c in cats]
F, p_F = stats.f_oneway(*groups)
H, p_H = stats.kruskal(*groups)
grand = df[amt].mean()
ssb = sum(len(g) * (g.mean() - grand) ** 2 for g in groups)
sst = ((df[amt] - grand) ** 2).sum()
eta2 = ssb / sst
means = {c: g.mean() for c, g in zip(cats, groups)}
hi, lo = max(means, key=means.get), min(means, key=means.get)
tuk = stats.tukey_hsd(*groups)
# bootstrap theo KHÁCH (cluster) cho khoảng cách hi-lo, vì 1 khách có nhiều dòng
cid = df["Customer ID"].astype("category").cat.codes.values
cat_idx = {c: i for i, c in enumerate(cats)}
S = np.zeros((cid.max() + 1, len(cats))); Cn = np.zeros_like(S)
np.add.at(S, (cid, df["Product Category"].map(cat_idx).values), df[amt].values)
np.add.at(Cn, (cid, df["Product Category"].map(cat_idx).values), 1)
gap = []
for _ in range(2000):
    w = np.bincount(rng.integers(0, len(S), len(S)), minlength=len(S))
    mu = (w @ S) / (w @ Cn)
    gap.append(mu[cat_idx[hi]] - mu[cat_idx[lo]])
ci_gap = np.percentile(gap, [2.5, 97.5])
out["H3"] = dict(means=means, F=F, p=p_F, p_kw=p_H, eta2=eta2, hi=hi, lo=lo, gap=means[hi] - means[lo],
                 ci_gap=ci_gap, n=len(df), tukey_min_p=float(np.min(tuk.pvalue[np.triu_indices(4, 1)])))

# ---------- Hình ----------
fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
ax[0].bar(["Nữ", "Nam"], [p0 * 100, p1 * 100], color=["#0E76C8", "#E07B27"])
ax[0].errorbar([0, 1], [p0 * 100, p1 * 100], yerr=[1.96 * np.sqrt(p0 * (1 - p0) / n0) * 100, 1.96 * np.sqrt(p1 * (1 - p1) / n1) * 100],
               fmt="none", color="black", capsize=5)
ax[0].set_ylim(0, 25); ax[0].set_ylabel("Tỉ lệ churn (%)"); ax[0].set_title(f"H1 · p = {p_h1:.3f}")
ax[1].bar(["Còn lại", "Churn"], [b.mean(), a.mean()], color=["#0E76C8", "#E07B27"])
ax[1].errorbar([0, 1], [b.mean(), a.mean()], yerr=[tcrit * b.std() / np.sqrt(len(b)), tcrit * a.std() / np.sqrt(len(a))], fmt="none", color="black", capsize=5)
ax[1].set_ylim(2500, 2900); ax[1].set_ylabel("Chi tiêu TB mỗi đơn"); ax[1].set_title(f"H2 · p = {p_t:.3f}  (trục y cắt từ 2500)")
mu = [means[c] for c in cats]; ci = [1.96 * g.std() / np.sqrt(len(g)) for g in groups]
ax[2].bar(cats, mu, yerr=ci, capsize=5, color="#0E76C8"); ax[2].set_ylim(2500, 2900)
ax[2].set_title(f"H3 · p = {p_F:.3f}  (trục y cắt từ 2500)"); ax[2].set_ylabel("Total Purchase Amount TB")
for x in ax: x.spines[["top", "right"]].set_visible(False)
plt.tight_layout(); FIG.parent.mkdir(parents=True, exist_ok=True); plt.savefig(FIG, dpi=150)

# ---------- Báo cáo ----------
h1, h2, h3 = out["H1"], out["H2"], out["H3"]
def verdict(p): return "bác bỏ H0" if p < ALPHA else "không bác bỏ H0"
md = f"""# Week 9 — EX9.1: kiểm định giả thuyết

**Dữ liệu:** `ecommerce_customer_cleaned.csv` — {len(df):,} dòng giao dịch, {N_CUST:,} khách hàng (trung bình ~5 đơn/khách).
**Mức ý nghĩa:** α = {ALPHA} (quy ước, không phải quy luật tự nhiên). Ba giả thuyết là toàn bộ các test đưa vào báo cáo; trước đó mình có chạy thử sơ bộ trên cùng dữ liệu nên đây là phân tích có xem trước, cần nói rõ khi nộp.
**Đơn vị phân tích:** `Churn` và `Gender` là thuộc tính của khách, một khách có nhiều dòng nên H1, H2 gộp về mức khách hàng để các quan sát độc lập.
**Phạm vi suy rộng:** chỉ là mẫu giao dịch của dataset này (Kaggle), không tự suy ra thị trường thương mại điện tử nói chung.

---

## H1 — Churn theo giới tính
- **H0:** tỉ lệ churn nam = nữ. **H1:** khác nhau (hai phía).
- **Test:** hai tỉ lệ z-test (tương đương chi-square 2×2 không hiệu chỉnh). Điều kiện: expected count nhỏ nhất = {h1['exp_min']:,.0f} (≫ 5), khách độc lập (mức khách hàng).
- **Kết quả:** churn nam {h1['p1']*100:.2f}% (n = {h1['n1']:,}), nữ {h1['p0']*100:.2f}% (n = {h1['n0']:,}).
- **Effect size:** chênh lệch **{h1['diff']*100:+.2f} điểm %** (nam − nữ), 95% CI [{h1['ci'][0]*100:+.2f}; {h1['ci'][1]*100:+.2f}] điểm %, Cohen's h = {h1['h']:.3f}, z = {h1['z']:.2f}, **p = {h1['p']:.3f}**.
- **Kết luận:** {verdict(h1['p'])} ở α = 0.05. Khoảng tin cậy chứa 0 và hẹp: nếu có chênh lệch giữa hai giới thì nhiều khả năng không quá ~{max(abs(h1['ci'][0]), abs(h1['ci'][1]))*100:.1f} điểm %. Không chứng minh được hai giới giống nhau, chỉ là chưa có đủ bằng chứng để nói khác.

## H2 — Chi tiêu của khách churn so với khách còn lại
- **H0:** chi tiêu trung bình mỗi đơn (trung bình theo khách) của nhóm churn = nhóm còn lại. **H1:** khác nhau (hai phía).
- **Test:** Welch t-test (mặc định theo slide). Kiểm tra hình dạng: skew = {h2['skew_a']:.2f} (churn), {h2['skew_b']:.2f} (còn lại), gần đối xứng, n lớn nên Welch phù hợp. Kiểm tra chéo bằng Mann-Whitney: p = {h2['p_mw']:.3f}.
- **Kết quả:** churn {h2['ma']:,.1f} (n = {h2['na']:,}), còn lại {h2['mb']:,.1f} (n = {h2['nb']:,}).
- **Effect size:** chênh lệch **{h2['diff']:+.1f}** (churn − còn lại), 95% CI [{h2['ci'][0]:+.1f}; {h2['ci'][1]:+.1f}], Cohen's d = {h2['d']:.3f}, t = {h2['t']:.2f}, **p = {h2['p']:.3f}**.
- **Kết luận:** {verdict(h2['p'])}. Chênh lệch ước tính chỉ ~{abs(h2['diff'])/h2['mb']*100:.2f}% so với mức chi tiêu trung bình và khoảng tin cậy chứa 0, nên không có bằng chứng rằng chi tiêu liên quan tới việc rời bỏ.

## H3 — Giá trị đơn hàng theo danh mục
- **H0:** trung bình Total Purchase Amount bằng nhau ở 4 danh mục. **H1:** ít nhất một danh mục khác.
- **Test:** one-way ANOVA (4 nhóm số). Kiểm chéo bằng Kruskal-Wallis: p = {h3['p_kw']:.3f}. Tukey HSD làm hậu kiểm: p nhỏ nhất giữa các cặp = {h3['tukey_min_p']:.3f}.
- **Kết quả:** trung bình theo danh mục: {", ".join(f"{c} {v:,.1f}" for c, v in h3['means'].items())} (n = {h3['n']:,} dòng).
- **Effect size:** khoảng cách lớn nhất ({h3['hi']} − {h3['lo']}) = **{h3['gap']:+.1f}**, 95% CI cluster-bootstrap theo khách [{h3['ci_gap'][0]:+.1f}; {h3['ci_gap'][1]:+.1f}]; η² = {h3['eta2']:.5f}. F = {h3['F']:.2f}, **p = {h3['p']:.3f}**.
- **Kết luận:** {verdict(h3['p'])} ở α = 0.05 (ANOVA p = {h3['p']:.3f}, Kruskal-Wallis p = {h3['p_kw']:.3f}), tức bằng chứng ở mức biên. Hậu kiểm Tukey có một cặp p = {h3['tukey_min_p']:.3f}, nhưng vì omnibus không đạt ngưỡng nên chỉ coi là gợi ý thăm dò. Điều quan trọng hơn p: độ lớn rất nhỏ, khoảng cách lớn nhất chỉ {h3['gap']/grand*100:.2f}% mức trung bình và η² ≈ {h3['eta2']*100:.3f}% phương sai, nên dù có đạt ý nghĩa thống kê cũng không có ý nghĩa kinh doanh.
- **Lưu ý 1:** ANOVA ở mức dòng giả định các dòng độc lập, trong khi một khách có nhiều dòng, nên p có thể hơi lạc quan. CI cluster-bootstrap theo khách ở trên đã tính đến điều này.
- **Lưu ý 2:** cặp ({h3['hi']}, {h3['lo']}) được chọn vì nó là cặp chênh nhiều nhất sau khi nhìn dữ liệu, nên CI [{h3['ci_gap'][0]:+.1f}; {h3['ci_gap'][1]:+.1f}] của khoảng cách này bị lệch lạc quan (winner's curse). Không nên đọc nó như bằng chứng riêng của cặp đó.

---

## Nhìn chung
- Cả ba giả thuyết đều **không bác bỏ được H0** ở α = 0.05 (H3 ở mức biên). Với 3 test, ngưỡng Bonferroni là 0.05/3 ≈ 0.0167 và kết luận không đổi.
- H1 và H2 có CI hẹp quanh 0 trên mẫu lớn (hàng chục nghìn khách), nên nếu có hiệu ứng thì rất nhỏ. H3 có hiệu ứng ước tính rất nhỏ ở mọi cách đọc.
- Nhất quán với Week 10: R² của mô hình hồi quy chỉ ~0.003, tức các biến hành vi giải thích rất ít biến thiên của giá trị đơn hàng.
- Giới hạn: Returns thiếu 19% và đã được fill 0 trong file đã clean, nên không đưa vào giả thuyết ở đây. Muốn test Returns thì chỉ dùng các dòng `Returns_was_missing == 0` (cột này được tạo ở bước 3.2 của notebook và không có trong file CSV; mình chưa đọc được file parquet để xác nhận nó có ở đó).
- Kết luận chỉ áp dụng cho mẫu của dataset này, không suy ra toàn bộ thị trường.
"""
REPORT.write_text(md, encoding="utf-8")
print(md)
