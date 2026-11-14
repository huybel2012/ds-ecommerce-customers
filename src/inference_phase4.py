"""Phase 4 / W9: customer-aware, synthetic-data statistical inference.

Three exploratory (not preregistered) hypotheses. Currency is not established.
All results use the single Phase 2 cleaned source; missing Returns are untouched.
"""
from __future__ import annotations

import json
from itertools import combinations
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from src.eda_metrics import AMOUNT, load_cleaned, validate_eda_data
from src.utils import REPO_ROOT

ALPHA = 0.05
RESULT_PATH = REPO_ROOT / "reports/phase4/hypothesis_results.json"
REPORT_PATH = REPO_ROOT / "reports/hypothesis_tests.md"
FIGURE_PATH = REPO_ROOT / "reports/figures/09_hypothesis_tests.png"


def holm(pvalues: list[float]) -> list[float]:
    """Holm step-down family-wise adjustment, monotone in sorted order."""
    n = len(pvalues)
    order = np.argsort(pvalues)
    adjusted = np.empty(n, dtype=float)
    running = 0.0
    for rank, ix in enumerate(order):
        running = max(running, (n-rank) * float(pvalues[ix]))
        adjusted[ix] = min(1.0, running)
    return adjusted.tolist()


def analyze(df: pd.DataFrame) -> dict:
    df = validate_eda_data(df)
    consistency = df.groupby("Customer ID")["Gender"].nunique(dropna=False)
    if not consistency.eq(1).all():
        raise ValueError("Gender inconsistent within Customer ID")
    customer = df.groupby("Customer ID", sort=False).agg(
        Gender=("Gender", "first"), Churn=("Churn", "first"),
        mean_amount=(AMOUNT, "mean"), transactions=(AMOUNT, "size"),
    )
    if set(customer.Gender.unique()) != {"Male", "Female"}:
        raise ValueError("H1 requires exactly Female and Male observations")

    # H1: two-sided two independent proportions, independent customers.
    male, female = customer[customer.Gender.eq("Male")], customer[customer.Gender.eq("Female")]
    p_m, p_f = male.Churn.mean(), female.Churn.mean()
    n_m, n_f = len(male), len(female)
    pooled = (male.Churn.sum() + female.Churn.sum()) / (n_m + n_f)
    pooled_se = np.sqrt(pooled * (1-pooled) * (1/n_m + 1/n_f))
    z = (p_m-p_f)/pooled_se
    ci_se = np.sqrt(p_m*(1-p_m)/n_m + p_f*(1-p_f)/n_f)
    ci = [(p_m-p_f)-stats.norm.ppf(.975)*ci_se,(p_m-p_f)+stats.norm.ppf(.975)*ci_se]
    h1 = {"unit":"customer", "n_male":n_m, "n_female":n_f, "male_rate":float(p_m),
          "female_rate":float(p_f), "difference_pp":float(100*(p_m-p_f)),
          "ci95_pp":[float(100*x) for x in ci], "cohen_h":float(2*np.arcsin(np.sqrt(p_m))-2*np.arcsin(np.sqrt(p_f))),
          "z":float(z), "p":float(2*stats.norm.sf(abs(z)))}

    # H2: Welch test on customer mean amounts. Do not count repeated purchases as independent people.
    a = customer.loc[customer.Churn.eq(1), "mean_amount"].to_numpy(float)
    b = customer.loc[customer.Churn.eq(0), "mean_amount"].to_numpy(float)
    if min(len(a),len(b)) < 2:
        raise ValueError("H2 needs at least two customers in each group")
    res = stats.ttest_ind(a,b,equal_var=False)
    diff = a.mean()-b.mean()
    va, vb = a.var(ddof=1), b.var(ddof=1)
    se = np.sqrt(va/len(a)+vb/len(b))
    dof = (va/len(a)+vb/len(b))**2 / ((va/len(a))**2/(len(a)-1)+(vb/len(b))**2/(len(b)-1))
    tcrit=stats.t.ppf(.975,dof)
    pooled_sd = np.sqrt(((len(a)-1)*va+(len(b)-1)*vb)/(len(a)+len(b)-2))
    h2 = {"unit":"customer", "n_churn":len(a), "n_retained":len(b),
          "mean_churn":float(a.mean()),"mean_retained":float(b.mean()),
          "difference":float(diff), "ci95":[float(diff-tcrit*se),float(diff+tcrit*se)],
          "cohen_d":float(diff/pooled_sd) if pooled_sd else None,
          "t":float(res.statistic), "p":float(res.pvalue), "df":float(dof)}

    # H3: category means with CLUSTER sandwich covariance at customer level.
    categories = sorted(df["Product Category"].unique())
    if len(categories)!=4:
        raise ValueError("H3 expects four product categories")
    codes = pd.Categorical(df["Product Category"],categories=categories).codes
    if np.any(codes<0): raise ValueError("Invalid category")
    y=df[AMOUNT].to_numpy(dtype=float)
    counts=np.bincount(codes,minlength=4).astype(float)
    sums=np.bincount(codes,weights=y,minlength=4)
    means=sums/counts
    cust_codes, uniques=pd.factorize(df["Customer ID"],sort=False)
    G=len(uniques)
    if G <= 4: raise ValueError("H3 needs at least five distinct customer clusters")
    scores=np.zeros((G,4),dtype=float)
    np.add.at(scores,(cust_codes,codes),y-means[codes])
    # CR1 correction, saturated group-means regression with K=4 coefficients.
    correction=(G/(G-1))*((len(df)-1)/(len(df)-4))
    cov=correction*(scores.T@scores)/np.outer(counts,counts)
    c=np.column_stack([-np.ones(3),np.eye(3)])
    delta=c@means
    v=c@cov@c.T
    if np.linalg.matrix_rank(v)<3:
        raise ValueError("H3 cluster covariance singular")
    wald=float(delta@np.linalg.solve(v,delta))
    p_robust=float(stats.chi2.sf(wald,3))
    overall=y.mean()
    eta2=float(sum(counts*(means-overall)**2)/np.sum((y-overall)**2))
    pairs=[]
    six=list(combinations(range(4),2))
    zcrit=stats.norm.ppf(1-ALPHA/(2*len(six)))
    raw_p=[]
    for i,j in six:
        d=float(means[i]-means[j]); se2=float(max(cov[i,i]+cov[j,j]-2*cov[i,j],0))
        sep=np.sqrt(se2)
        if sep==0: raise ValueError("H3 contrast has zero variance")
        p=float(2*stats.norm.sf(abs(d/sep)))
        raw_p.append(p)
        pairs.append({"pair":f"{categories[i]} vs {categories[j]}","difference":d,
                      "ci95_simultaneous_bonferroni":[float(d-zcrit*sep),float(d+zcrit*sep)],"raw_p":p})
    for pair, padj in zip(pairs,holm(raw_p)):
        pair["holm_p"] = padj
    h3={"unit":"transaction (SE clustered by customer)","categories":categories,
        "means":{k:float(v) for k,v in zip(categories,means)},
        "counts":{k:int(v) for k,v in zip(categories,counts)},
        "cluster_n":G, "cluster_robust_wald_chi2":wald,"df":3,"p":p_robust,
        "eta_squared_descriptive":eta2,"pairs_exploratory":pairs,
        "mean_se_clustered":{k:float(np.sqrt(cov[i,i])) for i,k in enumerate(categories)},
        "naive_anova_p_diagnostic_only":float(stats.f_oneway(*[y[codes==j] for j in range(4)]).pvalue)}
    adjusted = holm([h1["p"],h2["p"],h3["p"]])
    for result, p_adj in zip((h1,h2,h3),adjusted):
        result["holm_p_fwer_3_tests"]=p_adj
    return {"source_type":"synthetic", "currency":"unverified", "alpha":ALPHA,
            "status":"exploratory, hypotheses previously inspected", "rows":len(df),"customers":len(customer),
            "H1":h1,"H2":h2,"H3":h3}


def plot(result: dict, path: Path) -> None:
    h1,h2,h3=(result[k] for k in ("H1","H2","H3"))
    fig,axes=plt.subplots(1,3,figsize=(16,4.8),dpi=160)
    axes[0].bar(["Female","Male"],[100*h1["female_rate"],100*h1["male_rate"]], color=["#2563eb","#64748b"])
    axes[0].set_ylim(0,max(25,100*max(h1["female_rate"],h1["male_rate"])*1.22))
    axes[0].set_ylabel("Customer churn (%)")
    axes[0].set_title(f'H1  diff={h1["difference_pp"]:+.2f} pp\np={h1["p"]:.3g} (Holm {h1["holm_p_fwer_3_tests"]:.3g})')
    # Point/CI presentation avoids misleading nonzero-baseline bar plots.
    axes[1].errorbar([h2["difference"]],[0],xerr=[[h2["difference"]-h2["ci95"][0]],
                     [h2["ci95"][1]-h2["difference"]]],fmt="o",color="#2563eb",capsize=5)
    axes[1].axvline(0,c="gray",ls="--",lw=1)
    axes[1].set_yticks([0], ["Churn - retained"])
    axes[1].set_xlabel("Difference in mean recorded amount / customer")
    axes[1].set_title(f'H2  Welch p={h2["p"]:.3g}\n95% CI for difference')
    cs=list(h3["means"])
    xs=np.array([h3["means"][c] for c in cs]); ses=np.array([h3["mean_se_clustered"][c] for c in cs])
    axes[2].errorbar(xs,np.arange(len(cs)),xerr=1.96*ses,fmt="o",capsize=4,color="#2563eb")
    axes[2].set_yticks(np.arange(len(cs)),cs)
    axes[2].set_xlabel("Mean recorded amount / transaction")
    axes[2].set_title(f'H3  cluster-robust p={h3["p"]:.3g}\n95% pointwise CI on means')
    for ax in axes:
        ax.spines[["top","right"]].set_visible(False)
    fig.suptitle("Exploratory inference on synthetic data (currency unverified)",fontsize=13)
    fig.tight_layout()
    path.parent.mkdir(parents=True,exist_ok=True)
    fig.savefig(path,bbox_inches="tight")
    plt.close(fig)


def markdown(r:dict)->str:
    h1,h2,h3=(r[k] for k in ("H1","H2","H3"))
    def conclusion(x):return "Bác bỏ H0" if x["holm_p_fwer_3_tests"] < ALPHA else "Chưa bác bỏ H0"
    pair_table="\n".join(f'| {x["pair"]} | {x["difference"]:+.2f} | [{x["ci95_simultaneous_bonferroni"][0]:+.2f}, {x["ci95_simultaneous_bonferroni"][1]:+.2f}] | {x["holm_p"]:.4g} |' for x in h3["pairs_exploratory"])
    return f"""# Week 9: Statistical inference (Phase 4)

**Input:** `data/processed/ecommerce_customer_clean.csv` (Phase 2 validated); **{r['rows']:,} transactions / {r['customers']:,} customers**. Synthetic Kaggle data, monetary unit unverified; `Total Purchase Amount` cannot be reconciled with `Price × Quantity`. This is exploratory retrospective analysis: the existing hypotheses were inspected on this dataset, so do **not** present as preregistered confirmatory research. No causal or market-wide claims.

**Multiplicity:** Two-sided α=0.05; family of 3 primary hypotheses controlled by **Holm**. The H3 post-hoc family of six category pairs separately uses Holm-adjusted p-values and *Bonferroni simultaneous* confidence intervals. Avoid cherry-picking the widest observed gap.

## H1: Gender and customer churn
- **H0:** p(Male churn) = p(Female churn); **H1:** different. Unit: customer, deduplicated by `Customer ID`.
- Male: {h1['male_rate']:.3%} (n={h1['n_male']:,}), Female: {h1['female_rate']:.3%} (n={h1['n_female']:,}).
- Difference (male − female): **{h1['difference_pp']:+.3f} percentage points**, 95% unadjusted CI [{h1['ci95_pp'][0]:+.3f}, {h1['ci95_pp'][1]:+.3f}] pp; Cohen's h={h1['cohen_h']:.4f}.
- Two-proportion z={h1['z']:.3f}; p={h1['p']:.4g}; **Holm p={h1['holm_p_fwer_3_tests']:.4g}**. {conclusion(h1)} at α=.05. Statistical non-significance does not prove equality.

## H2: Per-customer average transaction amount by churn status
- **H0:** average of customer-level mean recorded amounts is equal between Churn=1 and 0; **H1:** different. Unit: customer, not independent transaction rows.
- Churn: {h2['mean_churn']:,.2f} (n={h2['n_churn']:,}), retained: {h2['mean_retained']:,.2f} (n={h2['n_retained']:,}).
- Difference: **{h2['difference']:+.2f}** (unverified units), Welch 95% CI [{h2['ci95'][0]:+.2f}, {h2['ci95'][1]:+.2f}], Cohen's d={h2['cohen_d']:.4f}.
- Welch t={h2['t']:.3f}, df={h2['df']:.0f}, p={h2['p']:.4g}; **Holm p={h2['holm_p_fwer_3_tests']:.4g}**. {conclusion(h2)} at α=.05. Association only.

## H3: Mean recorded amount across four categories
- **H0:** 4 category means are equal; **H1:** at least one differs.
- Unit: transaction; repeated transactions per customer are explicitly handled with **customer-clustered sandwich SE** and a chi-square Wald omnibus test (3 df, asymptotic approximation with {h3['cluster_n']:,} independent customer clusters).
- Category means: {', '.join(f'{c}: {v:,.2f} (n={h3["counts"][c]:,})' for c,v in h3['means'].items())}.
- Wald χ²={h3['cluster_robust_wald_chi2']:.3f}, p={h3['p']:.4g}; **Holm p={h3['holm_p_fwer_3_tests']:.4g}**. {conclusion(h3)} at α=.05.
- Descriptive η²={h3['eta_squared_descriptive']:.7f}. Ordinary row-level ANOVA p={h3['naive_anova_p_diagnostic_only']:.4g} is **diagnostic only**, because rows are not independent customers.

### Exploratory pairwise comparisons (6 pairs)

| Pair | Mean difference | 95% simultaneous CI (Bonferroni) | Holm p |
|---|---:|---|---:|
{pair_table}

**Notes:** The six comparisons were enumerated before selecting a largest observed gap. These adjusted intervals are valid for the pre-defined six-pair family under the large-cluster approximation, not proof of an economically meaningful effect. The category exposure is observational; no causal interpretation.

## Methodology and limits
- The dataset is synthetic; the statistical p-values describe this generated sample under working sampling assumptions, not the real retail population.
- Neither gender nor churn are assigned treatments; these comparisons cannot establish causal effects.
- `Returns` missingness (47,596 on canonical input) does not affect these tests, which do not use Returns.
- The transaction amount is a *recorded field with unresolved semantics* rather than confirmed revenue; do not label results as USD or VND.
- `reports/figures/09_hypothesis_tests.png` uses a zero-based churn bar chart and dot/CI plots instead of truncated bars for tiny differences.
"""


def run(path:Path|None=None, report_path:Path=REPORT_PATH, fig_path:Path=FIGURE_PATH, json_path:Path=RESULT_PATH)->dict:
    df=load_cleaned(path) if path is not None else load_cleaned()
    result=analyze(df)
    plot(result,fig_path)
    report_path.parent.mkdir(parents=True,exist_ok=True)
    json_path.parent.mkdir(parents=True,exist_ok=True)
    report_path.write_text(markdown(result),encoding="utf-8")
    json_path.write_text(json.dumps(result,indent=2,ensure_ascii=False)+"\n",encoding="utf-8")
    print(f"W9: H1 p={result['H1']['p']:.4g}, H2 p={result['H2']['p']:.4g}, H3 cluster p={result['H3']['p']:.4g}")
    print(f"Saved {report_path}, {fig_path}, {json_path}")
    return result


def main()->None:
    run()


if __name__=="__main__":
    main()
