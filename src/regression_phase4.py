"""Phase 4 / W10: reproducible grouped regression of a *recorded* synthetic amount.

Predictions do NOT represent actual revenue. Post-purchase Returns, Churn,
identifiers, and future information are excluded. OLS inference clusters at
Customer ID. All learned transforms fit only on training folds.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold, GroupShuffleSplit, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.eda_metrics import AMOUNT, load_cleaned, validate_eda_data
from src.utils import REPO_ROOT

NUM = ("Product Price", "Quantity", "Age")
CAT = ("Product Category", "Payment Method", "Gender")
REF = {"Product Category":"Clothing", "Payment Method":"Credit Card", "Gender":"Female"}
SEED=42
FIG_DIR=REPO_ROOT/'reports/figures'
REPORT=REPO_ROOT/'reports/regression_results.md'
METRICS=REPO_ROOT/'reports/phase4/regression_metrics.json'


def grouped_split(df: pd.DataFrame, test_size:float=.2, seed:int=SEED):
    groups=df['Customer ID']
    idx_train,idx_test=next(GroupShuffleSplit(n_splits=1,test_size=test_size,random_state=seed).split(df,groups=groups))
    if not set(groups.iloc[idx_train]).isdisjoint(set(groups.iloc[idx_test])):
        raise AssertionError('Customer ID leakage')
    return idx_train,idx_test


def design_matrix(df:pd.DataFrame)->pd.DataFrame:
    """Raw-unit OLS design with explicit reference groups for human explanation."""
    parts=[pd.Series(1.,index=df.index,name='const'),*[pd.to_numeric(df[c]).astype(float).rename(c) for c in NUM]]
    for c in CAT:
        levels=sorted(df[c].unique().tolist())
        if REF[c] not in levels: raise ValueError(f'Missing reference {REF[c]}')
        for level in levels:
            if level!=REF[c]:
                parts.append(df[c].eq(level).astype(float).rename(f'{c} = {level}'))
    return pd.concat(parts,axis=1).astype(float)


def cluster_ols(x:np.ndarray, y:np.ndarray, groups:np.ndarray)->dict:
    """OLS with cluster-sandwich CR1 covariance (customer as independent unit)."""
    n,k=x.shape
    group_codes,clusters=pd.factorize(groups,sort=False)
    G=len(clusters)
    if n<=k or G<=k: raise ValueError('Too few observations or customer clusters')
    gram=x.T@x
    if np.linalg.matrix_rank(gram)!=k:raise ValueError('Singular regression design')
    inv=np.linalg.inv(gram)
    beta=inv@(x.T@y)
    resid=y-x@beta
    scores=np.zeros((G,k),dtype=float)
    np.add.at(scores, group_codes, x*resid[:,None])
    cov=(G/(G-1))*((n-1)/(n-k))*(inv@(scores.T@scores)@inv)
    se=np.sqrt(np.maximum(np.diag(cov),0))
    tcrit=stats.t.ppf(.975,G-1)
    with np.errstate(divide='ignore', invalid='ignore'):
        t=np.divide(beta,se,out=np.full_like(beta,np.nan),where=se>0)
    p=2*stats.t.sf(np.abs(t),df=G-1)
    return {'coef':beta,'se':se,'ci_low':beta-tcrit*se,'ci_high':beta+tcrit*se,
            'p':p,'residuals':resid,'fitted':x@beta,'cov':cov,
            'n_groups':G,'n':n,'k':k,'x_inv':inv}


def mae_gain_cluster_bootstrap(y_true, y_base, y_ols, groups, n_boot=400, seed=SEED):
    """CI for baseline MAE - OLS MAE, bootstrap whole customers, not rows."""
    codes, unique=pd.factorize(groups,sort=False)
    G=len(unique)
    counts=np.bincount(codes,minlength=G)
    gains=np.bincount(codes,weights=np.abs(y_true-y_base)-np.abs(y_true-y_ols),minlength=G)
    rng=np.random.default_rng(seed)
    boot=[]
    for _ in range(n_boot):
        sampled=rng.integers(0,G,size=G)
        boot.append(float(gains[sampled].sum()/counts[sampled].sum()))
    return [float(x) for x in np.quantile(boot,[.025,.975])]


def model_metrics(y,p)->dict:
    return {'MAE':float(mean_absolute_error(y,p)),
            'RMSE':float(np.sqrt(mean_squared_error(y,p))),
            'R2':float(r2_score(y,p))}


def _prep():
    return ColumnTransformer([
        ('num',StandardScaler(),list(NUM)),
        ('cat',OneHotEncoder(handle_unknown='ignore',drop='first',sparse_output=False),list(CAT)),
    ])


def train_and_evaluate(df:pd.DataFrame, cv_splits:int=3)->tuple[dict,dict]:
    df=validate_eda_data(df)
    y=df[AMOUNT].to_numpy(float)
    groups=df['Customer ID'].to_numpy()
    train,test=grouped_split(df)
    X=df[list(NUM)+list(CAT)]
    Xtr,Xte=X.iloc[train],X.iloc[test]
    ytr,yte=y[train],y[test]
    gtr,gte=groups[train],groups[test]

    baseline=DummyRegressor(strategy='mean').fit(Xtr,ytr)
    ols=Pipeline([('prep',_prep()),('model',LinearRegression())]).fit(Xtr,ytr)
    if len(np.unique(gtr))<cv_splits:raise ValueError('Too few groups for CV')
    cv=GroupKFold(n_splits=cv_splits)
    ridge=GridSearchCV(Pipeline([('prep',_prep()),('model',Ridge())]),
                       {'model__alpha':[1.,100.,1000.]},cv=cv,scoring='neg_mean_absolute_error',n_jobs=1)
    lasso=GridSearchCV(Pipeline([('prep',_prep()),('model',Lasso(max_iter=5000))]),
                       {'model__alpha':[.1,1.,10.]},cv=cv,scoring='neg_mean_absolute_error',n_jobs=1)
    ridge.fit(Xtr,ytr,groups=gtr)
    lasso.fit(Xtr,ytr,groups=gtr)
    preds={'Baseline (mean)':baseline.predict(Xte),'OLS':ols.predict(Xte),
           'Ridge':ridge.predict(Xte),'Lasso':lasso.predict(Xte)}
    performance={name:model_metrics(yte,p) for name,p in preds.items()}
    gain=performance['Baseline (mean)']['MAE']-performance['OLS']['MAE']
    gain_ci=mae_gain_cluster_bootstrap(yte,preds['Baseline (mean)'],preds['OLS'],gte)

    raw_design=design_matrix(df.iloc[train])
    ols_inf=cluster_ols(raw_design.to_numpy(),ytr,gtr)
    beta_table=pd.DataFrame({'coefficient':ols_inf['coef'],'se_cluster':ols_inf['se'],
       'ci_low':ols_inf['ci_low'],'ci_high':ols_inf['ci_high'],'p_cluster':ols_inf['p']},index=raw_design.columns)
    residual=ols_inf['residuals']; n,k=ols_inf['n'],ols_inf['k']
    resid_variance=float(residual@residual/(n-k))
    leverage=np.einsum('ij,jk,ik->i',raw_design.to_numpy(),ols_inf['x_inv'],raw_design.to_numpy())
    cooks=(residual**2/(k*resid_variance))*(leverage/(1-leverage)**2)
    # VIF of non-intercept columns via inverse of correlation matrix.
    var_x=raw_design.drop(columns='const').to_numpy(float)
    corr=np.corrcoef(var_x,rowvar=False)
    vif=np.diag(np.linalg.inv(corr))
    train_r2=float(r2_score(ytr,ols.predict(Xtr)))
    result={'dataset':'Phase 2 validated cleaned CSV','source_type':'synthetic',
       'target':AMOUNT,'currency':'unverified','target_semantics':'unresolved vs Product Price x Quantity',
       'excluded_features':['Returns (missing and post-purchase)', 'Churn (post-outcome and customer attribute)',
                           'Customer ID (split only)','Purchase Date (not included for baseline comparability)'],
       'features':list(NUM)+list(CAT),'seed':SEED,
       'n_train_rows':len(train),'n_test_rows':len(test),
       'n_train_customers':int(len(np.unique(gtr))),'n_test_customers':int(len(np.unique(gte))),
       'no_customer_overlap':True,'models':performance,
       'ridge_best_alpha':float(ridge.best_params_['model__alpha']),
       'lasso_best_alpha':float(lasso.best_params_['model__alpha']),
       'grouped_cv_folds':cv_splits,'train_ols_R2':train_r2,
       'ols_mae_gain_over_baseline':gain,'ols_mae_gain_cluster_bootstrap_ci95':gain_ci,
       'coefficient_inference':'Customer cluster-robust CR1; t df = n_customers_train-1',
       'n_influential_cooks_4_over_n':int((cooks>4/n).sum()),
       'max_vif':float(np.max(vif)),
       'returns_missing_in_input':int(df['Returns'].isna().sum()),
       'amount_mismatches':int((df[AMOUNT] != df['Product Price']*df['Quantity']).sum())}
    # Return arrays to figure writer; never include model binaries in artifacts.
    detail={'coef':beta_table,'fitted':ols_inf['fitted'],'residual':residual,
      'cooks':cooks,'cooks_threshold':4/n, 'test_y':yte,'test_pred':preds['OLS']}
    return result,detail


def _md_table(df:pd.DataFrame,decimals=4)->str:
    header='| Metric | '+' | '.join(df.columns)+' |'
    sep='|---|'+'---|'*len(df.columns)
    rows=[]
    for idx,row in df.iterrows():
        rows.append('| '+str(idx)+' | '+' | '.join(f'{float(v):,.{decimals}f}' for v in row)+' |')
    return '\n'.join([header,sep]+rows)


def report_markdown(r:dict,coef:pd.DataFrame)->str:
    perf=_md_table(pd.DataFrame(r['models']).T[['MAE','RMSE','R2']],4)
    coefs=_md_table(coef,4)
    ci=r['ols_mae_gain_cluster_bootstrap_ci95']
    interpretations=[]
    for name,row in coef.iterrows():
        if name=='const':continue
        significant=bool(row.ci_low>0 or row.ci_high<0)
        interpretations.append(f'- `{name}`: {row.coefficient:+.3f} recorded units (95% customer-cluster CI [{row.ci_low:+.3f}; {row.ci_high:+.3f}]); '+
                               ('CI excludes zero' if significant else 'CI includes zero')+
                               '. Conditional association, **not causal**.')
    return f"""# Week 10: Regression (Phase 4)

**Input:** `data/processed/ecommerce_customer_clean.csv`, Phase 2 validated. **Target:** `Total Purchase Amount` in *unverified units*. Synthetic Kaggle dataset; do not call this measured USD/VND revenue. The target differs from Price × Quantity in **{r['amount_mismatches']:,} rows**; model results have limited substantive meaning.

**Design:** prediction at the transaction level. GroupShuffleSplit 80/20 by `Customer ID`, seed {r['seed']} (train {r['n_train_rows']:,} rows / {r['n_train_customers']:,} customers; test {r['n_test_rows']:,} rows / {r['n_test_customers']:,} customers). Customer overlap: **none**. Three-fold grouped CV inside train only; preprocessing is fit within each training fold. Numeric variables scaled, categorical one-hot encoded. No test-set tuning.

**Features:** {', '.join(r['features'])}. Excluded: {', '.join(r['excluded_features'])}. In particular, `Returns` is not used: it is 19% missing and could occur **after** the purchase being predicted. This avoids unsupported imputation and temporal leakage.

## Holdout test metrics

{perf}

Best grouped-CV alpha: Ridge {r['ridge_best_alpha']:g}, Lasso {r['lasso_best_alpha']:g}. Train OLS R²={r['train_ols_R2']:.5f}; test OLS R²={r['models']['OLS']['R2']:.5f}.

**Baseline comparison:** OLS improves MAE over mean baseline by **{r['ols_mae_gain_over_baseline']:+.3f}** units; 95% **customer cluster bootstrap** CI [{ci[0]:+.3f}, {ci[1]:+.3f}]. If CI includes zero, do not claim predictive improvement beyond sampling uncertainty. Statistical significance of individual coefficients must not be confused with useful prediction.

## OLS inference (TRAIN, raw units)

{coefs}

Standard errors: **CR1 cluster sandwich by Customer ID** (not ordinary iid OLS or HC3). This accounts for within-customer repeated transactions under the working assumption that different customers are independent. Data are observational and synthetic, not a causal experiment.

### Coefficient interpretations

{chr(10).join(interpretations)}

## Diagnostics and limitations

- Cook's D > 4/n: {r['n_influential_cooks_4_over_n']:,} observations (flag for review, not automatic deletion); max VIF {r['max_vif']:.2f}.
- Residual, QQ, Cook's influence diagnostics in `reports/figures/06_residual_diagnostics.png`; 95% customer-cluster coefficient CIs in `07_coefficients_ci.png`; holdout actual vs prediction in `08_actual_vs_predicted.png`.
- **Durbin–Watson omitted:** sorting unrelated customers' transactions by date does not create a meaningful single time series, so the old temporal-DW interpretation was invalid.
- Unverified target semantics and synthetic generation limit external validity. A low R² is itself an empirical finding: do not optimize on test data or promise an unrealistically high score.
- Cluster-robust inference does not repair confounding, selection effects, or an incorrectly defined prediction target.
"""


def plot(detail:dict,folder:Path)->None:
    folder.mkdir(parents=True,exist_ok=True)
    rng=np.random.default_rng(SEED)
    resid=detail['residual'];n=len(resid);sample=rng.choice(n,size=min(n,7000),replace=False)
    fig,axes=plt.subplots(1,3,figsize=(15.5,4.5),dpi=140)
    axes[0].scatter(detail['fitted'][sample],resid[sample],s=6,alpha=.3)
    axes[0].axhline(0,ls='--',c='gray');axes[0].set(xlabel='Fitted recorded amount',ylabel='Residual',title='Residual vs fitted')
    (theoretical,sample_ordered),(slope,intercept,_) = stats.probplot(resid[sample],dist='norm')
    axes[1].scatter(theoretical,sample_ordered,s=5,alpha=.3)
    axes[1].plot(theoretical,slope*theoretical+intercept,c='gray',lw=1)
    axes[1].set(xlabel='Theoretical quantiles',ylabel='Residual quantiles',title='QQ plot (sample)')
    axes[2].scatter(np.arange(len(sample)),detail['cooks'][sample],s=5,alpha=.5)
    axes[2].axhline(detail['cooks_threshold'],c='gray',ls='--')
    axes[2].set(xlabel='Sample index',ylabel="Cook's D",title='Influence (sample)')
    fig.suptitle('W10 diagnostic plots (recorded amount, unknown currency)')
    fig.tight_layout();fig.savefig(folder/'06_residual_diagnostics.png',bbox_inches='tight');plt.close(fig)

    c=detail['coef'].drop(index='const').sort_values('coefficient')
    fig,ax=plt.subplots(figsize=(9,6),dpi=140)
    vals=c.coefficient.to_numpy()
    errors=np.vstack([vals-c.ci_low.to_numpy(),c.ci_high.to_numpy()-vals])
    ax.errorbar(vals,np.arange(len(c)),xerr=errors,fmt='o',capsize=4,color='#2563eb')
    ax.axvline(0,ls='--',c='gray');ax.set_yticks(np.arange(len(c)),c.index)
    ax.set(xlabel='OLS coefficient (original feature units)',title='95% customer-cluster robust confidence intervals')
    fig.tight_layout();fig.savefig(folder/'07_coefficients_ci.png',bbox_inches='tight');plt.close(fig)

    yt,yp=detail['test_y'],detail['test_pred']
    sample=rng.choice(len(yt),size=min(6000,len(yt)),replace=False)
    fig,ax=plt.subplots(figsize=(6,5),dpi=140)
    ax.scatter(yt[sample],yp[sample],s=7,alpha=.25)
    valmin=min(float(yt.min()),float(yp.min()));valmax=max(float(yt.max()),float(yp.max()))
    ax.plot([valmin,valmax],[valmin,valmax],ls='--',c='gray',label='Ideal')
    ax.set(xlabel='Actual recorded amount',ylabel='OLS predicted amount',title='Holdout test actual vs predicted')
    ax.legend();fig.tight_layout();fig.savefig(folder/'08_actual_vs_predicted.png',bbox_inches='tight');plt.close(fig)


def run(path:Path|None=None,report_path:Path=REPORT,fig_dir:Path=FIG_DIR,json_path:Path=METRICS,cv_splits:int=3)->dict:
    df=load_cleaned(path) if path is not None else load_cleaned()
    result,details=train_and_evaluate(df,cv_splits=cv_splits)
    plot(details,fig_dir)
    report_path.parent.mkdir(parents=True,exist_ok=True)
    json_path.parent.mkdir(parents=True,exist_ok=True)
    report_path.write_text(report_markdown(result,details['coef']),encoding='utf-8')
    json_path.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print('W10 holdout:',json.dumps(result['models'],indent=2))
    print(f"MAE gain vs baseline={result['ols_mae_gain_over_baseline']:+.3f}; CI={result['ols_mae_gain_cluster_bootstrap_ci95']}")
    print(f'Saved {report_path}, {json_path}, and 06-08 PNGs in {fig_dir}')
    return result


def main()->None:
    run()

if __name__=='__main__':
    main()
