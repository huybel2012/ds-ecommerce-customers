"""Phase 5.1: train-only study of signal, temporal features, negative controls.

Exploratory investigation on a static synthetic Churn label, NOT future churn
forecasting. No overwrite of Phase 5 model, threshold, reports or figures.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.feature_selection import mutual_info_classif
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.classification_phase5 import (
    CUTOFF, FORBIDDEN_PREDICTORS, customer_holdout, grouped_cv, make_customer_features,
)
from src.eda_metrics import AMOUNT, validate_eda_data

SEED = 42  # Preserve the Phase 5 customer holdout definition
NEW_FEATURES = (
    'mean_gap_days', 'std_gap_days', 'transactions_last_90d',
    'transactions_last_180d', 'transactions_previous_90d',
    'frequency_delta_90d', 'category_switch_rate',
    'amount_trend_recent_vs_previous_180d', 'has_both_180d_windows',
)
# Pre-registered TRAIN-CV screening gate. It is NOT a deployment criterion.
GATE = {
    'min_mean_roc_auc': 0.55,
    'min_mean_ap_gain_vs_prevalence': 0.03,
    'min_mean_top_decile_lift': 1.20,
    'min_gain_vs_permuted_roc_auc': 0.03,
    'min_roc_auc_every_repeat': 0.52,
}


def extended_customer_features(df: pd.DataFrame, cutoff: pd.Timestamp = CUTOFF):
    """Return Phase 5 features plus solely pre-cutoff temporal behavior features."""
    base, labels, subgroup, audit = make_customer_features(df, cutoff=cutoff)
    cutoff = pd.Timestamp(cutoff)
    history = df.loc[pd.to_datetime(df['Purchase Date']) < cutoff, [
        'Customer ID', 'Purchase Date', 'Product Category', AMOUNT,
    ]].copy()
    history['Purchase Date'] = pd.to_datetime(history['Purchase Date'], errors='raise')
    history = history.sort_values(['Customer ID', 'Purchase Date', 'Product Category'], kind='mergesort')
    g = history.groupby('Customer ID', sort=True)
    history['gap_days'] = (history['Purchase Date'] - g['Purchase Date'].shift()).dt.total_seconds() / 86400.0
    history['category_switched'] = (
        history['Product Category'].ne(g['Product Category'].shift()) & history['gap_days'].notna()
    ).astype(float)
    gap = g['gap_days'].agg(mean_gap_days='mean', std_gap_days=lambda s: s.std(ddof=0)).fillna(0.0)
    transitions = (g['category_switched'].sum() / (g.size()-1).clip(lower=1)).rename('category_switch_rate')
    age_days = (cutoff - history['Purchase Date']).dt.total_seconds() / 86400.0
    history['last90'] = age_days.le(90).astype(int)
    history['last180'] = age_days.le(180).astype(int)
    history['previous90'] = age_days.gt(90).mul(age_days.le(180)).astype(int)
    history['previous180'] = age_days.gt(180).mul(age_days.le(360)).astype(int)
    history['recent_amount'] = history[AMOUNT].where(age_days.le(180))
    history['previous_amount'] = history[AMOUNT].where(age_days.gt(180) & age_days.le(360))
    periods = g.agg(
        transactions_last_90d=('last90','sum'),
        transactions_last_180d=('last180','sum'),
        transactions_previous_90d=('previous90','sum'),
        recent_amount_mean=('recent_amount','mean'),
        previous_amount_mean=('previous_amount','mean'),
    )
    periods['frequency_delta_90d'] = (
        (periods['transactions_last_90d'] - periods['transactions_previous_90d']) /
        (periods['transactions_previous_90d'] + 1.0)
    )
    periods['has_both_180d_windows'] = (
        periods['recent_amount_mean'].notna() & periods['previous_amount_mean'].notna()
    ).astype(int)
    # Trend of simulated recorded values, not real revenue; absent windows -> neutral zero + coverage flag.
    periods['amount_trend_recent_vs_previous_180d'] = (
        (periods['recent_amount_mean'] - periods['previous_amount_mean']) /
        periods['previous_amount_mean'].clip(lower=1)
    ).fillna(0.0)
    extra = periods.drop(columns=['recent_amount_mean','previous_amount_mean']).join(gap).join(transitions)
    extra = extra.loc[base.index, list(NEW_FEATURES)].astype(float)
    full = base.join(extra)
    if full.index.has_duplicates or not full.index.equals(labels.index):
        raise ValueError('Customer index contract violated')
    if set(full.columns) & FORBIDDEN_PREDICTORS:
        raise AssertionError('Leaked identifiers/label/demographics')
    if not np.isfinite(full.to_numpy()).all():
        raise ValueError('Non-finite temporal features')
    if not (history['Purchase Date'] < cutoff).all():
        raise AssertionError('Future transaction leakage')
    return base, full, labels, subgroup, {**audit, 'new_features': list(NEW_FEATURES)}


def candidate_models(seed: int = SEED, max_iter: int = 45):
    return {
        'Logistic Regression': make_pipeline(SimpleImputer(strategy='median'),
            StandardScaler(), LogisticRegression(max_iter=1000, random_state=seed)),
        'HistGradientBoosting': make_pipeline(SimpleImputer(strategy='median'),
            HistGradientBoostingClassifier(max_iter=max_iter, max_leaf_nodes=15,
                min_samples_leaf=45, learning_rate=0.06, l2_regularization=1.0,
                early_stopping=True, random_state=seed)),
    }


def top_decile_lift(y, prob, fraction=0.1):
    y = np.asarray(y, dtype=int)
    prob = np.asarray(prob, dtype=float)
    if len(y) != len(prob) or not 0 < fraction < 1:
        raise ValueError('Invalid ranking input')
    rate = np.mean(y)
    if not 0 < rate < 1:
        raise ValueError('Both label classes required')
    k = max(1, int(np.ceil(fraction * len(y))))
    return float(np.mean(y[np.argsort(-prob, kind='mergesort')[:k]]) / rate)


def audit_univariate_signal(X_train: pd.DataFrame, y_train: pd.Series, seed=SEED):
    """AUC orientation-insensitive and kNN mutual information, TRAIN ONLY."""
    y = np.asarray(y_train, dtype=int)
    x = X_train.astype(float)
    if x.isna().any().any() or not np.isfinite(x.to_numpy()).all():
        raise ValueError('Invalid features')
    # kNN MI is expensive on 40k rows and estimates become unstable for tied counts.
    # Use a seeded TRAIN-ONLY subsample solely for this exploratory diagnostic.
    mi_n = min(5000, len(y))
    mi_idx = np.random.default_rng(seed).choice(len(y), size=mi_n, replace=False)
    mi = mutual_info_classif(x.to_numpy()[mi_idx], y[mi_idx], discrete_features=False,
                            n_neighbors=3, random_state=seed)
    items = []
    for j, col in enumerate(x.columns):
        if x[col].nunique() < 2:
            auc = 0.5
        else:
            auc = float(roc_auc_score(y, x[col]))
        items.append({'feature':col, 'auc_absolute_distance_from_chance':abs(auc-.5),
                      'univariate_auc':auc, 'mutual_information':float(mi[j])})
    return sorted(items, key=lambda d:d['auc_absolute_distance_from_chance'], reverse=True)


def cv_metrics(X: pd.DataFrame, y: pd.Series, estimator, *, seeds=(42,137), folds=3):
    """Repeated group CV entirely inside the Phase 5 training customers."""
    if not X.index.is_unique or not X.index.equals(y.index):
        raise ValueError('One row per customer and identical label index required')
    rows=[]
    for seed in seeds:
        for tr, va in grouped_cv(X, y, n_splits=folds, seed=int(seed)):
            model = clone(estimator)
            model.fit(X.iloc[tr], y.iloc[tr])
            prob = model.predict_proba(X.iloc[va])[:, 1]
            target = y.iloc[va].to_numpy()
            rows.append({'repeat_seed':int(seed), 'fold_customers':len(va),
                         'ROC_AUC':float(roc_auc_score(target,prob)),
                         'PR_AUC':float(average_precision_score(target,prob)),
                         'prevalence':float(target.mean()),
                         'lift_at_top_10pct':top_decile_lift(target,prob)})
    avg={name:float(np.mean([r[name] for r in rows]))
         for name in ('ROC_AUC','PR_AUC','prevalence','lift_at_top_10pct')}
    avg['AP_gain_vs_prevalence'] = avg['PR_AUC'] - avg['prevalence']
    repeat_means={str(seed):float(np.mean([r['ROC_AUC'] for r in rows if r['repeat_seed']==seed]))
                  for seed in seeds}
    return {'mean':avg,'repeat_roc_auc':repeat_means,'folds':rows}


def promotion_screen(cv: dict, negative_controls: list[dict]) -> tuple[bool,dict]:
    """Exploratory TRAIN-only evidence gate. It never overwrites Phase 5."""
    actual=cv['mean']; perm_auc=float(np.mean([p['mean']['ROC_AUC'] for p in negative_controls]))
    checks={
        'ROC_AUC_at_least_0_55':actual['ROC_AUC'] >= GATE['min_mean_roc_auc'],
        'AP_gain_at_least_0_03':actual['AP_gain_vs_prevalence'] >= GATE['min_mean_ap_gain_vs_prevalence'],
        'top_decile_lift_at_least_1_20':actual['lift_at_top_10pct'] >= GATE['min_mean_top_decile_lift'],
        'AUC_gain_vs_permutations_at_least_0_03':actual['ROC_AUC']-perm_auc >= GATE['min_gain_vs_permuted_roc_auc'],
        'all_repeat_AUC_at_least_0_52':all(v>=GATE['min_roc_auc_every_repeat']
                                           for v in cv['repeat_roc_auc'].values()),
    }
    return all(checks.values()), {'checks':checks, 'average_permuted_roc_auc':perm_auc,
                                   'auc_gain_vs_permutations':actual['ROC_AUC']-perm_auc}


def run_improvement(df: pd.DataFrame, *, folds=3, repeats=2, permutations=2,
                    max_iter=45, seed=SEED):
    """Analyze without consuming holdout labels or altering Phase 5 artifacts."""
    if folds < 2 or repeats < 1 or permutations < 1 or max_iter < 5:
        raise ValueError('Insufficient CV/permutation parameters')
    print('Phase 5.1: validating source and building temporal features...',flush=True)
    validated = validate_eda_data(df)
    original, expanded, y, _, audit = extended_customer_features(validated)
    print(f'Phase 5.1: {len(original):,} eligible customers, {len(expanded.columns)} features',flush=True)
    tr, te = customer_holdout(original, y, seed=SEED)  # fixed Phase 5 grouping
    if set(original.index[tr]) & set(original.index[te]):
        raise AssertionError('Train/holdout leakage')
    ytr=y.iloc[tr]
    train={'Phase 5 (19 features)':original.iloc[tr],
           'Phase 5.1 (28 features)':expanded.iloc[tr]}
    seeds=tuple(seed + i * 95 for i in range(repeats))
    results={}
    for group_name, X in train.items():
        for model_name, est in candidate_models(seed=seed,max_iter=max_iter).items():
            name=f'{group_name} | {model_name}'
            print(f'Phase 5.1: repeated grouped CV: {name}',flush=True)
            results[name]=cv_metrics(X,ytr,est,seeds=seeds,folds=folds)
    # Primary model selection on train ROC-AUC. PR-AUC breaks ties, no test labels.
    selected=max(results,key=lambda k:(results[k]['mean']['ROC_AUC'],
                                        results[k]['mean']['PR_AUC']))
    chosen_group, chosen_model=selected.split(' | ',1)
    chosen_X=train[chosen_group]
    chosen_est=candidate_models(seed=seed,max_iter=max_iter)[chosen_model]
    shuffled=[]
    for index in range(permutations):
        print(f'Phase 5.1: shuffled-label negative control {index+1}/{permutations}',flush=True)
        rng=np.random.default_rng(seed+10001+index)
        null_y=pd.Series(rng.permutation(ytr.to_numpy()),index=ytr.index,name=ytr.name)
        shuffled.append(cv_metrics(chosen_X,null_y,chosen_est,
                                   seeds=(seed+211+index,),folds=folds))
    passed, screen = promotion_screen(results[selected], shuffled)
    print('Phase 5.1: train-only signal audit...',flush=True)
    signal=audit_univariate_signal(expanded.iloc[tr],ytr,seed=seed)
    return {
        'phase':'5.1', 'protocol':'EXPLORATORY; train-only; previous Phase 5 holdout NOT evaluated',
        'source_rows':int(len(validated)), 'eligible_customers':len(original),
        'train_customers':len(tr),'held_out_customers_not_scored':len(te),
        'positive_rate_training':float(ytr.mean()),
        'cutoff_exclusive':audit['cutoff_exclusive'],
        'n_features_original':original.shape[1], 'n_features_expanded':expanded.shape[1],
        'new_features':list(NEW_FEATURES),'feature_audit':audit,
        'tested_models':results,'train_signal_ranking':signal,
        'selected_exploratory_cv':selected,
        'negative_control':{'method':'independent random permutations of training labels',
                            'runs':shuffled},
        'screening_thresholds':GATE,'screening':screen,'passed_exploratory_screen':passed,
        'model_replacement_approved':False,
        'recommendation':('Promising TRAIN-CV signal; requires an independent fresh confirmatory dataset; '
                          'do NOT replace Phase 5 automatically.' if passed else
                          'No robust train-only evidence of improved churn discrimination; retain Phase 5 report.'),
        'limitations':[
            'Static synthetic Churn has no event/effective timestamp; never claim future prediction.',
            'Previous Phase 5 holdout was already inspected; it is not a fresh confirmatory test.',
            'Screening four model/feature candidates produces selection optimism even within CV.',
            'Time-derived features may be sparse because transaction history per customer is short.',
            'Recorded amount units and interpretation are unresolved; not verified revenue.',
            'Negative-control permutations assess spurious fit but cannot prove label-generation mechanism.',
            'No Returns, customer names, identifiers, age, gender, or post-cutoff data are predictors.',
        ],
    }


def report_markdown(result: dict) -> str:
    f=result
    lines=[
        '# Phase 5.1 — Classification Improvement Study', '',
        '> **Nghiên cứu thăm dò trên TRAIN ONLY.** Không ghi đè Phase 5; chưa đánh giá lại holdout đã được xem.',
        '> Nhãn Churn synthetic không có event timestamp. Không dự đoán rời bỏ tương lai.', '',
        '## Phạm vi và giao thức',
        f"- {f['source_rows']:,} transactions; {f['eligible_customers']:,} customers eligible; "
        f"train={f['train_customers']:,}; original holdout not scored={f['held_out_customers_not_scored']:,}.",
        f"- Cutoff exclusive: `{f['cutoff_exclusive']}`; giữ nguyên split theo khách của Phase 5.",
        f"- {f['n_features_original']} original features vs {f['n_features_expanded']} after 9 temporal features.",
        '- So sánh Logistic Regression và HistGradientBoosting trên 2 feature sets bằng repeated, group-disjoint CV.',
        '- ROC-AUC, average precision (PR-AUC), top-decile lift. Không tối ưu F1 threshold hay dùng test labels để chọn mô hình.',
        '- Negative control: xáo trộn nhãn trong tập train, duy trì phân bố lớp.', '',
        '## Kết quả repeated grouped CV (chỉ train)', '',
        '| Feature set / model | ROC-AUC | PR-AUC (AP) | AP - prevalence | Lift@10% |',
        '|---|---:|---:|---:|---:|',
    ]
    for name,r in f['tested_models'].items():
        m=r['mean']
        lines.append(f"| {name} | {m['ROC_AUC']:.4f} | {m['PR_AUC']:.4f} | "
                     f"{m['AP_gain_vs_prevalence']:+.4f} | {m['lift_at_top_10pct']:.3f} |")
    winner=f['selected_exploratory_cv']; gate=f['screening']
    lines += ['',f'**Ứng viên có ROC-AUC CV lớn nhất:** `{winner}` (chọn trên train, có selection bias).',
              f"**Negative control mean ROC-AUC:** {gate['average_permuted_roc_auc']:.4f}.",
              f"**Train-screen:** {'PASS (still requires fresh confirmation)' if f['passed_exploratory_screen'] else 'FAIL'}.",
              f"**Quyết định:** {f['recommendation']}", '', '## Pre-registered screening criteria', '']
    for k,v in gate['checks'].items():
        lines.append(f"- {'PASS' if v else 'FAIL'}: `{k}`")
    lines += ['', '## Tín hiệu univariate mạnh nhất trên train (thăm dò)',
              '| Feature | |AUC-0.5| | Mutual information |', '|---|---:|---:|']
    for x in f['train_signal_ranking'][:10]:
        lines.append(f"| `{x['feature']}` | {x['auc_absolute_distance_from_chance']:.4f} | {x['mutual_information']:.4f} |")
    lines += ['', '## Giới hạn và kết luận', '']
    lines.extend('- '+lim for lim in f['limitations'])
    lines += ['', '**Phase 5 original remains authoritative unless a new, independently validated evaluation justifies a change.**', '']
    return '\n'.join(lines)


def save_figures(result: dict, fig_dir: Path):
    fig_dir=Path(fig_dir); fig_dir.mkdir(parents=True,exist_ok=True)
    top=result['train_signal_ranking'][:12][::-1]
    fig, ax=plt.subplots(figsize=(8,5),dpi=135)
    ax.barh([d['feature'] for d in top], [d['auc_absolute_distance_from_chance'] for d in top], color='#2563eb')
    ax.set(xlabel='|Training univariate ROC-AUC - 0.5| (exploratory)',
           title='Feature signal audit · TRAIN customers only')
    ax.set_xlim(left=0);fig.tight_layout()
    p1=fig_dir/'13_phase51_univariate_signal.png';fig.savefig(p1,bbox_inches='tight');plt.close(fig)
    names=list(result['tested_models'])
    aucs=[result['tested_models'][n]['mean']['ROC_AUC'] for n in names]
    aps=[result['tested_models'][n]['mean']['PR_AUC'] for n in names]
    baseline=result['positive_rate_training']
    fig,(a,b)=plt.subplots(1,2,figsize=(12,5.2),dpi=135)
    pos=np.arange(len(names)); labels=[n.replace('Phase 5 (19 features)','19f').replace('Phase 5.1 (28 features)','28f') for n in names]
    a.barh(pos,aucs,color='#2563eb');a.axvline(.5,linestyle='--',color='#e8791b')
    b.barh(pos,aps,color='#2563eb');b.axvline(baseline,linestyle='--',color='#e8791b')
    for ax, vals, ref, cap in [(a,aucs,.5,'ROC-AUC'),(b,aps,baseline,'Average precision')]:
        ax.set_yticks(pos,labels,fontsize=7);ax.invert_yaxis()
        ax.set_xlim(0,max(.65,max(vals)*1.13));ax.set_xlabel(cap)
        ax.text(ref,.1,f'ref={ref:.3f}',fontsize=7,color='#c2410c')
    fig.suptitle('Repeated customer-disjoint CV · TRAIN ONLY, not future churn prediction',fontsize=10)
    fig.tight_layout()
    p2=fig_dir/'14_phase51_repeated_cv.png';fig.savefig(p2,bbox_inches='tight');plt.close(fig)
    return [p1,p2]


def write_outputs(result: dict, root: Path) -> list[Path]:
    root=Path(root)
    report=root/'reports/phase5/phase51_improvement_study.md'
    metrics=root/'reports/phase5/phase51_metrics.json'
    report.parent.mkdir(parents=True,exist_ok=True)
    figs=save_figures(result,root/'reports/figures')
    report.write_text(report_markdown(result),encoding='utf-8')
    metrics.write_text(json.dumps(result,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
    return [report,metrics,*figs]
