"""Phase 4 deterministic fixtures; never require the Kaggle raw CSV."""
from __future__ import annotations

import subprocess
import tempfile
import json
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from src.ab_design import required_per_arm
from src.eda_metrics import validate_eda_data
from src.inference_phase4 import analyze, holm, run as run_inference
from src.regression_phase4 import grouped_split, cluster_ols, design_matrix, train_and_evaluate, run as run_regression
from src.utils import REPO_ROOT


def fixture(n_customers=72, repeats=4, seed=18):
    rng=np.random.default_rng(seed)
    dates=pd.date_range('2021-01-01',periods=n_customers*repeats,freq='h')
    rows=[]
    for customer in range(n_customers):
        age=int(rng.integers(20,65)); gender='Female' if customer%2 else 'Male'
        churn=int(customer%5==0)
        for order in range(repeats):
            ix=customer*repeats+order
            cat=['Books','Clothing','Electronics','Home'][(customer+order)%4]
            payment=str(rng.choice(['Credit Card','Cash','Crypto','PayPal']))
            price=int(rng.integers(10,500)); quantity=int(rng.integers(1,6))
            amt=float(1400+age*3+{'Books':12,'Clothing':0,'Electronics':-10,'Home':-16}[cat]+rng.normal(0,900))
            missing=order==1
            rows.append({'Customer ID':customer+1,'Purchase Date':dates[ix],
                         'Product Category':cat,'Product Price':price,'Quantity':quantity,
                         'Total Purchase Amount':max(0,amt), 'Payment Method':payment,
                         'Returns':np.nan if missing else int(order%2==0),
                         'Returns_was_missing':int(missing),'Age':age,'Gender':gender,'Churn':churn})
    return pd.DataFrame(rows)


class Phase4InferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.df=fixture()

    def test_holm_monotone_original_order(self):
        self.assertEqual(holm([.04,.001,.03]),[.06,.003,.06])

    def test_inference_uses_independent_customer_counts(self):
        r=analyze(self.df)
        self.assertEqual(r['customers'],72)
        self.assertEqual(r['H1']['n_male']+r['H1']['n_female'],72)
        self.assertEqual(r['H2']['n_churn']+r['H2']['n_retained'],72)
        self.assertEqual(r['H3']['cluster_n'],72)
        self.assertEqual(len(r['H3']['pairs_exploratory']),6)
        self.assertTrue(0<=r['H3']['p']<=1)

    def test_h3_customer_cluster_robust_not_naive_rows(self):
        r1=analyze(self.df)
        d2=self.df.copy();d2['Purchase Date']=d2['Purchase Date']+pd.Timedelta(days=15)
        r2=analyze(pd.concat([self.df,d2],ignore_index=True))
        self.assertEqual(r1['H3']['cluster_n'],r2['H3']['cluster_n'])
        self.assertAlmostEqual(r1['H3']['p'],r2['H3']['p'],delta=.01)
        self.assertAlmostEqual(r1['H3']['mean_se_clustered']['Books'],r2['H3']['mean_se_clustered']['Books'],delta=.5)

    def test_w9_artifacts_and_json_created(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            inp=root/'fixture.csv'; self.df.to_csv(inp,index=False)
            out=run_inference(path=inp,report_path=root/'h9.md',fig_path=root/'09.png',json_path=root/'results.json')
            self.assertEqual(out['customers'],72)
            self.assertTrue((root/'09.png').stat().st_size>1000)
            self.assertIn('Holm', (root/'h9.md').read_text(encoding='utf-8'))
            self.assertIn('H3',json.loads((root/'results.json').read_text()))

    def test_no_null_returns_imputation(self):
        r=analyze(self.df)
        self.assertEqual(r['rows'],len(self.df))
        self.assertEqual(int(self.df.Returns.isna().sum()),72)

    def test_reject_customer_gender_disagreement(self):
        d=self.df.copy();d.loc[1,'Gender']='Female' if d.loc[1,'Gender']=='Male' else 'Male'
        with self.assertRaises(ValueError):analyze(d)

    def test_reject_legacy_missing_indicator(self):
        with self.assertRaises(ValueError):validate_eda_data(self.df.drop(columns=['Returns_was_missing']))

    def test_ab_sample_size_is_assumption_not_dataset(self):
        self.assertEqual(required_per_arm(.2,.22),6510)
        with self.assertRaises(ValueError):required_per_arm(.2,.2)


class Phase4RegressionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.df=fixture(n_customers=90,repeats=4)

    def test_grouped_holdout_disjoint(self):
        tr,te=grouped_split(self.df)
        self.assertTrue(set(self.df.iloc[tr]['Customer ID']).isdisjoint(set(self.df.iloc[te]['Customer ID'])))

    def test_design_excludes_future_or_missing_returns(self):
        x=design_matrix(self.df)
        self.assertNotIn('Returns',x.columns)
        self.assertNotIn('Churn',x.columns)
        self.assertNotIn('Customer ID',x.columns)
        self.assertIn('Product Category = Books',x.columns)
        self.assertNotIn('Product Category = Clothing',x.columns)

    def test_cluster_ols_se_and_ci_finite(self):
        x=design_matrix(self.df).to_numpy()
        r=cluster_ols(x,self.df['Total Purchase Amount'].to_numpy(),self.df['Customer ID'].to_numpy())
        self.assertTrue(np.all(np.isfinite(r['se'])))
        self.assertTrue(np.all(r['ci_low']<=r['ci_high']))
        self.assertEqual(r['n_groups'],90)

    def test_regression_group_cv_and_baseline(self):
        result,_=train_and_evaluate(self.df,cv_splits=3)
        self.assertTrue(result['no_customer_overlap'])
        self.assertEqual(result['returns_missing_in_input'],90)
        self.assertEqual(set(result['models']),{'OLS','Ridge','Lasso','Baseline (mean)'})
        self.assertEqual(len(result['ols_mae_gain_cluster_bootstrap_ci95']),2)
        self.assertNotIn('Returns',result['features'])

    def test_w10_artifacts_and_source_not_mutated(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            inp=root/'fixture.csv'; self.df.to_csv(inp,index=False)
            before=inp.read_bytes()
            result=run_regression(path=inp,report_path=root/'reg.md',fig_dir=root/'figs',json_path=root/'reg.json')
            self.assertEqual(inp.read_bytes(),before)
            self.assertTrue((root/'figs/07_coefficients_ci.png').stat().st_size>1000)
            self.assertIn('Baseline (mean)',result['models'])
            self.assertIn('cluster', (root/'reg.md').read_text(encoding='utf-8').lower())

    def test_dry_run_without_clean_source(self):
        output=subprocess.run([sys.executable,'-m','src.run_phase4','--dry-run'],
            cwd=REPO_ROOT,capture_output=True,text=True,check=True)
        self.assertIn('src.pipeline.07_hypothesis_tests',output.stdout)
        self.assertIn('src.pipeline.06_regression',output.stdout)


if __name__=='__main__':unittest.main()
