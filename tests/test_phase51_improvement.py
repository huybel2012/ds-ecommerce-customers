"""Tests for Phase 5.1 exploratory improvements: safety, temporal features, CV."""
from __future__ import annotations

import json
import subprocess
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd

from src.classification_phase5 import CUTOFF, customer_holdout, make_customer_features
from src.classification_improvement_phase51 import (
    GATE, NEW_FEATURES, audit_univariate_signal, extended_customer_features,
    promotion_screen, run_improvement, top_decile_lift, write_outputs,
)


def fixture(n=85, signal=False):
    """Small Phase 2 clean-schema example with explicit pre/post cutoff events."""
    rows=[]
    for cid in range(1,n+1):
        label = int(cid%5 == 0)
        gender='Male' if cid%2 else 'Female'
        for j,date in enumerate(['2022-02-01','2022-06-05','2022-11-20','2023-07-01']):
            if signal and j==2 and label:
                date='2022-12-20'
            missing=(cid+j)%7==0
            rows.append({'Customer ID':cid,'Purchase Date':pd.Timestamp(date)+pd.Timedelta(minutes=cid),
                'Product Category':['Books','Home','Clothing','Electronics'][(cid+j)%4],
                'Payment Method':['Cash','PayPal','Credit Card','Crypto'][(cid+j)%4],
                'Product Price':100+(cid+j)%20, 'Quantity':j%3+1,
                'Total Purchase Amount':1200+((cid*j)%50),
                'Returns':float('nan') if missing else float(j%2),
                'Returns_was_missing':int(missing),
                'Age':20+cid%35,'Gender':gender,'Churn':label})
    return pd.DataFrame(rows)


class TemporalFeatureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data=fixture()

    def test_customer_count_and_feature_count(self):
        x,expanded,y,meta,audit=extended_customer_features(self.data)
        self.assertEqual(x.shape,(85,19))
        self.assertEqual(expanded.shape,(85,28))
        self.assertEqual(list(expanded.columns[-9:]),list(NEW_FEATURES))
        self.assertTrue(expanded.index.is_unique)
        self.assertEqual(len(y),85)
        self.assertFalse(audit['label_timestamp_known'])

    def test_future_transactions_ignored_for_features(self):
        _,x,*_=extended_customer_features(self.data)
        changed=self.data.copy()
        changed.loc[changed['Purchase Date']>=CUTOFF,'Total Purchase Amount']=999999
        changed.loc[changed['Purchase Date']>=CUTOFF,'Product Category']='Home'
        _,got,*_=extended_customer_features(changed)
        pd.testing.assert_frame_equal(got,x)

    def test_post_cutoff_only_customer_excluded(self):
        later=self.data.iloc[:1].copy()
        later['Customer ID']=400000
        later['Purchase Date']=pd.Timestamp('2023-08-10')
        d=pd.concat([self.data,later],ignore_index=True)
        _,features,*_=extended_customer_features(d)
        self.assertNotIn(400000,features.index)

    def test_boundary_cutoff_exclusive(self):
        d=self.data.copy()
        row=d.iloc[:1].copy()
        row['Purchase Date']=pd.Timestamp('2023-01-01T00:00:00')
        d=pd.concat([d,row],ignore_index=True)
        _,before,*_=extended_customer_features(self.data)
        _,after,*_=extended_customer_features(d)
        pd.testing.assert_frame_equal(before,after)

    def test_shuffle_rows_does_not_change_features_when_timestamps_unique(self):
        _,x,*_=extended_customer_features(self.data)
        _,got,*_=extended_customer_features(self.data.sample(frac=1,random_state=15))
        pd.testing.assert_frame_equal(got,x)

    def test_interval_category_and_window_values(self):
        x,extra,*_=extended_customer_features(self.data)
        cid=1
        self.assertEqual(extra.loc[cid,'transactions_last_90d'],1)
        self.assertEqual(extra.loc[cid,'transactions_last_180d'],1)
        self.assertEqual(extra.loc[cid,'transactions_previous_90d'],0)
        self.assertAlmostEqual(extra.loc[cid,'category_switch_rate'],1.0)
        self.assertGreater(extra.loc[cid,'mean_gap_days'],0)
        self.assertTrue(np.isfinite(extra.to_numpy()).all())
        self.assertNotIn('Returns',extra.columns)
        self.assertNotIn('Customer ID',extra.columns)
        self.assertNotIn('Gender',extra.columns)

    def test_single_pre_cutoff_event_no_nan_gap(self):
        d=self.data.copy()
        keep=(d['Purchase Date']>=CUTOFF) | d['Purchase Date'].dt.month.eq(2)
        _,x,*_=extended_customer_features(d.loc[keep])
        self.assertEqual(x['mean_gap_days'].sum(),0)
        self.assertEqual(x['category_switch_rate'].sum(),0)
        self.assertEqual(x['has_both_180d_windows'].sum(),0)

    def test_input_unchanged(self):
        copy=self.data.copy(deep=True)
        extended_customer_features(self.data)
        pd.testing.assert_frame_equal(self.data,copy)

    def test_reject_label_inconsistency(self):
        d=self.data.copy(); d.loc[1,'Churn']=1-d.loc[1,'Churn']
        with self.assertRaisesRegex(ValueError,'Churn'):
            extended_customer_features(d)


class ExperimentTests(unittest.TestCase):
    def test_lift_at_top_decile(self):
        y=[1,0,0,0,0,0,0,0,0,0]
        p=[.9,.1,.1,.1,.1,.1,.1,.1,.1,.1]
        self.assertAlmostEqual(top_decile_lift(y,p),10)

    def test_reject_unaligned_customer_rows(self):
        d=fixture(45)
        base,*_=make_customer_features(d)
        self.assertTrue(base.index.is_unique)
        tr,te=customer_holdout(base,make_customer_features(d)[1])
        self.assertFalse(set(base.index[tr]) & set(base.index[te]))

    def test_univariate_signal_has_only_train_features(self):
        _,x,y,*_=extended_customer_features(fixture(60))
        selected=x.iloc[:50]
        out=audit_univariate_signal(selected,y.iloc[:50])
        self.assertEqual(len(out),28)
        self.assertTrue(all(a['mutual_information']>=0 for a in out))

    def test_screen_requires_all_checks(self):
        actual={'mean':{'ROC_AUC':.51,'AP_gain_vs_prevalence':.001,
                        'lift_at_top_10pct':1.03},'repeat_roc_auc':{'42':.52}}
        neg=[{'mean':{'ROC_AUC':.50}}]
        passed, audit=promotion_screen(actual,neg)
        self.assertFalse(passed)
        self.assertFalse(all(audit['checks'].values()))
        self.assertEqual(len(GATE),5)

    def test_train_only_experiment_outputs(self):
        df=fixture(90,signal=True)
        res=run_improvement(df,folds=2,repeats=1,permutations=1,max_iter=8)
        self.assertEqual(res['train_customers']+res['held_out_customers_not_scored'],90)
        self.assertEqual(res['n_features_original'],19)
        self.assertEqual(res['n_features_expanded'],28)
        self.assertEqual(len(res['tested_models']),4)
        self.assertEqual(len(res['negative_control']['runs']),1)
        self.assertFalse(res['model_replacement_approved'])
        self.assertIn('holdout NOT evaluated',res['protocol'])
        with TemporaryDirectory() as tmp:
            paths=write_outputs(res,Path(tmp))
            self.assertEqual(len(paths),4)
            self.assertTrue(all(p.exists() for p in paths))
            saved=json.loads(paths[1].read_text(encoding='utf-8'))
            self.assertFalse(saved['model_replacement_approved'])

    def test_dry_run_fails_closed_for_missing_real_source(self):
        with TemporaryDirectory() as tmp:
            p=subprocess.run([sys.executable,'-m','src.run_phase51',
                '--data',str(Path(tmp)/'not_found.csv'),'--dry-run'],
                capture_output=True,text=True)
            self.assertEqual(p.returncode,0)
            self.assertIn('Source exists: False',p.stdout)
            p=subprocess.run([sys.executable,'-m','src.run_phase51',
                '--data',str(Path(tmp)/'not_found.csv')],capture_output=True,text=True)
            self.assertNotEqual(p.returncode,0)
            self.assertIn('validated clean CSV required',p.stderr)


if __name__=='__main__':
    unittest.main()
