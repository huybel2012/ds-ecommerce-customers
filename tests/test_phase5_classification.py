"""Phase 5 W11 contract tests: grouped holdout, OOF thresholding and temporal caveats."""
from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
import unittest

import numpy as np
import pandas as pd

from src.classification_phase5 import (
    CUTOFF, FORBIDDEN_PREDICTORS, choose_f1_threshold, customer_holdout,
    grouped_cv, make_customer_features, run_classification, score_predictions,
    train_evaluate,
)


def fixture(n=90):
    rng = np.random.default_rng(94)
    rows=[]
    for i in range(n):
        label=int(i%5==0)
        age=22+i%44
        gender=['Female','Male'][i%2]
        for j, date in enumerate(['2021-02-01', '2022-07-12', '2023-04-11']):
            r=int(j==1 and i%7==0)
            missing=(i+j)%6==0
            rows.append({'Customer ID':i+1,'Purchase Date':pd.Timestamp(date)+pd.Timedelta(minutes=i),
                         'Product Category':['Books','Home','Clothing','Electronics'][(i+j)%4],
                         'Payment Method':['Cash','PayPal','Credit Card','Crypto'][(i+j)%4],
                         'Product Price':100+(i+j)%30,'Quantity':1+j,
                         'Total Purchase Amount':800+(i*3+j*5)%70,
                         'Returns':np.nan if missing else float(r),
                         'Returns_was_missing':int(missing),'Age':age,'Gender':gender,
                         'Churn':label})
    return pd.DataFrame(rows)


class FeatureEngineeringTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.df=fixture()

    def test_one_row_per_customer_and_history_only(self):
        x,y,meta,audit=make_customer_features(self.df)
        self.assertEqual(len(x),90)
        self.assertEqual(len(y),90)
        self.assertEqual(audit['history_transactions'],180)
        self.assertEqual(audit['excluded_post_cutoff_transactions'],90)
        self.assertTrue(x.index.is_unique)
        self.assertEqual(set(x.index),set(meta.index))

    def test_post_cutoff_mutation_does_not_change_features(self):
        x,*_=make_customer_features(self.df)
        altered=self.df.copy()
        altered.loc[altered['Purchase Date']>=CUTOFF,'Product Price']=999999
        altered.loc[altered['Purchase Date']>=CUTOFF,'Total Purchase Amount']=999999
        got,*_=make_customer_features(altered)
        pd.testing.assert_frame_equal(x,got)

    def test_post_cutoff_only_customer_excluded(self):
        f=self.df.iloc[:1].copy()
        f.loc[:,'Customer ID']=9999
        f.loc[:,'Purchase Date']=pd.Timestamp('2023-06-01')
        f['Churn']=0
        x,y,_,audit=make_customer_features(pd.concat([self.df,f],ignore_index=True))
        self.assertNotIn(9999,x.index)
        self.assertEqual(audit['ineligible_customers'],1)

    def test_reject_inconsistent_churn(self):
        d=self.df.copy();d.loc[1,'Churn']=1-d.loc[1,'Churn']
        with self.assertRaisesRegex(ValueError,'Churn'):
            make_customer_features(d)

    def test_reject_inconsistent_age(self):
        d=self.df.copy();d.loc[1,'Age']=99
        with self.assertRaisesRegex(ValueError,'Age'):
            make_customer_features(d)

    def test_reject_unexpected_category(self):
        d=self.df.copy();d.loc[0,'Product Category']='Gaming'
        with self.assertRaisesRegex(ValueError,'domain'):
            make_customer_features(d)

    def test_no_forbidden_columns_even_when_present_in_source(self):
        x,y,_,_=make_customer_features(self.df.assign(**{'Customer Name':'Synthetic Name'}))
        self.assertTrue(set(x.columns).isdisjoint(FORBIDDEN_PREDICTORS))
        self.assertNotIn('Churn',x)
        self.assertNotIn('Returns',x)
        self.assertNotIn('Gender',x)

    def test_latest_pre_cutoff(self):
        _,_,_,a=make_customer_features(self.df)
        self.assertLess(pd.Timestamp(a['latest_feature_transaction']),CUTOFF)
        self.assertFalse(a['label_timestamp_known'])
        self.assertFalse(a['prospective_claim_permitted'])

    def test_input_not_modified(self):
        d=self.df.copy(deep=True)
        make_customer_features(d)
        pd.testing.assert_frame_equal(d,self.df)

    def test_reject_no_history(self):
        d=self.df.copy();d['Purchase Date']=pd.Timestamp('2024-01-01')
        with self.assertRaisesRegex(ValueError,'before'):
            make_customer_features(d)


class ModelingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.df=fixture(n=90)
        cls.x,cls.y,_,_=make_customer_features(cls.df)

    def test_group_holdout_disjoint(self):
        tr,te=customer_holdout(self.x,self.y)
        self.assertEqual(len(set(self.x.index[tr])&set(self.x.index[te])),0)
        self.assertEqual(len(tr)+len(te),len(self.x))

    def test_group_cv_disjoint(self):
        tr,_=customer_holdout(self.x,self.y)
        xt=self.x.iloc[tr];yt=self.y.iloc[tr]
        for a,b in grouped_cv(xt,yt,3):
            self.assertFalse(set(xt.index[a])&set(xt.index[b]))

    def test_refuses_repeated_customer_as_feature_row(self):
        d=pd.concat([self.x,self.x.iloc[:1]])
        z=pd.concat([self.y,self.y.iloc[:1]])
        with self.assertRaisesRegex(ValueError,'unique customer'):
            customer_holdout(d,z)

    def test_threshold_is_deterministic_and_training_only(self):
        y=[0,0,1,1];p=[.1,.3,.4,.8]
        a=choose_f1_threshold(y,p)
        self.assertEqual(a,choose_f1_threshold(y,p))
        self.assertGreaterEqual(a[0],.05)
        self.assertLessEqual(a[0],.95)

    def test_confusion_matrix_and_imbalanced_accuracy(self):
        y=[0,0,0,0,1]
        p=[.2,.2,.2,.2,.2]
        s=score_predictions(y,p,.5)
        self.assertEqual(s['confusion_matrix_tn_fp_fn_tp'],[4,0,1,0])
        self.assertEqual(s['accuracy'],.8)
        self.assertEqual(s['F1'],0)
        self.assertAlmostEqual(s['ROC_AUC'],.5)

    def test_trains_baseline_and_two_models(self):
        result,detail=train_evaluate(self.x,self.y,trees=10,folds=2,bootstrap=20)
        self.assertEqual(set(result['models']),{'Prior baseline','All-positive baseline','Logistic Regression','Random Forest'})
        self.assertIn(result['selected_by_train_only'],('Logistic Regression','Random Forest'))
        self.assertTrue(0<=result['models']['Prior baseline']['test']['ROC_AUC']<=1)
        self.assertIn('train_all_positive_f1',result)
        self.assertIn('selected_holdout_f1_gain_vs_all_positive',result)
        self.assertIn('NOT FOR DEPLOYMENT',result['deployment_recommendation'])
        self.assertAlmostEqual(result['models']['All-positive baseline']['test']['recall'],1.0)
        self.assertTrue(set(detail['X_train'].index).isdisjoint(set(self.x.index)-set(detail['X_train'].index)))
        self.assertEqual(len(result['selected_holdout_ci95']),3)

    def test_rejects_bad_missing_indicator_at_entrypoint(self):
        with TemporaryDirectory() as t:
            d=self.df.copy();d.loc[0,'Returns_was_missing']=1-d.loc[0,'Returns_was_missing']
            with self.assertRaisesRegex(ValueError,'Returns_was_missing'):
                run_classification(d,Path(t)/'r.md',Path(t)/'m.json',Path(t)/'figs',trees=10,folds=2,bootstrap=20)

    def test_full_fixture_artifacts(self):
        with TemporaryDirectory() as t:
            root=Path(t)
            result=run_classification(self.df,root/'r.md',root/'m.json',root/'figs',trees=10,folds=2,bootstrap=20)
            self.assertEqual(len(result['artifacts']),5)
            self.assertTrue(all(p.is_file() for p in result['artifacts']))
            stored=json.loads((root/'m.json').read_text(encoding='utf-8'))
            self.assertFalse(stored['feature_audit']['prospective_claim_permitted'])
            self.assertIn('holdout', (root/'r.md').read_text(encoding='utf-8'))
            self.assertNotIn('Customer Name',stored['feature_columns'])

    def test_runner_end_to_end_fixture_creates_manifest_without_modifying_source(self):
        import hashlib
        with TemporaryDirectory() as t:
            root=Path(t)
            source=root/'clean.csv'
            self.df.to_csv(source,index=False)
            old_sha=hashlib.sha256(source.read_bytes()).hexdigest()
            # Preexisting W9 figure must survive a W11 run.
            prior=root/'reports/figures/09_hypothesis_tests.png'
            prior.parent.mkdir(parents=True,exist_ok=True)
            prior.write_bytes(b'keep previous phase')
            cmd=[sys.executable,'-m','src.run_phase5','--data',str(source),
                 '--output',str(root),'--trees','10','--folds','2','--bootstrap','20']
            r=subprocess.run(cmd,capture_output=True,text=True)
            self.assertEqual(r.returncode,0,r.stdout+'\n'+r.stderr)
            self.assertIn('PASS: Phase 5, source unchanged, 5 artifacts',r.stdout)
            self.assertEqual(hashlib.sha256(source.read_bytes()).hexdigest(),old_sha)
            self.assertEqual(prior.read_bytes(),b'keep previous phase')
            manifest=json.loads((root/'reports/phase5/run_manifest.json').read_text(encoding='utf-8'))
            self.assertFalse(manifest['prospective_churn_prediction_validated'])
            self.assertEqual(len(manifest['artifacts_sha256']),5)

    def test_rejects_legacy_csv_without_phase2_missing_indicator(self):
        with TemporaryDirectory() as t:
            root=Path(t)
            bad=root/'legacy.csv'
            self.df.drop(columns=['Returns_was_missing']).to_csv(bad,index=False)
            r=subprocess.run([sys.executable,'-m','src.run_phase5',
                '--data',str(bad),'--output',str(root),'--trees','10','--folds','2',
                '--bootstrap','20'],capture_output=True,text=True)
            self.assertNotEqual(r.returncode,0)
            self.assertIn('Returns_was_missing',r.stderr)

    def test_runner_dry_run_no_dataset_required(self):
        p=subprocess.run([sys.executable,'-m','src.run_phase5','--dry-run',
                         '--data','no_such_file_clean.csv'],capture_output=True,text=True)
        self.assertEqual(p.returncode,0,p.stderr)
        self.assertIn('Source exists: False',p.stdout)


if __name__=='__main__':
    unittest.main()
