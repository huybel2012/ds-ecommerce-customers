"""W12 fairness audit regression tests; no real dataset or network required."""
from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np
import pandas as pd

from src.classification_phase5 import CUTOFF, SEED, make_customer_features, customer_holdout, classifier_models, score_predictions
from src.fairness_phase6 import (
    wilson_ci, subgroup_frame, group_metrics, disparity, subgroup_audit,
    frozen_phase5_holdout, generate_phase6, make_model_card,
)
from src.run_phase6 import main as runner


def fixture(n=120):
    rows=[]
    for i in range(n):
        for j,date in enumerate(['2021-02-01','2022-07-12','2023-04-11']):
            missing=(i+j)%6==0
            rows.append({
                'Customer ID':i+1,'Purchase Date':pd.Timestamp(date)+pd.Timedelta(minutes=i),
                'Product Category':['Books','Home','Clothing','Electronics'][(i+j)%4],
                'Payment Method':['Cash','PayPal','Credit Card','Crypto'][(i+j)%4],
                'Product Price':100+(i+j)%30,'Quantity':1+j,
                'Total Purchase Amount':800+(i*3+j*5)%70,
                'Returns':np.nan if missing else float(j==1 and i%7==0),
                'Returns_was_missing':int(missing), 'Age':18+i%53,
                'Gender':['Female','Male'][i%2], 'Churn':int(i%5==0),
                'Customer Name':f'Faker name {i}',
            })
    return pd.DataFrame(rows)


def expected_phase5(df, *, trees=10, threshold=.45):
    X,y,_,audit=make_customer_features(df)
    tr,te=customer_holdout(X,y,seed=SEED)
    clf=classifier_models(trees=trees,seed=SEED)['Random Forest'].fit(X.iloc[tr],y.iloc[tr])
    probs=clf.predict_proba(X.iloc[te])[:,1]
    return {
        'seed':SEED, 'feature_columns':list(X.columns),
        'feature_audit':audit,'train_customers':len(tr),'test_customers':len(te),
        'selected_by_train_only':'Random Forest',
        'models':{'Random Forest':{'test':score_predictions(y.iloc[te].to_numpy(),probs,threshold)}}
    }


class FairnessMetricsTests(unittest.TestCase):
    def setUp(self):
        self.y=pd.Series([0,0,1,1, 0,0,1,1],index=range(8))
        self.p=np.array([.1,.8,.9,.9, .1,.1,.1,.1])
        self.meta=pd.DataFrame({'Gender':['Female']*4+['Male']*4,
                                'Age':[20,21,22,23,30,31,32,33]},index=range(8))

    def test_wilson_ci_boundaries_and_empty(self):
        self.assertIsNone(wilson_ci(0,0))
        self.assertLessEqual(wilson_ci(0,10)[0], 1e-12)
        self.assertGreaterEqual(wilson_ci(10,10)[1], 1-1e-12)
        lo,hi=wilson_ci(4,10)
        self.assertTrue(lo < .4 < hi)

    def test_confusion_and_rates_are_correct(self):
        frame=subgroup_frame(self.y,self.p,self.meta,.5)
        rows=group_metrics(frame,'Gender',min_group=4,min_class=2)
        female=next(x for x in rows if x['group']=='Female')
        self.assertEqual(female['confusion_matrix_tn_fp_fn_tp'],[1,1,0,2])
        self.assertAlmostEqual(female['tpr'],1.0)
        self.assertAlmostEqual(female['fpr'],.5)
        self.assertAlmostEqual(female['ppv'],2/3)
        self.assertAlmostEqual(female['selection_rate'],3/4)

    def test_all_positive_parity_is_not_fairness_certification(self):
        frame=subgroup_frame(self.y,np.ones(8),self.meta,.5)
        rows=group_metrics(frame,'Gender',min_group=4,min_class=2)
        g=disparity(rows)
        self.assertEqual(g['selection_rate_gap'],0.0)
        self.assertEqual(g['fpr_gap'],0.0)
        self.assertTrue(all(r['fpr']==1 and r['tpr']==1 for r in rows))

    def test_suppression_when_small_n_or_class(self):
        frame=subgroup_frame(self.y,self.p,self.meta,.5)
        rows=group_metrics(frame,'Gender',min_group=6,min_class=2)
        self.assertTrue(all(r['suppressed'] for r in rows))
        self.assertTrue(all(r['roc_auc'] is None and r['confusion_matrix_tn_fp_fn_tp'] is None for r in rows))
        self.assertIsNone(disparity(rows)['tpr_gap'])

    def test_missing_or_bad_groups_fail_closed(self):
        m=self.meta.copy();m.iloc[0,m.columns.get_loc('Gender')]='Unknown'
        with self.assertRaisesRegex(ValueError,'Gender'):
            subgroup_frame(self.y,self.p,m,.5)
        with self.assertRaisesRegex(ValueError,'align'):
            subgroup_frame(self.y,self.p,self.meta.sample(frac=1,random_state=7),.5)

    def test_age_boundaries_18_and_70(self):
        m=self.meta.copy();m.loc[0,'Age']=18;m.loc[7,'Age']=70
        d=subgroup_frame(self.y,self.p,m,.5)
        self.assertEqual(d.loc[0,'AgeBand'],'18-25')
        self.assertEqual(d.loc[7,'AgeBand'],'66-70')

    def test_subgroup_no_customer_ids_and_disjoint_count(self):
        res=subgroup_audit(self.y,self.p,self.meta,.5,min_group=4,min_class=2)
        self.assertEqual(res['n_holdout_customers'],8)
        self.assertEqual(len(res['groups']['GenderAge']),2)
        self.assertNotIn('Customer ID',json.dumps(res))

    def test_threshold_cannot_be_outside_unit_interval(self):
        with self.assertRaisesRegex(ValueError,'Threshold'):
            subgroup_frame(self.y,self.p,self.meta,1.0)


class FrozenModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data=fixture()
        cls.ref=expected_phase5(cls.data, trees=10)

    def test_replays_frozen_model_and_metrics(self):
        y,prob,sens,audit=frozen_phase5_holdout(self.data,self.ref,trees=10)
        self.assertEqual(len(y),self.ref['test_customers'])
        self.assertTrue(y.index.equals(sens.index))
        self.assertEqual(audit['latest_feature_transaction'][:4],'2022')

    def test_rejects_metric_drift(self):
        ref=json.loads(json.dumps(self.ref))
        ref['models']['Random Forest']['test']['ROC_AUC']=0.1
        with self.assertRaisesRegex(ValueError,'drifted'):
            frozen_phase5_holdout(self.data,ref,trees=10)

    def test_rejects_feature_or_seed_drift(self):
        ref=json.loads(json.dumps(self.ref));ref['seed']=19
        with self.assertRaisesRegex(ValueError,'seed/feature'):
            frozen_phase5_holdout(self.data,ref,trees=10)

    def test_rejects_old_legacy_missing_flag(self):
        data=self.data.copy()
        data['Returns_was_missing']=0
        with self.assertRaisesRegex(ValueError,'Returns_was_missing'):
            frozen_phase5_holdout(data,self.ref,trees=10)

    def test_outputs_are_aggregate_and_no_names(self):
        res,report,card=generate_phase6(self.data,self.ref,trees=10,min_group=2,min_class=2)
        self.assertEqual(res['provenance']['model_selection_or_threshold_tuning_in_phase6'],False)
        self.assertIn('NOT FOR DEPLOYMENT',card)
        self.assertIn('Faker',report)
        self.assertNotIn('Faker name',json.dumps(res)+report+card)
        self.assertNotIn('"Customer ID"',json.dumps(res))

    def test_runner_fixture_outputs_three_artifacts_and_keeps_source(self):
        import hashlib
        with TemporaryDirectory() as d:
            root=Path(d);data=root/'clean.csv';metrics=root/'phase5.json'
            self.data.to_csv(data,index=False);metrics.write_text(json.dumps(self.ref),encoding='utf-8')
            sha=hashlib.sha256(data.read_bytes()).hexdigest()
            self.assertEqual(runner(['--dry-run','--data',str(root/'none.csv'),'--phase5-metrics',str(metrics)]),0)
            self.assertEqual(runner(['--data',str(data),'--phase5-metrics',str(metrics),
                                     '--output',str(root),'--trees','10']),0)
            self.assertEqual(sha,hashlib.sha256(data.read_bytes()).hexdigest())
            for name in ('fairness_metrics.json','model_card.md','ethics_and_bias.md'):
                self.assertTrue((root/'reports/phase6'/name).exists())
            manifest=json.loads((root/'reports/phase6/run_manifest.json').read_text())
            self.assertFalse(manifest['deployment_approved'])
            self.assertEqual(len(manifest['artifacts_sha256']),3)


if __name__=='__main__':unittest.main()
