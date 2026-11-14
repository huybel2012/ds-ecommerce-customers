"""Phase 5 W11 entrypoint: python -m src.run_phase5 [--dry-run].

Requires the Phase 2 validated clean CSV. Refuses to use legacy processed data.
Does not write to source data or interfere with existing Phase 1–4 figures.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

import pandas as pd

from src.classification_phase5 import CUTOFF, run_classification
from src.eda_metrics import CLEAN_PATH, load_cleaned
from src.utils import REPO_ROOT


def sha256(path: Path) -> str:
    h=hashlib.sha256()
    with Path(path).open('rb') as src:
        for chunk in iter(lambda:src.read(1<<20),b''):
            h.update(chunk)
    return h.hexdigest()


def main(argv=None) -> int:
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dry-run',action='store_true')
    p.add_argument('--data',type=Path,default=CLEAN_PATH,help='Phase 2 clean CSV (for tests use fixture)')
    p.add_argument('--output',type=Path,default=REPO_ROOT,help='Output root for reports/ and figures/')
    p.add_argument('--cutoff',type=str,default=str(CUTOFF.date()))
    p.add_argument('--trees',type=int,default=80)
    p.add_argument('--folds',type=int,default=3)
    p.add_argument('--bootstrap',type=int,default=250)
    args=p.parse_args(argv)
    source=Path(args.data).resolve()
    out=Path(args.output).resolve()
    cutoff=pd.Timestamp(args.cutoff)
    if args.trees<10 or args.folds<2 or args.bootstrap<20:
        p.error('Need >=10 trees, >=2 folds and >=20 bootstrap repetitions')
    print(f'Phase 5: Python {sys.version.split()[0]}; source={source}',flush=True)
    print(f'Feature cutoff (exclusive): {cutoff.date()}; output root: {out}',flush=True)
    print(f'Source exists: {source.is_file()}',flush=True)
    if args.dry_run:
        return 0
    if not source.is_file():
        raise FileNotFoundError(f'Phase 2 cleaned CSV not found: {source}. Run python -m src.run_pipeline first.')
    original=sha256(source)
    df=load_cleaned(source)
    target_report=out/'reports/classification_results.md'
    target_metrics=out/'reports/phase5/classification_metrics.json'
    figures=out/'reports/figures'
    result=run_classification(df,target_report,target_metrics,figures,cutoff=cutoff,
                              trees=args.trees,folds=args.folds,bootstrap=args.bootstrap)
    if sha256(source)!=original:
        raise RuntimeError('Phase 5 unexpectedly modified its input data')
    paths=result['artifacts']
    manifest={
        'phase':5, 'created_utc':datetime.now(timezone.utc).isoformat(),
        'python':sys.version.split()[0], 'source':str(source),
        'source_sha256':original, 'cutoff_exclusive':cutoff.isoformat(),
        'model_selection':'train-only OOF F1',
        'customer_holdout_disjoint':True,
        'prospective_churn_prediction_validated':False,
        'artifacts_sha256':{str(p.relative_to(out)):sha256(p) for p in paths},
    }
    mf=out/'reports/phase5/run_manifest.json'
    mf.parent.mkdir(parents=True,exist_ok=True)
    mf.write_text(json.dumps(manifest,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
    res=result['summary']
    print(f"Phase 5: eligible customers={res['feature_audit']['eligible_customers']:,}; "
          f"train={res['train_customers']:,}; test={res['test_customers']:,}")
    print('SELECTED BY TRAIN OOF:',res['selected_by_train_only'])
    for name,r in res['models'].items():
        m=r['test']
        print(f"  {name}: F1={m['F1']:.4f}, ROC-AUC={m['ROC_AUC']:.4f}, "
              f"PR-AUC={m['PR_AUC_average_precision']:.4f}, threshold={m['threshold']:.3f}")
    print(f'PASS: Phase 5, source unchanged, {len(paths)} artifacts. Manifest: {mf}')
    print('LIMITATION: synthetic retrospective labels have no verified event timestamp; NOT future churn prediction.')
    return 0

if __name__=='__main__':
    raise SystemExit(main())
