"""Phase 6/W12 runner: python -m src.run_phase6 [--dry-run].

Recreates Phase 5 frozen Random Forest holdout and evaluates Gender, Age,
Gender x Age. No per-customer data written to disk. Does not alter source data.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

from src.eda_metrics import CLEAN_PATH, load_cleaned
from src.fairness_phase6 import generate_phase6
from src.utils import REPO_ROOT

PHASE5 = REPO_ROOT / 'reports/phase5/classification_metrics.json'


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(1<<20),b''):
            h.update(chunk)
    return h.hexdigest()


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dry-run',action='store_true')
    parser.add_argument('--data',type=Path,default=CLEAN_PATH)
    parser.add_argument('--phase5-metrics',type=Path,default=PHASE5)
    parser.add_argument('--output',type=Path,default=REPO_ROOT)
    parser.add_argument('--trees',type=int,default=80,help='Must match Phase 5 frozen forest trees')
    args=parser.parse_args(argv)
    src=Path(args.data).resolve();ref=Path(args.phase5_metrics).resolve();out=Path(args.output).resolve()
    if args.trees<10:parser.error('At least 10 trees required')
    print(f'Phase 6: Python {sys.version.split()[0]}; source={src}')
    print(f'Phase 5 frozen metrics exists: {ref.is_file()}; clean source exists: {src.is_file()}')
    if args.dry_run:return 0
    if not src.is_file():raise FileNotFoundError(f'Validated Phase 2 CSV unavailable: {src}')
    if not ref.is_file():raise FileNotFoundError(f'Phase 5 official metrics unavailable: {ref}')
    before=sha256(src);before_ref=sha256(ref)
    data=load_cleaned(src)
    phase5=json.loads(ref.read_text(encoding='utf-8'))
    res,report,card=generate_phase6(data,phase5,trees=args.trees)
    if before!=sha256(src) or before_ref!=sha256(ref):
        raise RuntimeError('Input changed during Phase 6')
    artifacts={
        'reports/phase6/fairness_metrics.json': json.dumps(res,indent=2,ensure_ascii=False)+'\n',
        'reports/phase6/ethics_and_bias.md': report,
        'reports/phase6/model_card.md': card,
    }
    for rel,contents in artifacts.items():
        p=out/rel;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(contents,encoding='utf-8')
    manifest={
        'phase':'6', 'created_utc':datetime.now(timezone.utc).isoformat(),
        'source_sha256':before,'phase5_metrics_sha256':before_ref,
        'python':sys.version.split()[0],
        'artifacts_sha256':{k:sha256(out/k) for k in artifacts},
        'frozen_holdout_decisions_reproduced':True, 'source_unchanged':True,
        'customer_data_exported':False,'deployment_approved':False,
    }
    mf=out/'reports/phase6/run_manifest.json';mf.parent.mkdir(parents=True,exist_ok=True)
    mf.write_text(json.dumps(manifest,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
    print(f"Phase 6: holdout={res['n_holdout_customers']:,}, fixed threshold={res['threshold']:.3f}")
    for k in ('Gender','AgeBand','GenderAge'):
        groups=res['groups'][k]
        print(f"  {k}: {len(groups)} groups; suppressed={sum(x['suppressed'] for x in groups)}; selection-rate gap={res['disparities'][k]['selection_rate_gap']}")
    repro=res['provenance']['phase5_metric_reproduction']
    print('Frozen model: binary confusion matrix exact; ROC-AUC delta=',repro['differences_current_minus_phase5']['ROC_AUC'])
    print(f'PASS: Phase 6, frozen Phase 5 decisions verified, source unchanged, {len(artifacts)} artifacts. Manifest: {mf}')
    print('LIMITATION: synthetic static label, near-all-positive predictions; NOT a fairness certification or deployment approval.')
    return 0


if __name__=='__main__':
    raise SystemExit(main())
