"""Phase 5.1 train-only improvement audit: python -m src.run_phase51 [--dry-run]."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

from src.classification_improvement_phase51 import run_improvement, write_outputs
from src.eda_metrics import CLEAN_PATH, load_cleaned
from src.utils import REPO_ROOT


def hash_file(path: Path) -> str:
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):
            h.update(block)
    return h.hexdigest()


def main(argv=None) -> int:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data',type=Path,default=CLEAN_PATH)
    parser.add_argument('--output',type=Path,default=REPO_ROOT)
    parser.add_argument('--dry-run',action='store_true')
    parser.add_argument('--folds',type=int,default=3)
    parser.add_argument('--repeats',type=int,default=2)
    parser.add_argument('--permutations',type=int,default=2)
    parser.add_argument('--max-iter',type=int,default=45)
    args=parser.parse_args(argv)
    source=args.data.resolve();out=args.output.resolve()
    print(f'Phase 5.1: Python {sys.version.split()[0]}, source={source}',flush=True)
    print(f'Source exists: {source.is_file()}; runs: {args.repeats}x{args.folds} CV, {args.permutations} permutations',flush=True)
    if args.dry_run:
        return 0
    if not source.is_file():
        raise FileNotFoundError('Phase 2 validated clean CSV required. Run python -m src.run_pipeline first; no legacy fallback.')
    before=hash_file(source)
    data=load_cleaned(source)
    result=run_improvement(data,folds=args.folds,repeats=args.repeats,
                           permutations=args.permutations,max_iter=args.max_iter)
    if hash_file(source)!=before:
        raise RuntimeError('Source modified during Phase 5.1')
    artifacts=write_outputs(result,out)
    manifest={
        'phase':'5.1','created_utc':datetime.now(timezone.utc).isoformat(),
        'python':sys.version.split()[0], 'source_path':str(source),
        'source_sha256':before,'phase5_original_untouched':True,
        'holdout_reused_for_selection':False,
        'artifacts_sha256':{str(p.relative_to(out)):hash_file(p) for p in artifacts},
    }
    path=out/'reports/phase5/phase51_run_manifest.json'
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(manifest,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
    print(f"Phase 5.1: eligible={result['eligible_customers']:,}; train={result['train_customers']:,}; HOLDOUT NOT SCORED={result['held_out_customers_not_scored']:,}")
    for name,metrics in result['tested_models'].items():
        m=metrics['mean']; print(f"  {name}: CV ROC-AUC={m['ROC_AUC']:.4f}, AP={m['PR_AUC']:.4f}, Lift@10%={m['lift_at_top_10pct']:.3f}")
    print(f"Best exploratory CV={result['selected_exploratory_cv']}; permutation AUC={result['screening']['average_permuted_roc_auc']:.4f}")
    print(f"TRAIN-ONLY SIGNAL SCREEN: {'PASS (new independent validation required)' if result['passed_exploratory_screen'] else 'FAIL (retain Phase 5)'}")
    print(f'PASS: Phase 5.1, source unchanged, {len(artifacts)} artifacts. Manifest: {path}')
    print('WARNING: static synthetic Churn label; NOT future churn forecasting. Phase 5 not modified.')
    return 0


if __name__=='__main__':
    raise SystemExit(main())
