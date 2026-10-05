from pathlib import Path as _ReleasePath
import sys as _release_sys
_release_sys.path.insert(0, str(_ReleasePath(__file__).resolve().parents[1]))
_release_sys.modules["bottleneck"] = None
"""Recreate source-only scores from the distributed frozen split indices.

Outputs go to results/refitted_0910; the evaluation inputs are never overwritten.
This optional check verifies model provenance independently of decision replay.
"""
from pathlib import Path
import json,time
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from threadpoolctl import threadpool_limits
ROOT=Path(__file__).resolve().parent;OUT=ROOT/'results/refitted_0910'
def main():
    OUT.mkdir(exist_ok=True);checks=[];start=time.perf_counter()
    for subject in ['deepscenario','sensodat','openpilot']:
        old=np.load(ROOT/'results'/f'{subject}_fresh_scores.npz')
        splits=json.loads((ROOT/'results'/f'{subject}_splits.json').read_text());store={}
        if subject!='openpilot':
            d=np.load(ROOT/'data'/f'{subject}_fresh.npz')
        else:
            report=json.loads((ROOT/'results/openpilot_run.json').read_text());spec=json.loads((ROOT/'data/openpilot_formal_specs.json').read_text());X=np.array([[s[k] for k in report['features']] for s in spec])
        for split in splits:
            key=split['key'];seed=int(key.rsplit('_',1)[1]);train=np.array(split['train']);hist=np.array(split['history']);target=np.array(split['target'])
            if subject=='openpilot':
                source=key.split('_')[0];y=pd.read_csv(ROOT/'results/openpilot/formal'/f'{source}_outcomes.csv').failure.to_numpy()
            else:X=d['X_config' if 'HGB-config' in key else 'X_spec'];y=d['y']
            model=make_pipeline(StandardScaler(),LogisticRegression(max_iter=2000,C=1.,random_state=seed)) if 'Logistic' in key else HistGradientBoostingClassifier(max_iter=200,learning_rate=.08,max_leaf_nodes=15,l2_regularization=1.,random_state=seed)
            with threadpool_limits(limits=4):
                model.fit(X[train],y[train]);s=model.predict_proba(X[target])[:,1];h=model.predict_proba(X[hist])[:,1]
            diff=float(np.max(np.abs(s-old[key])));assert diff<1e-12,(subject,key,diff)
            store[key]=s
            if subject!='openpilot':store[key+'_target_indices']=target
            hf=h[y[hist]==1];cut=float(np.quantile(hf,.05,method='lower')) if len(hf) else -np.inf
            checks.append(dict(subject=subject,key=key,max_score_difference=diff,history_cut=cut))
        np.savez_compressed(OUT/f'{subject}_scores.npz',**store)
        print(subject,'source-only fits reproduced',flush=True)
    (ROOT/'results/0910/refit_validation.json').write_text(json.dumps(dict(fits=len(checks),elapsed_seconds=time.perf_counter()-start,checks=checks),indent=2))
if __name__=='__main__':main()
