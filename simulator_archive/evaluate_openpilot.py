from pathlib import Path as _ReleasePath
import sys as _release_sys
_release_sys.path.insert(0, str(_ReleasePath(__file__).resolve().parents[1]))
_release_sys.modules["bottleneck"] = None
"""Evaluate frozen triage rules using newly executed version outcomes."""
import csv,json,time
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from threadpoolctl import threadpool_limits
from run_fresh import ROOT,PROTOCOL,FIELDS,sha,fit_model,metrics,evaluate

def main():
    start=time.perf_counter();out=ROOT/'results';p=json.loads((ROOT/'openpilot_protocol.json').read_text())
    specs=json.loads((ROOT/'data/openpilot_formal_specs.json').read_text())
    features=[k for k in specs[0] if k not in ['scenario_id','seed','duration']]
    X=np.asarray([[s[k] for k in features] for s in specs],float)
    ids=[s['scenario_id'] for s in specs];data={};overview=[]
    for tag in p['candidate_tags']:
        path=out/'openpilot/formal'/f'{tag}_outcomes.csv';d=pd.read_csv(path)
        assert d.scenario_id.tolist()==ids and len(d)==p['formal_scenarios']
        assert d.solver_errors.sum()==0
        data[tag]=d
        overview.append(dict(subject='openpilot',generator=tag,N=len(d),M=int(d.failure.sum()),
                             outcomes_sha256=sha(path),simulation_seconds=float(d.simulation_seconds.sum())))
    records=[];splits=[];store={};transitions=[]
    for source,target in p['source_target_pairs']:
        a=data[source].failure.to_numpy();b=data[target].failure.to_numpy()
        transitions.append(dict(source=source,target=target,pass_pass=int(((a==0)&(b==0)).sum()),
            pass_fail=int(((a==0)&(b==1)).sum()),fail_pass=int(((a==1)&(b==0)).sum()),fail_fail=int(((a==1)&(b==1)).sum())))
    with (out/'openpilot_draws.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=FIELDS);w.writeheader()
        for seed in PROTOCOL['model_seeds']:
            train,valid=train_test_split(np.arange(len(X)),test_size=.25,random_state=seed)
            for source,target in p['source_target_pairs']:
                ys=data[source].failure.to_numpy();yt=data[target].failure.to_numpy()
                for kind in (['HGB-spec','Logistic-spec'] if seed==PROTOCOL['model_seeds'][0] else ['HGB-spec']):
                    tic=time.perf_counter();model=fit_model(X[train],ys[train],seed,'Logistic' if kind.startswith('Logistic') else 'HGB')
                    with threadpool_limits(limits=4):
                        hist=model.predict_proba(X[valid])[:,1];s=model.predict_proba(X)[:,1]
                    cut=float(np.quantile(hist[ys[valid]==1],.05,method='lower')) if ys[valid].sum() else -np.inf
                    key=f'{source}_{target}_{kind}_{seed}';store[key]=s
                    records.append(dict(subject='openpilot',source=source,target=target,scorer=kind,model_seed=seed,
                        n_train=len(train),n_history=len(valid),history_failures=int(ys[valid].sum()),history_cut=cut,
                        fit_score_seconds=time.perf_counter()-tic,**metrics(yt,s)))
                    splits.append(dict(key=key,train=train.tolist(),history=valid.tolist(),target=list(range(len(X)))))
                    evaluate(w,'openpilot',source,target,kind,seed,s,yt,
                        data[target].simulation_seconds.to_numpy(),cut,
                        full=kind=='HGB-spec' and seed==PROTOCOL['model_seeds'][0]);f.flush()
    np.savez_compressed(out/'openpilot_fresh_scores.npz',**store)
    (out/'openpilot_splits.json').write_text(json.dumps(splits))
    report=dict(subject='openpilot',overview=overview,transitions=transitions,features=features,model_records=records,
        protocol_sha256=sha(ROOT/'protocol.json'),version_protocol_sha256=sha(ROOT/'openpilot_protocol.json'),
        input_sha256=sha(ROOT/'data/openpilot_formal_specs.json'),result_sha256=sha(out/'openpilot_draws.csv'),
        elapsed_seconds=time.perf_counter()-start,
        time_saved_unit='Fraction of simulated seconds, not measured wall time',
        inference_scope='Historical labels may describe the same scenario IDs; current-version labels never train scores')
    (out/'openpilot_run.json').write_text(json.dumps(report,indent=2));print(json.dumps(overview),flush=True)

if __name__=='__main__':main()
